"""mesh_axis_align — VLM-driven 6-candidate axis-snap per object.

Why this exists
---------------
``mesh_orient`` (next step) picks among NONE / flipX / flipY / flipZ — a
180° flip about each body axis — to correct a globally-flipped SAM3D
mesh. But the FP-given placement of the body frame in world is usually
NOT axis-aligned to world: the body axis closest to world +Z may be off
by a small residual angle (e.g. ~6°) or a large one (~46° if the
"most-vertical" body axis is semantically the wrong one).

This skill picks the right one out of **6 candidate orientations** —
one for each of (+X, -X, +Y, -Y, +Z, -Z) body axis pointing to world +Z
— by calling a VLM with the real frame-0 photo (target object outlined)
and 6 simulated panels at the same camera angle.

Algorithm (per object, processed in support_tree topological order):

  1. Read FP frame-0 → ``(R_FP, t_FP)`` in world.
  2. Generate 6 candidates: for each ``(axis_idx, sign)``, compute
     ``R_world`` = Rodrigues snap of ``sign * R_FP[:, axis_idx]`` to +Z.
  3. For each candidate:
     a. New world rotation ``R_FP_new = R_world @ R_FP`` (position
        preserved: ``t_FP`` unchanged — body-frame origin stays put in
        world; the body axes rotate around it).
     b. Project mesh vertices to world: compute ``bottom_z`` and
        ``top_z``.
     c. Parent attachment (scheme C: ground 贴 root, child 贴 parent):
        - If parent is ``GROUND`` (root): ``pos_offset.z = 0`` (root stays
          where FP placed it; the ground plane moves to meet the root's
          bottom — see step 6).
        - If parent is another object: ``pos_offset.z = parent_top_z −
          bottom_z`` (child slides to sit on parent's top surface).
     d. Render the full scene at frame 0 from the primary camera, with
        all already-decided objects placed at their chosen pose + this
        candidate object at the candidate pose. → panel image.
  4. Compose 7-panel layout: REFERENCE (real frame 0 with SAM3 mask
     outline) + 6 candidates (+X / -X / +Y / -Y / +Z / -Z).
  5. VLM picks one. Apply: rewrite every
     ``foundation_pose/frame_*_transform.npy`` with the chosen R_world
     (preserving t), patch manifest's per-object ``pos_offset`` =
     ``[0, 0, pos_offset_z]``.
  6. After loop: ``ground.offset = -min(world_bottom across decided)``
     so the MuJoCo ground plane sits at the lowest object's bottom
     (sim convention: ``ground_z = -offset``).

Why position is preserved
-------------------------
The previous axis_align (commit de6190c) did
``T_world_new = R_apply @ T_world`` which rotates BOTH R and t about the
world origin — the object visually swings to a new location. This rewrite
applies ``R_apply`` only to R, keeping t unchanged. The body-frame origin
(= what FP tracks) stays in place in world; only the orientation around
that point changes. The single intentional z-adjustment is the parent
attachment shift, recorded as ``pos_offset.z`` in the manifest.

State requirements
------------------
- ``<episode_dir>/manifest.yaml`` — object list, cameras, robot.
- ``<episode_dir>/objects/<name>/{visual.obj, scale.npy, foundation_pose/}``
- ``<run_root>/state.json`` — must carry ``support_tree.relations`` and
  ``segmentation_result[obj]['masks_dir' + 'keyframe']`` for mask overlay.

Idempotency
-----------
Each per-object call checks for ``foundation_pose_pre_align/``; if it
exists, refuse to re-apply (would compound rotations).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
import trimesh
import yaml
from PIL import Image

from ar2s.agents_geometry_prior.mesh_orient import (
    _earliest_pose,
    _bbox_of_projected_mesh,
    _label_strip,
    _load_camera,
    _load_real_rgb,
)


_AXIS_LABELS = ("X", "Y", "Z")
_DEFAULT_TARGET = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_CANDIDATE_LABELS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
_GROUND = "GROUND"


@dataclass
class AxisAlignResult:
    object_name: str
    applied: bool = False
    chosen_label: str = ""              # "+X" / "-X" / "+Y" / ...
    R_world: np.ndarray | None = None   # (3, 3) world-frame rotation applied
    pos_offset_z: float = 0.0
    bottom_z_world: float = 0.0         # world bottom after R_world + pos_offset
    top_z_world: float = 0.0            # world top after R_world + pos_offset
    keyframe: int = 0
    panel_path: str = ""
    vlm_raw: str = ""
    raw_mesh_backup: str = ""           # foundation_pose_pre_align/ path
    skipped_reason: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def _rodrigues_snap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal 3D rotation that maps unit vector ``a`` to unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-12:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        k = np.cross(a, perp)
        k /= np.linalg.norm(k)
        kx, ky, kz = k
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
        return np.eye(3) + 2.0 * (K @ K)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kx, ky, kz = v / s
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + K * s + (K @ K) * (1.0 - c)


# ---------------------------------------------------------------------------
# Support-tree adjacency + topological traversal
# ---------------------------------------------------------------------------

def _adjacency(relations: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    parent_of: dict[str, str] = {}
    children_of: dict[str, list[str]] = {}
    for r in relations:
        c = r["object"]
        p = r.get("supported_by", _GROUND)
        parent_of[c] = p
        children_of.setdefault(p, []).append(c)
    for k in children_of:
        children_of[k].sort()
    return parent_of, children_of


def _topological_order(parent_of: dict[str, str], children_of: dict[str, list[str]]) -> list[str]:
    """DFS from GROUND-children downward → parent-before-child order."""
    visited: set[str] = set()
    order: list[str] = []

    def dfs(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        order.append(node)
        for c in sorted(children_of.get(node, [])):
            dfs(c)

    roots = sorted(children_of.get(_GROUND, []))
    for r in roots:
        dfs(r)
    # Catch any node not reachable from GROUND
    for n in parent_of:
        dfs(n)
    return order


# ---------------------------------------------------------------------------
# Mesh + scale loading
# ---------------------------------------------------------------------------

def _load_scaled_mesh(obj_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (V_body, F) with per-object scale.npy applied."""
    mesh = trimesh.load(str(obj_dir / "visual.obj"), force="mesh", process=False)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    scale_path = obj_dir / "scale.npy"
    if scale_path.exists():
        s = np.asarray(np.load(scale_path), dtype=np.float64).reshape(-1)
        if s.size == 1:
            V = V * float(s[0])
        elif s.size >= 3:
            V = V * s[:3]
    return V, F


# ---------------------------------------------------------------------------
# SAM3 mask loading + overlay
# ---------------------------------------------------------------------------

def _resolve_local_masks_dir(run_root: Path, masks_dir_in_state: str) -> Path | None:
    """state.json's masks_dir may be /workspace/...; rewrite under run_root."""
    raw = Path(masks_dir_in_state)
    parts = raw.parts
    try:
        idx = parts.index("segmentation")
    except ValueError:
        return None
    return Path(run_root, *parts[idx:])


def _load_sam3_mask(
    run_root: Path, seg_result_for_obj: dict | None,
) -> tuple[np.ndarray | None, int]:
    """Return (mask_uint8_HxW, keyframe). mask is None if not found."""
    if not seg_result_for_obj:
        return None, 0
    masks_dir_str = seg_result_for_obj.get("masks_dir", "")
    keyframe = int(seg_result_for_obj.get("keyframe", 0))
    if not masks_dir_str:
        return None, keyframe
    masks_dir = _resolve_local_masks_dir(run_root, masks_dir_str)
    if masks_dir is None or not masks_dir.is_dir():
        return None, keyframe
    mask_path = masks_dir / f"{keyframe:06d}.png"
    if not mask_path.exists():
        pngs = sorted(masks_dir.glob("*.png"))
        if not pngs:
            return None, keyframe
        mask_path = pngs[0]
    arr = iio.imread(mask_path)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8), keyframe


def _overlay_mask_outline(rgb: np.ndarray, mask: np.ndarray,
                          color: tuple[int, int, int] = (255, 0, 0),
                          thickness: int = 3) -> np.ndarray:
    """Paint a coloured boundary of mask on rgb (returns copy).

    Uses morphological erosion: boundary = mask − erode(mask, thickness),
    then paints those pixels in ``color``. Pure-numpy/scipy, no cv2.
    """
    from scipy.ndimage import binary_erosion
    m = mask.astype(bool)
    eroded = binary_erosion(m, iterations=max(1, int(thickness)))
    boundary = m & ~eroded
    out = rgb.copy()
    out[boundary] = np.array(color, dtype=out.dtype)
    return out


# ---------------------------------------------------------------------------
# Multi-object MJCF panel render
# ---------------------------------------------------------------------------

# Optional: render the candidate (target) mesh in this saturated colour so the
# VLM can tell it apart from grey context objects. OFF by default — a flat solid
# colour kills the shading cues (e.g. a mug's concave opening vs its closed base)
# that distinguish up-vs-down, which hurts more than the object-ID help it gives
# in the typical 1-2 object scene. Enable with env AR2S_AXIS_HIGHLIGHT=1 or
# highlight=True (useful for cluttered multi-object scenes).
_HIGHLIGHT_RGBA = "1.0 0.1 0.9 1"        # bright magenta / pink
_HIGHLIGHT_NAME = "bright magenta (pink)"
_AXIS_HIGHLIGHT_DEFAULT = os.environ.get("AR2S_AXIS_HIGHLIGHT", "0") != "0"

# Auxiliary second-camera view: render the same 6 candidates from the scene's
# OTHER calibrated camera and append them below the primary panel, purely as a
# 3D-shape (flat-vs-standing) disambiguator — no real photo for that view. On by
# default when a secondary camera is resolvable; set AR2S_AXIS_AUX_VIEW=0 to skip.
_AXIS_AUX_DEFAULT = os.environ.get("AR2S_AXIS_AUX_VIEW", "1") != "0"

# Synthetic bird's-eye view: render each candidate from a camera looking straight
# DOWN at the object. Disambiguates cues neither grazing real camera shows — a
# container's opening reads as a ring (up) vs a closed disc (down); an elongated
# object reads as a full-length line (flat) vs a dot (standing). On by default;
# set AR2S_AXIS_TOPDOWN=0 to skip.
_AXIS_TOPDOWN_DEFAULT = os.environ.get("AR2S_AXIS_TOPDOWN", "1") != "0"


def _build_multi_object_mjcf(
    asset_meshes: list[tuple[str, Path]],
    body_specs: list[tuple[str, np.ndarray]],
    cam_pos: np.ndarray, cam_quat_wxyz: np.ndarray,
    fovy_deg: float, offscreen_side: int,
    ground_z: float = 0.0,
    highlight_idx: int | None = None,
    highlight_rgba: str = _HIGHLIGHT_RGBA,
) -> str:
    cp = " ".join(f"{x:.6f}" for x in cam_pos)
    cq = " ".join(f"{x:.6f}" for x in cam_quat_wxyz)

    assets = []
    for i, (_, path) in enumerate(asset_meshes):
        assets.append(f'    <mesh name="mesh_{i}" file="{path}"/>')
    assets_xml = "\n".join(assets)

    bodies = []
    for i, (_, pose_world) in enumerate(body_specs):
        body_pos = pose_world[:3, 3]
        body_quat = np.empty(4)
        mujoco.mju_mat2Quat(body_quat, pose_world[:3, :3].flatten())
        bp = " ".join(f"{x:.6f}" for x in body_pos)
        bq = " ".join(f"{x:.6f}" for x in body_quat)
        # Target (candidate) object is drawn in a saturated highlight colour so
        # the VLM can tell it apart from grey context objects; see _VLM_PROMPT_6.
        rgba = highlight_rgba if i == highlight_idx else "0.80 0.80 0.80 1"
        bodies.append(f'''
    <body name="body_{i}" pos="{bp}" quat="{bq}">
      <geom type="mesh" mesh="mesh_{i}" rgba="{rgba}"/>
    </body>''')

    return f"""<mujoco>
  <visual>
    <global offwidth="{offscreen_side}" offheight="{offscreen_side}"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7" specular="0.0 0.0 0.0"/>
  </visual>
  <asset>
{assets_xml}
  </asset>
  <worldbody>
    <camera name="cam0" pos="{cp}" quat="{cq}" fovy="{fovy_deg:.4f}"/>
    <geom type="plane" size="2 2 0.05" pos="0 0 {ground_z:.6f}" rgba="0.85 0.85 0.85 1"/>
{''.join(bodies)}
  </worldbody>
</mujoco>
"""


def _render_scene_panel(
    episode_dir: Path, primary_id: int,
    *,
    decided: dict[str, dict],
    current_obj_name: str,
    current_R_FP_new: np.ndarray,
    current_t_world: np.ndarray,
    current_pos_offset_z: float,
    current_bottom_z_world: float,
    current_V_body: np.ndarray,
    current_F: np.ndarray,
    tmp_dir: Path,
    tag: str,
    highlight: bool = _AXIS_HIGHLIGHT_DEFAULT,
    view_K: np.ndarray | None = None,
    view_w2c: np.ndarray | None = None,
    out_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Render the scene at frame 0.

    Objects are always placed in world via the PRIMARY camera (FP poses are in
    the primary frame). The VIEWING camera defaults to the primary too, but can
    be overridden with ``view_K`` / ``view_w2c`` to render the same world scene
    from a second calibrated camera (the auxiliary-view panels). ``out_hw`` is
    the viewing camera's image size (H, W); defaults to the primary real frame.

    Ground plane sits at world z = min(this candidate's bottom, decided objects'
    bottoms) so the candidate is always above the ground (no occlusion).
    """
    K, w2c = _load_camera(episode_dir, primary_id)   # placement (primary) frame
    c2w = np.linalg.inv(w2c)
    vK = view_K if view_K is not None else K
    v_c2w = np.linalg.inv(view_w2c) if view_w2c is not None else c2w

    if out_hw is not None:
        H, W = out_hw
    else:
        H, W = _load_real_rgb(episode_dir, frame_idx=0).shape[:2]
    S = 2 * int(np.ceil(max(vK[0, 2], W - vK[0, 2], vK[1, 2], H - vK[1, 2])))
    S += S % 2

    tmp_dir.mkdir(parents=True, exist_ok=True)
    asset_meshes: list[tuple[str, Path]] = []
    body_specs: list[tuple[str, np.ndarray]] = []

    for dec_name, dec in decided.items():
        dec_obj_dir = episode_dir / "objects" / dec_name
        V_dec, F_dec = _load_scaled_mesh(dec_obj_dir)
        dec_pose_cam, _ = _earliest_pose(dec_obj_dir)
        dec_pose_world = c2w @ dec_pose_cam
        dec_pose_world[2, 3] += float(dec["pos_offset_z"])

        mesh_path = tmp_dir / f"mesh_dec_{dec_name}.obj"
        trimesh.Trimesh(vertices=V_dec, faces=F_dec, process=False).export(str(mesh_path))
        asset_meshes.append((dec_name, mesh_path))
        body_specs.append((dec_name, dec_pose_world))

    cur_pose_world = np.eye(4)
    cur_pose_world[:3, :3] = current_R_FP_new
    cur_pose_world[:3, 3] = current_t_world
    cur_pose_world[2, 3] += float(current_pos_offset_z)

    cur_mesh_path = tmp_dir / f"mesh_candidate_{tag}.obj"
    trimesh.Trimesh(
        vertices=current_V_body, faces=current_F, process=False,
    ).export(str(cur_mesh_path))
    asset_meshes.append((current_obj_name, cur_mesh_path))
    body_specs.append((current_obj_name, cur_pose_world))
    target_idx = len(body_specs) - 1   # candidate object is always appended last

    from ar2s.droid_sim._util import opencv_c2w_to_mujoco_camera_pose, fovy_deg_from_K
    cam_pos, cam_quat = opencv_c2w_to_mujoco_camera_pose(v_c2w)
    fovy_deg = fovy_deg_from_K(vK, image_height=S)

    # Ground plane sits at the lowest bottom among (this candidate, decided
    # objects). Episode-level ground.offset = -min(decided.bottom_z_world);
    # this per-panel ground is the live preview of the same convention.
    candidate_floor = float(current_bottom_z_world)
    if decided:
        candidate_floor = min(candidate_floor,
                              min(d["bottom_z_world"] for d in decided.values()))

    xml = _build_multi_object_mjcf(
        asset_meshes, body_specs, cam_pos, cam_quat, fovy_deg, offscreen_side=S,
        ground_z=candidate_floor,
        highlight_idx=(target_idx if highlight else None),
    )
    spec = mujoco.MjSpec.from_string(xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ren = mujoco.Renderer(model, S, S)
    ren.update_scene(data, camera="cam0")
    img = ren.render()
    ren.close()
    x0 = int(round(S / 2 - vK[0, 2]))
    y0 = int(round(S / 2 - vK[1, 2]))
    return img[y0:y0 + H, x0:x0 + W]


def _load_aux_camera(
    run_root: Path, episode_dir: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray] | None:
    """Secondary camera (K, w2c, (H, W), real_rgb) for the auxiliary view, or None.

    Reuses ``scene_view_repair.load_scene_ctx``, which sources the second
    camera's real K + extrinsics + video from the local ``raw_episodes/`` dir.
    ``real_rgb`` is frame 0 of that camera (full frame, no mask — the target has
    no segmentation in the secondary view, so the VLM is told it may be occluded
    or absent there). Returns None (panel stays primary-only) if no secondary
    camera can be resolved — a single-camera episode or a bundle without raw
    frames.
    """
    if not _AXIS_AUX_DEFAULT:
        return None
    try:
        from ar2s.agents_visual.subagents.scene_view_repair import (
            load_scene_ctx, real_frame,
        )
        raw_dir = None
        for cand in (run_root / "raw_episodes",
                     episode_dir.parent.parent / "raw_episodes"):
            if cand.is_dir():
                raw_dir = cand
                break
        ctx = load_scene_ctx(run_root, raw_dir=raw_dir)
        sec = ctx["secondary_serial"]
        if not sec or sec not in ctx["K"] or sec not in ctx["c2w"]:
            return None
        K = np.asarray(ctx["K"][sec], dtype=np.float64)
        w2c = np.linalg.inv(np.asarray(ctx["c2w"][sec], dtype=np.float64))
        real_rgb = np.asarray(real_frame(ctx, sec, 0))[..., :3]
        H, W = real_rgb.shape[:2]
        return K, w2c, (int(H), int(W)), real_rgb
    except Exception as exc:                                   # noqa: BLE001
        print(f"[axis_align] auxiliary second-camera view unavailable: "
              f"{type(exc).__name__}: {exc}")
        return None


def _topdown_camera(
    center_world: np.ndarray, obj_extent: float, side: int = 512,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Synthetic camera looking straight DOWN at ``center_world``.

    Returns (K, w2c, (H, W)). Height scales with the object so it fills the
    frame; image is oriented world +Y = up, world +X = right. Shared by all 6
    candidates (their xy position is identical; only orientation differs).
    """
    r = max(0.5 * float(obj_extent), 1e-3)
    fov_deg = 45.0
    f = (side / 2) / np.tan(np.radians(fov_deg / 2))
    K = np.array([[f, 0, side / 2], [0, f, side / 2], [0, 0, 1]], dtype=np.float64)
    h = max(0.25, 4.0 * r)                       # clearance above the object
    # camera axes in world: x_cam=+X, y_cam=-Y, z_cam(optical)=-Z (looking down)
    R_c2w = np.array([[1., 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = [float(center_world[0]), float(center_world[1]),
                  float(center_world[2]) + h]
    w2c = np.linalg.inv(c2w)
    return K, w2c, (side, side)


# ---------------------------------------------------------------------------
# 7-panel composition
# ---------------------------------------------------------------------------

def _resize_panel(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_h = max(1, int(round(h * target_w / w)))
    pil = Image.fromarray(img)
    pil = pil.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(pil)


def _compose_seven_panel(
    reference: np.ndarray, candidates: list[np.ndarray],
    labels: list[str], panel_width: int = 360,
) -> np.ndarray:
    """Layout:
        row 1: [ REFERENCE (3-cell wide) ]
        row 2: [ +X | -X | +Y ]
        row 3: [ -Y | +Z | -Z ]
    """
    assert len(candidates) == 6 and len(labels) == 6
    cand_imgs = [_resize_panel(c, panel_width) for c in candidates]
    target_h = max(c.shape[0] for c in cand_imgs)
    cand_padded = []
    for c in cand_imgs:
        if c.shape[0] < target_h:
            pad = np.full((target_h - c.shape[0], c.shape[1], 3), 255, dtype=np.uint8)
            c = np.vstack([c, pad])
        cand_padded.append(c)
    cand_labels = [_label_strip(l, panel_width) for l in labels]

    sep_v_h = cand_labels[0].shape[0] + target_h
    sep_v = np.zeros((sep_v_h, 4, 3), dtype=np.uint8)

    def row(idxs):
        cols = []
        for j, i in enumerate(idxs):
            cell = np.vstack([cand_labels[i], cand_padded[i]])
            cols.append(cell)
            if j < len(idxs) - 1:
                cols.append(sep_v)
        return np.hstack(cols)

    row1 = row([0, 1, 2])
    row2 = row([3, 4, 5])

    # Reference at row1 width
    ref = _resize_panel(reference, row1.shape[1])
    ref_lbl = _label_strip("REFERENCE (real frame 0, target outlined in RED)", row1.shape[1])
    ref_row = np.vstack([ref_lbl, ref])

    sep_h = np.zeros((6, row1.shape[1], 3), dtype=np.uint8)
    return np.vstack([ref_row, sep_h, row1, sep_h, row2])


def _compose_aux_block(
    candidates: list[np.ndarray], labels: list[str],
    secondary_real: np.ndarray | None = None, panel_width: int = 360,
    header: str = ("SECOND CAMERA VIEW — real photo + same 6 candidates from "
                   "another angle (use to read flat vs standing)"),
) -> np.ndarray:
    """Second-camera block: real photo (if given) + the same 6 candidates.

        header:  SECOND CAMERA VIEW ...
        [ SECOND-VIEW REAL PHOTO (3-cell wide) ]   (only if secondary_real)
        row 1:   [ +X | -X | +Y ]
        row 2:   [ -Y | +Z | -Z ]
    """
    assert len(candidates) == 6 and len(labels) == 6
    cand_imgs = [_resize_panel(c, panel_width) for c in candidates]
    target_h = max(c.shape[0] for c in cand_imgs)
    padded = []
    for c in cand_imgs:
        if c.shape[0] < target_h:
            pad = np.full((target_h - c.shape[0], c.shape[1], 3), 255, dtype=np.uint8)
            c = np.vstack([c, pad])
        padded.append(c)
    cand_labels = [_label_strip(l, panel_width) for l in labels]
    sep_v = np.zeros((cand_labels[0].shape[0] + target_h, 4, 3), dtype=np.uint8)

    def row(idxs):
        cols = []
        for j, i in enumerate(idxs):
            cols.append(np.vstack([cand_labels[i], padded[i]]))
            if j < len(idxs) - 1:
                cols.append(sep_v)
        return np.hstack(cols)

    row1 = row([0, 1, 2])
    row2 = row([3, 4, 5])
    header_strip = _label_strip(header, row1.shape[1])
    sep_h = np.zeros((6, row1.shape[1], 3), dtype=np.uint8)
    parts = [header_strip]
    if secondary_real is not None:
        real_r = _resize_panel(secondary_real, row1.shape[1])
        real_lbl = _label_strip(
            "SECOND-VIEW REAL PHOTO (no outline; target may be occluded / not "
            "visible here — if so, ignore this view)", row1.shape[1])
        parts += [sep_h, real_lbl, real_r]
    parts += [sep_h, row1, sep_h, row2]
    return np.vstack(parts)


def _stack_primary_and_aux(primary: np.ndarray, aux: np.ndarray) -> np.ndarray:
    """Stack the auxiliary block below the primary 7-panel, width-matched."""
    w = primary.shape[1]
    aux_r = _resize_panel(aux, w)
    divider = np.zeros((12, w, 3), dtype=np.uint8)
    return np.vstack([primary, divider, aux_r])


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------

# Injected into _VLM_PROMPT_6 only when the target mesh is colour-highlighted.
_HIGHLIGHT_CLAUSE = (
    'In EVERY panel, the target object "{name}" is drawn in {hn}; all other '
    "scene objects and the ground are plain grey — judge ONLY the {hn} object, "
    "the grey objects are just context to help you read the scene.\n\n"
)

# Injected into _VLM_PROMPT_6 only when the auxiliary second-camera block exists.
_AUX_CLAUSE = (
    "\nBELOW the 6 panels is a SECOND CAMERA VIEW block, shot from a DIFFERENT "
    "angle: a REAL PHOTO from that camera, then the SAME 6 candidates rendered "
    "from it. The primary REFERENCE can be a grazing view where a flat object "
    "looks like it is standing up (and vice-versa); the second angle usually "
    "resolves this. Cross-check both real photos: pick the candidate that "
    "matches BOTH. IMPORTANT: the second-view real photo has NO outline and the "
    "target may be occluded or out of frame there — if you cannot confidently "
    "locate the target in it, IGNORE the second view entirely and judge from "
    "the primary REFERENCE alone. Do not force a match to something in the "
    "second photo that is not the target."
)

# Injected into _VLM_PROMPT_6 only when the synthetic top-down block exists.
_TOPDOWN_CLAUSE = (
    "\nThere is also a TOP-DOWN VIEW block: each candidate rendered from a "
    "synthetic camera looking straight DOWN (no real photo — a geometry aid). "
    "This is the clearest way to read cues the grazing real cameras hide: an "
    "open container (cup / bowl / mug) shows its OPENING as a hollow ring when "
    "it faces up, but a solid closed disc/dome when it faces down; an elongated "
    "object shows a full-length line when lying flat and a small dot when "
    "standing on end. Use it to settle up-vs-down and flat-vs-standing."
)

_VLM_PROMPT_6 = """You are choosing the correct world-frame orientation for an object
named "{name}" in a 3D physical scene reconstructed from a single RGB video.

The TOP image labelled REFERENCE shows the real scene's primary camera at
frame 0, with the target object outlined in RED.

The 6 panels below (labelled +X / -X / +Y / -Y / +Z / -Z) show the same
scene rendered from the SAME camera. {highlight_clause}In each panel, the object has been
rotated so that the labelled body axis points UP in world (+Z) — for
example "+Y" means the object's body +Y axis is now pointing up. The
object has also been slid vertically so it sits on its supporting parent
(e.g. table, ground).
{aux_clause}{topdown_clause}
Pick the ONE candidate whose orientation best matches the REFERENCE photo.
Focus on:
  - The object's UP direction (which face is on top vs bottom).
  - For a mug / bowl / cup: opening should face up unless the photo shows
    it upside-down.
  - For a flat package / box / lid: broad face up vs on its side.
  - For an elongated object (marker, pen, bottle): which end points up.

Texture / colour / lighting differences between mesh and photo are EXPECTED
and should be IGNORED — only the gross orientation matters. The in-plane
rotation about the vertical axis (e.g. mug handle on left vs right) is
FIXED IN A LATER STEP, so if two candidates only differ in such in-plane
rotation, pick either.

Respond on the FIRST line with EXACTLY one of: +X, -X, +Y, -Y, +Z, -Z
You may add a short justification on subsequent lines.
"""


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _judge_with_vlm_6(
    panel_path: Path, object_name: str,
    *, highlight: bool = _AXIS_HIGHLIGHT_DEFAULT, has_aux: bool = False,
    has_topdown: bool = False,
) -> tuple[int, str]:
    """Return (chosen_idx ∈ 0..5, raw_vlm_text)."""
    from ar2s.agent_configs.models import (
        chat_model_for, claude_to_lc_messages, parse_text,
    )
    llm = chat_model_for("geometry_prior.subagents.mesh_orient")
    if llm is None:
        raise RuntimeError(
            "no credential for the YAML-selected mesh_orient model provider"
        )
    clause = (_HIGHLIGHT_CLAUSE.format(name=object_name, hn=_HIGHLIGHT_NAME)
              if highlight else "")
    msgs = claude_to_lc_messages([
        {"type": "text", "text": _VLM_PROMPT_6.format(
            name=object_name, highlight_clause=clause,
            aux_clause=(_AUX_CLAUSE if has_aux else ""),
            topdown_clause=(_TOPDOWN_CLAUSE if has_topdown else ""))},
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _encode_image(panel_path),
        }},
    ])
    resp = llm.invoke(msgs)
    text = parse_text(resp).strip()
    first_line = text.splitlines()[0].strip().upper().replace(" ", "").rstrip(".:,")
    label_map = {l: i for i, l in enumerate(_CANDIDATE_LABELS)}
    if first_line in label_map:
        return label_map[first_line], text
    for label, idx in label_map.items():
        if label in text.upper():
            return idx, text
    raise ValueError(f"could not parse VLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# Manifest patching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-root ground-trust VLM picker
# ---------------------------------------------------------------------------

_VLM_PROMPT_GROUND_TRUST = """You are picking the most reliable FoundationPose (FP) pose recovery among
{n} rigid objects in a real-camera frame.

Each panel shows ONE object's name (top-left) plus the FP-recovered 6-DoF
pose visualised on that real frame as a GREEN 3D bounding box (8-vertex
wireframe) and body-frame XYZ axes (red/green/blue) at the body origin.

Your single criterion is **how tightly the green 3D bounding box wraps
the actual physical object**:

  - GOOD: the bbox wireframe closely follows the object's outline in the
    image — its faces appear flush with the object's surface, the bbox
    is barely larger than the object itself.
  - BAD: the bbox is noticeably larger / smaller than the actual object,
    or offset away from it, leaving a visible gap of empty pixels between
    the bbox edges and the object's silhouette.

Look ONLY at the bbox-vs-object fit. Do NOT use:
  - the axis arrows (always there, not informative)
  - the rest of the scene
  - any guess about which object would be "more useful" downstream

For tiny objects whose bbox is hard to see, judge by the visible portion
of the bbox you can see — if even that portion is offset, it's BAD.

Respond on the FIRST line with EXACTLY the object name (one of: {names}).
Add a short one-sentence justification on the second line.
"""


def _load_track_vis_frame0(obj_dir: Path) -> np.ndarray | None:
    """Load <obj_dir>/track_vis/frame_000000.png; None if missing."""
    p = obj_dir / "track_vis" / "frame_000000.png"
    if not p.exists():
        return None
    img = iio.imread(p)
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[..., :3]
    return img


def _crop_around_fp_bbox(
    track_vis: np.ndarray,
    V_scaled: np.ndarray,
    pose_obj_to_cam: np.ndarray,
    K: np.ndarray,
    pad_ratio: float = 0.6,
    min_side_px: int = 240,
) -> np.ndarray:
    """Crop track_vis to a square region around the FP-projected mesh bbox.

    The crop is centred on the projected mesh's 2D bounding box and padded by
    ``pad_ratio`` of its size on each side, so small objects (whose raw bbox
    is only a few pixels) get a meaningful context window. Width clamped to
    ``min_side_px`` so VLM can resolve detail even for tiny objects.
    """
    H, W = track_vis.shape[:2]
    x0, y0, x1, y1 = _bbox_of_projected_mesh(V_scaled, pose_obj_to_cam, K, H, W, pad=0)
    if x1 <= x0 or y1 <= y0:
        return track_vis   # degenerate bbox; return full frame
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0)
    side = max(side * (1 + 2 * pad_ratio), float(min_side_px))
    half = side / 2
    nx0 = int(max(0, cx - half))
    ny0 = int(max(0, cy - half))
    nx1 = int(min(W, cx + half))
    ny1 = int(min(H, cy + half))
    if nx1 <= nx0 or ny1 <= ny0:
        return track_vis
    return track_vis[ny0:ny1, nx0:nx1]


def _load_object_geometry_for_crop(
    obj_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (V_scaled_body, pose_obj_to_cam_frame0). None if assets missing.

    Prefers ``foundation_pose_pre_align/frame_000000_transform.npy`` (raw FP,
    matches the bbox burned into track_vis); falls back to the live
    ``foundation_pose/`` (which axis_align may have rewritten).
    """
    mesh_p = obj_dir / "visual.obj"
    if not mesh_p.is_file():
        return None
    scale_p = obj_dir / "scale.npy"

    # Prefer pre-axis-align FP (which is what track_vis was rendered against)
    # while supporting both packed and legacy pose stores.
    from ar2s.droid_sim.pose_store import earliest_frame, load_pose
    fp_root = obj_dir / "foundation_pose_pre_align"
    frame_idx = earliest_frame(fp_root)
    if frame_idx is None:
        fp_root = obj_dir / "foundation_pose"
        frame_idx = earliest_frame(fp_root)
    if frame_idx is None:
        return None

    mesh = trimesh.load(str(mesh_p), force="mesh", process=False)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    if scale_p.exists():
        s = np.asarray(np.load(scale_p), dtype=np.float64).reshape(-1)
        if s.size == 1:
            V = V * float(s[0])
        elif s.size >= 3:
            V = V * s[:3]
    pose = load_pose(fp_root, frame_idx)
    assert pose is not None, fp_root
    return V, pose


def _compose_n_candidate_panel(
    candidates: list[tuple[str, np.ndarray]],
    panel_width: int = 480,
) -> np.ndarray:
    """Horizontal strip of N labeled track_vis images."""
    cells = []
    for name, img in candidates:
        scaled = _resize_panel(img, panel_width)
        label = _label_strip(name, panel_width)
        cells.append(np.vstack([label, scaled]))
    target_h = max(c.shape[0] for c in cells)
    cells_padded = []
    for c in cells:
        if c.shape[0] < target_h:
            pad = np.full((target_h - c.shape[0], c.shape[1], 3), 255, dtype=np.uint8)
            c = np.vstack([c, pad])
        cells_padded.append(c)
    sep = np.zeros((target_h, 4, 3), dtype=np.uint8)
    out = cells_padded[0]
    for c in cells_padded[1:]:
        out = np.hstack([out, sep, c])
    return out


def _project_world_pose_bbox(
    V_body: np.ndarray, R_world: np.ndarray, t_world: np.ndarray,
    cam_w2c: np.ndarray, K: np.ndarray, H: int, W: int,
) -> tuple[int, int, int, int]:
    """2D bbox of mesh placed at (R_world, t_world), as seen by camera (w2c)."""
    pose_world = np.eye(4)
    pose_world[:3, :3] = R_world
    pose_world[:3, 3]  = t_world
    pose_cam = cam_w2c @ pose_world
    return _bbox_of_projected_mesh(V_body, pose_cam, K, H, W, pad=0)


def _union_bbox(bboxes: list[tuple[int, int, int, int]], W: int, H: int,
                pad_ratio: float = 0.3, min_side: int = 200,
                ) -> tuple[int, int, int, int]:
    """Compute union of N (x0,y0,x1,y1) bboxes, padded + clamped to image."""
    xs0 = min(b[0] for b in bboxes); ys0 = min(b[1] for b in bboxes)
    xs1 = max(b[2] for b in bboxes); ys1 = max(b[3] for b in bboxes)
    if xs1 <= xs0 or ys1 <= ys0:
        return 0, 0, W, H
    cx = (xs0 + xs1) / 2; cy = (ys0 + ys1) / 2
    side = max(xs1 - xs0, ys1 - ys0)
    side = max(side * (1 + 2 * pad_ratio), float(min_side))
    half = side / 2
    nx0 = int(max(0, cx - half)); ny0 = int(max(0, cy - half))
    nx1 = int(min(W, cx + half)); ny1 = int(min(H, cy + half))
    return nx0, ny0, nx1, ny1


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """2D bbox of nonzero mask region. None if mask is empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pick_trusted_ground_root(
    episode_dir: Path, root_names: list[str],
    *, primary_id: int, skip_vlm: bool = False,
) -> tuple[str, str, Path]:
    """N-candidate panel of FP-bbox crops + VLM picks most-trustworthy root.

    Each cell is the object's track_vis/frame_000000.png CROPPED around the
    FP-projected mesh bbox (with ~60% pad) so small objects (e.g. a 17mm pen)
    show their bbox at resolvable resolution instead of being a few pixels in
    a 1280×720 full frame.

    Returns (chosen_name, raw_vlm_text, panel_path).
    """
    K, _w2c = _load_camera(episode_dir, primary_id)
    cands: list[tuple[str, np.ndarray]] = []
    for name in root_names:
        obj_dir = episode_dir / "objects" / name
        img = _load_track_vis_frame0(obj_dir)
        if img is None:
            continue
        geom = _load_object_geometry_for_crop(obj_dir)
        if geom is not None:
            V_scaled, pose_oc = geom
            crop = _crop_around_fp_bbox(img, V_scaled, pose_oc, K)
        else:
            crop = img
        cands.append((name, crop))
    if not cands:
        raise RuntimeError(
            "no track_vis/frame_000000.png found for any candidate root"
        )

    panel = _compose_n_candidate_panel(cands)
    tmp_dir = episode_dir / "ground_trust_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    panel_path = tmp_dir / "ground_trust_panel.png"
    iio.imwrite(panel_path, panel)

    name_set = {n for n, _ in cands}
    if skip_vlm:
        chosen = sorted(name_set)[0]
        return chosen, f"(skip_vlm: defaulted to {chosen})", panel_path

    from ar2s.agent_configs.models import (
        chat_model_for, claude_to_lc_messages, parse_text,
    )
    llm = chat_model_for("geometry_prior.subagents.ground_trust")
    if llm is None:
        raise RuntimeError(
            "no credential for the YAML-selected ground_trust model provider"
        )
    names_list = ", ".join(n for n, _ in cands)
    msgs = claude_to_lc_messages([
        {"type": "text", "text": _VLM_PROMPT_GROUND_TRUST.format(
            n=len(cands), names=names_list,
        )},
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _encode_image(panel_path),
        }},
    ])
    resp = llm.invoke(msgs)
    text = parse_text(resp).strip()
    first_line = text.splitlines()[0].strip().rstrip(".:,").strip("\"'`")
    if first_line in name_set:
        return first_line, text, panel_path
    for n in name_set:
        if n.lower() in text.lower():
            return n, text, panel_path
    raise ValueError(
        f"VLM did not return a recognised object name (got: {first_line[:80]!r}); "
        f"allowed names: {sorted(name_set)}"
    )


# ---------------------------------------------------------------------------
# Manifest patching
# ---------------------------------------------------------------------------

def _patch_manifest_pos_offset_z(
    manifest_path: Path, object_name: str, pos_offset_z: float,
) -> None:
    data = yaml.safe_load(manifest_path.read_text()) or {}
    for obj in data.get("objects", []):
        if str(obj.get("name", "")) == object_name:
            obj["pos_offset"] = [0.0, 0.0, float(pos_offset_z)]
            break
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False))


def _patch_manifest_ground_offset(manifest_path: Path, ground_offset: float) -> None:
    """Patch ground.offset + force ground.reference = 'GROUND'.

    RobotSceneSim asserts ``ground.reference == 'GROUND'`` (the sentinel
    that means "ground plane is the world XY plane at -offset, not anchored
    to any object"). visual's ``ground_ref`` subagent sometimes writes a
    specific object name here as a hint; geometry_prior owns the final
    decision and always normalises to the sentinel.
    """
    data = yaml.safe_load(manifest_path.read_text()) or {}
    ground = data.setdefault("ground", {})
    ground["offset"] = float(ground_offset)
    ground["reference"] = "GROUND"
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False))


# ---------------------------------------------------------------------------
# Per-object driver
# ---------------------------------------------------------------------------

def axis_align_object(
    episode_dir: Path | str,
    run_root: Path | str,
    object_name: str,
    *,
    primary_id: int | None = None,
    decided: dict[str, dict],
    parent_name: str,
    state_dict: dict,
    skip_vlm: bool = False,
    force_idx: int | None = None,
    highlight: bool = _AXIS_HIGHLIGHT_DEFAULT,
    aux_cam: tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray] | None = None,
) -> AxisAlignResult:
    """One object: generate 6 candidates, VLM pick, apply to FP + manifest.

    ``aux_cam`` = (K, w2c, (H, W), real_rgb) of a second calibrated camera; when
    given, its real frame plus the same 6 candidates rendered from it are
    appended below the primary panel as a second anchor for flat-vs-standing.
    """
    episode_dir = Path(episode_dir).resolve()
    run_root = Path(run_root).resolve()

    manifest = yaml.safe_load((episode_dir / "manifest.yaml").read_text())
    if primary_id is None:
        primary_id = int(manifest["cameras"]["primary_id"])

    obj_dir = episode_dir / "objects" / object_name
    if not obj_dir.is_dir():
        return AxisAlignResult(
            object_name=object_name,
            error=f"object dir missing: {obj_dir}",
        )

    fp_dir = obj_dir / "foundation_pose"
    backup_fp_dir = obj_dir / "foundation_pose_pre_align"
    from ar2s.droid_sim.pose_store import exists, load_poses, save_poses
    if exists(backup_fp_dir):
        return AxisAlignResult(
            object_name=object_name,
            applied=False,
            skipped_reason=(
                f"foundation_pose_pre_align/ already exists at {backup_fp_dir}; "
                f"refusing to re-apply (would compound). Delete to force fresh run."
            ),
            raw_mesh_backup=str(backup_fp_dir),
        )
    if not exists(fp_dir):
        return AxisAlignResult(
            object_name=object_name,
            error=f"foundation_pose dir missing: {fp_dir}",
        )

    pose_obj_to_cam, frame_idx = _earliest_pose(obj_dir)
    K, cam_w2c = _load_camera(episode_dir, primary_id)
    pose_cam = pose_obj_to_cam
    c2w = np.linalg.inv(cam_w2c)
    pose_world = c2w @ pose_cam
    R_FP = pose_world[:3, :3]
    t_FP = pose_world[:3, 3].copy()

    V_body, F = _load_scaled_mesh(obj_dir)

    # Scheme C: root stays put (pos_offset.z = 0); the MuJoCo ground plane
    # moves to meet it via ground.offset (computed at episode level).
    # Children snap to their parent's top surface (preserves xy, adjusts z).
    is_root = (parent_name == _GROUND or parent_name not in decided)
    parent_top_z = 0.0 if is_root else float(decided[parent_name]["top_z_world"])

    candidates: list[dict] = []
    for axis_idx in (0, 1, 2):
        for sign in (+1, -1):
            body_axis_world = sign * R_FP[:, axis_idx]
            norm = np.linalg.norm(body_axis_world)
            if norm < 1e-12:
                continue
            body_axis_world = body_axis_world / norm

            R_world = _rodrigues_snap(body_axis_world, _DEFAULT_TARGET)
            R_FP_new = R_world @ R_FP

            V_world = (R_FP_new @ V_body.T).T + t_FP
            bottom_z_pre = float(V_world[:, 2].min())
            top_z_pre = float(V_world[:, 2].max())
            if is_root:
                pos_offset_z = 0.0                       # root: don't move
            else:
                pos_offset_z = parent_top_z - bottom_z_pre   # child: snap

            candidates.append({
                "label": _CANDIDATE_LABELS[axis_idx * 2 + (0 if sign > 0 else 1)],
                "axis_idx": axis_idx,
                "sign": sign,
                "R_world": R_world,
                "R_FP_new": R_FP_new,
                "pos_offset_z": pos_offset_z,
                "bottom_z_world": bottom_z_pre + pos_offset_z,
                "top_z_world": top_z_pre + pos_offset_z,
            })

    tmp_dir = obj_dir / "axis_align_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    panels: list[np.ndarray] = []
    for c in candidates:
        img = _render_scene_panel(
            episode_dir, primary_id,
            decided=decided,
            current_obj_name=object_name,
            current_R_FP_new=c["R_FP_new"],
            current_t_world=t_FP,
            current_pos_offset_z=c["pos_offset_z"],
            current_bottom_z_world=c["bottom_z_world"],
            current_V_body=V_body,
            current_F=F,
            tmp_dir=tmp_dir,
            tag=c["label"].replace("+", "p").replace("-", "n"),
            highlight=highlight,
        )
        panels.append(img)

    real_rgb = _load_real_rgb(episode_dir, frame_idx=0)
    seg_result = (state_dict.get("segmentation_result") or {}).get(object_name)
    mask, _ = _load_sam3_mask(run_root, seg_result)
    if mask is not None:
        if mask.shape != real_rgb.shape[:2]:
            mask_pil = Image.fromarray(mask * 255).resize(
                (real_rgb.shape[1], real_rgb.shape[0]), Image.NEAREST,
            )
            mask = (np.asarray(mask_pil) > 0).astype(np.uint8)
        real_with_overlay = _overlay_mask_outline(real_rgb, mask)
    else:
        real_with_overlay = real_rgb

    # Union-bbox crop: small objects (markers, pens) are a few pixels in
    # 1280×720 full scene → VLM can't judge orientation. Crop reference +
    # all 6 candidates to a common window = union(mask bbox, 6 candidate
    # projected bboxes) padded by 30%. Cells become object-zoomed and
    # cross-comparable.
    H, W = real_rgb.shape[:2]
    bboxes: list[tuple[int, int, int, int]] = []
    if mask is not None:
        mb = _mask_bbox(mask)
        if mb is not None:
            bboxes.append(mb)
    for c in candidates:
        t_with_offset = t_FP.copy()
        t_with_offset[2] += float(c["pos_offset_z"])
        bb = _project_world_pose_bbox(
            V_body, c["R_FP_new"], t_with_offset, cam_w2c, K, H, W,
        )
        if bb[2] > bb[0] and bb[3] > bb[1]:
            bboxes.append(bb)
    if bboxes:
        x0, y0, x1, y1 = _union_bbox(bboxes, W, H, pad_ratio=0.3, min_side=240)
        real_with_overlay = real_with_overlay[y0:y1, x0:x1]
        panels = [p[y0:y1, x0:x1] for p in panels]

    panel_img = _compose_seven_panel(
        real_with_overlay, panels,
        labels=[c["label"] for c in candidates],
    )

    # ---- auxiliary second-camera block (flat-vs-standing disambiguator) ----
    has_aux = False
    if aux_cam is not None:
        sec_K, sec_w2c, (secH, secW), sec_real = aux_cam
        try:
            sec_panels = [
                _render_scene_panel(
                    episode_dir, primary_id,
                    decided=decided,
                    current_obj_name=object_name,
                    current_R_FP_new=c["R_FP_new"],
                    current_t_world=t_FP,
                    current_pos_offset_z=c["pos_offset_z"],
                    current_bottom_z_world=c["bottom_z_world"],
                    current_V_body=V_body,
                    current_F=F,
                    tmp_dir=tmp_dir,
                    tag="sec_" + c["label"].replace("+", "p").replace("-", "n"),
                    highlight=highlight,
                    view_K=sec_K, view_w2c=sec_w2c, out_hw=(secH, secW),
                )
                for c in candidates
            ]
            sec_bboxes: list[tuple[int, int, int, int]] = []
            for c in candidates:
                t_off = t_FP.copy()
                t_off[2] += float(c["pos_offset_z"])
                bb = _project_world_pose_bbox(
                    V_body, c["R_FP_new"], t_off, sec_w2c, sec_K, secH, secW,
                )
                if bb[2] > bb[0] and bb[3] > bb[1]:
                    sec_bboxes.append(bb)
            sec_real_crop = sec_real
            if sec_bboxes:
                sx0, sy0, sx1, sy1 = _union_bbox(
                    sec_bboxes, secW, secH, pad_ratio=0.3, min_side=240)
                sec_panels = [p[sy0:sy1, sx0:sx1] for p in sec_panels]
                # zoom the second-view real photo to the same window (object
                # position is fixed; only orientation varies across candidates)
                sec_real_crop = sec_real[sy0:sy1, sx0:sx1]
            aux_block = _compose_aux_block(
                sec_panels, labels=[c["label"] for c in candidates],
                secondary_real=sec_real_crop)
            panel_img = _stack_primary_and_aux(panel_img, aux_block)
            has_aux = True
        except Exception as exc:                              # noqa: BLE001
            print(f"[axis_align] {object_name}: aux view render failed, "
                  f"primary-only ({type(exc).__name__}: {exc})")

    # ---- synthetic top-down block (up-vs-down / flat-vs-standing) ----
    has_topdown = False
    if _AXIS_TOPDOWN_DEFAULT:
        try:
            obj_extent = float((V_body.max(0) - V_body.min(0)).max())
            td_K, td_w2c, (tdH, tdW) = _topdown_camera(t_FP, obj_extent)
            td_panels = [
                _render_scene_panel(
                    episode_dir, primary_id,
                    decided=decided,
                    current_obj_name=object_name,
                    current_R_FP_new=c["R_FP_new"],
                    current_t_world=t_FP,
                    current_pos_offset_z=c["pos_offset_z"],
                    current_bottom_z_world=c["bottom_z_world"],
                    current_V_body=V_body,
                    current_F=F,
                    tmp_dir=tmp_dir,
                    tag="td_" + c["label"].replace("+", "p").replace("-", "n"),
                    highlight=highlight,
                    view_K=td_K, view_w2c=td_w2c, out_hw=(tdH, tdW),
                )
                for c in candidates
            ]
            td_bboxes: list[tuple[int, int, int, int]] = []
            for c in candidates:
                t_off = t_FP.copy()
                t_off[2] += float(c["pos_offset_z"])
                bb = _project_world_pose_bbox(
                    V_body, c["R_FP_new"], t_off, td_w2c, td_K, tdH, tdW,
                )
                if bb[2] > bb[0] and bb[3] > bb[1]:
                    td_bboxes.append(bb)
            if td_bboxes:
                tx0, ty0, tx1, ty1 = _union_bbox(
                    td_bboxes, tdW, tdH, pad_ratio=0.3, min_side=240)
                td_panels = [p[ty0:ty1, tx0:tx1] for p in td_panels]
            td_block = _compose_aux_block(
                td_panels, labels=[c["label"] for c in candidates],
                secondary_real=None,
                header="TOP-DOWN VIEW — same 6 candidates seen from straight "
                       "above (opening = ring up / disc down; line = flat, dot "
                       "= standing)")
            panel_img = _stack_primary_and_aux(panel_img, td_block)
            has_topdown = True
        except Exception as exc:                              # noqa: BLE001
            print(f"[axis_align] {object_name}: top-down view render failed "
                  f"({type(exc).__name__}: {exc})")

    panel_path = tmp_dir / f"axis_align_panel_{object_name}.png"
    iio.imwrite(panel_path, panel_img)

    if force_idx is not None:
        chosen_idx = int(force_idx)
        vlm_raw = f"(forced idx={chosen_idx})"
    elif skip_vlm:
        chosen_idx = 0
        vlm_raw = "(skip_vlm: defaulted to +X)"
    else:
        chosen_idx, vlm_raw = _judge_with_vlm_6(
            panel_path, object_name, highlight=highlight,
            has_aux=has_aux, has_topdown=has_topdown)

    chosen = candidates[chosen_idx]

    # Apply through pose_store so packed and legacy inputs both remain readable
    # and rewritten results use the packed, inode-friendly representation.
    poses = load_poses(fp_dir)
    save_poses(backup_fp_dir, poses)
    rewritten: dict[int, np.ndarray] = {}
    for pose_frame, pose_obj_to_cam in poses.items():
        pose_obj_to_world = c2w @ pose_obj_to_cam
        R_old = pose_obj_to_world[:3, :3]
        t_old = pose_obj_to_world[:3, 3]
        R_new = chosen["R_world"] @ R_old
        pose_world_new = pose_obj_to_world.copy()
        pose_world_new[:3, :3] = R_new
        pose_world_new[:3, 3] = t_old   # position preserved
        pose_cam_new = cam_w2c @ pose_world_new
        rewritten[pose_frame] = pose_cam_new.astype(np.float32)
    save_poses(fp_dir, rewritten)
    n_rewritten = len(rewritten)

    _patch_manifest_pos_offset_z(
        episode_dir / "manifest.yaml", object_name, chosen["pos_offset_z"],
    )

    return AxisAlignResult(
        object_name=object_name,
        applied=True,
        chosen_label=chosen["label"],
        R_world=chosen["R_world"],
        pos_offset_z=chosen["pos_offset_z"],
        bottom_z_world=chosen["bottom_z_world"],
        top_z_world=chosen["top_z_world"],
        keyframe=frame_idx,
        panel_path=str(panel_path),
        vlm_raw=vlm_raw,
        raw_mesh_backup=str(backup_fp_dir),
    )


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

def axis_align_episode(
    episode_dir: Path | str,
    run_root: Path | str,
    *,
    primary_id: int | None = None,
    skip_objects: tuple[str, ...] = ("robot",),
    skip_vlm: bool = False,
    highlight: bool = _AXIS_HIGHLIGHT_DEFAULT,
    aux_view: bool = _AXIS_AUX_DEFAULT,
) -> list[AxisAlignResult]:
    """Topological-order axis-align every non-robot object."""
    episode_dir = Path(episode_dir).resolve()
    run_root = Path(run_root).resolve()
    manifest_path = episode_dir / "manifest.yaml"

    aux_cam = _load_aux_camera(run_root, episode_dir) if aux_view else None
    if aux_cam is not None:
        print(f"[axis_align] auxiliary second-camera view ON "
              f"(K/extrinsics from raw_episodes; panel gets a 2nd-view block)")

    state = json.loads((run_root / "state.json").read_text())
    relations = (state.get("support_tree") or {}).get("relations") or []
    parent_of, children_of = _adjacency(relations)
    order = _topological_order(parent_of, children_of)

    bundle_objs = {
        str(o.get("name", ""))
        for o in (yaml.safe_load(manifest_path.read_text()) or {}).get("objects", [])
    }
    skip_set = {s.lower() for s in skip_objects}

    decided: dict[str, dict] = {}
    results: list[AxisAlignResult] = []

    for name in order:
        if name == _GROUND or name not in bundle_objs:
            continue
        if name.lower() in skip_set:
            continue
        try:
            r = axis_align_object(
                episode_dir, run_root, name,
                primary_id=primary_id,
                decided=decided,
                parent_name=parent_of.get(name, _GROUND),
                state_dict=state,
                skip_vlm=skip_vlm,
                highlight=highlight,
                aux_cam=aux_cam,
            )
        except Exception as exc:
            r = AxisAlignResult(
                object_name=name,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
        results.append(r)
        if r.applied:
            decided[name] = {
                "R_world": r.R_world,
                "pos_offset_z": r.pos_offset_z,
                "top_z_world": r.top_z_world,
                "bottom_z_world": r.bottom_z_world,
            }

    # ---- ground.offset derivation ----
    # support_tree's GROUND sentinel maps to one physical surface (typically
    # the table). The MuJoCo ground plane has a single z value.
    #
    #  • 0 roots → ground at world z=0 (degenerate).
    #  • 1 root  → scheme C: ground meets that root's bottom (preserves the
    #              root's FP-given world position).
    #  • >1 roots → FP-recovered bottoms typically disagree (different per-
    #               object scale / pose noise even though they sit on the same
    #               physical surface). Ask a VLM to pick the most-trustworthy
    #               root via its track_vis bbox overlay, use ITS bottom as the
    #               ground reference, and vertically snap every other root so
    #               its bottom touches the same plane (preserves xy, shifts z).
    ground_roots = [
        (name, dec) for name, dec in decided.items()
        if parent_of.get(name, _GROUND) == _GROUND
    ]
    if not ground_roots:
        ground_offset = 0.0
    elif len(ground_roots) == 1:
        ground_offset = float(-ground_roots[0][1]["bottom_z_world"])
    else:
        print(f"\n[geometry_prior] ④ ground-trust VLM on {len(ground_roots)} "
              f"GROUND-direct roots ...")
        root_names = [n for n, _ in ground_roots]
        if primary_id is None:
            _mfst = yaml.safe_load(manifest_path.read_text()) or {}
            _pid_ground = int(_mfst.get("cameras", {}).get("primary_id"))
        else:
            _pid_ground = primary_id
        try:
            trusted_name, vlm_raw, panel_path = _pick_trusted_ground_root(
                episode_dir, root_names,
                primary_id=_pid_ground, skip_vlm=skip_vlm,
            )
            print(f"  trusted root = {trusted_name}")
            print(f"  vlm panel    = {panel_path}")
            print(f"  vlm raw      = {vlm_raw[:300]}")
        except Exception as exc:
            # Fallback: pick smallest-height root (assume small objects are
            # most likely to sit accurately on the table; large meshes amplify
            # SAM3D scale errors).
            trusted_name = min(
                ground_roots,
                key=lambda kv: kv[1]["top_z_world"] - kv[1]["bottom_z_world"],
            )[0]
            print(f"  WARN: VLM picker failed ({type(exc).__name__}: {exc}); "
                  f"fallback → smallest-height root = {trusted_name}")

        trusted_bottom = next(
            d["bottom_z_world"] for n, d in ground_roots if n == trusted_name
        )
        ground_offset = float(-trusted_bottom)

        # Snap non-trusted roots so their bottoms align with trusted's ground.
        # Update manifest in place + propagate to decided + results.
        for name, dec in ground_roots:
            if name == trusted_name:
                continue
            shift_z = trusted_bottom - dec["bottom_z_world"]
            _patch_manifest_pos_offset_z(manifest_path, name, shift_z)
            dec["pos_offset_z"] = shift_z
            dec["bottom_z_world"] += shift_z
            dec["top_z_world"]    += shift_z
            for r in results:
                if r.object_name == name and r.applied:
                    r.pos_offset_z   = shift_z
                    r.bottom_z_world = dec["bottom_z_world"]
                    r.top_z_world    = dec["top_z_world"]
                    break
            print(f"  snap {name:<10s}  pos_offset.z={shift_z:+.4f}  "
                  f"→ bottom={dec['bottom_z_world']:+.4f}, "
                  f"top={dec['top_z_world']:+.4f}")

    # Only rewrite ground.offset when this run actually decided something.
    # On a re-entry (every object already has foundation_pose_pre_align/, so
    # each per-object call short-circuits with applied=False) `decided` is
    # empty, `ground_roots` is empty, and `ground_offset` defaults to 0.0 —
    # patching that would drop the MuJoCo plane while the objects keep their
    # existing offsets. `scene_view_repair` calls apply_geometry_priors a
    # second time, so this path is reachable in a normal run.
    if decided:
        _patch_manifest_ground_offset(manifest_path, ground_offset)
    return results


# ---------------------------------------------------------------------------
# Post-orient re-ground
# ---------------------------------------------------------------------------

def recompute_z_after_orient(
    episode_dir: Path | str,
    run_root: Path | str,
    *,
    primary_id: int | None = None,
    skip_objects: tuple[str, ...] = ("robot",),
) -> dict[str, dict]:
    """Re-derive ``pos_offset.z`` / ``ground.offset`` from the FINAL meshes.

    ``mesh_orient`` runs AFTER axis_align and may flip ``visual.obj`` 180°
    about its centroid, which moves the mesh's world bottom/top for any
    centroid-asymmetric object (24 mm observed on a mug). axis_align committed
    its z values from the pre-flip mesh, so they go stale.

    Pure geometry, no VLM: replays axis_align's z chain — trusted GROUND root
    keeps ``pos_offset.z = 0`` and sets ``ground.offset = -bottom``; other
    roots snap their bottoms to the trusted plane; children snap to their
    parent's top — against whatever ``visual.obj`` now contains. The FP
    per-frame poses already carry the axis-align rotation and are untouched
    by the flip, so re-reading them here is exact.

    The multi-root trusted-root *choice* is orientation-independent, so the
    VLM pick from axis_align is preserved rather than re-run: the trusted
    root is the GROUND-direct root whose current manifest ``pos_offset.z``
    is 0 (axis_align leaves the trusted root unshifted and snaps the rest);
    ties fall back to the smallest-height root, mirroring axis_align's own
    VLM-failure fallback.

    Idempotent: with no flips (or symmetric meshes) it recomputes the values
    axis_align already wrote.

    Returns {object_name: {"pos_offset_z", "bottom_z_world", "top_z_world"}}.
    """
    episode_dir = Path(episode_dir).resolve()
    run_root = Path(run_root).resolve()
    manifest_path = episode_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) or {}

    if primary_id is None:
        primary_id = int(manifest.get("cameras", {}).get("primary_id"))

    state = json.loads((run_root / "state.json").read_text())
    relations = (state.get("support_tree") or {}).get("relations") or []
    parent_of, children_of = _adjacency(relations)
    order = _topological_order(parent_of, children_of)

    bundle_objs = {str(o.get("name", "")) for o in manifest.get("objects", [])}
    manifest_offsets = {
        str(o.get("name", "")): float((o.get("pos_offset") or [0, 0, 0])[2])
        for o in manifest.get("objects", [])
    }
    skip_set = {s.lower() for s in skip_objects}

    K, cam_w2c = _load_camera(episode_dir, primary_id)
    c2w = np.linalg.inv(cam_w2c)

    # Pass 1 — raw world extents (pos_offset excluded), in topo order.
    raw: dict[str, dict] = {}
    for name in order:
        if name == _GROUND or name not in bundle_objs or name.lower() in skip_set:
            continue
        obj_dir = episode_dir / "objects" / name
        pose_cam, _ = _earliest_pose(obj_dir)
        pose_world = c2w @ pose_cam
        V_body, _F = _load_scaled_mesh(obj_dir)
        V_world = (pose_world[:3, :3] @ V_body.T).T + pose_world[:3, 3]
        raw[name] = {
            "bottom_pre": float(V_world[:, 2].min()),
            "top_pre": float(V_world[:, 2].max()),
        }

    if not raw:
        return {}

    # Trusted GROUND root: the one axis_align left unshifted (pos_offset.z==0);
    # ties -> smallest height (axis_align's own fallback rule).
    ground_roots = [n for n in raw if parent_of.get(n, _GROUND) == _GROUND]
    assert ground_roots, f"no GROUND-direct roots among {sorted(raw)}"
    unshifted = [n for n in ground_roots if abs(manifest_offsets.get(n, 0.0)) < 1e-9]
    pool = unshifted or ground_roots
    trusted = min(pool, key=lambda n: raw[n]["top_pre"] - raw[n]["bottom_pre"]) \
        if len(pool) > 1 else pool[0]
    trusted_bottom = raw[trusted]["bottom_pre"]

    # Pass 2 — replay the snap chain on the final meshes.
    decided: dict[str, dict] = {}
    for name in order:
        if name not in raw:
            continue
        is_root = parent_of.get(name, _GROUND) == _GROUND
        if name == trusted:
            pos_offset_z = 0.0
        elif is_root:
            pos_offset_z = trusted_bottom - raw[name]["bottom_pre"]
        else:
            parent = parent_of[name]
            assert parent in decided, (
                f"{name}: parent {parent!r} not processed before child "
                f"(support-tree order violated)"
            )
            pos_offset_z = decided[parent]["top_z_world"] - raw[name]["bottom_pre"]
        decided[name] = {
            "pos_offset_z": pos_offset_z,
            "bottom_z_world": raw[name]["bottom_pre"] + pos_offset_z,
            "top_z_world": raw[name]["top_pre"] + pos_offset_z,
        }
        if abs(pos_offset_z - manifest_offsets.get(name, 0.0)) > 1e-9:
            _patch_manifest_pos_offset_z(manifest_path, name, pos_offset_z)
            print(f"  [re-ground] {name:<12s} pos_offset.z "
                  f"{manifest_offsets.get(name, 0.0):+.4f} → {pos_offset_z:+.4f}")

    _patch_manifest_ground_offset(manifest_path, float(-trusted_bottom))
    return decided
