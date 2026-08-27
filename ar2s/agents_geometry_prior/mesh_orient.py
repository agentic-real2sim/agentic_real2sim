"""mesh_orient — VLM-driven orientation correction for per-object visual.obj.

Why this exists
---------------
SAM3D / FoundationPose can recover a mesh + pose pair that's *internally
consistent* (mesh registers cleanly to depth/RGB) but globally **flipped
180°** relative to ground truth — symmetric or weakly-textured objects
(bowls, mugs, lids, foil packets) have multiple equally-good registrations
and the picked one may be upside-down. We saw this on the snack v0 bowl,
and fixed it manually by body-frame `V * [1, -1, -1]` on the mesh.

This skill automates that decision per object:

  1. Render the mesh at its FoundationPose keyframe pose, from the primary
     camera, under FOUR orientation hypotheses (NONE / flip-X / flip-Y /
     flip-Z, where flip-axis means 180° rotation about that local axis).
  2. Compose a 5-panel comparison image: real RGB cropped to the object
     region + 4 rendered candidates same crop.
  3. Hand to a VLM, ask which candidate best matches the real photo's
     orientation. Output is one of {NONE, X, Y, Z}.
  4. If non-NONE, overwrite ``<episode_dir>/objects/<obj>/visual.obj``
     with the flipped vertices. The original is backed up to
     ``visual_pre_orient.obj`` next to it.

Body-frame flip preserves the FoundationPose pose: vertices rotate 180°
about the local axis passing through the **mesh centroid**, so the object
center stays put in world; only its visible orientation changes. Existing
pose .npy files don't need to be regenerated.

Pivoting about the mesh centroid (not the pose origin) matters because
FoundationPose's body-frame origin is generally offset from the mesh
centroid by tens of mm — rotating about the pose origin instead shifts
the visible centroid by ``2*|centroid_local|`` in world, which regressed
pose_view_critic in several bowl / pen cases.

Renderer: minimal MuJoCo MJCF (1 body + 1 mesh + 1 camera). Camera frame
convention matches RobotSceneSim (OpenCV→MuJoCo `R_flip = diag(1,-1,-1)`,
``fovy = 2 atan(H/2fy)``).

VLM: configured by `geometry_prior.subagents.mesh_orient` in
`ar2s/agent_configs/config_anthropic_opus47.yaml`.
"""
from __future__ import annotations

import base64
import io
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as iio
import mujoco

from ar2s.droid_sim._util import sanitize_name as _sanitize
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


_FLIPS: dict[str, np.ndarray] = {
    "NONE": np.eye(3),
    "X":    np.diag([1.0, -1.0, -1.0]),  # 180° about local X
    "Y":    np.diag([-1.0, 1.0, -1.0]),  # 180° about local Y
    "Z":    np.diag([-1.0, -1.0, 1.0]),  # 180° about local Z
}
_FLIP_KEYS = ("NONE", "X", "Y", "Z")


@dataclass
class OrientResult:
    object_name: str
    chosen_flip: str = "NONE"
    panel_image: str = ""             # path
    raw_mesh_backup: str = ""         # path to pre-flip mesh (only set if flipped)
    vlm_raw: str = ""
    keyframe: int = 0
    error: str = ""


# =========================================================================
# Episode/asset loading
# =========================================================================



def _earliest_pose(obj_dir: Path) -> tuple[np.ndarray, int]:
    """Return (T_cam_from_obj, frame_index) for the first available FP transform."""
    from ar2s.droid_sim.pose_store import earliest_frame, load_pose
    fp_dir = obj_dir / "foundation_pose"
    frame_idx = earliest_frame(fp_dir)
    if frame_idx is None:
        raise FileNotFoundError(f"no FoundationPose results under {obj_dir}")
    pose = load_pose(fp_dir, frame_idx)
    if pose is None:
        raise FileNotFoundError(
            f"FoundationPose frame {frame_idx} missing under {fp_dir}"
        )
    return pose, frame_idx


def _load_real_rgb(episode_dir: Path, frame_idx: int) -> np.ndarray:
    """Frame `frame_idx` of the real video (left half if stereo SBS).

    Decode from ``real/video.mp4`` — NOT ``real/first_frame.png``. The latter
    is the FoundationPose track_vis frame with the 3D bbox + body XYZ axes
    (red/green/blue) + status text baked in; those overlays clutter and occlude
    the object and mislead the orient VLM (it can read an axis line as part of
    the object's shape). The clean video frame is what the orient panel needs.
    """
    video = episode_dir / "real" / "video.mp4"
    with iio.get_reader(str(video)) as r:
        for i, frame in enumerate(r):
            if i == frame_idx:
                return frame[..., :3]
    raise IndexError(f"video has < {frame_idx + 1} frames: {video}")


def _load_camera(episode_dir: Path, primary_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, cam_w2c). cam_w2c is the 'refined' extrinsics as stored."""
    K = np.fromstring(
        (episode_dir / "cameras" / "intrinsics.txt").read_text().splitlines()[0],
        sep=" ",
    )[:9].reshape(3, 3).astype(np.float64)
    npz = np.load(episode_dir / "cameras" / "extrinsics.npz")
    cam_w2c = np.asarray(npz[f"cam_mat_{primary_id}"], dtype=np.float64).reshape(4, 4)
    return K, cam_w2c


# =========================================================================
# Rendering — minimal MuJoCo scene with 1 mesh + 1 camera
# =========================================================================

def _bbox_of_projected_mesh(
    verts_mesh: np.ndarray, pose_obj_to_cam: np.ndarray, K: np.ndarray,
    H: int, W: int, pad: int = 10,
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) bbox of mesh's projection in image. Clamped + padded."""
    V = (pose_obj_to_cam[:3, :3] @ verts_mesh.T).T + pose_obj_to_cam[:3, 3]
    V = V[V[:, 2] > 1e-4]
    if V.size == 0:
        return 0, 0, W, H
    uv = (K @ V.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x0 = int(np.floor(uv[:, 0].min())) - pad
    y0 = int(np.floor(uv[:, 1].min())) - pad
    x1 = int(np.ceil(uv[:, 0].max())) + pad
    y1 = int(np.ceil(uv[:, 1].max())) + pad
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, W, H
    return x0, y0, x1, y1


def _build_render_mjcf(
    mesh_obj_path: Path,
    body_pos: np.ndarray, body_quat_wxyz: np.ndarray,
    cam_pos: np.ndarray, cam_quat_wxyz: np.ndarray,
    fovy_deg: float,
    offscreen_side: int,
) -> str:
    """Tiny MJCF: 1 camera, 1 body+mesh, ambient lighting. No physics."""
    bp = " ".join(f"{x:.6f}" for x in body_pos)
    bq = " ".join(f"{x:.6f}" for x in body_quat_wxyz)
    cp = " ".join(f"{x:.6f}" for x in cam_pos)
    cq = " ".join(f"{x:.6f}" for x in cam_quat_wxyz)
    return f"""<mujoco>
  <visual>
    <global offwidth="{offscreen_side}" offheight="{offscreen_side}"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7" specular="0.0 0.0 0.0"/>
  </visual>
  <asset>
    <mesh name="obj_mesh" file="{mesh_obj_path}"/>
  </asset>
  <worldbody>
    <camera name="cam0" pos="{cp}" quat="{cq}" fovy="{fovy_deg:.4f}"/>
    <body name="obj_body" pos="{bp}" quat="{bq}">
      <geom type="mesh" mesh="obj_mesh" rgba="0.8 0.8 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _render_mesh(
    verts: np.ndarray, faces: np.ndarray,
    pose_obj_to_world: np.ndarray,
    cam_w2c: np.ndarray, K: np.ndarray,
    H: int, W: int,
    tag: str,
) -> np.ndarray:
    """Render mesh placed at world pose, viewed from a camera with extrinsics cam_w2c
    and intrinsics K. Returns (H, W, 3) uint8 RGB. Builds a fresh MJCF per call —
    cheap (<0.1 s) and we anyway need different mesh per candidate."""
    # MuJoCo's asset loader only takes a file path, so each candidate's vertices
    # must round-trip through a .obj. Those are full-resolution meshes (~13 MB
    # each, one per flip hypothesis) with no value once the panel is composed,
    # so they go to scratch rather than into the emitted episode bundle.
    with tempfile.TemporaryDirectory(prefix=f"mesh_orient_{tag}_") as scratch:
        mesh_obj_path = Path(scratch) / f"mesh_{tag}.obj"
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(
            str(mesh_obj_path)
        )

        body_pos = pose_obj_to_world[:3, 3]
        body_quat = np.empty(4)
        mujoco.mju_mat2Quat(body_quat, pose_obj_to_world[:3, :3].flatten())

        # Square frame at S = max-symmetric image side around principal point,
        # then crop to (H, W) using K's principal point — same trick as
        # tmp/snack_sim/run_sim.py:117.
        S = 2 * int(np.ceil(max(K[0, 2], W - K[0, 2], K[1, 2], H - K[1, 2])))
        S += S % 2

        # Camera world pose, OpenCV → MuJoCo convention. Render at offscreen
        # square side S so fovy must cover S (we crop to H × W after render).
        from ar2s.droid_sim._util import (
            opencv_c2w_to_mujoco_camera_pose, fovy_deg_from_K,
        )
        c2w = np.linalg.inv(cam_w2c)
        cam_pos, cam_quat = opencv_c2w_to_mujoco_camera_pose(c2w)
        fovy_deg = fovy_deg_from_K(K, image_height=S)

        xml = _build_render_mjcf(
            mesh_obj_path, body_pos, body_quat, cam_pos, cam_quat, fovy_deg,
            offscreen_side=S,
        )
        spec = mujoco.MjSpec.from_string(xml)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ren = mujoco.Renderer(model, S, S)
        ren.update_scene(data, camera="cam0")
        img = ren.render()
        ren.close()
        x0 = int(round(S / 2 - K[0, 2]))
        y0 = int(round(S / 2 - K[1, 2]))
        return img[y0:y0 + H, x0:x0 + W]


# =========================================================================
# Panel composition
# =========================================================================

def _label_strip(label: str, W: int, h: int = 28) -> np.ndarray:
    """White strip with centred black text — title above each panel."""
    img = Image.new("RGB", (W, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16,
        )
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 4), label, fill=(0, 0, 0), font=font)
    return np.asarray(img)


def _crop_and_resize(img: np.ndarray, bbox: tuple[int, int, int, int], side: int) -> np.ndarray:
    """Crop to bbox then resize to (side, side) preserving aspect with letterbox."""
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = side / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(crop).resize((new_w, new_h), Image.BICUBIC)
    )
    pad = np.full((side, side, 3), 255, dtype=np.uint8)
    py = (side - new_h) // 2
    px = (side - new_w) // 2
    pad[py:py + new_h, px:px + new_w] = resized
    return pad


def _compose_panel(real_rgb: np.ndarray, candidates: dict[str, np.ndarray],
                   bbox: tuple[int, int, int, int], panel_side: int = 320) -> np.ndarray:
    """Build a horizontal strip: [REFERENCE, NONE, X, Y, Z]. Each square = panel_side."""
    cols: list[np.ndarray] = []
    for name, img in [("REFERENCE", real_rgb)] + [(k, candidates[k]) for k in _FLIP_KEYS]:
        cropped = _crop_and_resize(img, bbox, panel_side)
        label = _label_strip(name, panel_side)
        cols.append(np.vstack([label, cropped]))
    # 4-px black separators between panels
    sep = np.zeros((cols[0].shape[0], 4, 3), dtype=np.uint8)
    out = cols[0]
    for c in cols[1:]:
        out = np.hstack([out, sep, c])
    return out


# =========================================================================
# VLM judgment
# =========================================================================

_VLM_PROMPT = """You are inspecting a 3D mesh recovered from a single-view RGB image
for an object named "{name}".

The leftmost panel labelled REFERENCE shows the real object photographed by
the scene's primary camera. The four panels to the right (NONE, X, Y, Z)
show the recovered mesh rendered FROM THE SAME CAMERA at the recovered
pose, under four orientation hypotheses:
  - NONE: mesh as-is.
  - X: mesh flipped 180° about its own local X axis.
  - Y: mesh flipped 180° about its own local Y axis.
  - Z: mesh flipped 180° about its own local Z axis.

ONE of these four candidate orientations matches the REFERENCE photo. You
must pick exactly one. Focus on coarse orientation cues:
  - For a bowl/cup/mug: opening up vs opening down.
  - For a flat package/box: which face is up vs which is the side.
  - For an elongated object: which end points which direction.
Texture/colour/lighting differences between mesh and photo are EXPECTED
and should be ignored — only shape and orientation matter.

Respond with EXACTLY one word on the first line: NONE, X, Y, or Z.
You may add a short justification on subsequent lines.
"""


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _judge_with_vlm(panel_path: Path, object_name: str) -> tuple[str, str]:
    """Return (choice, raw_vlm_text). choice ∈ {NONE,X,Y,Z}; raises on no key."""
    from ar2s.agent_configs.models import chat_model_for, claude_to_lc_messages, parse_text
    llm = chat_model_for("geometry_prior.subagents.mesh_orient")
    if llm is None:
        raise RuntimeError(
            "no credential for the YAML-selected mesh_orient model provider"
        )
    msgs = claude_to_lc_messages([
        {"type": "text", "text": _VLM_PROMPT.format(name=object_name)},
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _encode_image(panel_path),
        }},
    ])
    resp = llm.invoke(msgs)
    text = parse_text(resp).strip()
    first_line = text.splitlines()[0].strip().upper().rstrip(".:,")
    choice = first_line if first_line in _FLIP_KEYS else "NONE"
    return choice, text


# =========================================================================
# Per-object driver
# =========================================================================

def orient_object(
    episode_dir: Path | str,
    object_name: str,
    *,
    primary_id: int | None = None,
    tmp_root: Path | str | None = None,
    panel_side: int = 320,
    skip_vlm: bool = False,
    force_choice: str | None = None,
) -> OrientResult:
    """Decide orientation for one object; if non-NONE, overwrite visual.obj.

    Args:
        episode_dir: built episode (with manifest, cameras/, objects/).
        object_name: sanitised name of the object subdir under ``objects/``.
        primary_id: camera serial; defaults to manifest's ``cameras.primary_id``.
        tmp_root: where to save the comparison panel PNG. Defaults to the
            object dir itself. The per-hypothesis meshes are not written
            here — they go to scratch inside ``_render_mesh`` and are
            dropped after the render.
        skip_vlm: build the panel + return without calling VLM (debug).
        force_choice: override the VLM (debug). One of NONE/X/Y/Z.
    """
    import yaml
    episode_dir = Path(episode_dir).resolve()
    manifest = yaml.safe_load((episode_dir / "manifest.yaml").read_text())
    if primary_id is None:
        primary_id = int(manifest["cameras"]["primary_id"])

    obj_dir = episode_dir / "objects" / object_name
    if not obj_dir.is_dir():
        return OrientResult(object_name=object_name,
                            error=f"object dir missing: {obj_dir}")

    pose_obj_to_cam, frame_idx = _earliest_pose(obj_dir)
    K, cam_w2c = _load_camera(episode_dir, primary_id)
    c2w = np.linalg.inv(cam_w2c)
    pose_obj_to_world = c2w @ pose_obj_to_cam

    real_rgb = _load_real_rgb(episode_dir, frame_idx)
    H, W = real_rgb.shape[:2]

    # Load mesh, apply per-axis scale. process=False + maintain_order=True keep
    # the vertex list byte-for-byte as authored — trimesh's default processing
    # merges coincident vertices, which desynchronises the per-vertex colours
    # the USD exporter reads back out of visual.obj.
    mesh = trimesh.load(
        obj_dir / "visual.obj", force="mesh", process=False, maintain_order=True
    )
    V_raw = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if (obj_dir / "scale.npy").exists():
        s = np.asarray(np.load(obj_dir / "scale.npy"), dtype=np.float64).reshape(-1)
        if s.size == 1:
            V_scaled = V_raw * float(s[0])
        else:
            V_scaled = V_raw * s[:3]
    else:
        V_scaled = V_raw

    bbox = _bbox_of_projected_mesh(V_scaled, pose_obj_to_cam, K, H, W, pad=15)

    # Render 4 candidates. Only the composed panel is kept, and it lands
    # directly in the object dir — the per-hypothesis meshes live in scratch
    # inside _render_mesh, so there is nothing left to warrant a subdir.
    tmp_dir = Path(tmp_root) if tmp_root else obj_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, np.ndarray] = {}
    c_scaled = V_scaled.mean(axis=0)     # mesh centroid — pivot for flip
    for key in _FLIP_KEYS:
        R_off = _FLIPS[key]
        # Rotate 180° about the mesh centroid so the visible object center
        # stays put; rotating about the pose origin shifts the visible mesh.
        V_var = (V_scaled - c_scaled) @ R_off.T + c_scaled
        candidates[key] = _render_mesh(
            V_var, F, pose_obj_to_world, cam_w2c, K, H, W, key.lower(),
        )

    panel = _compose_panel(real_rgb, candidates, bbox, panel_side=panel_side)
    panel_path = tmp_dir / f"orient_panel_{_sanitize(object_name)}.png"
    iio.imwrite(panel_path, panel)

    if skip_vlm:
        return OrientResult(
            object_name=object_name, chosen_flip="NONE",
            panel_image=str(panel_path), keyframe=frame_idx,
        )

    if force_choice:
        choice, raw = force_choice.upper(), f"forced: {force_choice}"
    else:
        choice, raw = _judge_with_vlm(panel_path, object_name)

    result = OrientResult(
        object_name=object_name, chosen_flip=choice,
        panel_image=str(panel_path), vlm_raw=raw, keyframe=frame_idx,
    )

    if choice != "NONE":
        R_off = _FLIPS[choice]
        # Back up + overwrite using raw (un-scaled) vertices so scale.npy
        # semantics stay identical.
        backup = obj_dir / "visual_pre_orient.obj"
        if not backup.exists():
            (obj_dir / "visual.obj").rename(backup)
        # Rotate 180° about the mesh centroid — see module docstring +
        # candidates loop above. Pivoting about the local origin shifts the
        # visible centroid by 2*|centroid|; pivoting about the centroid keeps
        # the visible object where it is and only changes its orientation.
        # Transform the loaded mesh in place rather than rebuilding a bare
        # Trimesh from (V, F): a rebuilt mesh carries no visual, so the export
        # writes `v x y z` with no RGB and usd_exporter's vertex-colour read
        # falls back to visual_pre_orient.obj. Numerically identical to
        # (V_raw - c) @ R_off.T + c.
        c_raw = V_raw.mean(axis=0)
        T_off = np.eye(4, dtype=np.float64)
        T_off[:3, :3] = R_off
        T_off[:3, 3] = c_raw - R_off @ c_raw
        mesh.apply_transform(T_off)
        mesh.export(str(obj_dir / "visual.obj"))
        result.raw_mesh_backup = str(backup)

    return result


# =========================================================================
# Per-episode driver
# =========================================================================

def orient_episode(
    episode_dir: Path | str,
    *,
    objects: list[str] | None = None,
    skip_vlm: bool = False,
    force_choices: dict[str, str] | None = None,
) -> dict[str, OrientResult]:
    """Run orient_object across every object subdir under ``objects/``.

    ``robot`` is skipped automatically. Returns name → result dict.
    """
    import yaml
    episode_dir = Path(episode_dir).resolve()
    manifest = yaml.safe_load((episode_dir / "manifest.yaml").read_text())
    primary_id = int(manifest["cameras"]["primary_id"])

    if objects is None:
        objects = [
            p.name for p in sorted((episode_dir / "objects").iterdir())
            if p.is_dir() and p.name != "robot" and (p / "visual.obj").exists()
        ]

    out: dict[str, OrientResult] = {}
    for name in objects:
        fc = (force_choices or {}).get(name)
        try:
            out[name] = orient_object(
                episode_dir, name,
                primary_id=primary_id,
                skip_vlm=skip_vlm,
                force_choice=fc,
            )
        except Exception as e:
            out[name] = OrientResult(object_name=name, error=str(e))
    return out


__all__ = [
    "OrientResult", "orient_object", "orient_episode",
    "_FLIPS", "_FLIP_KEYS",
]
