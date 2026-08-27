"""Materialise an episode folder + manifest.yaml from VisualState (disk-state).

This is the ar2s.agents_visual equivalent of ar2s.agents_sysid's Stage 1 (Episode
Agent + build_episode). We already know the role of every artifact our
toolkit produced, so we skip the LLM and emit the structured folder
directly:

    outputs/run_pipeline/<run_id>/sysid_inputs/
        manifest.yaml
        robot_traj.h5
        objects/<name>/{visual.obj, scale.npy, foundation_pose.npz}
        cameras/{intrinsics.txt, extrinsics.npz}
        real/{first_frame.png, robot_mask.png, video.mp4}

The emitted folder is consumed by ``ar2s.agents_sysid.episode.load_episode``;
its strict validator catches any field/path mismatch we'd otherwise miss.

Reads (all dict access — disk-state):
  - state.entry (dict; task_description optional)
  - state.scene_name / pickup_objects / ground_reference_object
  - state.robot_traj_path
  - state.robot_mask_path / first_frame_rgb_path
  - state.cameras_intrinsics_path / cameras_extrinsics_path
  - state.camera_ids / object_camera_id
  - state.resolved_objects (list[dict])
  - state.stages.video_build.outputs.mp4_path

Run the complete pipeline with:
    python -m ar2s.run_pipeline --input outputs/run_pipeline/<run_id>/visual_<id>.py
"""
import os
import shutil
from pathlib import Path

import yaml

from ar2s.agents_visual.state import load_state
from ar2s.droid_sim._util import earliest_pose_frame as _earliest_pose_frame_or_none
from ar2s.droid_sim.scene.robot_profile import get_robot_profile


def _earliest_pose_frame(pose_source_path: str) -> int:
    """Earliest FP frame index, or 0 if missing/empty.

    Thin wrapper around ``ar2s.droid_sim._util.earliest_pose_frame`` that maps
    the shared helper's None return to 0 — the manifest schema wants an
    int regardless of whether the FP dir is populated.
    """
    if not pose_source_path:
        return 0
    return _earliest_pose_frame_or_none(pose_source_path) or 0


SCHEMA_VERSION = 1

# Fallback when the VisualInput predates the robot_type field. Every DROID
# episode is a Franka, so a missing field means Franka; self-collected episodes
# carry their own robot_type (VisualInput.robot_type, filled by
# build_raw_episode from the raw_data stage_manifest).
DEFAULT_ROBOT_TYPE = "franka_panda"


_REQUIRED_STATE = (
    "scene_name", "robot_traj_path", "resolved_objects",
    "ground_reference_object",
    "robot_mask_path", "first_frame_rgb_path", "pickup_objects",
    "cameras_intrinsics_path", "cameras_extrinsics_path",
    "camera_ids", "object_camera_id",
)


def write_episode_folder(run_root: str, target_dir: str) -> str:
    """Build the episode folder at ``target_dir`` and write manifest.yaml.

    Loads state from ``<run_root>/state.json`` and emits an episode folder
    ready for ``ar2s.agents_sysid.episode.load_episode``.

    Returns the resolved absolute path to ``target_dir``.
    """
    state = load_state(run_root)

    missing = [k for k in _REQUIRED_STATE if k not in state]
    if missing:
        raise ValueError(f"VisualState missing required fields for emit: {missing}")

    video_build = state.get("stages", {}).get("video_build", {})
    mp4_path = video_build.get("outputs", {}).get("mp4_path")
    if not mp4_path:
        raise ValueError("stages.video_build.outputs.mp4_path missing — run video_build first")

    if not state["resolved_objects"]:
        raise ValueError("resolved_objects is empty; nothing to emit")

    _assert_first_frame_matches_primary(state)

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Visual meshes: trimesh convert to .obj at the destination
    for obj in state["resolved_objects"]:
        src = Path(obj["visual_mesh_path"])
        if not src.exists():
            raise FileNotFoundError(f"visual mesh not found for {obj['name']!r}: {src}")
        _convert_visual_to_obj(src, target / _visual_dst(obj))

    file_moves = _build_file_moves(state, mp4_path)
    _execute_file_moves(file_moves, target)

    manifest = _build_manifest(state)
    manifest_path = target / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, default_flow_style=False, sort_keys=False)

    return str(target.resolve())


# View-mismatch gate: track_vis overlay drawings add noise, but same-view
# comparisons land well under this while a different physical camera lands
# in the thousands (measured: AUTOLab same-view ≈ 90, GuptaLab manual-retry
# cross-view ≈ 4788).
_FIRST_FRAME_VIEW_MSE_MAX = 1000.0


def _assert_first_frame_matches_primary(state: dict) -> None:
    """Fail finalize when pose_tracking ran on a different camera view than
    the declared primary video.

    OPT-IN (default OFF): set ``AR2S_STRICT_EMIT_GATE=1`` to enable. The
    check's two built-in assumptions — the compared view is the *primary*
    stream, and the first tracked frame is raw frame 0 — are both broken by
    features in this same change set (dual-view tracking can put an object's
    track_vis on the secondary view; ``_early_frame_with_mask`` accepts
    ``init_frame`` in 1-4, several raw frames of arm motion away), so as a
    default-on gate it hard-fails otherwise-correct runs at the last stage.

    Root cause this guards against: a manual --camera retry (or stale remote
    work/ copy) leaving old-camera content under the primary video's name, so
    FoundationPose silently tracks the wrong view (seen on
    GuptaLab_success_2023_04_20_13_13_17). ``first_frame_rgb_path`` is
    pose_tracking's frame-0 overlay — same view as whatever FP consumed —
    so a pixel-MSE against the primary stream's frame 0 catches the mismatch.

    Skips silently when either image is unreadable (the gate is
    defense-in-depth, not a new hard dependency).
    """
    if os.environ.get("AR2S_STRICT_EMIT_GATE", "") != "1":
        return
    first_frame_path = state.get("first_frame_rgb_path") or ""
    stereo_path = state.get("stereo_stream_path") or ""
    if not first_frame_path or not stereo_path:
        return
    if not Path(first_frame_path).is_file() or not Path(stereo_path).is_file():
        return
    try:
        import imageio.v2 as iio
        import numpy as np

        overlay = np.asarray(iio.imread(first_frame_path))[..., :3]
        reader = iio.get_reader(stereo_path)
        try:
            raw0 = np.asarray(reader.get_data(0))[..., :3]
        finally:
            reader.close()
    except Exception as exc:  # noqa: BLE001 — never block emit on IO quirks
        print(f"[emit] WARNING: first-frame view check skipped ({exc})")
        return

    if raw0.shape[1] >= 2 * raw0.shape[0]:      # side-by-side stereo → left half
        raw0 = raw0[:, : raw0.shape[1] // 2]
    h = min(overlay.shape[0], raw0.shape[0])
    w = min(overlay.shape[1], raw0.shape[1])
    if h == 0 or w == 0:
        return
    mse = float(np.mean(
        (overlay[:h, :w].astype(np.float32) - raw0[:h, :w].astype(np.float32)) ** 2
    ))
    if mse > _FIRST_FRAME_VIEW_MSE_MAX:
        raise ValueError(
            f"pose_tracking view mismatch: first_frame_rgb_path "
            f"({first_frame_path}) does not match frame 0 of the primary "
            f"stream ({stereo_path}); pixel MSE {mse:.0f} > "
            f"{_FIRST_FRAME_VIEW_MSE_MAX:.0f}. FoundationPose likely tracked "
            f"a different camera than the declared primary — check for stale "
            f"seq/ or work/ copies from a --camera retry."
        )
    print(f"[emit] first-frame view check OK (MSE {mse:.0f})")


# ---------------------------------------------------------------------------
# File move plan (src absolute -> dst relative-to-episode-dir)
# ---------------------------------------------------------------------------

def _visual_dst(obj: dict) -> str:
    """Canonical destination — always .obj.

    Downstream sysid (load_obj_vertices etc.) parses .obj as text. Source
    meshes from sam3d-objects are .glb; we convert at emit time so the
    manifest's `visual:` always points to a sysid-readable file.
    """
    return f"objects/{obj['name']}/visual.obj"


def _convert_visual_to_obj(src: Path, dst: Path) -> None:
    """Load any trimesh-supported mesh format and export as .obj at dst.

    Plain copy when src is already .obj — avoids the trimesh roundtrip and
    keeps the conversion runnable without trimesh installed.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".obj":
        shutil.copy2(src, dst)
        return

    import trimesh
    mesh = trimesh.load(src, force="mesh")
    if not hasattr(mesh, "export"):
        raise ValueError(f"trimesh did not return an exportable mesh from {src}")
    mesh.export(dst)


def _build_file_moves(state: dict, mp4_path: str) -> list[dict]:
    """Plain copies. Visual meshes are NOT in here — they need conversion."""
    moves: list[dict] = [
        {"src": str(Path(state["robot_traj_path"]).resolve()),
         "dst": "robot_traj.h5", "type": "file"},

        {"src": state["cameras_intrinsics_path"],
         "dst": "cameras/intrinsics.txt", "type": "file"},
        {"src": str(Path(state["cameras_extrinsics_path"]).resolve()),
         "dst": "cameras/extrinsics.npz", "type": "file"},

        {"src": state["first_frame_rgb_path"],
         "dst": "real/first_frame.png", "type": "file"},
        {"src": mp4_path,
         "dst": "real/video.mp4", "type": "file"},
    ]
    # robot_mask is optional: resolve() returns "" when SAM3 (incl. Tier-1
    # synonyms) cannot ground the robot; downstream pose-alignment then skips
    # the robot-IoU term but otherwise proceeds.
    if state.get("robot_mask_path"):
        moves.append({
            "src": state["robot_mask_path"],
            "dst": "real/robot_mask.png", "type": "file",
        })

    # Per-object track_vis dirs (FoundationPose's pose-overlay PNGs) come from
    # state.stages.pose_tracking, NOT resolved_objects (resolve only kept the
    # packed pose source). Look them up here so the emitted bundle includes
    # the visualisation alongside the poses.
    track_vis_by_name: dict[str, str] = {}
    pt_outputs = (state.get("stages", {}).get("pose_tracking") or {}).get("outputs") or {}
    for po in pt_outputs.get("per_object", []):
        if po.get("success") and po.get("track_vis_dir"):
            track_vis_by_name[po["object_name"]] = po["track_vis_dir"]

    for obj in state["resolved_objects"]:
        moves.append({"src": obj["scale_path"],
                      "dst": f"objects/{obj['name']}/scale.npy", "type": "file"})
        # Re-packed rather than copied: pose_source_path resolves in either
        # on-disk layout, so this normalises a legacy per-frame dir to the
        # single archive the bundle ships.
        moves.append({"src": obj["pose_source_path"],
                      "dst": f"objects/{obj['name']}/foundation_pose.npz",
                      "type": "poses"})
        tv = track_vis_by_name.get(obj["name"])
        if tv and Path(tv).is_dir():
            moves.append({"src": tv,
                          "dst": f"objects/{obj['name']}/track_vis", "type": "dir"})

    return moves


def _execute_file_moves(moves: list[dict], target: Path) -> None:
    seen_dsts: set[str] = set()
    for fm in moves:
        if fm["dst"] in seen_dsts:
            raise ValueError(f"duplicate dst in file_moves: {fm['dst']!r}")
        seen_dsts.add(fm["dst"])

        src = Path(fm["src"])
        if not src.exists():
            raise FileNotFoundError(f"file_moves src not found: {src} (-> {fm['dst']})")

        dst = target / fm["dst"]
        dst.parent.mkdir(parents=True, exist_ok=True)

        if fm["type"] == "poses":
            from ar2s.droid_sim.pose_store import load_poses, save_poses
            poses = load_poses(src)
            if not poses:
                raise FileNotFoundError(f"no FoundationPose poses under {src}")
            save_poses(dst, poses)
        elif fm["type"] == "dir":
            if not src.is_dir():
                raise ValueError(f"expected dir, got file: {src}")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=False)
        else:
            if not src.is_file():
                raise ValueError(f"expected file, got dir: {src}")
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Manifest dict (matches ar2s.agents_sysid.episode._validate schema v1)
# ---------------------------------------------------------------------------

def _build_manifest(state: dict) -> dict:
    objects = []
    for obj in state["resolved_objects"]:
        entry = {
            "name":                obj["name"],
            "visual":              _visual_dst(obj),
            "scale":               f"objects/{obj['name']}/scale.npy",
            # Key name kept for manifest-schema compatibility; the value is
            # now the packed archive. pose_store resolves either form.
            "foundation_pose_dir": f"objects/{obj['name']}/foundation_pose.npz",
            "init_frame":          _earliest_pose_frame(obj.get("pose_source_path", "")),
            "friction":            [float(x) for x in obj.get("friction", (1.0, 0.005, 0.0001))],
        }
        # ``mass`` is omitted entirely when neither resolve.mass_hints nor
        # physical_prior has set it. Sysid's validator catches that and tells
        # the user to run physical_prior first. Never write a default
        # placeholder here — downstream would treat it as truth.
        mass = obj.get("mass")
        if mass is not None:
            entry["mass"] = float(mass)
        objects.append(entry)

    pickup = list(state["pickup_objects"])
    entry = state.get("entry") or {}
    description = entry.get("task_description") or f"pickup {', '.join(pickup)}"

    # manifest.robot is what sysid reads to pick the RobotProfile, so resolve it
    # here rather than emitting a constant: a wrong robot.type does not fail, it
    # silently simulates a different arm replaying this arm's joint angles.
    # get_robot_profile raises on an unknown name — better here, while the
    # episode folder is being written, than deep inside the grasp stage.
    robot_profile = get_robot_profile(entry.get("robot_type") or DEFAULT_ROBOT_TYPE)

    # robot_mask is conditional: SAM3 can fail to ground the robot (no mask),
    # in which case the file isn't copied AND the manifest field is omitted.
    # Downstream pose-alignment skips the robot-IoU term when the mask is
    # absent. Mirrors the file_moves logic in _build_file_moves.
    real_observation = {
        "first_frame": "real/first_frame.png",
        "video":       "real/video.mp4",
    }
    if state.get("robot_mask_path"):
        real_observation["robot_mask"] = "real/robot_mask.png"

    return {
        "schema_version": SCHEMA_VERSION,
        "scene_name": state["scene_name"],
        "robot": {
            "type":       robot_profile.name,
            "gripper":    robot_profile.gripper_name,
            "trajectory": "robot_traj.h5",
        },
        "objects": objects,
        "cameras": {
            "intrinsics": "cameras/intrinsics.txt",
            "extrinsics": "cameras/extrinsics.npz",
            "cam_ids":    [int(x) for x in state["camera_ids"]],
            "primary_id": int(state["object_camera_id"]),
        },
        "real_observation": real_observation,
        "ground": {
            "reference":       state["ground_reference_object"],
            # MuJoCo z-up world floor normal. geometry_prior.z_anchor may
            # later flip `reference` to the GROUND sentinel and set `offset`;
            # scene prep consumes both verbatim (no recomputation).
            "local_down_axis": [0.0, 0.0, -1.0],
        },
        "task": {
            "type":          "manipulation",
            "action":        "pickup",
            "pickup_objects": pickup,
            "description":   description,
        },
    }
