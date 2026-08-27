"""Load a declarative episode folder into an InputBundle.

The v2 episode format is a folder with:
  - manifest.yaml  (schema v1, see sysid_inputs/manifest.yaml)
  - real/, objects/, trajectory/  (data referenced by manifest)

This loader bridges the declarative manifest format to the runtime input
types used by the sysid calibration and grasp-probe stages. The strict
validator catches typos and missing fields up front.

Usage
-----
    from ar2s.agents_sysid.episode import load_episode

    bundle = load_episode(
        "outputs/run_pipeline/marker_in_mug_v0/sysid_inputs/marker_in_mug_v0"
    )
    # bundle is an ar2s.agents_sysid.inputs.InputBundle.
"""
from pathlib import Path
from typing import Any, Mapping

import yaml

from ar2s.agents_sysid.inputs import GroundInput, InputBundle, ObjectInput
from ar2s.droid_sim.scene.config import CameraSpec


SCHEMA_VERSION = 1


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def load_episode(episode_dir: str | Path) -> InputBundle:
    """Read ``episode_dir/manifest.yaml`` + folder, return ``InputBundle``.

    Strict validation: any missing required field, schema_version mismatch,
    or non-existent referenced file raises with a path-prefixed message.
    """
    episode_dir = Path(episode_dir).resolve()
    manifest_path = episode_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.yaml missing in {episode_dir}")

    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.yaml in {episode_dir} did not parse as dict")

    _validate(manifest, episode_dir)
    return _to_input_bundle(manifest, episode_dir)


# -------------------------------------------------------------------
# Validation (strict)
# -------------------------------------------------------------------

def _real_frame_size(episode_dir: Path, manifest: dict) -> tuple[int, int] | None:
    """Native (W, H) of the real cameras — the size the stored K describes.

    Read from ``real_observation.video``, the rectified left stream (a hardlink
    to ``seq/<serial>/rectified_left.mp4``), which is always at sensor
    resolution. Deliberately NOT ``real_observation.first_frame``: that is a
    FoundationPose ``track_vis`` overlay, so it inherits
    ``VisualInput.pose_shorter_side`` and can be smaller than the sensor.

    K cannot supply the size either — cx is not W/2 on a real camera. Returns
    None when the video is missing or unreadable; callers then fall back to the
    DROID default rather than guessing.
    """
    rel = (manifest.get("real_observation") or {}).get("video")
    if not rel:
        return None
    path = Path(episode_dir) / rel
    if not path.is_file():
        return None
    try:
        import imageio.v2 as imageio
        with imageio.get_reader(str(path)) as reader:
            w, h = reader.get_meta_data()["size"]
    except Exception as e:  # noqa: BLE001
        print(f"[episode] WARNING: cannot read {path} for camera size ({e})")
        return None
    return int(w), int(h)


def _validate(manifest: Mapping[str, Any], episode_dir: Path) -> None:
    sv = manifest.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION}. "
            f"Move the old manifest aside and re-run scanner."
        )

    # Top-level required keys
    for k in ("scene_name", "robot", "objects", "cameras",
              "real_observation", "ground", "task"):
        if k not in manifest:
            raise ValueError(f"manifest missing top-level field: {k!r}")

    # robot block
    for k in ("type", "gripper", "trajectory"):
        if k not in manifest["robot"]:
            raise ValueError(f"manifest.robot missing: {k!r}")
    _check_path(episode_dir, manifest["robot"]["trajectory"], "robot.trajectory")

    # objects block
    objs = manifest["objects"]
    if not isinstance(objs, list) or not objs:
        raise ValueError("manifest.objects must be a non-empty list")
    obj_names: set[str] = set()
    for i, obj in enumerate(objs):
        for k in ("name", "visual", "scale", "foundation_pose_dir", "friction"):
            if k not in obj:
                raise ValueError(f"manifest.objects[{i}] missing: {k!r}")
        # mass is required, but the diagnostic is more useful than a generic
        # "missing: 'mass'" because the contract is: visual emits geometry
        # + pose only; physical_prior writes the mass (or user provides
        # ``entry.object_mass_hints[name]`` to skip the VLM call).
        mass = obj.get("mass")
        if mass is None:
            raise ValueError(
                f"manifest.objects[{i}] ({obj.get('name')!r}) is missing "
                "``mass``. The visual stage no longer writes a placeholder "
                "for mass — it must be supplied by either:\n"
                "  1. the physical-prior stage inside ``python -m "
                "ar2s.run_pipeline`` (material x density x scaled volume), OR\n"
                "  2. ``entry.object_mass_hints[<name>]`` in your VisualInput "
                "(skips the VLM for that object)."
            )
        if obj["name"] in obj_names:
            raise ValueError(f"duplicate object name: {obj['name']!r}")
        obj_names.add(obj["name"])
        _check_path(episode_dir, obj["visual"], f"objects[{i}].visual")
        _check_path(episode_dir, obj["scale"],  f"objects[{i}].scale")
        _check_pose_source(episode_dir, obj["foundation_pose_dir"],
                           f"objects[{i}].foundation_pose_dir")
        if obj.get("collision_dir"):
            _check_path(episode_dir, obj["collision_dir"],
                        f"objects[{i}].collision_dir")

    # cameras block — accepts combined OR per-camera form
    cams = manifest["cameras"]
    if "list" in cams:
        # per-camera form (vision-team-friendly,  not yet implemented)
        raise NotImplementedError(
            "cameras.list[] (per-camera-file form) not yet supported by "
            "loader; use combined form (cameras.intrinsics + cameras.extrinsics "
            "+ cameras.cam_ids) for now."
        )
    for k in ("intrinsics", "extrinsics", "cam_ids", "primary_id"):
        if k not in cams:
            raise ValueError(f"manifest.cameras (combined form) missing: {k!r}")
    _check_path(episode_dir, cams["intrinsics"], "cameras.intrinsics")
    _check_path(episode_dir, cams["extrinsics"], "cameras.extrinsics")
    if not isinstance(cams["cam_ids"], list) or not cams["cam_ids"]:
        raise ValueError("manifest.cameras.cam_ids must be non-empty list")
    if cams["primary_id"] not in cams["cam_ids"]:
        raise ValueError(
            f"cameras.primary_id {cams['primary_id']} not in cam_ids "
            f"{cams['cam_ids']}"
        )

    # real_observation block — robot_mask is optional (SAM3 may fail to
    # ground the robot; downstream pose-alignment skips the IoU term).
    real = manifest["real_observation"]
    for k in ("first_frame", "video"):
        if k not in real:
            raise ValueError(f"manifest.real_observation missing: {k!r}")
        _check_path(episode_dir, real[k], f"real_observation.{k}")
    if "robot_mask" in real:
        _check_path(episode_dir, real["robot_mask"], "real_observation.robot_mask")

    # ground block (semantic)
    for k in ("reference", "local_down_axis"):
        if k not in manifest["ground"]:
            raise ValueError(f"manifest.ground missing: {k!r}")
    ground_ref = manifest["ground"]["reference"]
    if ground_ref != "GROUND" and ground_ref not in obj_names:
        # Two regimes:
        #
        # 1. ``"GROUND"`` sentinel: ``ar2s.agents_geometry_prior`` Stage 0's
        #    z_anchor writes this to mean "world XY plane". Downstream
        #    consumers (``derive_ground_pose_world_xy`` in robot_scene_sim,
        #    ``_ground_normal_at_frame`` in grasp_sweep) detect it and skip
        #    the FP-derived ground path. Always valid — don't validate
        #    against obj_names.
        #
        # 2. Real object name (legacy / pre-Stage-0 bundles): the visual
        #    pipeline's ``ground_ref`` subagent occasionally wrote a stale
        #    reference (an object that segment ACCEPTED but resolve.py later
        #    dropped at the mesh × scale × pose intersection — e.g. DROID
        #    episode IRIS_0303 named "induction stove" as ground ref while
        #    the emitted objects/ only contained the bowl). Silently
        #    rewrite to the first surviving object so old episodes still
        #    load. Stage 0's z_anchor normally rewrites this to the GROUND
        #    sentinel anyway; this branch is only reached for bundles that
        #    skipped geometry_prior.
        stale = ground_ref
        if not obj_names:
            raise ValueError(
                f"ground.reference {stale!r} not in objects (and objects "
                f"list is empty — no fallback available)"
            )
        new_ref = sorted(obj_names)[0]
        print(f"[episode] WARNING: ground.reference {stale!r} not in "
              f"objects {sorted(obj_names)}; falling back to {new_ref!r}")
        manifest["ground"]["reference"] = new_ref

    # task block (semantic)
    for k in ("type", "action", "description"):
        if k not in manifest["task"]:
            raise ValueError(f"manifest.task missing: {k!r}")
    picks = _pickup_objects(manifest["task"])
    if not picks:
        raise ValueError("manifest.task missing: 'pickup_objects'")
    missing = [p for p in picks if p not in obj_names]
    if missing:
        raise ValueError(
            f"task.pickup_objects {missing} not in objects {sorted(obj_names)}"
        )


def _pickup_objects(task: Mapping[str, Any]) -> list[str]:
    """Read ``task.pickup_objects``, falling back to legacy ``pickup_object``.

    The visual pipeline now emits a list; episode folders emitted before that
    change carry a single ``pickup_object`` string.
    """
    if "pickup_objects" in task:
        return list(task["pickup_objects"])
    if "pickup_object" in task:
        return [task["pickup_object"]]
    return []


def _check_path(base: Path, rel: str, field: str) -> None:
    """Resolve `rel` relative to `base` and assert it exists."""
    p = base / rel
    if not p.exists():
        raise FileNotFoundError(
            f"{field}: file/dir not found at {p} "
            f"(manifest path = {rel!r}, episode = {base})"
        )


def _check_pose_source(base: Path, rel: str, field: str) -> None:
    """Assert the per-object pose source resolves in either on-disk layout.

    ``foundation_pose_dir`` may be a legacy directory of per-frame
    ``frame_NNNNNN_transform.npy`` files or a packed ``poses.npz`` — see
    ``ar2s.droid_sim.pose_store``. Existence alone is not enough: an emptied
    directory would pass ``_check_path`` and only fail later inside the sim.
    """
    from ar2s.droid_sim.pose_store import exists as pose_source_exists
    if not pose_source_exists(base / rel):
        raise FileNotFoundError(
            f"{field}: no FoundationPose poses at {base / rel} — expected "
            f"either poses.npz or frame_NNNNNN_transform.npy files "
            f"(manifest path = {rel!r}, episode = {base})"
        )


# -------------------------------------------------------------------
# Conversion to InputBundle
# -------------------------------------------------------------------

def _to_input_bundle(manifest: Mapping[str, Any], episode_dir: Path) -> InputBundle:
    objects = [
        ObjectInput(
            name=o["name"],
            visual_mesh_path=str(episode_dir / o["visual"]),
            scale_path=str(episode_dir / o["scale"]),
            pose_source_path=str(episode_dir / o["foundation_pose_dir"]),
            collision_mesh_dir=(
                str(episode_dir / o["collision_dir"]) if o.get("collision_dir")
                else ""
            ),
            mass=float(o["mass"]),
            friction=tuple(float(x) for x in o["friction"]),
            init_frame=int(o.get("init_frame", 0)),
            static=bool(o.get("static", False)),
            pos_offset=tuple(float(x) for x in o.get("pos_offset", (0.0, 0.0, 0.0))),
        )
        for o in manifest["objects"]
    ]

    cams = manifest["cameras"]
    cameras_spec = CameraSpec(
        camera_ids=tuple(int(x) for x in cams["cam_ids"]),
        object_camera_id=int(cams["primary_id"]),
        intrinsics_path=str(episode_dir / cams["intrinsics"]),
        extrinsics_path=str(episode_dir / cams["extrinsics"]),
        image_size=_real_frame_size(episode_dir, manifest),
    )

    ground = GroundInput(
        reference_object=manifest["ground"]["reference"],
        local_down_axis=tuple(float(x) for x in manifest["ground"]["local_down_axis"]),
        offset=float(manifest["ground"].get("offset", 0.0)),
    )

    picks = _pickup_objects(manifest["task"])

    real = manifest["real_observation"]
    robot_mask_path = (
        str(episode_dir / real["robot_mask"]) if real.get("robot_mask") else ""
    )

    return InputBundle(
        scene_name=manifest["scene_name"],
        robot_traj_path=str(episode_dir / manifest["robot"]["trajectory"]),
        objects=objects,
        cameras=cameras_spec,
        ground=ground,
        robot_type=manifest["robot"].get("type", "franka_panda"),
        robot_mask_path=robot_mask_path,
        first_frame_rgb_path=str(episode_dir / real["first_frame"]),
        pickup_objects=picks,
        # Single-object sysid stages (grasp probe / sweep / loop) still act on
        # one object; the first pickup is the one they drive.
        pickup_object=picks[0],
    )


__all__ = ["load_episode", "SCHEMA_VERSION"]
