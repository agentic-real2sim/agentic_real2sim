"""Fold per-stage tool outputs into bundle-shape fields (disk-state).

This is the deterministic glue between visual-agent skill outputs (sliced by
stage, persisted as JSON under state.stages.<name>) and the final
``outputs.write_episode_folder`` step (which needs everything sliced by
object). It also materialises one artefact the toolkit doesn't:

  - ``ObjectInput.scale_path`` expects a (3,) ``.npy``; mesh_scale only
    produces a uniform float per frame. resolve writes
    ``<seq>/<object>_scale.npy`` from their median.

Run AFTER the ReAct agent completes (ground_ref ok=True) and BEFORE emit.
No LLM/VLM in here.

Reads (all dict access on disk-loaded state.json):
  - state.entry                                      (VisualInput as dict)
  - state.pickup_objects                             (list[str] — set by pickup_objects subagent)
  - state.stages.svo_extract.outputs.{K_path, ...}
  - state.cameras_extrinsics_path                    (cam_mat_<serial> key[s], w2c)
  - state.stages.segmentation.outputs.objects[*]
  - state.stages.mesh_recover.outputs.per_object[*]
  - state.stages.mesh_scale.outputs.per_object[*]
  - state.stages.pose_tracking.outputs.per_object[*]

Writes into state.json (via update_state):
  - resolved_objects                  list[dict] (ResolvedObject-shaped)
  - pickup_objects                    pruned to the ones that survived all stages
  - robot_mask_path
  - first_frame_rgb_path
  - cameras_intrinsics_path
  - camera_ids                        list[int]   (PR-4: includes secondary when present)
  - object_camera_id                  int          (primary serial — kept for sim compat;
                                                    per-obj camera_id lives in resolved_objects
                                                    + mask_camera_id_by_object)
"""
from pathlib import Path

import numpy as np

from ar2s.agents_visual.state import ResolvedObject, load_state, save_state


_ROBOT_TEXT = "robot"

_REQUIRED_STAGES = ("svo_extract", "segmentation", "mesh_recover",
                    "mesh_scale", "pose_tracking")

_REQUIRED_TOPLEVEL = ("entry", "pickup_objects")


def _camera_serials_from_extrinsics(extrinsics_path: str) -> list[str]:
    """Return camera serials encoded as cam_mat_<serial> keys in the NPZ."""
    if not extrinsics_path:
        raise RuntimeError("cameras_extrinsics_path is empty; cannot derive camera id")
    path = Path(extrinsics_path)
    if not path.exists():
        raise RuntimeError(f"cameras_extrinsics_path not found: {path}")
    with np.load(path) as data:
        keys = [k for k in data.files if k.startswith("cam_mat_")]
    if not keys:
        raise RuntimeError(
            f"expected at least one cam_mat_<serial> key in {path}; got {keys}"
        )
    return [k[len("cam_mat_"):] for k in keys]


def _numeric_camera_id(serial_str: str, *, source: str) -> int:
    try:
        return int(serial_str)
    except ValueError as e:
        raise RuntimeError(
            f"camera serial from {source} must be numeric; got {serial_str!r}"
        ) from e


def _primary_camera_id_from_state(state: dict) -> tuple[int, list[str]]:
    """Resolve primary camera id from state and dual-view-capable extrinsics.

    Dual-camera NPZs are valid when state.primary_camera_id disambiguates the
    primary ``cam_mat_<serial>``. Old one-camera inputs may omit the state key.
    Matrices are ``T_cam_from_world`` / w2c; pose files remain primary-frame.
    """
    extrinsics_path = state.get("cameras_extrinsics_path", "")
    serials = _camera_serials_from_extrinsics(extrinsics_path)
    primary_serial = state.get("primary_camera_id", "") or ""
    if primary_serial:
        if primary_serial not in serials:
            raise RuntimeError(
                f"state.primary_camera_id={primary_serial!r} is not present in "
                f"{extrinsics_path}; available cam_mat serials: {serials}"
            )
        return _numeric_camera_id(primary_serial, source="state.primary_camera_id"), serials
    if len(serials) != 1:
        raise RuntimeError(
            "state.primary_camera_id is required when cameras_extrinsics_path "
            f"contains multiple cam_mat_<serial> keys: {serials}"
        )
    return _numeric_camera_id(serials[0], source=extrinsics_path), serials


def _zip_object_reports(
    mesh_per_object: list[dict],
    scale_per_object: list[dict],
    pose_per_object: list[dict],
) -> tuple[list[tuple[dict, dict, dict]], list[str]]:
    """Intersect successful reports across mesh_recover / mesh_scale / pose_tracking.

    Returns (triples, dropped). An object survives only when:
      - mesh_recover.per_object[name].success is True
      - mesh_scale.per_object[name].num_frames_used > 0
      - pose_tracking.per_object[name].success is True
    """
    mesh_ok = {r["object_name"]: r for r in mesh_per_object if r.get("success")}
    scale_ok = {
        s["object_name"]: s
        for s in scale_per_object
        if (s.get("num_frames_used") or 0) > 0
    }
    pose_ok = {r["object_name"]: r for r in pose_per_object if r.get("success")}

    names = set(mesh_ok) & set(scale_ok) & set(pose_ok)
    union = set(mesh_ok) | set(scale_ok) | set(pose_ok)
    dropped = sorted(union - names)
    triples = [(mesh_ok[n], scale_ok[n], pose_ok[n]) for n in sorted(names)]
    return triples, dropped


def _write_aggregate_scale_npy(
    median_scale: float, meta_dir: str, object_name: str,
) -> str:
    """Write (s, s, s) float32 to ``<seq>/<object>_scale.npy``.

    ``meta_dir`` is mesh_scale's per-object metadata dir
    (``<seq>/scale/<object>``); the aggregate goes one level up beside the
    sequence so it isn't the lone file in a two-deep tree, and so its name
    says which object it belongs to rather than relying on the parent dir.
    """
    seq_dir = Path(meta_dir).parent.parent
    out_path = seq_dir / f"{object_name}_scale.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.array([median_scale] * 3, dtype=np.float32))
    return str(out_path)


def _find_robot_mask(seg_outputs: dict) -> str:
    """Find robot mask path in segmentation outputs.

    Returns <robot_masks_dir>/000000.png — the first-frame mask. Used by
    downstream pose-alignment to optimise the camera-to-robot-base transform
    (IoU between rendered sim robot and the real robot mask). If SAM3 can't
    find the robot in this scene (no text variant grounds + no Tier-1 synonym
    rescues it), return "" and warn: pose-alignment can still operate from
    other cues, just without the robot-IoU term.
    """
    for obj in seg_outputs.get("objects", []):
        if obj.get("text") == _ROBOT_TEXT and obj.get("masks_dir"):
            return str(Path(obj["masks_dir"]) / "000000.png")
    print(
        f"[resolve] WARNING: no segmentation entry with text={_ROBOT_TEXT!r} "
        f"(or its masks_dir is empty). Downstream pose-alignment will skip "
        f"the robot-mask IoU term."
    )
    return ""


def resolve(run_root: str) -> dict:
    """Aggregate stage outputs into ResolvedObject list + manifest paths.

    Mutates state.json in place. Returns a dict of the fields it set
    (handy for printing / debug from the CLI).

    Raises:
        ValueError on missing prerequisites or if the first pickup object did
        not survive every stage.
    """
    state = load_state(run_root)

    missing_top = [k for k in _REQUIRED_TOPLEVEL if not state.get(k)]
    if missing_top:
        raise ValueError(f"resolve() top-level state missing or empty: {missing_top}")

    stages = state.get("stages", {})
    missing_stages = [
        s for s in _REQUIRED_STAGES
        if not stages.get(s, {}).get("ok")
    ]
    if missing_stages:
        raise ValueError(
            f"resolve() stages not complete: {missing_stages} "
            f"(have: {sorted(k for k, v in stages.items() if v.get('ok'))})"
        )

    svo = stages["svo_extract"]
    seg = stages["segmentation"]
    mr  = stages["mesh_recover"]
    ms  = stages["mesh_scale"]
    pt  = stages["pose_tracking"]

    pickup_objects = list(state["pickup_objects"])
    entry = state.get("entry") or {}
    mass_hints = entry.get("object_mass_hints") or state.get("object_mass_hints") or {}
    friction_hints = entry.get("object_friction_hints") or state.get("object_friction_hints") or {}

    triples, dropped = _zip_object_reports(
        mr["outputs"]["per_object"],
        ms["outputs"]["per_object"],
        pt["outputs"]["per_object"],
    )
    if dropped:
        print(f"[resolve] WARNING: dropping objects with partial-stage failures: {dropped}")

    pose_by_name = {pose_r["object_name"]: pose_r for _, _, pose_r in triples}
    # The FIRST pickup object must survive every stage. pickup_objects is in
    # temporal grasp order, and sysid's sweep can only study the demo's first
    # grasp — it anchors on engage_frame, the first gripper-close edge (see
    # droid_sim/grasp_probe.py::_find_engage_frame) — so falling back to a later
    # pickup would sample the right mesh against the wrong instant. Secondary
    # pickups may drop; they are never swept, just pruned from the emitted list.
    first_pickup = pickup_objects[0]
    if first_pickup not in pose_by_name:
        raise ValueError(
            f"first pickup object {first_pickup!r} did not survive all stages "
            f"(succeeded: {sorted(pose_by_name)}; dropped: {dropped}). The grasp "
            f"sweep anchors on the demo's first gripper-close, so it cannot fall "
            f"back to a later pickup object."
        )
    surviving_pickups = [p for p in pickup_objects if p in pose_by_name]
    if len(surviving_pickups) < len(pickup_objects):
        print(f"[resolve] WARNING: secondary pickup objects dropped by earlier "
              f"stages: {[p for p in pickup_objects if p not in pose_by_name]} "
              f"(keeping {surviving_pickups})")

    resolved: list[ResolvedObject] = []
    for mesh_r, scale_r, pose_r in triples:
        scale_npy = _write_aggregate_scale_npy(
            scale_r["median_scale"], scale_r["meta_dir"], mesh_r["object_name"],
        )
        ro = ResolvedObject(
            name=mesh_r["object_name"],
            visual_mesh_path=mesh_r["glb_path"],
            scale_path=scale_npy,
            pose_source_path=pose_r["pose_dir"],
        )
        # Mass: explicit user hint wins; otherwise leave as None so
        # ``ar2s.agents_physical_prior`` (material × density × scaled volume)
        # is the source. The manifest emits no ``mass`` field at all when
        # both are absent — sysid's validator rejects that loudly so a
        # forgotten physical_prior run can't silently slide into the sim.
        if ro.name in mass_hints:
            ro.mass = float(mass_hints[ro.name])
        if ro.name in friction_hints:
            ro.friction = tuple(float(x) for x in friction_hints[ro.name])
        resolved.append(ro)

    serial, extrinsic_serials = _primary_camera_id_from_state(state)

    # Dual-view (PR-4): camera_ids includes BOTH primary + secondary when a
    # secondary stereo was kept (cam_mat_<secondary_serial> lives in the same
    # cameras_extrinsics.npz). object_camera_id stays as primary's serial —
    # downstream sim still treats poses as primary-frame; secondary poses
    # were rebased to primary frame in pose_tracking (_rebase_secondary_poses).
    secondary_id_str = state.get("secondary_camera_id", "") or ""
    extra_ids: list[int] = []
    if secondary_id_str:
        if secondary_id_str in extrinsic_serials:
            secondary_id = _numeric_camera_id(
                secondary_id_str,
                source="state.secondary_camera_id",
            )
            if secondary_id != serial:
                extra_ids.append(secondary_id)

    update = {
        "resolved_objects":         resolved,
        # Prune to survivors so the emitted manifest only names pickup objects
        # that actually made it into objects/ (sysid's validator rejects the
        # rest).
        "pickup_objects":           surviving_pickups,
        "robot_mask_path":          _find_robot_mask(seg["outputs"]),
        "first_frame_rgb_path":     pose_by_name[first_pickup]["first_frame_rgb_path"],
        "cameras_intrinsics_path":  svo["outputs"]["K_path"],
        "camera_ids":               [serial, *extra_ids],
        "object_camera_id":         serial,
    }
    state.update(update)
    save_state(state, run_root)
    return update
