"""Stage 2 orchestrator — turn an episode folder into a CalibratedScene.

Pure deterministic pipeline (no LLM, no Agent loop). Two skills run in
sequence; ground.{reference, local_down_axis, offset} are passed through
from the manifest (Stage 0's ``z_anchor`` already populated them).

    load_episode(episode_dir)
         │
         ▼   InputBundle (includes manifest's ground spec)
    select_primary_camera ─► CameraSelection
         │
         ▼   (true primary)
    align_robot_base      ─► RobotBaseAlignment
         │
         ▼   (robot pose in world)
    [manifest pass-through] ─► GroundCalibration  (local_down_axis + offset)
         │
         ▼
    CalibratedScene  +  <episode_dir>/calibration.yaml

The yaml is a side-effect cache so Stage 3 doesn't have to re-run Stage 2
(which is expensive when ``--optimize-robot-base`` is set).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ar2s.agents_sysid.episode import load_episode
from ar2s.agents_sysid.scene import CalibratedScene
from ar2s.agents_sysid.skills.camera_select import select_primary_camera
from ar2s.agents_sysid.skills.ground_calibrate import GroundCalibration, _load_fp_first_frame
from ar2s.agents_sysid.skills.robot_base_align import align_robot_base


def calibrate_scene(
    episode_dir: str | Path,
    *,
    write_yaml: bool = True,
    update_manifest: bool = True,
    optimize_robot_base: bool = False,
) -> CalibratedScene:
    """Run the Stage-2 pipeline on an episode folder.

    Args:
        episode_dir: path to a built episode folder (with manifest.yaml).
        write_yaml: if True, write `episode_dir/calibration.yaml` as a cache.
        update_manifest: if True, also patch `manifest.yaml` in-place so its
            `cameras.primary_id` and `ground.local_down_axis` match what
            Stage 2 derived. Keeps the two files in sync. Stage 3 still
            reads `calibration.yaml` (full record), but a consistent
            manifest is useful for re-running Stage 1+2 idempotently.
        optimize_robot_base: forwarded to ``align_robot_base``. Default False
            → identity base pose (DROID convention). Set True to run the v1
            IoU optimizer (requires robot_mask.png; falls back to identity
            with a warning when absent).

    Returns:
        CalibratedScene with all three skill outputs + first-frame FP poses.
    """
    episode_dir = Path(episode_dir).expanduser().resolve()
    bundle = load_episode(episode_dir)

    print(f"[stage2] {bundle.scene_name}")
    print("[stage2] camera_select ...")
    sel = select_primary_camera(bundle)
    n_cams = len(bundle.cameras.camera_ids)
    if n_cams == 1:
        print(f"         single-camera default -> primary_id={sel.primary_id} (trivial)")
    else:
        print(f"         primary_id={sel.primary_id} (manifest had "
              f"{bundle.cameras.object_camera_id}), confidence={sel.confidence:.4f}")

    print(f"[stage2] align_robot_base (optimize={optimize_robot_base}) ...")
    align = align_robot_base(
        bundle, primary_camera_index=sel.primary_index,
        optimize=optimize_robot_base,
    )
    if align.n_seeds_tried == 0:
        print(f"         identity base pose (skipped IoU optimizer)")
    else:
        print(f"         IoU={align.iou:.4f}, converged={align.converged}, "
              f"seed_ious={[f'{x:.3f}' for x in align.seed_ious]}")

    # Ground.{reference,local_down_axis,offset} are produced upstream by
    # geometry_prior's z_anchor and live in the manifest verbatim. Pass
    # them through here; no freefall refinement, no FP-rotation axis pick.
    print("[stage2] ground (manifest pass-through) ...")
    ground = GroundCalibration(
        reference_object=bundle.ground.reference_object,
        local_down_axis=tuple(bundle.ground.local_down_axis),
        offset=float(bundle.ground.offset),
        coarse_offset=float(bundle.ground.offset),
        final_drift=0.0,
        converged=True,
        history=[],
    )
    print(f"         reference={ground.reference_object!r}, "
          f"local_down_axis={ground.local_down_axis}, "
          f"offset={ground.offset:.5f} m")

    # Earliest-tracked FP pose per object (cam frame) — read once, cache in
    # CalibratedScene. Uses obj.init_frame so partially-tracked objects
    # (visual per-object keyframe > 0) load their actual first available
    # transform rather than the missing frame_000000.
    fp_first_cam = {
        obj.name: _load_fp_first_frame(obj.pose_source_path, init_frame=obj.init_frame)
        for obj in bundle.objects
    }

    scene = CalibratedScene(
        bundle=bundle,
        camera_selection=sel,
        robot_alignment=align,
        ground=ground,
        object_fp_first_cam=fp_first_cam,
    )

    if write_yaml:
        out = episode_dir / "calibration.yaml"
        save_calibration_yaml(scene, out)
        print(f"[stage2] wrote {out}")

    if update_manifest:
        manifest_path = episode_dir / "manifest.yaml"
        patched = patch_manifest_with_calibration(manifest_path, scene)
        if patched:
            print(f"[stage2] patched {manifest_path}: {', '.join(patched)}")

    return scene


# -------------------------------------------------------------------
# calibration.yaml serialization
# -------------------------------------------------------------------

def _mat_to_list(M: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in M]


def save_calibration_yaml(scene: CalibratedScene, path: str | Path) -> None:
    """Write the human-readable calibration record. Derived; can be regenerated
    by re-running `calibrate_scene`."""
    sel = scene.camera_selection
    align = scene.robot_alignment
    g = scene.ground
    bundle = scene.bundle

    doc = {
        "schema_version": 1,
        "scene_name": scene.scene_name,
        "cameras": {
            "primary_id": int(sel.primary_id),
            "primary_index": int(sel.primary_index),
            "confidence": float(sel.confidence),
            "iou_per_camera": {int(k): float(v) for k, v in sel.iou_per_camera.items()},
            "manifest_primary_id": int(bundle.cameras.object_camera_id),
        },
        "robot": {
            "base_pose": _mat_to_list(align.pose_world),
            "pos": [float(x) for x in align.pos],
            "quat_wxyz": [float(x) for x in align.quat_wxyz],
            "iou": float(align.iou),
            "converged": bool(align.converged),
            "seed_ious": [float(x) for x in align.seed_ious],
        },
        "ground": {
            "reference": g.reference_object,
            "local_down_axis": list(g.local_down_axis),
            "offset": float(g.offset),
            "coarse_offset": float(g.coarse_offset),
            "final_drift_m": float(g.final_drift),
            "converged": bool(g.converged),
        },
    }
    Path(path).write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


# -------------------------------------------------------------------
# manifest patching — keep manifest in sync with calibration
# -------------------------------------------------------------------

def patch_manifest_with_calibration(
    manifest_path: str | Path,
    scene: CalibratedScene,
) -> list[str]:
    """Rewrite ``manifest.yaml`` in-place so its calibration-derived fields
    match what Stage 2 actually produced.

    Patched fields:
      - ``cameras.primary_id``  (the auto-picked primary)
      - ``ground.local_down_axis``  (FP-derived 6-axis pick)

    Returns:
        list of human-readable change strings (empty if nothing changed).
    """
    manifest_path = Path(manifest_path)
    doc = yaml.safe_load(manifest_path.read_text())

    changes: list[str] = []

    new_primary = int(scene.camera_selection.primary_id)
    old_primary = doc.get("cameras", {}).get("primary_id")
    if old_primary != new_primary:
        doc.setdefault("cameras", {})["primary_id"] = new_primary
        changes.append(f"cameras.primary_id {old_primary}→{new_primary}")

    new_axis = list(scene.ground.local_down_axis)
    old_axis = doc.get("ground", {}).get("local_down_axis")
    if list(old_axis or []) != new_axis:
        doc.setdefault("ground", {})["local_down_axis"] = new_axis
        changes.append(f"ground.local_down_axis {old_axis}→{new_axis}")

    if changes:
        manifest_path.write_text(
            yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
        )
    return changes


__all__ = [
    "calibrate_scene",
    "save_calibration_yaml",
    "patch_manifest_with_calibration",
    "CalibratedScene",
]
