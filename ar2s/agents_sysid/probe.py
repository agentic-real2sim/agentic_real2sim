"""Stage 3 orchestrator — load calibration + run probe + save result.

Single-skill stage today (`probe_grasp`). Output goes to
``artifacts/<run-id>/probe_result.yaml``.

Future expansions (envelope characterization, tube, USD render) will plug
in here as additional skills. The orchestrator stays a simple linear
chain — when we eventually agent-ify Stage 3, each skill is a clean
``@tool``-shaped pure function (same pattern as Stage 1+2).

Calibration handling:
  - If ``<episode_dir>/calibration.yaml`` exists, load it (fast).
  - Otherwise run Stage 2 fresh (~200s, dominated by robot align).

Output layout:
  artifacts/<run-id>/
      probe_result.yaml
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import yaml

from ar2s.agents_sysid.calibrate import calibrate_scene
from ar2s.agents_sysid.episode import load_episode
from ar2s.agents_sysid.scene import CalibratedScene
from ar2s.agents_sysid.skills.camera_select import CameraSelection
from ar2s.agents_sysid.skills.ground_calibrate import GroundCalibration, _load_fp_first_frame
from ar2s.agents_sysid.skills.probe_grasp import GraspProbeResult, probe_grasp
from ar2s.agents_sysid.skills.robot_base_align import RobotBaseAlignment


# -------------------------------------------------------------------
# load CalibratedScene from cached yaml
# -------------------------------------------------------------------

def _load_calibration_yaml(episode_dir: Path) -> CalibratedScene:
    """Reconstruct a CalibratedScene from a cached calibration.yaml."""
    bundle = load_episode(episode_dir)
    doc = yaml.safe_load((episode_dir / "calibration.yaml").read_text())

    cam = doc["cameras"]
    sel = CameraSelection(
        primary_id=int(cam["primary_id"]),
        primary_index=int(cam["primary_index"]),
        iou_per_camera={int(k): float(v)
                         for k, v in cam.get("iou_per_camera", {}).items()},
        confidence=float(cam.get("confidence", 0.0)),
    )

    rob = doc["robot"]
    align = RobotBaseAlignment(
        pose_world=np.array(rob["base_pose"], dtype=np.float64),
        pos=np.array(rob["pos"], dtype=np.float64),
        quat_wxyz=np.array(rob["quat_wxyz"], dtype=np.float64),
        rotvec=np.zeros(3, dtype=np.float64),   # not stored in yaml; not needed downstream
        iou=float(rob["iou"]),
        converged=bool(rob["converged"]),
        n_seeds_tried=len(rob.get("seed_ious", [])),
        seed_ious=list(rob.get("seed_ious", [])),
    )

    g = doc["ground"]
    ground = GroundCalibration(
        reference_object=g["reference"],
        local_down_axis=tuple(g["local_down_axis"]),
        offset=float(g["offset"]),
        coarse_offset=float(g.get("coarse_offset", g["offset"])),
        final_drift=float(g.get("final_drift_m", 0.0)),
        converged=bool(g.get("converged", True)),
    )

    fp_first = {
        obj.name: _load_fp_first_frame(obj.pose_source_path, init_frame=obj.init_frame)
        for obj in bundle.objects
    }

    return CalibratedScene(
        bundle=bundle,
        camera_selection=sel,
        robot_alignment=align,
        ground=ground,
        object_fp_first_cam=fp_first,
    )


def _load_or_compute_calibration(episode_dir: Path) -> CalibratedScene:
    """Load cached calibration if present, else run Stage 2."""
    if (episode_dir / "calibration.yaml").exists():
        return _load_calibration_yaml(episode_dir)
    print("[stage3] no cached calibration.yaml — running Stage 2 (~200s) ...")
    return calibrate_scene(episode_dir, write_yaml=True, update_manifest=True)


# -------------------------------------------------------------------
# orchestrator
# -------------------------------------------------------------------

def _make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def probe_scene(
    episode_dir: str | Path,
    *,
    artifacts_root: str | Path = "artifacts",
    run_id: str | None = None,
) -> tuple[GraspProbeResult, Path]:
    """Run Stage 3 (basic grasp probe) on a calibrated episode.

    Args:
        episode_dir: built episode folder (with manifest.yaml + ideally
            calibration.yaml).
        artifacts_root: where to dump run artifacts (default "artifacts/").
        run_id: optional explicit id; default = timestamp + uuid6.

    Returns:
        (GraspProbeResult, run_dir).
    """
    episode_dir = Path(episode_dir).expanduser().resolve()
    artifacts_root = Path(artifacts_root).expanduser().resolve()
    run_id = run_id or _make_run_id()
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage3] episode = {episode_dir}")
    print(f"[stage3] run_dir = {run_dir}")
    print("[stage3] load calibration ...")
    scene = _load_or_compute_calibration(episode_dir)
    print(f"         primary_id={scene.camera_selection.primary_id}, "
          f"robot_iou={scene.robot_alignment.iou:.3f}, "
          f"ground_offset={scene.ground.offset:.4f}m")

    pickup = scene.bundle.pickup_object
    print(f"[stage3] probe_grasp (pickup={pickup}) ...")
    t0 = time.time()
    result, _frame_cache = probe_grasp(scene)
    elapsed = time.time() - t0
    verdict = ("GRASPED" if result.grasped
               else "LOOSE" if result.grasped_loose
               else "FAILED")
    print(f"         [{verdict}] {elapsed:.1f}s  "
          f"peak_lift={result.peak_lift_m*1000:.1f}mm  "
          f"dist@apex={result.dist_at_peak_lift_m*100:.1f}cm  "
          f"closest={result.closest_distance_m*100:.1f}cm")
    if result.failure_reason:
        print(f"         reason: {result.failure_reason}")

    out = run_dir / "probe_result.yaml"
    _save_probe_result(out, episode_dir, scene, result)
    print(f"[stage3] wrote {out}")

    return result, run_dir


# -------------------------------------------------------------------
# probe_result.yaml serializer
# -------------------------------------------------------------------

def _save_probe_result(
    path: Path,
    episode_dir: Path,
    scene: CalibratedScene,
    result: GraspProbeResult,
) -> None:
    doc = {
        "schema_version": 1,
        "scene_name": scene.scene_name,
        "episode_dir": str(episode_dir),
        "pickup_object": scene.bundle.pickup_object,
        "grasp": {
            "grasped": result.grasped,
            "grasped_loose": result.grasped_loose,
            "failure_reason": result.failure_reason,
            "peak_lift_m": result.peak_lift_m,
            "peak_lift_time_frac": result.peak_lift_time_frac,
            "lift_window_sec": result.lift_window_sec,
            "dist_at_peak_lift_m": result.dist_at_peak_lift_m,
            "closest_distance_m": result.closest_distance_m,
            "duration_sec": result.duration_sec,
            "engage_frame": result.engage_frame,
        },
        "criteria": result.criteria,
        "diagnostics": {
            "pen_quat_at_engage_wxyz": list(result.pen_quat_at_engage),
            "gripper_quat_at_engage_wxyz": list(result.gripper_quat_at_engage),
        },
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))


__all__ = ["probe_scene"]
