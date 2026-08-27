"""Stage 2 skill: derive `ground.local_down_axis` + `ground.offset`.

Two-step calibration:

  Step 1 (deterministic, fast):
    Pick `local_down_axis` from the ref object's first-frame FoundationPose
    rotation. The local axis whose world-frame expression is closest to
    [0, 0, -1] (world down) wins. Handles the "mug.obj uses -Y as up" type
    of mesh-author conventions automatically.

  Step 2 (physics, slower):
    Run v1's `compute_theoretical_ground_offset` (geometric init) →
    `coarse_align_ground` (Strategy A) → `fine_align_ground` (Nelder-Mead).
    Goal: object freefall drift ≈ 0 — i.e., objects rest stably on the
    derived ground plane.

This skill is a thin wrapper around the v1 library functions in
`ar2s/droid_sim/ground_alignment.py`. No re-implementation of the math.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ar2s.agents_sysid.inputs import InputBundle
from ar2s.droid_sim.ground_alignment import (
    coarse_align_ground,
    compute_theoretical_ground_offset,
    fine_align_ground,
)
from ar2s.droid_sim.scene.config import SceneConfig
from ar2s.droid_sim.scene.robot_scene_sim import RobotSceneSim

from ar2s.agents_sysid.scene_config import build_scene_config
from ar2s.droid_sim.scene.robot_profile import get_robot_profile
from ar2s.agents_sysid.skills.camera_select import CameraSelection
from ar2s.agents_sysid.skills.robot_base_align import RobotBaseAlignment


_WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass
class GroundCalibration:
    reference_object: str
    local_down_axis: tuple[float, float, float]    # in ref object's local frame
    offset: float                                   # signed, along local_down_axis
    coarse_offset: float
    final_drift: float                              # m, |p_N - p_0| of ref obj
    converged: bool
    history: list[dict] = field(default_factory=list)


# -------------------------------------------------------------------
# Step 1: pick local_down_axis from FP rotation
# -------------------------------------------------------------------

def _load_extrinsics(extrinsics_path: str, cam_id: int) -> np.ndarray:
    """Return T_cam_from_world (4x4) for the given camera id."""
    ext = np.load(extrinsics_path)
    key = f"cam_mat_{cam_id}"
    if key not in ext:
        raise KeyError(f"Camera extrinsics key {key!r} missing in {extrinsics_path}")
    return np.asarray(ext[key], dtype=np.float64).reshape(4, 4)


def _load_fp_first_frame(fp_dir: str, init_frame: int = 0) -> np.ndarray:
    """Load the FoundationPose pose at the object's earliest tracked frame.

    Args:
        fp_dir: per-object foundation_pose root, in either the packed
            ``poses.npz`` or the legacy ``frame_NNNNNN_transform.npy`` layout.
        init_frame: which frame index to load. Default 0 (legacy behaviour
            for fully-tracked objects). For objects whose visual keyframe
            fell mid-demo (``manifest.objects[i].init_frame > 0``), pass that
            value so we read the actual earliest available pose rather than
            the missing frame 0.

    Returns:
        ``T_cam_from_obj`` (4×4, float64).
    """
    from ar2s.droid_sim.pose_store import load_pose
    T = load_pose(fp_dir, int(init_frame))
    if T is None:
        raise FileNotFoundError(
            f"Missing FP frame_{int(init_frame):06d} (init_frame={init_frame}) "
            f"under {fp_dir}"
        )
    return T


def _pick_local_down_axis(R_world_from_obj: np.ndarray) -> tuple[float, float, float]:
    """Among ±e_x, ±e_y, ±e_z, find the local axis whose world-frame
    expression is closest to [0, 0, -1].

    World expression of ±e_i = ±R_world_from_obj[:, i]. We want this closest
    to [0, 0, -1], i.e. its z-component closest to -1.
    """
    best: tuple[float, int, int] | None = None  # (dist, axis_idx, sign)
    for axis_idx in range(3):
        col = R_world_from_obj[:, axis_idx]
        for sign in (+1, -1):
            world_axis = sign * col
            dist = float(np.linalg.norm(world_axis - _WORLD_DOWN))
            if best is None or dist < best[0]:
                best = (dist, axis_idx, sign)
    assert best is not None
    _, axis_idx, sign = best
    out = [0.0, 0.0, 0.0]
    out[axis_idx] = float(sign)
    return tuple(out)


def _derive_local_down_axis(
    bundle: InputBundle,
    primary_cam_id: int,
) -> tuple[float, float, float]:
    """Step 1: derive local_down_axis from the ref object's FP first frame.

    Robust to a stale ``ground.reference`` that no longer matches any
    surviving object (e.g. visual wrote ``induction stove`` as the ref
    but that object was DEGRADED → didn't make it through resolve.py's
    mesh × scale × pose intersection → not in ``bundle.objects``). When
    that happens, fall back to the first available object with a warning
    rather than raising a hard StopIteration.
    """
    ref_name = bundle.ground.reference_object
    obj_names = [o.name for o in bundle.objects]
    ref_obj = next((o for o in bundle.objects if o.name == ref_name), None)
    if ref_obj is None:
        if not bundle.objects:
            raise ValueError(
                "ground_calibrate: no objects available — visual emitted "
                "zero usable objects (segment / mesh / pose all failed)"
            )
        fallback = bundle.objects[0]
        print(f"[ground_calibrate] WARNING: ground.reference {ref_name!r} not "
              f"in surviving objects {obj_names}; falling back to "
              f"{fallback.name!r}")
        ref_obj = fallback

    T_cam_from_obj = _load_fp_first_frame(
        ref_obj.pose_source_path, init_frame=ref_obj.init_frame,
    )
    T_cam_from_world = _load_extrinsics(bundle.cameras.extrinsics_path, primary_cam_id)
    T_world_from_cam = np.linalg.inv(T_cam_from_world)
    T_world_from_obj = T_world_from_cam @ T_cam_from_obj
    R_world_from_obj = T_world_from_obj[:3, :3]
    return _pick_local_down_axis(R_world_from_obj)


# -------------------------------------------------------------------
# Step 2: physics-based ground.offset (theoretical → coarse → fine)
# -------------------------------------------------------------------

def _build_cfg_for_ground_align(
    bundle: InputBundle,
    primary_cam_id: int,
    local_down_axis: tuple[float, float, float],
) -> SceneConfig:
    """Build a SceneConfig with the corrected primary cam + derived down axis.

    Uses dataclasses.replace so we don't mutate the bundle's CameraSpec / the
    GroundSpec that build_scene_config returned (which would leak into other
    callers).
    """
    cfg = build_scene_config(bundle)
    cfg.cameras = replace(cfg.cameras, object_camera_id=primary_cam_id)
    cfg.ground = replace(cfg.ground, local_down_axis=local_down_axis, offset=0.0)
    return cfg


def calibrate_ground(
    bundle: InputBundle,
    camera_selection: CameraSelection,
    robot_alignment: RobotBaseAlignment,
    *,
    drift_thresh_m: float = 0.01,
) -> GroundCalibration:
    """Run the 2-step ground calibration.

    Args:
        bundle: episode InputBundle.
        camera_selection: from `select_primary_camera` — provides the
            corrected primary_id used to transform FP poses to world frame.
        robot_alignment: from `align_robot_base` — provides robot base pose
            so the freefall sim has the robot in roughly the right place.
        drift_thresh_m: convergence cutoff for `converged` flag.

    Returns:
        GroundCalibration with the derived local_down_axis + tuned offset.
        `converged` is False on physics non-convergence (no exception —
        downstream decides what to do).
    """
    ref_name = bundle.ground.reference_object

    # Step 1: derive local_down_axis from FP rotation
    local_down = _derive_local_down_axis(bundle, camera_selection.primary_id)

    # Step 2: build sim + theoretical_offset + coarse + fine
    cfg = _build_cfg_for_ground_align(bundle, camera_selection.primary_id, local_down)

    initial = compute_theoretical_ground_offset(cfg)
    cfg.ground.offset = initial

    sim = RobotSceneSim(
        cfg,
        is_kinematic=False,
        robot_base_pos=robot_alignment.pos,
        robot_base_quat=robot_alignment.quat_wxyz,
        robot_profile=get_robot_profile(bundle.robot_type),
    )

    history: list[dict] = []

    coarse = coarse_align_ground(sim, initial_offset=initial)
    history.append({"stage": "coarse", **coarse.__dict__})

    fine = fine_align_ground(sim, initial_offset=coarse.offset)
    history.append({"stage": "fine", **fine.__dict__})

    final_drift = float(fine.final_drift)
    return GroundCalibration(
        reference_object=ref_name,
        local_down_axis=local_down,
        offset=float(fine.offset),
        coarse_offset=float(coarse.offset),
        final_drift=final_drift,
        converged=bool(final_drift < drift_thresh_m),
        history=history,
    )


__all__ = ["calibrate_ground", "GroundCalibration"]
