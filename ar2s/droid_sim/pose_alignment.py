"""Robot base pose alignment against a real segmentation mask.

Pure-Python optimizer. Given a RobotOnlySim, a real binary mask, and an
initial seed, runs Nelder-Mead on [pos, rotvec] (6-d) to minimize 1 - IoU.
Does not touch scene objects, ground, or disk state beyond what the caller
requests via the returned `overlay` array.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from ar2s.droid_sim.scene.robot_only_sim import RobotOnlySim


def _quat_wxyz_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return intersection / union if union else 0.0


def _render_robot_mask(
    sim: RobotOnlySim,
    cam_idx: int,
    width: int,
    height: int,
) -> np.ndarray:
    frames = sim.render_cameras(
        width=width, height=height,
        visible_groups=(RobotOnlySim.GEOM_GROUP_ROBOT_VISUAL,),
    )
    distorted = sim.apply_distortion(frames)
    img = distorted[cam_idx]
    return np.any(img > 10, axis=2)


def _overlay(real_mask: np.ndarray, sim_mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*real_mask.shape, 3), dtype=np.uint8)
    overlay[real_mask, 1] = 128
    overlay[sim_mask, 2] = 128
    overlay[sim_mask & real_mask] = [0, 255, 0]
    return overlay


@dataclass
class PoseAlignOutput:
    pos: np.ndarray           # (3,)
    rotvec: np.ndarray        # (3,)
    quat_wxyz: np.ndarray     # (4,)
    iou: float
    overlay: np.ndarray       # (H, W, 3) uint8 — green=overlap, dark-green=real, blue=sim


def optimize_robot_base_pose(
    sim: RobotOnlySim,
    real_mask: np.ndarray,
    *,
    init_pos: np.ndarray,
    init_rotvec: np.ndarray,
    cam_idx: int = 1,
    maxiter: int = 400,
    width: int = 1280,
    height: int = 720,
) -> PoseAlignOutput:
    """Minimize `1 - IoU(real_mask, sim_robot_mask)` over robot base pos+rotvec."""
    real_mask = real_mask.astype(bool)
    init_pos = np.asarray(init_pos, dtype=np.float64)
    init_rotvec = np.asarray(init_rotvec, dtype=np.float64)

    tracker = {"best_iou": -1.0, "best_params": None}

    def objective(params: np.ndarray) -> float:
        pos = params[:3]
        rotvec = params[3:6]
        quat = _quat_wxyz_from_rotvec(rotvec)
        sim.set_base_pose(pos, quat)
        sim_mask = _render_robot_mask(sim, cam_idx, width, height)
        iou = _compute_iou(sim_mask, real_mask)
        if iou > tracker["best_iou"]:
            tracker["best_iou"] = iou
            tracker["best_params"] = params.copy()
        return 1.0 - iou

    x0 = np.concatenate([init_pos, init_rotvec])
    minimize(
        objective, x0, method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4, "adaptive": True},
    )

    best = tracker["best_params"]
    best_pos = best[:3]
    best_rotvec = best[3:6]
    best_quat = _quat_wxyz_from_rotvec(best_rotvec)

    # Render the best overlay for critic / debugging.
    sim.set_base_pose(best_pos, best_quat)
    sim_mask = _render_robot_mask(sim, cam_idx, width, height)

    return PoseAlignOutput(
        pos=best_pos,
        rotvec=best_rotvec,
        quat_wxyz=best_quat,
        iou=float(tracker["best_iou"]),
        overlay=_overlay(real_mask, sim_mask),
    )
