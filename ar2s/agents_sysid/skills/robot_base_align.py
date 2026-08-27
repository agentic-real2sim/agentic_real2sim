"""Stage 2 skill: resolve the robot base SE(3).

Default behaviour (``optimize=False``)
--------------------------------------
Return the identity base pose immediately. No mask read, no render, no
optimizer. This is the right answer for DROID episodes — the refined
cam_extrinsics produced by visual's raw_data prep is already well
calibrated against the robot's recorded joint trajectory, so the
identity base is within a couple of centimetres of truth. The IoU
optimizer below was developed for the v1 hand-built marker_in_mug
episodes (where cam_extrinsics was rougher); on DROID data it
sometimes introduces drift instead of correcting it.

Opt-in behaviour (``optimize=True``)
------------------------------------
Run the v1 IoU optimizer: 3 fixed seeds × Nelder-Mead on
``1 − IoU(rendered_robot_mask, real_mask)`` against a single camera
(the primary, as picked by ``camera_select``). 6-DoF variable
``(pos[3], rotvec[3])``; no gripper / no joint zero offset. If
``robot_mask_path`` is empty (visual SAM3 failed to ground the
robot), we warn and fall back to identity — opt-in flag with missing
input is not a hard error since the default already handles this case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ar2s.agents_sysid.inputs import InputBundle
from ar2s.droid_sim.pose_alignment import optimize_robot_base_pose
from ar2s.droid_sim.scene.robot_only_sim import RobotOnlySim


# ---- v1 carryover: 3 hardcoded seeds + 100 iters per seed ----------------
@dataclass
class _Seed:
    init_pos: tuple[float, float, float]
    init_rotvec: tuple[float, float, float]


_SEEDS: tuple[_Seed, ...] = (
    _Seed(init_pos=(0.01, 0.02, 0.0),   init_rotvec=(0.0, 0.0, 0.0)),
    _Seed(init_pos=(0.03, 0.00, 0.01),  init_rotvec=(0.05, 0.0, 0.0)),
    _Seed(init_pos=(-0.02, 0.04, -0.01),init_rotvec=(0.0, 0.05, 0.05)),
)

_DEFAULT_MAXITER = 100
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_IOU_THRESHOLD = 0.7


@dataclass
class RobotBaseAlignment:
    pose_world: np.ndarray            # 4x4 in world frame
    pos: np.ndarray                   # (3,)
    quat_wxyz: np.ndarray             # (4,) [w, x, y, z]
    rotvec: np.ndarray                # (3,)
    iou: float
    converged: bool                   # iou >= iou_threshold
    n_seeds_tried: int
    seed_ious: list[float] = field(default_factory=list)
    overlay_image: np.ndarray | None = None   # (H, W, 3) — for debugging


def _load_real_mask(path: str) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img > 127


def _pose_4x4(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from pos + quaternion (w, x, y, z)."""
    from scipy.spatial.transform import Rotation
    w, x, y, z = quat_wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _identity_alignment() -> RobotBaseAlignment:
    """Identity base pose, used as the DROID default + opt-in-with-no-mask fallback."""
    return RobotBaseAlignment(
        pose_world=np.eye(4),
        pos=np.zeros(3, dtype=np.float64),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        rotvec=np.zeros(3, dtype=np.float64),
        iou=0.0,
        converged=True,            # identity is accepted as "the answer" by convention
        n_seeds_tried=0,
        seed_ious=[],
        overlay_image=None,
    )


def align_robot_base(
    bundle: InputBundle,
    primary_camera_index: int,
    *,
    optimize: bool = False,
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
    maxiter_per_seed: int = _DEFAULT_MAXITER,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> RobotBaseAlignment:
    """Resolve the robot base pose for Stage 2.

    Args:
        bundle: episode InputBundle.
        primary_camera_index: which camera's IoU to optimize against (from
            CameraSelection.primary_index). Ignored when ``optimize=False``.
        optimize: if False (default), return the identity base pose immediately
            — see module docstring for the DROID-default rationale. If True,
            run the v1 multi-seed Nelder-Mead IoU optimizer (requires
            ``robot_mask_path``; warns + falls back to identity when absent).
        iou_threshold: success cutoff for the ``converged`` flag (optimize=True).
        maxiter_per_seed: scipy maxiter passed to each Nelder-Mead run.
        width / height: render resolution.

    Returns:
        RobotBaseAlignment with the chosen base pose. ``n_seeds_tried==0``
        flags the identity-default path (vs ``n_seeds_tried==len(_SEEDS)``
        for the optimized path).
    """
    if not optimize:
        return _identity_alignment()

    if not bundle.robot_mask_path:
        print("[robot_base_align] WARNING: optimize=True requested but episode "
              "has no robot_mask.png; falling back to identity base pose")
        return _identity_alignment()

    real_mask = _load_real_mask(bundle.robot_mask_path)

    # Fresh sim per seed (optimizer mutates base pose).
    best = None
    best_iou = -1.0
    seed_ious: list[float] = []

    for seed in _SEEDS:
        sim = RobotOnlySim(
            robot_traj_path=bundle.robot_traj_path,
            cameras=bundle.cameras,
        )
        out = optimize_robot_base_pose(
            sim, real_mask,
            init_pos=np.array(seed.init_pos, dtype=np.float64),
            init_rotvec=np.array(seed.init_rotvec, dtype=np.float64),
            cam_idx=primary_camera_index,
            maxiter=maxiter_per_seed,
            width=width,
            height=height,
        )
        seed_ious.append(float(out.iou))
        if out.iou > best_iou:
            best_iou = float(out.iou)
            best = out

    assert best is not None  # _SEEDS is non-empty
    return RobotBaseAlignment(
        pose_world=_pose_4x4(best.pos, best.quat_wxyz),
        pos=np.asarray(best.pos, dtype=np.float64),
        quat_wxyz=np.asarray(best.quat_wxyz, dtype=np.float64),
        rotvec=np.asarray(best.rotvec, dtype=np.float64),
        iou=float(best.iou),
        converged=bool(best.iou >= iou_threshold),
        n_seeds_tried=len(_SEEDS),
        seed_ious=seed_ious,
        overlay_image=best.overlay,
    )


__all__ = ["align_robot_base", "RobotBaseAlignment"]
