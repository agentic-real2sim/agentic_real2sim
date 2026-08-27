"""Input bundle for the G1 SysID pipeline.

Produced by the Stage-1 motion reader agent and consumed by Stages 2 and 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MotionBundle:
    """Validated inputs for one sysID run.

    Attributes
    ----------
    motion_path:   absolute path to reference .npz motion file
    pkl_path:      absolute path to paired latent context .pkl file
    onnx_model_path: absolute path to the ONNX policy model
    task_type:     "goal" — fixed z for the whole rollout (from one pkl row)
                   "tracking" — per-step z from a row-slice of the pkl
    goal_frame:    (goal only) frame index in .npz whose pose we want to reach
                   for the GoalReachReward.  Defaults to last frame.
    z_idx:         (goal only, raw-array pkl) explicit row index into the
                   (N, z_dim) pkl array to use as the fixed latent z.
                   Overrides the goal_frame-based index when set.
                   Use this when the pkl is a big sequence file (e.g. zs_3.pkl)
                   and you want a specific single z vector from it.
    z_row_start:   (tracking only) first row in the (N, z_dim) pkl array
    z_row_end:     (tracking only) exclusive end row (length = end - start)
    n_frames:      total frames in reference .npz
    z_dim:         latent space dimension
    n_z_rows:      total rows in pkl array
    description:   one-sentence natural-language description of the motion
    min_height:    pelvis height (m) below which the robot is considered fallen
    """
    motion_path: str
    pkl_path: str
    onnx_model_path: str
    task_type: str               # "goal" | "tracking"
    goal_frame: Optional[int]    # for task_type="goal": pose frame in .npz
    z_row_start: Optional[int]   # for task_type="tracking"
    z_row_end: Optional[int]     # for task_type="tracking" (exclusive)
    n_frames: int
    z_dim: int
    n_z_rows: int
    description: str
    min_height: float = 0.3
    z_idx: Optional[int] = None  # goal only: explicit pkl row index for z


__all__ = ["MotionBundle"]
