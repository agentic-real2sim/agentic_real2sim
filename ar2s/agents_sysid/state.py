"""Shared lightweight result types used by sysid tools and ar2s.droid_sim."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PoseResult:
    pos: tuple[float, float, float]
    rotvec: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    iou: float
    overlay_path: str = ""
    attempt_idx: int = 0


@dataclass
class FrameSnapshot:
    frame_idx: int = 0
    time_sec: float = 0.0
    frac: float = 0.0
    pen_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pen_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gripper_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gripper_close_frac: float = 0.0
    distance: float = 0.0
    image_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class FirstContactInfo:
    frame_idx: int = -1
    side: str = "none"
    body_name: str = ""
    world_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_frame_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    corrective_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class GraspProbeReport:
    grasped: bool = False
    peak_lift: float = 0.0
    peak_lift_time_frac: float = 0.0
    lift_window_sec: float = 0.0
    closest_distance_overall: float = 0.0
    dist_at_peak_lift: float = 0.0
    grasped_loose: bool = False
    snapshots: list[FrameSnapshot] = field(default_factory=list)
    duration_sec: float = 0.0
    first_contact: Optional[FirstContactInfo] = None


__all__ = [
    "FirstContactInfo",
    "FrameSnapshot",
    "GraspProbeReport",
    "PoseResult",
]
