"""CalibratedScene — in-memory result of Stage 2.

The orchestrator (`ar2s.agents_sysid.calibrate.calibrate_scene`) returns this; Stage 3
consumes it. It is also what `calibration.yaml` round-trips through (via
`save_calibration_yaml` / `load_calibration_yaml`).

Fields are the union of the two skill outputs (camera_select,
robot_base_align) plus a `GroundCalibration` built as a pass-through of the
manifest ground spec (populated upstream by ``geometry_prior.z_anchor``),
plus the original `InputBundle`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ar2s.agents_sysid.inputs import InputBundle

from ar2s.agents_sysid.skills.camera_select import CameraSelection
from ar2s.agents_sysid.skills.ground_calibrate import GroundCalibration
from ar2s.agents_sysid.skills.robot_base_align import RobotBaseAlignment


@dataclass
class CalibratedScene:
    bundle: InputBundle
    camera_selection: CameraSelection
    robot_alignment: RobotBaseAlignment
    ground: GroundCalibration

    # Convenience: first-frame FP transforms (T_cam_from_obj) for each obj.
    # Kept as a separate cache so Stage 3 can read it without re-doing file I/O.
    # The conversion to world-frame uses `camera_selection.primary_id`'s
    # extrinsics — same conversion as RobotSceneSim does internally.
    object_fp_first_cam: dict[str, np.ndarray] = None  # type: ignore[assignment]

    @property
    def scene_name(self) -> str:
        return self.bundle.scene_name


__all__ = ["CalibratedScene"]
