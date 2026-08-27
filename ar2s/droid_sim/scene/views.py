"""CameraPose dataclass + sanity bounds + extrinsics → pose conversion.

Used by ViewFinderAgent to represent the two camera poses chosen for
capturing grasp keyframes. A `CameraPose` is the (pos, lookat, up, fovy)
quadruple that `RobotSceneSim.render_free_camera` consumes.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Sanity bounds applied at proposer output (no LLM round wasted on out-of-range proposals).
DIST_MIN_M = 0.20
DIST_MAX_M = 1.50
POS_Z_MIN_M = 0.0
POS_Z_MAX_M = 1.50
FOVY_MIN_DEG = 20.0
FOVY_MAX_DEG = 80.0


@dataclass
class CameraPose:
    """A free-camera pose in world coordinates (meters, z-up convention)."""
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fovy_deg: float = 45.0

    def as_np(self) -> dict[str, np.ndarray]:
        return {
            "pos": np.asarray(self.pos, dtype=np.float64),
            "lookat": np.asarray(self.lookat, dtype=np.float64),
            "up": np.asarray(self.up, dtype=np.float64),
        }

    @classmethod
    def from_cam2world(
        cls,
        cam2world: np.ndarray,
        fovy_deg: float = 45.0,
        lookat_dist_m: float = 0.5,
    ) -> "CameraPose":
        """Convert a (4, 4) cam→world transform (OpenCV +Z forward, +Y down)
        into a CameraPose. The lookat point is placed `lookat_dist_m`
        forward along the camera's optical axis.
        """
        cam2world = np.asarray(cam2world, dtype=np.float64).reshape(4, 4)
        R = cam2world[:3, :3]
        pos = cam2world[:3, 3]
        forward_world = R @ np.array([0.0, 0.0, 1.0])    # +Z is forward in OpenCV
        up_world = R @ np.array([0.0, -1.0, 0.0])        # -Y is up in OpenCV
        lookat = pos + forward_world * float(lookat_dist_m)
        return cls(
            pos=tuple(pos.astype(float)),
            lookat=tuple(lookat.astype(float)),
            up=tuple(up_world.astype(float)),
            fovy_deg=float(fovy_deg),
        )


def validate_bounds(pose: CameraPose) -> Optional[str]:
    """Return None if pose is within sanity bounds, otherwise a short reason."""
    pos = np.asarray(pose.pos, dtype=np.float64)
    lookat = np.asarray(pose.lookat, dtype=np.float64)
    dist = float(np.linalg.norm(pos - lookat))
    if not (DIST_MIN_M <= dist <= DIST_MAX_M):
        return f"distance {dist:.2f}m not in [{DIST_MIN_M}, {DIST_MAX_M}]"
    if not (POS_Z_MIN_M <= pos[2] <= POS_Z_MAX_M):
        return f"pos.z {pos[2]:.2f}m not in [{POS_Z_MIN_M}, {POS_Z_MAX_M}]"
    if not (FOVY_MIN_DEG <= pose.fovy_deg <= FOVY_MAX_DEG):
        return f"fovy {pose.fovy_deg:.1f}° not in [{FOVY_MIN_DEG}, {FOVY_MAX_DEG}]"
    if np.linalg.norm(pose.up) < 1e-6:
        return "up vector has zero length"
    return None
