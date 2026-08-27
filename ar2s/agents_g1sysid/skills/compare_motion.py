"""Motion comparison skill - wraps BFM-Zero compare_motion.py metrics.

Used by Stage 3 after the final rollout to measure how well the recovered
physics reproduces the reference motion.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from langchain_core.tools import tool as _lc_tool

_BFM_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bfm_zero")
)


def _ensure_bfm_on_path() -> None:
    if _BFM_ROOT not in sys.path:
        sys.path.insert(0, _BFM_ROOT)


@_lc_tool
def compare_trajectory(recording_path: str, reference_path: str,
                        frame_start: int = 0,
                        frame_end: int | None = None) -> str:
    """Compare a simulation recording NPZ against a reference motion NPZ.

    Computes trajectory-mode metrics (all frames aligned in time):
      mpjpe_m           - mean per-joint position error (metres)
      mean_dof_err_deg  - mean joint angle error (degrees)
      mean_rot_err_deg  - mean body rotation error, geodesic (degrees)
      mean_root_drift_m - mean pelvis drift vs reference (metres)
      max_root_drift_m  - peak pelvis drift (metres)

    Args:
        recording_path: path to the recorded (simulated) .npz
        reference_path: path to the reference (ground-truth) .npz
        frame_start:    first reference frame to include (default 0)
        frame_end:      exclusive end frame (default = all frames)

    Returns a JSON string with the scalar metrics above.
    """
    rec_p = Path(recording_path).expanduser().resolve()
    ref_p = Path(reference_path).expanduser().resolve()
    for p in (rec_p, ref_p):
        if not p.exists():
            return json.dumps({"error": f"File not found: {p}"})

    import numpy as np
    _ensure_bfm_on_path()
    from compare_motion import compute_metrics  # type: ignore[import]

    rec_raw = np.load(str(rec_p))
    ref_raw = np.load(str(ref_p))

    T_rec = rec_raw["body_positions"].shape[0]
    fe = frame_end if frame_end is not None else frame_start + T_rec
    ref = {
        "body_positions": ref_raw["body_positions"][frame_start:fe],
        "body_rotations": ref_raw["body_rotations"][frame_start:fe],
        "dof_positions":  ref_raw["dof_positions"][frame_start:fe],
    }
    rec = {
        "body_positions": rec_raw["body_positions"],
        "body_rotations": rec_raw["body_rotations"],
        "dof_positions":  rec_raw["dof_positions"],
    }
    m = compute_metrics(rec, ref, root_align=True)
    return json.dumps({
        "recording": str(rec_p),
        "reference": str(ref_p),
        "T": int(m["T"]),
        "mpjpe_m": float(m["mpjpe_m"]),
        "mean_dof_err_deg": float(m["mean_dof_err_deg"]),
        "mean_rot_err_deg": float(m["mean_rot_err_deg"]),
        "mean_root_drift_m": float(m["mean_root_drift_m"]),
        "max_root_drift_m": float(m["max_root_drift_m"]),
    }, indent=2)


def compute_trajectory_metrics(recording_path: str, reference_path: str,
                                frame_start: int = 0,
                                frame_end: int | None = None) -> dict:
    """Non-tool version: returns the metrics dict directly (used by sysid.py)."""
    import numpy as np
    _ensure_bfm_on_path()
    from compare_motion import compute_metrics  # type: ignore[import]

    rec_raw = np.load(recording_path)
    ref_raw = np.load(reference_path)

    T_rec = rec_raw["body_positions"].shape[0]
    fe = frame_end if frame_end is not None else frame_start + T_rec
    ref = {
        "body_positions": ref_raw["body_positions"][frame_start:fe],
        "body_rotations": ref_raw["body_rotations"][frame_start:fe],
        "dof_positions":  ref_raw["dof_positions"][frame_start:fe],
    }
    rec = {
        "body_positions": rec_raw["body_positions"],
        "body_rotations": rec_raw["body_rotations"],
        "dof_positions":  rec_raw["dof_positions"],
    }
    return compute_metrics(rec, ref, root_align=True)


__all__ = ["compare_trajectory", "compute_trajectory_metrics"]
