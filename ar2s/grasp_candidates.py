"""Shared deterministic policy for selecting grasp artifacts.

The cleanup and Blender handoff paths use the same candidate policy so that
they retain and package the same successful samples.  This module deliberately
contains no evaluation-specific dependencies.
"""
from __future__ import annotations

import math
from typing import Any


CANDIDATE_TOP_K = 5
MIN_PEAK_LIFT_MM = -100.0
MAX_PEAK_DISPLACEMENT_MM = 2000.0
METRIC_FIELDS = ("peak_lift_mm", "peak_displacement_mm")


def _metric_float(entry: dict[str, Any], key: str) -> float:
    """Return a metric as a float, using NaN for missing values."""
    value = entry.get(key)
    return float(value if value is not None else float("nan"))


def candidate_rejection_reason(
    entry: dict[str, Any],
    *,
    check_video: bool = True,
) -> str | None:
    """Return why an entry is not an eligible grasp candidate, if any."""
    if not entry.get("grasped"):
        return "not_grasped"
    if check_video and not entry.get("video_path"):
        return "missing_video"
    lift_mm = _metric_float(entry, "peak_lift_mm")
    displacement_mm = _metric_float(entry, "peak_displacement_mm")
    if not math.isfinite(lift_mm) or not math.isfinite(displacement_mm):
        return "non_finite_metric"
    if lift_mm < MIN_PEAK_LIFT_MM:
        return "lift_below_min"
    if displacement_mm > MAX_PEAK_DISPLACEMENT_MM:
        return "displacement_above_max"
    return None


def expected_candidate_indices(
    entries: list[dict[str, Any]],
    *,
    check_video: bool = True,
) -> list[str]:
    """Return eligible sample indices in deterministic best-first order."""
    candidates = [
        entry
        for entry in entries
        if candidate_rejection_reason(entry, check_video=check_video) is None
    ]
    candidates = sorted(
        candidates,
        key=lambda entry: (
            -_metric_float(entry, "peak_displacement_mm"),
            str(entry.get("sample_idx", "")),
        ),
    )[:CANDIDATE_TOP_K]
    return [str(entry["sample_idx"]) for entry in candidates]


__all__ = [
    "CANDIDATE_TOP_K",
    "MAX_PEAK_DISPLACEMENT_MM",
    "METRIC_FIELDS",
    "MIN_PEAK_LIFT_MM",
    "candidate_rejection_reason",
    "expected_candidate_indices",
]
