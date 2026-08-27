"""Stage-1 termination skill - validates and commits a MotionBundle.

The LLM calls submit_bundle(submission_json=...) as its final action.
Strict validation is done here; errors are returned as strings so the agent
can fix and retry (up to MAX_ATTEMPTS).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from langchain_core.tools import tool

from ar2s.agents_g1sysid.inputs import MotionBundle

MAX_ATTEMPTS = 5


# ---------------------------------------------------------------------------
# Submission slot (shared mutable state between make_submit_tool and the agent)
# ---------------------------------------------------------------------------

@dataclass
class _SubmissionSlot:
    submitted: Optional[MotionBundle] = None
    failed_attempts: int = 0
    last_error: Optional[str] = None
    aborted: bool = False


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def _validate_submission(data: dict) -> tuple[Optional[MotionBundle], Optional[str]]:
    """Parse and validate the submission JSON.

    Returns (bundle, None) on success or (None, error_message) on failure.
    """
    required = [
        "motion_path", "pkl_path", "onnx_model_path", "task_type",
        "n_frames", "z_dim", "n_z_rows", "description",
    ]
    for key in required:
        if key not in data:
            return None, f"Missing required field: {key!r}"

    task = data["task_type"]
    if task not in ("goal", "tracking"):
        return None, f"task_type must be 'goal' or 'tracking', got {task!r}"

    if task == "goal":
        if data.get("goal_frame") is None:
            return None, "goal task requires goal_frame"
        if not isinstance(data["goal_frame"], int) or data["goal_frame"] < 0:
            return None, f"goal_frame must be a non-negative int, got {data.get('goal_frame')!r}"
    elif task == "tracking":
        if data.get("z_row_start") is None or data.get("z_row_end") is None:
            return None, "tracking task requires z_row_start and z_row_end"
        if data["z_row_end"] <= data["z_row_start"]:
            return None, (
                f"z_row_end ({data['z_row_end']}) must be greater than "
                f"z_row_start ({data['z_row_start']})"
            )
        seg_len = data["z_row_end"] - data["z_row_start"]
        if seg_len != data["n_frames"]:
            return None, (
                f"z segment length ({seg_len}) must equal n_frames ({data['n_frames']})"
            )

    # File existence checks
    for field in ("motion_path", "pkl_path", "onnx_model_path"):
        p = Path(data[field])
        if not p.exists():
            return None, f"{field}: file not found at {p}"

    bundle = MotionBundle(
        motion_path=str(Path(data["motion_path"]).resolve()),
        pkl_path=str(Path(data["pkl_path"]).resolve()),
        onnx_model_path=str(Path(data["onnx_model_path"]).resolve()),
        task_type=task,
        goal_frame=data.get("goal_frame"),
        z_row_start=data.get("z_row_start"),
        z_row_end=data.get("z_row_end"),
        n_frames=int(data["n_frames"]),
        z_dim=int(data["z_dim"]),
        n_z_rows=int(data["n_z_rows"]),
        description=str(data["description"]),
        min_height=float(data.get("min_height", 0.3)),
    )
    return bundle, None


# ---------------------------------------------------------------------------
# Factory - creates a closure-wrapped @tool bound to a slot
# ---------------------------------------------------------------------------

def make_submit_tool(slot: _SubmissionSlot) -> Callable:
    """Return a LangChain @tool that writes validated submissions into *slot*."""

    @tool
    def submit_bundle(submission_json: str) -> str:
        """Submit the MotionBundle description for validation.

        Call this LAST, after you have inspected both the .npz and .pkl files.

        Args:
            submission_json: JSON string with ALL of the following fields:
              motion_path      - absolute path to reference .npz
              pkl_path         - absolute path to latent .pkl
              onnx_model_path  - absolute path to .onnx policy model
              task_type        - "goal" or "tracking"
              goal_frame       - (goal only) int frame index
              z_row_start      - (tracking only) int first pkl row
              z_row_end        - (tracking only) int exclusive end pkl row
              n_frames         - int total frames in .npz
              z_dim            - int latent dimension
              n_z_rows         - int total rows in pkl
              description      - one-sentence description of the motion
              min_height       - float pelvis fall threshold (default 0.3)

        On success, returns "OK".
        On failure, returns a JSON object with "errors" so you can fix and retry.
        """
        if slot.aborted:
            return json.dumps({"errors": ["Max submission attempts reached. Stop."]})

        try:
            data = json.loads(submission_json)
        except json.JSONDecodeError as e:
            slot.failed_attempts += 1
            slot.last_error = f"JSON parse error: {e}"
            if slot.failed_attempts >= MAX_ATTEMPTS:
                slot.aborted = True
            return json.dumps({"errors": [slot.last_error]})

        bundle, err = _validate_submission(data)
        if err:
            slot.failed_attempts += 1
            slot.last_error = err
            if slot.failed_attempts >= MAX_ATTEMPTS:
                slot.aborted = True
            return json.dumps({
                "errors": [err],
                "attempts_remaining": MAX_ATTEMPTS - slot.failed_attempts,
            })

        slot.submitted = bundle
        return "OK"

    return submit_bundle


__all__ = ["MAX_ATTEMPTS", "_SubmissionSlot", "make_submit_tool"]
