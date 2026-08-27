"""Stage 1 - Motion Reader Agent.

Given a directory that contains a paired reference motion (.npz) and latent
context (.pkl) plus a path to the ONNX policy model, this LangGraph ReAct
agent:

  1. Lists the directory to discover the .npz, .pkl, and .onnx files.
  2. Inspects the .npz to learn the frame count, joint names, and array keys.
  3. Inspects the .pkl to learn the latent dimension, row count, and structure.
  4. Determines the task type (goal vs tracking) and the correct z parameters.
  5. Submits a validated MotionBundle via submit_bundle.

Public API
----------
    from ar2s.agents_g1sysid.agents.motion_reader_agent import run_motion_reader_agent
    bundle = run_motion_reader_agent(
        data_dir="raw_motions/dance_v0/",
        onnx_model_path="model/exported/FBcprAuxModel.onnx",
    )
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ar2s.agent_configs import chat_model_for
from ar2s.agents_g1sysid.inputs import MotionBundle
from ar2s.agents_g1sysid.skills.motion_inspect import inspect_npz, inspect_pkl, list_motion_dir
from ar2s.agents_g1sysid.skills.bundle_submit import _SubmissionSlot, make_submit_tool

_RECURSION_LIMIT = 20

_SYSTEM_PROMPT = """\
You are a Motion Reader Agent for a Unitree G1 humanoid SysID pipeline.

# Goal

Given a directory containing a reference motion (.npz) and a paired latent
context (.pkl), plus a path to an ONNX policy model, inspect the files and
produce a valid MotionBundle by calling submit_bundle.

# Available Tools

Read-only inspection:
  - list_motion_dir(path)           list .npz / .pkl / .onnx files with sizes
  - inspect_npz(path)               keys, shapes, frame count, joint names
  - inspect_pkl(path)               z_dim, n_z_rows, structure type

Termination (call last):
  - submit_bundle(submission_json)
    Strict validation. Returns "OK" on success, or a JSON error to fix+retry.
    submission_json is a single JSON string with ALL required fields (see below).

# MotionBundle Fields

  motion_path      - absolute path to the reference .npz
  pkl_path         - absolute path to the paired .pkl
  onnx_model_path  - absolute path to the .onnx policy model (given to you)
  task_type        - "goal" or "tracking" (see rules below)
  goal_frame       - (goal only) integer frame index in .npz for the goal pose
  z_row_start      - (tracking only) first row in the pkl array (inclusive)
  z_row_end        - (tracking only) last row + 1 (exclusive); MUST equal z_row_start + n_frames
  n_frames         - total frames in the .npz (shape[0] of dof_positions)
  z_dim            - latent dimension from inspect_pkl
  n_z_rows         - total rows in pkl from inspect_pkl
  description      - one-sentence natural-language description of the motion
  min_height       - float, pelvis fall threshold in metres (default 0.3;
                      use 0.0 for fall-and-get-up motions)

# Task Type Rules

Use "tracking" when:
  - The pkl is a raw (N, z_dim) array (structure="array") AND n_z_rows >= n_frames.
    In this case set z_row_start=0, z_row_end=n_frames.
  - If the user or data directory name hints at a specific pkl row slice,
    use those numbers instead.

Use "goal" when:
  - The pkl is a dict (structure="dict") - it contains a single z vector.
    Set goal_frame = n_frames - 1 (last frame) unless the name hints otherwise.
  - Or when n_z_rows < n_frames (not enough z rows for tracking).

# Workflow

1. list_motion_dir(data_dir) to see all files.
2. inspect_npz on the .npz file -> record n_frames, z_dim check (if possible).
3. inspect_pkl on the .pkl file -> record z_dim, n_z_rows, structure.
4. Determine task_type and z parameters using the rules above.
5. submit_bundle with all fields filled in.
   On error, fix the specified fields and retry (up to 5 attempts total).

# Important

- Use ABSOLUTE paths for all path fields (expand ~ and resolve relative paths).
- Do NOT guess file paths - only use paths you confirmed exist via list_motion_dir
  or inspect_* tools.
- The .npz and .pkl are PAIRED: trust the pairing, just validate dimensions.
"""


def _system_with_caching() -> SystemMessage:
    return SystemMessage(content=[{
        "type": "text",
        "text": _SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }])


def run_motion_reader_agent(
    data_dir: str | Path,
    onnx_model_path: str | Path,
    recursion_limit: int = _RECURSION_LIMIT,
) -> MotionBundle:
    """Run the Stage-1 Motion Reader Agent on a raw data directory.

    Args:
        data_dir:        directory containing the paired .npz and .pkl.
        onnx_model_path: path to the ONNX policy model.
        recursion_limit: cap on LLM ↔ tool round-trips.

    Returns:
        Validated MotionBundle.

    Raises:
        RuntimeError: no API key, agent crashed, or max submission attempts hit.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    onnx_model_path = Path(onnx_model_path).expanduser().resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"data_dir {data_dir} is not a directory")
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    slot = _SubmissionSlot()
    submit_tool = make_submit_tool(slot)

    llm = chat_model_for("g1sysid.main.motion_reader_agent")
    if llm is None:
        raise RuntimeError(
            "No LLM provider/API key configured. "
            "Set the credential env var for the provider selected in "
            "ar2s/agent_configs/config_anthropic_opus47.yaml, or pass --agent-config."
        )

    agent = create_react_agent(
        model=llm,
        tools=[list_motion_dir, inspect_npz, inspect_pkl, submit_tool],
        prompt=_system_with_caching(),
    )

    initial_msg = (
        f"Data directory to scan: {data_dir}\n"
        f"ONNX model path: {onnx_model_path}\n\n"
        "Start by listing the data directory, then inspect the .npz and .pkl "
        "files to determine the task type and z parameters, then call submit_bundle."
    )

    try:
        agent.invoke(
            {"messages": [HumanMessage(content=initial_msg)]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as e:
        if slot.submitted:
            return slot.submitted
        raise RuntimeError(
            f"Motion Reader Agent crashed before submitting: {type(e).__name__}: {e}. "
            f"Failed attempts: {slot.failed_attempts}. "
            f"Last error: {slot.last_error or '(none)'}"
        ) from e

    if slot.aborted:
        raise RuntimeError(
            f"Motion Reader Agent aborted: max submission attempts reached "
            f"({slot.failed_attempts}). Last error: {slot.last_error or '(none)'}"
        )

    if slot.submitted is None:
        raise RuntimeError(
            f"Motion Reader Agent terminated without calling submit_bundle. "
            f"Failed attempts: {slot.failed_attempts}. "
            f"Last error: {slot.last_error or '(none)'}"
        )

    return slot.submitted


__all__ = ["run_motion_reader_agent"]
