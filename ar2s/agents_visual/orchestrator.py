"""Top-level ReAct orchestrator for ar2s.agents_visual.

Three exported callables — the unified pipeline driver runs them in this order:

  data_ingest(visual_input, run_root=None) -> run_root
      Validate paths + seed state.json. NO LLM, plain Python.

  run_visual(run_root, *, max_iterations) -> dict
      ReAct agent over the 13 visual tools (see orchestrator_prompt.md).
      Dependency graph lives in the prompt; segment owns its retry loop
      internally.

  finalize(run_root, *, target_dir) -> str
      Call resolve() + emit() (both plain Python). Returns the path to
      the materialised episode folder. The CLI invokes this after the
      ReAct agent completes successfully.

The 13 tools are imported below and bound to ``create_react_agent``.
data_ingest / resolve / emit are intentionally NOT exposed as tools —
they're deterministic glue the orchestrator controls, not LLM-decided
steps. (See plan doc Section "三类节点": top-level direct-own.)
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ar2s.pipeline_artifacts import pipeline_run_dir
from ar2s.agent_configs.models import chat_model_for
from ar2s.agents_visual.droid_episode_id import is_valid_episode_id
from ar2s.agents_visual.inputs import VisualInput
from ar2s.agents_visual.outputs import write_episode_folder
from ar2s.agents_visual.resolve import resolve as _resolve
from ar2s.agents_visual.state import load_state, save_state, state_path
from ar2s.droid_sim.scene.robot_profile import get_robot_profile

# ---- 13 @tool entries (7 skill incl. scale_critic + 6 subagent) ----
from ar2s.agents_visual.skills.svo_extract   import svo_extract
from ar2s.agents_visual.skills.video_build   import video_build
from ar2s.agents_visual.skills.mesh_recover  import mesh_recover
from ar2s.agents_visual.skills.stereo_depth  import stereo_depth
from ar2s.agents_visual.skills.mesh_scale    import mesh_scale
from ar2s.agents_visual.skills.scale_critic  import scale_critic
from ar2s.agents_visual.skills.pose_tracking import pose_tracking

from ar2s.agents_visual.subagents.object_discovery   import (
    object_discovery,
    _MAX_OD_CONSECUTIVE_FAILURES,
    _OD_FAST_FAIL_KEY,
)
from ar2s.agents_visual.subagents.pickup_objects     import pickup_objects
from ar2s.agents_visual.subagents.support_tree       import support_tree
from ar2s.agents_visual.subagents.segment_controller import segment
from ar2s.agents_visual.subagents.ground_ref         import ground_ref
from ar2s.agents_visual.subagents.tracking_critic    import tracking_critic


_TOOLS = [
    # Stage skills
    svo_extract, video_build, mesh_recover,
    stereo_depth, mesh_scale, scale_critic, pose_tracking,
    # Decision subagents
    object_discovery, pickup_objects, support_tree,
    # Unified segment controller (replaces the old keyframe_select +
    # segmentation + mask_critic chain — see subagents/segment_controller.py).
    segment,
    ground_ref, tracking_critic,
]

_PROMPT_PATH = Path(__file__).with_name("orchestrator_prompt.md")
# Default model lives in ``ar2s/agent_configs/config_anthropic_opus47.yaml`` under
# ``visual.main.orchestrator``.
_DEFAULT_MAX_ITERATIONS = 50                    # 13 tools * ~2 trips = 26 baseline;
                                                # 50 gives ~20 extra round-trips of headroom
                                                # for segment-controller retries and
                                                # warning-only critic calls.
_MAX_NUDGES = 40                                 # recovery attempts for Rule 6 violations
                                                # (model emits a "next: <stage>" message
                                                # with no tool call before ground_ref) —
                                                # see run_visual().
_TERMINAL_PREFIXES = ("Pipeline complete", "Pipeline halted")

# data_ingest is a consumer of already-exported artifacts: it validates that
# the rectified stereo MP4/K sidecar and PointWorld camera extrinsics already
# exist and fails fast if not. It never generates, converts, or copies those
# artifacts itself — that is the export step's job (see
# docs/droid_data_utils.md and ar2s.agents_visual.raw_episode.build_raw_episode).
_EXPORT_HINT = (
    "run the export step first (`python -m scripts.export_episodes_from_raw`; "
    "see docs/droid_data_utils.md#raw-episode-export-artifacts) — data_ingest "
    "does not generate or fetch this on its own"
)


# ---------------------------------------------------------------------------
# data_ingest — plain Python, no LLM
# ---------------------------------------------------------------------------

def data_ingest(
    visual_input: VisualInput,
    run_root: str | None = None,
) -> str:
    """Validate VisualInput + seed state.json. Returns the run_root path.

    Mirrors v1's data_ingest node but writes to disk-state instead of
    returning a graph dict.

    Args:
        visual_input: user's filled-in VisualInput dataclass.
        run_root: optional override. Defaults to
            ``outputs/run_pipeline/<episode_id>``.

    Raises:
        ValueError if any required path is missing.
    """
    errors: list[str] = []
    stereo_mp4_path = Path(visual_input.stereo_stream_path).expanduser()
    stereo_intrinsics_path = Path(visual_input.stereo_intrinsics_path).expanduser()
    robot_traj_path = Path(visual_input.robot_traj_path).expanduser()

    if not visual_input.stereo_stream_path:
        errors.append(
            "stereo_stream_path is empty; expected an exported rectified "
            f"stereo .mp4 — {_EXPORT_HINT}"
        )
    elif not stereo_mp4_path.exists():
        errors.append(
            f"stereo_stream_path not found: {visual_input.stereo_stream_path} "
            f"— {_EXPORT_HINT}"
        )
    elif stereo_mp4_path.suffix.lower() != ".mp4":
        errors.append(
            "stereo_stream_path must point to an exported rectified stereo .mp4; "
            f"got {visual_input.stereo_stream_path}"
        )
    if not visual_input.stereo_intrinsics_path:
        errors.append(
            "stereo_intrinsics_path is empty; expected exporter-produced "
            f"<serial>-stereo.K.txt — {_EXPORT_HINT}"
        )
    elif not stereo_intrinsics_path.exists():
        errors.append(
            f"stereo_intrinsics_path not found: {visual_input.stereo_intrinsics_path} "
            f"— {_EXPORT_HINT}"
        )
    if not robot_traj_path.exists():
        errors.append(f"robot_traj_path not found: {visual_input.robot_traj_path}")

    # Validation-only read: robot_type reaches the manifest through
    # state["entry"] (asdict of this VisualInput), the same route
    # task_description takes. It is not consumed until finalize writes
    # manifest.robot.type, so without this an unknown name would only surface
    # after the whole visual chain has run. getattr-guarded for VisualInput
    # bundles written before the field existed.
    try:
        get_robot_profile(getattr(visual_input, "robot_type", "") or "franka_panda")
    except KeyError as e:
        errors.append(f"robot_type: {e}")

    secondary_stream = getattr(visual_input, "secondary_stereo_stream_path", "") or ""
    secondary_alias = getattr(visual_input, "secondary_svo_path", "") or ""
    if not secondary_stream and secondary_alias:
        secondary_stream = secondary_alias
    secondary_intrinsics = getattr(visual_input, "secondary_stereo_intrinsics_path", "") or ""
    secondary_camera_id = getattr(visual_input, "secondary_camera_id", "") or ""
    secondary_stream_path = Path(secondary_stream).expanduser() if secondary_stream else None
    secondary_intrinsics_path = (
        Path(secondary_intrinsics).expanduser() if secondary_intrinsics else None
    )
    secondary_configured = bool(secondary_stream or secondary_intrinsics or secondary_camera_id)
    if secondary_configured:
        if not secondary_stream:
            errors.append(
                "secondary_stereo_stream_path is required when secondary "
                "intrinsics or secondary_camera_id is set"
            )
        elif not secondary_stream_path.exists():
            errors.append(f"secondary_stereo_stream_path not found: {secondary_stream}")
        elif secondary_stream_path.suffix.lower() != ".mp4":
            errors.append(
                "secondary_stereo_stream_path must point to an exported "
                f"rectified stereo .mp4; got {secondary_stream}"
            )
        if not secondary_intrinsics:
            errors.append(
                "secondary_stereo_intrinsics_path is required when a secondary "
                "view is configured"
            )
        elif not secondary_intrinsics_path.exists():
            errors.append(
                f"secondary_stereo_intrinsics_path not found: {secondary_intrinsics}"
            )
        if not secondary_camera_id:
            errors.append("secondary_camera_id is required when a secondary view is configured")

    # cameras_extrinsics is required by write_episode_folder() at finalize time;
    # validate up front to avoid burning 20-30 min of pipeline on a missing file.
    extrinsics_path = (
        Path(visual_input.cameras_extrinsics_path).expanduser()
        if visual_input.cameras_extrinsics_path else None
    )
    cam_keys: set[str] = set()
    if not visual_input.cameras_extrinsics_path:
        errors.append(
            "cameras_extrinsics_path is empty — expected a .npz with one "
            "or more `cam_mat_<serial>` keys (4x4 T_cam_from_world / w2c) "
            f"derived from the PointWorld cameras json; {_EXPORT_HINT}"
        )
    elif not extrinsics_path.exists():
        errors.append(
            f"cameras_extrinsics_path not found: {visual_input.cameras_extrinsics_path} "
            f"— {_EXPORT_HINT}"
        )
    else:
        import numpy as np
        with np.load(extrinsics_path) as data:
            cam_keys = {k for k in data.files if k.startswith("cam_mat_")}
        primary_camera_id = getattr(visual_input, "primary_camera_id", "") or ""
        if primary_camera_id:
            expected = f"cam_mat_{primary_camera_id}"
            if expected not in cam_keys:
                errors.append(
                    f"cameras_extrinsics_path lacks {expected}; found {sorted(cam_keys)}"
                )
        elif len(cam_keys) != 1:
            errors.append(
                "primary_camera_id is required when cameras_extrinsics_path "
                f"contains {len(cam_keys)} cam_mat_<serial> keys: {sorted(cam_keys)}"
            )
        if secondary_configured and secondary_camera_id:
            expected = f"cam_mat_{secondary_camera_id}"
            if expected not in cam_keys:
                errors.append(
                    f"cameras_extrinsics_path lacks {expected}; found {sorted(cam_keys)}"
                )
    if not visual_input.episode_id:
        errors.append("episode_id is empty")
    elif not is_valid_episode_id(visual_input.episode_id):
        errors.append(
            "episode_id must be identifier-safe "
            "(start with a letter, then letters/digits/underscores); "
            f"got {visual_input.episode_id!r}"
        )
    if errors:
        raise ValueError("data_ingest: " + "; ".join(errors))

    if run_root is None:
        run_root = str(pipeline_run_dir(visual_input.episode_id))
    run_root_abs = str(Path(run_root).expanduser().resolve())
    Path(run_root_abs).mkdir(parents=True, exist_ok=True)

    secondary_stream_resolved = str(secondary_stream_path.resolve()) if secondary_stream_path else ""
    secondary_intrinsics_resolved = (
        str(secondary_intrinsics_path.resolve()) if secondary_intrinsics_path else ""
    )

    # Resolve input paths now while CWD is authoritative; toolkit subprocesses
    # may run from stage-specific working directories.
    state: dict = {
        "entry":                  asdict(visual_input) if is_dataclass(visual_input) else dict(visual_input),
        "stereo_stream_path":     str(stereo_mp4_path.resolve()),
        "stereo_intrinsics_path": str(stereo_intrinsics_path.resolve()),
        "secondary_stereo_stream_path": secondary_stream_resolved,
        "secondary_stereo_intrinsics_path": secondary_intrinsics_resolved,
        # Deprecated compatibility alias for incoming PR-3 code. This is a
        # rectified MP4 path, not an SVO path.
        "secondary_svo_path":     secondary_stream_resolved,
        "primary_camera_id":      getattr(visual_input, "primary_camera_id", "") or "",
        "secondary_camera_id":    secondary_camera_id,
        "robot_traj_path":        str(robot_traj_path.resolve()),
        "episode_id":             visual_input.episode_id,
        "scene_name":             visual_input.scene_name or visual_input.episode_id,
        "run_root":               run_root_abs,
        "max_frames":             visual_input.max_frames,
        "heavy_max_frames":       visual_input.heavy_max_frames,
        "pose_shorter_side":      getattr(visual_input, "pose_shorter_side", None),
        "frame_step":             visual_input.frame_step,
        # stereo_depth (FoundationStereo) speed/quality knobs; read by the
        # stereo_depth skill. getattr-guarded for older VisualInput bundles.
        "stereo_valid_iters":     getattr(visual_input, "stereo_valid_iters", 22),
        "get_pointcloud":         getattr(visual_input, "stereo_get_pointcloud", False),
        "cameras_extrinsics_path": (
            str(extrinsics_path.resolve())
            if extrinsics_path else ""
        ),
        # Seed stages dict so skills can immediately record_stage().
        "stages":  {},
        # Empty audit trail; subagents append to it via append_history().
        "history": [],
    }
    save_state(state, run_root_abs)

    # Mkdir the standard subdirs early so subprocess log writes succeed even
    # before the first tool runs.
    for sub in ("logs", "seq", "segmentation", "meshes", "poses"):
        (Path(run_root_abs) / sub).mkdir(exist_ok=True)

    print(f"[data_ingest] seeded {state_path(run_root_abs)}")
    print(f"[data_ingest] run_root: {run_root_abs}")
    return run_root_abs


# ---------------------------------------------------------------------------
# run_visual — top-level ReAct agent
# ---------------------------------------------------------------------------

def _system_message_with_caching() -> SystemMessage:
    """Build a SystemMessage with Anthropic prompt caching on the static body.

    The system prompt is long (~150 lines including the dependency graph
    + tool inventory + rules). With caching, subsequent agent turns read
    it from cache (~10× cheaper). Non-Anthropic providers ignore the
    ``cache_control`` field harmlessly.
    """
    return SystemMessage(content=[{
        "type":          "text",
        "text":          _PROMPT_PATH.read_text(),
        "cache_control": {"type": "ephemeral"},
    }])


def _last_message_text(result: dict) -> str:
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    content = getattr(msgs[-1], "content", "") or ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _stopped_without_tool_call(result: dict) -> bool:
    """True if the agent ended on a text-only message that isn't terminal.

    Rule 6 in orchestrator_prompt.md requires a tool call in every response
    before ``ground_ref`` succeeds. Weaker models sometimes announce a
    "next: <stage>" step in plain text instead of calling it; LangGraph
    treats any tool-call-free AIMessage as "agent is done", which would
    otherwise abort the run early with later stages unrun.
    """
    msgs = result.get("messages") or []
    if not msgs or getattr(msgs[-1], "tool_calls", None):
        return False
    return not _last_message_text(result).startswith(_TERMINAL_PREFIXES)


def _raise_if_od_fast_fail(run_root: str) -> None:
    """Abort the visual pipeline if object_discovery latched its fast-fail flag.

    object_discovery latches ``_OD_FAST_FAIL_KEY`` in state.json after
    ``_MAX_OD_CONSECUTIVE_FAILURES`` consecutive all-VLM failures — the
    "wrist-camera" truncation loop where the discovery VLM degenerates into an
    over-long response that hits the completion-token cap and never parses.
    Without this guard the orchestrator keeps nudging object_discovery, each
    retry regenerating a ~16k-token response (minutes each), until the batch
    driver's 90-min wall-clock timeout kills the episode.

    We raise here — in run_visual, OUTSIDE the create_react_agent ToolNode that
    would otherwise CATCH a tool-level exception and feed it back to the LLM as
    just another retry. A RuntimeError from run_visual propagates up to
    run_pipeline's ``visual_processing`` tool, which turns it into a clean
    ``stage error: visual_processing`` within ~1-2 min instead of a 60-90 min
    nudge loop.
    """
    if load_state(run_root).get(_OD_FAST_FAIL_KEY):
        raise RuntimeError(
            "object_discovery fast-fail: all VLMs failed "
            f"{_MAX_OD_CONSECUTIVE_FAILURES} times in a row (wrist-cam "
            "truncation loop); aborting the visual pipeline instead of "
            "nudge-looping to the batch wall-clock timeout"
        )


def run_visual(
    run_root: str,
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> dict:
    """Run the ReAct agent over the 13 visual tools.

    Args:
        run_root: absolute path to ``outputs/run_pipeline/<run_id>/``. Must already contain
            state.json seeded by data_ingest().
        max_iterations: cap on LLM↔tool round-trips. 50 = ~1.9x the 13-tool
            ideal (26 trips); covers segment-controller retry overhead.

    Returns:
        The LangGraph agent's final state (`{"messages": [...]}`). The
        last AIMessage in messages is the agent's terminal summary;
        the CLI parses it for success/halt. If the agent stops on a
        Rule 6 violation (text-only message, no tool call, before
        ``ground_ref``), it is nudged to continue, up to ``_MAX_NUDGES``
        times, before giving up and returning the stalled result as-is.

    Raises:
        FileNotFoundError if state.json doesn't exist at run_root.
        RuntimeError if no LLM API key is configured.
    """
    if not Path(state_path(run_root)).exists():
        raise FileNotFoundError(
            f"state.json missing at {state_path(run_root)} — call data_ingest first"
        )

    llm = chat_model_for("visual.main.orchestrator")
    if llm is None:
        raise RuntimeError(
            "No API key for the configured top-level visual orchestrator. "
            "Set the credential env var for the provider selected in "
            "ar2s/agent_configs/config_anthropic_opus47.yaml or pass --agent-config "
            "to choose another YAML config."
        )

    agent = create_react_agent(
        model=llm,
        tools=_TOOLS,
        prompt=_system_message_with_caching(),
    )

    initial_msg = (
        f"run_root = {run_root}\n\n"
        f"Run the visual pipeline end-to-end on this run directory. "
        f"Follow the dependency graph in the system prompt; segment owns "
        f"its own retry loop, so don't call segmentation/keyframe_select/"
        f"mask_critic as peer tools."
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=initial_msg)]},
        config={"recursion_limit": max_iterations},
    )
    # Fast-fail guard (wrist-cam truncation loop): bail before nudging if
    # object_discovery already latched its terminal flag during this run.
    _raise_if_od_fast_fail(run_root)

    nudge_msg = (
        "You stopped with a text-only response and no tool call, which Rule 6 "
        "forbids before `ground_ref` succeeds. Check state.json's `stages` and "
        "`history` for what has already run, then call the next tool in the "
        "dependency graph now — do not just announce it."
    )
    nudges_left = _MAX_NUDGES
    while nudges_left > 0 and _stopped_without_tool_call(result):
        nudges_left -= 1
        print(
            "[run_visual] agent stopped without a tool call before ground_ref "
            f"(text: {_last_message_text(result)[:200]!r}); nudging "
            f"({_MAX_NUDGES - nudges_left}/{_MAX_NUDGES})"
        )
        result = agent.invoke(
            {"messages": result["messages"] + [HumanMessage(content=nudge_msg)]},
            config={"recursion_limit": max_iterations},
        )
        # Re-nudging a doomed wrist-cam episode just burns another ~16k-token
        # truncated regeneration per attempt; abort the moment the guard latches.
        _raise_if_od_fast_fail(run_root)

    return result


# ---------------------------------------------------------------------------
# finalize — resolve + emit (post-agent glue)
# ---------------------------------------------------------------------------

def finalize(
    run_root: str,
    *,
    target_dir: str | None = None,
) -> str:
    """Run resolve() + emit() after the ReAct agent finishes.

    Args:
        run_root: absolute path to the run dir.
        target_dir: optional override. Defaults to
            ``<run_root>/sysid_inputs`` (see pipeline_artifacts).

    Returns:
        Absolute path to the materialised episode folder.
    """
    # resolve mutates state.json with resolved_objects etc.
    _resolve(run_root)

    if target_dir is None:
        # Emits flat; episode_bundle_dir keeps returning a legacy
        # <episode_id>_agent/ when a previous run of this dir made one, so a
        # re-emit patches that bundle rather than orphaning it beside a new one.
        from ar2s.pipeline_artifacts import episode_bundle_dir
        target_dir = str(episode_bundle_dir(Path(run_root).expanduser().resolve()))

    return write_episode_folder(run_root, target_dir)


__all__ = ["data_ingest", "run_visual", "finalize"]
