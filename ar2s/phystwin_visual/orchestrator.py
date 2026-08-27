"""Top-level ReAct orchestrator for the PhysTwin visual pipeline.

Mirrors ``ar2s.agents_visual.orchestrator`` / ``ar2s.agents_sysid.orchestrator``
so the pipelines share a shape: a ``data_ingest`` validation step (plain
Python), then a ReAct agent (``run_phystwin``) that drives the deterministic
``ar2s.phystwin_visual`` stage tools in dependency order.

Two exported callables — the CLI runs them in this order:

  data_ingest(inp: PhysTwinInput) -> PhysTwinInput
      Validate base_path/case_dir, copy the shape-prior mesh into place if
      shape_prior=True. NO LLM, plain Python. Raises on a bad input.

  run_phystwin(inp, *, max_iterations) -> PhysTwinPipelineResult
      ReAct agent over the phystwin_visual stage tools (video_segmentation
      -> dense_tracking/pcd_projection -> mask_cleanup -> track_processing
      -> [shape_alignment] -> final_export). The dependency graph lives in
      ``orchestrator_prompt.md``.

There is intentionally NO ``finalize`` step: ``final_export`` IS the terminal
deliverable (final_data.pkl + split.json), mirroring process_data.py.

Design notes
------------
* **No LLM decisions today.** The agent calls the stage tools in the fixed
  order named in the initial message (with ``run_shape_alignment`` included
  only when ``shape_prior=True``); the value is the extensible framework.
* **Disk is the source of truth.** Inter-stage data flows through
  ``<base_path>/<case_name>/`` (mask/, cotracker/, pcd/, ...). The only
  orchestration state is an in-memory ``_StageSink`` the tools write their
  typed Report into so ``run_phystwin`` can return a structured result.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ar2s.agent_configs.models import chat_model_for
from ar2s.phystwin_visual.inputs import PhysTwinInput
from ar2s.phystwin_visual import (
    video_segmentation,
    dense_tracking,
    pcd_projection,
    mask_cleanup,
    track_processing,
    shape_alignment,
    final_export,
)
from ar2s.phystwin_visual.video_segmentation import VideoSegmentationReport
from ar2s.phystwin_visual.dense_tracking import DenseTrackingReport
from ar2s.phystwin_visual.pcd_projection import PcdProjectionReport
from ar2s.phystwin_visual.mask_cleanup import MaskCleanupReport
from ar2s.phystwin_visual.track_processing import TrackProcessingReport
from ar2s.phystwin_visual.shape_alignment import ShapeAlignmentReport
from ar2s.phystwin_visual.final_export import FinalExportReport


_PROMPT_PATH = Path(__file__).with_name("orchestrator_prompt.md")
_RUN_ROOT_TEMPLATE = "runs/{pipeline_id}_phystwin"
# 7 candidate tools * ~2 trips = 14 baseline; 24 gives headroom.
_DEFAULT_MAX_ITERATIONS = 24
_TERMINAL_PREFIXES = (
    "PhysTwin visual pipeline complete:",
    "PhysTwin visual pipeline halted at",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class _StageSink:
    """In-memory capture of each stage's typed Report (orchestration only)."""
    video_segmentation: VideoSegmentationReport | None = None
    dense_tracking: DenseTrackingReport | None = None
    pcd_projection: PcdProjectionReport | None = None
    mask_cleanup: MaskCleanupReport | None = None
    track_processing: TrackProcessingReport | None = None
    shape_alignment: ShapeAlignmentReport | None = None
    final_export: FinalExportReport | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class PhysTwinPipelineResult:
    ok: bool
    sink: _StageSink
    agent_final: str = ""
    final_data_path: str = ""


# ---------------------------------------------------------------------------
# data_ingest — plain Python, no LLM
# ---------------------------------------------------------------------------

def data_ingest(inp: PhysTwinInput) -> PhysTwinInput:
    """Validate paths and seed the shape-prior mesh. Returns a resolved copy of ``inp``.

    Mirrors the pre-refactor phystwin_orchestrator's data_ingest node.

    Raises:
        ValueError on a missing base_path/case_dir or a bad shape-prior config.
    """
    errors: list[str] = []
    base_path = Path(inp.base_path)
    if not base_path.exists():
        errors.append(f"base_path not found: {base_path}")
    case_path = base_path / inp.case_name
    if not case_path.exists():
        errors.append(f"case_dir not found: {case_path}")
    if inp.shape_prior and not inp.shape_mesh_path:
        errors.append("shape_prior=True but shape_mesh_path is empty")
    if inp.shape_prior and inp.shape_mesh_path and not Path(inp.shape_mesh_path).exists():
        errors.append(f"shape_mesh_path not found: {inp.shape_mesh_path}")
    if errors:
        raise ValueError("PhysTwin data_ingest: " + "; ".join(errors))

    resolved = replace(inp, base_path=str(base_path.resolve()))

    # If shape_prior, copy the mesh to case_dir/shape/object.glb (mirrors process_data.py)
    if resolved.shape_prior and resolved.shape_mesh_path:
        shape_dir = case_path / "shape"
        shape_dir.mkdir(parents=True, exist_ok=True)
        target = shape_dir / "object.glb"
        src = Path(resolved.shape_mesh_path).resolve()
        if src != target.resolve():
            shutil.copy2(src, target)
        matching_dir = shape_dir / "matching"
        if matching_dir.exists():
            shutil.rmtree(matching_dir)

    return resolved


# ---------------------------------------------------------------------------
# Stage tools (factory — closes over inp + sink + log_dir)
# ---------------------------------------------------------------------------

def _write_split_json(inp: PhysTwinInput) -> None:
    """Write split.json (mirrors process_data.py / the pre-refactor orchestrator)."""
    case_path = Path(inp.base_path) / inp.case_name
    pcd_dir = case_path / "pcd"
    frame_len = len(list(pcd_dir.glob("*.npz"))) if pcd_dir.exists() else 0
    if frame_len > 0:
        split = {
            "frame_len": frame_len,
            "train": [0, int(frame_len * 0.7)],
            "test": [int(frame_len * 0.7), frame_len],
        }
        (case_path / "split.json").write_text(json.dumps(split))


def _build_tools(inp: PhysTwinInput, sink: _StageSink, log_dir: str) -> list:
    """Build the LangChain @tool set for this pipeline run.

    Each tool takes no arguments — base_path/case_name/per-stage config come
    from ``inp``. Each tool runs the underlying phystwin_visual stage,
    records the typed Report into ``sink``, and returns a compact JSON
    string for the agent.
    """
    text_prompt = f"{inp.category}.{inp.controller_name}"

    @tool
    def run_video_segmentation() -> str:
        """Stage 1: multi-camera Grounded-SAM-2 segmentation. Writes mask/.
        Run FIRST."""
        report = video_segmentation.run(
            video_segmentation.VideoSegmentationInput(
                base_path=inp.base_path,
                case_name=inp.case_name,
                text_prompt=text_prompt,
                camera_indices=inp.camera_indices,
                box_threshold=inp.box_threshold,
                text_threshold=inp.text_threshold,
            ),
            log_dir=log_dir,
        )
        sink.video_segmentation = report
        if not report.success:
            sink.errors.append(f"video_segmentation: {report.error}")
        return json.dumps({"ok": report.success, "error": report.error, "mask_dir": report.mask_dir})

    @tool
    def run_dense_tracking() -> str:
        """Stage 2: dense 2D tracking with CoTracker. Writes cotracker/.
        Run AFTER run_video_segmentation."""
        report = dense_tracking.run(
            dense_tracking.DenseTrackingInput(
                base_path=inp.base_path,
                case_name=inp.case_name,
                max_query_points=inp.cotracker_query_points,
            ),
            log_dir=log_dir,
        )
        sink.dense_tracking = report
        if not report.success:
            sink.errors.append(f"dense_tracking: {report.error}")
        return json.dumps({"ok": report.success, "error": report.error, "cotracker_dir": report.cotracker_dir})

    @tool
    def run_pcd_projection() -> str:
        """Stage 3: RGB-D lifting to world-space point clouds. Writes pcd/.
        Run AFTER run_video_segmentation (reads metadata.json/calibrate.pkl;
        independent of run_dense_tracking)."""
        report = pcd_projection.run(
            pcd_projection.PcdProjectionInput(base_path=inp.base_path, case_name=inp.case_name),
            log_dir=log_dir,
        )
        sink.pcd_projection = report
        if not report.success:
            sink.errors.append(f"pcd_projection: {report.error}")
        return json.dumps({"ok": report.success, "error": report.error, "pcd_dir": report.pcd_dir})

    @tool
    def run_mask_cleanup() -> str:
        """Stage 4: depth + semantic + outlier mask filtering. Writes
        mask/processed_masks.pkl. Run AFTER run_pcd_projection."""
        report = mask_cleanup.run(
            mask_cleanup.MaskCleanupInput(
                base_path=inp.base_path,
                case_name=inp.case_name,
                controller_name=inp.controller_name,
            ),
            log_dir=log_dir,
        )
        sink.mask_cleanup = report
        if not report.success:
            sink.errors.append(f"mask_cleanup: {report.error}")
        return json.dumps({
            "ok": report.success, "error": report.error,
            "processed_masks_path": report.processed_masks_path,
        })

    @tool
    def run_track_processing() -> str:
        """Stage 5: 3D track filtering and packaging. Writes
        track_process_data.pkl. Run AFTER run_dense_tracking and
        run_mask_cleanup."""
        report = track_processing.run(
            track_processing.TrackProcessingInput(base_path=inp.base_path, case_name=inp.case_name),
            log_dir=log_dir,
        )
        sink.track_processing = report
        if not report.success:
            sink.errors.append(f"track_processing: {report.error}")
        return json.dumps({"ok": report.success, "error": report.error, "track_data_path": report.track_data_path})

    @tool
    def run_shape_alignment() -> str:
        """Stage 6 (optional): align the shape-prior mesh (shape/object.glb)
        to the observed point cloud. Writes shape/matching/final_mesh.glb and
        mesh_transform.npz. Run AFTER run_track_processing, only if
        shape_prior=True."""
        report = shape_alignment.run(
            shape_alignment.ShapeAlignmentInput(
                base_path=inp.base_path,
                case_name=inp.case_name,
                controller_name=inp.controller_name,
            ),
            log_dir=log_dir,
        )
        sink.shape_alignment = report
        if not report.success:
            sink.errors.append(f"shape_alignment: {report.error}")
        return json.dumps({
            "ok": report.success, "error": report.error,
            "aligned_mesh_path": report.aligned_mesh_path,
        })

    @tool
    def run_final_export() -> str:
        """Stage 7: final data export — volume-sampled track data, train/test
        split, and preview videos. Writes final_data.pkl + split.json.
        Run LAST."""
        report = final_export.run(
            final_export.FinalExportInput(
                base_path=inp.base_path,
                case_name=inp.case_name,
                shape_prior=inp.shape_prior,
                num_surface_points=inp.num_surface_points,
                volume_sample_size=inp.volume_sample_size,
            ),
            log_dir=log_dir,
        )
        sink.final_export = report
        if report.success:
            _write_split_json(inp)
        else:
            sink.errors.append(f"final_export: {report.error}")
        return json.dumps({"ok": report.success, "error": report.error, "final_data_path": report.final_data_path})

    tools = [
        run_video_segmentation, run_dense_tracking, run_pcd_projection,
        run_mask_cleanup, run_track_processing,
    ]
    if inp.shape_prior:
        tools.append(run_shape_alignment)
    tools.append(run_final_export)
    return tools


def _plan(inp: PhysTwinInput) -> list[str]:
    """Ordered list of tool names the agent should call."""
    plan = [
        "run_video_segmentation", "run_dense_tracking", "run_pcd_projection",
        "run_mask_cleanup", "run_track_processing",
    ]
    if inp.shape_prior:
        plan.append("run_shape_alignment")
    plan.append("run_final_export")
    return plan


# ---------------------------------------------------------------------------
# run_phystwin — top-level ReAct agent
# ---------------------------------------------------------------------------

def _system_message_with_caching() -> SystemMessage:
    """SystemMessage with Anthropic prompt caching on the static body.

    Non-Anthropic providers ignore the ``cache_control`` field harmlessly.
    """
    return SystemMessage(content=[{
        "type": "text",
        "text": _PROMPT_PATH.read_text(),
        "cache_control": {"type": "ephemeral"},
    }])


def _is_terminal(text: str) -> bool:
    """True if ``text`` is one of the terminal lines from orchestrator_prompt.md."""
    stripped = text.strip()
    return any(stripped.startswith(p) for p in _TERMINAL_PREFIXES)


def _remaining_tools(sink: _StageSink, plan: list[str]) -> list[str]:
    """Tools in ``plan`` whose Report hasn't landed in ``sink`` yet, in order."""
    return [t for t in plan if getattr(sink, t.removeprefix("run_")) is None]


def _nudge_message(remaining: list[str]) -> str:
    return (
        f"That was a text-only reply, but the plan isn't finished. "
        f"Remaining tools, in order: {remaining}. Call `{remaining[0]}` now, "
        f"then keep going through the rest without stopping for narration."
    )


def run_phystwin(
    inp: PhysTwinInput,
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> PhysTwinPipelineResult:
    """Run the PhysTwin visual pipeline under a ReAct orchestrator.

    Args:
        inp: validated PhysTwinInput, e.g. the return value of ``data_ingest``.
        max_iterations: LangGraph recursion_limit for the ReAct loop.

    Returns:
        PhysTwinPipelineResult — ``ok`` reflects whether final_export
        succeeded and produced final_data.pkl.

    Raises:
        RuntimeError if no LLM is configured for phystwin_visual.main.orchestrator.
    """
    sink = _StageSink()
    plan = _plan(inp)
    pipeline_id = inp.pipeline_id or inp.case_name
    log_dir = str(Path(_RUN_ROOT_TEMPLATE.format(pipeline_id=pipeline_id)).resolve() / "logs")

    llm = chat_model_for("phystwin_visual.main.orchestrator")
    if llm is None:
        raise RuntimeError(
            "No API key for the configured phystwin_visual.main.orchestrator model. "
            "Set the credential env var for the provider selected in "
            "ar2s/agent_configs/config_default.yaml or pass --agent-config "
            "to choose another YAML config."
        )

    agent = create_react_agent(
        model=llm,
        tools=_build_tools(inp, sink, log_dir),
        prompt=_system_message_with_caching(),
    )

    initial_msg = (
        f"base_path = {inp.base_path}\n"
        f"case_name = {inp.case_name}\n\n"
        f"Call these tools exactly once each, in this order: {plan}. "
        f"After the last tool returns ok=true, stop and reply with the "
        f"terminal success line. If any tool returns ok=false, stop and "
        f"reply with the terminal halt line."
    )

    # Worst case: one tool advances per invoke (model stalls on every tool),
    # so allow up to len(plan) nudges to cover the whole plan.
    max_nudges = len(plan)
    messages: list = [HumanMessage(content=initial_msg)]
    agent_final = ""
    for attempt in range(max_nudges + 1):
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": max_iterations},
        )
        messages = result["messages"]
        agent_final = _last_message_text(result)
        if _is_terminal(agent_final):
            break
        remaining = _remaining_tools(sink, plan)
        if not remaining or attempt == max_nudges:
            break
        messages = messages + [HumanMessage(content=_nudge_message(remaining))]

    final_data_path = sink.final_export.final_data_path if sink.final_export else ""
    ok = bool(sink.final_export and sink.final_export.success) and not sink.errors
    return PhysTwinPipelineResult(ok=ok, sink=sink, agent_final=agent_final, final_data_path=final_data_path)


def _last_message_text(agent_result: dict) -> str:
    msgs = agent_result.get("messages") or []
    if not msgs:
        return ""
    content = getattr(msgs[-1], "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return content


__all__ = ["PhysTwinPipelineResult", "data_ingest", "run_phystwin"]
