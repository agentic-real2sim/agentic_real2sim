"""ReAct orchestrator for the physical-prior stage.

The LLM is only a fixed-order tool caller. Material classification remains in
``material_classify.classify_object()`` via the deterministic tool functions in
``apply.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ar2s.agent_configs.models import chat_model_for
from ar2s.agents_physical_prior import apply as physical_apply
from ar2s.agents_physical_prior.apply import ApplyReport, PhysicalPriorContext


_PROMPT_PATH = Path(__file__).with_name("orchestrator_prompt.md")
_DEFAULT_MAX_ITERATIONS = 14
_TOOL_ORDER = [
    "load_manifest",
    "classify_materials",
    "compute_masses",
    "write_physical_priors",
    "patch_manifest",
]


@dataclass(frozen=True)
class PhysicalPriorInput:
    episode_dir: str
    run_root: str


@dataclass
class _PhysicalPriorSink:
    context: PhysicalPriorContext | None = None
    mass_by_name: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def data_ingest(episode_dir: str | Path, run_root: str | Path) -> PhysicalPriorInput:
    """Validate the physical-prior inputs and return absolute paths."""
    episode_dir = Path(episode_dir).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve()
    assert episode_dir.is_dir(), f"episode_dir does not exist: {episode_dir}"
    assert (episode_dir / "manifest.yaml").is_file(), (
        f"manifest.yaml not found at {episode_dir / 'manifest.yaml'}"
    )
    assert (run_root / "state.json").is_file(), (
        f"state.json not found at {run_root / 'state.json'}"
    )
    return PhysicalPriorInput(episode_dir=str(episode_dir), run_root=str(run_root))


def _system_message_with_caching() -> SystemMessage:
    return SystemMessage(content=[{
        "type": "text",
        "text": _PROMPT_PATH.read_text(),
        "cache_control": {"type": "ephemeral"},
    }])


def _build_tools(stage_input: PhysicalPriorInput, sink: _PhysicalPriorSink) -> list:
    @tool
    def load_manifest() -> str:
        """Load manifest.yaml plus visual state.json. Run this first."""
        try:
            sink.context = physical_apply.load_physical_prior_context(
                stage_input.episode_dir,
                stage_input.run_root,
            )
        except Exception as exc:  # noqa: BLE001 - return a structured tool halt
            sink.errors.append(f"load_manifest: {type(exc).__name__}: {exc}")
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return json.dumps({
            "ok": True,
            "episode_dir": stage_input.episode_dir,
            "run_root": stage_input.run_root,
            "n_candidate_objects": len(sink.context.candidates),
        })

    @tool
    def classify_materials() -> str:
        """Classify each emitted non-robot object's material. Run second."""
        if sink.context is None:
            sink.errors.append("classify_materials: load_manifest has not run")
            return json.dumps({"ok": False, "error": "load_manifest has not run"})
        try:
            report = physical_apply.classify_materials(sink.context)
        except Exception as exc:  # noqa: BLE001
            sink.errors.append(f"classify_materials: {type(exc).__name__}: {exc}")
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return json.dumps({
            "ok": True,
            "classified": len(report.classified),
            "skipped": len(report.skipped),
        })

    @tool
    def compute_masses() -> str:
        """Compute masses from material density, raw mesh volume, and scale."""
        if sink.context is None:
            sink.errors.append("compute_masses: load_manifest has not run")
            return json.dumps({"ok": False, "error": "load_manifest has not run"})
        try:
            sink.mass_by_name = physical_apply.compute_masses(sink.context)
        except Exception as exc:  # noqa: BLE001
            sink.errors.append(f"compute_masses: {type(exc).__name__}: {exc}")
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        failed = [
            p.name for p in sink.context.report.classified
            if p.mass_kg is None
        ]
        return json.dumps({
            "ok": True,
            "masses_computed": len(sink.mass_by_name),
            "mass_failed_for": failed,
        })

    @tool
    def write_physical_priors() -> str:
        """Write physical_priors.json. Run after compute_masses."""
        if sink.context is None:
            sink.errors.append("write_physical_priors: load_manifest has not run")
            return json.dumps({"ok": False, "error": "load_manifest has not run"})
        try:
            path = physical_apply.write_physical_priors(sink.context.report)
        except Exception as exc:  # noqa: BLE001
            sink.errors.append(f"write_physical_priors: {type(exc).__name__}: {exc}")
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return json.dumps({"ok": True, "physical_priors_path": str(path)})

    @tool
    def patch_manifest() -> str:
        """Patch manifest.yaml object masses. Run last."""
        if sink.context is None:
            sink.errors.append("patch_manifest: load_manifest has not run")
            return json.dumps({"ok": False, "error": "load_manifest has not run"})
        try:
            patched = physical_apply.patch_manifest_masses(
                sink.context,
                sink.mass_by_name,
            )
        except Exception as exc:  # noqa: BLE001
            sink.errors.append(f"patch_manifest: {type(exc).__name__}: {exc}")
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return json.dumps({
            "ok": True,
            "manifest_patched_for": patched,
            "n_patched": len(patched),
        })

    return [
        load_manifest,
        classify_materials,
        compute_masses,
        write_physical_priors,
        patch_manifest,
    ]


def run_physical_prior(
    stage_input: PhysicalPriorInput,
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> ApplyReport:
    """Run physical-prior tools under a fixed-order ReAct orchestrator."""
    sink = _PhysicalPriorSink()
    llm = chat_model_for("physical_prior.main.orchestrator")
    if llm is None:
        raise RuntimeError(
            "No API key for the configured physical_prior.main.orchestrator model. "
            "Set the credential env var for the provider selected in the active "
            "agent config."
        )

    agent = create_react_agent(
        model=llm,
        tools=_build_tools(stage_input, sink),
        prompt=_system_message_with_caching(),
    )

    initial_msg = (
        f"episode_dir = {stage_input.episode_dir}\n"
        f"run_root = {stage_input.run_root}\n\n"
        f"Call these tools exactly once each, in this order: {_TOOL_ORDER}. "
        f"If any tool returns ok=false, stop and reply with the terminal halt line."
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=initial_msg)]},
        config={"recursion_limit": max_iterations},
    )
    agent_final = _last_message_text(result)
    if sink.errors:
        raise RuntimeError(
            "physical-prior ReAct agent halted: "
            + "; ".join(sink.errors)
            + (f"; final={agent_final}" if agent_final else "")
        )
    assert sink.context is not None, "physical-prior agent did not call load_manifest"
    assert sink.context.report.physical_priors_path is not None, (
        "physical-prior agent did not write physical_priors.json"
    )
    return sink.context.report


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


__all__ = [
    "PhysicalPriorInput",
    "data_ingest",
    "run_physical_prior",
]
