"""Unified Stage 3 sysid agent — supports kind=droid|phystwin.

Complementary to the Stage 1 phystwin episode setup agent:

  - kind="droid"    → forwards to existing grasp_sweep() / grasp_loop()
                       (already complete + has its own internal LLM sub-agents;
                       no point putting an extra LLM dispatcher in front).
  - kind="phystwin" → LangGraph ReAct agent with 5 tools (read_setup_yaml,
                       run_cma, run_train, run_inference, submit_sysid_done).
                       The agent reads cache_state, decides which sub-stages
                       to skip, calls them, then submits done. Decisions are
                       small (skip vs not) but the path is exposed.

Public API:
    from ar2s.agents_sysid.agents.sysid_agent import run_sysid_agent

    # droid (forwards to grasp_sweep)
    result = run_sysid_agent("episodes/marker_in_mug_v0", kind="droid")

    # phystwin (LLM ReAct agent)
    result = run_sysid_agent("episodes/double_stretch_sloth", kind="phystwin")

Returns:
    For kind=droid:   GraspSweepResult | GraspLoopResult (existing types)
    For kind=phystwin: PhystwinSysidResult (new in sysid_tools.py)
"""
from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ar2s.agent_configs.models import chat_model_for
from ar2s.agents_sysid.skills.sysid_tools import (
    PhystwinSysidResult,
    _SysidSlot,
    make_phystwin_sysid_tools,
)


_RECURSION_LIMIT = 16             # phystwin path: ~6 LLM-tool round trips suffice


_PHYSTWIN_SYSID_PROMPT = """\
You are the Stage 3 sysid agent (phystwin kind).

# Goal

Run the phystwin system-identification pipeline:
  1. Stage 3.1 CMA-ES sparse optimization     (`run_cma_tool`)
  2. Stage 3.2 Warp differentiable training   (`run_train_tool`)
  3. Stage 3.3 forward-rollout inference      (`run_inference_tool`)

Each sub-stage writes its outputs to disk. Stage 2 (sim_initialization) has
already run and produced ``phystwin_setup.yaml``, which tells you which
sub-stages are already cached (so you can skip them).

# Workflow

1. FIRST call `read_setup_yaml()` to see the current state, especially
   ``cache_state.cma_done`` and ``cache_state.train_done``.
2. For each sub-stage:
     - If its cache_state field is true, SKIP (do not call the tool).
     - Else call the corresponding tool.
3. After all 3 stages have either run or been skipped, call
   `submit_sysid_done(summary)` with a 1-2 sentence summary.

# Defaults

  - `run_cma_tool` default ``max_iter=20`` — leave it unless instructed.
  - You don't need to pass any args to `run_train_tool` / `run_inference_tool`.
  - Don't use ``force=True`` unless explicitly told to retry.

# Important

  - Sanity_check.passed=false IS NOT a reason to skip. Even with warnings,
    Stage 3 can run. The agent should proceed unless sim_initialization
    itself failed (data_loaded=false). In that case, return early via
    submit_sysid_done explaining the issue — do NOT call run_cma/train/infer.
  - Don't loop: each tool only needs to be called once per session.
  - When CMA succeeds, immediately move to train. When train succeeds, move
    to inference. The order is fixed.
"""


def _run_phystwin_sysid_agent(
    episode_dir: Path,
    *,
    raw_dir: Path,
    case_name: str,
    train_frame: int,
    config_choice: str,
    log_dir: Path,
    recursion_limit: int,
    device: str,
) -> PhystwinSysidResult:
    """Inner: LangGraph ReAct agent for phystwin Stage 3."""
    slot = _SysidSlot()
    tools = make_phystwin_sysid_tools(
        episode_dir=episode_dir,
        raw_dir=raw_dir,
        case_name=case_name,
        train_frame=train_frame,
        config_choice=config_choice,
        log_dir=log_dir,
        slot=slot,
        device=device,
    )

    llm = chat_model_for("sysid.phystwin.main.sysid_agent")
    if llm is None:
        raise RuntimeError(
            "No LLM provider/API key configured. "
            "Set the credential env var for the provider selected in the agent YAML."
        )

    sys_msg = SystemMessage(content=[
        {
            "type": "text",
            "text": _PHYSTWIN_SYSID_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ])
    agent = create_react_agent(model=llm, tools=tools, prompt=sys_msg)

    initial_msg = (
        f"Episode dir: {episode_dir}\n"
        f"Case name: {case_name}\n"
        f"Config choice: {config_choice} (decided by Stage 1)\n"
        f"Train frame: {train_frame}\n\n"
        "Begin by calling read_setup_yaml() to see the current cache state, "
        "then run/skip CMA + train + inference accordingly. End with submit_sysid_done."
    )

    try:
        agent.invoke(
            {"messages": [HumanMessage(content=initial_msg)]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as e:
        if slot.result is not None:
            # Already submitted before the crash — return what we have.
            return slot.result
        raise RuntimeError(
            f"phystwin sysid agent crashed: {type(e).__name__}: {e}"
        ) from e

    if slot.result is None:
        raise RuntimeError(
            f"phystwin sysid agent terminated without calling submit_sysid_done."
        )
    return slot.result


# -------------------------------------------------------------------
# Public entry — unified across kinds
# -------------------------------------------------------------------

def run_sysid_agent(
    episode_dir: str | Path,
    *,
    kind: str = "droid",
    recursion_limit: int = _RECURSION_LIMIT,
    device: str = "cuda:0",
    # droid-specific:
    droid_strategy: str = "sweep",     # "sweep" | "loop" — only used when kind=droid
    droid_kwargs: dict | None = None,  # passthrough to grasp_sweep/grasp_loop
    # phystwin-specific:
    log_dir: str | Path | None = None,  # for phystwin sub-stage subprocess logs
):
    """Run Stage 3 sysid for an already-prepared episode.

    Args:
        episode_dir: dir containing manifest + (for phystwin) phystwin_setup.yaml.
        kind: ``"droid"`` or ``"phystwin"``.
        recursion_limit: ReAct round-trip cap (phystwin only; default 16).
        device: cuda device (phystwin only).
        droid_strategy: ``"sweep"`` (deterministic, fast) or ``"loop"`` (LLM retry).
            Only used when kind="droid". Default "sweep".
        droid_kwargs: extra kwargs forwarded to grasp_sweep/grasp_loop.
        log_dir: subprocess log dir (phystwin only). Default = episode_dir/logs/.

    Returns:
        - kind="droid":    GraspSweepResult or GraspLoopResult
        - kind="phystwin": PhystwinSysidResult
    """
    episode_dir = Path(episode_dir).expanduser().resolve()
    if not episode_dir.is_dir():
        raise ValueError(f"episode_dir {episode_dir} not a directory")

    if kind == "droid":
        # Forward to existing droid Stage 3 — no extra LLM wrapper.
        if droid_strategy == "sweep":
            from ar2s.agents_sysid.grasp_sweep import grasp_sweep
            return grasp_sweep(episode_dir, **(droid_kwargs or {}))
        elif droid_strategy == "loop":
            from ar2s.agents_sysid.grasp_loop import grasp_loop
            return grasp_loop(episode_dir, **(droid_kwargs or {}))
        else:
            raise ValueError(
                f"droid_strategy must be 'sweep' or 'loop', got {droid_strategy!r}"
            )

    elif kind == "phystwin":
        # Load manifest (from Stage 1)
        manifest_path = episode_dir / "phystwin_manifest.yaml"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"phystwin_manifest.yaml not found at {manifest_path}. "
                f"Stage 1 (Episode Agent) must run first."
            )
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        if log_dir is None:
            log_dir = episode_dir / "logs"
        log_dir = Path(log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        return _run_phystwin_sysid_agent(
            episode_dir=episode_dir,
            raw_dir=Path(manifest["raw_dir"]).expanduser().resolve(),
            case_name=manifest["case_name"],
            train_frame=int(manifest["train_frame"]),
            config_choice=manifest["config_choice"],
            log_dir=log_dir,
            recursion_limit=recursion_limit,
            device=device,
        )

    else:
        raise ValueError(f"unknown kind {kind!r}; expected 'droid' or 'phystwin'")


__all__ = ["run_sysid_agent", "PhystwinSysidResult"]
