"""LangChain tool factory for the phystwin Stage 3 sysid agent.

The agent's job is dispatch + skip decisions; the heavy lifting is done in
``ar2s.agents_sysid.phystwin.sysid`` (subprocess wrappers around the phystwin
optimize_cma / train_warp / inference_warp scripts).

Tools are closure-wrapped so they share state (episode_dir, case_name, etc.)
without making the agent pass these around.

Tool set (5 total):
  - read_setup_yaml()            → JSON of phystwin_setup.yaml key fields
  - run_cma(max_iter, force)     → invoke Stage 3.1, returns ckpt path or skip msg
  - run_train(force)             → invoke Stage 3.2 (requires CMA done)
  - run_inference()              → invoke Stage 3.3 (requires train done)
  - submit_sysid_done(summary)   → terminator; agent calls this to exit cleanly
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.tools import tool

from ar2s.agents_sysid.phystwin.sysid import (
    PhystwinCmaResult,
    PhystwinInferResult,
    PhystwinTrainResult,
    run_cma,
    run_infer,
    run_train,
)


@dataclass
class PhystwinSysidResult:
    """Structured outcome of Stage 3 (phystwin sysid)."""
    case_name: str
    cma_result: PhystwinCmaResult | None = None    # None if skipped
    train_result: PhystwinTrainResult | None = None
    infer_result: PhystwinInferResult | None = None
    agent_summary: str = ""                         # last words from submit_sysid_done
    n_submit_attempts: int = 0                      # how many times agent tried to finish


@dataclass
class _SysidSlot:
    """Shared state across the tool calls. submit_sysid_done writes here."""
    result: PhystwinSysidResult | None = None
    submit_attempts: int = 0


# -------------------------------------------------------------------
# Tool factory
# -------------------------------------------------------------------

def make_phystwin_sysid_tools(
    episode_dir: Path,
    raw_dir: Path,
    case_name: str,
    train_frame: int,
    config_choice: str,
    log_dir: Path,
    slot: _SysidSlot,
    *,
    device: str = "cuda:0",
) -> list:
    """Build the 5 closure-wrapped tools the phystwin sysid agent uses.

    Args:
        episode_dir: contains phystwin_manifest.yaml + phystwin_setup.yaml.
        raw_dir: source phystwin data folder (the agent doesn't need to know,
            but the underlying wrappers need this).
        case_name / train_frame / config_choice: from Stage 1 manifest.
        log_dir: where per-stage logs go (phystwin_cma.{stdout,stderr,...}).
        slot: SysidSlot the agent writes results into.
        device: cuda device passed through to subprocesses.
    """

    setup_yaml_path = episode_dir / "phystwin_setup.yaml"

    @tool
    def read_setup_yaml() -> str:
        """Read phystwin_setup.yaml. Returns key fields as JSON.

        Use this FIRST to learn:
          - sim_initialization (data_loaded, springs_built, num_total_springs)
          - sanity_check (passed, issues)
          - cache_state (cma_done, train_done) — to decide which stages to skip
          - data_shape (frame_len etc.)
        """
        if not setup_yaml_path.exists():
            return f"ERROR: {setup_yaml_path} does not exist. Was Stage 2 run?"
        try:
            with open(setup_yaml_path) as f:
                doc = yaml.safe_load(f)
        except Exception as e:
            return f"ERROR loading {setup_yaml_path}: {type(e).__name__}: {e}"

        # Return only the agent-relevant fields, not the full dump.
        slim = {
            "case_name": doc.get("case_name"),
            "config_choice": doc.get("config_choice"),
            "frame_num": doc.get("frame_num"),
            "fps": doc.get("fps"),
            "n_cams": doc.get("n_cams"),
            "sim_initialization": doc.get("sim_initialization", {}),
            "sanity_check": doc.get("sanity_check", {}),
            "cache_state": doc.get("cache_state", {}),
            "data_shape": doc.get("data_shape", {}),
        }
        return json.dumps(slim, indent=2, default=str)

    @tool
    def run_cma_tool(max_iter: int = 20, force: bool = False) -> str:
        """Stage 3.1: CMA-ES sparse parameter optimization.

        If cache_state.cma_done=true and force=False, you should skip this
        and NOT call this tool. Calling it anyway will re-run and overwrite
        the cached optimal_params.pkl.

        Args:
            max_iter: CMA generations. Default 20 (good for most cases).
                Each generation runs 11 forward sims, so cost scales linearly.
            force: if True, run even when cache exists.
        """
        try:
            result = run_cma(
                base_path=str(raw_dir.parent),
                case_name=case_name,
                train_frame=train_frame,
                max_iter=max_iter,
                config_choice=config_choice,
                log_dir=str(log_dir),
                device=device,
            )
        except Exception as e:
            return f"ERROR: run_cma failed: {type(e).__name__}: {e}"

        slot.result_cma = result  # transient cache; not yet sealed into slot.result
        return (
            f"OK: CMA finished. optimal_params.pkl at "
            f"{result.optimal_params_path.relative_to(result.optimal_params_path.parents[2])}, "
            f"config={config_choice}, max_iter={max_iter}."
        )

    @tool
    def run_train_tool(force: bool = False) -> str:
        """Stage 3.2: Warp differentiable training.

        Requires CMA's optimal_params.pkl. If cache_state.train_done=true
        and force=False, skip.
        """
        try:
            result = run_train(
                base_path=str(raw_dir.parent),
                case_name=case_name,
                train_frame=train_frame,
                config_choice=config_choice,
                log_dir=str(log_dir),
                device=device,
            )
        except FileNotFoundError as e:
            return f"ERROR: run_train precondition unmet: {e}"
        except Exception as e:
            return f"ERROR: run_train failed: {type(e).__name__}: {e}"

        slot.result_train = result
        return f"OK: train_warp finished. best_ckpt at {result.best_ckpt_path.name}."

    @tool
    def run_inference_tool() -> str:
        """Stage 3.3: forward-only rollout with the trained checkpoint.

        Requires train to have produced best_*.pth.
        """
        try:
            result = run_infer(
                base_path=str(raw_dir.parent),
                case_name=case_name,
                config_choice=config_choice,
                log_dir=str(log_dir),
                device=device,
            )
        except FileNotFoundError as e:
            return f"ERROR: run_inference precondition unmet: {e}"
        except Exception as e:
            return f"ERROR: run_inference failed: {type(e).__name__}: {e}"

        slot.result_infer = result
        return f"OK: inference finished. output_dir={result.output_dir}."

    @tool
    def submit_sysid_done(summary: str) -> str:
        """Signal that you are done with Stage 3.

        Call this AFTER you've run (or deliberately skipped) the relevant
        sub-stages. Provide a 1-2 sentence summary of what you did and the
        final state — e.g.:
          "Both CMA and train were cached, so I skipped them; ran inference
          to produce the final rollout video."

        Calling this terminates the agent.
        """
        slot.submit_attempts += 1
        slot.result = PhystwinSysidResult(
            case_name=case_name,
            cma_result=getattr(slot, "result_cma", None),
            train_result=getattr(slot, "result_train", None),
            infer_result=getattr(slot, "result_infer", None),
            agent_summary=str(summary),
            n_submit_attempts=slot.submit_attempts,
        )
        return f"OK: Stage 3 sysid done. You may end the turn now."

    return [
        read_setup_yaml,
        run_cma_tool,
        run_train_tool,
        run_inference_tool,
        submit_sysid_done,
    ]


__all__ = [
    "PhystwinSysidResult",
    "_SysidSlot",
    "make_phystwin_sysid_tools",
]
