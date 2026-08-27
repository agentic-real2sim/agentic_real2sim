"""Stage 3 - Convergence Assessment Agent (one-shot LLM call).

Given the completed hill-climbing search results (reward curve, best alpha,
motion comparison metrics), this agent makes a single LLM call to assess
whether the search converged and how well the recovered physics reproduces
the reference motion.

No tool calls, no internal retry - pure structured generation.
The prompt is loaded from convergence_prompt.md and cached via Anthropic
prompt caching so repeated calls in a run only pay for the input once.

Public API
----------
    from ar2s.agents_g1sysid.agents.sysid_loop.convergence_agent import assess_convergence
    verdict = assess_convergence(
        baseline_reward=0.31,
        oracle_reward=0.87,
        recovered_reward=0.79,
        n_iters=200,
        reward_curve=[...],
        alpha_best=np.ones(3),
        motion_metrics={...},
    )
    # verdict.outcome in ("converged", "max_iters", "give_up")
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from ar2s.agent_configs import chat_model_for, parse_text

_PROMPT_PATH = Path(__file__).parent / "convergence_prompt.md"


@dataclass
class ConvergenceVerdict:
    outcome: str               # "converged" | "max_iters" | "give_up"
    recovery_quality: str      # "excellent" | "good" | "partial" | "poor"
    reasoning: str
    suggestions: str | None = None
    raw_response: str = ""
    parse_error: str | None = None


def _load_system_prompt() -> SystemMessage:
    text = _PROMPT_PATH.read_text()
    return SystemMessage(content=[{
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }])


def _parse_response(text: str) -> ConvergenceVerdict:
    """Extract the JSON object from the LLM response."""
    # Strip markdown fences if present
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return ConvergenceVerdict(
            outcome="max_iters",
            recovery_quality="poor",
            reasoning="(parse failed)",
            raw_response=text,
            parse_error="No JSON object found in response",
        )
    try:
        data = json.loads(match.group())
        return ConvergenceVerdict(
            outcome=data.get("outcome", "max_iters"),
            recovery_quality=data.get("recovery_quality", "poor"),
            reasoning=data.get("reasoning", ""),
            suggestions=data.get("suggestions"),
            raw_response=text,
        )
    except json.JSONDecodeError as e:
        return ConvergenceVerdict(
            outcome="max_iters",
            recovery_quality="poor",
            reasoning="(parse failed)",
            raw_response=text,
            parse_error=str(e),
        )


def assess_convergence(
    baseline_reward: float,
    oracle_reward: float,
    recovered_reward: float,
    n_iters: int,
    reward_curve: list[float],
    alpha_best: np.ndarray,
    motion_metrics: dict,
) -> ConvergenceVerdict:
    """One-shot LLM assessment of whether the sysID search converged.

    Args:
        baseline_reward:  reward with broken physics, alpha = 1
        oracle_reward:    reward with true oracle correction
        recovered_reward: best reward found by hill-climbing
        n_iters:          total iterations run
        reward_curve:     list of best-so-far fitness values
        alpha_best:       best correction vector (numpy)
        motion_metrics:   dict from compare_trajectory_metrics()

    Returns:
        ConvergenceVerdict with outcome + recovery_quality + reasoning.
    """
    llm = chat_model_for("g1sysid.subagents.convergence_agent")
    if llm is None:
        return ConvergenceVerdict(
            outcome="max_iters",
            recovery_quality="poor",
            reasoning="No credential for the configured convergence model provider.",
        )

    # Summarise the reward curve tail (last 30%)
    tail_start = max(0, int(len(reward_curve) * 0.70))
    tail = reward_curve[tail_start:]
    tail_improvement = (max(tail) - tail[0]) if tail else 0.0

    user_content = json.dumps({
        "baseline_reward": round(float(baseline_reward), 4),
        "oracle_reward": round(float(oracle_reward), 4),
        "recovered_reward": round(float(recovered_reward), 4),
        "recovery_fraction": round(float(recovered_reward) / max(float(oracle_reward), 1e-9), 3),
        "n_iters": n_iters,
        "reward_curve_summary": {
            "first": round(float(reward_curve[0]), 4) if reward_curve else None,
            "best": round(float(max(reward_curve)), 4) if reward_curve else None,
            "last": round(float(reward_curve[-1]), 4) if reward_curve else None,
            "tail_improvement": round(float(tail_improvement), 4),
        },
        "alpha_best_summary": {
            "mean": round(float(np.mean(alpha_best)), 4),
            "min": round(float(np.min(alpha_best)), 4),
            "max": round(float(np.max(alpha_best)), 4),
            "values": [round(float(v), 4) for v in alpha_best.flatten().tolist()[:10]],
        },
        "motion_metrics": {
            k: round(float(v), 4)
            for k, v in motion_metrics.items()
            if isinstance(v, (int, float))
        },
    }, indent=2)

    system_msg = _load_system_prompt()
    human_msg = HumanMessage(content=user_content)

    try:
        response = llm.invoke([system_msg, human_msg])
        text = parse_text(response)
    except Exception as e:
        return ConvergenceVerdict(
            outcome="max_iters",
            recovery_quality="poor",
            reasoning=f"LLM call failed: {e}",
            parse_error=str(e),
        )

    return _parse_response(text)


__all__ = ["ConvergenceVerdict", "assess_convergence"]
