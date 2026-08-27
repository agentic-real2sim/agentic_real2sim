"""tracking_critic subagent — per-object VLM verdict on FoundationPose drift.

Uses the disk-backed visual-agent state and ``@tool(run_root: str)``
signature. Voting and sampling behavior are preserved from the original
implementation.

For each per-object track_vis directory written by pose_tracking, sample
N=4 evenly-spaced overlay PNGs and ask YAML-configured VLMs to vote
on whether the rendered 6-DoF pose follows the real object across those
frames. Majority-accept on each object.

Reads from state.json:
  - stages.pose_tracking.outputs.per_object  (each has track_vis_dir,
                                              error if failed)

Writes to state.json:
  - state.tracking_critic_verdict = {
        accept, reason, detail: {per_object: {...}, rejected: [...]}
    }
  - history += {"agent": "tracking_critic", "note": "..."}

Returns small JSON: {ok, verdict, n_rejected, error?}.

Design note:
  - NOT a ReAct mini-agent — single fan-out vote per object.
  - WARNING-ONLY in v0 (matches v1 — deferred init_frame retry loop).
    The top-level orchestrator sees verdict.accept and may choose to act,
    but tracking_critic itself does NOT re-invoke pose_tracking.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import vlm_call
from ar2s.agent_configs import image_to_data_url
from ar2s.agents_visual.state import append_history, load_state, save_state


_PROMPT_PATH = Path(__file__).with_name("tracking_critic_prompt.md")
# Capped at gateway Mistral-Small's image limit (per v1 ablation note).
_N_KEYFRAMES = 4


class _TrackVerdict(BaseModel):
    accept: bool = Field(description="True iff pose tracks the object across the sampled frames.")
    poor_frames: list[int] = Field(
        default_factory=list,
        description="1-based positions of frames showing drift; empty when accept=True.",
    )
    reasoning: str = Field(description="One sentence summary.")


def _evenly_sample(items: list[Path], n: int) -> list[Path]:
    if len(items) <= n:
        return items
    return [items[int(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _record(run_root: str, verdict: dict, note: str) -> str:
    state = load_state(run_root)
    state["tracking_critic_verdict"] = verdict
    save_state(state, run_root)
    append_history(run_root, "tracking_critic", note=note)
    return json.dumps({
        "ok":          True,
        "verdict":     verdict,
        "n_rejected":  len(verdict.get("detail", {}).get("rejected", [])),
    })


@tool
def tracking_critic(run_root: str) -> str:
    """Judge FoundationPose drift per object via VLM majority vote on track_vis frames.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``stages.pose_tracking`` to have completed (success or
            partial-failure both OK — we report per-object).

    Returns:
        JSON string:
          {"ok":         True,
           "verdict":    {"accept": bool, "reason": str, "detail": {
                              "per_object": {<name>: {...}, ...},
                              "rejected":   [<name>, ...]}},
           "n_rejected": int}
        accept=True iff every reviewed object passes majority vote.
        WARNING-ONLY: caller may decide to retry pose_tracking with a
        different init_frame; this subagent does NOT loop.
    """
    state = load_state(run_root)
    pt = state.get("stages", {}).get("pose_tracking", {})
    per_object_in = pt.get("outputs", {}).get("per_object", [])

    if not per_object_in:
        verdict = {"accept": True, "reason": "no pose_tracking outputs", "detail": {}}
        return _record(run_root, verdict, note="no pose_tracking outputs to review")

    per_object_out: dict[str, dict] = {}
    system_prompt = _PROMPT_PATH.read_text()

    for rep in per_object_in:
        name = rep.get("object_name", "?")
        if not rep.get("success") or not rep.get("track_vis_dir"):
            per_object_out[name] = {
                "accept": False,
                "reason": f"tracking failed: {rep.get('error', 'no track_vis_dir')[:80]}",
            }
            continue

        vis_files = sorted(Path(rep["track_vis_dir"]).glob("*.png"))
        if not vis_files:
            per_object_out[name] = {
                "accept": False,
                "reason": f"no track_vis PNGs in {rep['track_vis_dir']}",
            }
            continue

        sampled = _evenly_sample(vis_files, _N_KEYFRAMES)
        images = [image_to_data_url(p) for p in sampled]
        user_text = (
            f"These are {len(sampled)} evenly-spaced track_vis frames for "
            f"object {name!r} (in temporal order). Each frame shows the real "
            f"camera image with the estimated 6-DoF pose rendered as a 3D bbox "
            f"+ XYZ axes. Decide whether the pose tracks the {name} across "
            f"these frames."
        )

        print(f"[tracking_critic] reviewing {name} over {len(sampled)} frames")
        responses = vlm_call(
            "visual.subagents.tracking_critic",
            system=system_prompt,
            user_text=user_text,
            images=images,
            output_schema=_TrackVerdict,
        )

        successes = [r for r in responses.values() if r is not None]
        if not successes:
            # v1 policy: default to accept on total VLM failure.
            per_object_out[name] = {
                "accept": True,
                "reason": "all VLMs failed; defaulting to accept",
            }
            continue

        n_accept = sum(1 for r in successes if r.accept)
        accept = n_accept * 2 >= len(successes)        # majority accept
        per_object_out[name] = {
            "accept":      accept,
            "n_accept":    n_accept,
            "n_responses": len(successes),
            "per_model_reasoning": {
                m: r.reasoning for m, r in responses.items() if r is not None
            },
        }

    rejected = [n for n, info in per_object_out.items() if not info["accept"]]
    overall_accept = not rejected
    if rejected:
        reason = f"{len(rejected)}/{len(per_object_out)} objects show drift: {rejected}"
        print(f"[tracking_critic] reject: {reason}")
    else:
        reason = f"all {len(per_object_out)} objects track cleanly"
        print(f"[tracking_critic] {reason}")

    verdict = {
        "accept": overall_accept,
        "reason": reason,
        "detail": {"per_object": per_object_out, "rejected": rejected},
    }
    return _record(run_root, verdict, note=reason)
