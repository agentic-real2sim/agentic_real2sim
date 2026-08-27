"""ground_ref subagent — VLM picks which scene object defines the floor.

Uses the disk-backed visual-agent state and ``@tool(run_root: str)``
signature. Voting and tie-break behavior are preserved from the original
ablation-tuned implementation.

Reads from state.json:
  - discovered_objects                          (list[str])
  - pickup_objects                              (set by pickup_objects subagent;
                                                 we exclude them from candidates)
  - stages.svo_extract.outputs.left_frames_dir  (we use frame_000000.png)
  - entry.ground_reference_object               (optional user hint;
                                                 authoritative fallback)

Writes to state.json:
  - ground_reference_object = <chosen name>
  - history += {"agent": "ground_ref", "note": "..."}

Returns small JSON: {ok, ground_reference_object, votes, rationale, error?}.

Design note: not a ReAct mini-agent — single fan-out vote on the first frame.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import get_agent_config, vlm_call
from ar2s.agent_configs import image_to_data_url
from ar2s.agents_visual.state import append_history, load_state, save_state
from ar2s.droid_sim._util import snake_case_name


_PROMPT_PATH = Path(__file__).with_name("ground_ref_prompt.md")
_AGENT_PATH = "visual.subagents.ground_ref"


class _GroundChoice(BaseModel):
    object_name: str = Field(
        description=(
            "Exact name of the ground reference object — must match one of "
            "the candidates from the user prompt, case-insensitive."
        )
    )
    reasoning: str = Field(description="One sentence justifying the choice.")


def _scene_candidates(state: dict) -> list[str]:
    """Candidates for the ground reference vote.

    Prefer ``segmentation_verdict.accepted`` (objects that came through the
    SAM3 round-1 / round-2 retries cleanly) over the full
    ``discovered_objects`` list. Degraded objects (pick_best_or_drop
    sparse picks) typically end up with sparse / unstable
    pose tracks; picking one as the ground reference makes sysid's stage 2
    fail because the obj is later dropped at resolve.py (intersection of
    mesh_recover / mesh_scale / pose_tracking successes) and never appears
    in the emitted ``objects/`` dir. Fall back to discovered_objects only
    when segmentation_verdict hasn't been written yet (shouldn't happen at
    this stage of the pipeline, but defensive).
    """
    pickup = set(state.get("pickup_objects") or [])
    verdict = state.get("segmentation_verdict") or {}
    if "accepted" in verdict:
        pool = verdict.get("accepted") or []
    else:
        pool = state.get("discovered_objects") or []
    return [o for o in pool if o and o != "robot" and o not in pickup]


def _emit(run_root: str, pick: str, note: str, **extras) -> str:
    state = load_state(run_root)
    state["ground_reference_object"] = pick
    save_state(state, run_root)
    append_history(run_root, "ground_ref", note=note)
    return json.dumps({"ok": True, "ground_reference_object": pick, **extras})


def _fallback(run_root: str, candidates: list[str], reason: str) -> str:
    state = load_state(run_root)
    hint = (state.get("entry") or {}).get("ground_reference_object") or ""
    hint_id = snake_case_name(hint) if hint else ""
    pick = hint_id if hint_id in set(candidates) else (candidates[0] if candidates else "")
    return _emit(
        run_root, pick,
        note=f"fallback: {reason} -> {pick!r}",
        fallback_reason=reason,
    )


@tool
def ground_ref(run_root: str) -> str:
    """Pick which scene object defines the floor / ground, via multi-model VLM vote.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``discovered_objects``, ``pickup_objects`` (set by pickup_objects
            subagent), and ``stages.svo_extract`` to have completed.

    Returns:
        JSON string:
          {"ok": True,
           "ground_reference_object": str,
           "votes":                   {object_name: count, ...},
           "rationale":               str}
        Fallbacks (still ok=True with `fallback_reason` set):
          - single candidate (after excluding robot + pickup) -> picks it
          - first frame missing / all VLMs fail -> entry.ground_reference_object
            hint or first candidate
    """
    state = load_state(run_root)
    candidates = _scene_candidates(state)

    if not candidates:
        return _fallback(run_root, [], "no non-robot non-pickup candidates")
    if len(candidates) == 1:
        only = candidates[0]
        return _emit(run_root, only, note=f"single candidate -> {only!r}")

    svo = state.get("stages", {}).get("svo_extract", {})
    if not svo.get("ok"):
        return _fallback(run_root, candidates, "no completed svo_extract stage")

    left_dir = svo.get("outputs", {}).get("left_frames_dir")
    if not left_dir:
        return _fallback(run_root, candidates, "svo_extract.outputs.left_frames_dir missing")

    first_frame = Path(left_dir) / "frame_000000.png"
    if not first_frame.exists():
        return _fallback(run_root, candidates, f"first frame missing at {first_frame}")

    image = image_to_data_url(first_frame)
    system_prompt = _PROMPT_PATH.read_text()

    pickup_for_msg = ", ".join(state.get("pickup_objects") or []) or "<unknown>"
    user_text = (
        "Look at this workspace image. The robot is picking up "
        f"{pickup_for_msg} (already chosen). "
        f"Pick the ground reference from these candidates: "
        f"{', '.join(candidates)}."
    )

    print("[ground_ref] querying VLMs ...")
    responses = vlm_call(
        _AGENT_PATH,
        system=system_prompt,
        user_text=user_text,
        images=[image],
        output_schema=_GroundChoice,
    )

    cand_lower = {snake_case_name(c): c for c in candidates}
    votes: Counter[str] = Counter()
    per_model_pick: dict[str, str] = {}
    for model_id, r in responses.items():
        if r is None:
            continue
        chosen = snake_case_name(r.object_name)
        if chosen in cand_lower:
            votes[cand_lower[chosen]] += 1
            per_model_pick[model_id] = cand_lower[chosen]
        else:
            print(f"[ground_ref] {model_id.split('/')[-1]}: "
                  f"chose unknown {chosen!r}, dropping")

    if not votes:
        return _fallback(run_root, candidates, "all VLMs failed or chose unknowns")

    top_count = votes.most_common(1)[0][1]
    top_picks = [obj for obj, n in votes.items() if n == top_count]
    preferred_model = get_agent_config(_AGENT_PATH)["models"][0]
    if len(top_picks) == 1:
        winner = top_picks[0]
        rationale = f"majority among {sum(votes.values())} votes"
    elif preferred_model in per_model_pick and per_model_pick[preferred_model] in top_picks:
        winner = per_model_pick[preferred_model]
        rationale = f"tied {top_picks}; broken by preferred model"
    else:
        winner = sorted(top_picks)[0]
        rationale = f"tied {top_picks}; broken alphabetically"

    print(f"[ground_ref] vote tally: {dict(votes)}")
    print(f"[ground_ref] ground_reference: {winner!r}  ({rationale})")

    return _emit(
        run_root, winner,
        note=f"tally={dict(votes)} -> {winner!r} ({rationale})",
        votes=dict(votes),
        rationale=rationale,
    )
