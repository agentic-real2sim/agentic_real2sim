"""pickup_objects subagent — multi-model VLM vote on which objects the robot grasps.

Runs right after object_discovery and owns the pipeline's authoritative
``pickup_objects`` key.

Reads from state.json:
  - discovered_objects                          (list[str]; "robot" filtered out)
  - stages.svo_extract.outputs.left_frames_dir  (required for VLM frames)
  - entry.pickup_object                         (optional user hint;
                                                 authoritative fallback if
                                                 all VLMs fail)

Writes to state.json:
  - pickup_objects = list[str]
  - history += {"agent": "pickup_objects", "note": "..."}

Returns small JSON: {ok, pickup_objects, votes, rationale, error?}.

Design note: not a ReAct mini-agent. Single fan-out vote + threshold + emit.
Each model names every object it saw grasped, in the order it grasped them;
an object is kept when it reaches ``_MIN_VOTES`` (capped at the number of
models that answered, so a single-model run keeps that model's picks). If
nothing clears the bar the top-voted object alone is kept, so the list is
never empty when a VLM answered at all.

The emitted list stays in temporal grasp order (mean position across the
models that named it) rather than vote order, because ``pickup_objects[0]``
is what the sysid grasp stages act on — see agents_sysid/episode.py.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import vlm_call
from ar2s.agent_configs import image_to_data_url
from ar2s.agents_visual.state import append_history, load_state, save_state
from ar2s.droid_sim._util import snake_case_name


_PROMPT_PATH = Path(__file__).with_name("pickup_objects_prompt.md")
_N_KEYFRAMES = 10
_AGENT_PATH = "visual.subagents.pickup_objects"

# Same consensus threshold object_discovery uses: 2 votes when 2+ models
# answered, 1 when only one did.
_MIN_VOTES = 2


class _PickupChoice(BaseModel):
    object_names: list[str] = Field(
        description=(
            "Every object the gripper grasps and lifts over the demo, in the "
            "order it grasps them. Each entry must exactly match one of the "
            "candidates from the user prompt."
        )
    )
    reasoning: str = Field(description="One sentence justifying the choice.")


def _scene_candidates(state: dict) -> list[str]:
    return [o for o in (state.get("discovered_objects") or []) if o and o != "robot"]


def _evenly_sample(items: list[Path], n: int) -> list[Path]:
    if len(items) <= n:
        return items
    return [items[int(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _emit(run_root: str, picks: list[str], note: str, **extras) -> str:
    state = load_state(run_root)
    state["pickup_objects"] = picks
    save_state(state, run_root)
    append_history(run_root, "pickup_objects", note=note)
    return json.dumps({"ok": True, "pickup_objects": picks, **extras})


def _fallback(run_root: str, candidates: list[str], reason: str) -> str:
    state = load_state(run_root)
    hint = (state.get("entry") or {}).get("pickup_object") or ""
    hint_id = snake_case_name(hint) if hint else ""
    picks = (
        [hint_id] if hint_id in set(candidates)
        else (candidates[:1] if candidates else [])
    )
    return _emit(
        run_root, picks,
        note=f"fallback: {reason} -> {picks!r}",
        fallback_reason=reason,
    )


@tool
def pickup_objects(run_root: str) -> str:
    """Pick which scene objects the robot grasps, via multi-model VLM vote.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``discovered_objects`` (set by object_discovery) and
            ``stages.svo_extract`` to have completed.

    Returns:
        JSON string:
          {"ok": True,
           "pickup_objects": list[str],
           "votes":          {object_name: count, ...},
           "rationale":      str}
        Fallbacks (still ok=True with `fallback_reason` set):
          - single candidate -> picks it directly
          - no frames / all VLMs fail -> entry.pickup_object hint or first candidate
    """
    state = load_state(run_root)
    candidates = _scene_candidates(state)

    if not candidates:
        return _fallback(run_root, [], "no non-robot candidates in discovered_objects")
    if len(candidates) == 1:
        only = candidates[0]
        return _emit(run_root, [only], note=f"single candidate -> {only!r}")

    svo = state.get("stages", {}).get("svo_extract", {})
    if not svo.get("ok"):
        return _fallback(run_root, candidates, "no completed svo_extract stage")

    left_dir = svo.get("outputs", {}).get("left_frames_dir")
    if not left_dir:
        return _fallback(run_root, candidates, "svo_extract.outputs.left_frames_dir missing")

    all_frames = sorted(Path(left_dir).glob("frame_*.png"))
    if not all_frames:
        return _fallback(run_root, candidates, f"no frames in {left_dir}")

    sampled = _evenly_sample(all_frames, _N_KEYFRAMES)
    images = [image_to_data_url(p) for p in sampled]
    system_prompt = _PROMPT_PATH.read_text()

    user_text = (
        f"These are {len(sampled)} evenly-spaced frames from the demo "
        f"(first to last). Candidate objects: {', '.join(candidates)}.\n\n"
        f"Which of them does the robot pick up?"
    )

    print(f"[pickup_objects] querying VLMs over {len(sampled)} frames ...")
    responses = vlm_call(
        _AGENT_PATH,
        system=system_prompt,
        user_text=user_text,
        images=images,
        output_schema=_PickupChoice,
    )

    # Normalize against snake_case candidates; one vote per (model, object).
    # ``rank_sum`` accumulates each object's position in the model's list, which
    # the prompt asks to be temporal grasp order — averaging those positions
    # orders the final picks by when the robot grasps them, not by vote count.
    cand_lower = {snake_case_name(c): c for c in candidates}
    votes: Counter[str] = Counter()
    rank_sum: Counter[str] = Counter()
    n_answered = 0
    for model_id, r in responses.items():
        if r is None:
            continue
        n_answered += 1
        seen: set[str] = set()
        for raw in r.object_names:
            chosen = snake_case_name(raw)
            if chosen not in cand_lower:
                print(f"[pickup_objects] {model_id.split('/')[-1]}: "
                      f"chose unknown {chosen!r}, dropping")
                continue
            if chosen in seen:
                continue
            name = cand_lower[chosen]
            rank_sum[name] += len(seen)
            seen.add(chosen)
            votes[name] += 1

    if not votes:
        return _fallback(run_root, candidates, "all VLMs failed or chose unknowns")

    min_votes = min(_MIN_VOTES, n_answered)
    passers = [obj for obj, n in votes.items() if n >= min_votes]
    if passers:
        rationale = f">={min_votes} of {n_answered} models agreed"
    else:
        # No object cleared the bar (each model named a different item); keep
        # the single top-voted one rather than emitting an empty list.
        passers = [votes.most_common(1)[0][0]]
        rationale = f"no object reached {min_votes} votes; kept top-voted only"
    # Temporal grasp order first: picks[0] is what the robot grasps FIRST, and
    # that is the object sysid drives its grasp study on (episode.py sets
    # bundle.pickup_object from it). Vote count only breaks ties.
    picks = sorted(passers, key=lambda o: (rank_sum[o] / votes[o], -votes[o], o))

    print(f"[pickup_objects] vote tally: {dict(votes)}")
    print(f"[pickup_objects] pickup_objects: {picks!r}  ({rationale})")

    return _emit(
        run_root, picks,
        note=f"tally={dict(votes)} -> {picks!r} ({rationale})",
        votes=dict(votes),
        rationale=rationale,
    )
