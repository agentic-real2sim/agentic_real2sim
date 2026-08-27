"""object_discovery subagent — task-focused multi-model VLM object discovery.

This is the canonical object discovery agent. It prefers wrist-camera evidence
when available, falls back to the primary extracted left frames. The VLM
prompt itself filters to task-relevant objects only (see
object_discovery_prompt.md), so no separate relevance-pruning stage runs
downstream.

Reads from state.json:
  - entry.wrist_stream_path                     (wrist-cam rectified .mp4, if staged)
  - entry.object_texts                          (authoritative hint override)

Primary frame source: entry.wrist_stream_path, if present.
Fallback: stages.svo_extract.outputs.left_frames_dir (backward compat).

Writes to state.json:
  - state.discovered_objects = list[str]        (snake_case object ids,
                                                 always contains "robot")
  - state.discovered_object_descriptions = {name: description, ...}
                                           (empty only if the merge call
                                           itself fails)
  - state.history.append({"agent": "object_discovery", "note": "..."})

Returns small JSON to the orchestrator: {ok, discovered_objects,
n_models_used, error?} plus path-specific keys (see tool docstring).

Design note: not a ReAct mini-agent — fixed two-stage control flow:
  1. Fan-out: Default vision models independently label the scene.
  2. Merge: Preferred model always runs next — reconciling N>=2 models' label
     sets into one deduplicated list, or simply augmenting a single model's
     list — attaching a per-object description either way (see
     object_merging_prompt.md).
If the merge call itself fails, falls back to simple vote-threshold filtering
(no descriptions in that case). Recall is preferred over precision
(downstream SAM3 returns empty masks for hallucinated objects).
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ar2s.agents_visual._models import vlm_call, vlm_call_first_success, get_agent_config
from ar2s.agent_configs import image_to_data_url
from ar2s.agents_visual.state import append_history, load_state, save_state
from ar2s.droid_sim._util import snake_case_name


_PROMPT_PATH = Path(__file__).with_name("object_discovery_prompt.md")
_MERGE_PROMPT_PATH = Path(__file__).with_name("object_merging_prompt.md")

_DISCOVERY_AGENT_PATH = "visual.subagents.object_discovery"
_MERGE_AGENT_PATH = "visual.subagents.object_merging"

_N_FRAMES = 6

# --- Fast-fail guard for the wrist-cam truncation loop -----------------------
# On "wrist-camera" episodes the sampled frames can be cluttered, and the 
# discovery VLMs may occasionally degenerate into an over-long response that 
# hits the completion-token cap, so nothing parses and ``successes`` comes
# back empty every time. Left alone, the ReAct orchestrator keeps nudging
# object_discovery, and each retry regenerates a ~16k-token response (minutes
# each) that can never parse. We persist a consecutive all-VLM-failure counter
# in state.json and, after this many failures in a row, latch a terminal flag
# so the pipeline fails fast (the flag is read by run_visual in orchestrator.py,
# which aborts the run).
_MAX_OD_CONSECUTIVE_FAILURES = 5
_OD_FAILS_KEY = "_od_consecutive_failures"   # int, persisted in state.json
_OD_FAST_FAIL_KEY = "_od_fast_fail"          # bool latch, also read by run_visual

# High-precision VLM consensus, now a backup if merger fails.
# Tuned in v1 ablation:
#   1 vote  → noisy (Mistral-Large lists every object in sight)
#   2 votes → matches well to graspable scene objects
# Items the user explicitly named in entry.object_texts override the vote
# threshold (treated as authoritative).
_MIN_VOTES = 2
_ROBOT_NAME = "robot"
_ROBOT_SYNONYMS = {
    "robot", "robot arm", "robotic arm", "robot manipulator", "arm",
    "robotic gripper", "gripper",
}
_ROBOT_SYNONYM_IDS = {snake_case_name(n) for n in _ROBOT_SYNONYMS}


class _SceneObjects(BaseModel):
    objects: list[str] = Field(
        description=(
            "List of distinct physical objects directly involved in the task. "
            "Each entry names exactly one object (e.g. 'mug', 'pen'). Never "
            "combine objects. Use lowercase singular 1-3 word labels."
        )
    )
    # Forcing the model to state each object's appearance measurably improves
    # grounding: on REAL_07_04 the colorless schema conflated a transparent
    # cup and an orange spoon into one 'scoop' (and mis-identified the grasped
    # object); with per-object colors the same model separated them and got
    # grasped right. The color also seeds segmentation's "<color> <noun>"
    # prompt candidate.
    object_colors: list[str] = Field(
        description=(
            "object_colors[i] is the dominant visible color or material "
            "appearance of objects[i] — e.g. 'red', 'black', 'transparent', "
            "'metallic silver'. Be literal about what you see. MUST have the "
            "same length and order as `objects`."
        )
    )
    notes: str = Field(description="One short sentence describing the scene.")


class _MergedObject(BaseModel):
    name: str = Field(description="The merged object label.")
    color: str = Field(
        default="",
        description=(
            "Dominant visible color/material of the object, e.g. 'red', "
            "'transparent', 'metallic silver'."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "Detailed visual description for a tight bounding box: color, "
            "material, shape, size, marks, location relative to others."
        ),
    )


class _MergeOutput(BaseModel):
    objects: list[_MergedObject] = Field(
        description="Merged array of best-chosen labels with descriptions, one per matched pair."
    )
    notes: str = Field(
        description=(
            "One short sentence describing any conflicts or ambiguities encountered, "
            "or 'none' if the merge was clean."
        )
    )


def _normalize(name: str) -> str:
    n = snake_case_name(name)
    if name.strip().lower() in _ROBOT_SYNONYMS or n in _ROBOT_SYNONYM_IDS:
        return _ROBOT_NAME
    return n


def _ensure_robot(objs: list[str]) -> list[str]:
    if _ROBOT_NAME not in objs:
        objs = objs + [_ROBOT_NAME]
    return objs


def _fallback_to_hint(run_root: str, hint: list[str], reason: str) -> str:
    """Use user-provided object_texts hint when VLMs unavailable."""
    discovered = _ensure_robot([_normalize(h) for h in hint if h])
    state = load_state(run_root)
    state["discovered_objects"] = discovered
    save_state(state, run_root)
    append_history(
        run_root, "object_discovery",
        note=f"fallback: {reason} -> discovered={discovered}",
    )
    return json.dumps({
        "ok":                  True,
        "discovered_objects":  discovered,
        "n_models_used":       0,
        "votes":               {},
        "fallback_reason":     reason,
    })


def _merge_labels(
    successes: dict[str, _SceneObjects],
    images: list[str],
) -> _MergeOutput | None:
    """Reconcile (or, for a single model, augment) VLM label output(s) via the
    configured merge agent.

    Builds a vlm1_objects … vlmN_objects payload for however many models are in
    ``successes`` (caller ensures >= 1) — the merge prompt handles N=1 by
    passing every object through and still attaching descriptions, since it
    always needs to run that step now.
    Returns None on failure; caller falls back to voting in that case.
    """
    payload: dict[str, list[str]] = {}
    for i, (_model_id, r) in enumerate(successes.items(), start=1):
        payload[f"vlm{i}_objects"] = r.objects
        payload[f"vlm{i}_colors"] = list(getattr(r, "object_colors", []) or [])

    n = len(successes)
    user_text = (
        f"Here are {len(images)} evenly-spaced frames from the demo "
        f"(the same frames the {n} VLM{'s' if n != 1 else ''} analyzed).\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    preferred_merge_model = get_agent_config(_MERGE_AGENT_PATH)["models"][0]
    print(f"[object_discovery] merging labels via {preferred_merge_model.split('/')[-1]} ...")
    try:
        return vlm_call_first_success(
            _MERGE_AGENT_PATH,
            system=_MERGE_PROMPT_PATH.read_text(),
            user_text=user_text,
            images=images,
            output_schema=_MergeOutput,
        )
    except Exception as exc:
        print(f"[object_discovery] merge step failed ({exc}); falling back to voting")
        return None


def _find_wrist_mp4(state: dict) -> Path | None:
    """Return the staged wrist-cam MP4 path, or None if not available.

    Reads entry.wrist_stream_path (populated by build_raw_episode when the
    staged raw_data dir has a wrist <serial>-stereo.mp4). Returns None if the
    field is empty/absent or the file no longer exists on disk.
    """
    entry = state.get("entry") or {}
    wrist_path = entry.get("wrist_stream_path")
    if not wrist_path:
        return None
    p = Path(wrist_path)
    return p if p.exists() else None


def _extract_frames_from_mp4(mp4: Path, out_dir: Path, n_frames: int) -> list[Path]:
    """Extract n_frames evenly-spaced PNGs from mp4 using ffmpeg.

    Writes to out_dir/frame_XXXXXX.png. Existing files are overwritten.
    Returns the list of written paths.
    """
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(mp4),
        ],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip())
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n_frames):
        t = i * duration / n_frames  # avoids seeking to EOF which ffmpeg silently drops
        out_path = out_dir / f"frame_{i:06d}.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{t:.4f}",
                "-i", str(mp4),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_path),
            ],
            capture_output=True, check=True,
        )
        paths.append(out_path)

    # Stereo SBS videos (wrist/external ZED exports are 2560x720) waste half
    # the width on the redundant right-eye view; after image_to_data_url's
    # max-dim cap that costs HALF the effective vertical resolution — enough
    # to make small/transparent objects invisible to the VLM (REAL_07_04's
    # grasped transparent cup was found at full res 6/6 and missed at
    # 1024x288 3/3). Keep only the left eye.
    from PIL import Image
    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
            if w >= 2 * h:
                img.crop((0, 0, w // 2, h)).save(p)
    return paths


def _sample_frames(state: dict, run_root: str) -> list[Path]:
    """Evenly-spaced frames (first→last) for multi-frame scene inventory.

    Primary source: wrist-cam MP4 (smallest numeric ID under
    {episode_dir}/recordings/MP4/). Frames are extracted to
    {run_root}/od_frames/ via ffmpeg and kept for debugging.

    Fallback: svo_extract left-frame directory (backward compat when the
    MP4 directory is absent or extraction fails).
    """
    mp4 = _find_wrist_mp4(state)
    if mp4 is not None:
        print(f"[object_discovery] using wrist-cam MP4: {mp4.name}")
        try:
            return _extract_frames_from_mp4(
                mp4, Path(run_root) / "od_frames", _N_FRAMES
            )
        except Exception as exc:
            print(f"[object_discovery] wrist-cam extraction failed ({exc}); "
                  "falling back to svo_extract frames")

    # Fallback: svo_extract output
    sve = state.get("stages", {}).get("svo_extract", {})
    if not sve.get("ok"):
        return []
    left_dir = sve.get("outputs", {}).get("left_frames_dir")
    if not left_dir:
        return []
    frames = sorted(Path(left_dir).glob("frame_*.png"))
    if not frames:
        return []
    if len(frames) <= _N_FRAMES:
        return frames
    return [frames[int(i * (len(frames) - 1) / (_N_FRAMES - 1))] for i in range(_N_FRAMES)]


@tool
def object_discovery(run_root: str) -> str:
    """Identify objects on the workspace by VLM consensus over sampled frames.

    Args:
        run_root: absolute path to runs/<episode_id>/. Frames are sourced
            from entry.wrist_stream_path if present; falls back to
            svo_extract output otherwise.

    Returns:
        JSON string (merge path):
          {"ok": True, "discovered_objects": list[str],
           "n_models_used": int, "hint_only": list[str], "merge_notes": str,
           "object_descriptions": {name: str}}
        JSON string (vote fallback — merge call itself failed):
          {"ok": True, "discovered_objects": list[str],
           "n_models_used": int, "hint_only": list[str],
           "votes": {name: count}}
        JSON string (hint fallback — no frames):
          {"ok": True, ..., "n_models_used": 0, "fallback_reason": str}
        On unrecoverable failure (no frames and no hint):
          {"ok": False, "error": "<reason>"}
    """
    state = load_state(run_root)
    entry = state.get("entry") or {}
    hint = list(entry.get("object_texts") or [])

    # Fast-fail guard (part a): if a previous invocation already latched the
    # terminal flag (_MAX_OD_CONSECUTIVE_FAILURES consecutive all-VLM failures —
    # the wrist-cam truncation loop), return the hint fallback IMMEDIATELY
    # without touching the VLM. This kills the expensive ~16k-token regeneration
    # that is the main time/budget sink; run_visual detects the same flag and
    # aborts the pipeline so we never get re-invoked more than a couple of times.
    if state.get(_OD_FAST_FAIL_KEY):
        return _fallback_to_hint(
            run_root, hint,
            f"fast-fail latched after {_MAX_OD_CONSECUTIVE_FAILURES} consecutive "
            "VLM failures (wrist-cam truncation loop); skipped VLM call",
        )

    frames = _sample_frames(state, run_root)
    if not frames:
        if hint:
            return _fallback_to_hint(run_root, hint, "no frames available (wrist-cam MP4 and svo_extract both missing)")
        return json.dumps({
            "ok":    False,
            "error": "no frames available: wrist-cam MP4 not found and svo_extract has no frames; entry.object_texts is also empty",
        })

    # 1280 keeps a left-eye 1280x720 frame unscaled (the default 1024 cap
    # exists for verbose OSS models; the configured judges handle this fine).
    images = [image_to_data_url(p, max_dim=1280) for p in frames]
    system_prompt = _PROMPT_PATH.read_text()

    discovery_models = get_agent_config(_DISCOVERY_AGENT_PATH)["models"]
    print(f"[object_discovery] querying {len(discovery_models)} VLM(s) over {len(frames)} frames ...")
    responses = vlm_call(
        _DISCOVERY_AGENT_PATH,
        system=system_prompt,
        user_text=(
            f"These are {len(frames)} evenly-spaced frames from the demo, in "
            f"temporal order (first to last). List every distinct physical "
            f"object that is directly involved in the task, WITH each "
            f"object's visible color/material — look at every object "
            f"individually before naming it. "
            f"Follow the rules in the system prompt."
        ),
        images=images,
        output_schema=_SceneObjects,
    )

    successes = {m: r for m, r in responses.items() if r is not None}
    if not successes:
        # Fast-fail guard (part b): count consecutive all-VLM failures. For these
        # wrist-cam episodes "all VLMs failed" == truncation (vlm_call exposes
        # only None, not the reason), and 5 in a row is a sound abort trigger.
        # Latch the terminal flag on reaching the threshold so run_visual stops
        # nudge-looping a doomed episode. save_state here persists the counter;
        # _fallback_to_hint reloads state from disk, so the write is preserved.
        fails = int(state.get(_OD_FAILS_KEY, 0)) + 1
        state[_OD_FAILS_KEY] = fails
        reason = f"all VLMs failed ({fails}/{_MAX_OD_CONSECUTIVE_FAILURES} consecutive)"
        if fails >= _MAX_OD_CONSECUTIVE_FAILURES:
            state[_OD_FAST_FAIL_KEY] = True
            reason += "; latching fast-fail (wrist-cam truncation loop)"
            print(f"[object_discovery] {reason}")
        save_state(state, run_root)
        return _fallback_to_hint(run_root, hint, reason)

    # A VLM parsed → real objects discovered. Reset the consecutive-failure
    # counter (persisted by the merge/vote success paths below, which save
    # this same `state` object).
    state[_OD_FAILS_KEY] = 0

    for model_id, r in successes.items():
        print(f"[object_discovery] {model_id.split('/')[-1]}: "
              f"objects={r.objects}  colors={getattr(r, 'object_colors', [])}")

    # Per-model {normalized_name: color} — first model to name an object wins;
    # used directly in the voting fallback and as backfill for the merge path.
    model_colors: dict[str, str] = {}
    for r in successes.values():
        cols = list(getattr(r, "object_colors", []) or [])
        for j, raw in enumerate(r.objects):
            n = _normalize(raw)
            c = (cols[j].strip() if j < len(cols) and cols[j] else "")
            if n and c and n not in model_colors:
                model_colors[n] = c

    hint_passers = [_normalize(h) for h in hint]
    n_models = len(successes)

    # --- Stage 2: always merge/augment — even a single model's list still
    # needs per-object descriptions attached. ---
    merged = _merge_labels(successes, images)

    if merged is not None:
        raw_objects = [_normalize(o.name) for o in merged.objects if o.name]
        descriptions = {_normalize(o.name): o.description for o in merged.objects if o.name}
        colors = {
            _normalize(o.name): (o.color.strip() or model_colors.get(_normalize(o.name), ""))
            for o in merged.objects if o.name
        }
        colors = {k: v for k, v in colors.items() if v}
        merged_set = set(raw_objects)
        hint_only = [h for h in hint_passers if h and h not in merged_set]
        discovered = raw_objects + sorted(hint_only)
        discovered = _ensure_robot(discovered)

        preferred_merge_model = get_agent_config(_MERGE_AGENT_PATH)["models"][0]
        print(f"[object_discovery] {n_models}/{len(responses)} models OK; "
              f"merged via {preferred_merge_model.split('/')[-1]}")
        print(f"[object_discovery] merge notes: {merged.notes}")
        if hint_only:
            print(f"[object_discovery] hint-only objects added: {hint_only}")
        print(f"[object_discovery] discovered: {discovered}")

        state["discovered_objects"] = discovered
        state["discovered_object_descriptions"] = descriptions
        state["object_colors"] = colors
        print(f"[object_discovery] object_colors: {colors}")
        save_state(state, run_root)
        append_history(
            run_root, "object_discovery",
            note=(
                f"merged {n_models} VLM outputs via {preferred_merge_model.split('/')[-1]} "
                f"-> {discovered}; hint-only adds {hint_only}; "
                f"notes={merged.notes!r}"
            ),
        )
        return json.dumps({
            "ok":                  True,
            "discovered_objects":  discovered,
            "n_models_used":       n_models,
            "hint_only":           hint_only,
            "merge_notes":         merged.notes,
            "object_descriptions": descriptions,
        })

    # --- Fallback: voting (only reached if the merge call itself failed) ---
    votes: dict[str, int] = defaultdict(int)
    for r in successes.values():
        seen_this_model: set[str] = set()
        for raw in r.objects:
            n = _normalize(raw)
            if not n or n in seen_this_model:
                continue
            seen_this_model.add(n)
            votes[n] += 1

    # Threshold relative to how many models actually answered: full consensus
    # (2) when both models respond, graceful single-model mode (1) when
    # one provider's key is missing or it errored.
    effective_min_votes = min(_MIN_VOTES, len(successes))
    vote_passers = [obj for obj, count in votes.items() if count >= effective_min_votes]
    hint_only = [h for h in hint_passers if h and h not in set(vote_passers)]

    discovered = sorted(vote_passers, key=lambda o: (-votes[o], o)) + sorted(hint_only)
    discovered = _ensure_robot(discovered)

    print(f"[object_discovery] {n_models}/{len(responses)} models OK; "
          f"vote tally: {dict(votes)}")
    if hint_only:
        print(f"[object_discovery] hint-only objects added: {hint_only}")
    print(f"[object_discovery] discovered: {discovered}")

    state["discovered_objects"] = discovered
    state["object_colors"] = {
        n: c for n, c in model_colors.items() if n in set(discovered)
    }
    print(f"[object_discovery] object_colors: {state['object_colors']}")
    save_state(state, run_root)
    append_history(
        run_root, "object_discovery",
        note=(
            f"voted across {n_models} VLMs (>={_MIN_VOTES} votes) -> {discovered}; "
            f"hint-only adds {hint_only}"
        ),
    )
    return json.dumps({
        "ok":                  True,
        "discovered_objects":  discovered,
        "n_models_used":       n_models,
        "votes":               dict(votes),
        "hint_only":           hint_only,
    })
