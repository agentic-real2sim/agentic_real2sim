"""scale_critic skill — deterministic numerical check on mesh-scale stability.

Uses the disk-backed visual-agent state and ``@tool(run_root: str)``
signature. The original heuristic and thresholds are preserved.

This is NOT a VLM subagent — pure numerical signal. It's listed in the visual
plan under "top-level direct-own" alongside data_ingest / resolve / emit;
we place it in skills/ because it shares the deterministic-side-effect
shape with other skills (read state.json → compute → write back).

Heuristic per object (computed on the keyframe-anchored kept-frame set):
  - std_scale / median_scale > _STD_RATIO_THRESH (0.15)  → "unstable"
  - num_frames_used < _MIN_FRAMES (3)                    → "small sample"
  - keyframe_anchored AND kept/total < _AGREEMENT_THRESH (0.5)
                                                          → "low keyframe agreement"
  - keyframe_anchored AND num_frames_used == 1 AND total > 1
                                                          → "keyframe-only fallback"
    (every non-keyframe frame disagreed by > 15% — keyframe itself may be
    a suspect anchor, or the depth / mask quality varied wildly across frames)

**Warning-only**: always emits ``verdict.accept = True``. The "right" retry
path (re-segment with a different keyframe → re-recover meshes → re-score)
is the same expensive cycle the mask_critic already owns, so for now we
surface concerns in ``verdict.detail`` without triggering an extra loop.

Reads from state.json:
  - stages.mesh_scale.outputs.per_object

Writes to state.json:
  - state["scale_critic_verdict"] = {"accept": True, "reason": str, "detail": {...}}
  - state.history += {"agent": "scale_critic", "note": "..."}
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from ar2s.agents_visual.state import append_history, load_state, save_state


_STD_RATIO_THRESH = 0.15      # std/median > 15% counts as unstable
_MIN_FRAMES = 3               # fewer than this is a small-sample warning
_AGREEMENT_THRESH = 0.5       # warn when < 50% of raw frames agreed with keyframe


@tool
def scale_critic(run_root: str) -> str:
    """Flag objects with unstable mesh scale; always accepts (warning-only).

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``stages.mesh_scale`` to have completed.

    Returns:
        JSON string:
          {"ok":      True,
           "verdict": {"accept": True,
                       "reason": str,
                       "detail": {"per_object": {...}, "issues": [...]}},
           "n_issues": int}
        If stages.mesh_scale is missing/incomplete, still accepts with
        reason="no mesh_scale stage" (matches v1 behaviour).
    """
    state = load_state(run_root)
    rep = state.get("stages", {}).get("mesh_scale", {})

    if not rep.get("ok"):
        verdict = {"accept": True, "reason": "no mesh_scale stage; nothing to critique", "detail": {}}
        state["scale_critic_verdict"] = verdict
        save_state(state, run_root)
        append_history(run_root, "scale_critic", note="no mesh_scale stage")
        return json.dumps({"ok": True, "verdict": verdict, "n_issues": 0})

    per_object_in = rep.get("outputs", {}).get("per_object", [])
    issues: list[str] = []
    per_object_out: dict[str, dict] = {}

    for o in per_object_in:
        name = o.get("object_name", "?")
        median = float(o.get("median_scale", 0.0))
        std = float(o.get("std_scale", 0.0))
        n_frames = int(o.get("num_frames_used", 0))           # post-filter
        total = int(o.get("total_frames", n_frames))          # pre-filter
        kf_anchored = bool(o.get("keyframe_anchored", False))
        ratio = (std / median) if median > 0 else float("inf")
        agreement = (n_frames / total) if total > 0 else 0.0

        per_object_out[name] = {
            "median":             median,
            "std":                std,
            "std_ratio":          ratio,
            "num_frames":         n_frames,
            "total_frames":       total,
            "keyframe_anchored":  kf_anchored,
            "agreement_ratio":    agreement,
        }

        if kf_anchored and n_frames == 1 and total > 1:
            issues.append(
                f"{name}: keyframe-only fallback ({total-1}/{total} non-keyframe "
                f"frames disagreed by >15%; keyframe scale may be unreliable)"
            )
        elif kf_anchored and total >= 3 and agreement < _AGREEMENT_THRESH:
            issues.append(
                f"{name}: low keyframe agreement ({n_frames}/{total} = "
                f"{agreement:.0%}; threshold {_AGREEMENT_THRESH:.0%})"
            )
        elif n_frames < _MIN_FRAMES:
            issues.append(
                f"{name}: only {n_frames} usable frames (threshold {_MIN_FRAMES})"
            )
        elif ratio > _STD_RATIO_THRESH:
            issues.append(
                f"{name}: std/median = {ratio:.1%} (threshold {_STD_RATIO_THRESH:.0%})"
            )

    if issues:
        reason = f"{len(issues)} object(s) have unstable scale: {'; '.join(issues)}"
        print(f"[scale_critic] WARNING: {reason}")
    else:
        reason = f"all {len(per_object_out)} object scales stable"
        print(f"[scale_critic] {reason}")

    verdict = {
        "accept": True,
        "reason": reason,
        "detail": {"per_object": per_object_out, "issues": issues},
    }
    state["scale_critic_verdict"] = verdict
    save_state(state, run_root)
    append_history(run_root, "scale_critic", note=reason)

    return json.dumps({
        "ok":       True,
        "verdict":  verdict,
        "n_issues": len(issues),
    })
