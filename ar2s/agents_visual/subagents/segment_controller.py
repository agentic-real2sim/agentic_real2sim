"""segment — batched, per-object segmentation controller (orchestrator tool).

Owns the keyframe → SAM3 → critique → retry loop for every object in one
deterministic Python state machine. Two rounds, no escalation ladder:

  ROUND 1 (primary camera only). Per object, keyframe_select returns a prompt
    frame AND 3 SAM3 synonyms; box_detect returns one VLM bounding box on that
    frame. Five jobs go out in a single SAM3 batch: base name, 3 synonyms, and
    a hybrid text+box prompt. The robot is text-only (no box) and uses its own
    fixed synonym list, so it contributes 6 jobs.
  ROUND 2 (only for objects still pending). The secondary view is
    lazy-bootstrapped here — ``svo_extract.run_for_secondary`` +
    ``video_build.run_for_secondary`` (RGB only). The primary view RESELECTS a
    keyframe, shown round 1's frame and asked for one unlike it, and re-detects
    its box there; the secondary view picks a keyframe independently. Both
    reuse round 1's synonyms — the vocabulary is not what round 2 varies.
    Two SAM3 batches (one per camera video), judged together; the first ACCEPT
    wins and primary is preferred on a same-round tie.

An object stays pending after a round when nothing was accepted OR when the
best candidate covers at most one frame — a single masked frame is not evidence
a track exists, even if the judge liked the look of it.

Per-candidate judging is cheap → expensive: coverage ``COVERAGE_MIN`` →
``0 < n < _MIN_FRAMES_FOR_PIPELINE`` sparse gate → VLM ``mask_critic``. The
robot is auto-accepted and never reaches the VLM. The VLM judge is also LAZY:
within each camera view, candidates are judged in descending
mask-count order and judging stops at the first the critic accepts in each
view — that is necessarily the highest-count accepted candidate in that view.
Both views still evaluate independently before primary wins a same-round tie,
for far fewer VLM calls.

Finalization: an object pending after round 2 gets its least-bad saved
candidate via ``pick_best_or_drop`` (across both rounds and both cameras,
including the 1-frame ones) and is marked degraded, or dropped. The chosen
candidate is materialised to the canonical ``segmentation/<object>/`` layout
downstream stages expect. The robot is exempt — with no mask after round 2 it
is dropped and reported as ``robot_mask_missing=true``; default non-strict runs
continue without the robot-IoU alignment term, while ``run_pipeline --strict``
halts before finalize.

Reads: discovered_objects, discovered_object_descriptions,
       stages.{svo_extract, video_build}, primary_camera_id,
       secondary_camera_id, secondary_stereo_stream_path,
       secondary_stereo_intrinsics_path. ``secondary_svo_path`` may appear as
       a deprecated alias for the secondary rectified MP4.
Writes: stages.segmentation, object_keyframes, segmentation_result,
        segmentation_verdict, mask_camera_id_by_object,
        stages.{svo_extract_secondary, video_build_secondary} (when
        secondary is lazy-bootstrapped).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_core.tools import tool

from ar2s.agents_visual._toolkit.segmentation import SegJob
from ar2s.agents_visual.skills import svo_extract as _svo_skill
from ar2s.agents_visual.skills import video_build as _vb_skill
from ar2s.agents_visual.skills.segmentation import run_jobs
from ar2s.agents_visual.state import append_history, load_state, record_stage, save_state
from ar2s.agents_visual.subagents.keyframe_select import select_keyframe_for_object
from ar2s.agents_visual.subagents.mask_critic import (
    _MIN_FRAMES_FOR_PIPELINE,
    judge_candidate,
    pick_best_or_drop,
)
from ar2s.droid_sim._util import sanitize_name

MAX_ATTEMPTS = 2
ROBOT = "robot"
COVERAGE_MIN = 0.2   # reject candidates whose masks cover < this fraction of frames (propagation collapse)
# A candidate masking one lone frame is not a track. Even when the VLM likes the
# look of that frame, retry rather than freeze the object on it.
MIN_NONEMPTY_TO_FREEZE = 2
EARLY_RESCUE_WINDOW = 5   # frames 0..4 checked for a non-empty mask when coverage/sparse gates fail


def _early_frame_with_mask(masks_dir: str) -> int | None:
    """Return the index of the earliest frame in [0, EARLY_RESCUE_WINDOW) whose
    mask file is non-empty, or None if none of them are.

    Used by the coverage/sparse gate rescue path: SAM3 sometimes gets one
    clean pen/marker mask on the prompt frame but its tracker collapses to
    empty on later frames, tanking coverage. If that one good mask lives in
    the first few frames, we still route the candidate to the VLM critic so
    the pipeline can accept it and let pose_tracking redo the propagation.
    """
    import numpy as np
    from PIL import Image
    from pathlib import Path as _P
    if not masks_dir:
        return None
    for k in range(EARLY_RESCUE_WINDOW):
        p = _P(masks_dir) / f"{k:06d}.png"
        if not p.is_file():
            continue
        try:
            m = np.array(Image.open(p))
        except Exception:
            continue
        if int(m.sum()) > 0:
            return k
    return None


# Robot has no visual instance description, but downstream pose alignment still
# needs its early-frame mask. These prompts mirror skills/segmentation.
_ROBOT_SYNONYMS = ["robot arm", "robotic arm", "gripper", "franka arm", "robot manipulator"]


def _out_subdir(name: str, round_idx: int, prompt: str, keyframe: int) -> str:
    return f"{sanitize_name(name)}__r{round_idx}__{sanitize_name(prompt)}__kf{keyframe}"


def _left_frames_dir(state: dict, view: str) -> str | None:
    stage_key = "svo_extract" if view == "primary" else "svo_extract_secondary"
    sve = state.get("stages", {}).get(stage_key, {})
    return (sve.get("outputs") or {}).get("left_frames_dir") if sve.get("ok") else None


def _detect_boxes(
    state: dict, view: str, objs_kf: list[tuple[dict, int]],
) -> dict[str, list[int]]:
    """Batch box detection for ``objs_kf`` on ``view``, one VLM call per keyframe.

    ``objs_kf`` is ``[(object, keyframe), ...]``; objects sharing a keyframe go
    out in a single ``detect_boxes`` call. Robot (text-only) and objects with no
    object_discovery description — the description is what disambiguates which
    instance to box — are skipped. Returns ``{object_name: [x1,y1,x2,y2]}``.
    """
    # Lazy import keeps detection VLM/PIL setup out of the normal controller
    # import path.
    from ar2s.agents_visual.subagents.box_propose import detect_boxes

    left_dir = _left_frames_dir(state, view)
    if not left_dir:
        return {}
    descriptions = state.get("discovered_object_descriptions") or {}

    by_kf: dict[int, list[tuple[str, str]]] = {}
    for o, keyframe in objs_kf:
        if o["name"] == ROBOT:
            continue
        desc = descriptions.get(o["name"]) or ""
        if not desc:
            print(f"[segment]   {o['name']}: no box — no object description")
            continue
        by_kf.setdefault(int(keyframe), []).append((o["name"], desc))

    out: dict[str, list[int]] = {}
    for kf, targets in by_kf.items():
        kf_path = Path(left_dir) / f"frame_{kf:06d}.png"
        if not kf_path.is_file():
            print(f"[segment]   box: keyframe {kf_path} missing for {[n for n, _ in targets]}")
            continue
        try:
            out.update(detect_boxes(str(kf_path), targets))
        except Exception as e:                              # noqa: BLE001 — VLM boundary
            print(f"[segment]   box_detect failed on kf{kf}: {e}")
    return out


def _build_specs(
    o: dict, keyframe: int, synonyms: list[str], box: list[int] | None,
) -> list[dict]:
    """One (object, view)'s round of SAM3 job specs.

    The batch shape: base name, its synonyms (3 from the VLM, or the robot's
    fixed 5), and — when ``box`` is set — one hybrid text+box job carrying the
    base name plus the detected box. ``out_subdir`` is filled in per round at
    job-build time.
    """
    prompts = [o["name"]]
    color = str(o.get("color") or "").strip()
    if color and o["name"] != ROBOT and color.lower() not in o["name"].replace("_", " ").lower():
        prompts.append(f"{color} {o['name'].replace('_', ' ')}")
    for s in synonyms:
        if s not in prompts:
            prompts.append(s)
    specs = [{"prompt": p, "keyframe": keyframe, "box": None} for p in prompts]
    if box:
        specs.append({"prompt": o["name"], "keyframe": keyframe, "box": box})
    return specs


def _cheap_verdict(o: dict, cand: dict) -> bool | None:
    """Free pre-VLM gate. Sets ``cand["reasoning"]`` for decided cases.

    Returns True (accept without the VLM — robot only), False (reject without
    the VLM — coverage / sparse), or None (passes both cheap gates → the caller
    must run the VLM ``judge_candidate``).
    """
    n, total = cand.get("num_nonempty_masks") or 0, cand.get("num_frames") or 0
    coverage = (n / total) if total else 0.0
    if o["name"] == ROBOT:
        if coverage < COVERAGE_MIN:
            cand["reasoning"] = f"propagation collapsed: {n}/{total} frames masked"
            return False
        cand["reasoning"] = f"robot detected ({n}/{total} frames)"
        return True

    failed_gate = coverage < COVERAGE_MIN or (0 < n < _MIN_FRAMES_FOR_PIPELINE)
    if failed_gate:
        rescue_kf = _early_frame_with_mask(cand["masks_dir"])
        if rescue_kf is None:
            cand["reasoning"] = (
                f"coverage/sparse gate failed ({n}/{total}); no non-empty mask "
                f"in first {EARLY_RESCUE_WINDOW} frames"
            )
            return False
        cand["keyframe"] = rescue_kf
        print(f"[segment]   {o['name']}: early-frame rescue at {rescue_kf} "
              f"after gate failure ({n}/{total})")
    return None


def _preferred_accepted_candidate(
    candidates_by_view: dict[str, list[dict]],
) -> tuple[str, dict] | None:
    """Return the highest-coverage accepted candidate, preferring primary.

    This small deterministic seam makes the same-round camera tie rule
    independently testable without loading SAM3 or a VLM.
    """
    for view in ("primary", "secondary"):
        accepted = [c for c in candidates_by_view.get(view, []) if c["accept"]]
        if accepted:
            return view, max(accepted, key=lambda c: c["num_nonempty_masks"])
    return None


def _round_one_object(name: str, descriptions: dict, colors: dict, run_root: str) -> dict:
    """Build one object's first-round state without touching SAM3 or a VLM."""
    o = {
        "name": name, "description": descriptions.get(name, ""),
        "color": colors.get(name, ""), "status": "pending", "winner": None,
        "winning_view": "", "winning_camera_id": "", "keyframe": 0,
        "prompt": name, "synonyms": [], "all_candidates": [], "specs": {},
    }
    if name == ROBOT:
        o["synonyms"] = list(_ROBOT_SYNONYMS)
    else:
        kf, synonyms = select_keyframe_for_object(run_root, name, set(), view="primary")
        o["keyframe"] = 0 if kf is None else kf
        o["synonyms"] = synonyms
    return o


def _judge_view_lazily(o: dict, candidates: list[dict], judge) -> None:
    """Evaluate one camera view until its first accepted candidate."""
    for cand in sorted(candidates, key=lambda c: c["num_nonempty_masks"], reverse=True):
        verdict = _cheap_verdict(o, cand)
        if verdict is False:
            continue
        if verdict is None:
            accepted, reasoning = judge(cand)
            cand["accept"], cand["reasoning"] = accepted, reasoning
        else:
            cand["accept"] = True
        if cand["accept"]:
            return


def _materialize(winner: dict, canonical: Path) -> dict:
    """Copy the winning candidate's masks (+overlay) into segmentation/<object>/."""
    masks_dst = canonical / "masks"
    src = Path(winner["masks_dir"]).resolve()
    src_overlay = winner.get("overlay_mp4") or ""

    if canonical.exists():
        if canonical.is_symlink() or canonical.is_file():
            canonical.unlink()
        else:
            shutil.rmtree(canonical)
    canonical.mkdir(parents=True)
    shutil.copytree(src, masks_dst)
    overlay_dst = ""
    if src_overlay and Path(src_overlay).is_file():
        overlay_dst = str(canonical / Path(src_overlay).name)
        shutil.copy2(src_overlay, overlay_dst)
    return {"masks_dir": str(masks_dst), "overlay_mp4": overlay_dst}


def _bootstrap_secondary(run_root: str) -> bool:
    """Extract secondary RGB frames + build secondary mp4 (lazy).

    Called once when round 1 leaves at least one object pending AND the run was
    built with a secondary view. Does NOT run stereo_depth — that's only needed
    for objects that actually land on the secondary view. Returns True iff both
    substages succeed.

    Idempotent: if ``stages.svo_extract_secondary`` or
    ``stages.video_build_secondary`` is already ok (e.g. a prior segment
    call), the corresponding skill helper returns the cached record.
    """
    state = load_state(run_root)
    svo_done = state.get("stages", {}).get("svo_extract_secondary", {}).get("ok")
    if not svo_done:
        print("[segment] bootstrapping secondary: svo_extract.run_for_secondary ...")
        r = _svo_skill.run_for_secondary(run_root)
        if not r.get("ok"):
            print(f"[segment] secondary svo_extract failed: {r.get('error')}")
            return False
    vb_done = load_state(run_root).get("stages", {}).get("video_build_secondary", {}).get("ok")
    if not vb_done:
        print("[segment] bootstrapping secondary: video_build.run_for_secondary ...")
        r = _vb_skill.run_for_secondary(run_root)
        if not r.get("ok"):
            print(f"[segment] secondary video_build failed: {r.get('error')}")
            return False
    return True


@tool
def segment(run_root: str) -> str:
    """Segment every scene object with per-object keyframes/prompts.

    Runs the full keyframe → SAM3 → critique → retry loop internally: one
    primary-only round, then a retry round racing a reselected primary keyframe
    against the secondary view. Writes stages.segmentation +
    mask_camera_id_by_object. Call this once; on ok=True proceed to
    mesh_recover. ok=False means the robot mask is missing after both rounds,
    non-strict runs continue in a degraded mode and ``run_pipeline --strict``

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            stages.svo_extract + stages.video_build complete and
            discovered_objects set by object_discovery. For dual-view,
            state.secondary_stereo_stream_path,
            state.secondary_stereo_intrinsics_path, and
            state.secondary_camera_id must be set. ``secondary_svo_path`` is
            only accepted as a deprecated alias for the secondary rectified
            MP4 path.

    Returns:
        {"ok": True, "per_object": {name: status}, "accepted": [...],
         "degraded": [...], "dropped": [...],
         "mask_camera_id_by_object": {name: serial}}
        ``robot_mask_missing`` is True when the robot had to be dropped.
    """
    state = load_state(run_root)
    if not state.get("stages", {}).get("video_build", {}).get("ok"):
        err = "stages.video_build has not completed; run video_build first"
        record_stage(run_root, "segmentation", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    discovered = [str(o).strip() for o in (state.get("discovered_objects") or []) if o]
    if not discovered:
        entry = state.get("entry") or {}
        discovered = [str(o).strip() for o in (entry.get("object_texts") or []) if o]
    if not discovered:
        err = "no objects to segment; run object_discovery or set entry.object_texts"
        record_stage(run_root, "segmentation", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    primary_id   = state.get("primary_camera_id", "") or ""
    secondary_id = state.get("secondary_camera_id", "") or ""
    secondary_stream = state.get("secondary_stereo_stream_path") or state.get("secondary_svo_path")
    secondary_intrinsics = state.get("secondary_stereo_intrinsics_path")
    has_secondary = bool(secondary_stream and secondary_intrinsics and secondary_id)

    descriptions = state.get("discovered_object_descriptions") or {}
    colors = state.get("object_colors") or {}

    # ---- ROUND 1 setup: primary view only ----
    objs: dict[str, dict] = {}
    for name in discovered:
        objs[name] = _round_one_object(name, descriptions, colors, run_root)

    # One batched box call per shared keyframe, then build the primary specs.
    boxes = _detect_boxes(state, "primary", [(o, o["keyframe"]) for o in objs.values()])
    for o in objs.values():
        o["specs"]["primary"] = _build_specs(
            o, o["keyframe"], o["synonyms"], boxes.get(o["name"]),
        )

    secondary_bootstrapped = False

    # ---- lockstep rounds ----
    for round_idx in range(MAX_ATTEMPTS):
        jobs_by_view: dict[str, list[SegJob]] = {"primary": [], "secondary": []}
        for o in objs.values():
            if o["status"] != "pending":
                continue
            for view, specs in o["specs"].items():
                for spec in specs:
                    spec["out_subdir"] = _out_subdir(
                        o["name"], round_idx, spec["prompt"], spec["keyframe"],
                    ) + ("__box" if spec.get("box") else "")
                    jobs_by_view[view].append(SegJob(
                        text=spec["prompt"], frame_index=spec["keyframe"],
                        out_subdir=spec["out_subdir"], box=spec.get("box"),
                    ))

        if not jobs_by_view["primary"] and not jobs_by_view["secondary"]:
            break

        results_by_view: dict[str, dict] = {}
        for view in ("primary", "secondary"):
            if not jobs_by_view[view]:
                continue
            n_objs = sum(1 for o in objs.values()
                         if o["status"] == "pending" and o["specs"].get(view))
            print(f"[segment] round {round_idx + 1}/{MAX_ATTEMPTS}: "
                  f"{len(jobs_by_view[view])} {view} jobs over {n_objs} objects")
            try:
                results_by_view[view] = run_jobs(
                    run_root, jobs_by_view[view], view=view,
                )
            except Exception as exc:                    # subprocess boundary
                err = f"SAM3 {view} segmentation failed: {exc}"
                record_stage(run_root, "segmentation", ok=False, error=err)
                return json.dumps({"ok": False, "error": err})

        # Per-obj per-view: build round candidates (mask info only), then judge
        # lazily below — the VLM is the expensive part and most candidates never
        # need it.
        for o in objs.values():
            if o["status"] != "pending":
                continue
            view_round_cands: dict[str, list[dict]] = {}
            for view, specs in o["specs"].items():
                if not specs:
                    continue
                round_cands: list[dict] = []
                for spec in specs:
                    info = (results_by_view.get(view) or {}).get(spec["out_subdir"])
                    cand = {
                        "view":     view,
                        "prompt":   spec["prompt"],
                        "keyframe": spec["keyframe"],
                        "box":      spec.get("box"),
                        "out_subdir": spec["out_subdir"],
                        "masks_dir":   info.masks_dir if info else "",
                        "overlay_mp4": info.overlay_mp4 if info else "",
                        "num_nonempty_masks": info.num_nonempty_masks if info else 0,
                        "num_frames":         info.num_frames if info else 0,
                        "accept":    False,
                        "reasoning": "",
                    }
                    o["all_candidates"].append(cand)
                    round_cands.append(cand)
                view_round_cands[view] = round_cands

            # Judge both views independently. Within a view, judge the highest
            # mask-count candidates first and stop at the first critic acceptance:
            # it is the max-count accepted candidate for that view. The primary
            # preference is applied only after both per-view evaluations finish.
            for view in ("primary", "secondary"):
                def _judge(cand: dict):
                    v = judge_candidate(run_root, o["name"], cand["prompt"],
                                        cand["keyframe"], cand["masks_dir"],
                                        description=o.get("description", ""))
                    return v.accept, v.reasoning
                _judge_view_lazily(o, view_round_cands.get(view, []), _judge)

            preferred = _preferred_accepted_candidate(view_round_cands)
            if preferred is not None:
                view, cand = preferred
                o["winner"]            = cand
                o["status"]            = "accepted"
                o["winning_view"]      = view
                o["winning_camera_id"] = primary_id if view == "primary" else secondary_id
                o["keyframe"]          = cand["keyframe"]
                o["prompt"]            = cand["prompt"]
                print(f"[segment]   {o['name']}: ACCEPT on {view} "
                      f"(prompt={o['prompt']!r}, kf={o['keyframe']}"
                      f"{', box' if cand.get('box') else ''})")
            if o["status"] != "accepted":
                print(f"[segment]   {o['name']}: reject (no candidate accepted)")

        pending = [o for o in objs.values() if o["status"] == "pending"]
        if not pending or round_idx == MAX_ATTEMPTS - 1:
            break

        # ---- ROUND 2 setup: reselect primary keyframe + open the secondary view ----
        if has_secondary and not secondary_bootstrapped:
            print(f"[segment] secondary bootstrap: {len(pending)} pending objs after "
                  f"round 1 → extracting secondary view ...")
            secondary_bootstrapped = _bootstrap_secondary(run_root)
            if not secondary_bootstrapped:
                print("[segment] secondary bootstrap failed — staying on primary only")

        # Pass 1: reselect the keyframe for each pending object per view.
        primary_kf: dict[str, int] = {}
        secondary_kf: dict[str, int] = {}
        for o in pending:
            o["specs"] = {}
            # Shown round 1's frame, the VLM is asked for one UNLIKE it — a
            # near-duplicate of a failed frame just fails the same way.
            if o["name"] == ROBOT:
                kf = 0
            else:
                kf, _ = select_keyframe_for_object(
                    run_root, o["name"], {o["keyframe"]},
                    view="primary", reference_frame=o["keyframe"],
                )
            if kf is not None:
                primary_kf[o["name"]] = kf
            if secondary_bootstrapped and o["name"] != ROBOT:
                kf_sec, _ = select_keyframe_for_object(
                    run_root, o["name"], set(), view="secondary",
                )
                if kf_sec is not None:
                    secondary_kf[o["name"]] = kf_sec

        # Pass 2: batched box detection per view (grouped by keyframe), then specs.
        by_name = {o["name"]: o for o in pending}
        p_boxes = _detect_boxes(
            state, "primary", [(by_name[n], kf) for n, kf in primary_kf.items()],
        )
        s_boxes = _detect_boxes(
            state, "secondary", [(by_name[n], kf) for n, kf in secondary_kf.items()],
        )
        for name, kf in primary_kf.items():
            o = by_name[name]
            o["specs"]["primary"] = _build_specs(
                o, kf, o["synonyms"], p_boxes.get(name),
            )
        for name, kf in secondary_kf.items():
            o = by_name[name]
            o["specs"]["secondary"] = _build_specs(
                o, kf, o["synonyms"], s_boxes.get(name),
            )

    # ---- finalization: best-of-saved, or drop ----
    for o in objs.values():
        if o["status"] != "pending":
            continue
        if o["name"] == ROBOT:
            continue                                  # robot decided below (HALT on still-pending)

        idx = (pick_best_or_drop(run_root, o["name"], o["all_candidates"],
                                 description=o.get("description", ""))
               if o["all_candidates"] else None)
        # A pick is only usable if it actually has masks; SAM3 may have found
        # no detection on every tried frame.
        if idx is not None and (o["all_candidates"][idx].get("num_nonempty_masks") or 0) > 0:
            chosen = o["all_candidates"][idx]
            o["winner"]            = chosen
            o["status"]            = "degraded"
            o["winning_view"]      = chosen.get("view") or "primary"
            o["winning_camera_id"] = (primary_id if o["winning_view"] == "primary"
                                       else secondary_id)
            o["keyframe"]          = chosen["keyframe"]
            o["prompt"]            = chosen["prompt"]
        else:
            o["status"] = "dropped"

    robot = objs.get(ROBOT)
    robot_mask_missing = bool(robot is not None and robot["status"] == "pending")
    if robot_mask_missing:
        robot["status"] = "dropped"
        print(
            "[segment] WARNING: robot mask missing on frame 0; keyframe "
            "rotation is disabled so there is no retry. Continuing in "
            "degraded mode without the robot-IoU alignment term."
        )

    # ---- materialize winners + write state ----
    seg_root = Path(run_root) / "segmentation"
    objects_out: list[dict] = []
    object_keyframes: dict[str, int] = {}
    mask_camera_id_by_object: dict[str, str] = {}
    for o in objs.values():
        if o["status"] in ("accepted", "degraded") and o["winner"]:
            mat = _materialize(o["winner"], seg_root / sanitize_name(o["name"]))
            objects_out.append({
                "text": o["name"],
                "masks_dir": mat["masks_dir"],
                "overlay_mp4": mat["overlay_mp4"],
                "num_nonempty_masks": o["winner"]["num_nonempty_masks"],
            })
            object_keyframes[o["name"]] = int(o["winner"]["keyframe"])
            if o["winning_camera_id"]:
                mask_camera_id_by_object[o["name"]] = o["winning_camera_id"]

    per_object = {o["name"]: o["status"] for o in objs.values()}
    accepted = [n for n, s in per_object.items() if s == "accepted"]
    degraded = [n for n, s in per_object.items() if s == "degraded"]
    dropped  = [n for n, s in per_object.items() if s == "dropped"]
    sec_winners = [n for n in (accepted + degraded)
                   if mask_camera_id_by_object.get(n) == secondary_id]

    segmentation_result = {
        o["name"]: {
            "status": o["status"], "keyframe": o["keyframe"], "prompt": o["prompt"],
            "winning_view": o["winning_view"],
            "winning_camera_id": o["winning_camera_id"],
            "masks_dir": (o["winner"] or {}).get("masks_dir", "") if o["winner"] else "",
            "candidates": o["all_candidates"],
        }
        for o in objs.values()
    }

    outputs = {"output_dir": str(seg_root), "objects": objects_out}
    stats = {
        "num_objects":  len(objs),
        "num_accepted": len(accepted),
        "num_degraded": len(degraded),
        "num_dropped":  len(dropped),
        "num_on_secondary": len(sec_winners),
        "robot_mask_missing": robot_mask_missing,
        "object_keyframes": object_keyframes,
    }
    record_stage(run_root, "segmentation", ok=True, outputs=outputs, stats=stats)

    state = load_state(run_root)
    state["object_keyframes"] = object_keyframes
    if object_keyframes:
        state["prompt_frame_index"] = int(next(iter(object_keyframes.values())))
    state["segmentation_result"] = segmentation_result
    state["segmentation_verdict"] = {
        "per_object": per_object, "accepted": accepted,
        "degraded": degraded, "dropped": dropped, "robot_ok": not robot_mask_missing,
        "robot_mask_missing": robot_mask_missing,
        "on_secondary": sec_winners,
    }
    if mask_camera_id_by_object:
        state["mask_camera_id_by_object"] = mask_camera_id_by_object
    save_state(state, run_root)
    append_history(run_root, "segment",
                   note=(f"accepted={accepted} degraded={degraded} dropped={dropped}"
                         + (f" on_secondary={sec_winners}" if sec_winners else "")))

    print(f"[segment] done: accepted={accepted} degraded={degraded} dropped={dropped}"
          + (f" (on secondary view: {sec_winners})" if sec_winners else ""))
    return json.dumps({
        "ok": True, "per_object": per_object,
        "accepted": accepted, "degraded": degraded, "dropped": dropped,
        "robot_mask_missing": robot_mask_missing,
        "mask_camera_id_by_object": mask_camera_id_by_object,
    })
