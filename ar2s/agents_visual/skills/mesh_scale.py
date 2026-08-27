"""mesh_scale skill — per-object scale factor (depth pointcloud ↔ mesh).

Reads from state.json:
  - stages.svo_extract.outputs.sequence_dir       (primary depth/ + K.txt)
  - stages.stereo_depth                           (primary; required — orchestrator
                                                    now defers this stage until just
                                                    before mesh_scale, after segment/
                                                    mesh_recover, so FoundationStereo
                                                    inference is skipped on earlier
                                                    pipeline failures)
  - stages.svo_extract_secondary.outputs.sequence_dir   (secondary; PR-3)
  - stages.stereo_depth_secondary                  (lazy-triggered by PR-4 below
                                                    when at least one obj's
                                                    mask landed on secondary)
  - stages.segmentation.outputs.output_dir        (SAM 3 mask root; per-text subdirs)
  - stages.mesh_recover.outputs.{mesh_dir, per_object}
    Only objects with mesh_recover.success=True are passed; skipping failed
    ones avoids feeding compute_mesh_scale.py a missing .glb path (mirrors
    v1 orchestrator behaviour).
  - mask_camera_id_by_object                      (PR-3; routes per-obj depth)

Dual-view (PR-4): objects are grouped by their winning camera and the
compute_mesh_scale.py subprocess is invoked once per group with that
camera's ``sequence_dir`` (depth + K). The per-object on-disk mask layout
(``segmentation/<obj>/masks/``) is camera-agnostic — segment_controller's
``_materialize`` puts the winner there regardless of source view — so a
single ``mask_dir`` works across both groups. If at least one object is
on secondary and ``stages.stereo_depth_secondary`` isn't done, mesh_scale
auto-runs ``stereo_depth.run_for_secondary`` upfront.

Writes to disk:
  - <sequence_dir>/scale/<object>/frame_*_meta.npz  (compute_mesh_scale output)
  - logs under <run_root>/logs/mesh_scale.{cmd,stdout,stderr,returncode}

Writes to state.json:
  - state["stages"]["mesh_scale"] = {
        ok, outputs.per_object = [{
            object_name, median_scale, std_scale, num_frames_used, meta_dir
        }, ...], stats, error
    }

The downstream scale_critic skill reads outputs.per_object to surface
unstable scales (std/median > 15% or num_frames_used < 3); resolve will
broadcast median_scale to (s, s, s) for the per-object scale_path .npy.

Subprocess env: ``qianjun_foundation_stereo`` — same env as stereo_depth /
svo_extract because compute_mesh_scale.py lives under FoundationStereo's
scripts/ and uses the same FoundationStereo helpers. Auto-injected
XFORMERS_DISABLED=1 from _runners.py still applies (sm_120 dinov2 fallback).
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from ar2s.agents_visual._toolkit import mesh_scale as _toolkit
from ar2s.agents_visual.skills import stereo_depth as _stereo_skill
from ar2s.agents_visual.state import load_state, record_stage


@tool
def mesh_scale(run_root: str) -> str:
    """Compute per-object metric scale by aligning depth pointcloud to the recovered mesh.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``stages.svo_extract``, ``stages.segmentation``, and
            ``stages.mesh_recover`` to have completed.

    Returns:
        JSON string:
          {"ok": True,
           "outputs": {"per_object": [{
                "object_name", "median_scale", "std_scale",
                "num_frames_used", "meta_dir"}, ...]},
           "stats":   {"n_objects", "n_with_frames", "n_total_frames"}}
        On failure:
          {"ok": False, "error": "<reason>"}
    """
    state = load_state(run_root)
    svo = state.get("stages", {}).get("svo_extract", {})
    seg = state.get("stages", {}).get("segmentation", {})
    mr = state.get("stages", {}).get("mesh_recover", {})
    sd = state.get("stages", {}).get("stereo_depth", {})

    for stage_name, stage_val in (
        ("svo_extract", svo), ("segmentation", seg),
        ("mesh_recover", mr), ("stereo_depth", sd),
    ):
        if not stage_val.get("ok"):
            err = f"stages.{stage_name} has not completed successfully; run it first"
            record_stage(run_root, "mesh_scale", ok=False, error=err)
            return json.dumps({"ok": False, "error": err})

    primary_seq = svo.get("outputs", {}).get("sequence_dir")
    mesh_dir    = mr.get("outputs", {}).get("mesh_dir")
    mask_dir    = seg.get("outputs", {}).get("output_dir")
    if not all([primary_seq, mesh_dir, mask_dir]):
        err = (
            "missing one of primary sequence_dir / mesh_dir / mask_dir in upstream "
            f"stage outputs (got {primary_seq!r}, {mesh_dir!r}, {mask_dir!r})"
        )
        record_stage(run_root, "mesh_scale", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    # Per v1 mesh_scale_node: only score objects whose mesh_recover succeeded.
    successful = [
        o["object_name"] for o in mr.get("outputs", {}).get("per_object", [])
        if o.get("success")
    ]
    if not successful:
        err = "no objects with successful mesh_recover; cannot score scale for anything"
        record_stage(run_root, "mesh_scale", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    # Dual-view (PR-4): partition successful objects by their winning camera.
    # segment_controller wrote mask_camera_id_by_object in PR-3; for old single-
    # camera episodes this is absent → everything stays on primary.
    primary_id   = state.get("primary_camera_id", "") or ""
    secondary_id = state.get("secondary_camera_id", "") or ""
    mask_cam_by  = state.get("mask_camera_id_by_object") or {}
    primary_objs:   list[str] = []
    secondary_objs: list[str] = []
    for name in successful:
        cid = mask_cam_by.get(name) or ""
        if cid and cid == secondary_id and cid != primary_id:
            secondary_objs.append(name)
        else:
            primary_objs.append(name)

    # Lazy-trigger secondary stereo_depth — mesh_scale needs depth maps for
    # every obj we'll score; secondary depth wasn't computed at end of PR-3's
    # bootstrap (only RGB + mp4 were). Skip the call when no secondary objs
    # so single-cam episodes pay nothing.
    secondary_seq = ""
    if secondary_objs:
        sd_sec = state.get("stages", {}).get("stereo_depth_secondary", {})
        if not sd_sec.get("ok"):
            print(f"[mesh_scale] {len(secondary_objs)} obj(s) on secondary "
                  f"({secondary_objs}) — running stereo_depth.run_for_secondary ...")
            r = _stereo_skill.run_for_secondary(run_root)
            if not r.get("ok"):
                err = (f"could not compute secondary depth for "
                       f"{secondary_objs}: {r.get('error')}")
                record_stage(run_root, "mesh_scale", ok=False, error=err)
                return json.dumps({"ok": False, "error": err})
            # Re-read state to pick up the new stage record's sequence_dir.
            state = load_state(run_root)
        svo_sec = state.get("stages", {}).get("svo_extract_secondary", {})
        secondary_seq = svo_sec.get("outputs", {}).get("sequence_dir") or ""
        if not secondary_seq:
            err = ("stages.svo_extract_secondary.outputs.sequence_dir missing; "
                   "cannot resolve depth for secondary objs")
            record_stage(run_root, "mesh_scale", ok=False, error=err)
            return json.dumps({"ok": False, "error": err})

    # Multi-word object names (e.g. "snack package", "pot lid") collide with
    # SAM3's underscore-sanitised dir names. Sanitise for path lookup; map
    # back to canonical names in the per_object output.
    import re as _re
    _SANITIZE = _re.compile(r"[^a-zA-Z0-9._-]+")
    def _sanitize(text: str) -> str:
        return _SANITIZE.sub("_", text).strip("_") or "unnamed"

    pfb_raw = state.get("object_keyframes") or state.get("prompt_frame_index_by_object") or {}
    global_kf = state.get("prompt_frame_index")

    def _per_object_records(raw_names: list[str], sequence_dir: str) -> list[dict] | tuple[str, dict]:
        """Run the toolkit for one camera group, return per-obj records.

        Returns either a list of report dicts on success, or a (err, raw_record)
        tuple on failure where raw_record is the partial info for state.
        """
        if not raw_names:
            return []
        san = [_sanitize(n) for n in raw_names]
        raw_by_san = {s: r for s, r in zip(san, raw_names)}
        keyframe_by_san: dict[str, int] = {}
        for raw_name in raw_names:
            kf = pfb_raw.get(raw_name, global_kf)
            if kf is not None:
                keyframe_by_san[_sanitize(raw_name)] = int(kf)
        inp = _toolkit.MeshScaleInput(
            sequence_dir=sequence_dir,
            mesh_dir=mesh_dir,
            mask_dir=mask_dir,
            objects=san,
            keyframe_by_object=keyframe_by_san,
        )
        report = _toolkit.run(inp, log_dir=str(Path(run_root) / "logs"))
        if not report.success:
            return ("error", {"error": report.error, "raw_names": raw_names})
        return [
            {
                "object_name":       raw_by_san.get(o.object_name, o.object_name),
                "median_scale":      o.median_scale,
                "std_scale":         o.std_scale,
                "num_frames_used":   o.num_frames_used,
                "total_frames":      o.total_frames,
                "keyframe_anchored": o.keyframe_anchored,
                "meta_dir":          o.meta_dir,
                "camera_id":         (secondary_id if sequence_dir == secondary_seq else primary_id),
            }
            for o in report.objects
        ]

    per_object: list[dict] = []
    for group_name, raw_names, seq in (
        ("primary",   primary_objs,   primary_seq),
        ("secondary", secondary_objs, secondary_seq),
    ):
        if not raw_names:
            continue
        print(f"[mesh_scale] running {group_name} group ({len(raw_names)} obj(s)): {raw_names}")
        result = _per_object_records(raw_names, seq)
        if isinstance(result, tuple) and result[0] == "error":
            err = f"mesh_scale {group_name} failed: {result[1]['error']}"
            record_stage(run_root, "mesh_scale", ok=False, error=err)
            return json.dumps({"ok": False, "error": err})
        per_object.extend(result)

    n_with_frames = sum(1 for o in per_object if o["num_frames_used"] > 0)
    n_total_frames = sum(o["num_frames_used"] for o in per_object)
    outputs = {"per_object": per_object}
    stats = {
        "n_objects":      len(per_object),
        "n_with_frames":  n_with_frames,
        "n_total_frames": n_total_frames,
        "n_on_secondary": len(secondary_objs),
    }
    record_stage(run_root, "mesh_scale", ok=True, outputs=outputs, stats=stats)

    return json.dumps({"ok": True, "outputs": outputs, "stats": stats})
