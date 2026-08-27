"""mesh_recover skill — sam-3d-objects mesh recovery (one .glb per non-robot object).

Builds a list of per-object jobs (each with its own keyframe RGB + mask) from
state.discovered_objects + state.stages.segmentation, then makes ONE batched
call into the _toolkit subprocess wrapper, which invokes
ar2s/agents_visual/_toolkit/scripts/run_sam3d_batch.py inside the
`qianjun_sam3d` conda env (LIDRA_SKIP_INIT auto-injected by _runners.py).
The batch wrapper loads SAM3D (~13 GB) ONCE for all objects.

Dual-view (PR-4): each obj's RGB frame is sourced from THE VIEW that produced
its accepted mask (state.mask_camera_id_by_object[obj] → primary if absent
or matching primary_camera_id, else secondary). The single batched sam3d
call still loads the model once; the per-job paths simply point at different
seq subdirs.

Reads from state.json:
  - discovered_objects                          (list[str], "robot" filtered out)
  - object_keyframes or prompt_frame_index_by_object     (per-object SAM3D frames)
  - prompt_frame_index                                  (fallback frame)
  - stages.svo_extract.outputs.left_frames_dir            (primary rgb source)
  - stages.svo_extract_secondary.outputs.left_frames_dir  (secondary rgb source, PR-3)
  - stages.segmentation.outputs.objects                   (per-text masks_dir lookup)
  - mask_camera_id_by_object                              (PR-3; routes per-obj rgb)

Writes to disk:
  - <run_root>/meshes/<object>.glb
  - <run_root>/meshes/<object>.glb.json   (sidecar from run_sam3d_batch.py)
  - logs under <run_root>/logs/mesh_recover.{cmd,stdout,stderr,returncode}
  - <run_root>/logs/mesh_recover_jobs.json (batched job manifest)

Writes to state.json:
  - state["stages"]["mesh_recover"] = {
        ok, outputs.{mesh_dir, per_object: [...]}, stats, error
    }
  - `ok` is True iff AT LEAST ONE object succeeded; per-object failures live
    in outputs.per_object so downstream stages (mesh_scale) can filter.

Subprocess env: ``qianjun_sam3d`` (auto-LIDRA_SKIP_INIT=1 via _runners.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from ar2s.agents_visual._toolkit import mesh_recover as _toolkit
from ar2s.agents_visual.state import load_state, record_stage


@tool
def mesh_recover(run_root: str) -> str:
    """Recover one .glb mesh per non-robot object via sam-3d-objects.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``stages.svo_extract`` and ``stages.segmentation`` to have
            completed, plus ``discovered_objects`` set by object_discovery.

    Returns:
        JSON string:
          {"ok": True (iff at least one object succeeded),
           "outputs": {"mesh_dir", "per_object": [{
                "object_name", "success", "glb_path"?, "error"?}, ...]},
           "stats":   {"n_objects", "n_success", "prompt_frame_index"}}
        On unrecoverable failure (missing inputs):
          {"ok": False, "error": "<reason>"}
    """
    state = load_state(run_root)
    svo = state.get("stages", {}).get("svo_extract", {})
    seg = state.get("stages", {}).get("segmentation", {})
    if not svo.get("ok"):
        err = "stages.svo_extract has not completed; run svo_extract first"
        record_stage(run_root, "mesh_recover", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})
    if not seg.get("ok"):
        err = "stages.segmentation has not completed; run segment first"
        record_stage(run_root, "mesh_recover", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    discovered = state.get("discovered_objects") or []
    non_robot = [o for o in discovered if o and o != "robot"]
    if not non_robot:
        err = (
            "no non-robot objects in state.discovered_objects; nothing to mesh. "
            "Run object_discovery and confirm at least one scene object found."
        )
        record_stage(run_root, "mesh_recover", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    prompt_idx = int(state.get("prompt_frame_index", 0))
    # Phase-B per-object: each object's RGB + mask is sampled at its own keyframe.
    pfb = state.get("object_keyframes") or state.get("prompt_frame_index_by_object") or {}
    left_dir = svo.get("outputs", {}).get("left_frames_dir")
    if not left_dir:
        err = "stages.svo_extract.outputs.left_frames_dir missing"
        record_stage(run_root, "mesh_recover", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    # Dual-view (PR-4): if any obj was accepted on secondary view, its RGB
    # frame must come from secondary's left_frames_dir, not primary's.
    primary_id   = state.get("primary_camera_id", "") or ""
    mask_cam_by  = state.get("mask_camera_id_by_object") or {}
    svo_sec_left = ""
    secondary_mask_cids = sorted({cid for cid in mask_cam_by.values() if cid and cid != primary_id})
    if secondary_mask_cids:
        svo_sec = state.get("stages", {}).get("svo_extract_secondary", {})
        svo_sec_left = svo_sec.get("outputs", {}).get("left_frames_dir") if svo_sec.get("ok") else ""
        if not svo_sec_left:
            err = (
                "objects accepted on secondary camera(s) "
                f"{secondary_mask_cids}, but stages.svo_extract_secondary.outputs.left_frames_dir "
                "is missing; cannot recover meshes from the correct RGB view"
            )
            record_stage(run_root, "mesh_recover", ok=False, error=err)
            return json.dumps({"ok": False, "error": err})

    def _left_dir_for(obj_name: str) -> str:
        """Return the frames dir for ``obj_name``'s winning camera."""
        cid = mask_cam_by.get(obj_name) or ""
        if cid and cid != primary_id:
            return svo_sec_left
        return left_dir

    # Map sam3 text prompt → masks_dir from segmentation outputs.
    masks_dirs: dict[str, str] = {}
    for obj in seg.get("outputs", {}).get("objects", []):
        text = obj.get("text") or ""
        masks_dir = obj.get("masks_dir") or ""
        if text and masks_dir:
            masks_dirs[text] = masks_dir

    mesh_dir = Path(run_root) / "meshes"
    log_dir = Path(run_root) / "logs"

    # Multi-word discovered names (e.g. "snack package", "pot lid") become
    # path-unsafe when used as filenames; the per_object record still carries
    # the canonical name so downstream keying is unchanged — only the on-disk
    # .glb filename is sanitised. Canonical impl: ar2s.droid_sim._util.sanitize_name
    from ar2s.droid_sim._util import sanitize_name as _sanitize

    # Build the batched job list. Objects with no segmentation mask are
    # short-circuited here (recorded as failures) so they don't cost a slot
    # in the SAM3D subprocess; the surviving jobs go in one shot.
    per_object: list[dict] = []          # final report entries (success + early failures)
    job_meta: list[dict] = []            # frame index per surviving object (for the report)
    jobs: list[_toolkit.MeshRecoverJob] = []
    for name in non_robot:
        masks_dir = masks_dirs.get(name)
        if not masks_dir:
            per_object.append({
                "object_name": name,
                "success":     False,
                "error":       f"no segmentation masks_dir for text='{name}'",
            })
            continue

        # Use this object's own per-object frame (Phase B); fall back to global.
        obj_frame = int(pfb.get(name, prompt_idx))
        obj_left_dir = _left_dir_for(name)
        rgb_path = f"{obj_left_dir}/frame_{obj_frame:06d}.png"
        mask_path = f"{masks_dir}/{obj_frame:06d}.png"
        glb_path = str(mesh_dir / f"{_sanitize(name)}.glb")
        jobs.append(_toolkit.MeshRecoverJob(
            object_name=name,
            rgb_frame_path=rgb_path,
            mask_path=mask_path,
            output_glb_path=glb_path,
        ))
        job_meta.append({"object_name": name, "frame": obj_frame})

    if jobs:
        report = _toolkit.run(
            _toolkit.MeshRecoverInput(jobs=jobs),
            log_dir=str(log_dir),
        )
        by_name = {o.object_name: o for o in report.objects}
        for meta in job_meta:
            name = meta["object_name"]
            obj_info = by_name.get(name)
            if obj_info is not None and obj_info.success:
                per_object.append({
                    "object_name": name,
                    "success":     True,
                    "glb_path":    obj_info.glb_path,
                    "frame":       meta["frame"],
                })
            else:
                per_object.append({
                    "object_name": name,
                    "success":     False,
                    "error":       obj_info.error if obj_info else "no result from batch",
                    "frame":       meta["frame"],
                })

    n_success = sum(1 for o in per_object if o["success"])
    outputs = {
        "mesh_dir":   str(mesh_dir),
        "per_object": per_object,
    }
    stats = {
        "n_objects":          len(non_robot),
        "n_success":          n_success,
        "prompt_frame_index": prompt_idx,
    }
    record_stage(run_root, "mesh_recover", ok=(n_success > 0), outputs=outputs, stats=stats)

    return json.dumps({
        "ok":      n_success > 0,
        "outputs": outputs,
        "stats":   stats,
    })
