"""pose_tracking skill — FoundationPose 6-DoF tracking, one run per recovered mesh.

Per-object loop (mirrors v1's pose_tracking_node). For every object whose
mesh_recover succeeded, invokes ``run_demo_foundationstereo_video.py`` in the
qianjun_foundationpose env via _toolkit.pose_tracking. Each per-object call
writes its outputs under ``<run_root>/poses/<object>/{ob_in_cam,track_vis}``
plus a packed ``poses.npz``.

Reads from state.json:
  - discovered_objects                           (list[str]; "robot" filtered)
  - stages.svo_extract.outputs.sequence_dir      (K.txt + frames + depth/)
  - stages.segmentation.outputs.output_dir       (SAM3 masks root)
  - stages.mesh_recover.outputs.per_object       (only success=True objects
                                                  are tracked — skipping
                                                  failures matches v1 logic)
  - init_frame                                   (optional override; -1 = auto-pick.
                                                  Future tracking_critic retry hook
                                                  was planned to set this; that
                                                  retry loop is deferred, so in
                                                  practice this is always -1.)

Writes to disk:
  - <run_root>/poses/<object>/ob_in_cam/frame_NNNNNN.txt   (toolkit script)
  - <run_root>/poses/<object>/track_vis/<id>.png           (toolkit script)
  - <run_root>/poses/<object>/poses.npz                    (toolkit packs;
                                                            see ar2s.droid_sim.pose_store)
  - logs under <run_root>/logs/pose_tracking_<object>.{cmd,stdout,stderr,returncode}

Writes to state.json:
  - state["stages"]["pose_tracking"] = {ok, outputs, stats, error}
  - ok=True iff at least one object's tracking succeeded; per-object detail
    in outputs.per_object so resolve can broadcast pose_source_path correctly.

Subprocess env: ``qianjun_foundationpose``.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

import numpy as np

from ar2s.agents_visual._toolkit import pose_tracking as _toolkit
from ar2s.agents_visual.skills import stereo_depth as _stereo_skill
from ar2s.agents_visual.state import load_state, record_stage


def _rebase_secondary_poses(pose_source: str, primary_cam_mat: np.ndarray,
                            secondary_cam_mat: np.ndarray) -> int:
    """In-place transform the packed poses from secondary to primary frame.

    ``pose_source`` is the toolkit's reported pose path (``<obj>/poses.npz``),
    read and rewritten through ``ar2s.droid_sim.pose_store`` so it works on
    either on-disk layout.

    Previously this rewrote ``<pose_dir>/ob_in_cam/*.txt``, but ``pose_dir``
    is the packed-pose path and ``ob_in_cam`` is its SIBLING — so the glob
    never matched, the rebase silently did nothing, and any object tracked on
    the secondary view kept secondary-frame poses. No shipped run hit it
    (every object had won on primary), but it would have mislocated objects
    the moment segment_controller's dual-view fallback fired.

    Dual-view: FoundationPose's per-frame ``ob_in_cam`` files express each
    object pose in the tracked camera's frame (T_cam_from_obj). When secondary
    was the tracking view, downstream sim (which uses a single primary cam_mat
    for world-frame lifting) would get a wrong answer. We pre-rebase here so
    all on-disk poses are in PRIMARY camera frame.

    Convention check (decided 2026-06-05): cameras_extrinsics.npz's
    ``cam_mat_<serial>`` is **w2c** (T_cam_from_world), NOT c2w as the
    raw_episode.py docstring previously claimed. Verified against
    droid_sim/scene/robot_scene_sim.py:111-127, which loads cam_mat as
    ``xforms_bot2cam`` and computes ``cam2world = inv(cam_mat)`` before
    multiplying with T_obj_in_cam — only sensible if cam_mat is w2c.

    With cam_mat = w2c, the rebase math is:

        T_world_from_obj  = T_world_from_sec_cam @ T_obj_in_sec_cam
                          = inv(cam_mat_sec)     @ T_obj_in_sec_cam
        T_obj_in_pri_cam  = T_pri_cam_from_world @ T_world_from_obj
                          = cam_mat_pri          @ inv(cam_mat_sec) @ T_obj_in_sec_cam

    So the multiplicative factor is ``cam_mat_pri @ inv(cam_mat_sec)``.

    Returns the number of poses rebased. Idempotency is the caller's job —
    re-running pose_tracking blasts the pose source before re-collecting.
    """
    from ar2s.droid_sim import pose_store

    poses = pose_store.load_poses(pose_source)
    if not poses:
        return 0
    transform = primary_cam_mat @ np.linalg.inv(secondary_cam_mat)
    pose_store.save_poses(
        pose_source, {idx: transform @ m for idx, m in poses.items()},
    )
    return len(poses)


@tool
def pose_tracking(run_root: str) -> str:
    """Track each recovered mesh through the rectified video via FoundationPose.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``stages.svo_extract``, ``stages.segmentation``, and
            ``stages.mesh_recover`` to have completed.

    Returns:
        JSON string:
          {"ok": True (iff at least one object tracked),
           "outputs": {"poses_dir",
                       "per_object": [{
                           "object_name", "success",
                           "pose_dir"?, "track_vis_dir"?, "first_frame_rgb_path"?,
                           "error"?}, ...]},
           "stats":   {"n_objects", "n_success", "init_frame"}}
        On unrecoverable failure (missing inputs):
          {"ok": False, "error": "<reason>"}
    """
    state = load_state(run_root)
    svo = state.get("stages", {}).get("svo_extract", {})
    seg = state.get("stages", {}).get("segmentation", {})
    mr = state.get("stages", {}).get("mesh_recover", {})

    for stage_name, stage_val in (("svo_extract", svo), ("segmentation", seg), ("mesh_recover", mr)):
        if not stage_val.get("ok"):
            err = f"stages.{stage_name} has not completed; run it first"
            record_stage(run_root, "pose_tracking", ok=False, error=err)
            return json.dumps({"ok": False, "error": err})

    primary_seq = svo.get("outputs", {}).get("sequence_dir")
    mask_root   = seg.get("outputs", {}).get("output_dir")
    if not primary_seq or not mask_root:
        err = "stages.svo_extract.outputs.sequence_dir or stages.segmentation.outputs.output_dir missing"
        record_stage(run_root, "pose_tracking", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    successful = [
        o for o in mr.get("outputs", {}).get("per_object", [])
        if o.get("success") and o.get("glb_path")
    ]
    if not successful:
        err = "no objects with successful mesh_recover; cannot track anything"
        record_stage(run_root, "pose_tracking", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    # Dual-view (PR-4): identify which objs are on secondary so we can pull
    # depth/frames/K from the right sequence_dir per object. mesh_scale.run
    # already lazy-triggered stages.stereo_depth_secondary; verify it
    # actually ran (idempotent: if mesh_scale was skipped for some reason,
    # auto-run it here too).
    primary_id   = state.get("primary_camera_id", "") or ""
    secondary_id = state.get("secondary_camera_id", "") or ""
    mask_cam_by  = state.get("mask_camera_id_by_object") or {}
    has_secondary_obj = any(
        (mask_cam_by.get(o["object_name"]) or "") == secondary_id and secondary_id
        for o in successful
    )
    secondary_seq = ""
    if has_secondary_obj:
        if not state.get("stages", {}).get("stereo_depth_secondary", {}).get("ok"):
            print("[pose_tracking] secondary objs present but stereo_depth_secondary "
                  "missing — running stereo_depth.run_for_secondary ...")
            r = _stereo_skill.run_for_secondary(run_root)
            if not r.get("ok"):
                err = f"could not compute secondary depth: {r.get('error')}"
                record_stage(run_root, "pose_tracking", ok=False, error=err)
                return json.dumps({"ok": False, "error": err})
            state = load_state(run_root)
        svo_sec = state.get("stages", {}).get("svo_extract_secondary", {})
        secondary_seq = svo_sec.get("outputs", {}).get("sequence_dir") or ""
        if not secondary_seq:
            err = "stages.svo_extract_secondary.outputs.sequence_dir missing"
            record_stage(run_root, "pose_tracking", ok=False, error=err)
            return json.dumps({"ok": False, "error": err})

    def _seq_for(obj_name: str) -> str:
        cid = mask_cam_by.get(obj_name) or ""
        if cid and cid == secondary_id and cid != primary_id and secondary_seq:
            return secondary_seq
        return primary_seq

    # Pre-load both cam_mats — needed for the secondary-frame rebase below.
    primary_cam_mat = None
    secondary_cam_mat = None
    if has_secondary_obj:
        extr_path = state.get("cameras_extrinsics_path") or ""
        if extr_path and Path(extr_path).exists():
            extr = np.load(extr_path)
            pkey = f"cam_mat_{primary_id}"
            skey = f"cam_mat_{secondary_id}"
            if pkey in extr.files and skey in extr.files:
                primary_cam_mat = np.asarray(extr[pkey], dtype=np.float64)
                secondary_cam_mat = np.asarray(extr[skey], dtype=np.float64)
        if primary_cam_mat is None or secondary_cam_mat is None:
            err = (f"cameras_extrinsics.npz at {extr_path!r} missing "
                   f"cam_mat_{primary_id} or cam_mat_{secondary_id}; cannot "
                   f"rebase secondary-view poses to primary frame")
            record_stage(run_root, "pose_tracking", ok=False, error=err)
            return json.dumps({"ok": False, "error": err})

    init_frame = int(state.get("init_frame", -1))
    # Optional validation cap: track only the first N frames (matches the
    # heavy_max_frames depth cap so every tracked frame has a depth map).
    # end_frame is inclusive, so N frames -> end_frame = N - 1. -1 = to end.
    heavy_cap = state.get("heavy_max_frames")
    end_frame = (int(heavy_cap) - 1) if heavy_cap else -1
    # Registration memory scales with full-frame pixel count; wide sensors
    # (AIRBOT ZED 2i, 2208x1242) OOM a 32 GB card where DROID's 1280x720 fits.
    shorter_side = state.get("pose_shorter_side") or None
    output_dir = str(Path(run_root) / "poses")
    log_dir = str(Path(run_root) / "logs")

    # FoundationPose's reader reconstructs the mask path as
    # ``<mask_root>/<object_name>/masks/NN.png``. SAM3's segmentation toolkit
    # writes those dirs under the SANITISED name (spaces -> '_', so
    # "snack package" lives at ``segmentation/snack_package/``). Pass the
    # SANITISED form to the pose-tracking script so its mask lookup hits the
    # right dir; we still surface the canonical (with-space) name in the
    # per_object report so downstream consumers key by the original name.
    from ar2s.droid_sim._util import sanitize_name as _sanitize

    per_object: list[dict] = []
    for obj in successful:
        name = obj["object_name"]
        san_name = _sanitize(name)
        glb = obj["glb_path"]
        sequence_dir = _seq_for(name)
        camera_id    = (secondary_id if sequence_dir == secondary_seq else primary_id)
        inp = _toolkit.PoseTrackingInput(
            sequence_dir=sequence_dir,
            mask_root=mask_root,
            mesh_file=glb,
            object_name=san_name,                  # path-safe; matches SAM3 dir
            output_dir=output_dir,
            init_frame=init_frame,
            end_frame=end_frame,
            shorter_side=shorter_side,
        )
        report = _toolkit.run(inp, log_dir=log_dir)
        if report.success:
            rebased = 0
            if camera_id == secondary_id and camera_id != primary_id:
                rebased = _rebase_secondary_poses(
                    report.pose_dir, primary_cam_mat, secondary_cam_mat,
                )
                print(f"[pose_tracking] {name!r}: rebased {rebased} pose files "
                      f"secondary→primary frame")
            per_object.append({
                "object_name":          name,      # canonical (with-space) name
                "success":              True,
                "pose_dir":             report.pose_dir,
                "track_vis_dir":        report.track_vis_dir,
                "first_frame_rgb_path": report.first_frame_rgb_path,
                "camera_id":            camera_id,
                "frame_rebased":        rebased,
            })
        else:
            per_object.append({
                "object_name": name,
                "success":     False,
                "error":       report.error,
                "camera_id":   camera_id,
            })

    n_success = sum(1 for o in per_object if o["success"])
    outputs = {
        "poses_dir":  output_dir,
        "per_object": per_object,
    }
    stats = {
        "n_objects":  len(successful),
        "n_success":  n_success,
        "init_frame": init_frame,
    }
    record_stage(run_root, "pose_tracking", ok=(n_success > 0), outputs=outputs, stats=stats)

    return json.dumps({
        "ok":      n_success > 0,
        "outputs": outputs,
        "stats":   stats,
    })
