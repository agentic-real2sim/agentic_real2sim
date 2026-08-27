"""stereo_depth skill — run FoundationStereo on rectified frame pairs.

Reads from state.json:
  - stages.svo_extract.outputs.sequence_dir            (primary; required)
  - stages.svo_extract_secondary.outputs.sequence_dir  (PR-3 lazy via
                                                        ``run_for_secondary``)
  - get_pointcloud      (optional; default False — .ply is debug-only)
  - stereo_valid_iters  (optional; default 22; GRU refinement iterations)

Note: max_frames / frame_step are applied ONCE, by svo_extract (which also
re-indexes the kept frames to a consecutive sequence). stereo_depth processes
ALL frames svo_extract produced — it does NOT re-read those knobs, otherwise
frame_step would double-decimate and leave downstream stages short on depth.

Writes to disk (under each call's sequence_dir):
  - <sequence_dir>/depth/frame_NNNNNN_depth.npy
  - <sequence_dir>/vis/frame_NNNNNN_vis.png      (deleted as the stage returns)
  - <sequence_dir>/pointcloud/frame_NNNNNN.ply   (when get_pointcloud is True)
  - logs under <run_root>/logs/stereo_depth[_<role>].{cmd,...}

Writes to state.json:
  - state["stages"]["stereo_depth"]           = {ok, outputs, stats, error}   (primary)
  - state["stages"]["stereo_depth_secondary"] = same shape, set by PR-3 only

Subprocess env: ``qianjun_foundation_stereo``.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from ar2s.agents_visual._toolkit import stereo_depth as _toolkit
from ar2s.agents_visual.state import load_state, record_stage

# Pipeline defaults (overridable via VisualInput -> state). These differ from
# FoundationStereo's own script defaults to trade negligible quality for speed:
#   - get_pointcloud OFF: the .ply clouds are debug-only (excluded from sync,
#     not read by mesh_scale/pose_tracking, which build their own from depth).
#     Skipping them also drops the per-frame open3d radius-outlier denoise.
#   - valid_iters 22 (vs script default 32): near-lossless, ~linear speedup.
_DEFAULT_GET_POINTCLOUD = False
_DEFAULT_VALID_ITERS = 22


def _stereo_knobs(state: dict) -> dict:
    return {
        "get_pointcloud": state.get("get_pointcloud", _DEFAULT_GET_POINTCLOUD),
        "valid_iters":    state.get("stereo_valid_iters", _DEFAULT_VALID_ITERS),
        # Optional cap for the HEAVY per-frame model ONLY (validation shortcut).
        # svo_extract still owns real decimation (max_frames/frame_step); with
        # frame_step=1 this simply processes the first N extracted frames, so no
        # double-decimation. None = all frames svo_extract produced.
        "heavy_max_frames": state.get("heavy_max_frames"),
    }


def _prune_vis_dir(vis_dir: str) -> int:
    """Delete FoundationStereo's disparity-visualisation PNGs. Returns the count.

    One disparity-colormap PNG per frame (~1 MB each). Nothing in the pipeline
    reads them and they are reconstructable from the depth this stage keeps
    plus K, so they are never worth carrying — and dropping them here rather
    than in pipeline_cleanup is what keeps them off the PEAK disk footprint,
    which is what binds when a SLURM array runs many episodes concurrently.

    Proper fix lives upstream in ``run_stereo_on_frames.py`` (a ``--save_vis 0``
    flag next to the existing ``--get_pc``), which would avoid encoding them at
    all. That script is in the FoundationStereo_droid fork rather than this
    repo, so until the flag lands we delete instead of not-writing.
    """
    d = Path(vis_dir)
    if not d.is_dir():
        return 0
    n = 0
    for p in d.glob("*_vis.png"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    try:
        d.rmdir()
    except OSError:
        pass
    return n


def _run_stereo_depth(
    run_root: str,
    *,
    sequence_dir: str,
    stage_key: str,
    log_subdir: str,
    get_pointcloud: bool,
    valid_iters: int,
    heavy_max_frames: int | None = None,
) -> dict:
    """Camera-aware core. Runs FoundationStereo over ``sequence_dir`` and
    records under ``stage_key``. svo_extract is the sole owner of real frame
    decimation, so we keep frame_step=1 here. ``heavy_max_frames`` is an
    OPTIONAL validation cap that limits FoundationStereo to the first N frames
    only (None = process ALL frames svo_extract produced).
    """
    log_dir = (str(Path(run_root) / "logs" / log_subdir)
               if log_subdir else str(Path(run_root) / "logs"))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    inp = _toolkit.StereoDepthInput(
        sequence_dir=sequence_dir,
        max_frames=heavy_max_frames,
        frame_step=1,
        get_pointcloud=get_pointcloud,
        valid_iters=valid_iters,
    )
    report = _toolkit.run(inp, log_dir=log_dir)

    if not report.success:
        record_stage(run_root, stage_key, ok=False, error=report.error)
        return {"ok": False, "error": report.error}

    n_vis_pruned = _prune_vis_dir(report.vis_dir) if report.vis_dir else 0

    outputs = {
        "depth_dir":      report.depth_dir,
        "vis_dir":        "",
        "pointcloud_dir": report.pointcloud_dir,
    }
    stats = {
        "num_frames_processed": report.num_frames_processed,
        "num_vis_pruned": n_vis_pruned,
    }
    record_stage(run_root, stage_key, ok=True, outputs=outputs, stats=stats)
    return {"ok": True, "outputs": outputs, "stats": stats}


@tool
def stereo_depth(run_root: str) -> str:
    """Compute primary-view depth maps + (optional) point clouds.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``stages.svo_extract.outputs.sequence_dir`` from a previous svo_extract run.

    Returns:
        JSON string:
          {"ok": True,
           "outputs": {"depth_dir", "vis_dir", "pointcloud_dir"},
           "stats":   {"num_frames_processed"}}
        On failure:
          {"ok": False, "error": "<reason>"}
    """
    state = load_state(run_root)
    svo = state.get("stages", {}).get("svo_extract", {})
    if not svo.get("ok"):
        err = "stages.svo_extract has not completed successfully; run svo_extract first"
        record_stage(run_root, "stereo_depth", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    sequence_dir = svo.get("outputs", {}).get("sequence_dir")
    if not sequence_dir:
        err = "stages.svo_extract.outputs.sequence_dir missing"
        record_stage(run_root, "stereo_depth", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    result = _run_stereo_depth(
        run_root,
        sequence_dir=sequence_dir,
        stage_key="stereo_depth",
        log_subdir="",
        **_stereo_knobs(state),
    )
    return json.dumps(result)


def run_for_secondary(run_root: str) -> dict:
    """Compute the secondary view's depth maps (PR-3+ lazy callers).

    Called by PR-4's ``depth_for_fallback_objs`` stage when at least one
    object's segment fell back to secondary. Requires
    ``stages.svo_extract_secondary`` to be ok. Records under
    ``stages.stereo_depth_secondary``.
    """
    state = load_state(run_root)
    svo_sec = state.get("stages", {}).get("svo_extract_secondary", {})
    if not svo_sec.get("ok"):
        return {
            "ok": False,
            "error": "stages.svo_extract_secondary not ok "
                     "(run svo_extract.run_for_secondary first)",
        }
    sequence_dir = svo_sec.get("outputs", {}).get("sequence_dir")
    if not sequence_dir:
        return {
            "ok": False,
            "error": "stages.svo_extract_secondary.outputs.sequence_dir missing",
        }
    return _run_stereo_depth(
        run_root,
        sequence_dir=sequence_dir,
        stage_key="stereo_depth_secondary",
        log_subdir="secondary",
        **_stereo_knobs(state),
    )
