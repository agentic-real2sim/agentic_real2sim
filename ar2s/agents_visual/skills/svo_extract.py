"""svo_extract skill — split exported rectified stereo MP4 frames.

Reads from state.json:
  - stereo_stream_path                 (primary rectified .mp4; required)
  - stereo_intrinsics_path             (primary SDK/SVO K sidecar; required)
  - secondary_stereo_stream_path       (optional fallback rectified .mp4)
  - secondary_stereo_intrinsics_path   (optional fallback K sidecar)
  - primary_camera_id    (optional; non-empty => writes under seq/<id>/)
  - secondary_camera_id  (optional; PR-3 uses for secondary subdir)
  - max_frames           (optional cap)
  - frame_step           (optional; default 1)

Writes to disk:
  - Single-camera builds (no primary_camera_id):
      <run_root>/seq/K.txt
      <run_root>/seq/frames/{left,right}/frame_NNNNNN.png
  - Dual-view builds (primary_camera_id set):
      <run_root>/seq/<primary_id>/K.txt
      <run_root>/seq/<primary_id>/frames/{left,right}/frame_NNNNNN.png
  - Logs: <run_root>/logs/svo_extract[_<role>].{cmd,stdout,stderr,returncode}

Writes to state.json:
  - state["stages"]["svo_extract"]            = {ok, outputs, stats, error}   (primary)
  - state["stages"]["svo_extract_secondary"]  = same shape, set by PR-3 only

Subprocess env: ``qianjun_foundation_stereo`` (auto-injected XFORMERS_DISABLED=1
by _toolkit/_runners.py for sm_120 fallback).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from langchain_core.tools import tool

from ar2s.agents_visual._toolkit import svo_extract as _toolkit
from ar2s.agents_visual.state import load_state, record_stage


def _serial_from_extrinsics(
    extr_path: str | None,
    *,
    prefer_serial: str = "",
) -> str | None:
    """Return the ZED serial from cameras_extrinsics.npz, else None.

    Dual-view (PR-2+): with multiple ``cam_mat_<serial>`` keys, ``prefer_serial``
    disambiguates which one to return (typically the camera being extracted
    right now). Without a preference, the single-key case still returns the
    sole serial; multi-key with no preference returns None and lets the
    toolkit fall back to filename auto-detection.
    """
    if not extr_path or not Path(extr_path).exists():
        return None
    try:
        keys = [k for k in np.load(extr_path).files if k.startswith("cam_mat_")]
    except Exception:
        return None
    serials = [k[len("cam_mat_"):] for k in keys]
    if prefer_serial and prefer_serial in serials:
        return prefer_serial
    if len(serials) == 1:
        return serials[0]
    return None


def _seq_subdir(run_root: str, camera_id: str) -> Path:
    """Resolve where this camera's seq artifacts live.

    Empty ``camera_id`` (single-camera build / old episodes) returns the
    legacy ``<run_root>/seq`` directly so all existing state.stages.* paths
    keep resolving. Non-empty namespaces under ``seq/<camera_id>/`` so
    primary + secondary frames coexist without collisions (PR-3 wiring).
    """
    root = Path(run_root) / "seq"
    return root / camera_id if camera_id else root


def _run_svo_extract(
    run_root: str,
    *,
    camera_id: str,
    stereo_stream_path: str,
    intrinsic_file: str,
    stage_key: str,
    log_subdir: str,
) -> dict:
    """Camera-aware core. Used by the @tool wrapper for primary, and by
    PR-3's segment_controller for secondary. Returns the JSON-ready dict
    AND records it under ``stages.<stage_key>`` so the call site can stay
    thin.

    ``camera_id`` may be empty (legacy single-camera path → writes under
    ``seq/``). When non-empty, frames + K.txt live under ``seq/<id>/`` so
    multiple cameras can share the same run_root without overwriting one
    another's frames.
    """
    if not stereo_stream_path:
        err = f"stereo stream path is empty for stage_key={stage_key!r}"
        record_stage(run_root, stage_key, ok=False, error=err)
        return {"ok": False, "error": err}
    if not intrinsic_file:
        err = f"intrinsic_file is empty for stage_key={stage_key!r}"
        record_stage(run_root, stage_key, ok=False, error=err)
        return {"ok": False, "error": err}

    state = load_state(run_root)
    seq_dir = str(_seq_subdir(run_root, camera_id))
    log_dir = str(Path(run_root) / "logs" / log_subdir) if log_subdir else str(Path(run_root) / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    inp = _toolkit.SVOExtractInput(
        source_path=stereo_stream_path,
        intrinsic_file=intrinsic_file,
        out_dir=seq_dir,
        max_frames=state.get("max_frames"),
        frame_step=state.get("frame_step", 1),
    )
    report = _toolkit.run(inp, log_dir=log_dir)

    if not report.success:
        record_stage(run_root, stage_key, ok=False, error=report.error)
        return {"ok": False, "error": report.error}

    outputs = {
        "sequence_dir":     seq_dir,
        "K_path":           report.K_path,
        "left_frames_dir":  report.left_frames_dir,
        "right_frames_dir": report.right_frames_dir,
    }
    stats = {
        "fps":           report.fps,
        "width":         report.width,
        "height":        report.height,
        "total_frames":  report.total_frames,
        "baseline_m":    report.baseline_m,
        "camera_id":     camera_id,
    }
    record_stage(run_root, stage_key, ok=True, outputs=outputs, stats=stats)
    return {"ok": True, "outputs": outputs, "stats": stats}


@tool
def svo_extract(run_root: str) -> str:
    """Extract rectified left/right stereo frames from the PRIMARY camera.

    Args:
        run_root: absolute path to runs/<episode_id>/. The state.json there
            must already contain ``stereo_stream_path`` and
            ``stereo_intrinsics_path`` (seeded by data_ingest). For dual-view
            builds, also ``primary_camera_id`` — frames go
            under ``seq/<primary_camera_id>/`` instead of legacy ``seq/``.

    Returns:
        JSON string:
          {"ok": True,
           "outputs": {"sequence_dir", "K_path", "left_frames_dir", "right_frames_dir"},
           "stats":   {"fps", "width", "height", "total_frames", "baseline_m", "camera_id"}}
        On failure:
          {"ok": False, "error": "<reason>"}
    """
    state = load_state(run_root)
    stereo_stream_path = state.get("stereo_stream_path")
    if not stereo_stream_path:
        err = "state.stereo_stream_path is missing — has data_ingest been run?"
        record_stage(run_root, "svo_extract", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})
    intrinsic_file = state.get("stereo_intrinsics_path")
    if not intrinsic_file:
        err = "state.stereo_intrinsics_path is missing — has data_ingest been run?"
        record_stage(run_root, "svo_extract", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    result = _run_svo_extract(
        run_root,
        camera_id=state.get("primary_camera_id", "") or "",
        stereo_stream_path=stereo_stream_path,
        intrinsic_file=intrinsic_file,
        stage_key="svo_extract",
        log_subdir="",
    )
    return json.dumps(result)


def run_for_secondary(run_root: str) -> dict:
    """Extract rectified frames for the SECONDARY camera (PR-3 callers).

    Reads canonical ``secondary_stereo_*`` fields + ``secondary_camera_id``
    from state. ``secondary_svo_path`` is accepted only as a deprecated
    rectified-MP4 alias for the stream path. Records under
    ``stages.svo_extract_secondary`` so primary's stage record stays untouched.
    """
    state = load_state(run_root)
    secondary_stream = (
        state.get("secondary_stereo_stream_path", "")
        or state.get("secondary_svo_path", "")
        or ""
    )
    secondary_intrinsics = state.get("secondary_stereo_intrinsics_path", "") or ""
    secondary_id  = state.get("secondary_camera_id", "") or ""
    if not secondary_stream or not secondary_intrinsics or not secondary_id:
        return {
            "ok": False,
            "error": "secondary_stereo_stream_path / "
                     "secondary_stereo_intrinsics_path / secondary_camera_id "
                     "not set (single-camera episode — nothing to extract)",
        }
    return _run_svo_extract(
        run_root,
        camera_id=secondary_id,
        stereo_stream_path=secondary_stream,
        intrinsic_file=secondary_intrinsics,
        stage_key="svo_extract_secondary",
        log_subdir="secondary",
    )
