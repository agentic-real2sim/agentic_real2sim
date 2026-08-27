"""video_build skill — assemble rectified left frames into an mp4 for SAM3.

Reads from state.json:
  - stages.svo_extract.outputs.{left_frames_dir, sequence_dir}  (primary)
  - stages.svo_extract.stats.fps                                (ffmpeg framerate)
  - stages.svo_extract_secondary.* (PR-3 path; consumed via run_for_secondary)

Writes to disk:
  - <sequence_dir>/rectified_left.mp4 (mp4 lives alongside the frames so
    dual-view builds get a per-camera mp4 without colliding)
  - logs under <run_root>/logs/video_build[_<role>].{cmd,...}

Writes to state.json:
  - state["stages"]["video_build"]           = {ok, outputs, stats, error}   (primary)
  - state["stages"]["video_build_secondary"] = same shape, set by PR-3 only

Pure ffmpeg call — no conda env switch (ffmpeg resolved via imageio_ffmpeg,
already a dep of the visual orchestrator env).
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from ar2s.agents_visual._toolkit import video_build as _toolkit
from ar2s.agents_visual.state import load_state, record_stage


def _run_video_build(
    run_root: str,
    *,
    svo_stage: dict,
    stage_key: str,
    log_subdir: str,
) -> dict:
    """Camera-aware core. Encodes ``rectified_left.mp4`` next to the frames
    (under ``svo_stage.outputs.sequence_dir``) so dual-view builds get
    per-camera mp4s without overwriting each other.
    """
    left_frames_dir = svo_stage.get("outputs", {}).get("left_frames_dir")
    sequence_dir    = svo_stage.get("outputs", {}).get("sequence_dir")
    fps             = svo_stage.get("stats", {}).get("fps")
    if not left_frames_dir or not sequence_dir or not fps:
        err = ("svo stage outputs.left_frames_dir / outputs.sequence_dir / "
               "stats.fps missing")
        record_stage(run_root, stage_key, ok=False, error=err)
        return {"ok": False, "error": err}

    mp4_path = str(Path(sequence_dir) / "rectified_left.mp4")
    log_dir = (str(Path(run_root) / "logs" / log_subdir)
               if log_subdir else str(Path(run_root) / "logs"))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    inp = _toolkit.VideoBuildInput(
        frames_dir=left_frames_dir,
        output_mp4_path=mp4_path,
        fps=float(fps),
    )
    report = _toolkit.run(inp, log_dir=log_dir)

    if not report.success:
        record_stage(run_root, stage_key, ok=False, error=report.error)
        return {"ok": False, "error": report.error}

    outputs = {"mp4_path": report.mp4_path}
    stats = {"fps": float(fps), "crf": inp.crf}
    record_stage(run_root, stage_key, ok=True, outputs=outputs, stats=stats)
    return {"ok": True, "outputs": outputs, "stats": stats}


@tool
def video_build(run_root: str) -> str:
    """Encode the PRIMARY view's rectified frames into an mp4 for SAM3.

    Args:
        run_root: absolute path to runs/<episode_id>/. Requires
            ``stages.svo_extract`` to be ok.

    Returns:
        JSON string:
          {"ok": True,
           "outputs": {"mp4_path"},
           "stats":   {"fps", "crf"}}
        On failure:
          {"ok": False, "error": "<reason>"}
    """
    state = load_state(run_root)
    svo = state.get("stages", {}).get("svo_extract", {})
    if not svo.get("ok"):
        err = "stages.svo_extract has not completed successfully; run svo_extract first"
        record_stage(run_root, "video_build", ok=False, error=err)
        return json.dumps({"ok": False, "error": err})

    result = _run_video_build(
        run_root,
        svo_stage=svo,
        stage_key="video_build",
        log_subdir="",
    )
    return json.dumps(result)


def run_for_secondary(run_root: str) -> dict:
    """Encode the secondary view's rectified frames into an mp4 (PR-3 callers).

    Requires ``stages.svo_extract_secondary`` to be ok (i.e. PR-3's
    ``svo_extract.run_for_secondary`` was called first). Records under
    ``stages.video_build_secondary``.
    """
    state = load_state(run_root)
    svo_sec = state.get("stages", {}).get("svo_extract_secondary", {})
    if not svo_sec.get("ok"):
        return {
            "ok": False,
            "error": "stages.svo_extract_secondary not ok "
                     "(run svo_extract.run_for_secondary first)",
        }
    return _run_video_build(
        run_root,
        svo_stage=svo_sec,
        stage_key="video_build_secondary",
        log_subdir="secondary",
    )
