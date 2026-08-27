"""Stage 4: mp4 + per-job (text, frame) prompts -> per-candidate masks.

Wraps ``run_sam3_segment.py`` (vendored copy of upstream
``batch_video_segment.py`` with a one-line API patch — see _runners.SCRIPTS;
env: ``qianjun_sam3``).

Each *job* is a (text, prompt frame, output subdir) triple, optionally carrying
a pixel ``box``. SAM3 grounds the text on that job's frame — with the box, when
given, sent in the same prompt to say which instance is meant — keeps the
highest-scoring detection, and propagates it through the whole video. For each
job the script writes::

    <output_dir>/<out_subdir>/masks/NNNNNN.png         (one PNG per frame)
    <output_dir>/<out_subdir>/<out_subdir>_overlay.mp4
    <output_dir>/<out_subdir>/metadata.json

The ``segment`` controller drives this with one job per pending candidate so a
whole round of segmentation runs in a single subprocess (one SAM3 model load).
When SAM3 finds no detection on a job's prompt frame, only ``metadata.json`` is
written (selected_obj_id=null) and that candidate's ``num_nonempty_masks`` is 0.

Winner picking is the caller's concern; this module only runs jobs and reports
what landed on disk.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ar2s.agents_visual._toolkit._runners import env_for_stage, run_in_env, script_path


@dataclass
class SegJob:
    text: str                               # SAM3 noun-phrase prompt
    frame_index: int                        # frame the text is grounded on
    out_subdir: str                         # masks land under <output_dir>/<out_subdir>/
    box: list[int] | None = None            # optional [x1,y1,x2,y2] hybrid box prompt


@dataclass
class SegmentationInput:
    video_path: str
    output_dir: str                         # candidate dirs land under here
    jobs: list[SegJob]


@dataclass
class JobMaskInfo:
    out_subdir: str
    masks_dir: str                          # <output_dir>/<out_subdir>/masks; empty when no detection
    overlay_mp4: str = ""                   # empty when no detection
    num_nonempty_masks: int = 0             # count of mask PNGs with any foreground pixel
    num_frames: int = 0                     # total frames (video frame count; 0 when no detection)


@dataclass
class SegmentationReport:
    success: bool
    error: str = ""
    jobs: list[JobMaskInfo] = field(default_factory=list)


def run(inp: SegmentationInput, *, log_dir: str) -> SegmentationReport:
    if not inp.jobs:
        return SegmentationReport(success=False, error="jobs is empty")

    out_root = Path(inp.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs_payload = []
    for j in inp.jobs:
        payload = {
            "text": j.text, "frame_index": int(j.frame_index), "out_subdir": j.out_subdir,
        }
        if j.box:
            payload["box"] = [int(v) for v in j.box]
        jobs_payload.append(payload)
    jobs_json_path = out_root / "_jobs.json"
    jobs_json_path.write_text(json.dumps(jobs_payload, indent=2))

    args = [
        "--video-path", inp.video_path,
        "--output-dir", inp.output_dir,
        "--jobs-json",  str(jobs_json_path),
    ]

    result = run_in_env(
        env=env_for_stage("segmentation"),
        script_path=script_path("segmentation"),
        args=args,
        log_dir=log_dir,
        stage="segmentation",
    )

    if result.returncode != 0:
        return SegmentationReport(
            success=False,
            error=(
                f"sam3 segmentation failed (rc={result.returncode}); see {result.stderr_path}\n"
                f"{result.stderr_tail}"
            ),
        )

    return _collect_outputs(inp)


def _collect_outputs(inp: SegmentationInput) -> SegmentationReport:
    output_root = Path(inp.output_dir)
    jobs: list[JobMaskInfo] = []

    for job in inp.jobs:
        job_dir = output_root / job.out_subdir
        meta_path = job_dir / "metadata.json"
        if not meta_path.exists():
            jobs.append(JobMaskInfo(out_subdir=job.out_subdir, masks_dir=""))
            continue

        meta = json.loads(meta_path.read_text())
        mask_dir = meta.get("mask_dir") or ""
        overlay = meta.get("overlay_video_path") or ""

        nonempty = _count_nonempty_masks(Path(mask_dir)) if mask_dir else 0

        jobs.append(JobMaskInfo(
            out_subdir=job.out_subdir,
            masks_dir=mask_dir,
            overlay_mp4=overlay,
            num_nonempty_masks=nonempty,
            num_frames=int(meta.get("frame_count") or 0),
        ))

    return SegmentationReport(success=True, jobs=jobs)


def _count_nonempty_masks(mask_dir: Path) -> int:
    count = 0
    for png in mask_dir.glob("*.png"):
        if np.array(Image.open(png)).any():
            count += 1
    return count
