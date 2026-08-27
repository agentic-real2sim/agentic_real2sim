"""segmentation skill — sam3 video-promptable segmentation.

``run_jobs(run_root, jobs, *, view)`` is the only public surface: a thin wrapper
that delegates to ``_toolkit.run`` after pulling ``video_path`` + ``output_dir``
from state for the requested view. Returns ``{out_subdir: JobMaskInfo}`` so the
``segment`` controller can index its candidate state directly. Winner-picking
and retry logic are the controller's concerns, not this module's.

Reads from state.json:
  - stages.video_build.outputs.mp4_path            (view="primary")
  - stages.video_build_secondary.outputs.mp4_path  (view="secondary")

Writes to disk:
  - <run_root>/segmentation/<out_subdir>/masks/NNNNNN.png
  - <run_root>/segmentation/<out_subdir>/<out_subdir>_overlay.mp4
  - <run_root>/segmentation/<out_subdir>/metadata.json
  - logs under <run_root>/logs/segmentation.{cmd,stdout,stderr,returncode}
  (secondary lands under segmentation_secondary/ and logs/secondary/)

Subprocess env: ``sam3`` (no XFORMERS_DISABLED inject — sam3 doesn't
use xformers; pkg_resources / numpy<2 pins live inside the install script).
"""
from __future__ import annotations

from pathlib import Path

from ar2s.agents_visual._toolkit import segmentation as _toolkit
from ar2s.agents_visual.state import load_state


def run_jobs(
    run_root: str,
    jobs: list[_toolkit.SegJob],
    *,
    view: str = "primary",
) -> dict[str, _toolkit.JobMaskInfo]:
    """Run SAM3 on a batch of jobs; return ``{out_subdir: JobMaskInfo}``.

    Used by ``subagents.segment_controller`` — gets one out_subdir per pending
    candidate so a whole round runs in one subprocess (one SAM3 model load).
    Raises on missing prerequisites (video_build) or a SAM3 subprocess error;
    the controller treats those as a hard failure.

    Does NOT do winner-picking or retry planning — the controller owns those.

    Dual-view: ``view="primary"`` reads from ``stages.video_build``
    and writes candidates under ``segmentation/``. ``view="secondary"`` reads
    from ``stages.video_build_secondary`` and writes under
    ``segmentation_secondary/`` so primary + secondary candidate dirs cannot
    collide on identical ``(name, prompt, kf)`` triples. The CALLER is
    responsible for routing each job to the correct view (segment_controller
    holds two separate job lists, one per lane).
    """
    state = load_state(run_root)
    if view == "primary":
        vb_key, out_subdir = "video_build", "segmentation"
    elif view == "secondary":
        vb_key, out_subdir = "video_build_secondary", "segmentation_secondary"
    else:
        raise ValueError(f"run_jobs view must be 'primary' or 'secondary', got {view!r}")

    vb = state.get("stages", {}).get(vb_key, {})
    if not vb.get("ok"):
        raise RuntimeError(
            f"stages.{vb_key} has not completed; "
            f"run video_build{'_for_secondary' if view == 'secondary' else ''} first"
        )
    video_path = vb.get("outputs", {}).get("mp4_path")
    if not video_path:
        raise RuntimeError(f"stages.{vb_key}.outputs.mp4_path missing")

    output_dir = str(Path(run_root) / out_subdir)
    log_dir = (str(Path(run_root) / "logs" / "secondary")
               if view == "secondary" else str(Path(run_root) / "logs"))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    report = _toolkit.run(
        _toolkit.SegmentationInput(video_path=video_path, output_dir=output_dir, jobs=jobs),
        log_dir=log_dir,
    )
    if not report.success:
        raise RuntimeError(f"segmentation ({view}) failed: {report.error}")
    return {j.out_subdir: j for j in report.jobs}


