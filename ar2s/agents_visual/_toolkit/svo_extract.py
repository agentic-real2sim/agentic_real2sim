"""Stage 1: exported rectified stereo MP4 -> frames + copied intrinsics.

The visual-agent path consumes the raw exporter's rectified side-by-side MP4
and explicit SDK/SVO K sidecar. It does not read SVO files, fetch ZED
calibration, or run OpenCV rectification. The split-only helper writes::

    <out_dir>/K.txt
    <out_dir>/frames/left/frame_NNNNNN.png
    <out_dir>/frames/right/frame_NNNNNN.png

Env: ``qianjun_foundation_stereo``.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ar2s.agents_visual._toolkit._runners import env_for_stage, run_in_env, script_path


_FPS_REGEX = re.compile(r"FPS:\s*([\d.]+)")


@dataclass
class SVOExtractInput:
    source_path: str                        # exported rectified stereo side-by-side .mp4
    intrinsic_file: str                     # exporter-produced SDK/SVO K sidecar
    out_dir: str
    max_frames: int | None = None
    frame_step: int = 1


@dataclass
class SVOExtractReport:
    success: bool
    error: str = ""
    K_path: str = ""                        # <out_dir>/K.txt
    left_frames_dir: str = ""               # <out_dir>/frames/left
    right_frames_dir: str = ""              # <out_dir>/frames/right
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0                   # frames actually written
    baseline_m: float = 0.0


def run(inp: SVOExtractInput, *, log_dir: str) -> SVOExtractReport:
    suffix = Path(inp.source_path).suffix.lower()
    if suffix == ".svo":
        return SVOExtractReport(
            success=False,
            error="visual-agent requires exported rectified .mp4; .svo is no longer accepted",
        )
    if suffix != ".mp4":
        return SVOExtractReport(
            success=False,
            error=f"unsupported source extension {suffix!r}; expected exported rectified .mp4",
        )

    source_path = Path(inp.source_path)
    if not source_path.exists():
        return SVOExtractReport(
            success=False,
            error=f"stereo stream missing: {source_path}",
        )

    intrinsic_path = Path(inp.intrinsic_file)
    if not intrinsic_path.exists():
        return SVOExtractReport(
            success=False,
            error=f"intrinsic_file missing: {intrinsic_path}",
        )

    report = _run_dispatch(
        inp,
        log_dir=log_dir,
        script_key="split_rectified_mp4",
        file_flag="--video_file",
        extra_args=["--intrinsic_file", inp.intrinsic_file],
    )

    # frame_step>1 makes the upstream script write NON-consecutive original
    # indices (frame_000000, 000005, ...). Downstream assumes a consecutive
    # 0..N sequence (video_build's ffmpeg `frame_%06d.png`, prompt_frame_index,
    # mesh_recover's `frame_{idx:06d}.png`). Re-index the kept frames to
    # consecutive and divide the reported fps by the step (we kept 1/step).
    if report.success and inp.frame_step > 1:
        n = _reindex_consecutive(report.left_frames_dir, report.right_frames_dir)
        report.total_frames = n
        if report.fps:
            report.fps = report.fps / inp.frame_step
    return report


def _reindex_consecutive(*frame_dirs: str) -> int:
    """Rename ``frame_*.png`` in each dir to a consecutive ``frame_000000..`` run.

    frame_step subsampling leaves gaps (000000, 000005, ...); this collapses
    them to 0,1,2,... Two-pass via temp names so shrinking indices can't collide
    with an existing target. All dirs are processed in the same sorted order, so
    left/right stay frame-aligned. Returns the per-dir frame count.
    """
    count = 0
    for d in frame_dirs:
        dpath = Path(d)
        if not dpath.is_dir():
            continue
        frames = sorted(dpath.glob("frame_*.png"))
        tmps: list[Path] = []
        for i, p in enumerate(frames):
            tmp = dpath / f".reindex_{i:06d}.png"
            p.rename(tmp)
            tmps.append(tmp)
        for i, tmp in enumerate(tmps):
            tmp.rename(dpath / f"frame_{i:06d}.png")
        count = len(tmps)
    return count


def _run_dispatch(
    inp: SVOExtractInput,
    *,
    log_dir: str,
    script_key: str,
    file_flag: str,
    extra_args: list[str] | None = None,
) -> SVOExtractReport:
    Path(inp.out_dir).mkdir(parents=True, exist_ok=True)

    args = [file_flag, inp.source_path, "--out_dir", inp.out_dir]
    if inp.max_frames is not None:
        args += ["--max_frames", str(inp.max_frames)]
    if inp.frame_step != 1:
        args += ["--frame_step", str(inp.frame_step)]
    if extra_args:
        args += extra_args

    result = run_in_env(
        env=env_for_stage(script_key),
        script_path=script_path(script_key),
        args=args,
        log_dir=log_dir,
        stage="svo_extract",
    )

    if result.returncode != 0:
        return SVOExtractReport(
            success=False,
            error=(
                f"{script_key} failed (rc={result.returncode}); see {result.stderr_path}\n"
                f"{result.stderr_tail}"
            ),
        )

    return _parse_outputs(inp, result)


def _parse_outputs(inp: SVOExtractInput, result) -> SVOExtractReport:
    out_dir   = Path(inp.out_dir)
    k_path    = out_dir / "K.txt"
    left_dir  = out_dir / "frames" / "left"
    right_dir = out_dir / "frames" / "right"

    if not k_path.exists():
        return SVOExtractReport(success=False, error=f"K.txt missing at {k_path}")

    lines = [ln for ln in k_path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return SVOExtractReport(success=False, error=f"K.txt malformed at {k_path}")
    baseline_m = float(lines[1])

    left_frames = sorted(left_dir.glob("frame_*.png"))
    right_frames = sorted(right_dir.glob("frame_*.png"))
    if not left_frames:
        return SVOExtractReport(success=False, error=f"no frames written to {left_dir}")
    if len(right_frames) != len(left_frames):
        return SVOExtractReport(
            success=False,
            error=f"left/right frame count mismatch: {len(left_frames)} vs {len(right_frames)}",
        )
    with Image.open(left_frames[0]) as img:
        width, height = img.size

    # FPS isn't stored on disk; parse it from the script's log output.
    # split_rectified_mp4_frames.py logs FPS to stderr. Fall back to stdout
    # just in case the logging configuration changes.
    fps = _extract_fps(result.stderr_path, result.stdout_path)

    return SVOExtractReport(
        success=True,
        K_path=str(k_path),
        left_frames_dir=str(left_dir),
        right_frames_dir=str(right_dir),
        fps=fps,
        width=width,
        height=height,
        total_frames=len(left_frames),
        baseline_m=baseline_m,
    )


def _extract_fps(*log_paths: Path) -> float:
    for p in log_paths:
        m = _FPS_REGEX.search(p.read_text(errors="replace"))
        if m:
            return float(m.group(1))
    return 0.0
