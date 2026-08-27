"""Stage 3: left frames -> rectified_left.mp4 (input for SAM3).

Pure ffmpeg invocation — no conda env switch. The ffmpeg binary is resolved
via ``imageio_ffmpeg`` (already a project dep), falling back to PATH lookup
if the package isn't available.
"""
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg

from ar2s.agents_visual._toolkit._runners import run_command


def _resolve_ffmpeg() -> str:
    """Locate ffmpeg via imageio_ffmpeg's bundled binary."""
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class VideoBuildInput:
    frames_dir: str                         # directory containing frame_NNNNNN.png
    output_mp4_path: str
    fps: float                              # match source (from SVOExtractReport.fps)
    crf: int = 18


@dataclass
class VideoBuildReport:
    success: bool
    error: str = ""
    mp4_path: str = ""


def run(inp: VideoBuildInput, *, log_dir: str | Path) -> VideoBuildReport:
    Path(inp.output_mp4_path).parent.mkdir(parents=True, exist_ok=True)
    pattern = str(Path(inp.frames_dir) / "frame_%06d.png")

    cmd = [
        _resolve_ffmpeg(), "-y",
        "-framerate", str(inp.fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(inp.crf),
        inp.output_mp4_path,
    ]

    result = run_command(cmd, log_dir=log_dir, stage="video_build")

    if result.returncode != 0:
        return VideoBuildReport(
            success=False,
            error=(
                f"ffmpeg failed (rc={result.returncode}); see {result.stderr_path}\n"
                f"{result.stderr_tail}"
            ),
        )

    return VideoBuildReport(success=True, mp4_path=inp.output_mp4_path)
