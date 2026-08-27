#!/usr/bin/env python3
"""Create real-vs-sim side-by-side videos for paired episode renders.

Expected input layout:

    <real-root>/<episode_id>/real/video.mp4
    <sim-dir>/<episode_id>.mp4

For each episode, the simulated video is sampled down to the real video's
frame count, then stacked with the real video on the left and the sampled
simulation on the right. The output video uses the real video's FPS.

Usage:
    .conda/droid_sim/bin/python scripts/make_real_vs_sim_side_by_side.py \\
        --real-root /media/eric/data/droid_sim/outputs/collect_blender_grasp_assets/qianjun_2026-07-02_blender_best_grasps_easy-20260706T142738Z-3-001_reorg \\
        --sim-dir /media/eric/data/droid_sim/episodic_renders/2026-07-07_rendered \\
        --output-dir /media/eric/data/droid_sim/episodic_renders/2026-07-07_side-by-side \\
        --overwrite
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


DEFAULT_REAL_ROOT = Path(
    "/media/eric/data/droid_sim/outputs/collect_blender_grasp_assets/"
    "qianjun_2026-07-02_blender_best_grasps_easy-20260706T142738Z-3-001_reorg"
)
DEFAULT_SIM_DIR = Path("/media/eric/data/droid_sim/episodic_renders/2026-07-07_rendered")
DEFAULT_OUTPUT_DIR = Path(
    "/media/eric/data/droid_sim/episodic_renders/2026-07-07_side-by-side"
)
DEFAULT_FFMPEG = Path("/usr/bin/ffmpeg")
DEFAULT_FFPROBE = Path("/usr/bin/ffprobe")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frame_count: int
    duration_s: float


@dataclass(frozen=True)
class VideoPair:
    episode_id: str
    real_path: Path
    sim_path: Path
    out_path: Path


def parse_fraction(value: str) -> Fraction:
    num, den = value.split("/")
    fps = Fraction(int(num), int(den))
    assert fps > 0, f"invalid frame rate: {value}"
    return fps


def ffprobe_video(path: Path, ffprobe: Path) -> VideoInfo:
    assert path.is_file(), f"missing video: {path}"
    proc = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    assert len(streams) == 1, f"expected one video stream in {path}, got {len(streams)}"
    stream = streams[0]
    frame_count_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    assert frame_count_raw not in (None, "N/A"), f"ffprobe did not report frames for {path}"
    fps_raw = stream.get("avg_frame_rate")
    if fps_raw in (None, "0/0", "N/A"):
        fps_raw = stream["r_frame_rate"]
    duration_raw = stream.get("duration")
    assert duration_raw not in (None, "N/A"), f"ffprobe did not report duration for {path}"
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=parse_fraction(fps_raw),
        frame_count=int(frame_count_raw),
        duration_s=float(duration_raw),
    )


def discover_pairs(
    real_root: Path,
    sim_dir: Path,
    output_dir: Path,
    episodes: set[str],
) -> list[VideoPair]:
    assert real_root.is_dir(), f"real root does not exist: {real_root}"
    assert sim_dir.is_dir(), f"sim dir does not exist: {sim_dir}"

    pairs: list[VideoPair] = []
    for real_path in sorted(real_root.glob("*/real/video.mp4")):
        episode_id = real_path.parents[1].name
        if episodes and episode_id not in episodes:
            continue
        sim_path = sim_dir / f"{episode_id}.mp4"
        assert sim_path.is_file(), f"missing sim video for {episode_id}: {sim_path}"
        pairs.append(
            VideoPair(
                episode_id=episode_id,
                real_path=real_path,
                sim_path=sim_path,
                out_path=output_dir / f"{episode_id}_real_vs_sim.mp4",
            )
        )

    if episodes:
        found = {pair.episode_id for pair in pairs}
        missing = sorted(episodes - found)
        assert not missing, f"requested episodes without real videos: {missing}"
    assert pairs, f"no real videos found under {real_root}/*/real/video.mp4"
    return pairs


def fps_arg(fps: Fraction) -> str:
    if fps.denominator == 1:
        return str(fps.numerator)
    return f"{fps.numerator}/{fps.denominator}"


def sim_select_expr(sim_frame_count: int, target_frame_count: int) -> str:
    assert target_frame_count > 0, target_frame_count
    assert sim_frame_count >= target_frame_count, (
        f"sim video has fewer frames than target: sim={sim_frame_count} "
        f"target={target_frame_count}"
    )
    if target_frame_count == 1:
        return "eq(n\\,0)"
    # selected_n is the number of already-selected frames. This chooses
    # round(i * (M - 1) / (N - 1)) for i in [0, N - 1], including endpoints.
    return (
        "eq(n\\,"
        f"round(selected_n*{sim_frame_count - 1}/{target_frame_count - 1})"
        ")"
    )


def build_filter(real: VideoInfo, sim: VideoInfo, *, height: int) -> tuple[str, str]:
    assert real.frame_count > 0, "real video has no frames"
    assert sim.frame_count > 0, "sim video has no frames"
    assert sim.frame_count >= real.frame_count, (
        f"sim frame count must be >= real frame count: sim={sim.frame_count} "
        f"real={real.frame_count}"
    )
    assert height > 0 and height % 2 == 0, f"--height must be a positive even integer: {height}"
    fps = fps_arg(real.fps)
    select_expr = sim_select_expr(sim.frame_count, real.frame_count)
    filter_complex = (
        f"[0:v]trim=end_frame={real.frame_count},"
        f"setpts=N/({fps}*TB),scale=-2:{height},setsar=1[real];"
        f"[1:v]select='{select_expr}',"
        f"setpts=N/({fps}*TB),scale=-2:{height},setsar=1[sim];"
        "[real][sim]hstack=inputs=2:shortest=1,format=yuv420p[v]"
    )
    return filter_complex, select_expr


def render_pair(
    pair: VideoPair,
    *,
    ffmpeg: Path,
    ffprobe: Path,
    height: int,
    crf: int,
    preset: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    real = ffprobe_video(pair.real_path, ffprobe)
    sim = ffprobe_video(pair.sim_path, ffprobe)
    assert real.fps > 0, f"invalid real fps for {pair.episode_id}: {real.fps}"
    if pair.out_path.exists() and not overwrite:
        print(f"[skip] {pair.episode_id}: {pair.out_path}", flush=True)
        return "skipped"

    pair.out_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex, select_expr = build_filter(real, sim, height=height)
    cmd = [
        str(ffmpeg),
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(pair.real_path),
        "-i",
        str(pair.sim_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-frames:v",
        str(real.frame_count),
        "-r",
        fps_arg(real.fps),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        "-an",
        str(pair.out_path),
    ]
    print(
        f"[render] {pair.episode_id}: real={real.frame_count}@{fps_arg(real.fps)}fps "
        f"sim={sim.frame_count}@{fps_arg(sim.fps)}fps select={select_expr} "
        f"out={pair.out_path}",
        flush=True,
    )
    if dry_run:
        print(" ".join(cmd), flush=True)
        return "dry-run"
    subprocess.run(cmd, check=True)
    return "rendered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Episode id to render. May be passed multiple times.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output panel height for each side. Must be even.",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    real_root = args.real_root.expanduser().resolve()
    sim_dir = args.sim_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()

    assert ffmpeg.is_file(), f"ffmpeg not found: {ffmpeg}"
    assert ffprobe.is_file(), f"ffprobe not found: {ffprobe}"

    pairs = discover_pairs(real_root, sim_dir, output_dir, set(args.episode))
    counts = {"rendered": 0, "skipped": 0, "dry-run": 0}
    for pair in pairs:
        status = render_pair(
            pair,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            height=args.height,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        counts[status] += 1

    print(
        f"[done] processed={len(pairs)} rendered={counts['rendered']} "
        f"skipped={counts['skipped']} dry_run={counts['dry-run']} "
        f"output_dir={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
