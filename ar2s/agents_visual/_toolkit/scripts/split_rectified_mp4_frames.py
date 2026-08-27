#!/usr/bin/env python3
"""Split an exported rectified stereo SBS MP4 into left/right PNG frames.

This helper assumes the raw exporter already used the ZED SDK/SVO metadata for
rectification and wrote a matching ``<serial>-stereo.K.txt`` sidecar. It only
copies that sidecar to ``K.txt`` and splits each RGB frame in half.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np


def _video_fps(video_file: Path) -> float:
    reader = imageio.get_reader(str(video_file))
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    return float(meta.get("fps") or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", required=True)
    parser.add_argument("--intrinsic_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    video_file = Path(args.video_file)
    intrinsic_file = Path(args.intrinsic_file)
    out_dir = Path(args.out_dir)
    assert video_file.exists(), video_file
    assert intrinsic_file.exists(), intrinsic_file
    assert args.frame_step >= 1, args.frame_step
    if args.max_frames is not None:
        assert args.max_frames >= 0, args.max_frames

    left_dir = out_dir / "frames" / "left"
    right_dir = out_dir / "frames" / "right"
    if left_dir.exists():
        shutil.rmtree(left_dir)
    if right_dir.exists():
        shutil.rmtree(right_dir)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(intrinsic_file, out_dir / "K.txt")

    fps = _video_fps(video_file)
    logging.info("FPS: %s", fps)

    written = 0
    for src_idx, frame in enumerate(iio.imiter(str(video_file))):
        if src_idx % args.frame_step != 0:
            continue
        if args.max_frames is not None and written >= args.max_frames:
            break

        assert frame.ndim == 3 and frame.shape[2] == 3, (
            video_file,
            src_idx,
            frame.shape,
        )
        height, width, _ = frame.shape
        assert width % 2 == 0, (video_file, src_idx, width)
        half_width = width // 2
        assert half_width > 0 and height > 0, (video_file, src_idx, frame.shape)

        left = np.ascontiguousarray(frame[:, :half_width, :])
        right = np.ascontiguousarray(frame[:, half_width:, :])
        name = f"frame_{src_idx:06d}.png"
        iio.imwrite(left_dir / name, left)
        iio.imwrite(right_dir / name, right)
        written += 1

    logging.info("Frames written: %d", written)


if __name__ == "__main__":
    main()
