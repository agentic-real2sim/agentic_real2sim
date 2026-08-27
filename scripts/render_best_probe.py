"""Re-run a single grasp probe with USD + primary-cam video.

Used by the sysid daemon after a `--no-usd` sweep identifies the best
sample. We re-execute that ONE probe with the best sample's local-pos
shift so the heavy USD + video work happens only once per episode.

Usage:
    python scripts/render_best_probe.py \\
        --episode-dir episodes_droid_300/<id>_agent \\
        --shift-m 0.001,0.007,0.026 \\
        --out-dir artifacts_droid_300/<id>/best/

Writes:
    <out-dir>/result.usd                         (MuJoCo USDExporter output)
    <out-dir>/primary_view.mp4                   (rendered from primary cam)
    <out-dir>/probe_result.yaml                  (the GraspProbeResult fields)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

from ar2s.agents_sysid.probe import _load_or_compute_calibration
from ar2s.agents_sysid.skills.probe_grasp import probe_grasp


def parse_shift(s: str) -> tuple[float, float, float]:
    """Parse '0.001,0.007,0.026' (METERS) or '1,7,26' (mm, auto-detected) → (dx,dy,dz)."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"--shift-m must be 3 comma-separated floats, got: {s!r}")
    # Heuristic: |v| > 1 → assume mm; else meters. Most real shifts are < 100mm = 0.1m.
    return (parts[0], parts[1], parts[2])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode-dir", type=Path, required=True)
    p.add_argument("--shift-m",     type=str, required=True,
                   help="Local-pos shift in METERS as 'dx,dy,dz'.")
    p.add_argument("--out-dir",     type=Path, required=True)
    p.add_argument("--video-sample-every", type=int, default=3)
    p.add_argument("--video-fps",          type=float, default=30.0)
    p.add_argument("--video-width",        type=int, default=1280)
    p.add_argument("--video-height",       type=int, default=720)
    p.add_argument("--no-video", action="store_true",
                   help="Skip mp4 render (just save USD).")
    p.add_argument("--no-usd",   action="store_true",
                   help="Skip USD (just render mp4).")
    args = p.parse_args()

    shift = parse_shift(args.shift_m)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = _load_or_compute_calibration(args.episode_dir)
    print(f"[render_best] episode_dir : {args.episode_dir}")
    print(f"[render_best] pickup       : {scene.bundle.pickup_object}")
    print(f"[render_best] shift_m      : {shift}")
    print(f"[render_best] out_dir      : {out_dir}")

    usd_dir = None if args.no_usd else out_dir
    video_path = None if args.no_video else (out_dir / "primary_view.mp4")

    result, _frame_cache = probe_grasp(
        scene,
        cumulative_shift_m=shift,
        usd_save_dir=usd_dir,
        usd_sample_every=1,
        video_save_path=video_path,
        video_sample_every=args.video_sample_every,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    # Persist a small summary so the daemon can verify success.
    res_dict = dataclasses.asdict(result)
    (out_dir / "probe_result.yaml").write_text(yaml.safe_dump(res_dict))
    print(f"[render_best] grasped     : {result.grasped}")
    print(f"[render_best] closest_mm  : {result.closest_distance_m * 1000:.2f}")
    print(f"[render_best] peak_lift_mm: {result.peak_lift_m * 1000:.2f}")
    if video_path is not None:
        print(f"[render_best] video       : {video_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
