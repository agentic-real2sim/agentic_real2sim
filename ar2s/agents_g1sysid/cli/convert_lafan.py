"""CLI: convert a retargeted-LAFAN1 G1 CSV into a reference-motion NPZ.

Usage (from the g1sysid env — needs mujoco + numpy):

    .conda/g1sysid/bin/python -m ar2s.agents_g1sysid.cli.convert_lafan \
        <clip>.csv --out assets_lafan/<clip>.npz

The CSV is the Unitree ``LAFAN1_Retargeting_Dataset`` G1 format (30 fps,
root xyz + quat xyzw + 29 joints per row). The output NPZ is 50 fps (one
frame per control step) with MuJoCo-FK body poses — directly consumable by
the pipeline's tracking task and compare_motion.

Exit code 0 = written; 1 = conversion error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ar2s.agents_g1sysid.motion_convert import (
    CONTROL_FPS,
    LAFAN_FPS,
    convert_lafan_csv,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert a retargeted-LAFAN1 G1 CSV to a reference NPZ.",
    )
    p.add_argument("csv", help="Unitree LAFAN1_Retargeting_Dataset G1 CSV.")
    p.add_argument("--out", required=True,
                   help="Output .npz path (created; parent dirs made).")
    p.add_argument("--src-fps", type=float, default=LAFAN_FPS,
                   help=f"Source frame rate (default {LAFAN_FPS}).")
    p.add_argument("--dst-fps", type=float, default=CONTROL_FPS,
                   help=f"Output frame rate (default {CONTROL_FPS} = one "
                        f"frame per control step).")
    p.add_argument("--frame-start", type=int, default=0,
                   help="First source frame of the slice (default 0).")
    p.add_argument("--frame-end", type=int, default=None,
                   help="End of the slice, exclusive (default: clip end).")
    args = p.parse_args()

    try:
        summary = convert_lafan_csv(
            args.csv, args.out,
            src_fps=args.src_fps, dst_fps=args.dst_fps,
            frame_start=args.frame_start, frame_end=args.frame_end,
        )
    except Exception as e:
        print(f"[convert_lafan] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[convert_lafan] {Path(args.csv).name}: {summary['T']} frames @ "
          f"{args.dst_fps:g} fps ({summary['duration_s']} s, "
          f"{summary['n_bodies']} bodies) -> {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
