"""CLI: render a side-by-side comparison MP4 of two G1 motion NPZ files.

Usage (g1sysid env; headless-safe — EGL is forced before mujoco loads):

    .conda/g1sysid/bin/python -m ar2s.agents_g1sysid.cli.render_compare \
        reference.npz rollout_recovered.npz --out compare.mp4

Typical inputs: the retargeted-LAFAN1 reference (convert_lafan output) on
the left, a recorded closed-loop rollout (rollout_recovered.npz from
run_sysid, or --record output of run_mujoco_env_demo) on the right.

Exit code 0 = written; 1 = render error.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render two motion NPZ files side by side into one MP4.",
    )
    p.add_argument("reference", help="Left clip: reference-format .npz.")
    p.add_argument("rollout", help="Right clip: reference-format .npz.")
    p.add_argument("--out", required=True, help="Output .mp4 path.")
    p.add_argument("--labels", default="reference,rollout",
                   help="Comma-separated overlay labels (default "
                        "'reference,rollout').")
    args = p.parse_args()

    labels = tuple(args.labels.split(",", 1))
    if len(labels) != 2:
        print("[render_compare] --labels needs two comma-separated names",
              file=sys.stderr)
        return 1

    from ar2s.agents_g1sysid.motion_render import render_side_by_side

    try:
        summary = render_side_by_side(
            args.reference, args.rollout, args.out, labels=labels,
        )
    except Exception as e:
        print(f"[render_compare] FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    print(f"[render_compare] {summary['T']} frames @ {summary['fps']:g} fps "
          f"-> {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
