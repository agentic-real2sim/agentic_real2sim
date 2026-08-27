"""CLI: run Stage 3 brute-force grasp sweep (grasp_sweep).

Usage (default 40 uniform mesh-surface samples):
    python -m ar2s.agents_sysid.cli.run_grasp_sweep \
        outputs/run_pipeline/marker_in_mug_v0/sysid_inputs/marker_in_mug_v0

Usage (custom sample count + frame, no USD, 16 workers):
    python -m ar2s.agents_sysid.cli.run_grasp_sweep \
        outputs/run_pipeline/marker_in_mug_v0/sysid_inputs/marker_in_mug_v0 \
        --n-samples 80 --sample-seed 42 --reference-frame 400 --no-usd --n-workers 16

Triggers Stage 2 automatically if no cached calibration.yaml.

Zero LLM cost. Wall clock ~6s per probe / n_workers. Exit code 0 if any
sample grasped, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ar2s.pipeline_artifacts import pipeline_run_dir
from ar2s.agents_sysid.grasp_sweep import (
    N_SAMPLES_DEFAULT,
    SAMPLE_SEED_DEFAULT,
    USD_SAMPLE_EVERY_DEFAULT,
    grasp_sweep,
)


def main():
    p = argparse.ArgumentParser(description="Run Stage 3 brute-force grasp sweep.")
    p.add_argument("episode_dir",
                   help="Built episode folder (manifest.yaml + calibration.yaml).")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT,
                   help=f"Total uniform surface samples on pickup mesh "
                        f"(default {N_SAMPLES_DEFAULT}).")
    p.add_argument("--sample-seed", type=int, default=SAMPLE_SEED_DEFAULT,
                   help="Numpy seed for trimesh.sample.sample_surface "
                        "(reproducibility).")
    p.add_argument("--reference-frame", type=int, default=None,
                   help="Frame to anchor pad-midpoint at (default engage_frame).")
    p.add_argument("--no-usd", action="store_true",
                   help="Disable USD export (~47 MB / sample).")
    p.add_argument("--usd-sample-every", type=int, default=USD_SAMPLE_EVERY_DEFAULT,
                   help=f"USD frame stride (default {USD_SAMPLE_EVERY_DEFAULT}).")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Multi-process worker count (default = cpu_count - 1).")
    p.add_argument("--artifacts-root", default=None,
                   help="Parent of <run-id>/ output. Defaults to "
                        "outputs/run_pipeline/<episode-run-id>/artifacts.")
    p.add_argument("--run-id", default=None,
                   help="Explicit run id (default = 'sweep-<ts>-<hex>').")
    p.add_argument("--full-collision", action="store_true",
                   help="Enable robot(arm+gripper) <-> ALL scene objects collision "
                        "(default: robot collides only with the pickup object + ground).")
    args = p.parse_args()

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    if args.artifacts_root is None:
        if episode_dir.name == "sysid_inputs":
            pipeline_id = episode_dir.parent.name          # flat run-dir bundle
        elif episode_dir.parent.name in ("sysid_inputs", "episodes"):
            pipeline_id = episode_dir.parent.parent.name   # legacy <id>_agent/
        else:
            pipeline_id = episode_dir.name
        args.artifacts_root = str(pipeline_run_dir(pipeline_id) / "artifacts")

    try:
        result = grasp_sweep(
            episode_dir,
            artifacts_root=args.artifacts_root,
            run_id=args.run_id,
            n_samples=args.n_samples,
            sample_seed=args.sample_seed,
            reference_frame=args.reference_frame,
            save_usd=not args.no_usd,
            usd_sample_every=args.usd_sample_every,
            n_workers=args.n_workers,
            full_collision=args.full_collision,
        )
    except Exception as e:
        print(f"[run_grasp_sweep] FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if result.n_grasped > 0 else 1)


if __name__ == "__main__":
    main()
