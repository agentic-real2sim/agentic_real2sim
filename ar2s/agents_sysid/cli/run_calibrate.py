"""CLI: run Stage 2 (scene calibration) on a built episode folder.

Usage:
    python -m ar2s.agents_sysid.cli.run_calibrate \
        outputs/run_pipeline/marker_in_mug_v0/sysid_inputs/marker_in_mug_v0

Writes:
    <episode_dir>/calibration.yaml      (cache; deletable, recomputable)
    <episode_dir>/manifest.yaml         (patched in place — primary_id +
                                         local_down_axis match calibration)

Wall clock: ~200s on marker_in_mug_v0 (dominated by robot base IoU opt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ar2s.agents_sysid._gl import configure_headless_gl
from ar2s.agents_sysid.calibrate import calibrate_scene


def main():
    p = argparse.ArgumentParser(description="Run Stage 2 calibration.")
    p.add_argument("episode_dir",
                   help="Built episode folder (containing manifest.yaml).")
    p.add_argument("--no-write-yaml", action="store_true",
                   help="Skip writing calibration.yaml (in-memory only).")
    p.add_argument("--no-update-manifest", action="store_true",
                   help="Skip patching manifest.yaml in place.")
    args = p.parse_args()
    configure_headless_gl()

    try:
        calibrate_scene(
            Path(args.episode_dir),
            write_yaml=not args.no_write_yaml,
            update_manifest=not args.no_update_manifest,
        )
    except Exception as e:
        print(f"[run_calibrate] FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
