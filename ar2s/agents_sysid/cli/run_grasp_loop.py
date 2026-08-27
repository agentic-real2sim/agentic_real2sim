"""CLI: run Stage 3 LLM retry loop (grasp_loop).

Usage:
    python -m ar2s.agents_sysid.cli.run_grasp_loop \
        outputs/run_pipeline/marker_in_mug_v0/sysid_inputs/marker_in_mug_v0 --max-rounds 10

Strategy: Round 0 deterministic shift from first_contact, then LLM-driven
(Agent A view-sufficiency + Agent B shift decision) rounds 1-N. Writes
per-round artifacts including USD animations (47 MB / round; disable with
``--no-usd`` if disk-constrained).

Triggers Stage 2 automatically if no cached calibration.yaml.

Exit code 0 on grasped success, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ar2s.agent_configs.models import set_config_path
from ar2s.pipeline_artifacts import pipeline_run_dir
from ar2s.agents_sysid.grasp_loop import (
    CUMULATIVE_SHIFT_CAP_MM,
    MAX_ROUNDS,
    PER_ROUND_SHIFT_CAP_MM,
    USD_SAMPLE_EVERY_DEFAULT,
    grasp_loop,
)


def main():
    p = argparse.ArgumentParser(description="Run Stage 3 LLM grasp retry loop.")
    p.add_argument("episode_dir",
                   help="Built episode folder (manifest.yaml + calibration.yaml).")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                   help=f"Max outer rounds (default {MAX_ROUNDS}).")
    p.add_argument("--per-round-cap-mm", type=int, default=PER_ROUND_SHIFT_CAP_MM,
                   help=f"Per-round |shift| cap per axis (default {PER_ROUND_SHIFT_CAP_MM}).")
    p.add_argument("--cumulative-cap-mm", type=int, default=CUMULATIVE_SHIFT_CAP_MM,
                   help=f"Cumulative |shift| cap per axis (default {CUMULATIVE_SHIFT_CAP_MM}).")
    p.add_argument("--agent-config", default=None, type=Path,
                   help="YAML agent model config path. Defaults to "
                        "ar2s/agent_configs/config_anthropic_opus47.yaml.")
    p.add_argument("--no-usd", action="store_true",
                   help="Disable USD export per round.")
    p.add_argument("--usd-sample-every", type=int, default=USD_SAMPLE_EVERY_DEFAULT,
                   help=f"USD frame stride (default {USD_SAMPLE_EVERY_DEFAULT}).")
    p.add_argument("--artifacts-root", default=None,
                   help="Parent of <run-id>/ output. Defaults to "
                        "outputs/run_pipeline/<episode-run-id>/artifacts.")
    p.add_argument("--run-id", default=None,
                   help="Explicit run id (default = timestamp + uuid).")
    args = p.parse_args()
    set_config_path(args.agent_config)

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
        result = grasp_loop(
            episode_dir,
            artifacts_root=args.artifacts_root,
            run_id=args.run_id,
            max_rounds=args.max_rounds,
            per_round_cap_mm=args.per_round_cap_mm,
            cumulative_cap_mm=args.cumulative_cap_mm,
            save_usd=not args.no_usd,
            usd_sample_every=args.usd_sample_every,
        )
    except Exception as e:
        print(f"[run_grasp_loop] FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if result.outcome == "success" else 1)


if __name__ == "__main__":
    main()
