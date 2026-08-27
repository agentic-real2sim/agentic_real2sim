"""CLI wrapper around :func:`ar2s.agents_visual.raw_episode.build_raw_episode`.

Usage::

    python -m ar2s.agents_visual.cli.build_raw_episode <raw_data_dir> \\
        [--episode-id droid_100_episode_<idx>_<raw_id_slug>] \\
        [--camera auto|ext1|ext2|<serial>] \\
        [--overwrite]

Produces (role-keyed layout — files are named by physical camera role;
"primary" is only an attribute recorded in camera_selection.json + the stub):
    outputs/run_pipeline/<run_id>/
        raw_episodes/{ext1-stereo.mp4, ext1-stereo.K.txt,
                           ext2-stereo.mp4, ext2-stereo.K.txt (if usable),
                           wrist.mp4 (if staged),
                           robot_traj.h5, cameras_extrinsics.npz,
                           camera_selection.json, source/}
        visual_<id>.py

If the run uses --camera auto (default) and the raw_data folder has >=2
external cameras, the camera_select VLM votes on which view to keep.
Single-external sources skip the vote. Re-running with a different --camera
only rewrites camera_selection.json + the stub — the role-named videos never
change content, so stale copies can't silently carry wrong-view footage.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from ar2s.agent_configs import set_config_path
from ar2s.agents_visual.raw_episode import build_raw_episode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DROID raw_data folder into "
            "outputs/run_pipeline/<run_id>/{raw_episodes/,visual_<id>.py}/."
        ),
    )
    parser.add_argument(
        "raw_data_dir",
        type=Path,
        help="Path to the DROID raw_data folder (contains metadata.json, "
             "cameras.json, trajectory.h5, and one or more exported "
             "<serial>-stereo.mp4 + <serial>-stereo.K.txt pairs).",
    )
    parser.add_argument(
        "--episode-id", default=None,
        help=(
            "DROID-100 episode id. Staged folders with stage_manifest.json "
            "derive this automatically; if provided it must match the manifest."
        ),
    )
    parser.add_argument(
        "--camera", default="auto",
        help="Camera selection mode: 'auto' (default; VLM votes when >=2), "
             "'ext1' / 'ext2' (pick by role from metadata), or a literal "
             "ZED serial number.",
    )
    parser.add_argument(
        "--run-suffix", default=None,
        help="Optional tag distinguishing config variants of the SAME episode; "
             "the run dir becomes <artifact-root>/<episode_id>_<suffix>/. The "
             "episode id always prefixes the run dir.",
    )
    parser.add_argument(
        "--artifact-root", default=None, type=Path,
        help="Root containing per-run artifacts. Defaults to "
             "outputs/run_pipeline or AR2S_RUN_PIPELINE_ROOT.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="If this run's flat raw_episodes/ or visual_<id>.py "
             "already exist, replace them.",
    )
    parser.add_argument(
        "--agent-config", default=None, type=Path,
        help="Agent-config YAML used for the --camera auto VLM vote. Defaults "
             "to the built-in default (config_anthropic_opus47.yaml). Set this "
             "to route the camera vote through the same model as the pipeline.",
    )
    args = parser.parse_args()

    set_config_path(args.agent_config)

    try:
        result = build_raw_episode(
            args.raw_data_dir,
            episode_id=args.episode_id,
            run_suffix=args.run_suffix,
            artifact_root=args.artifact_root,
            forced_camera=args.camera,
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"[build_raw_episode] FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        traceback.print_exc()
        return 1

    print()
    print(f"[build_raw_episode] ok.")
    print(f"  raw_episode dir : {result.raw_episode_dir}")
    print(f"  visual input    : {result.visual_input_path}")
    print(f"  run dir         : {result.run_dir}")
    print(f"  chosen camera   : {result.chosen_serial} ({result.chosen_role})")
    print(f"  task            : {result.task_description!r}")
    print(f"  trajectory len  : {result.n_traj_frames}")
    print()
    print("Run the full DROID pipeline with:")
    print(f"  python -m ar2s.run_pipeline --input {result.visual_input_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
