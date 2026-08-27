"""Convert AR2S self-collected episodes into DROID-shaped raw_data folders.

Input — one AR2S collection drop (see ``scripts.collected_episode_common``)::

    <drop>/.../<n>/{camera.svo2, robot.mcap, episode_manifest.json,
                    station_calibration.json}

Output — exactly the layout ``ar2s.agents_visual.raw_episode`` documents::

    <out_root>/<episode_id>/
    ├── <serial>-stereo.mp4         rectified side-by-side stereo from the SVO
    ├── <serial>-stereo.K.txt       left-camera K (row-major) + baseline in m
    ├── trajectory.h5               action/joint_position + action/gripper_position
    ├── metadata_<episode_id>.json  task text + ext1_cam_serial
    ├── <episode_id>_cameras.json   refined_extrinsics (w2c) per serial
    └── stage_manifest.json         episode_id + provenance

From there the episode is indistinguishable from a staged DROID episode::

    python -m ar2s.run_pipeline --raw-data <out_root>/<episode_id>

Robot state is paired to camera frames by nearest timestamp, the rule
``episode_manifest.json`` declares: the SVO's per-frame image timestamps
against the MCAP publish times, both in ns.

This step needs the ZED SDK (pyzed) and a CUDA GPU for SVO rectification, so
pyzed is imported at module load and the GPU is checked before any work — same
contract as ``scripts.extract_stereo_from_svo``. Everything else lives in
``scripts.collected_episode_common``, which imports cleanly without either.

Usage::

    python -m scripts.export_collected_episode <episode_or_drop_dir> \\
        --out-root droid_sim_data/real_episodes/raw_data

    # pin the pipeline-facing name instead of deriving it from the manifest
    python -m scripts.export_collected_episode <episode_dir> \\
        --out-root <root> --episode-id AIRBOT_success_2026_08_03_21_21_47
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyzed.sl  # noqa: F401  (import early: a broken SDK must fail before any work)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ar2s.agents_visual.droid_episode_id import validate_episode_id
from scripts.collected_episode_common import (
    CollectedEpisode,
    coverage_gap_ns,
    find_collected_episodes,
    load_collected_episode,
    normalize_gripper,
    pair_nearest,
    read_mcap_state_stream,
    write_cameras_json,
    write_metadata_json,
    write_stage_manifest,
    write_trajectory_h5,
)
from scripts.droid_raw_common import MISSING_PREREQ_EXIT
from scripts.export_episodes_from_raw import (
    close_mp4_writer,
    get_svo_camera_metadata,
    open_mp4_writer,
    save_intrinsics,
    write_frame,
    zed_image_to_bgr,
)
from scripts.extract_stereo_from_svo import check_prerequisites
from scripts.svo_reader import SVOReader

# A camera frame paired with a robot sample this far away is not synchronised
# in any useful sense; both recordings run well above 10 Hz.
_MAX_PAIRING_GAP_NS = 100_000_000


def extract_stereo_with_timestamps(
    svo_path: Path,
    output_path: Path,
    camera_serial: str,
) -> np.ndarray:
    """Write ``<serial>-stereo.mp4`` + ``.K.txt``; return per-frame image ns.

    Mirrors ``export_episodes_from_raw.extract_stereo_mp4_from_svo`` — same
    reader parameters, same writer, same intrinsics sidecar — and additionally
    records the image timestamp of every frame written, which is what the
    robot-state pairing needs and what a second decode pass would have to
    reproduce exactly.
    """
    reader = SVOReader(str(svo_path), camera_serial)
    writer = None
    timestamps: list[int] = []
    try:
        reader.set_reading_parameters(
            image=True,
            depth=False,
            pointcloud=False,
            concatenate_images=False,
            rectified_images=True,
            resolution=(0, 0),
            resize_func=None,
        )
        fps, K, baseline = get_svo_camera_metadata(reader)

        left_key = f"{camera_serial}_left"
        right_key = f"{camera_serial}_right"

        while True:
            data = reader.read_camera()
            if data is None:
                break
            left = data.get("image", {}).get(left_key)
            right = data.get("image", {}).get(right_key)
            if left is None or right is None:
                raise RuntimeError(f"Missing stereo image data while reading {svo_path}")
            frame = np.concatenate(
                [zed_image_to_bgr(left), zed_image_to_bgr(right)], axis=1
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = open_mp4_writer(output_path, fps, (width, height))
            write_frame(writer, frame)
            timestamps.append(int(reader.get_image_timestamp_ns()))

        if writer is None:
            raise RuntimeError(
                f"Failed to read any stereo frame from {svo_path}. "
                "Check pyzed/ZED SDK installation and GPU availability."
            )
    finally:
        try:
            if writer is not None:
                close_mp4_writer(writer, output_path)
        finally:
            reader.disable_camera()

    save_intrinsics(K, baseline, output_path.with_suffix(".K.txt"))
    return np.asarray(timestamps, dtype=np.int64)


def export_episode(
    episode: CollectedEpisode,
    out_root: Path,
    *,
    episode_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert one collected episode; returns the raw_data folder written."""
    episode_id = (
        validate_episode_id(episode_id) if episode_id else episode.default_episode_id
    )
    out_dir = Path(out_root) / episode_id
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"[collected] episode dir: {episode.episode_dir}")
    print(f"[collected] episode id:  {episode_id} "
          f"(manifest: {episode.manifest_episode_id})")
    print(f"[collected] task:        {episode.task_description}")
    print(f"[collected] robot:       {episode.robot_model} -> profile "
          f"{episode.robot_type} ({len(episode.arm_joint_indices)} arm joints)")

    # ---- camera ----
    mp4_path = out_dir / f"{episode.camera_serial}-stereo.mp4"
    print(f"[collected] extracting {episode.svo_path.name} -> {mp4_path.name}")
    camera_ts = extract_stereo_with_timestamps(
        episode.svo_path, mp4_path, episode.camera_serial
    )
    print(f"[collected] wrote {mp4_path.name}: {camera_ts.size} frames, "
          f"image ts {camera_ts[0]}..{camera_ts[-1]} ns")

    # ---- robot ----
    robot_ts, state = read_mcap_state_stream(
        episode.mcap_path, episode.state_topic, vector_length=episode.vector_length
    )
    print(f"[collected] read {episode.mcap_path.name} {episode.state_topic}: "
          f"{state.shape[0]} samples, publish ts {robot_ts[0]}..{robot_ts[-1]} ns")

    gap_ns = coverage_gap_ns(camera_ts, robot_ts)
    print(f"[collected] worst camera->robot pairing gap: {gap_ns / 1e6:.1f} ms")
    if gap_ns > _MAX_PAIRING_GAP_NS:
        raise RuntimeError(
            f"worst pairing gap is {gap_ns / 1e6:.1f} ms — the camera and robot "
            f"recordings do not overlap closely enough to pair by nearest "
            f"timestamp. Check that both files come from the same take."
        )

    paired = state[pair_nearest(camera_ts, robot_ts)]
    arm = paired[:, episode.arm_joint_indices]
    gripper = normalize_gripper(paired[:, episode.gripper_index], episode.gripper_range)
    traj_path = out_dir / "trajectory.h5"
    write_trajectory_h5(traj_path, arm, gripper)
    print(f"[collected] wrote {traj_path.name}: {arm.shape[0]} steps, "
          f"{arm.shape[1]} arm joints, gripper norm "
          f"[{gripper.min():.4f}, {gripper.max():.4f}]")

    # ---- metadata + extrinsics ----
    write_metadata_json(
        out_dir / f"metadata_{episode_id}.json",
        task_description=episode.task_description,
        camera_serial=episode.camera_serial,
    )
    calibration = json.loads(episode.calibration_path.read_text(encoding="utf-8"))
    mat = write_cameras_json(
        out_dir / f"{episode_id}_cameras.json",
        camera_serial=episode.camera_serial,
        calibration=calibration,
    )
    print(f"[collected] wrote {episode_id}_cameras.json: cam_mat_"
          f"{episode.camera_serial} translation {mat[:3, 3]}")

    write_stage_manifest(out_dir / "stage_manifest.json", {
        "episode_id": episode_id,
        "source": "ar2s_collected",
        "collected_episode_dir": str(episode.episode_dir),
        "manifest_episode_id": episode.manifest_episode_id,
        "task_description": episode.task_description,
        "robot": episode.robot_model,
        # RobotProfile registry name. build_raw_episode reads this key and
        # bakes it into the VisualInput stub, from where it reaches
        # manifest.robot.type — the value sysid uses to pick the arm model.
        "robot_type": episode.robot_type,
        "arm_dof": len(episode.arm_joint_indices),
        "gripper_range": list(episode.gripper_range),
        "gripper_unit": episode.gripper_unit,
        "camera_serial": episode.camera_serial,
        "n_frames": int(camera_ts.size),
        "n_robot_samples": int(robot_ts.size),
        "worst_pairing_gap_ns": gap_ns,
        "stage_dir": str(out_dir),
    })

    print(f"[collected] DONE: {out_dir}")
    print(f"[collected] next: python -m ar2s.run_pipeline --raw-data {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AR2S self-collected episodes (SVO2 + MCAP) into the "
            "DROID-shaped raw_data folders ar2s.run_pipeline --raw-data consumes."
        )
    )
    parser.add_argument(
        "source",
        nargs="+",
        help=(
            "Collected-episode folder(s), or any parent of them (a whole "
            "collection drop is fine — every episode_manifest.json underneath "
            "is converted)."
        ),
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Directory to write <episode_id>/ raw_data folders into.",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help=(
            "Pipeline-facing episode id. Default: derived from the manifest's "
            "episode_id. Only valid when converting a single episode."
        ),
    )
    parser.add_argument(
        "--robot-type",
        default=None,
        help=(
            "RobotProfile registry name (e.g. airbot_play). Default: derived "
            "from episode_manifest.json robot.model. A wrong value silently "
            "simulates the wrong arm, so an underivable model is an error "
            "rather than a Franka fallback."
        ),
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing <out_root>/<episode_id>/.")
    args = parser.parse_args()

    check_prerequisites()

    episode_dirs: list[Path] = []
    for src in args.source:
        found = find_collected_episodes(Path(src))
        if not found:
            print(f"no episode_manifest.json at or under {src}", file=sys.stderr)
            sys.exit(MISSING_PREREQ_EXIT)
        episode_dirs.extend(found)

    if args.episode_id and len(episode_dirs) > 1:
        print(
            f"--episode-id names one episode but {len(episode_dirs)} were found: "
            + ", ".join(str(d) for d in episode_dirs),
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"[collected] {len(episode_dirs)} episode(s) to convert")
    for episode_dir in episode_dirs:
        export_episode(
            load_collected_episode(episode_dir, robot_type=args.robot_type),
            Path(args.out_root),
            episode_id=args.episode_id,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
