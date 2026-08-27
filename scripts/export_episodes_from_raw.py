# Author: Guanxiong Chen


import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.svo_reader import SVOReader
from scripts.droid_raw_common import (
    find_svo_path,
    load_raw_ids,
    normalize_serial,
    pointworld_cameras_path,
)


def zed_image_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        raise ValueError(f"Expected HxWxC or HxW frame, got shape {frame.shape}")
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] == 3:
        return frame.copy()
    raise ValueError(f"Expected 3 or 4 channels, got shape {frame.shape}")


def get_svo_camera_metadata(reader: SVOReader):
    camera_info = reader._cam.get_camera_information()
    if hasattr(camera_info, "camera_configuration"):
        camera_cfg = camera_info.camera_configuration
        fps = camera_cfg.fps
        calib_params = camera_cfg.calibration_parameters
    else:
        fps = camera_info.camera_fps
        calib_params = camera_info.calibration_parameters

    left_cam = calib_params.left_cam
    K = np.array(
        [
            [left_cam.fx, 0, left_cam.cx],
            [0, left_cam.fy, left_cam.cy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    baseline = abs(calib_params.get_camera_baseline()) / 1000.0
    return float(fps), K, baseline


def save_intrinsics(K: np.ndarray, baseline: float, output_path: Path) -> None:
    assert K.shape == (3, 3), K.shape
    assert baseline > 0, baseline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(map(str, K.flatten())) + "\n")
        f.write(f"{baseline}\n")


def open_mp4_writer(output_path: Path, fps: float, size: tuple[int, int]):
    width, height = size
    assert width > 0 and height > 0, size
    assert fps > 0, fps
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "-", "-an",
        "-vf", "scale=in_range=full:out_range=full",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "medium",
        "-g", "400", "-keyint_min", "400",
        "-color_range", "pc",
        "-movflags", "+faststart",
        str(output_path),
    ]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert writer.stdin is not None
    return writer


def write_frame(writer, frame: np.ndarray) -> None:
    assert writer.stdin is not None
    assert frame.ndim == 3 and frame.shape[2] == 3, frame.shape
    writer.stdin.write(np.ascontiguousarray(frame).tobytes())


def close_mp4_writer(writer, output_path: Path) -> None:
    assert writer.stdin is not None
    writer.stdin.close()
    returncode = writer.wait()
    if returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with return code {returncode} while writing {output_path}"
        )


def extract_stereo_mp4_from_svo(
    svo_path: Path,
    output_path: Path,
    camera_serial: str,
) -> None:
    reader = SVOReader(str(svo_path), camera_serial)
    writer = None
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
        intrinsics_output_path = output_path.with_suffix(".K.txt")

        left_key = f"{camera_serial}_left"
        right_key = f"{camera_serial}_right"

        data = reader.read_camera()
        if not data or "image" not in data:
            raise RuntimeError(
                f"Failed to read stereo frame from {svo_path}. "
                "Check pyzed/ZED SDK installation and GPU availability."
            )

        left = data["image"].get(left_key)
        right = data["image"].get(right_key)
        if left is None or right is None:
            raise RuntimeError(f"Missing stereo image data while reading {svo_path}")

        first_frame = np.concatenate(
            [zed_image_to_bgr(left), zed_image_to_bgr(right)],
            axis=1,
        )
        height, width = first_frame.shape[:2]
        writer = open_mp4_writer(output_path, fps, (width, height))

        write_frame(writer, first_frame)
        while True:
            data = reader.read_camera()
            if data is None:
                break
            left = data.get("image", {}).get(left_key)
            right = data.get("image", {}).get(right_key)
            if left is None or right is None:
                raise RuntimeError(f"Missing stereo image data while reading {svo_path}")
            frame = np.concatenate(
                [zed_image_to_bgr(left), zed_image_to_bgr(right)],
                axis=1,
            )
            write_frame(writer, frame)
    finally:
        try:
            if writer is not None:
                close_mp4_writer(writer, output_path)
        finally:
            reader.disable_camera()
    save_intrinsics(K, baseline, intrinsics_output_path)


def main():
    parser = argparse.ArgumentParser(
    description="Inspect raw DROID dataset and export episode-wise data.")
    parser.add_argument(
        "--generate_video", action="store_true") # extract stereo MP4s from SVO
    parser.add_argument(
        "--export_extrinsics", action="store_true") # export pointworld extrinsics
    parser.add_argument(
        "--export_intrinsics", action="store_true") # export pointworld intrinsics
    parser.add_argument(
        "--raw_ids_file",
        default=None,
        help="Optional newline-delimited raw episode ids to export.",
    )
    args = parser.parse_args()

    # confirm data exists
    data_dir = Path("droid_data/raw")
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    # create output dir first
    output_dir = Path("outputs/export_episodes_from_raw")
    os.makedirs(output_dir, exist_ok=True)
    raw_ids = load_raw_ids(Path(args.raw_ids_file) if args.raw_ids_file else None)
    pointworld_root = Path("droid_data/pointworld_cam_extrinsics")

    for episode_dir in sorted(
        (p for p in data_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    ):
        if raw_ids is not None and episode_dir.name not in raw_ids:
            continue

        # read episode metadata
        metadata_files = list(episode_dir.glob("*.json"))
        if len(metadata_files) != 1:
            print(
                f"Skipping {episode_dir.name}: expected 1 metadata json, found {len(metadata_files)}"
            )
            continue
        with metadata_files[0].open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        try:
            serials = [
                metadata["wrist_cam_serial"],
                metadata["ext1_cam_serial"],
                metadata["ext2_cam_serial"],
            ] # always in wrist, ext1, ext2 order
        except KeyError as exc:
            print(f"Skipping {episode_dir.name}: missing field {exc}")
            continue

        camera_serials = [normalize_serial(s) for s in serials]

        if args.generate_video:
            # Generate stereo MP4s from SVO only. Do not use existing stereo
            # MP4s as a fallback, and do not back-fill droid_data/raw.
            out_dir = output_dir / episode_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            camera_output_names = ["wrist", "ext1", "ext2"]
            for camera_name, camera_serial in zip(camera_output_names, camera_serials):
                svo_path = find_svo_path(episode_dir, camera_serial)
                if svo_path is None:
                    print(
                        f"Skipping {episode_dir.name} camera {camera_name} "
                        f"{camera_serial}: missing SVO"
                    )
                    continue

                output_path = out_dir / f"{camera_serial}-stereo.mp4"
                try:
                    print(
                        f"Extracting stereo MP4 for {episode_dir.name} camera "
                        f"{camera_name} {camera_serial} from {svo_path.name}"
                    )
                    extract_stereo_mp4_from_svo(
                        svo_path=svo_path,
                        output_path=output_path,
                        camera_serial=camera_serial,
                    )
                except Exception as exc:
                    print(
                        f"Failed to extract stereo MP4 for {episode_dir.name} "
                        f"camera {camera_name} {camera_serial}: {exc}"
                    )

        cameras_path = None
        cameras = None
        if args.export_extrinsics or args.export_intrinsics:
            cameras_path, reason = pointworld_cameras_path(
                episode_dir,
                metadata,
                pointworld_root=pointworld_root,
            )
            if cameras_path is None:
                print(f"Skipping calibration for {episode_dir.name}: {reason}")
                continue
            out_dir = output_dir / episode_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cameras_path, out_dir / cameras_path.name)

            with cameras_path.open("r", encoding="utf-8") as f:
                cameras = json.load(f)

        if args.export_extrinsics:
            # get the robot base->cam frame matrix for the two external
            # cameras (wrist cam per-frame matrix is not available in the
            # PointWorld extrinsics json, so we skip it here)
            cam_mats = {}
            for serial in camera_serials[1:]:
                data = cameras.get(str(serial))
                if not isinstance(data, dict):
                    print(
                        f"Skipping camera {serial} in {cameras_path}: expected dict, got {type(data)}"
                    )
                    continue
                refined = data.get("refined_extrinsics")
                if refined is None:
                    print(
                        f"Skipping camera {serial}: missing 'refined_extrinsics' in {cameras_path}"
                    )
                    continue
                cam_key = f"cam_mat_{serial}"
                cam_mats[cam_key] = np.asarray(refined, dtype=np.float32)
            if cam_mats:
                npz_path = out_dir / f"{cameras_path.stem}_extrinsics.npz"
                np.savez(npz_path, **cam_mats)
            else:
                print(f"Skipping extrinsics NPZ for {episode_dir.name}: no refined external cameras")

        if args.export_intrinsics:
            # get the intrinsics (K) matrix
            intrinsics_mats = {}
            for serial in camera_serials[1:]:
                data = cameras.get(str(serial))
                if not isinstance(data, dict):
                    print(
                        f"Skipping camera {serial} intrinsics in {cameras_path}: expected dict, got {type(data)}"
                    )
                    continue
                vggt_intrinsics = data.get("vggt_intrinsics")
                if vggt_intrinsics is None:
                    print(
                        f"Missing 'vggt_intrinsics' for camera {serial} in {cameras_path}"
                    )
                    continue
                cam_key = f"cam_K_{serial}"
                K = np.asarray(vggt_intrinsics, dtype=np.float32)
                if K.shape != (3, 3):
                    raise ValueError(
                        f"Expected 3x3 'vggt_intrinsics' for camera {serial} in {cameras_path}, got shape {K.shape}"
                    )
                intrinsics_mats[cam_key] = K
            if intrinsics_mats:
                intrinsics_npz_path = out_dir / f"{cameras_path.stem}_intrinsics.npz"
                np.savez(intrinsics_npz_path, **intrinsics_mats)
            else:
                print(f"Skipping intrinsics NPZ for {episode_dir.name}: no external VGGT intrinsics")


if __name__ == "__main__":
    main()
