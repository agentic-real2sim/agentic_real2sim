# Inspect stereo data directly from SVO.
#
# This script:
# 1. Opens SVO and extracts stream metadata (frame count, FPS, resolution)
# 2. Extracts intrinsics and baseline
# 3. Iterates stereo frames and prints basic per-frame stats
# 4. Saves left/right images

import os
import argparse
import logging
from copy import deepcopy

import cv2
import numpy as np
import imageio
from tqdm import tqdm

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    sl = None


class SVOReader:
    """Minimal local SVO reader for image stream inspection."""

    def __init__(self, filepath, serial_number):
        if sl is None:
            raise RuntimeError(
                "pyzed is not installed. Install ZED SDK Python API (pyzed) to read SVO files."
            )
        self.serial_number = serial_number
        self._index = 0
        self._sbs_img = sl.Mat()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._cam = sl.Camera()

        init_parameters = sl.InitParameters()
        init_parameters.camera_image_flip = sl.FLIP_MODE.OFF
        init_parameters.set_from_svo_file(filepath)
        status = self._cam.open(init_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO file: {repr(status)}")

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        self.image = image
        self.depth = depth
        self.pointcloud = pointcloud
        self.concatenate_images = concatenate_images
        self.resize_func = cv2.resize if resize_func == "cv2" else None

        if self.resize_func is None:
            self.zed_resolution = sl.Resolution(*resolution)
            self.resizer_resolution = (0, 0)
        else:
            self.zed_resolution = sl.Resolution(0, 0)
            self.resizer_resolution = resolution

        self.skip_reading = not any([image, depth, pointcloud])

    def get_frame_resolution(self):
        camera_cfg = self._cam.get_camera_information().camera_configuration
        return (camera_cfg.resolution.width, camera_cfg.resolution.height)

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        return self._cam.get_svo_number_of_frames()

    def _process_frame(self, frame):
        out = deepcopy(frame.get_data())
        if self.resizer_resolution == (0, 0):
            return out
        return self.resize_func(out, self.resizer_resolution)

    def read_camera(self):
        if self.skip_reading:
            return {}
        self._index += 1
        err = self._cam.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            return None

        data_dict = {}
        if self.image:
            if self.concatenate_images:
                self._cam.retrieve_image(self._sbs_img, sl.VIEW.SIDE_BY_SIDE, resolution=self.zed_resolution)
                data_dict["image"] = {self.serial_number: self._process_frame(self._sbs_img)}
            else:
                self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=self.zed_resolution)
                self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT, resolution=self.zed_resolution)
                data_dict["image"] = {
                    self.serial_number + "_left": self._process_frame(self._left_img),
                    self.serial_number + "_right": self._process_frame(self._right_img),
                }
        return data_dict

    def disable_camera(self):
        if hasattr(self, "_cam"):
            self._cam.close()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def save_intrinsics(K, baseline, output_path):
    with open(output_path, "w") as f:
        f.write(" ".join(map(str, K.flatten())) + "\n")
        f.write(f"{baseline}\n")
    logging.info("Intrinsics saved to %s", output_path)


def zed_image_to_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.copy()


def create_svo_reader_bundle(svo_path):
    serial_number = os.path.splitext(os.path.basename(svo_path))[0]
    reader = SVOReader(filepath=svo_path, serial_number=serial_number)
    reader.set_reading_parameters(
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    )

    width, height = reader.get_frame_resolution()
    total_frames = reader.get_frame_count()

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

    return {
        "reader": reader,
        "serial_number": serial_number,
        "left_key": f"{serial_number}_left",
        "right_key": f"{serial_number}_right",
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "K": K,
        "baseline": baseline,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect stereo data loaded from SVO")
    parser.add_argument("--svo_file", type=str, required=True, help="Path to the SVO file")
    parser.add_argument(
        "--intrinsic_file",
        type=str,
        default=None,
        help="Path to a pre-existing K.txt file (override SVO intrinsics)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "output_svo_inspect"),
        help="Output directory",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Max number of frames to inspect")
    parser.add_argument("--frame_step", type=int, default=1, help="Inspect every N-th frame")
    parser.add_argument("--print_stats", type=int, default=1, help="Print per-frame image stats")
    parser.add_argument("--stats_every", type=int, default=1, help="Print stats every N processed frames")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.out_dir, exist_ok=True)

    logging.info("Initializing SVOReader...")
    svo = create_svo_reader_bundle(args.svo_file)
    reader = svo["reader"]
    total_frames = svo["total_frames"]
    fps = svo["fps"]
    width = svo["width"]
    height = svo["height"]
    left_key = svo["left_key"]
    right_key = svo["right_key"]

    logging.info("SVO: %s", args.svo_file)
    logging.info("Serial: %s", svo["serial_number"])
    logging.info("Total frames: %d, FPS: %s", total_frames, fps)
    logging.info("Single-eye frame size: %dx%d", width, height)

    if args.intrinsic_file is not None:
        logging.info("Loading intrinsics from %s", args.intrinsic_file)
        with open(args.intrinsic_file, "r") as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
            baseline = float(lines[1])
    else:
        K = svo["K"]
        baseline = svo["baseline"]

    intrinsic_output = os.path.join(args.out_dir, "K.txt")
    save_intrinsics(K, baseline, intrinsic_output)
    logging.info("Camera intrinsic matrix:\n%s", K)
    logging.info("Baseline: %.6f m", baseline)

    frames_dir = os.path.join(args.out_dir, "frames")
    left_dir = os.path.join(frames_dir, "left")
    right_dir = os.path.join(frames_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    processed_count = 0
    sampled_total = (total_frames + args.frame_step - 1) // args.frame_step
    target_total = sampled_total if args.max_frames is None else min(args.max_frames, sampled_total)
    pbar = tqdm(total=target_total, desc="Inspecting SVO frames")

    try:
        for raw_frame_idx in range(total_frames):
            data = reader.read_camera()
            if data is None:
                break
            if raw_frame_idx % args.frame_step != 0:
                continue

            left = zed_image_to_rgb(data["image"][left_key])
            right = zed_image_to_rgb(data["image"][right_key])

            if args.print_stats and (processed_count % max(args.stats_every, 1) == 0):
                logging.info(
                    "frame=%06d | left shape=%s dtype=%s min=%d max=%d | right shape=%s dtype=%s min=%d max=%d",
                    raw_frame_idx,
                    tuple(left.shape),
                    left.dtype,
                    int(left.min()),
                    int(left.max()),
                    tuple(right.shape),
                    right.dtype,
                    int(right.min()),
                    int(right.max()),
                )

            imageio.imwrite(os.path.join(left_dir, f"frame_{raw_frame_idx:06d}.png"), left)
            imageio.imwrite(os.path.join(right_dir, f"frame_{raw_frame_idx:06d}.png"), right)

            processed_count += 1
            pbar.update(1)
            if args.max_frames is not None and processed_count >= args.max_frames:
                break
    finally:
        reader.disable_camera()
        pbar.close()

    logging.info("Done! Inspected %d frames.", processed_count)
    logging.info("Results saved to %s", args.out_dir)
    logging.info("  - Intrinsics: %s", intrinsic_output)
    logging.info("  - Left frames: %s", left_dir)
    logging.info("  - Right frames: %s", right_dir)


if __name__ == "__main__":
    main()
