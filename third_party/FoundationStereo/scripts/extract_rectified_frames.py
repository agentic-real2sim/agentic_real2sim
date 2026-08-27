# Extract rectified stereo frames from DROID side-by-side MP4.
#
# This script:
# 1. Fetches factory calibration from ZED online API (no pyzed/ZED SDK needed)
# 2. Computes stereo rectification maps via cv2.stereoRectify
# 3. Splits side-by-side frames, applies undistort + rectify
# 4. Saves rectified left/right PNGs and rectified K.txt (same format as read_from_svo.py)
#
# Usage:
#   module load conda && conda activate foundation_stereo
#   python extract_rectified_frames.py --video_file ../episode_4_raw_videos/25947356-stereo.mp4

import os
import argparse
import logging
import urllib.request

import cv2
import numpy as np
import imageio
from tqdm import tqdm


# ZED camera resolution presets (single-eye width x height)
ZED_RESOLUTIONS = {
    '2K':  (2208, 1242),
    'FHD': (1920, 1080),
    'HD':  (1280, 720),
    'VGA': (672,  376),
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def detect_zed_resolution(single_eye_width, single_eye_height):
    for name, (w, h) in ZED_RESOLUTIONS.items():
        if single_eye_width == w and single_eye_height == h:
            return name
    best_name = min(ZED_RESOLUTIONS, key=lambda n: abs(ZED_RESOLUTIONS[n][1] - single_eye_height))
    logging.warning(
        f"Exact resolution match not found for {single_eye_width}x{single_eye_height}, "
        f"using closest preset: {best_name} ({ZED_RESOLUTIONS[best_name]})"
    )
    return best_name


def fetch_zed_calibration(serial_number):
    url = f"https://www.stereolabs.com/developers/calib/?SN={serial_number}"
    logging.info(f"Fetching ZED calibration from {url}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode('utf-8')


def parse_zed_calibration(calib_text, resolution_name):
    """
    Parse ZED online calibration to extract:
      - Left/right intrinsics (K) and distortion (D)
      - Stereo rotation (R) and translation (T)
      - Baseline in meters

    Calibration sections used:
      [LEFT_CAM_{res}]  -> fx, fy, cx, cy, k1, k2, p1, p2, k3
      [RIGHT_CAM_{res}] -> fx, fy, cx, cy, k1, k2, p1, p2, k3
      [STEREO]          -> Baseline, TY, TZ, RX_{res}, CV_{res}, RZ_{res}
    """
    sections = {}
    current_section = None
    for line in calib_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            sections[current_section] = {}
        elif '=' in line and current_section is not None:
            key, value = line.split('=', 1)
            sections[current_section][key.strip()] = value.strip()

    suffix = resolution_name

    # --- Left camera ---
    left_sec = sections[f'LEFT_CAM_{suffix}']
    K_left = np.array([
        [float(left_sec['fx']), 0, float(left_sec['cx'])],
        [0, float(left_sec['fy']), float(left_sec['cy'])],
        [0, 0, 1],
    ], dtype=np.float64)
    D_left = np.array([
        float(left_sec['k1']), float(left_sec['k2']),
        float(left_sec['p1']), float(left_sec['p2']),
        float(left_sec['k3']),
    ], dtype=np.float64)

    # --- Right camera ---
    right_sec = sections[f'RIGHT_CAM_{suffix}']
    K_right = np.array([
        [float(right_sec['fx']), 0, float(right_sec['cx'])],
        [0, float(right_sec['fy']), float(right_sec['cy'])],
        [0, 0, 1],
    ], dtype=np.float64)
    D_right = np.array([
        float(right_sec['k1']), float(right_sec['k2']),
        float(right_sec['p1']), float(right_sec['p2']),
        float(right_sec['k3']),
    ], dtype=np.float64)

    # --- Stereo extrinsics ---
    stereo_sec = sections['STEREO']
    baseline_mm = float(stereo_sec['Baseline'])
    TY = float(stereo_sec['TY'])
    TZ = float(stereo_sec['TZ'])

    # Rotation between cameras (small angles in radians)
    rx = float(stereo_sec[f'RX_{suffix}'])
    ry = float(stereo_sec[f'CV_{suffix}'])   # convergence = rotation around Y
    rz = float(stereo_sec[f'RZ_{suffix}'])

    # Rotation vector -> rotation matrix
    rvec = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)

    # Translation: left-to-right, baseline is along negative X in ZED convention
    T = np.array([-baseline_mm, TY, TZ], dtype=np.float64)

    baseline_m = baseline_mm / 1000.0

    return K_left, D_left, K_right, D_right, R, T, baseline_m


def compute_rectification(K_left, D_left, K_right, D_right, R, T, image_size):
    """
    Compute stereo rectification transforms and rectified intrinsics.

    Returns:
        map_left:  (map1, map2) remap arrays for left camera
        map_right: (map1, map2) remap arrays for right camera
        K_rect:    3x3 rectified intrinsic matrix (from left projection matrix)
        baseline:  rectified baseline in meters (from P2)
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D_left, K_right, D_right,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,  # 0 = crop to valid pixels only
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, image_size, cv2.CV_32FC1,
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, image_size, cv2.CV_32FC1,
    )

    # Rectified intrinsics from left projection matrix P1 (3x4)
    # P1 = K_rect @ [I | 0] so K_rect = P1[:3, :3]
    K_rect = P1[:3, :3].astype(np.float32)

    # Rectified baseline from P2: P2[0,3] = -fx * Tx  =>  Tx = -P2[0,3] / P2[0,0]
    baseline_rect = abs(P2[0, 3] / P2[0, 0])  # in same units as T (mm)
    baseline_rect_m = baseline_rect / 1000.0

    logging.info(f"Rectification ROI left:  {roi1}")
    logging.info(f"Rectification ROI right: {roi2}")
    logging.info(f"Rectified K:\n{K_rect}")
    logging.info(f"Rectified baseline: {baseline_rect_m:.6f} m")

    return (map1_left, map2_left), (map1_right, map2_right), K_rect, baseline_rect_m


def save_intrinsics(K, baseline, output_path):
    with open(output_path, 'w') as f:
        f.write(' '.join(map(str, K.flatten())) + '\n')
        f.write(f'{baseline}\n')
    logging.info(f"Intrinsics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract rectified stereo frames from DROID side-by-side MP4'
    )
    parser.add_argument('--video_file', type=str, required=True,
                        help='Path to the side-by-side stereo MP4 (e.g., 25947356-stereo.mp4)')
    parser.add_argument('--serial_number', type=str, default=None,
                        help='ZED serial number (auto-detected from filename if omitted)')
    parser.add_argument('--intrinsic_file', type=str, default=None,
                        help='Pre-existing K.txt to use instead of online API')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: ../output_droid_<serial>)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to extract')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Extract every N-th frame')
    parser.add_argument('--save_video', type=int, default=1,
                        help='Also save rectified left frames as an MP4 video')
    parser.add_argument('--video_crf', type=int, default=15,
                        help='H.264 CRF quality for output video (lower = better, 0 = lossless)')
    args = parser.parse_args()

    setup_logging()

    # --- Open video ---
    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_file}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    single_w = vid_width // 2
    single_h = vid_height

    logging.info(f"Video: {args.video_file}")
    logging.info(f"Total frames: {total_frames}, FPS: {fps}")
    logging.info(f"Side-by-side: {vid_width}x{vid_height} -> each eye: {single_w}x{single_h}")

    # --- Serial number ---
    serial_number = args.serial_number
    if serial_number is None:
        basename = os.path.splitext(os.path.basename(args.video_file))[0]
        serial_number = basename.split('-')[0]
        logging.info(f"Auto-detected serial number: {serial_number}")

    # --- Output directory ---
    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        args.out_dir = os.path.join(script_dir, '..', f'output_droid_{serial_number}')
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Calibration & rectification ---
    if args.intrinsic_file is not None:
        logging.info(f"Loading intrinsics from {args.intrinsic_file}")
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K_rect = np.array(list(map(float, lines[0].split())), dtype=np.float32).reshape(3, 3)
            baseline = float(lines[1])
        map_left = map_right = None
        logging.info("Using provided intrinsics (no rectification will be applied)")
    else:
        resolution_name = detect_zed_resolution(single_w, single_h)
        calib_text = fetch_zed_calibration(serial_number)
        K_left, D_left, K_right, D_right, R, T, baseline = parse_zed_calibration(
            calib_text, resolution_name
        )

        logging.info(f"Unrectified left  K:\n{K_left}")
        logging.info(f"Unrectified left  D: {D_left}")
        logging.info(f"Unrectified right K:\n{K_right}")
        logging.info(f"Unrectified right D: {D_right}")
        logging.info(f"R:\n{R}")
        logging.info(f"T: {T}")

        map_left, map_right, K_rect, baseline = compute_rectification(
            K_left, D_left, K_right, D_right, R, T, (single_w, single_h)
        )

    # --- Save intrinsics ---
    intrinsic_output = os.path.join(args.out_dir, 'K.txt')
    save_intrinsics(K_rect, baseline, intrinsic_output)

    # --- Output directories ---
    left_dir = os.path.join(args.out_dir, 'frames', 'left')
    right_dir = os.path.join(args.out_dir, 'frames', 'right')
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    # --- Video writer (optional) ---
    video_writer = None
    video_path = None
    if args.save_video:
        video_path = os.path.join(args.out_dir, 'rectified_left.mp4')
        effective_fps = fps / args.frame_step
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, effective_fps, (single_w, single_h))
        logging.info(f"Will save rectified video to {video_path} (fps={effective_fps:.1f})")

    # --- Extract frames ---
    sampled_total = (total_frames + args.frame_step - 1) // args.frame_step
    target_total = sampled_total if args.max_frames is None else min(args.max_frames, sampled_total)
    pbar = tqdm(total=target_total, desc="Extracting rectified frames")

    processed = 0
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_step != 0:
            continue

        # Split side-by-side
        left_bgr = frame[:, :single_w, :]
        right_bgr = frame[:, single_w:, :]

        # Rectify
        if map_left is not None:
            left_bgr = cv2.remap(left_bgr, map_left[0], map_left[1], cv2.INTER_LINEAR)
            right_bgr = cv2.remap(right_bgr, map_right[0], map_right[1], cv2.INTER_LINEAR)

        # BGR -> RGB for saving PNGs
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        imageio.imwrite(os.path.join(left_dir, f'frame_{frame_idx:06d}.png'), left_rgb)
        imageio.imwrite(os.path.join(right_dir, f'frame_{frame_idx:06d}.png'), right_rgb)

        if video_writer is not None:
            video_writer.write(left_bgr)

        processed += 1
        pbar.update(1)
        if args.max_frames is not None and processed >= args.max_frames:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        # Re-encode with ffmpeg for better compression and compatibility
        final_video_path = video_path.replace('.mp4', '_final.mp4')
        effective_fps = fps / args.frame_step
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{video_path}" -c:v libx264 -pix_fmt yuv420p '
            f'-crf {args.video_crf} -r {effective_fps} "{final_video_path}" -loglevel warning'
        )
        logging.info(f"Re-encoding with ffmpeg for better compatibility...")
        ret = os.system(ffmpeg_cmd)
        if ret == 0:
            os.replace(final_video_path, video_path)
            logging.info(f"Rectified video: {video_path}")
        else:
            logging.warning(f"ffmpeg re-encode failed (code {ret}), keeping cv2 output: {video_path}")
            if os.path.exists(final_video_path):
                os.remove(final_video_path)

    pbar.close()

    logging.info(f"Done! Extracted {processed} rectified frame pairs.")
    logging.info(f"  - Intrinsics: {intrinsic_output}")
    logging.info(f"  - Left frames:  {left_dir}")
    logging.info(f"  - Right frames: {right_dir}")
    if video_path:
        logging.info(f"  - Video: {video_path}")


if __name__ == '__main__':
    main()
