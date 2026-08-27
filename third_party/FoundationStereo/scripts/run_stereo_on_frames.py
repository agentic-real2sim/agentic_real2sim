# Run FoundationStereo on pre-extracted rectified frame pairs.
#
# Input:  output of extract_rectified_frames.py (or read_from_svo.py)
#         <sequence_dir>/
#           K.txt                   (rectified intrinsics + baseline)
#           frames/left/*.png
#           frames/right/*.png
#
# Output: depth maps, disparity visualisations, and (optionally) point clouds
#         written into the same <sequence_dir>.
#
# Usage:
#   module load conda && conda activate foundation_stereo
#   python run_stereo_on_frames.py --sequence_dir ../output_droid_25947356

import os
import sys
import argparse
import logging
import glob

import cv2
import numpy as np
import torch
import imageio
import open3d as o3d
from tqdm import tqdm

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo


def load_intrinsics(K_file):
    with open(K_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    K = np.array(list(map(float, lines[0].split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline


def run_foundation_stereo(model, left_img, right_img, args):
    scale = args.scale
    if scale < 1:
        left_img = cv2.resize(left_img, fx=scale, fy=scale, dsize=None)
        right_img = cv2.resize(right_img, fx=scale, fy=scale, dsize=None)

    H, W = left_img.shape[:2]
    left_ori = left_img.copy()

    left_tensor = torch.as_tensor(left_img).cuda().float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right_img).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    return disp, left_ori, H, W


def main():
    parser = argparse.ArgumentParser(
        description='Run FoundationStereo on pre-extracted rectified stereo frames'
    )
    parser.add_argument('--sequence_dir', type=str, required=True,
                        help='Directory containing K.txt and frames/left/, frames/right/')
    parser.add_argument('--ckpt_dir', type=str,
                        default=os.path.join(code_dir, '..', 'pretrained_models', '23-51-11', 'model_best_bp2.pth'),
                        help='Pretrained model path')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Downsize factor (must be <=1)')
    parser.add_argument('--hiera', type=int, default=0,
                        help='Hierarchical inference for high-res images (>1K)')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='Number of flow-field updates during forward pass')
    parser.add_argument('--z_far', type=float, default=10.0,
                        help='Max depth to clip in point cloud')
    parser.add_argument('--get_pc', type=int, default=1,
                        help='Save point cloud output')
    parser.add_argument('--remove_invisible', type=int, default=1,
                        help='Remove non-overlapping observations')
    parser.add_argument('--denoise_cloud', type=int, default=1,
                        help='Whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30)
    parser.add_argument('--denoise_radius', type=float, default=0.03)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max number of frames to process (None = all)')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Process every N-th frame (1 = every frame)')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    sequence_dir = os.path.abspath(args.sequence_dir)

    # ---- Load intrinsics ----
    K_file = os.path.join(sequence_dir, 'K.txt')
    K, baseline = load_intrinsics(K_file)
    logging.info(f"Intrinsics from {K_file}:\n{K}")
    logging.info(f"Baseline: {baseline:.6f} m")

    # ---- Discover frame pairs ----
    left_dir = os.path.join(sequence_dir, 'frames', 'left')
    right_dir = os.path.join(sequence_dir, 'frames', 'right')

    left_files = sorted(glob.glob(os.path.join(left_dir, '*.png')))
    if len(left_files) == 0:
        raise FileNotFoundError(f'No PNG frames found in {left_dir}')

    # Build matched pairs
    frame_pairs = []
    for lf in left_files:
        stem = os.path.basename(lf)
        rf = os.path.join(right_dir, stem)
        if not os.path.exists(rf):
            logging.warning(f'No matching right frame for {stem}, skipping')
            continue
        frame_pairs.append((lf, rf, os.path.splitext(stem)[0]))

    logging.info(f"Found {len(frame_pairs)} frame pairs in {sequence_dir}")

    # Apply frame_step
    frame_pairs = frame_pairs[::args.frame_step]
    if args.max_frames is not None:
        frame_pairs = frame_pairs[:args.max_frames]

    logging.info(f"Will process {len(frame_pairs)} frames (step={args.frame_step}, max={args.max_frames})")

    # ---- Output directories ----
    depth_dir = os.path.join(sequence_dir, 'depth')
    vis_dir = os.path.join(sequence_dir, 'vis')
    pc_dir = os.path.join(sequence_dir, 'pointcloud')
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    if args.get_pc:
        os.makedirs(pc_dir, exist_ok=True)

    # ---- Load model ----
    logging.info("Loading FoundationStereo model...")
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    model_args = OmegaConf.create(cfg)

    model = FoundationStereo(model_args)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"Checkpoint: global_step={ckpt['global_step']}, epoch={ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    # ---- Process frames ----
    scale = args.scale
    pbar = tqdm(frame_pairs, desc="FoundationStereo inference")

    for left_file, right_file, frame_id in pbar:
        left_img = imageio.imread(left_file)[..., :3]
        right_img = imageio.imread(right_file)[..., :3]

        disp, left_ori, H, W = run_foundation_stereo(model, left_img, right_img, args)

        # Visualise disparity
        vis_img = vis_disparity(disp)
        vis_combined = np.concatenate([left_ori, vis_img], axis=1)
        imageio.imwrite(os.path.join(vis_dir, f'{frame_id}_vis.png'), vis_combined)

        # Remove invisible regions (left pixels that project outside right image)
        if args.remove_invisible:
            xx = np.arange(disp.shape[1])[None, :].repeat(disp.shape[0], axis=0)
            us_right = xx - disp
            disp[us_right < 0] = np.inf

        # Disparity -> depth
        K_scaled = K.copy()
        K_scaled[:2] *= scale
        depth = K_scaled[0, 0] * baseline / disp
        np.save(os.path.join(depth_dir, f'{frame_id}_depth.npy'), depth)

        # Point cloud
        if args.get_pc:
            xyz_map = depth2xyzmap(depth, K_scaled)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), left_ori.reshape(-1, 3))

            pts = np.asarray(pcd.points)
            keep = (pts[:, 2] > 0) & (pts[:, 2] <= args.z_far)
            pcd = pcd.select_by_index(np.where(keep)[0])

            if args.denoise_cloud:
                _, ind = pcd.remove_radius_outlier(
                    nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                pcd = pcd.select_by_index(ind)

            o3d.io.write_point_cloud(os.path.join(pc_dir, f'{frame_id}.ply'), pcd)

    pbar.close()

    logging.info(f"Done! Processed {len(frame_pairs)} frames.")
    logging.info(f"Results in {sequence_dir}:")
    logging.info(f"  - Depth maps:      {depth_dir}")
    logging.info(f"  - Visualisations:  {vis_dir}")
    if args.get_pc:
        logging.info(f"  - Point clouds:    {pc_dir}")


if __name__ == '__main__':
    main()
