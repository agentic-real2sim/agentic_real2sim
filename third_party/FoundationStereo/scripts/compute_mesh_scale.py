# Compute per-object mesh scale from the first N frames of a sequence.
#
# The scale is the bounding-box-diagonal ratio between the masked observed
# depth point cloud and the sampled mesh surface — exactly what
# compute_initial_alignment() in run_icp_alignment.py computes before ICP.
# ICP only refines rigid rotation/translation, so skipping it does not affect
# scale.
#
# Output (read by load_scale_median in run_demo_foundationstereo_video.py):
#   <data_dir>/scale/<obj>/frame_XXXXXX_meta.npz   (contains 'scale')
#
# Usage:
#   python compute_mesh_scale.py \
#       --data_dir ../output_droid_25947356 \
#       --mesh_dir ../meshes \
#       --mask_dir ../segment_results_droid \
#       --objects mug pen \
#       --num_frames 20

import os
import glob
import argparse
import logging

import cv2
import numpy as np
import open3d as o3d
import trimesh

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_intrinsics(k_path):
    with open(k_path, 'r') as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].split()))).reshape(3, 3).astype(np.float64)
    baseline = float(lines[1])
    return K, baseline


def depth_to_pointcloud(depth, K, mask=None, z_min=0.01, z_max=5.0):
    H, W = depth.shape
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)

    if mask is not None:
        m = mask.reshape(-1) > 0
        vs, us, zs = vs[m], us[m], zs[m]

    valid = (zs > z_min) & (zs < z_max) & np.isfinite(zs)
    vs, us, zs = vs[valid], us[valid], zs[valid]

    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    return np.stack([xs, ys, zs], axis=-1).astype(np.float64)


def load_mesh_points(glb_path, n_samples=50000):
    mesh = trimesh.load(glb_path, force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float64), mesh


def find_mask_file(mask_root, object_name, frame_idx):
    """SAM3 batch_video_segment layout: <mask_root>/<object>/masks/<frame_idx:06d>.png"""
    path = os.path.join(mask_root, object_name, 'masks', f'{frame_idx:06d}.png')
    return path if os.path.exists(path) else None


def bbox_diag(points):
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def compute_scale(observed_pts, mesh_pts):
    obs_diag = bbox_diag(observed_pts)
    mesh_diag = bbox_diag(mesh_pts)
    if mesh_diag < 1e-8:
        return 1.0
    return obs_diag / mesh_diag


def main():
    parser = argparse.ArgumentParser(
        description='Fast mesh-scale estimation from first N frames (no ICP).')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to output_droid_XXXXXXXX directory (contains K.txt and depth/)')
    parser.add_argument('--mesh_dir', type=str, required=True,
                        help='Directory containing <object>.glb meshes')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to segment_results_droid directory')
    parser.add_argument('--objects', type=str, nargs='+', default=['mug', 'pen'],
                        help='Object names to process')
    parser.add_argument('--num_frames', type=int, default=20,
                        help='Process only the first N depth frames')
    parser.add_argument('--n_mesh_samples', type=int, default=50000)
    parser.add_argument('--z_min', type=float, default=0.01)
    parser.add_argument('--z_max', type=float, default=5.0)
    parser.add_argument('--min_observed_pts', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: <data_dir>/scale)')
    args = parser.parse_args()

    depth_dir = os.path.join(args.data_dir, 'depth')
    out_dir = args.out_dir or os.path.join(args.data_dir, 'scale')
    os.makedirs(out_dir, exist_ok=True)

    K, _ = load_intrinsics(os.path.join(args.data_dir, 'K.txt'))
    logging.info(f"K:\n{K}")

    mesh_clouds = {}
    for obj in args.objects:
        glb_path = os.path.join(args.mesh_dir, f'{obj}.glb')
        if not os.path.exists(glb_path):
            logging.warning(f"Mesh not found: {glb_path}, skipping {obj}")
            continue
        pts, mesh_obj = load_mesh_points(glb_path, args.n_mesh_samples)
        mesh_clouds[obj] = pts
        logging.info(f"Loaded mesh '{obj}': {len(pts)} pts, extents={mesh_obj.extents}")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, 'frame_*_depth.npy')))
    if not depth_files:
        logging.error(f"No depth files in {depth_dir}")
        return
    depth_files = depth_files[:args.num_frames]
    logging.info(f"Will process {len(depth_files)} frames (requested {args.num_frames})")

    per_object_scales = {obj: [] for obj in mesh_clouds}

    for depth_path in depth_files:
        frame_name = os.path.basename(depth_path).replace('_depth.npy', '')
        # frame_name = 'frame_000042' -> frame_idx = 42
        frame_idx = int(frame_name.split('_')[1])

        depth = np.load(depth_path).astype(np.float64)

        for obj, mesh_pts in mesh_clouds.items():
            mask_path = find_mask_file(args.mask_dir, obj, frame_idx)
            if mask_path is None:
                logging.info(f"[{frame_name}] no mask for '{obj}'")
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            observed = depth_to_pointcloud(depth, K, mask=mask,
                                           z_min=args.z_min, z_max=args.z_max)
            if len(observed) < args.min_observed_pts:
                logging.info(f"[{frame_name}] '{obj}' too few pts ({len(observed)})")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(observed)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            observed = np.asarray(pcd.points)
            if len(observed) < args.min_observed_pts:
                logging.info(f"[{frame_name}] '{obj}' too few after denoise")
                continue

            scale = compute_scale(observed, mesh_pts)
            per_object_scales[obj].append(scale)

            obj_out_dir = os.path.join(out_dir, obj)
            os.makedirs(obj_out_dir, exist_ok=True)
            np.savez(os.path.join(obj_out_dir, f'{frame_name}_meta.npz'),
                     scale=scale,
                     n_observed_pts=len(observed),
                     mask_path=mask_path)

            logging.info(f"[{frame_name}] {obj}: scale={scale:.6f} "
                         f"(n_obs={len(observed)})")

    logging.info("=" * 50)
    logging.info("SUMMARY")
    for obj, scales in per_object_scales.items():
        if not scales:
            logging.info(f"  {obj}: no valid frames")
            continue
        arr = np.array(scales)
        logging.info(f"  {obj}: n={len(arr)}  median={np.median(arr):.6f}  "
                     f"mean={arr.mean():.6f}  std={arr.std():.6f}  "
                     f"min={arr.min():.6f}  max={arr.max():.6f}")
    logging.info(f"Results saved to {out_dir}")


if __name__ == '__main__':
    main()
