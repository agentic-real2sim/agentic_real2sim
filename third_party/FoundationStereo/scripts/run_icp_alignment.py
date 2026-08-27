# ICP alignment of mesh models to depth-map point clouds using segmentation masks
#
# For each frame in output_droid_XXXXXXXX:
#   1. Load the depth map and camera intrinsics
#   2. Load segmentation masks for each object (mug, pen) from segment_results_droid
#   3. Back-project masked depth pixels into a 3D observed point cloud
#   4. Load the corresponding .glb mesh, sample its surface
#   5. Compute an initial alignment (scale + translation) from bounding-box matching
#   6. Run ICP (Open3D) to refine the pose
#   7. Save the 4x4 transform, the mesh scale, and optional visualizations

import os
import sys
import glob
import argparse
import logging
import numpy as np
import cv2
import open3d as o3d
import trimesh
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_intrinsics(k_path):
    """Load K matrix and baseline from K.txt (FoundationStereo format)."""
    with open(k_path, 'r') as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].split()))).reshape(3, 3).astype(np.float64)
    baseline = float(lines[1])
    return K, baseline


def depth_to_pointcloud(depth, K, mask=None, z_min=0.01, z_max=5.0):
    """
    Back-project a depth image to a 3D point cloud.

    Args:
        depth: (H, W) float32 depth map in meters
        K: 3x3 intrinsic matrix
        mask: optional (H, W) uint8 binary mask (>0 = valid)
        z_min, z_max: depth clipping range
    Returns:
        points: (N, 3) np.float64
    """
    H, W = depth.shape
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)

    # Apply mask
    if mask is not None:
        m = mask.reshape(-1) > 0
        vs, us, zs = vs[m], us[m], zs[m]

    # Filter by depth range
    valid = (zs > z_min) & (zs < z_max) & np.isfinite(zs)
    vs, us, zs = vs[valid], us[valid], zs[valid]

    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    points = np.stack([xs, ys, zs], axis=-1)
    return points.astype(np.float64)


def load_mesh_pointcloud(glb_path, n_samples=50000):
    """
    Load a .glb mesh and uniformly sample its surface.
    Returns the sampled points (N, 3) and the trimesh Scene/Mesh.
    """
    mesh = trimesh.load(glb_path, force='mesh')
    # Sample the surface uniformly
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float64), mesh


def find_mask_file(masks_dir, object_name):
    """
    Find the mask file for a given object in a frame's mask directory.
    Filename pattern: {object_name}_mask*_score*.png
    """
    pattern = os.path.join(masks_dir, f'{object_name}_mask*_score*.png')
    matches = glob.glob(pattern)
    if len(matches) == 0:
        return None
    # If multiple masks, take the one with highest score
    if len(matches) > 1:
        def get_score(p):
            base = os.path.basename(p)
            # e.g. mug_mask0_score0.93.png
            score_str = base.split('score')[-1].replace('.png', '')
            return float(score_str)
        matches.sort(key=get_score, reverse=True)
    return matches[0]


def compute_initial_alignment(observed_pts, mesh_pts):
    """
    Compute an initial alignment (scale + translation) that brings the mesh
    point cloud close to the observed point cloud.

    Strategy:
      1. Scale the mesh so its bounding-box diagonal matches the observed one
      2. Translate the mesh centroid to the observed centroid

    Returns:
        scaled_mesh_pts: (N, 3)
        scale: float
        T_init: 4x4 np.float64 transform (scale baked in)
    """
    # Observed bounding box
    obs_min = observed_pts.min(axis=0)
    obs_max = observed_pts.max(axis=0)
    obs_diag = np.linalg.norm(obs_max - obs_min)
    obs_center = (obs_min + obs_max) / 2.0

    # Mesh bounding box
    mesh_min = mesh_pts.min(axis=0)
    mesh_max = mesh_pts.max(axis=0)
    mesh_diag = np.linalg.norm(mesh_max - mesh_min)
    mesh_center = (mesh_min + mesh_max) / 2.0

    # Scale
    scale = obs_diag / mesh_diag if mesh_diag > 1e-8 else 1.0

    # Apply: first center mesh at origin, scale, then translate to obs center
    scaled_pts = (mesh_pts - mesh_center) * scale + obs_center

    # Build 4x4 transform: T = translate(obs_center) @ Scale @ translate(-mesh_center)
    T_init = np.eye(4)
    T_init[:3, :3] = np.eye(3) * scale
    T_init[:3, 3] = obs_center - scale * mesh_center

    return scaled_pts, scale, T_init


def preprocess_pcd(pcd, voxel_size):
    """Downsample and compute FPFH features for global registration."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh


def run_global_registration(source_pts, target_pts, voxel_size=0.005):
    """
    FPFH feature-based RANSAC global registration.
    Good for when bounding-box alignment fails (partial views, thin objects).

    Returns:
        result: Open3D registration result with .transformation
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pts)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)

    source_down, source_fpfh = preprocess_pcd(source, voxel_size)
    target_down, target_fpfh = preprocess_pcd(target, voxel_size)

    distance_threshold = voxel_size * 2.0
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def run_icp(source_pts, target_pts, T_init=np.eye(4),
            max_correspondence_distance=0.02, max_iteration=200):
    """
    Run point-to-plane ICP using Open3D.

    Args:
        source_pts: (N, 3) mesh points (already scaled & initially aligned)
        target_pts: (M, 3) observed depth points
        T_init: 4x4 initial transform (identity if source already pre-aligned)
        max_correspondence_distance: ICP distance threshold
        max_iteration: max ICP iterations
    Returns:
        result: Open3D registration result
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pts)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)

    # Estimate normals for point-to-plane ICP (better convergence)
    radius = max_correspondence_distance * 2.5
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # Point-to-plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration,
            relative_fitness=1e-8,
            relative_rmse=1e-8,
        ),
    )
    return result


def save_colored_clouds(observed_pts, transformed_mesh_pts, out_path):
    """Save a combined .ply with observed (blue) + aligned mesh (red) for debugging."""
    n_obs = len(observed_pts)
    n_mesh = len(transformed_mesh_pts)

    all_pts = np.vstack([observed_pts, transformed_mesh_pts])
    colors = np.zeros((n_obs + n_mesh, 3))
    colors[:n_obs] = [0.2, 0.4, 1.0]       # blue for observed
    colors[n_obs:] = [1.0, 0.2, 0.2]        # red for mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)


# Object rendering colors: (B, G, R) for OpenCV
OBJECT_COLORS = {
    'mug':  (50, 205, 50),     # green
    'pen':  (0, 165, 255),     # orange
    'cup':  (255, 100, 50),    # blue-ish
    'bowl': (180, 50, 255),    # magenta
}
DEFAULT_COLOR = (100, 200, 100)


def project_points(pts_3d, K):
    """
    Project 3D points (N,3) to 2D pixel coordinates using intrinsic matrix K.
    Returns (N,2) float array of (u, v) pixel coordinates and (N,) depths.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    zs = pts_3d[:, 2]
    us = fx * pts_3d[:, 0] / zs + cx
    vs = fy * pts_3d[:, 1] / zs + cy
    return np.stack([us, vs], axis=-1), zs


def render_mesh_overlay(image, mesh_trimesh, T_final, K, color_bgr,
                        alpha=0.4, edge_alpha=0.7, max_faces=None):
    """
    Render a transformed mesh onto an image as a semi-transparent overlay.

    Args:
        image: (H, W, 3) BGR uint8 image
        mesh_trimesh: trimesh.Trimesh object (original, untransformed)
        T_final: 4x4 transform (maps original mesh vertices -> camera frame)
        K: 3x3 camera intrinsic matrix
        color_bgr: (B, G, R) tuple for the mesh color
        alpha: transparency for filled faces
        edge_alpha: transparency for wireframe edges
        max_faces: limit faces rendered (for performance); None = all
    Returns:
        overlay: (H, W, 3) BGR uint8 image with mesh overlay
    """
    H, W = image.shape[:2]
    vertices = np.asarray(mesh_trimesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh_trimesh.faces)

    # Transform vertices to camera frame
    verts_cam = (T_final[:3, :3] @ vertices.T).T + T_final[:3, 3]

    # Filter faces whose vertices are in front of the camera
    z_vals = verts_cam[:, 2]
    face_z_min = z_vals[faces].min(axis=1)
    valid_faces_mask = face_z_min > 0.01
    faces = faces[valid_faces_mask]

    if len(faces) == 0:
        return image.copy()

    # Sort faces by mean depth (back-to-front for painter's algorithm)
    face_mean_z = z_vals[faces].mean(axis=1)
    depth_order = np.argsort(-face_mean_z)  # far to near
    faces = faces[depth_order]

    # Limit faces for performance if mesh is very dense
    if max_faces is not None and len(faces) > max_faces:
        # Subsample evenly
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
        faces = faces[indices]

    # Project all vertices to 2D
    pts_2d, _ = project_points(verts_cam, K)

    # Create overlay layers
    face_overlay = image.copy()
    edge_overlay = image.copy()

    # Draw filled triangles
    for tri in faces:
        pts = pts_2d[tri].astype(np.int32)
        # Clip check: skip if all points are far outside the image
        if (pts[:, 0].max() < -50 or pts[:, 0].min() > W + 50 or
                pts[:, 1].max() < -50 or pts[:, 1].min() > H + 50):
            continue
        cv2.fillPoly(face_overlay, [pts.reshape(-1, 1, 2)], color_bgr)

    # Draw wireframe edges (slightly brighter)
    edge_color = tuple(min(int(c * 1.3), 255) for c in color_bgr)
    for tri in faces:
        pts = pts_2d[tri].astype(np.int32)
        if (pts[:, 0].max() < -50 or pts[:, 0].min() > W + 50 or
                pts[:, 1].max() < -50 or pts[:, 1].min() > H + 50):
            continue
        for i in range(3):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 3])
            cv2.line(edge_overlay, p1, p2, edge_color, 1, cv2.LINE_AA)

    # Blend: faces then edges
    result = cv2.addWeighted(image, 1.0 - alpha, face_overlay, alpha, 0)
    result = cv2.addWeighted(result, 1.0 - edge_alpha, edge_overlay, edge_alpha, 0)
    return result


def render_all_objects_overlay(left_img_path, frame_results, mesh_objects,
                               K, out_path, alpha=0.4, max_faces_per_obj=50000):
    """
    Render all aligned objects onto the left image and save.

    Args:
        left_img_path: path to the left frame PNG
        frame_results: dict {obj_name: {'T_final': ..., 'fitness': ..., ...}}
        mesh_objects: dict {obj_name: trimesh.Trimesh}
        K: 3x3 intrinsic matrix
        out_path: output image path
        alpha: overlay transparency
        max_faces_per_obj: max faces to render per object
    """
    image = cv2.imread(left_img_path)
    if image is None:
        logging.warning(f"Cannot read image: {left_img_path}")
        return

    overlay = image.copy()

    for obj_name, res in frame_results.items():
        if res['fitness'] < 0.1:
            continue  # skip failed alignments

        if obj_name not in mesh_objects:
            continue

        color = OBJECT_COLORS.get(obj_name, DEFAULT_COLOR)
        T_final = res['T_final']

        overlay = render_mesh_overlay(
            overlay, mesh_objects[obj_name], T_final, K,
            color_bgr=color, alpha=alpha, edge_alpha=0.15,
            max_faces=max_faces_per_obj)

        logging.info(f"    Rendered '{obj_name}' on overlay "
                     f"(fitness={res['fitness']:.3f})")

    # Add legend in top-left corner
    y_offset = 30
    for obj_name, res in frame_results.items():
        color = OBJECT_COLORS.get(obj_name, DEFAULT_COLOR)
        fitness = res['fitness']
        label = f"{obj_name}: fitness={fitness:.3f}"
        if fitness < 0.1:
            label += " [FAILED]"
        # Draw colored rectangle + text
        cv2.rectangle(overlay, (10, y_offset - 18), (30, y_offset), color, -1)
        cv2.putText(overlay, label, (35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(overlay, label, (35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                    cv2.LINE_AA)
        y_offset += 30

    # Save side-by-side: original | overlay
    combined = np.concatenate([image, overlay], axis=1)
    cv2.imwrite(out_path, combined)
    logging.info(f"    Saved render overlay: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ICP alignment of mesh models to depth-map point clouds')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to output_droid_XXXXXXXX directory')
    parser.add_argument('--mesh_dir', type=str, required=True,
                        help='Path to meshes directory containing .glb files')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to segment_results_droid directory')
    parser.add_argument('--objects', type=str, nargs='+', default=['mug', 'pen'],
                        help='Object names to align (must match mask filenames and mesh filenames)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: <data_dir>/icp_results)')
    parser.add_argument('--n_mesh_samples', type=int, default=50000,
                        help='Number of points to sample from mesh surface')
    parser.add_argument('--icp_max_dist', type=float, default=0.02,
                        help='ICP max correspondence distance (meters)')
    parser.add_argument('--icp_max_iter', type=int, default=200,
                        help='ICP max iterations')
    parser.add_argument('--z_min', type=float, default=0.01,
                        help='Min valid depth (meters)')
    parser.add_argument('--z_max', type=float, default=5.0,
                        help='Max valid depth (meters)')
    parser.add_argument('--save_vis', type=int, default=1,
                        help='Save visualization point clouds')
    parser.add_argument('--icp_refine_steps', type=int, default=3,
                        help='Number of multi-scale ICP refinement steps')
    args = parser.parse_args()

    # ---- Setup ----
    data_dir = args.data_dir
    depth_dir = os.path.join(data_dir, 'depth')
    out_dir = args.out_dir or os.path.join(data_dir, 'icp_results')
    os.makedirs(out_dir, exist_ok=True)

    K, baseline = load_intrinsics(os.path.join(data_dir, 'K.txt'))
    logging.info(f"Loaded intrinsics from {data_dir}/K.txt")
    logging.info(f"K:\n{K}")

    # ---- Load meshes ----
    mesh_clouds = {}
    mesh_objects = {}
    for obj_name in args.objects:
        glb_path = os.path.join(args.mesh_dir, f'{obj_name}.glb')
        if not os.path.exists(glb_path):
            logging.warning(f"Mesh not found: {glb_path}, skipping {obj_name}")
            continue
        pts, mesh_obj = load_mesh_pointcloud(glb_path, args.n_mesh_samples)
        mesh_clouds[obj_name] = pts
        mesh_objects[obj_name] = mesh_obj
        logging.info(f"Loaded mesh '{obj_name}': {len(pts)} sampled points, "
                     f"extents={mesh_obj.extents}")

    # ---- Find depth frames ----
    depth_files = sorted(glob.glob(os.path.join(depth_dir, 'frame_*_depth.npy')))
    if not depth_files:
        logging.error(f"No depth files found in {depth_dir}")
        return
    logging.info(f"Found {len(depth_files)} depth frames")

    # ---- Process each frame ----
    all_results = {}

    for depth_path in depth_files:
        fname = os.path.basename(depth_path)
        # e.g. frame_000000_depth.npy -> frame_000000
        frame_name = fname.replace('_depth.npy', '')
        frame_idx = frame_name  # e.g. "frame_000000"

        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {frame_name}")

        depth = np.load(depth_path).astype(np.float64)
        masks_dir = os.path.join(args.mask_dir, f'{frame_name}_masks')

        if not os.path.isdir(masks_dir):
            logging.warning(f"Masks directory not found: {masks_dir}, skipping")
            continue

        frame_results = {}

        for obj_name in args.objects:
            if obj_name not in mesh_clouds:
                continue

            # Find mask
            mask_path = find_mask_file(masks_dir, obj_name)
            if mask_path is None:
                logging.warning(f"  No mask found for '{obj_name}' in {masks_dir}")
                continue

            logging.info(f"  Object: {obj_name}")
            logging.info(f"    Mask: {os.path.basename(mask_path)}")

            # Load mask and extract observed point cloud
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            observed_pts = depth_to_pointcloud(depth, K, mask=mask,
                                               z_min=args.z_min, z_max=args.z_max)
            if len(observed_pts) < 50:
                logging.warning(f"    Too few observed points ({len(observed_pts)}), skipping")
                continue
            logging.info(f"    Observed points: {len(observed_pts)}")

            # Denoise observed point cloud (statistical outlier removal)
            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(observed_pts)
            obs_pcd, inlier_idx = obs_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)
            observed_pts = np.asarray(obs_pcd.points)
            logging.info(f"    After denoising: {len(observed_pts)} points")

            # Initial alignment: scale + translate (bbox-based)
            mesh_pts = mesh_clouds[obj_name].copy()
            scaled_mesh_pts, scale, T_init = compute_initial_alignment(
                observed_pts, mesh_pts)
            logging.info(f"    Initial scale: {scale:.6f}")
            logging.info(f"    Initial translation: {T_init[:3, 3]}")

            # ---- Phase 1: Multi-scale ICP with bbox initialization ----
            dist_schedule = np.geomspace(
                args.icp_max_dist * 5, args.icp_max_dist,
                num=args.icp_refine_steps)

            current_pts = scaled_mesh_pts.copy()
            T_icp_cumulative = np.eye(4)

            for step, max_dist in enumerate(dist_schedule):
                result = run_icp(
                    current_pts, observed_pts,
                    T_init=np.eye(4),
                    max_correspondence_distance=max_dist,
                    max_iteration=args.icp_max_iter,
                )
                T_step = result.transformation
                T_icp_cumulative = T_step @ T_icp_cumulative
                current_pts = (T_step[:3, :3] @ current_pts.T).T + T_step[:3, 3]

                logging.info(f"    ICP step {step+1}/{args.icp_refine_steps} "
                             f"(max_dist={max_dist:.4f}): "
                             f"fitness={result.fitness:.4f}, "
                             f"RMSE={result.inlier_rmse:.6f}")

            bbox_fitness = result.fitness
            bbox_rmse = result.inlier_rmse

            # ---- Phase 2: If bbox-ICP failed, try FPFH global registration ----
            if bbox_fitness < 0.1:
                logging.info(f"    BBox-ICP failed (fitness={bbox_fitness:.4f}), "
                             f"trying FPFH global registration...")

                # Try multiple voxel sizes x multiple random restarts
                best_global_fitness = 0.0
                best_T_global = np.eye(4)
                n_restarts = 5  # RANSAC is stochastic, try multiple times

                for voxel_size in [0.003, 0.005, 0.008]:
                    for attempt in range(n_restarts):
                        try:
                            global_result = run_global_registration(
                                scaled_mesh_pts, observed_pts, voxel_size=voxel_size)
                            # Refine with ICP
                            refined_pts = (global_result.transformation[:3, :3] @
                                           scaled_mesh_pts.T).T + global_result.transformation[:3, 3]
                            icp_result = run_icp(
                                refined_pts, observed_pts,
                                T_init=np.eye(4),
                                max_correspondence_distance=args.icp_max_dist * 3,
                                max_iteration=args.icp_max_iter,
                            )
                            total_fitness = icp_result.fitness
                            if attempt == 0 or total_fitness > 0:
                                logging.info(f"      FPFH (voxel={voxel_size:.3f}, "
                                             f"attempt={attempt+1}): "
                                             f"global={global_result.fitness:.4f} -> "
                                             f"refined={total_fitness:.4f}")
                            if total_fitness > best_global_fitness:
                                best_global_fitness = total_fitness
                                best_T_global = icp_result.transformation @ global_result.transformation
                            if total_fitness > 0.5:
                                break  # good enough, no more restarts
                        except Exception as e:
                            if attempt == 0:
                                logging.warning(f"      FPFH (voxel={voxel_size:.3f}) failed: {e}")
                    if best_global_fitness > 0.5:
                        break  # good enough, no more voxel sizes

                # Use global result if it's better
                if best_global_fitness > bbox_fitness:
                    logging.info(f"    Using FPFH result (fitness={best_global_fitness:.4f})")
                    T_icp_cumulative = best_T_global
                    current_pts = (T_icp_cumulative[:3, :3] @
                                   scaled_mesh_pts.T).T + T_icp_cumulative[:3, 3]

                    # Final refinement with tight ICP
                    for max_dist in [args.icp_max_dist * 2, args.icp_max_dist]:
                        result = run_icp(
                            current_pts, observed_pts,
                            T_init=np.eye(4),
                            max_correspondence_distance=max_dist,
                            max_iteration=args.icp_max_iter,
                        )
                        T_icp_cumulative = result.transformation @ T_icp_cumulative
                        current_pts = (result.transformation[:3, :3] @
                                       current_pts.T).T + result.transformation[:3, 3]
                        logging.info(f"    FPFH refinement (max_dist={max_dist:.4f}): "
                                     f"fitness={result.fitness:.4f}, "
                                     f"RMSE={result.inlier_rmse:.6f}")

            # Final combined transform: T_icp @ T_init
            # This transforms the ORIGINAL mesh points to the camera frame
            T_final = T_icp_cumulative @ T_init
            final_fitness = result.fitness
            final_rmse = result.inlier_rmse

            logging.info(f"    Final fitness={final_fitness:.4f}, RMSE={final_rmse:.6f}")
            logging.info(f"    Final transform:\n{T_final}")

            # Save results
            obj_out_dir = os.path.join(out_dir, obj_name)
            os.makedirs(obj_out_dir, exist_ok=True)

            # Save transform
            transform_path = os.path.join(obj_out_dir, f'{frame_name}_transform.npy')
            np.save(transform_path, T_final)

            # Save metadata
            meta_path = os.path.join(obj_out_dir, f'{frame_name}_meta.npz')
            np.savez(meta_path,
                     T_final=T_final,
                     T_init=T_init,
                     T_icp=T_icp_cumulative,
                     scale=scale,
                     fitness=final_fitness,
                     inlier_rmse=final_rmse,
                     n_observed_pts=len(observed_pts),
                     mask_path=mask_path)

            # Save visualization
            if args.save_vis:
                vis_path = os.path.join(obj_out_dir, f'{frame_name}_aligned.ply')
                save_colored_clouds(observed_pts, current_pts, vis_path)
                logging.info(f"    Saved visualization: {vis_path}")

            frame_results[obj_name] = {
                'T_final': T_final,
                'scale': scale,
                'fitness': final_fitness,
                'rmse': final_rmse,
            }

        all_results[frame_name] = frame_results

        # ---- Render combined overlay for this frame ----
        if args.save_vis and frame_results:
            left_img_path = os.path.join(data_dir, 'frames', 'left',
                                         f'{frame_name}.png')
            if os.path.exists(left_img_path):
                render_dir = os.path.join(out_dir, 'render')
                os.makedirs(render_dir, exist_ok=True)
                render_path = os.path.join(render_dir,
                                           f'{frame_name}_overlay.png')
                render_all_objects_overlay(
                    left_img_path, frame_results, mesh_objects,
                    K, render_path, alpha=0.45)

    # ---- Post-processing: propagate successful poses to failed frames ----
    frame_names_sorted = sorted(all_results.keys())
    for obj_name in args.objects:
        # Collect frames that succeeded and failed
        success_frames = []
        failed_frames = []
        for fn in frame_names_sorted:
            if obj_name in all_results[fn]:
                if all_results[fn][obj_name]['fitness'] > 0.1:
                    success_frames.append(fn)
                else:
                    failed_frames.append(fn)

        if not failed_frames or not success_frames:
            continue

        logging.info(f"\n  Propagating '{obj_name}' poses to failed frames: {failed_frames}")

        for fail_fn in failed_frames:
            # Find nearest successful frame
            fail_idx = frame_names_sorted.index(fail_fn)
            nearest_fn = min(success_frames,
                             key=lambda s: abs(frame_names_sorted.index(s) - fail_idx))
            nearest_res = all_results[nearest_fn][obj_name]
            logging.info(f"    {fail_fn}: using pose from {nearest_fn} as initialization")

            # Load this frame's depth and mask
            depth_path = os.path.join(depth_dir, f'{fail_fn}_depth.npy')
            depth = np.load(depth_path).astype(np.float64)
            masks_dir = os.path.join(args.mask_dir, f'{fail_fn}_masks')
            mask_path = find_mask_file(masks_dir, obj_name)
            if mask_path is None:
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            observed_pts = depth_to_pointcloud(depth, K, mask=mask,
                                               z_min=args.z_min, z_max=args.z_max)
            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(observed_pts)
            obs_pcd, _ = obs_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            observed_pts = np.asarray(obs_pcd.points)

            if len(observed_pts) < 50:
                continue

            # Use the nearest successful frame's transform as initialization
            T_neighbor = nearest_res['T_final']
            scale = nearest_res['scale']

            # Apply neighbor's transform to original mesh
            mesh_pts = mesh_clouds[obj_name].copy()
            init_pts = (T_neighbor[:3, :3] @ mesh_pts.T).T + T_neighbor[:3, 3]

            # Refine with ICP
            T_refine_cum = np.eye(4)
            current_pts = init_pts.copy()
            for max_dist in [args.icp_max_dist * 3, args.icp_max_dist * 1.5, args.icp_max_dist]:
                result = run_icp(
                    current_pts, observed_pts,
                    T_init=np.eye(4),
                    max_correspondence_distance=max_dist,
                    max_iteration=args.icp_max_iter,
                )
                T_refine_cum = result.transformation @ T_refine_cum
                current_pts = (result.transformation[:3, :3] @
                               current_pts.T).T + result.transformation[:3, 3]

            final_fitness = result.fitness
            final_rmse = result.inlier_rmse
            T_final = T_refine_cum @ T_neighbor

            logging.info(f"    Propagated result: fitness={final_fitness:.4f}, "
                         f"RMSE={final_rmse:.6f}")

            if final_fitness > all_results[fail_fn].get(obj_name, {}).get('fitness', 0):
                # Update results
                all_results[fail_fn][obj_name] = {
                    'T_final': T_final,
                    'scale': scale,
                    'fitness': final_fitness,
                    'rmse': final_rmse,
                }

                # Save updated results
                obj_out_dir = os.path.join(out_dir, obj_name)
                os.makedirs(obj_out_dir, exist_ok=True)
                np.save(os.path.join(obj_out_dir, f'{fail_fn}_transform.npy'), T_final)
                np.savez(os.path.join(obj_out_dir, f'{fail_fn}_meta.npz'),
                         T_final=T_final, T_init=T_neighbor, T_icp=T_refine_cum,
                         scale=scale, fitness=final_fitness, inlier_rmse=final_rmse,
                         n_observed_pts=len(observed_pts), mask_path=mask_path,
                         propagated_from=nearest_fn)
                if args.save_vis:
                    vis_path = os.path.join(obj_out_dir, f'{fail_fn}_aligned.ply')
                    save_colored_clouds(observed_pts, current_pts, vis_path)

    # ---- Re-render overlays for propagated frames ----
    if args.save_vis:
        render_dir = os.path.join(out_dir, 'render')
        os.makedirs(render_dir, exist_ok=True)
        for frame_name, frame_res in all_results.items():
            # Re-render if any object was propagated (check for improvements)
            left_img_path = os.path.join(data_dir, 'frames', 'left',
                                         f'{frame_name}.png')
            if os.path.exists(left_img_path) and frame_res:
                render_path = os.path.join(render_dir,
                                           f'{frame_name}_overlay.png')
                render_all_objects_overlay(
                    left_img_path, frame_res, mesh_objects,
                    K, render_path, alpha=0.45)

    # ---- Summary ----
    logging.info(f"\n{'='*60}")
    logging.info("SUMMARY")
    logging.info(f"{'='*60}")
    for frame_name, frame_res in all_results.items():
        for obj_name, res in frame_res.items():
            logging.info(f"  {frame_name} / {obj_name}: "
                         f"fitness={res['fitness']:.4f}, "
                         f"RMSE={res['rmse']:.6f}, "
                         f"scale={res['scale']:.6f}")

    logging.info(f"\nResults saved to {out_dir}")


if __name__ == '__main__':
    main()
