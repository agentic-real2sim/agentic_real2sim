# Render ICP alignment results: project aligned meshes onto left images
#
# Loads saved transforms from run_icp_alignment.py and renders mesh overlays.
# No ICP computation needed — purely visualization from saved results.
#
# Usage:
#   python scripts/render_icp_results.py \
#       --data_dir output_droid_25947356 \
#       --mesh_dir meshes \
#       --icp_dir output_droid_25947356/icp_results \
#       --objects mug pen

import os
import glob
import argparse
import logging
import numpy as np
import cv2
import trimesh

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ---------------------------------------------------------------------------
# Predefined object colors (BGR for OpenCV)
# ---------------------------------------------------------------------------
OBJECT_COLORS = {
    'mug':    (50, 205, 50),     # green
    'pen':    (0, 165, 255),     # orange
    'cup':    (255, 100, 50),    # blue-ish
    'bowl':   (180, 50, 255),    # magenta
    'bottle': (255, 200, 0),     # cyan-ish
    'box':    (50, 50, 255),     # red
}
DEFAULT_COLOR = (100, 200, 100)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_intrinsics(k_path):
    """Load K matrix from K.txt (FoundationStereo format)."""
    with open(k_path, 'r') as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].split()))).reshape(3, 3).astype(np.float64)
    return K


def load_icp_meta(meta_path):
    """
    Load ICP result metadata from .npz file.
    Returns dict with T_final, scale, fitness, inlier_rmse, etc.
    """
    data = np.load(meta_path, allow_pickle=True)
    result = {}
    for k in data.keys():
        v = data[k]
        result[k] = v.item() if v.ndim == 0 else v
    return result


def load_mesh(glb_path):
    """Load a .glb mesh via trimesh."""
    return trimesh.load(glb_path, force='mesh')


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------

def project_points(pts_3d, K):
    """Project 3D points (N,3) to 2D pixel coordinates (N,2) using K."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    zs = pts_3d[:, 2]
    us = fx * pts_3d[:, 0] / zs + cx
    vs = fy * pts_3d[:, 1] / zs + cy
    return np.stack([us, vs], axis=-1), zs


def render_mesh_overlay(image, mesh, T_final, K, color_bgr,
                        alpha=0.4, edge_alpha=0.15, max_faces=None):
    """
    Render a transformed mesh onto an image as a semi-transparent overlay.

    Args:
        image:      (H, W, 3) BGR uint8 image
        mesh:       trimesh.Trimesh object (original coordinates)
        T_final:    4x4 transform (original mesh vertices -> camera frame)
        K:          3x3 camera intrinsic matrix
        color_bgr:  (B, G, R) color tuple
        alpha:      face fill transparency (0 = invisible, 1 = opaque)
        edge_alpha: wireframe edge transparency
        max_faces:  limit faces for performance (None = all)
    Returns:
        overlay image (H, W, 3) BGR uint8
    """
    H, W = image.shape[:2]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    # Transform vertices to camera frame
    verts_cam = (T_final[:3, :3] @ vertices.T).T + T_final[:3, 3]

    # Keep only faces in front of the camera
    z_vals = verts_cam[:, 2]
    face_z_min = z_vals[faces].min(axis=1)
    faces = faces[face_z_min > 0.01]

    if len(faces) == 0:
        return image.copy()

    # Sort back-to-front (painter's algorithm)
    face_mean_z = z_vals[faces].mean(axis=1)
    faces = faces[np.argsort(-face_mean_z)]

    # Subsample if too many faces
    if max_faces is not None and len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
        faces = faces[indices]

    # Project vertices to 2D
    pts_2d, _ = project_points(verts_cam, K)

    # Draw filled faces
    face_layer = image.copy()
    for tri in faces:
        pts = pts_2d[tri].astype(np.int32)
        if (pts[:, 0].max() < -50 or pts[:, 0].min() > W + 50 or
                pts[:, 1].max() < -50 or pts[:, 1].min() > H + 50):
            continue
        cv2.fillPoly(face_layer, [pts.reshape(-1, 1, 2)], color_bgr)

    # Draw wireframe edges
    edge_layer = image.copy()
    edge_color = tuple(min(int(c * 1.3), 255) for c in color_bgr)
    for tri in faces:
        pts = pts_2d[tri].astype(np.int32)
        if (pts[:, 0].max() < -50 or pts[:, 0].min() > W + 50 or
                pts[:, 1].max() < -50 or pts[:, 1].min() > H + 50):
            continue
        for i in range(3):
            cv2.line(edge_layer, tuple(pts[i]), tuple(pts[(i + 1) % 3]),
                     edge_color, 1, cv2.LINE_AA)

    # Blend
    result = cv2.addWeighted(image, 1.0 - alpha, face_layer, alpha, 0)
    result = cv2.addWeighted(result, 1.0 - edge_alpha, edge_layer, edge_alpha, 0)
    return result


def draw_legend(image, obj_results, y_start=30):
    """Draw object legend (name + fitness score) in top-left corner."""
    y = y_start
    for obj_name, meta in obj_results.items():
        color = OBJECT_COLORS.get(obj_name, DEFAULT_COLOR)
        fitness = meta.get('fitness', 0.0)
        label = f"{obj_name}: fitness={fitness:.3f}"
        if fitness < 0.1:
            label += " [FAILED]"
        cv2.rectangle(image, (10, y - 18), (30, y), color, -1)
        cv2.putText(image, label, (35, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, label, (35, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                    cv2.LINE_AA)
        y += 30
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Render ICP-aligned meshes onto left images')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory containing frames/left/ and K.txt')
    parser.add_argument('--mesh_dir', type=str, required=True,
                        help='Directory containing .glb mesh files')
    parser.add_argument('--icp_dir', type=str, default=None,
                        help='ICP results directory (default: <data_dir>/icp_results)')
    parser.add_argument('--objects', type=str, nargs='+', default=['mug', 'pen'],
                        help='Object names to render')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output render directory (default: <icp_dir>/render)')
    parser.add_argument('--alpha', type=float, default=0.45,
                        help='Mesh overlay transparency (0=invisible, 1=opaque)')
    parser.add_argument('--edge_alpha', type=float, default=0.15,
                        help='Wireframe edge transparency')
    parser.add_argument('--max_faces', type=int, default=50000,
                        help='Max faces per object to render (for speed)')
    parser.add_argument('--min_fitness', type=float, default=0.1,
                        help='Skip objects with fitness below this threshold')
    parser.add_argument('--side_by_side', type=int, default=1,
                        help='Save side-by-side (original | overlay); 0 = overlay only')
    parser.add_argument('--frames', type=str, nargs='*', default=None,
                        help='Specific frame names to render (e.g. frame_000000); '
                             'None = all available')
    args = parser.parse_args()

    # ---- Resolve paths ----
    data_dir = args.data_dir
    icp_dir = args.icp_dir or os.path.join(data_dir, 'icp_results')
    out_dir = args.out_dir or os.path.join(icp_dir, 'render')
    left_dir = os.path.join(data_dir, 'frames', 'left')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load intrinsics ----
    K = load_intrinsics(os.path.join(data_dir, 'K.txt'))
    logging.info(f"Intrinsics K:\n{K}")

    # ---- Load meshes ----
    meshes = {}
    for obj_name in args.objects:
        glb_path = os.path.join(args.mesh_dir, f'{obj_name}.glb')
        if not os.path.exists(glb_path):
            logging.warning(f"Mesh not found: {glb_path}, skipping '{obj_name}'")
            continue
        meshes[obj_name] = load_mesh(glb_path)
        logging.info(f"Loaded mesh '{obj_name}': "
                     f"{len(meshes[obj_name].vertices)} verts, "
                     f"{len(meshes[obj_name].faces)} faces")

    # ---- Discover frames ----
    if args.frames:
        frame_names = args.frames
    else:
        # Auto-discover from left image directory
        left_images = sorted(glob.glob(os.path.join(left_dir, 'frame_*.png')))
        frame_names = [os.path.splitext(os.path.basename(p))[0] for p in left_images]

    if not frame_names:
        logging.error(f"No frames found in {left_dir}")
        return

    logging.info(f"Rendering {len(frame_names)} frames: {frame_names}")

    # ---- Render each frame ----
    for frame_name in frame_names:
        left_path = os.path.join(left_dir, f'{frame_name}.png')
        if not os.path.exists(left_path):
            logging.warning(f"Left image not found: {left_path}, skipping")
            continue

        image = cv2.imread(left_path)
        overlay = image.copy()

        # Load ICP results for each object
        obj_results = {}
        for obj_name in args.objects:
            meta_path = os.path.join(icp_dir, obj_name, f'{frame_name}_meta.npz')
            if not os.path.exists(meta_path):
                logging.info(f"  {frame_name}/{obj_name}: no meta file, skipping")
                continue

            meta = load_icp_meta(meta_path)
            obj_results[obj_name] = meta
            fitness = meta.get('fitness', 0.0)

            if fitness < args.min_fitness:
                logging.info(f"  {frame_name}/{obj_name}: fitness={fitness:.3f} < "
                             f"{args.min_fitness} [SKIPPED]")
                continue

            if obj_name not in meshes:
                continue

            T_final = meta['T_final']
            color = OBJECT_COLORS.get(obj_name, DEFAULT_COLOR)

            overlay = render_mesh_overlay(
                overlay, meshes[obj_name], T_final, K,
                color_bgr=color,
                alpha=args.alpha,
                edge_alpha=args.edge_alpha,
                max_faces=args.max_faces)

            logging.info(f"  {frame_name}/{obj_name}: "
                         f"fitness={fitness:.3f}, rendered")

        # Draw legend
        overlay = draw_legend(overlay, obj_results)

        # Save
        if args.side_by_side:
            out_img = np.concatenate([image, overlay], axis=1)
        else:
            out_img = overlay

        out_path = os.path.join(out_dir, f'{frame_name}_overlay.png')
        cv2.imwrite(out_path, out_img)
        logging.info(f"  Saved: {out_path}")

    logging.info(f"\nDone! Rendered {len(frame_names)} frames to {out_dir}")


if __name__ == '__main__':
    main()
