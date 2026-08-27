# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
    TexturesVertex,
)
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, matrix_to_quaternion, quaternion_multiply
import random
import open3d as o3d
from scipy.ndimage import label, binary_dilation, binary_fill_holes, binary_erosion, minimum_filter
import copy
from sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render import GaussianRenderer
from loguru import logger
from utils3d.numpy import depth_edge

def remove_small_regions(mask, min_area=100):
    """
    Remove small disconnected regions (floating points) from the mask.
    Keeps all regions with area >= min_area.
    """
    labeled_mask, num_labels = label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_labels + 1):
        region = (labeled_mask == i)
        if region.sum() >= min_area:
            cleaned |= region
    return cleaned

def is_near_image_border(mask, border_thickness=10):
    """
    Check if the mask touches the image border within a given thickness.
    """
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[:border_thickness, :] = True
    border_mask[-border_thickness:, :] = True
    border_mask[:, :border_thickness] = True
    border_mask[:, -border_thickness:] = True
    return np.any(mask & border_mask)    

def is_occluded_by_others(mask, point_map, dilation_iter=2, z_thresh=0.05, filter_size=3):
    """
    Efficient occlusion detection using depth map and internal/external edges.
    """
    z_map = point_map[..., 2]
    if not np.any(mask):
        return False

    # Create internal and external edge masks
    eroded = binary_erosion(mask, iterations=dilation_iter)
    dilated = binary_dilation(mask, iterations=dilation_iter)

    internal_edge = mask & (~eroded)
    external_edge = dilated & (~mask)

    # Set invalid areas to +inf so they don't affect min-pooling
    z_ext = np.where(external_edge, z_map, np.inf)

    # Apply minimum filter to get local min depth around internal edges
    z_ext_min = minimum_filter(z_ext, size=filter_size, mode='constant', cval=np.inf)

    # Depth values at internal edge
    z_int = np.where(internal_edge, z_map, np.nan)

    # Compare depth difference
    diff = z_int - z_ext_min
    occlusion_mask = (diff > z_thresh) & (~np.isnan(diff))

    # return np.any(occlusion_mask)
    return np.sum(occlusion_mask) > 10

def has_internal_occlusion(mask, min_hole_area=20):
    """
    Check if the mask has internal holes or has been split into fragments.
    This may indicate internal occlusion.
    """
    # Check number of connected components
    labeled, num_features = label(mask)
    if num_features > 1:
        return True  # Mask is fragmented

    # Check for internal holes
    filled = binary_fill_holes(mask)
    holes = filled & (~mask)
    return np.sum(holes) >= min_hole_area

def check_occlusion(mask, point_map,
                    min_region_area=25,
                    border_thickness=5,
                    z_thresh=0.3,
                    min_hole_area=100):
    """
    Main function to check different types of occlusion for a given mask and 3D point map.
    """
    # clean mask by removing floating points
    cleaned_mask = remove_small_regions(mask, min_area=min_region_area)
    dilation_iter = 2
    filter_size = 2 * dilation_iter + 1

    # run occlusion checks
    return (
        is_near_image_border(cleaned_mask, border_thickness)
        or is_occluded_by_others(cleaned_mask, point_map, dilation_iter, z_thresh, filter_size)
        or has_internal_occlusion(cleaned_mask, min_hole_area)
    )

def get_mesh(Mesh, tfm_ori, device):
    mesh_vertices = Mesh.vertices.copy()
    # rotate mesh (from z-up to y-up)
    mesh_vertices = mesh_vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    mesh_vertices = torch.from_numpy(mesh_vertices).float().to(device)
    points_world = tfm_ori.transform_points(mesh_vertices.unsqueeze(0))
    Mesh.vertices = points_world[0].cpu().numpy()  # pytorch3d, y-up, x left, z inwards.
    verts, faces_idx = load_and_simplify_mesh(Mesh, device)
    # === Add dummy white texture ===
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])  # (1, V, 3)
    mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

    return mesh, faces_idx, textures


def get_mask_renderer(Mask, min_size, Intrinsics, device):
    orig_h, orig_w = Mask.shape[-2:]
    min_orig_size = min(orig_w, orig_h)
    scale_factor = min_size / min_orig_size
    mask = F.interpolate(
        Mask[None, None],
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )
    H, W = mask.shape[-2:]

    intrinsics = denormalize_f(Intrinsics.cpu().numpy(), H, W)
    cameras = PerspectiveCameras(
        focal_length=torch.tensor(
            [[intrinsics[0, 0], intrinsics[1, 1]]], device=device, dtype=torch.float32
        ),
        principal_point=torch.tensor(
            [[intrinsics[0, 2], intrinsics[1, 2]]], device=device, dtype=torch.float32
        ),
        image_size=torch.tensor([[H, W]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=1e-6,
        faces_per_pixel=50,
        max_faces_per_bin=50000,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    return mask, renderer


def run_alignment(
    Point_Map,
    mask,
    mesh,
    center,
    faces_idx,
    textures,
    renderer,
    device,
    align_pm_coordinate=False,
):

    # Get rid of flying points using depth edge detection
    # Convert mask to 2D for depth_edge function
    mask_2d = mask[0, 0].bool().cpu().numpy()
    depth = Point_Map[..., 2].cpu().numpy()

    # Remove flying points (large depth discontinuities)
    depth_edge_mask = depth_edge(depth, rtol=0.03, mask=mask_2d)
    cleaned_mask = mask_2d & ~depth_edge_mask

    # Convert back to torch tensor and apply to get target points
    cleaned_mask_tensor = torch.from_numpy(cleaned_mask).to(Point_Map.device)
    target_object_points = Point_Map[cleaned_mask_tensor]

    # Remove inf values
    finite_mask = torch.isfinite(target_object_points).all(dim=1)
    target_object_points = target_object_points[finite_mask]

    # Apply coordinate alignment if needed
    if align_pm_coordinate:
        target_object_points[:, 0] *= -1
        target_object_points[:, 1] *= -1
    flag_notgt = False

    if target_object_points.shape[0] == 0:
        flag_notgt = True
        return None, None, None, None, None, None, None, flag_notgt    

    source_points, target_points = mesh.verts_packed(), target_object_points
    # align to moge object points.
    height_src = torch.max(source_points[:, 1]) - torch.min(source_points[:, 1])
    height_tgt = torch.max(target_points[:, 1]) - torch.min(target_points[:, 1])
    scale_1 = height_tgt / height_src
    source_points *= scale_1
    center *= scale_1

    center_src = torch.mean(source_points, dim=0)
    center_tgt = torch.mean(target_points, dim=0)
    translation_1 = center_tgt - center_src

    source_points += translation_1
    center += translation_1

    # manually align based on moge point cloud.
    tfm1 = (
        Transform3d(device=device)
        .scale(scale_1.expand(3)[None])
        .translate(translation_1[None])
    )
    mesh = Meshes(verts=[source_points], faces=[faces_idx], textures=textures)
    rendered = renderer(mesh)
    ori_iou = compute_iou(rendered[..., 3][0][None, None], mask, threshold=0.5)
    final_iou = ori_iou.cpu().item()

    return source_points, target_points, center, tfm1, mesh, ori_iou, final_iou, flag_notgt


def apply_transform(mesh, center, quat, translation, scale):
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)
    # transform to the world coordinate system center.
    verts = mesh.verts_packed() - center
    # perform operation
    verts = verts * scale
    verts = verts @ R.transpose(0, 1)
    # transform back to the original position after rotation.
    verts += center
    verts = verts + translation

    transformed_mesh = Meshes(
        verts=[verts], faces=[mesh.faces_packed()], textures=mesh.textures
    )
    return transformed_mesh


def compute_loss(rendered, mask_gt, loss_weights, quat, translation, scale):

    pred_mask = rendered[..., 3][0]
    # === 1. MSE Loss on mask ===
    loss_mask = F.mse_loss(pred_mask, mask_gt[0, 0])

    # === 2. Reg Loss on quaternion ===
    quat_normalized = quat / quat.norm()
    loss_reg_q = F.mse_loss(
        quat_normalized, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device)
    )
    loss_reg_t = torch.norm(translation) ** 2
    loss_reg_s = (scale - 1.0) ** 2

    # === Total weighted loss ===
    total_loss = (
        loss_weights["mask"] * loss_mask
        + loss_weights["reg_q"] * loss_reg_q
        + loss_weights["reg_t"] * loss_reg_t
        + loss_weights["reg_s"] * loss_reg_s
    )

    return total_loss


def export_transformed_mesh_glb(
    verts, mesh_obj, center, quat, translation, scale, output_path
):
    quat_normalized = quat / quat.norm()

    R = quaternion_to_matrix(quat_normalized)
    # transform to the world coordinate system center.
    verts -= center
    # perform operations.
    verts = verts * scale
    verts = verts @ R.transpose(0, 1)
    # transform back to the original position after rotation.
    verts += center
    verts = verts + translation

    mesh_obj.vertices = verts.cpu().numpy()
    output_path = os.path.join(output_path, "result.glb")
    mesh_obj.export(output_path)
    return


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_simplify_mesh(Mesh, device, target_triangles=5000):

    vertices = np.asarray(Mesh.vertices)
    faces = np.asarray(Mesh.faces)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_non_manifold_edges()

    if len(mesh_o3d.triangles) > target_triangles:
        mesh_simplified = mesh_o3d.simplify_quadric_decimation(target_triangles)
    else:
        mesh_simplified = mesh_o3d

    verts = torch.tensor(
        np.asarray(mesh_simplified.vertices), dtype=torch.float32, device=device
    )
    faces = torch.tensor(
        np.asarray(mesh_simplified.triangles), dtype=torch.int64, device=device
    )

    return verts, faces


def compute_iou(render_mask_obj, mask_obj_gt, threshold=0.5):

    # Binarize masks
    pred = (render_mask_obj > threshold).float()
    gt_obj = (mask_obj_gt > threshold).float()

    # Compute intersection and union
    intersection = (pred * gt_obj).sum()
    union = ((pred + gt_obj) > 0).float().sum()

    if union == 0:
        return torch.tensor(1.0 if intersection == 0 else 0.0)  # avoid division by zero

    iou = intersection / union
    return iou


def denormalize_f(norm_K, height, width):
    # Extract cx and cy from the normalized K matrix
    cx_norm = norm_K[0][2]  # c_x is at K[0][2]
    cy_norm = norm_K[1][2]  # c_y is at K[1][2]

    fx_norm = norm_K[0][0]  # Normalized fx
    fy_norm = norm_K[1][1]  # Normalized fy
    s_norm = norm_K[0][1]  # Skew (usually 0)

    # Scale to absolute values
    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    s_abs = s_norm * width

    # Construct absolute K matrix
    abs_K = np.array([[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]])
    return abs_K


# Convert torch tensors to Open3D point clouds
def tensor_to_o3d_pcd(tensor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor.cpu().numpy())
    return pcd


# Convert Open3D back to torch tensor
def o3d_to_tensor(pcd):
    return torch.tensor(np.asarray(pcd.points), dtype=torch.float32)


def run_ICP(source_points_mesh, source_points, target_points, threshold):
    # Convert your point clouds
    mesh_src_pcd = tensor_to_o3d_pcd(source_points_mesh.verts_padded().squeeze(0))
    src_pcd = tensor_to_o3d_pcd(source_points)
    tgt_pcd = tensor_to_o3d_pcd(target_points)

    # Run ICP
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Apply transformation
    mesh_src_pcd.transform(reg_p2p.transformation)
    points_aligned_icp = o3d_to_tensor(mesh_src_pcd).to(source_points.device)

    return points_aligned_icp, reg_p2p.transformation


def run_render_compare(mesh, center, renderer, mask, device):

    quat = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True)
    )
    translation = torch.nn.Parameter(
        torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
    )
    scale = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))

    def get_optimizer(stage):
        if stage == 1:
            return torch.optim.Adam([translation, scale], lr=1e-2)
        elif stage == 2:
            return torch.optim.Adam([quat, translation, scale], lr=5e-3)

    loss_weights = {"mask": 200, "reg_q": 0.1, "reg_t": 0.05, "reg_s": 0.05}
    prev_loss = None

    global_step = 0
    for stage in [1, 2]:
        optimizer = get_optimizer(stage)
        iters = [5, 25]
        for i in range(iters[stage - 1]):
            optimizer.zero_grad()
            transformed = apply_transform(mesh, center, quat, translation, scale)
            rendered = renderer(transformed)
            loss = compute_loss(rendered, mask, loss_weights, quat, translation, scale)
            loss.backward()
            optimizer.step()
            global_step += 1
            if prev_loss is not None and abs(loss.item() - prev_loss) < 1e-5:
                break
            prev_loss = loss.item()

    quat, translation, scale = quat.detach(), translation.detach(), scale.detach()
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)

    return quat, translation, scale, R

# =============================================================================
# Gaussian Splatting Utilities
# =============================================================================

def extract_mask_from_gs_rendering(rendered, mode="evaluation"):
    """
    Extract mask from GS rendering with consistent logic.
    Returns:
        mask tensor [H, W]
    """
    # Priority 1: Try alpha channel
    if "alpha" in rendered:
        mask = rendered["alpha"][0] if rendered["alpha"].dim() == 3 else rendered["alpha"]
        return mask

    # Priority 2: Try rgba format
    if "rgba" in rendered:
        return rendered["rgba"][3]

    # Priority 3: Extract from color/render channels
    if "color" in rendered:
        mask = rendered["color"]
    else:
        raise ValueError("Cannot extract mask from GS rendering - missing 'color', 'alpha', or 'rgba' key")

    # Handle different channel formats
    if mask.shape[0] == 3:  # RGB
        if mode == "evaluation":
            # Use max across channels for sharper binary decisions
            mask = mask.max(dim=0)[0]
        else:  # optimization
            # Use mean across channels for better gradient flow
            mask = mask.mean(dim=0)
    elif mask.shape[0] == 4:  # RGBA
        mask = mask[3]  # Use alpha channel
    elif mask.shape[0] == 1:  # Grayscale
        mask = mask[0]  # Squeeze channel dimension

    return mask


def extract_rgb_from_gs_rendering(rendered):
    """
    Extract RGB from GS rendering with consistent logic.

    Args:
        rendered: GS rendered output dictionary

    Returns:
        RGB tensor [3, H, W]
    """
    if "color" in rendered:
        return rendered["color"]
    else:
        raise ValueError("Cannot extract RGB from GS rendering - missing 'color' key")


def get_mask_colors_for_gs(gaussian):
    """
    Get efficient white colors for mask rendering with GS.
    Returns colors tensor compatible with colors_overwrite parameter.
    """
    # Create white colors (1.0, 1.0, 1.0) for all Gaussians for mask rendering
    num_gaussians = gaussian._features_dc.shape[0]
    return torch.ones(num_gaussians, 3, device=gaussian._features_dc.device, dtype=gaussian._features_dc.dtype)

def get_gs_transformed(Gaussian, tfm_ori, scale_factor, device):
    """
    Apply initial transformation to Gaussian Splatting object.
    Similar to get_mesh but for GS.

    Args:
        Gaussian: Original Gaussian object
        tfm_ori: Transformation to apply to positions AND rotations
        scale_factor: Scale factor to apply to Gaussian sizes (can be scalar or tensor)
        device: Device
    """
    # Work with a copy to avoid modifying the original
    gs_copy = safe_copy_gaussian(Gaussian)

    # Get initial Gaussian positions
    initial_positions = gs_copy.get_xyz

    logger.info(f"loaded gs shape is {initial_positions.shape}")

    # Apply transformation to positions
    points_world = tfm_ori.transform_points(initial_positions.unsqueeze(0))
    gs_copy.from_xyz(points_world[0])

    # Apply scale to Gaussian scaling parameters (correct log-space handling)
    if scale_factor is not None:
        if torch.is_tensor(scale_factor):
            if scale_factor.dim() == 0:  # scalar
                scale_tensor = scale_factor.expand_as(gs_copy._scaling)
            else:  # vector [sx, sy, sz]
                scale_tensor = scale_factor.expand_as(gs_copy._scaling)
        else:  # float/int
            scale_tensor = torch.tensor(scale_factor, device=device).expand_as(gs_copy._scaling)

        # _scaling is in log space, so add log(scale_factor)
        gs_copy._scaling = gs_copy._scaling + torch.log(scale_tensor)

    # Apply transformation rotation to Gaussian _rotation parameters
    # Extract rotation component from transformation matrix
    tfm_matrix = tfm_ori.get_matrix()[0]  # [4, 4]
    rotation_matrix = tfm_matrix[:3, :3]  # [3, 3]

    # Normalize the rotation matrix to handle scaling effects
    # Extract scale factors to get pure rotation matrix
    scale_factors = rotation_matrix.norm(dim=0)  # [3]
    pure_rotation_matrix = rotation_matrix / scale_factors[None, :]  # [3, 3]

    # Convert rotation matrix to quaternion
    tfm_rotation_quat = matrix_to_quaternion(pure_rotation_matrix[None])  # [1, 4]

    # Get current Gaussian rotations as quaternions
    current_rotations = gs_copy.get_rotation  # [N, 4]

    # Apply transformation rotation to each Gaussian's rotation
    # Broadcast transformation rotation to all Gaussians and multiply
    tfm_quat_broadcasted = tfm_rotation_quat.expand_as(current_rotations)  # [N, 4]
    new_rotations = quaternion_multiply(tfm_quat_broadcasted, current_rotations)
    gs_copy.from_rotation(new_rotations)

    return gs_copy, initial_positions


def get_gs_mask_renderer(Mask, min_size, Intrinsics, device, backend="gsplat"):
    """
    Setup GS renderer for mask rendering.
    Forces square rendering (H=W) to work with unmodified gaussian_render.py.
    """
    # Use exact same mask resize code as get_mask_renderer
    orig_h, orig_w = Mask.shape[-2:]
    min_orig_size = min(orig_w, orig_h)
    scale_factor = min_size / min_orig_size
    mask = F.interpolate(
        Mask[None, None],
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )
    H, W = mask.shape[-2:]

    # Force square rendering: pad the smaller dimension to make it square
    square_size = max(H, W)
    if H != W:
        if H < W:
            # Pad height to match width
            pad_h = (W - H) // 2
            pad_remaining = W - H - pad_h
            mask = F.pad(mask, (0, 0, pad_h, pad_remaining), mode='constant', value=0)
        else:  # W < H
            # Pad width to match height
            pad_w = (H - W) // 2
            pad_remaining = H - W - pad_w
            mask = F.pad(mask, (pad_w, pad_remaining, 0, 0), mode='constant', value=0)

        # Update dimensions after padding
        H, W = mask.shape[-2:]
        assert H == W, f"Expected square mask after padding, got {H}x{W}"

    # GS renderer expects NORMALIZED intrinsics (gsplat backend unnormalizes them internally)
    intrinsics_tensor = Intrinsics.to(device)

    # Setup GS renderer with square resolution only (no modifications to gaussian_render.py needed)
    gs_renderer = GaussianRenderer({
        "resolution": square_size,
        "near": 0.8,
        "far": 1.6,
        "ssaa": 1,
        "bg_color": (0.0, 0.0, 0.0),
        "backend": backend
    })

    return mask, gs_renderer, intrinsics_tensor


def run_gs_alignment(
    Point_Map,
    mask,
    gaussian,
    center,
    renderer,
    intrinsics,
    device,
    align_pm_coordinate=False,
):
    """
    Manual alignment between Gaussian positions and point cloud.
    Similar to run_alignment but for GS.
    """

    # Get rid of flying points using depth edge detection
    # Convert mask to 2D for depth_edge function
    mask_2d = mask[0, 0].bool().cpu().numpy()
    depth = Point_Map[..., 2].cpu().numpy()

    # Remove flying points (large depth discontinuities)
    depth_edge_mask = depth_edge(depth, rtol=0.03, mask=mask_2d)
    cleaned_mask = mask_2d & ~depth_edge_mask

    # Convert back to torch tensor and apply to get target points
    cleaned_mask_tensor = torch.from_numpy(cleaned_mask).to(Point_Map.device)
    target_object_points = Point_Map[cleaned_mask_tensor]

    # Remove inf values
    finite_mask = torch.isfinite(target_object_points).all(dim=1)
    target_object_points = target_object_points[finite_mask]

    # Apply coordinate alignment if needed
    if align_pm_coordinate:
        target_object_points[:, 0] *= -1
        target_object_points[:, 1] *= -1
    flag_notgt = False

    if target_object_points.shape[0] == 0:
        flag_notgt = True
        return None, None, None, None, None, None, None, flag_notgt

    # Get source points (Gaussian positions) and target points
    source_points, target_points = gaussian.get_xyz, target_object_points

    # Align based on height scaling (same logic as mesh version)
    height_src = torch.max(source_points[:, 1]) - torch.min(source_points[:, 1])
    height_tgt = torch.max(target_points[:, 1]) - torch.min(target_points[:, 1])
    scale_1 = height_tgt / height_src

    # Apply scaling to Gaussian positions and scaling parameters
    scaled_positions = source_points * scale_1
    gaussian_aligned = safe_copy_gaussian(gaussian)
    gaussian_aligned.from_xyz(scaled_positions)
    # Scale the Gaussian scaling parameters (correct log-space handling)
    gaussian_aligned._scaling = gaussian._scaling + torch.log(scale_1.expand_as(gaussian._scaling))
    center *= scale_1

    # Center alignment (same as mesh version)
    center_src = torch.mean(scaled_positions, dim=0)
    center_tgt = torch.mean(target_points, dim=0)
    translation_1 = center_tgt - center_src

    # Apply translation
    translated_positions = scaled_positions + translation_1
    gaussian_aligned.from_xyz(translated_positions)
    center += translation_1

    # Create transformation (same as mesh version)
    tfm1 = (
        Transform3d(device=device)
        .scale(scale_1.expand(3)[None])
        .translate(translation_1[None])
    )

    # Apply coordinate conversion for rendering
    gaussian_aligned_opencv = safe_copy_gaussian(gaussian_aligned)
    flip_coords_pytorch3d_to_opencv(gaussian_aligned_opencv)
    extrinsics = torch.eye(4, device=device, dtype=torch.float32)

    # Get mask dimensions for intrinsics correction
    rendered = renderer.render(
        gaussian_aligned_opencv,
        extrinsics,
        intrinsics,
        colors_overwrite=get_mask_colors_for_gs(gaussian_aligned_opencv)
    )
    ori_iou = compute_iou_gs(rendered, mask, threshold=0.5)
    final_iou = ori_iou.cpu().item()

    return translated_positions, target_points, center, tfm1, gaussian_aligned, ori_iou, final_iou, flag_notgt


def run_gs_ICP(source_points, target_points, threshold):
    """
    Run ICP alignment on Gaussian positions.
    Similar to run_ICP but for GS.
    """
    # Convert to Open3D point clouds
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_points.detach().cpu().numpy())

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_points.detach().cpu().numpy())

    # Run ICP
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Apply transformation to source points
    src_pcd.transform(reg_p2p.transformation)
    points_aligned_icp = torch.tensor(
        np.asarray(src_pcd.points), dtype=torch.float32, device=source_points.device
    )

    return points_aligned_icp, reg_p2p.transformation


def apply_icp_transformation_to_gaussian(gaussian, transformation, device):
    """
    Apply ICP transformation matrix to Gaussian scaling and rotation parameters.

    Returns:
        Tuple of (R, scale_icp, t) where:
            - R: Rotation matrix [3, 3]
            - scale_icp: Scale vector [3]
            - t: Translation vector [3]
    """
    # Convert transformation matrix to torch
    T_o3d = torch.tensor(transformation, dtype=torch.float32, device=device)
    T_o3d = T_o3d.T

    # Decompose transformation matrix
    A = T_o3d[:3, :3]
    scale_icp = A.norm(dim=1)
    R = A / scale_icp[:, None]
    t = T_o3d[3, :3]

    # Apply scale to Gaussian scaling parameters (log-space)
    gaussian._scaling = gaussian._scaling + torch.log(scale_icp.expand_as(gaussian._scaling))

    # Apply rotation to Gaussian rotation parameters
    icp_rotation_quat = matrix_to_quaternion(R[None])  # [1, 4]
    current_rotations = gaussian.get_rotation  # [N, 4]
    icp_quat_broadcasted = icp_rotation_quat.expand_as(current_rotations)  # [N, 4]
    new_rotations = quaternion_multiply(icp_quat_broadcasted, current_rotations)
    gaussian.from_rotation(new_rotations)

    return R, scale_icp, t


def prepare_rgb_for_supervision(rgb_gt, mask):
    """
    Prepare RGB ground truth image for supervision by handling format conversion and resizing.

    Args:
        rgb_gt: Ground truth RGB image [3, H, W] or [1, 3, H, W]
        mask: Target mask [1, 1, H, W] to match dimensions

    Returns:
        Prepared RGB image [3, H, W] with dimensions matching mask

    Raises:
        ValueError: If RGB_GT has unexpected shape
    """
    # Handle format conversion
    if rgb_gt.dim() == 3:  # [3, H, W]
        rgb_gt_processed = rgb_gt
    elif rgb_gt.dim() == 4:  # [1, 3, H, W]
        rgb_gt_processed = rgb_gt[0]
    else:
        raise ValueError(f"Unexpected RGB_GT shape: {rgb_gt.shape}. Expected [3, H, W] or [1, 3, H, W]")

    # Check RGB and mask size compatibility
    orig_h, orig_w = rgb_gt_processed.shape[-2:]
    target_h, target_w = mask.shape[-2:]

    if orig_h != target_h or orig_w != target_w:
        logger.warning(f"RGB size ({orig_h}x{orig_w}) doesn't match mask size ({target_h}x{target_w}), resizing RGB")
        # Resize RGB to match processed mask size
        rgb_gt_processed = F.interpolate(
            rgb_gt_processed[None],
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        logger.info(f"RGB resized to {rgb_gt_processed.shape}")

    return rgb_gt_processed


def copy_and_update_gaussian_positions(gaussian, new_positions):
    """
    Create a copy of Gaussian with updated positions.
    """
    gs_copy = safe_copy_gaussian(gaussian)
    gs_copy.from_xyz(new_positions)
    return gs_copy


def safe_copy_gaussian(gaussian):
    """
    Safely copy a Gaussian object, handling tensors with gradients.
    """
    try:
        return copy.deepcopy(gaussian)
    except RuntimeError as e:
        if "deepcopy" in str(e) and "graph leaves" in str(e):
            # Handle tensors with gradients - detach first, then copy
            gs_copy = copy.copy(gaussian)  # Shallow copy first

            # Deep copy the tensor attributes, detaching if needed
            for attr_name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
                if hasattr(gaussian, attr_name):
                    attr_value = getattr(gaussian, attr_name)
                    if attr_value is not None:
                        if torch.is_tensor(attr_value) and attr_value.requires_grad:
                            # Detach tensor and clone
                            setattr(gs_copy, attr_name, attr_value.detach().clone())
                        elif torch.is_tensor(attr_value):
                            # Clone tensor without gradients
                            setattr(gs_copy, attr_name, attr_value.clone())
                        else:
                            # Non-tensor attribute
                            setattr(gs_copy, attr_name, copy.deepcopy(attr_value))

            # Copy other attributes
            for attr_name, attr_value in gaussian.__dict__.items():
                if not attr_name.startswith('_') or attr_name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
                    continue  # Skip already handled or special attributes

                if torch.is_tensor(attr_value):
                    if attr_value.requires_grad:
                        setattr(gs_copy, attr_name, attr_value.detach().clone())
                    else:
                        setattr(gs_copy, attr_name, attr_value.clone())
                else:
                    setattr(gs_copy, attr_name, copy.deepcopy(attr_value))

            return gs_copy
        else:
            raise


def apply_gs_transform_inplace(gaussian, center, quat, translation, scale, backup_data=None):
    """
    Apply transformation to Gaussian Splatting object IN-PLACE for efficiency.
    Returns backup data to restore later.
    """
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)

    # Store original values for restoration if needed
    if backup_data is None:
        backup_data = {
            '_xyz': gaussian._xyz.clone(),
            '_scaling': gaussian._scaling.clone(),
            '_rotation': gaussian.get_rotation.clone()
        }

    # Transform Gaussian positions (same logic as mesh version)
    positions = gaussian.get_xyz
    # transform to the world coordinate system center.
    verts = positions - center
    # perform operation
    verts = verts * scale
    verts = verts @ R.transpose(0, 1)
    # transform back to the original position after rotation.
    verts += center
    verts = verts + translation

    # Update Gaussian in-place
    gaussian.from_xyz(verts)

    # Also scale the Gaussian scaling parameters (correct log-space handling)
    if scale.dim() == 0:
        scale_tensor = scale.expand_as(backup_data['_scaling'])
    else:
        scale_tensor = scale
    gaussian._scaling = backup_data['_scaling'] + torch.log(scale_tensor)

    # Also apply rotation to Gaussian rotation parameters (consistent with Steps 1 & 2)
    # Convert rotation matrix to quaternion
    rotation_quat = matrix_to_quaternion(R[None])  # [1, 4]

    # Apply rotation to each Gaussian's rotation
    current_rotations = backup_data['_rotation']
    rotation_quat_broadcasted = rotation_quat.expand_as(current_rotations)  # [N, 4]
    new_rotations = quaternion_multiply(rotation_quat_broadcasted, current_rotations)
    gaussian.from_rotation(new_rotations)

    return backup_data


def apply_gs_transform_inplace_no_backup(gaussian, center, quat, translation, scale, backup_data):
    """
    Apply transformation to Gaussian Splatting object IN-PLACE for efficiency.
    Uses provided backup_data without creating new backups (more efficient for optimization loops).
    """
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)

    # Transform Gaussian positions (same logic as mesh version)
    positions = gaussian.get_xyz
    # transform to the world coordinate system center.
    verts = positions - center
    # perform operation
    verts = verts * scale
    verts = verts @ R.transpose(0, 1)
    # transform back to the original position after rotation.
    verts += center
    verts = verts + translation

    # Update Gaussian in-place (differentiable)
    gaussian.from_xyz(verts)

    # Also scale the Gaussian scaling parameters (correct log-space handling)
    if scale.dim() == 0:
        scale_tensor = scale.expand_as(backup_data['_scaling'])
    else:
        scale_tensor = scale
    gaussian._scaling = backup_data['_scaling'] + torch.log(scale_tensor)

    # Also apply rotation to Gaussian rotation parameters (consistent with Steps 1 & 2)
    # Convert rotation matrix to quaternion
    rotation_quat = matrix_to_quaternion(R[None])  # [1, 4]

    # Apply rotation to each Gaussian's rotation
    current_rotations = backup_data['_rotation']
    rotation_quat_broadcasted = rotation_quat.expand_as(current_rotations)  # [N, 4]
    new_rotations = quaternion_multiply(rotation_quat_broadcasted, current_rotations)
    gaussian.from_rotation(new_rotations)


def restore_gs_transform(gaussian, backup_data):
    """
    Restore Gaussian to its original state using backup data.
    """
    gaussian._xyz = backup_data['_xyz']
    gaussian._scaling = backup_data['_scaling']
    gaussian.from_rotation(backup_data['_rotation'])


def apply_gs_transform(gaussian, center, quat, translation, scale):
    """
    Legacy function for compatibility - creates a copy.
    Use apply_gs_transform_inplace for better performance.
    """
    gs_transformed = safe_copy_gaussian(gaussian)
    apply_gs_transform_inplace(gs_transformed, center, quat, translation, scale)
    return gs_transformed


def run_gs_render_compare_rgb_mask(gaussian, center, renderer, intrinsics, mask, rgb_image, device,
                                  return_renderings=False):
    """
    Args:
        gaussian: Gaussian splatting object
        center: Center point for rotation
        renderer: GS renderer
        intrinsics: Camera intrinsics
        mask: Target mask [1, 1, H, W]
        rgb_image: Target RGB image [3, H, W] or [1, 3, H, W]
        device: Torch device
        return_renderings: If True, return initial and final renderings for visualization

    Returns:
        If return_renderings=False: Tuple of (quaternion, translation, scale, rotation_matrix)
        If return_renderings=True: Tuple of (quaternion, translation, scale, rotation_matrix, initial_rendering, final_rendering)
    """
    initial_rendering = None
    if return_renderings:
        with torch.no_grad():
            gaussian_initial_copy = safe_copy_gaussian(gaussian)
            flip_coords_pytorch3d_to_opencv(gaussian_initial_copy)
            extrinsics = torch.eye(4, device=device, dtype=torch.float32)
            initial_rendering = renderer.render(
                gaussian_initial_copy,
                extrinsics,
                intrinsics,
            )

    # Full quaternion rotation (4 DOF)
    quat = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True)
    )
    # Full XYZ translation optimization (3 DOF)
    translation_xyz = torch.nn.Parameter(
        torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)
    )
    # SCALE
    scale = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))

    def get_optimizer(stage):
        if stage == 1:
            return torch.optim.Adam([translation_xyz, scale], lr=1e-2)
        elif stage == 2:
            return torch.optim.Adam([quat, translation_xyz, scale], lr=5e-3)

    # Enhanced loss weights including RGB
    loss_weights = {
        "rgb": 10,       # RGB supervision weight
        "mask": 10,       # Mask supervision weight
        "reg_q": 0.1,    # Quaternion regularization
        "reg_t": 0.05,   # Translation regularization
        "reg_s": 0.05    # Scale regularization
    }
    prev_loss = None

    backup_data = {
        '_xyz': gaussian._xyz.clone(),
        '_scaling': gaussian._scaling.clone(),
        '_rotation': gaussian.get_rotation.clone()
    }

    global_step = 0
    for stage in [1, 2]:
        optimizer = get_optimizer(stage)
        iters = [5, 25]
        for i in range(iters[stage - 1]):
            optimizer.zero_grad()
            translation = translation_xyz
            apply_gs_transform_inplace_no_backup(gaussian, center, quat, translation, scale, backup_data)

            # Apply coordinate conversion for rendering
            flip_coords_pytorch3d_to_opencv(gaussian)
            extrinsics = torch.eye(4, device=device, dtype=torch.float32)

            rendered = renderer.render(
                gaussian,
                extrinsics,
                intrinsics,
            )

            loss, loss_details = compute_gs_loss_rgb_mask(
                rendered, mask, rgb_image, loss_weights, quat, translation_xyz, scale
            )

            loss.backward()
            optimizer.step()
            global_step += 1

            # Restore to original clean state
            restore_gs_transform(gaussian, backup_data)

            # Early stopping with loss details logging
            if prev_loss is not None and abs(loss.item() - prev_loss) < 1e-6:
                break
            prev_loss = loss.item()

            # Log progress occasionally
            if i % 15 == 0:
                logger.info(f"   Stage {stage}, Iter {i}: RGB={loss_details['loss_rgb']:.4f}, Mask={loss_details['loss_mask']:.4f} (src: {loss_details['mask_source']}), Total={loss_details['total_loss']:.4f}")

    # Convert final results to the same format as the original function
    quat, translation_xyz, scale = quat.detach(), translation_xyz.detach(), scale.detach()
    quat_normalized = quat / quat.norm()
    R = quaternion_to_matrix(quat_normalized)
    translation = translation_xyz

    # Capture final rendering if requested
    final_rendering = None
    if return_renderings:
        with torch.no_grad():
            # Apply final transformation to get optimized Gaussian
            gaussian_final_copy = safe_copy_gaussian(gaussian)
            apply_gs_transform_inplace(gaussian_final_copy, center, quat_normalized, translation, scale)
            flip_coords_pytorch3d_to_opencv(gaussian_final_copy)
            extrinsics = torch.eye(4, device=device, dtype=torch.float32)
            final_rendering = renderer.render(
                gaussian_final_copy,
                extrinsics,
                intrinsics,
            )

    if return_renderings:
        return quat_normalized, translation, scale, R, initial_rendering, final_rendering
    else:
        return quat_normalized, translation, scale, R

def compute_gs_loss(rendered, mask_gt, loss_weights, quat, translation, scale):
    """
    Compute loss for GS render-and-compare optimization.
    Similar to compute_loss but for GS.
    """
    # Extract mask using helper function with optimization mode
    pred_mask = extract_mask_from_gs_rendering(rendered, mode="optimization")

    # === 1. MSE Loss on mask (same as mesh version) ===
    loss_mask = F.mse_loss(pred_mask, mask_gt[0, 0])

    # === 2. Reg Loss (same as mesh version) ===
    quat_normalized = quat / quat.norm()
    loss_reg_q = F.mse_loss(
        quat_normalized, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device)
    )
    loss_reg_t = torch.norm(translation) ** 2
    loss_reg_s = (scale - 1.0) ** 2

    # === Total weighted loss (same as mesh version) ===
    total_loss = (
        loss_weights["mask"] * loss_mask
        + loss_weights["reg_q"] * loss_reg_q
        + loss_weights["reg_t"] * loss_reg_t
        + loss_weights["reg_s"] * loss_reg_s
    )

    return total_loss


def flip_coords_pytorch3d_to_opencv(gaussian):
    """
    Flip X,Y coordinates from PyTorch3D to OpenCV convention IN-PLACE.

    Transforms both positions AND rotations consistently with other transformation functions.

    Returns: The same gaussian object (modified in-place)
    """
    # Get AABB-denormalized coordinates, flip X,Y, then set back using proper API
    denormalized_xyz = gaussian.get_xyz
    denormalized_xyz[:, 0] *= -1  # Flip X
    denormalized_xyz[:, 1] *= -1  # Flip Y
    gaussian.from_xyz(denormalized_xyz)

    # Also transform rotations to be consistent with the coordinate system flip
    current_rotations = gaussian.get_rotation  # [N, 4] - actual unit quaternions
    
    coord_flip_matrix = torch.tensor([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ], device=current_rotations.device, dtype=current_rotations.dtype)

    # Convert coordinate flip to quaternion
    coord_flip_quat = matrix_to_quaternion(coord_flip_matrix[None])  # [1, 4]

    # Apply coordinate flip transformation to each Gaussian's rotation
    coord_flip_quat_broadcasted = coord_flip_quat.expand_as(current_rotations)  # [N, 4]
    new_rotations = quaternion_multiply(coord_flip_quat_broadcasted, current_rotations)
    gaussian.from_rotation(new_rotations)

    return gaussian


def compute_gs_loss_rgb_mask(rendered, mask_gt, rgb_gt, loss_weights, quat, translation_xyz, scale):
    """
    Loss function for GS optimization using both RGB and mask supervisions.

    Args:
        rendered: GS rendered output with 'color' key and 'alpha' key
        mask_gt: Ground truth mask [1, 1, H, W]
        rgb_gt: Ground truth RGB image [3, H, W] or [1, 3, H, W]
        loss_weights: Dictionary with loss weights
        quat: Full quaternion parameters [4] (w, x, y, z)
        translation_xyz: Full XYZ translation parameters [3]
        scale: Scale parameter [1]

    Returns:
        total_loss: Combined loss value
        loss_details: Dictionary with individual loss components
    """

    # Extract RGB and mask using helper functions
    pred_rgb_channels = extract_rgb_from_gs_rendering(rendered)
    pred_mask = extract_mask_from_gs_rendering(rendered, mode="optimization")

    # Track mask source for logging
    if "alpha" in rendered:
        mask_source = "alpha_channel"
    elif "rgba" in rendered:
        mask_source = "rgba_alpha"
    else:
        mask_source = "rgb_grayscale"
        logger.warning("Using RGB->grayscale for mask loss - may not work well with dark objects.")

    # Prepare ground truth RGB - handle both [3, H, W] and [1, 3, H, W] formats
    if rgb_gt.dim() == 4 and rgb_gt.shape[0] == 1:
        rgb_gt = rgb_gt[0]  # Remove batch dimension: [1, 3, H, W] -> [3, H, W]

    # Prepare ground truth mask - extract 2D mask
    mask_gt_2d = mask_gt[0, 0]  # [1, 1, H, W] -> [H, W]

    # === 1. RGB Loss (MSE on RGB channels in masked regions) ===
    # Apply mask to focus RGB loss on valid regions
    mask_expanded = (mask_gt_2d > 0.5).float()  # [H, W]
    mask_3d = mask_expanded[None, :, :].expand_as(pred_rgb_channels)  # [3, H, W]

    # RGB loss only in masked regions - compute loss only on valid pixels to avoid gradient dilution
    # Extract only the valid (masked) pixels for loss computation
    valid_pixels = mask_3d > 0.5
    if torch.sum(valid_pixels) > 0:
        pred_rgb_valid = pred_rgb_channels[valid_pixels]
        gt_rgb_valid = rgb_gt[valid_pixels]
        loss_rgb = F.mse_loss(pred_rgb_valid, gt_rgb_valid)
    else:
        loss_rgb = torch.tensor(0.0, device=pred_rgb_channels.device, requires_grad=True)

    # === 2. Mask Loss (MSE on mask) ===
    loss_mask = F.mse_loss(pred_mask, mask_gt_2d)

    # === 3. Regularization Losses ===
    # Full quaternion regularization (encourage identity rotation)
    quat_normalized = quat / quat.norm()
    loss_reg_q = F.mse_loss(
        quat_normalized, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device)
    )
    # Full XYZ translation regularization
    loss_reg_t = torch.norm(translation_xyz) ** 2
    # Scale regularization
    loss_reg_s = (scale - 1.0) ** 2

    # === Total weighted loss ===
    total_loss = (
        loss_weights["rgb"] * loss_rgb +
        loss_weights["mask"] * loss_mask +
        loss_weights["reg_q"] * loss_reg_q +
        loss_weights["reg_t"] * loss_reg_t +
        loss_weights["reg_s"] * loss_reg_s
    )

    loss_details = {
        'loss_rgb': loss_rgb.item(),
        'loss_mask': loss_mask.item(),
        'loss_reg_q': loss_reg_q.item(),
        'loss_reg_t': loss_reg_t.item(),
        'loss_reg_s': loss_reg_s.item(),
        'total_loss': total_loss.item(),
        'mask_source': mask_source
    }

    return total_loss, loss_details


def compute_iou_gs(rendered, mask_obj_gt, threshold=0.5):
    """
    Compute IoU for GS rendering.
    Similar to compute_iou but for GS.
    """
    # Extract mask using helper function with evaluation mode (sharper boundaries)
    render_mask = extract_mask_from_gs_rendering(rendered, mode="evaluation")

    # Ensure correct shape
    if render_mask.dim() == 2:
        render_mask = render_mask[None, None]

    # Binarize masks (same as mesh version)
    pred = (render_mask > threshold).float()
    gt_obj = (mask_obj_gt > threshold).float()

    # Compute intersection and union (same as mesh version)
    intersection = (pred * gt_obj).sum()
    union = ((pred + gt_obj) > 0).float().sum()

    if union == 0:
        return torch.tensor(1.0 if intersection == 0 else 0.0)  # avoid division by zero

    iou = intersection / union
    return iou