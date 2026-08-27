import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PYTORCH3D_DIR = (
    Path(__file__).resolve().parents[3] / "third_party" / "pytorch3d_phystwin"
)
if PYTORCH3D_DIR.exists():
    sys.path.insert(0, str(PYTORCH3D_DIR))

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    AmbientLights,
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from scipy.spatial import cKDTree


def select_point(points, raw_matching_points, object_mask):
    mask_coords = np.column_stack(np.where(object_mask > 0))
    mask_coords = mask_coords[:, ::-1]
    tree = cKDTree(mask_coords)

    distances, indices = tree.query(raw_matching_points)

    new_match = mask_coords[indices]
    # Pay attention to the case that the keypoint is in different order
    matched_points = points[new_match[:, 1], new_match[:, 0]]
    return mask_coords[indices], matched_points


def plot_mesh_with_points(mesh, points, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        alpha=0.5,
        edgecolor="none",
        color="lightgrey",
    )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="red", s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    ax.set_title("3D Mesh with Projected Points")
    plt.savefig(filename)
    plt.clf()


def plot_image_with_points(image, points, save_dir, points2=None):
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], color="red", s=5)
    if points2 is not None:
        plt.scatter(points2[:, 0], points2[:, 1], color="blue", s=5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Points on Original Image")
    plt.savefig(save_dir)
    plt.clf()


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):

        # Extract all meshes from the scene
        meshes = []
        for name, geometry in scene_or_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        # Combine all meshes if there are multiple
        if len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        elif len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            raise ValueError("No valid meshes found in the GLB file")

        # Get model metadata
        metadata = {
            "vertices": combined_mesh.vertices.shape[0],
            "faces": combined_mesh.faces.shape[0],
            "bounds": combined_mesh.bounds.tolist(),
            "center_mass": combined_mesh.center_mass.tolist(),
            "is_watertight": combined_mesh.is_watertight,
            "original_scene": combined_mesh,  # Keep reference to original scene
        }

        mesh = combined_mesh
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def project_2d_to_3d(image_points, depth, camera_intrinsics, camera_pose):
    """
    Project 2D image points to 3D space using the depth map, camera intrinsics, and pose.

    :param image_points: Nx2 array of image points
    :param depth: Depth map
    :param camera_intrinsics: Camera intrinsic matrix
    :param camera_pose: 4x4 camera pose matrix
    :return: Nx3 array of 3D points in world coordinates
    """
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    # Convert image points to normalized device coordinates (NDC)
    ndc_points = np.zeros((image_points.shape[0], 3))
    valid_depth_coords = np.column_stack(np.where(depth > 0))
    depth_tree = None
    if len(valid_depth_coords) > 0:
        depth_tree = cKDTree(valid_depth_coords[:, ::-1])
    for i, (u, v) in enumerate(image_points):
        u = int(u)
        v = int(v)
        z = depth[v, u]
        if z <= 0 and depth_tree is not None:
            _, nearest_idx = depth_tree.query([u, v])
            u, v = valid_depth_coords[nearest_idx, ::-1]
            z = depth[v, u]
        x = -(u - cx) * z / fx
        y = -(v - cy) * z / fy
        ndc_points[i] = [x, y, z]
    valid_mask = ndc_points[:, 2] > 0
    ndc_points = ndc_points[valid_mask]
    # ndc_points = np.vstack((ndc_points, np.zeros(3), [[0, 0, 0]])) # modified
    # Convert from camera coordinates to world coordinates
    ndc_points_homogeneous = np.hstack((ndc_points, np.ones((ndc_points.shape[0], 1))))
    world_points_homogeneous = ndc_points_homogeneous @ np.linalg.inv(camera_pose)
    return world_points_homogeneous[:, :3], valid_mask


def sample_camera_poses(radius, num_samples, num_up_samples=4, device="cpu"):
    """
    Generate camera poses around a sphere with a given radius.
    camera_poses: A list of 4x4 transformation matrices representing the camera poses.
    camera_view_coord = word_coord @ camera_pose
    """
    camera_poses = []
    phi = np.linspace(0, np.pi, num_samples)  # Elevation angle
    phi = phi[1:-1]  # Exclude poles
    theta = np.linspace(0, 2 * np.pi, num_samples)  # Azimuthal angle

    # Generate different up vectors
    up_vectors = [np.array([0, 0, 1])]  # z-axis is up
    for i in range(1, num_up_samples):
        angle = (i / num_up_samples) * np.pi * 2
        up = np.array([np.sin(angle), 0, np.cos(angle)])  # Rotate around y-axis
        up_vectors.append(up)

    for p in phi:
        for t in theta:
            for up in up_vectors:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = radius * np.cos(p)
                position = np.array([x, y, z])[None, :]
                lookat = np.array([0, 0, 0])[None, :]
                up = up[None, :]
                R, T = look_at_view_transform(radius, t, p, False, position, lookat, up)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = R
                camera_pose[3, :3] = T
                camera_poses.append(camera_pose)

    print("total poses", len(camera_poses))
    return torch.tensor(np.array(camera_poses), device=device)


def render_image(mesh, camera_poses, width=640, height=480, fov=1, device="cpu"):
    camera_poses = torch.tensor(camera_poses, device=device)
    if len(camera_poses.shape) == 2:
        camera_poses = camera_poses[None, :]

    from pytorch3d.io import IO
    from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    mesh = io.load_mesh(mesh)
    mesh = mesh.to(device)
    if mesh.textures is None:
        mesh.textures = TexturesVertex(
            verts_features=torch.full_like(mesh.verts_padded(), 0.7)
        )

    R = camera_poses[:, :3, :3]
    T = camera_poses[:, 3, :3]
    num_poses = camera_poses.shape[0]
    cameras = PerspectiveCameras(
        R=R,
        T=T,
        device=device,
        focal_length=torch.ones(num_poses, 1)
        * 0.5
        * width
        / np.tan(fov / 2),  # Calculate focal length from FOV in radians
        principal_point=torch.tensor((width / 2, height / 2))
        .repeat(num_poses)
        .reshape(-1, 2),  # different order from image_size!!
        image_size=torch.tensor((height, width)).repeat(num_poses).reshape(-1, 2),
        in_ndc=False,
    )

    lights = AmbientLights(device=device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            blend_params=BlendParams(background_color=(0, 0, 0)),
            cameras=cameras,
            lights=lights,
        ),
    )
    extended_mesh = mesh.extend(num_poses).to(device)
    fragments = renderer.rasterizer(extended_mesh)
    depth = fragments.zbuf.squeeze().cpu().numpy()
    rendered_images = renderer(mesh.extend(num_poses))
    color = (rendered_images[..., :3].cpu().numpy() * 255).astype(np.uint8)

    return color, depth


def render_multi_images(
    mesh,
    width=640,
    height=480,
    fov=1,
    radius=3.0,
    num_samples=6,
    num_ups=2,
    device="cpu",
):
    # Sample camera poses
    camera_poses = sample_camera_poses(radius, num_samples, num_ups, device)

    # Calculate intrinsics
    fx = 0.5 * width / np.tan(fov / 2)
    fy = fx  # * aspect_ratio
    cx, cy = width / 2, height / 2
    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    num_cameras = camera_poses.shape[0]

    # Render two times to avoid memory overflow
    split = num_cameras // 2
    color1, depth1 = render_image(
        mesh, camera_poses[:split], width, height, fov, device
    )
    color2, depth2 = render_image(
        mesh, camera_poses[split:], width, height, fov, device
    )
    color = np.concatenate([color1, color2], axis=0)
    depth = np.concatenate([depth1, depth2], axis=0)
    return color, depth, camera_poses, camera_intrinsics
