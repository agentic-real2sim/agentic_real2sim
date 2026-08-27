import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import pickle
import trimesh
import cv2
import json
import torch
import os
import math
from ar2s.phystwin_visual.utils.align_util import (
    render_multi_images,
    render_image,
    as_mesh,
    project_2d_to_3d,
    plot_mesh_with_points,
    plot_image_with_points,
    select_point,
)
from ar2s.phystwin_visual.match_pairs import image_pair_matching
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree

VIS = True
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name
output_dir = f"{base_path}/{case_name}/shape/matching"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def matrix_to_realsim_euler_deg(rotation_matrix):
    sp = -rotation_matrix[2, 0]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = math.asin(sp)

    if abs(sp) < 0.99999:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = 0.0

    return np.array(
        [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
        dtype=np.float64,
    )


def pose_selection_render_superglue(
    raw_img, fov, mesh_path, mesh, crop_img, output_dir
):
    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)

    # Render multimle images and feature matching
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    # Use superglue to match the features
    best_idx, match_result, match_results, match_nums = image_pair_matching(
        grays, crop_img, output_dir, viz_best=True, return_all=True
    )
    depth_match_nums = []
    for depth, result in zip(depths, match_results):
        valid_matches = result["matches"] > -1
        render_matching_points = result["keypoints0"][valid_matches]
        valid_depth_num = 0
        for u, v in render_matching_points:
            u = int(u)
            v = int(v)
            if (
                0 <= v < depth.shape[0]
                and 0 <= u < depth.shape[1]
                and depth[v, u] > 0
            ):
                valid_depth_num += 1
        depth_match_nums.append(valid_depth_num)
    depth_match_nums = np.array(depth_match_nums)
    projectable_indices = np.where(depth_match_nums >= 4)[0]
    if len(projectable_indices) > 0:
        best_idx = projectable_indices[
            np.argmax(np.array(match_nums)[projectable_indices])
        ]
        match_result = match_results[best_idx]
    print("matched point number", np.sum(match_result["matches"] > -1))
    print("projectable matched point number", depth_match_nums[best_idx])

    best_color = colors[best_idx]
    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()
    return best_color, best_depth, best_pose, match_result, camera_intrinsics


def registration_pnp(mesh_matching_points, raw_matching_points, intrinsic):
    # Solve the PNP and verify the reprojection error
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    assert success, "solvePnP failed"
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed for this case {case_name}.$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()

    return mesh2raw_camera


def registration_scale(mesh_matching_points_cam, matching_points_cam):
    # After PNP, optimize the scale in the camera coordinate
    def objective(scale, mesh_points, pcd_points):
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        return loss

    initial_scale = 1
    result = minimize(
        objective,
        initial_scale,
        args=(mesh_matching_points_cam, matching_points_cam),
        method="L-BFGS-B",
    )
    optimal_scale = result.x[0]
    print("Rescale:", optimal_scale)
    return optimal_scale


def deform_ARAP(initial_mesh_world, mesh_matching_points_world, matching_points):
    # Do the ARAP deformation based on the matching keypoints
    mesh_vertices = np.asarray(initial_mesh_world.vertices)
    kdtree = KDTree(mesh_vertices)
    _, mesh_points_indices = kdtree.query(mesh_matching_points_world)
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)
    deform_mesh = initial_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matching_points),
        max_iter=1,
    )
    return deform_mesh, mesh_points_indices


def get_matching_ray_registration(
    mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
):
    # Get the matching indices and targets based on the viewpoint
    obs_points_cam = np.dot(
        w2c,
        np.hstack((obs_points_world, np.ones((obs_points_world.shape[0], 1)))).T,
    ).T
    obs_points_cam = obs_points_cam[:, :3]
    vertices_cam = np.dot(
        w2c,
        np.hstack(
            (
                np.asarray(mesh_world.vertices),
                np.ones((np.asarray(mesh_world.vertices).shape[0], 1)),
            )
        ).T,
    ).T
    vertices_cam = vertices_cam[:, :3]

    obs_kd = KDTree(obs_points_cam)

    new_indices = []
    new_targets = []
    # trimesh used to do the ray-casting test
    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]
    for index, vertex in enumerate(vertices_cam):
        ray_origins = np.array([[0, 0, 0]])
        ray_direction = vertex
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        ray_directions = np.array([ray_direction])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )

        ignore_flag = False

        if len(locations) > 0:
            first_intersection = locations[0]
            vertex_distance = np.linalg.norm(vertex)
            intersection_distance = np.linalg.norm(first_intersection)
            if intersection_distance < vertex_distance - 1e-4:
                # If the intersection point is not the vertex, it means the vertex is not visible from the camera viewpoint
                ignore_flag = True

        if ignore_flag:
            continue
        else:
            # Select the closest point to the ray of the observation points as the matching point
            indices = obs_kd.query_ball_point(vertex, 0.02)
            line_distances = line_point_distance(vertex, obs_points_cam[indices])
            # Get the closest point
            if len(line_distances) > 0:
                closest_index = np.argmin(line_distances)
                target = np.dot(
                    c2w, np.hstack((obs_points_cam[indices][closest_index], 1))
                )
                new_indices.append(index)
                new_targets.append(target[:3])

    new_indices = np.asarray(new_indices)
    new_targets = np.asarray(new_targets)

    return new_indices, new_targets


def deform_ARAP_ray_registration(
    deform_kp_mesh_world,
    obs_points_world,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    mesh_points_indices,
    matching_points,
):
    final_indices = []
    final_targets = []
    for index, target in zip(mesh_points_indices, matching_points):
        if index not in final_indices:
            final_indices.append(index)
            final_targets.append(target)

    for c2w, w2c in zip(c2ws, w2cs):
        new_indices, new_targets = get_matching_ray_registration(
            deform_kp_mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
        )
        for index, target in zip(new_indices, new_targets):
            if index not in final_indices:
                final_indices.append(index)
                final_targets.append(target)

    # Also need to adjust the positions to make sure they are above the table
    indices = np.where(np.asarray(deform_kp_mesh_world.vertices)[:, 2] > 0)[0]
    for index in indices:
        if index not in final_indices:
            final_indices.append(index)
            target = np.asarray(deform_kp_mesh_world.vertices)[index].copy()
            target[2] = 0
            final_targets.append(target)
        else:
            target = final_targets[final_indices.index(index)]
            if target[2] > 0:
                target[2] = 0
                final_targets[final_indices.index(index)] = target

    final_mesh_world = deform_kp_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )
    return final_mesh_world


def line_point_distance(p, points):
    # Compute the distance between points and the line between p and [0, 0, 0]
    p = p / np.linalg.norm(p)
    points_to_origin = points
    cross_product = np.linalg.norm(np.cross(points_to_origin, p), axis=1)
    return cross_product / np.linalg.norm(p)


if __name__ == "__main__":
    existDir(output_dir)

    cam_idx = 0
    img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"
    mesh_path = f"{base_path}/{case_name}/shape/object.glb"
    # Get the mask index of the object
    with open(f"{base_path}/{case_name}/mask/mask_info_{cam_idx}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_img_path = f"{base_path}/{case_name}/mask/{cam_idx}/{obj_idx}/0.png"
    # Load the metadata
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsic = np.array(data["intrinsics"])[cam_idx]

    # Load the c2w for the camera
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
        c2w = c2ws[cam_idx]
        w2c = np.linalg.inv(c2w)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]

    # Load the shape prior
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh = as_mesh(mesh)

    # Load and process the image to get a cropped version for easy superglue
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # Get mask bounding box, larger than the original bounding box
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    if not os.path.exists(f"{output_dir}/best_match.pkl"):
        # 2D feature Matching to get the best pose of the object
        bbox = np.argwhere(mask_img > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2),
        )
        # Make sure the bounding box is within the image
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(raw_img.shape[1], bbox[2]),
            min(raw_img.shape[0], bbox[3]),
        )
        # Get the masked cropped image used for superglue
        crop_img = raw_img.copy()
        mask_bool = mask_img > 0
        crop_img[~mask_bool] = 0
        crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

        # Render the object and match the features
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                fov,
                mesh_path,
                mesh,
                crop_img,
                output_dir=output_dir,
            )
        )
        with open(f"{output_dir}/best_match.pkl", "wb") as f:
            pickle.dump(
                [
                    best_color,
                    best_depth,
                    best_pose,
                    match_result,
                    camera_intrinsics,
                    bbox,
                ],
                f,
            )
    else:
        with open(f"{output_dir}/best_match.pkl", "rb") as f:
            best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                pickle.load(f)
            )

    # Process to get the matching points on the mesh and on the image
    # Get the projected 3D matching points on the mesh
    valid_matches = match_result["matches"] > -1
    render_matching_points = match_result["keypoints0"][valid_matches]
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )
    render_matching_points = render_matching_points[valid_mask]
    # Get the matching points on the raw image
    raw_matching_points_box = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]
    raw_matching_points_box = raw_matching_points_box[valid_mask]
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])

    if VIS:
        # Do visualization for the matching
        plot_mesh_with_points(
            mesh,
            mesh_matching_points,
            f"{output_dir}/mesh_matching.png",
        )
        plot_image_with_points(
            best_depth,
            render_matching_points,
            f"{output_dir}/render_matching.png",
        )
        plot_image_with_points(
            raw_img,
            raw_matching_points,
            f"{output_dir}/raw_matching.png",
        )

    # Do PnP optimization to optimize the rotation between the 3D mesh keypoints and the 2D image keypoints
    mesh2raw_camera = registration_pnp(
        mesh_matching_points, raw_matching_points, intrinsic
    )

    if VIS:
        pnp_camera_pose = np.eye(4, dtype=np.float32)
        pnp_camera_pose[:3, :3] = np.linalg.inv(mesh2raw_camera[:3, :3])
        pnp_camera_pose[3, :3] = mesh2raw_camera[:3, 3]
        pnp_camera_pose[:, :2] = -pnp_camera_pose[:, :2]
        color, depth = render_image(
            mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
        )
        vis_mask = depth > 0
        color[0][~vis_mask] = raw_img[~vis_mask]
        plt.imsave(f"{output_dir}/pnp_results.png", color[0])

    # Transform the mesh into the real world coordinate
    mesh_matching_points_cam = np.dot(
        mesh2raw_camera,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_cam = mesh_matching_points_cam[:, :3]

    # Load the pcd in world coordinate of raw image matching points
    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
    mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    for i in range(3):
        points = data["points"][i]
        colors = data["colors"][i]
        mask = processed_masks[0][i]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])
        if i == 0:
            first_points = points
            first_mask = mask

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    # Find the cloest points for the raw_matching_points
    new_match, matching_points = select_point(
        first_points, raw_matching_points, first_mask
    )
    matching_points_cam = np.dot(
        w2c, np.hstack((matching_points, np.ones((matching_points.shape[0], 1)))).T
    ).T
    matching_points_cam = matching_points_cam[:, :3]

    if VIS:
        # Draw the raw_matching_points and new matching points on the masked
        vis_img = raw_img.copy()
        vis_img[~first_mask] = 0
        plot_image_with_points(
            vis_img,
            raw_matching_points,
            f"{output_dir}/raw_matching_valid.png",
            new_match,
        )

    # Use the matching points in the camera coordinate to optimize the scame between the mesh and the observation
    optimal_scale = registration_scale(mesh_matching_points_cam, matching_points_cam)

    # Compute the rigid transformation from the original mesh to the final world coordinate
    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    mesh2world = np.dot(c2w, np.dot(scale_matrix, mesh2raw_camera))

    linear = mesh2world[:3, :3].astype(np.float64)
    position = mesh2world[:3, 3].astype(np.float64)
    scale_value = float(np.mean(np.linalg.norm(linear, axis=0)))
    rotation_matrix = linear / scale_value
    rotation = matrix_to_realsim_euler_deg(rotation_matrix)
    scale = np.full(3, scale_value, dtype=np.float64)
    np.savez(
        f"{base_path}/{case_name}/mesh_transform.npz",
        position=position,
        rotation=rotation,
        scale=scale,
    )

    mesh_matching_points_world = np.dot(
        mesh2world,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_world = mesh_matching_points_world[:, :3]

    # Do the ARAP based on the matching keypoints
    # Convert the mesh to open3d to use the ARAP function
    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    # Need to remove the duplicated vertices to enable open3d, however, the duplicated points are important in trimesh for texture
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    # Get the index from original vertices to the mesh vertices, mapping between trimesh and open3d
    kdtree = KDTree(initial_mesh_world.vertices)
    _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(mesh2world)

    # ARAP based on the keypoints
    deform_kp_mesh_world, mesh_points_indices = deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, matching_points
    )

    # Do the ARAP based on both the ray-casting matching and the keypoints
    # Identify the vertex which blocks or blocked by the observation, then match them with the observation points on the ray
    final_mesh_world = deform_ARAP_ray_registration(
        deform_kp_mesh_world,
        obs_points,
        mesh,
        trimesh_indices,
        c2ws,
        w2cs,
        mesh_points_indices,
        matching_points,
    )

    if VIS:
        final_mesh_world.compute_vertex_normals()

        # Visualize the partial observation and the mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Render the final stuffs as a turntable video
        vis = o3d.visualization.Visualizer()
        if vis.create_window(visible=False):
            dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            height, width, _ = dummy_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(
                f"{output_dir}/final_matching.mp4", fourcc, 30, (width, height)
            )
            # final_mesh_world.compute_vertex_normals()
            # final_mesh_world.translate([0, 0, 0.2])
            # mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(final_mesh_world)
            # o3d.visualization.draw_geometries([pcd, final_mesh_world], window_name="Matching")
            vis.add_geometry(pcd)
            vis.add_geometry(final_mesh_world)
            # vis.add_geometry(coordinate)
            view_control = vis.get_view_control()

            for j in range(360):
                view_control.rotate(10, 0)
                vis.poll_events()
                vis.update_renderer()
                frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                frame = (frame * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            vis.destroy_window()

    mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]
    mesh.export(f"{output_dir}/final_mesh.glb")
