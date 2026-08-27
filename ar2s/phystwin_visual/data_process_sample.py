# Optionally do the shape completion for the object points (including both suface and interior points)
# Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points

import numpy as np
import open3d as o3d
import pickle
import matplotlib.pyplot as plt
import trimesh
import cv2
from ar2s.phystwin_visual.utils.align_util import as_mesh
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument("--num_surface_points", type=int, default=1024)
parser.add_argument("--volume_sample_size", type=float, default=0.005)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Used to judge if using the shape prior
SHAPE_PRIOR = args.shape_prior
num_surface_points = args.num_surface_points
volume_sample_size = args.volume_sample_size


def sample_volume_mesh_chunked(mesh, count, batch_size=256):
    points = (np.random.random((count, 3)) * mesh.extents) + mesh.bounds[0]
    contained = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        contained.append(mesh.contains(points[start:end]))
    return points[np.concatenate(contained)][:count]


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def process_unique_points(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    # Get the unique index in the object points
    first_object_points = object_points[0]
    unique_idx = np.unique(first_object_points, axis=0, return_index=True)[1]
    object_points = object_points[:, unique_idx, :]
    object_colors = object_colors[:, unique_idx, :]
    object_visibilities = object_visibilities[:, unique_idx]
    object_motions_valid = object_motions_valid[:, unique_idx]

    # Make sure all points are above the ground
    object_points[object_points[..., 2] > 0, 2] = 0

    if SHAPE_PRIOR:
        shape_mesh_path = f"{base_path}/{case_name}/shape/matching/final_mesh.glb"
        trimesh_mesh = trimesh.load(shape_mesh_path, force="mesh")
        trimesh_mesh = as_mesh(trimesh_mesh)
        # Sample the surface points
        surface_points, _ = trimesh.sample.sample_surface(
            trimesh_mesh, num_surface_points
        )
        # Sample the interior points
        interior_points = sample_volume_mesh_chunked(trimesh_mesh, 10000)

    if SHAPE_PRIOR:
        all_points = np.concatenate(
            [surface_points, interior_points, object_points[0]], axis=0
        )
    else:
        all_points = object_points[0]
    # Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
    min_bound = np.min(all_points, axis=0)
    index = []
    grid_flag = {}
    for i in range(object_points.shape[1]):
        grid_index = tuple(
            np.floor((object_points[0, i] - min_bound) / volume_sample_size).astype(int)
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            index.append(i)
    if SHAPE_PRIOR:
        final_surface_points = []
        for i in range(surface_points.shape[0]):
            grid_index = tuple(
                np.floor((surface_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_surface_points.append(surface_points[i])
        final_interior_points = []
        for i in range(interior_points.shape[0]):
            grid_index = tuple(
                np.floor((interior_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_interior_points.append(interior_points[i])
        all_points = np.concatenate(
            [final_surface_points, final_interior_points, object_points[0][index]],
            axis=0,
        )
    else:
        all_points = object_points[0][index]

    # Render the final pcd with interior filling as a turntable video
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(all_points)
    coorindate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    vis = o3d.visualization.Visualizer()
    if vis.create_window(visible=False):
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            f"{base_path}/{case_name}/final_pcd.mp4", fourcc, 30, (width, height)
        )

        vis.add_geometry(all_pcd)
        # vis.add_geometry(coorindate)
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

    track_data.pop("object_points")
    track_data.pop("object_colors")
    track_data.pop("object_visibilities")
    track_data.pop("object_motions_valid")
    track_data["object_points"] = object_points[:, index, :]
    track_data["object_colors"] = object_colors[:, index, :]
    track_data["object_visibilities"] = object_visibilities[:, index]
    track_data["object_motions_valid"] = object_motions_valid[:, index]
    if SHAPE_PRIOR:
        track_data["surface_points"] = np.array(final_surface_points)
        track_data["interior_points"] = np.array(final_interior_points)
    else:
        track_data["surface_points"] = np.zeros((0, 3))
        track_data["interior_points"] = np.zeros((0, 3))

    return track_data


def visualize_track(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(visible=False):
        return
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_data.mp4", fourcc, 30, (width, height)
    )

    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_visibilities[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_visibilities[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
        track_data = pickle.load(f)

    track_data = process_unique_points(track_data)

    with open(f"{base_path}/{case_name}/final_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    visualize_track(track_data)
