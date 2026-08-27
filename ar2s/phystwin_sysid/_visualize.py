import os
import open3d as o3d
import numpy as np
import torch
import time
import cv2
from ._logger import logger
from .config import PhysTwinConfig
import pyrender
import trimesh


def _render_video_egl(
    object_points,
    object_colors,
    controller_points,
    object_visibilities,
    cfg: PhysTwinConfig,
    vis_cam_idx,
    save_path,
    fps,
    width,
    height,
    intrinsic,
    w2c,
):
    """Render ``object_points`` to ``save_path`` via pyrender's EGL backend.

    Fallback for machines where Open3D's legacy Visualizer can't get a GL
    context (no X11/Wayland and no OSMesa). EGL talks to the GPU driver
    directly, so this works on headless machines with an NVIDIA GPU.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

    # w2c is OpenCV convention (X right, Y down, Z forward into the scene);
    # pyrender/OpenGL cameras look down -Z with Y up, so flip Y and Z of the
    # camera-local axes when going from c2w_cv to c2w_gl.
    c2w_cv = np.linalg.inv(w2c)
    c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])

    cam_pos = c2w_cv[:3, 3]
    dists = np.linalg.norm(object_points.reshape(-1, 3) - cam_pos, axis=-1)
    znear = max(0.01, float(dists.min()) * 0.5)
    zfar = float(dists.max()) * 1.5 + 0.1

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)

    sphere_mesh = trimesh.creation.icosphere(radius=0.01)
    sphere_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.0, 0.0, 1.0], metallicFactor=0.0, roughnessFactor=1.0,
    )
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT

    renderer = pyrender.OffscreenRenderer(width, height, point_size=4.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    try:
        for i in range(object_points.shape[0]):
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])

            if object_visibilities is None:
                points_i = object_points[i]
                colors_i = object_colors[i]
            else:
                visible = np.where(object_visibilities[i])[0]
                points_i = object_points[i, visible, :]
                colors_i = object_colors[i, visible, :]
            scene.add(pyrender.Mesh.from_points(
                points_i.astype(np.float32), colors=colors_i.astype(np.float32)
            ))

            if controller_points is not None:
                poses = np.tile(np.eye(4), (controller_points.shape[1], 1, 1))
                poses[:, :3, 3] = controller_points[i]
                scene.add(pyrender.Mesh.from_trimesh(
                    sphere_mesh, material=sphere_material, poses=poses,
                ))

            scene.add(camera, pose=c2w_gl)

            rgba, _ = renderer.render(scene, flags=flags)
            frame = rgba[:, :, :3].copy()
            if cfg.overlay_path is not None:
                background = rgba[:, :, 3] == 0
                image_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
                overlay = cv2.imread(image_path)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                frame[background] = overlay[background]

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
    finally:
        video_writer.release()
        renderer.delete()


def visualize_pc(
    object_points,
    cfg: PhysTwinConfig,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    visualize=True,
    save_video=False,
    save_path=None,
    vis_cam_idx=0,
):
    FPS = cfg.fps
    width, height = cfg.WH
    intrinsic = cfg.intrinsics[vis_cam_idx]
    w2c = cfg.w2cs[vis_cam_idx]

    # Convert the stuffs to numpy if it's tensor
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors is None:
        object_colors = np.tile(
            [1, 0, 0], (object_points.shape[0], object_points.shape[1], 1)
        )
    else:
        if object_colors.shape[1] < object_points.shape[1]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            object_colors.shape[0],
                            object_points.shape[1] - object_colors.shape[1],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=1,
            )

    # The pcs is a 4d pcd numpy array with shape (n_frames, n_points, 3)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visualize, width=width, height=height)

    if vis.get_view_control() is None:
        # open3d's legacy Visualizer needs a GLFW-backed GL context even for
        # visible=False; on headless machines without OSMesa, window
        # creation silently fails and there is nothing to render.
        vis.destroy_window()
        if save_video:
            logger.warning(
                "[VIS]: no GL context available for Open3D Visualizer (headless); "
                "falling back to pyrender EGL rendering"
            )
            _render_video_egl(
                object_points, object_colors, controller_points, object_visibilities,
                cfg, vis_cam_idx, save_path, FPS, width, height, intrinsic, w2c,
            )
        else:
            logger.warning("[VIS]: no GL context available (headless); skipping visualize_pc")
        return

    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")

    # Initialize video writer if save_video is True
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for .mp4 (avc1 = h264 sometimes maps to h264_v4l2m2m hardware encoder, fails on headless)
        video_writer = cv2.VideoWriter(save_path, fourcc, FPS, (width, height))

    if controller_points is not None:
        controller_meshes = []
        prev_center = []
    for i in range(object_points.shape[0]):
        object_pcd = o3d.geometry.PointCloud()
        if object_visibilities is None:
            object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        else:
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            object_pcd.colors = o3d.utility.Vector3dVector(
                object_colors[i, np.where(object_visibilities[i])[0], :]
            )
        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            if controller_points is not None:
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            if controller_points is not None:
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
        vis.poll_events()
        vis.update_renderer()

        # Capture frame and write to video file if save_video is True
        if save_video:
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            if cfg.overlay_path is not None:
                # Get the mask where the pixel is white
                mask = np.all(frame == [255, 255, 255], axis=-1)
                image_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
                overlay = cv2.imread(image_path)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                frame[mask] = overlay[mask]
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        if visualize:
            time.sleep(1 / FPS)

    vis.destroy_window()
    if save_video:
        video_writer.release()
