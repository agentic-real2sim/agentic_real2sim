"""Robot-only MuJoCo simulator for pose alignment against segmentation masks.

Scene-agnostic: given a robot trajectory (for first-frame joint config) and
a `CameraSpec`, builds a MuJoCo scene containing only the Franka + Robotiq
2F-85 and the calibrated cameras. No objects, no ground — pose alignment
does not need them.

Shares loading helpers in spirit with `RobotSceneSim` but intentionally
duplicates a small amount of code to keep the abstraction boundary clean.
If the duplication becomes painful, factor into a shared base later.
"""
from pathlib import Path

import h5py
import mujoco
import numpy as np

from ar2s.droid_sim.scene.config import CameraSpec


def _absolutize_mesh_paths(spec: mujoco.MjSpec, asset_dir: Path) -> None:
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str(asset_dir / mesh.file)


class RobotOnlySim:
    GEOM_GROUP_ROBOT_VISUAL    = 2
    GEOM_GROUP_ROBOT_COLLISION = 3

    def __init__(
        self,
        robot_traj_path: str,
        cameras: CameraSpec,
        *,
        robot_base_pos: np.ndarray | None = None,
        robot_base_quat: np.ndarray | None = None,
    ):
        self.cameras_spec = cameras
        self.robot_traj_path = Path(robot_traj_path)

        self.arm_dof = 7
        self.gripper_dof = 1
        self.total_dof = self.arm_dof + self.gripper_dof

        self.robot_base_pos = (
            np.asarray(robot_base_pos, dtype=np.float64) if robot_base_pos is not None
            else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        self.robot_base_quat = (
            np.asarray(robot_base_quat, dtype=np.float64) if robot_base_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )

        self._load_camera_params()
        self._load_initial_joint_config()
        self._build_mjcf_model()

    @property
    def camera_ids(self) -> tuple[int, ...]:
        return self.cameras_spec.camera_ids

    # --------------------------------------------------------------- Cameras
    def _load_camera_params(self):
        ext_path = Path(self.cameras_spec.extrinsics_path)
        if not ext_path.exists():
            raise FileNotFoundError(f"Missing camera extrinsics file: {ext_path}")
        ext_data = np.load(str(ext_path))
        xforms_bot2cam = []
        for cam_id in self.camera_ids:
            key = f"cam_mat_{cam_id}"
            if key not in ext_data:
                raise KeyError(f"Camera extrinsics key '{key}' missing in {ext_path}.")
            xforms_bot2cam.append(np.asarray(ext_data[key], dtype=np.float64).reshape(4, 4))
        xforms_bot2cam = np.stack(xforms_bot2cam, axis=0)

        num_cameras = xforms_bot2cam.shape[0]
        xform_bot2world = np.eye(4, dtype=np.float64)
        self.xforms_cam2world = np.zeros((num_cameras, 4, 4), dtype=np.float64)
        for i in range(num_cameras):
            xform_cam2bot = np.linalg.inv(xforms_bot2cam[i])
            self.xforms_cam2world[i] = xform_bot2world @ xform_cam2bot

        int_path = Path(self.cameras_spec.intrinsics_path)
        if not int_path.exists():
            raise FileNotFoundError(f"Missing camera intrinsics file: {int_path}")

        if int_path.suffix == ".txt":
            lines = [l for l in int_path.read_text().splitlines() if l.strip()]
            values = np.fromstring(lines[0], sep=' ', dtype=np.float64)
            if values.size != 9:
                raise ValueError(f"Expected 9 values in {int_path}, got {values.size}.")
            K = values.reshape(3, 3)
            self.cam_intrinsics = np.stack([K] * num_cameras, axis=0)
            k1 = float(lines[1]) if len(lines) > 1 else 0.0
            self.dist_k1 = np.full(num_cameras, k1, dtype=np.float64)
        else:
            int_data = np.load(str(int_path))
            intrinsics = []
            for cam_id in self.camera_ids:
                key = f"cam_K_{cam_id}"
                if key not in int_data:
                    raise KeyError(f"Camera intrinsics key '{key}' missing in {int_path}.")
                intrinsics.append(np.asarray(int_data[key], dtype=np.float64).reshape(3, 3))
            self.cam_intrinsics = np.stack(intrinsics, axis=0)
            self.dist_k1 = np.zeros(num_cameras, dtype=np.float64)

    # ------------------------------------------------- Initial joint config
    def _load_initial_joint_config(self):
        """Only the first frame — enough to place the robot for mask rendering."""
        if not self.robot_traj_path.exists():
            raise FileNotFoundError(f"Missing robot trajectory file: {self.robot_traj_path}")
        with h5py.File(str(self.robot_traj_path), "r") as f:
            joint_pos = np.asarray(f["joint_position"][0], dtype=np.float32)
            gripper_pos = np.asarray(f["gripper_position"][0], dtype=np.float32)
        if joint_pos.shape != (self.arm_dof,):
            raise ValueError(
                f"joint_position[0] must have shape ({self.arm_dof},), got {joint_pos.shape}.")

        q = np.concatenate([joint_pos, gripper_pos])
        q[6] -= np.pi / 4                       # DROID / MuJoCo convention offset
        gripper_norm = float(np.clip(q[-1], 0.0, 1.0))
        self._initial_q_arm = q[:self.arm_dof]
        self._initial_gripper_norm = gripper_norm

    # ------------------------------------------------------- MJCF building
    def _build_mjcf_model(self):
        workspace_root = Path(__file__).parent.parent.parent.parent.absolute()
        arm_mjcf_path = workspace_root / "assets/robot/franka_emika_panda/panda_nohand.xml"
        gripper_mjcf_path = workspace_root / "assets/robot/robotiq_2f85/2f85.xml"

        spec = mujoco.MjSpec()
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.iterations = 15
        spec.option.tolerance = 1e-7
        spec.option.impratio = 5.0
        spec.compiler.degree = False
        spec.visual.global_.offwidth = 2048
        spec.visual.global_.offheight = 2048

        robot_spec = mujoco.MjSpec.from_file(str(arm_mjcf_path))
        gripper_spec = mujoco.MjSpec.from_file(str(gripper_mjcf_path))
        _absolutize_mesh_paths(robot_spec, workspace_root / "assets/robot/franka_emika_panda/assets")
        _absolutize_mesh_paths(gripper_spec, workspace_root / "assets/robot/robotiq_2f85/assets")

        gripper_frame = robot_spec.body("attachment").add_frame()
        gripper_frame.attach_body(gripper_spec.body("base_mount"), "", "")

        world_body = spec.worldbody

        key_light = world_body.add_light()
        key_light.name = "scene_key_light"
        key_light.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        key_light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        key_light.dir[:] = [0.2, 0.1, -1.0]
        key_light.pos[:] = [0.0, -0.4, 1.5]
        key_light.ambient[:] = [0.25, 0.25, 0.25]
        key_light.diffuse[:] = [0.9, 0.9, 0.9]
        key_light.specular[:] = [0.2, 0.2, 0.2]
        key_light.castshadow = 0

        robot_frame = world_body.add_frame()
        robot_frame.pos[:] = self.robot_base_pos
        robot_frame.quat[:] = self.robot_base_quat
        robot_frame.attach_body(robot_spec.body("link0"), "", "")

        from ar2s.droid_sim._util import (
            opencv_c2w_to_mujoco_camera_pose, fovy_deg_from_K,
        )
        for i, cam_id in enumerate(self.camera_ids):
            xform = self.xforms_cam2world[i]
            cam_pos, cam_quat = opencv_c2w_to_mujoco_camera_pose(xform)
            cam = world_body.add_camera()
            cam.name = f"cam_{cam_id}"
            cam.pos[:] = cam_pos
            cam.quat[:] = cam_quat
            cam.fovy = fovy_deg_from_K(self.cam_intrinsics[i], image_height=720)

        self._spec = spec
        self.model = spec.compile()
        # Pose alignment works kinematically; contacts would waste time.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)

        # Place the robot in its first-frame joint configuration.
        joint_names = ["joint1", "joint2", "joint3", "joint4",
                       "joint5", "joint6", "joint7"]
        for i, name in enumerate(joint_names):
            adr = self.model.joint(name).qposadr.item()
            self.data.qpos[adr] = self._initial_q_arm[i]
        gripper_q = 0.8 * self._initial_gripper_norm
        for jname in ("right_driver_joint", "left_driver_joint"):
            adr = self.model.joint(jname).qposadr.item()
            self.data.qpos[adr] = gripper_q

        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------- Update API
    def set_base_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Update robot base pose in-place (no re-compile)."""
        self.model.body_pos[1] = pos
        self.model.body_quat[1] = quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------- Rendering
    def render_cameras(
        self,
        width: int = 1280,
        height: int = 720,
        visible_groups: tuple[int, ...] | None = None,
    ) -> list[np.ndarray]:
        scene_option = mujoco.MjvOption()
        if visible_groups is not None:
            for g in range(len(scene_option.geomgroup)):
                scene_option.geomgroup[g] = 1 if g in visible_groups else 0

        frames = []
        for i, cam_id in enumerate(self.camera_ids):
            K = self.cam_intrinsics[i]
            fy = K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            S = 2 * int(np.ceil(max(cx, width - cx, cy, height - cy)))
            if S % 2 != 0:
                S += 1

            cam_name = f"cam_{cam_id}"
            cam_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            orig_fovy = float(self.model.cam_fovy[cam_idx])
            self.model.cam_fovy[cam_idx] = float(np.degrees(2.0 * np.arctan(S / (2.0 * fy))))

            renderer = mujoco.Renderer(self.model, height=S, width=S)
            renderer.update_scene(self.data, camera=cam_name, scene_option=scene_option)
            square_img = renderer.render().copy()
            renderer.close()
            self.model.cam_fovy[cam_idx] = orig_fovy

            x0 = int(round(S / 2 - cx))
            y0 = int(round(S / 2 - cy))
            frames.append(square_img[y0:y0 + height, x0:x0 + width].copy())

        return frames

    def apply_distortion(self, images: list[np.ndarray]) -> list[np.ndarray]:
        from scipy.ndimage import map_coordinates

        results = []
        for img, K, k1 in zip(images, self.cam_intrinsics, self.dist_k1):
            if k1 == 0.0:
                results.append(img.copy())
                continue
            h, w = img.shape[:2]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u_d, v_d = np.meshgrid(np.arange(w, dtype=np.float64),
                                   np.arange(h, dtype=np.float64))
            xn = (u_d - cx) / fx
            yn = (v_d - cy) / fy
            xn_u = xn.copy()
            yn_u = yn.copy()
            for _ in range(10):
                r2 = xn_u ** 2 + yn_u ** 2
                scale = 1.0 / (1.0 + k1 * r2)
                xn_u = xn * scale
                yn_u = yn * scale
            map_u = fx * xn_u + cx
            map_v = fy * yn_u + cy
            out = np.empty_like(img)
            for c in range(img.shape[2]):
                out[..., c] = map_coordinates(
                    img[..., c], [map_v, map_u],
                    order=1, mode='constant', cval=0,
                ).astype(np.uint8)
            results.append(out)
        return results
