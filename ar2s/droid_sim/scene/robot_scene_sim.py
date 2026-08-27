"""Generic robot + scene MuJoCo simulator, driven by a SceneConfig.

Replaces the single-scene MarkerInMugSim. The robot stack (Franka + Robotiq
2F-85) is fixed; objects, ground and cameras come from the SceneConfig.

See docs/agent_pipeline/architecture.md §3.

Geom group assignment (objects vs old MarkerInMugSim are reversed for
episode_000004: pen=4, mug=5 instead of pen=5, mug=4). Group 3 contains
physics-only collision geometry and is hidden from rendered outputs.
"""
from pathlib import Path

import h5py
import mujoco
import numpy as np
from mujoco import sysid

from ar2s.droid_sim import pose_store
from ar2s.droid_sim.scene.config import SceneConfig
from ar2s.droid_sim.scene.ground import derive_ground_pose_world_xy
from ar2s.droid_sim.scene.mesh_utils import get_effective_scale
from ar2s.droid_sim.scene.robot_profile import FRANKA_PANDA, RobotProfile


def _absolutize_mesh_paths(spec: mujoco.MjSpec, asset_dir: Path) -> None:
    """Rewrite each mesh.file in `spec` to an absolute path under `asset_dir`."""
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str(asset_dir / mesh.file)


class RobotSceneSim:
    # NOTE: group 0 is MuJoCo's *default-visible* group (a bare MjvOption shows
    # 0..2), so any renderer that does not set geomgroup explicitly will draw
    # the coacd hulls and hide nothing. Every in-repo render path sets it; new
    # ones must too. See skills/probe_grasp.py `_USD_HIDDEN_GROUPS`.
    GEOM_GROUP_OBJECT_COLLISION = 0  # per-object collision sub-meshes (coacd output)
    GEOM_GROUP_GROUND          = 1
    GEOM_GROUP_ROBOT_VISUAL    = 2  # set in panda.xml default class
    GEOM_GROUP_ROBOT_COLLISION = 3  # set in panda.xml default class
    GEOM_GROUP_OBJECT_FIRST    = 4  # objects[i] visual mesh => min(4 + i, 5); MuJoCo only has 6 groups

    GRIPPER_Q_MAX = 0.78  # Robotiq 2F-85 driver joint cap (< 0.8 = fully closed)

    def __init__(
        self,
        scene_config: SceneConfig,
        *,
        is_kinematic: bool = True,
        robot_base_pos: np.ndarray | None = None,
        robot_base_quat: np.ndarray | None = None,
        exclude_robot_nonpickup_collision: tuple[str, ...] = (),
        robot_profile: RobotProfile = FRANKA_PANDA,
    ):
        """If `exclude_robot_nonpickup_collision` is non-empty, it is the
        tuple `(pickup_name,)` — robot ↔ all SceneObjects whose name is NOT
        `pickup_name` are filtered out by the unified collision policy.
        Object↔object and object↔ground collisions remain enabled. The policy
        also applies `profile.exclude_arm_ground_collision` in the same pass,
        so these options cannot overwrite each other's contype/conaffinity
        masks.
        """
        self.config = scene_config
        self.is_kinematic = is_kinematic
        self._exclude_robot_nonpickup = tuple(exclude_robot_nonpickup_collision)
        self.profile = robot_profile

        # --- Timing ---
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_frame = 0
        self.steps_per_frame = 10
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / self.steps_per_frame

        self.droid_control_freq = 15
        assert self.fps % self.droid_control_freq == 0, \
            "FPS must be a multiple of DROID's control frequency"

        # --- Robot DOF (from the profile) ---
        self.arm_dof = self.profile.arm_dof
        self.gripper_dof = 1
        self.total_dof = self.arm_dof + self.gripper_dof

        # --- Robot base pose ---
        self.robot_base_pos = (
            np.asarray(robot_base_pos, dtype=np.float64) if robot_base_pos is not None
            else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        self.robot_base_quat = (
            np.asarray(robot_base_quat, dtype=np.float64) if robot_base_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )

        # --- Per-object buffers ---
        self._obj_pos_traj: dict[str, list[np.ndarray]] = {}
        self._obj_quat_traj: dict[str, list[np.ndarray]] = {}
        self._obj_scales: dict[str, np.ndarray] = {}
        self.object_geom_groups: dict[str, int] = {}

        # --- Sensor recording ---
        self._sensor_traj: list[np.ndarray] = []

        # --- Reused object-camera renderer (lazy; see render_object_camera) ---
        self._obj_cam_renderer: dict | None = None

        # --- Build sequence ---
        self._load_camera_params()
        self._load_robot_trajectory()
        self._load_object_meshes()
        self._load_object_trajectories()
        self._build_mjcf_model()

    # ----------------------------------------------- convenience properties
    @property
    def camera_ids(self) -> tuple[int, ...]:
        return self.config.cameras.camera_ids

    @property
    def object_camera_id(self) -> int:
        return self.config.cameras.object_camera_id

    # -------------------------------------------------------------- Cameras
    def _load_camera_params(self):
        # --- Extrinsics (NPZ): cam_mat_{cam_id} → (4, 4) bot-to-cam ---
        ext_path = Path(self.config.cameras.extrinsics_path)
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

        xform_bot2world = np.eye(4, dtype=np.float64)
        num_cameras = xforms_bot2cam.shape[0]
        self.xforms_cam2world = np.zeros((num_cameras, 4, 4), dtype=np.float64)
        for i in range(num_cameras):
            xform_cam2bot = np.linalg.inv(xforms_bot2cam[i])
            self.xforms_cam2world[i] = xform_bot2world @ xform_cam2bot

        # --- Intrinsics ---
        int_path = Path(self.config.cameras.intrinsics_path)
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

    # --------------------------------------------------- Robot trajectory
    def _load_robot_trajectory(self):
        acts_path = Path(self.config.robot_traj_path)
        if not acts_path.exists():
            raise FileNotFoundError(f"Missing robot trajectory file: {acts_path}")
        with h5py.File(str(acts_path), "r") as f:
            joint_pos = np.asarray(f["joint_position"][:], dtype=np.float32)
            gripper_pos = np.asarray(f["gripper_position"][:], dtype=np.float32)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != self.arm_dof:
            raise ValueError(
                f"joint_position must have shape (N, {self.arm_dof}), got {joint_pos.shape}.")
        if gripper_pos.ndim != 2 or gripper_pos.shape[1] != self.gripper_dof:
            raise ValueError(
                f"gripper_position must have shape (N, {self.gripper_dof}), got {gripper_pos.shape}.")

        q_robot = np.concatenate([joint_pos, gripper_pos], axis=1)
        # Per-arm-joint zero-position offset (Franka joint7: DROID/libfranka vs
        # MuJoCo panda_nohand convention; AIRBOT: all zero).
        q_robot[:, : self.arm_dof] -= np.asarray(self.profile.arm_zero_offsets)
        gripper_norm = np.clip(q_robot[:, -1], 0.0, 1.0)
        kernel = np.ones(5) / 5
        gripper_norm = np.convolve(gripper_norm, kernel, mode='same')
        gripper_norm = np.clip(gripper_norm, 0.0, 1.0)
        self.gripper_norm = gripper_norm
        # Store the gripper column in actuator-ctrl units (Robotiq 0-255 tendon;
        # AIRBOT g2_joint metres). Kinematic mode re-derives norm = col/ctrl_scale.
        q_robot[:, -1] = self.profile.gripper_ctrl_scale * gripper_norm

        self.num_control_frames = q_robot.shape[0]
        self.num_frames = int(self.fps * (self.num_control_frames / self.droid_control_freq))
        self.num_steps = self.num_frames * self.steps_per_frame
        self._robot_q_traj = q_robot
        self.q_target = self._interpolate_q_target(q_robot=q_robot)

    # --------------------------------------------------- Object meshes
    def _load_object_meshes(self):
        for obj in self.config.objects:
            visual_path = Path(obj.visual_mesh_path)
            if not visual_path.exists():
                raise FileNotFoundError(f"[{obj.name}] missing visual mesh: {visual_path}")
            for col_path_str in obj.collision_mesh_paths:
                if not Path(col_path_str).exists():
                    raise FileNotFoundError(
                        f"[{obj.name}] missing collision mesh: {col_path_str}")
            scale_path = Path(obj.scale_path)
            if not scale_path.exists():
                raise FileNotFoundError(f"[{obj.name}] missing scale file: {scale_path}")
            self._obj_scales[obj.name] = get_effective_scale(obj)

    # --------------------------------------------------- Object trajectories
    def _load_object_trajectories(self):
        from ar2s.droid_sim.scene.shift import (
            shift_to_world_disp_obj,
        )

        N = self.num_control_frames
        obj_cam_idx = list(self.camera_ids).index(self.object_camera_id)
        cam2world = self.xforms_cam2world[obj_cam_idx]

        # Pass 1: load all trajectories with per-object pos_offset / quat_offset
        # but WITHOUT phase5_shift (need ref's first-frame quat to compute the
        # ground-projected basis, which we get below).
        for obj in self.config.objects:
            pose_dir = Path(obj.pose_source.path)
            poses_cam = self._load_pose_sequence(
                pose_dir, N, obj.name,
                init_frame=obj.pose_source.init_frame,
            )  # (N, 4, 4)

            pos_traj: list[np.ndarray] = []
            quat_traj: list[np.ndarray] = []
            pos_offset = np.asarray(obj.pos_offset, dtype=np.float64)
            quat_offset = np.asarray(obj.quat_offset, dtype=np.float64)
            for i in range(N):
                pose_world = cam2world @ poses_cam[i]
                quat_w = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat_w, pose_world[:3, :3].flatten())
                pos = pose_world[:3, 3] + pos_offset
                quat_out = np.empty(4, dtype=np.float64)
                mujoco.mju_mulQuat(quat_out, quat_offset, quat_w)
                pos_traj.append(pos)
                quat_traj.append(quat_out)

            self._obj_pos_traj[obj.name] = pos_traj
            self._obj_quat_traj[obj.name] = quat_traj

        # Pass 2: compute world-frame Phase-5 displacement once and add it to
        # every object's pos_traj. Stage 0 writes the GROUND sentinel; this
        # simulator no longer supports object-referenced ground planes.
        ref_name = self.config.ground.reference_object
        assert ref_name == "GROUND", (
            "RobotSceneSim requires ground.reference_object == 'GROUND'; "
            f"got {ref_name!r}. Run geometry_prior before scene_prep."
        )
        n = -np.asarray(self.config.ground.local_down_axis, dtype=np.float64)
        n /= max(np.linalg.norm(n), 1e-12)
        self._world_disp_obj = shift_to_world_disp_obj(
            self.config.phase5_shift, n,
        )
        for obj in self.config.objects:
            for i in range(len(self._obj_pos_traj[obj.name])):
                self._obj_pos_traj[obj.name][i] = (
                    self._obj_pos_traj[obj.name][i] + self._world_disp_obj
                )

    @staticmethod
    def _load_pose_sequence(
        pose_dir: Path, num_frames: int, object_name: str,
        init_frame: int = 0,
    ) -> np.ndarray:
        """Load FoundationPose per-frame transforms.

        Frames 0..init_frame-1 are filled with the init_frame pose (the
        object is treated as static during the pre-tracking window).
        Subsequent frames load from disk or fall back to the previous frame
        if missing (FoundationPose occasionally drops a frame).
        """
        if not pose_store.exists(pose_dir):
            raise FileNotFoundError(f"Missing pose directory for {object_name}: {pose_dir}")

        by_frame = pose_store.load_poses(pose_dir)
        init_pose = by_frame.get(int(init_frame))
        if init_pose is None:
            raise FileNotFoundError(
                f"Missing init-frame pose for {object_name} at init_frame="
                f"{init_frame}: {pose_dir}"
            )

        poses = np.zeros((num_frames, 4, 4), dtype=np.float64)
        for i in range(num_frames):
            if i < init_frame:
                poses[i] = init_pose                           # pre-init: hold static
                continue
            value = by_frame.get(i)
            if value is not None:
                poses[i] = value
            elif i > init_frame:
                poses[i] = poses[i - 1]                        # FP dropped a frame
            else:                                              # i == init_frame
                poses[i] = init_pose                           # unreachable in practice
        return poses

    # ------------------------------------------------------- MJCF building
    def _required_render_side(self) -> int:
        """Largest square side any real-camera render will ask for.

        Mirrors the S computation in render_cameras() over every camera at its
        native size, so the offscreen buffer is sized from the episode's actual
        sensors instead of a constant.
        """
        size = getattr(self.config.cameras, "image_size", None) or (1280, 720)
        width, height = int(size[0]), int(size[1])
        sides = []
        for K in self.cam_intrinsics:
            cx, cy = float(K[0, 2]), float(K[1, 2])
            S = 2 * int(np.ceil(max(cx, width - cx, cy, height - cy)))
            sides.append(S + (S % 2))
        return max(sides) if sides else 0

    def _build_mjcf_model(self):
        workspace_root = Path(__file__).parent.parent.parent.parent.absolute()

        # Empty top-level spec — replaces the near-empty base XML.
        spec = mujoco.MjSpec()
        spec.option.timestep = self.sim_dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.iterations = 15
        spec.option.impratio = 5.0
        spec.option.tolerance = 1e-7
        spec.compiler.degree = False  # radians

        # All mesh references are absolute paths, so meshdir is irrelevant.
        # Offscreen framebuffer. render_cameras/render_object_camera handle an
        # off-centre principal point by rendering a square of side
        # S = 2*ceil(max(cx, W-cx, cy, H-cy)) and cropping the requested window
        # out of it, so the buffer must hold S, not W. DROID's 1280x720 needs
        # S~1280; the AIRBOT ZED 2i (2208x1242, cx~1130) needs S~2260, which is
        # why a flat 2048 used to make every real-camera render fail with
        # "Image width 2260 > framebuffer width 2048".
        square = self._required_render_side()
        spec.visual.global_.offwidth = max(2048, square)
        spec.visual.global_.offheight = max(2048, square)

        p = self.profile
        if p.is_combined:
            # Single combined arm+gripper MJCF (AIRBOT): the gripper is already
            # part of the tree, so there is no runtime attach.
            robot_spec = mujoco.MjSpec.from_file(str(workspace_root / p.combined_mjcf))
            _absolutize_mesh_paths(robot_spec, workspace_root / p.arm_mesh_dir)
        else:
            # Arm + gripper loaded separately, gripper mounted on the arm's
            # attach body (Franka + Robotiq).
            robot_spec = mujoco.MjSpec.from_file(str(workspace_root / p.arm_mjcf))
            gripper_spec = mujoco.MjSpec.from_file(str(workspace_root / p.gripper_mjcf))
            _absolutize_mesh_paths(robot_spec, workspace_root / p.arm_mesh_dir)
            _absolutize_mesh_paths(gripper_spec, workspace_root / p.gripper_mesh_dir)
            gripper_frame = robot_spec.body(p.arm_attach_body).add_frame()
            gripper_frame.attach_body(gripper_spec.body(p.gripper_attach_body), "", "")

        world_body = spec.worldbody

        # --- Light ---
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

        # --- Robot (link0 attached at robot_frame) ---
        robot_frame = world_body.add_frame()
        robot_frame.pos[:] = self.robot_base_pos
        robot_frame.quat[:] = self.robot_base_quat
        if p.is_combined:
            # Single combined MJCF: the URDF's fixed root (base_link) is merged
            # into worldbody, so there is no single root body to attach. Compose
            # the whole robot spec at the frame; prefix="" keeps joint/body names
            # (joint1.., g2_joint, link6, ...) so profile lookups still resolve.
            spec.attach(robot_spec, prefix="", frame=robot_frame)
        else:
            robot_frame.attach_body(robot_spec.body(p.arm_root_body), "", "")

        # --- Objects (one body per SceneObject) ---
        object_visual_geoms: dict[str, mujoco.MjsGeom] = {}
        object_collision_geoms: dict[str, list[mujoco.MjsGeom]] = {}

        for i, obj in enumerate(self.config.objects):
            scale = self._obj_scales[obj.name]
            group = min(self.GEOM_GROUP_OBJECT_FIRST + i, 5)
            self.object_geom_groups[obj.name] = group

            # Visual mesh asset (absolute path)
            visual_mesh = spec.add_mesh()
            visual_mesh.name = f"{obj.name}_visual"
            visual_mesh.file = str(Path(obj.visual_mesh_path).resolve())
            visual_mesh.scale[:] = scale

            # Collision mesh assets (absolute paths)
            collision_mesh_names: list[str] = []
            for c_idx, col_path in enumerate(obj.collision_mesh_paths):
                m = spec.add_mesh()
                m.name = f"{obj.name}_col_{c_idx}"
                m.file = str(Path(col_path).resolve())
                m.scale[:] = scale
                collision_mesh_names.append(m.name)

            # Body + free joint
            body = world_body.add_body()
            body.name = obj.name
            body.pos[:] = self._obj_pos_traj[obj.name][0]
            body.quat[:] = self._obj_quat_traj[obj.name][0]
            # NOTE: body.mass is NOT set here. Mesh geoms at the default density
            # would otherwise sum to a mass that OVERRIDES this and blows up the
            # object mass (a CoACD plate/spoon summed to ~5x its manifest mass).
            # Instead we assign the geom masses explicitly below so the body's
            # total == obj.mass exactly, with per-mesh inertia from the shapes.
            if not obj.static:
                fj = body.add_freejoint()
                fj.name = f"{obj.name}_freejoint"

            col_geoms: list[mujoco.MjsGeom] = []
            if collision_mesh_names:
                # Separate visual (no collision) + per-piece collision geoms
                vis_geom = body.add_geom()
                vis_geom.type = mujoco.mjtGeom.mjGEOM_MESH
                vis_geom.name = f"{obj.name}_visual_geom"
                vis_geom.meshname = f"{obj.name}_visual"
                vis_geom.contype = 0
                vis_geom.conaffinity = 0
                vis_geom.group = group
                vis_geom.mass = 0.0   # visual-only; mass lives on the collision pieces
                if obj.rgba is not None:
                    vis_geom.rgba[:] = obj.rgba
                object_visual_geoms[obj.name] = vis_geom

                _piece_mass = obj.mass / max(len(collision_mesh_names), 1)
                for c_idx, cname in enumerate(collision_mesh_names):
                    g = body.add_geom()
                    g.type = mujoco.mjtGeom.mjGEOM_MESH
                    g.name = f"{obj.name}_collision_geom_{c_idx}"
                    g.meshname = cname
                    g.friction[:] = obj.friction
                    # Split obj.mass across the CoACD pieces so the body total
                    # equals the manifest mass (see note at body creation).
                    g.mass = _piece_mass
                    # condim 6 enables torsional + rolling friction; the
                    # MuJoCo default (3) silently ignores friction[1:].
                    g.condim = 6
                    # Collision sub-meshes go to a dedicated group so USD
                    # export can hide them while keeping the visual mesh
                    # (which stays at `group` = OBJECT_FIRST + i).
                    g.group = self.GEOM_GROUP_OBJECT_COLLISION
                    col_geoms.append(g)
            else:
                # Single geom does both visual and collision (matches old single-mesh objects).
                g = body.add_geom()
                g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.name = f"{obj.name}_visual_geom"
                g.meshname = f"{obj.name}_visual"
                g.friction[:] = obj.friction
                g.mass = obj.mass
                g.condim = 6
                g.group = group
                col_geoms.append(g)
                object_visual_geoms[obj.name] = g
            object_collision_geoms[obj.name] = col_geoms

        # --- Ground: world XY plane, z driven by ground_spec.offset ---
        # geometry_prior Stage 0 fills ``manifest.ground.offset`` from
        # z_anchor's BFS propagation, so the plane sits at the bottom of
        # the deepest support surface by construction. See
        # ``ar2s.droid_sim.scene.ground.derive_ground_pose_world_xy``.
        ground_spec = self.config.ground
        ground_pos, quat_plane = derive_ground_pose_world_xy(
            ground_spec, self.config.phase5_shift[2],
        )

        ground_geom = world_body.add_geom()
        ground_geom.name = "ground"
        ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground_geom.size[:] = ground_spec.plane_size
        ground_geom.pos[:] = ground_pos
        ground_geom.quat[:] = quat_plane
        ground_geom.rgba[:] = [0.9, 0.9, 0.9, 1.0]
        ground_geom.group = self.GEOM_GROUP_GROUND

        # --- Cameras (calibrated extrinsics + intrinsics → MuJoCo cameras) ---
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

        # End-effector site (TCP). z=0.160 puts it slightly past Robotiq's
        # `pinch` site (at base_mount z=0.149), inside the pad envelope —
        # the closer the TCP is to the actual grasp contact zone, the more
        # useful the `ee - contact` corrective vector is for DET shifts and
        # the more meaningful the closest_distance / dist_at_peak_lift signals.
        gripper_base = spec.body(self.profile.ee_site_body)
        ee_site = gripper_base.add_site()
        ee_site.name = "ee_frame"
        ee_site.pos[:] = list(self.profile.ee_site_pos)
        ee_site.size[0] = 0.0001

        # Robot base coordinate frame site
        base_site = world_body.add_site()
        base_site.name = "robot_base_frame"
        base_site.pos[:] = self.robot_base_pos.astype(np.float64)
        base_site.size[0] = 0.0001

        self.model = spec.compile()
        if self.is_kinematic:
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        if self._exclude_robot_nonpickup or self.profile.exclude_arm_ground_collision:
            self._apply_collision_policy()
        self._apply_gripper_contact_params()
        self.data = mujoco.MjData(self.model)

        # Initial robot joint positions (arm joints from the profile).
        self.joint_id_map = []
        for i, name in enumerate(self.profile.arm_joint_names):
            joint_id = self.model.joint(name).qposadr.item()
            self.joint_id_map.append(joint_id)
            self.data.qpos[joint_id] = self.q_target[0][i]
            self.data.ctrl[i] = self.q_target[0][i]
        self._seed_gripper_qpos(self.gripper_norm[0])
        self.data.ctrl[self.profile.gripper_ctrl_index] = self.q_target[0][self.arm_dof]

        mujoco.mj_forward(self.model, self.data)
        self.initial_state = sysid.create_initial_state(
            self.model, self.data.qpos, self.data.qvel, self.data.act
        )

    # --------------------------------------------------------- Sim loop
    def _seed_gripper_qpos(self, norm: float) -> None:
        """Write each gripper joint's qpos directly (kinematic mode does not
        solve the mimic equality, so every finger joint is seeded explicitly)."""
        norm = float(np.clip(norm, 0.0, 1.0))
        for jname, scale in self.profile.gripper_seed_joints:
            adr = self.model.joint(jname).qposadr.item()
            self.data.qpos[adr] = scale * norm

    def set_joint_targets(self, frame_idx: int) -> None:
        assert frame_idx >= 0
        if frame_idx >= self.q_target.shape[0]:
            frame_idx = self.q_target.shape[0] - 1
        if self.data.ctrl.size >= self.total_dof:
            self.data.ctrl[: self.total_dof] = self.q_target[frame_idx]

    def sim_one_frame(self):
        frames_per_control = self.fps // self.droid_control_freq
        control_idx = min(self.sim_frame // frames_per_control,
                          self.num_control_frames - 1)

        if self.is_kinematic:
            for i, joint_id in enumerate(self.joint_id_map):
                self.data.qpos[joint_id] = self.q_target[self.sim_frame][i]
            norm = self.q_target[self.sim_frame][self.arm_dof] / self.profile.gripper_ctrl_scale
            self._seed_gripper_qpos(norm)

            for obj in self.config.objects:
                if obj.static:
                    continue
                adr = self.model.joint(f"{obj.name}_freejoint").qposadr.item()
                self.data.qpos[adr:adr + 3] = self._obj_pos_traj[obj.name][control_idx]
                self.data.qpos[adr + 3:adr + 7] = self._obj_quat_traj[obj.name][control_idx]

            for _ in range(self.steps_per_frame):
                mujoco.mj_forward(self.model, self.data)
                self._sensor_traj.append(self.data.sensordata.copy())
        else:
            self.set_joint_targets(frame_idx=self.sim_frame)
            gi = self.profile.gripper_ctrl_index
            ctrl_max = self.profile.gripper_close_cap_frac * self.profile.gripper_ctrl_scale
            self.data.ctrl[gi] = min(self.data.ctrl[gi], ctrl_max)
            for _ in range(self.steps_per_frame):
                mujoco.mj_step(self.model, self.data)
                self._sensor_traj.append(self.data.sensordata.copy())

        self.sim_time += self.frame_dt
        self.sim_frame += 1

    # --------------------------------------------------------- TimeSeries
    def get_control_ts(self) -> sysid.TimeSeries:
        t = np.arange(self.num_steps) * self.sim_dt
        q_expanded = np.repeat(self.q_target, self.steps_per_frame, axis=0)
        return sysid.TimeSeries(t, q_expanded)

    def get_sensor_ts(self) -> sysid.TimeSeries:
        times = np.arange(1, self.num_steps) * self.sim_dt
        sensor_data = np.stack(self._sensor_traj[:-1])
        return sysid.TimeSeries.from_names(times, sensor_data, self.model)

    def save_traj(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        control_ts = self.get_control_ts()
        sensor_ts = self.get_sensor_ts()
        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("initial_state", data=self.initial_state)
            g = f.create_group("control_ts")
            g.create_dataset("times", data=control_ts.times)
            g.create_dataset("data", data=control_ts.data)
            g = f.create_group("sensor_ts")
            g.create_dataset("times", data=sensor_ts.times)
            g.create_dataset("data", data=sensor_ts.data)
        print(f"Saved trajectory to {out_path}")

    def _interpolate_q_target(self, q_robot: np.ndarray) -> np.ndarray:
        q_robot = np.asarray(q_robot, dtype=np.float32)
        assert q_robot.ndim == 2, f"q_robot must be 2D, got shape {q_robot.shape}"
        frames_per_control = self.fps // self.droid_control_freq
        q_target = np.zeros((self.num_frames, self.total_dof), dtype=np.float32)
        if q_robot.shape[0] == 0:
            return q_target
        if q_robot.shape[0] == 1:
            q_target[:] = q_robot[0]
            return q_target
        alpha = (np.arange(frames_per_control, dtype=q_robot.dtype) / frames_per_control)[:, None]
        for i in range(q_robot.shape[0] - 1):
            start = q_robot[i]
            end = q_robot[i + 1]
            q_target[i * frames_per_control:(i + 1) * frames_per_control] = (
                start + alpha * (end - start))
        q_target[(q_robot.shape[0] - 1) * frames_per_control:] = q_robot[-1]
        return q_target

    # --------------------------------------------------------- Rendering
    def render_cameras(
        self,
        width: int = 1280,
        height: int = 720,
        visible_groups: tuple[int, ...] | None = None,
        show_axes: bool = False,
    ) -> list[np.ndarray]:
        scene_option = mujoco.MjvOption()
        if visible_groups is not None:
            for g in range(len(scene_option.geomgroup)):
                scene_option.geomgroup[g] = 1 if g in visible_groups else 0
        else:
            # MuJoCo's default MjvOption hides groups 3-5, which silently
            # drops object[0]/object[1+] (groups 4-5) from the rendered
            # frames. Match render_free_camera() and show everything except
            # the collision groups (robot collision = 3, object collision
            # sub-meshes = 0; both are physics-only geoms with no rgba styling).
            for g in range(len(scene_option.geomgroup)):
                scene_option.geomgroup[g] = 0 if g in (
                    self.GEOM_GROUP_ROBOT_COLLISION,
                    self.GEOM_GROUP_OBJECT_COLLISION,
                ) else 1
        if show_axes:
            scene_option.frame = mujoco.mjtFrame.mjFRAME_WORLD

        frames = []
        for i, cam_id in enumerate(self.camera_ids):
            K = self.cam_intrinsics[i]
            fy = K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            S = 2 * int(np.ceil(max(cx, width - cx, cy, height - cy)))
            if S % 2 != 0:
                S += 1

            cam_name = f"cam_{cam_id}"
            cam_idx = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            orig_fovy = float(self.model.cam_fovy[cam_idx])
            self.model.cam_fovy[cam_idx] = float(
                np.degrees(2.0 * np.arctan(S / (2.0 * fy))))

            renderer = mujoco.Renderer(self.model, height=S, width=S)
            renderer.update_scene(self.data, camera=cam_name, scene_option=scene_option)
            square_img = renderer.render().copy()
            renderer.close()
            self.model.cam_fovy[cam_idx] = orig_fovy

            x0 = int(round(S / 2 - cx))
            y0 = int(round(S / 2 - cy))
            frames.append(square_img[y0:y0 + height, x0:x0 + width].copy())

        return frames

    def render_object_camera(
        self,
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        """Render ONLY the object camera, reusing a cached ``mujoco.Renderer``.

        Produces a frame byte-identical to ``render_cameras(width, height)``
        at the object-camera index, but builds the GL context + uploads the
        scene to the GPU once (on first call) and reuses it across calls.
        ``render_cameras`` instead constructs and destroys a ``Renderer`` for
        every camera on every call — fine for the few-shot agent renders, but
        catastrophic for the grasp sweep's per-frame video capture (hundreds
        of context create/destroy cycles per probe, serialized across workers
        on a single GPU's EGL driver).

        Call ``close_object_camera_renderer()`` when done to free the context.
        """
        obj_cam_idx = list(self.camera_ids).index(self.object_camera_id)
        K = self.cam_intrinsics[obj_cam_idx]
        fy = K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        S = 2 * int(np.ceil(max(cx, width - cx, cy, height - cy)))
        if S % 2 != 0:
            S += 1

        cam_name = f"cam_{self.object_camera_id}"
        cam_idx = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = float(np.degrees(2.0 * np.arctan(S / (2.0 * fy))))

        cache = self._obj_cam_renderer
        if cache is None or cache["S"] != S or cache["wh"] != (width, height):
            if cache is not None:
                cache["renderer"].close()
            # Match render_cameras: show everything except collision groups
            # (robot collision + object collision sub-meshes).
            scene_option = mujoco.MjvOption()
            for g in range(len(scene_option.geomgroup)):
                scene_option.geomgroup[g] = 0 if g in (
                    self.GEOM_GROUP_ROBOT_COLLISION,
                    self.GEOM_GROUP_OBJECT_COLLISION,
                ) else 1
            self._obj_cam_renderer = cache = {
                "renderer": mujoco.Renderer(self.model, height=S, width=S),
                "S": S, "wh": (width, height), "scene_option": scene_option,
                "cam_idx": cam_idx, "fovy": fovy, "cx": cx, "cy": cy,
            }

        renderer = cache["renderer"]
        # Temporarily override fovy exactly as render_cameras does, so the
        # rendered+cropped pixels match it bit-for-bit; restore afterwards.
        orig_fovy = float(self.model.cam_fovy[cam_idx])
        self.model.cam_fovy[cam_idx] = cache["fovy"]
        renderer.update_scene(
            self.data, camera=cam_name, scene_option=cache["scene_option"])
        square_img = renderer.render()
        self.model.cam_fovy[cam_idx] = orig_fovy

        x0 = int(round(S / 2 - cache["cx"]))
        y0 = int(round(S / 2 - cache["cy"]))
        return square_img[y0:y0 + height, x0:x0 + width].copy()

    def close_object_camera_renderer(self) -> None:
        """Free the cached object-camera Renderer (GL context), if any."""
        if self._obj_cam_renderer is not None:
            self._obj_cam_renderer["renderer"].close()
            self._obj_cam_renderer = None

    def render_free_camera(
        self,
        pos: np.ndarray,
        lookat: np.ndarray,
        fovy: float = 45.0,
        width: int = 1280,
        height: int = 720,
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        show_axes: bool = False,
    ) -> np.ndarray:
        """Render from an arbitrary camera at world `pos` looking at world `lookat`.

        MuJoCo's `MjvCamera` (FREE type) parameterises the view by the
        camera's *forward* (look-direction) angles, NOT by the angles of
        the camera's position relative to `lookat`. An earlier version of
        this method used `d = pos - lookat` and produced images that
        appeared mirrored / inverted because the resulting azimuth and
        elevation pointed AWAY from the lookat instead of toward it.

        Correct conversion: take `d = lookat - pos` (the look direction
        in world coords) and read its azimuth + elevation.
        """
        pos = np.asarray(pos, dtype=np.float64)
        lookat = np.asarray(lookat, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = float(np.linalg.norm(pos - lookat))
        # Azimuth + elevation of the LOOK DIRECTION (camera → lookat).
        d = lookat - pos
        cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
        cam.elevation = float(np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2]))))
        scene_option = mujoco.MjvOption()
        for g in range(len(scene_option.geomgroup)):
            scene_option.geomgroup[g] = 0 if g in (
                self.GEOM_GROUP_ROBOT_COLLISION,
                self.GEOM_GROUP_OBJECT_COLLISION,
            ) else 1
        if show_axes:
            scene_option.frame = mujoco.mjtFrame.mjFRAME_WORLD
        # Override fov via model; MjvCamera has no fovy.
        # Simpler: use a temporary free camera rendered through the renderer's default.
        self.model.vis.global_.fovy = fovy
        renderer.update_scene(self.data, camera=cam, scene_option=scene_option)
        img = renderer.render().copy()
        renderer.close()
        return img

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

    def reset(self):
        self.sim_time = 0.0
        self.sim_frame = 0
        self._sensor_traj = []
        self._build_mjcf_model()

    # ------------------------------------------------ Collision-filter helper
    def _apply_collision_policy(self) -> None:
        """Generate one consistent contype/conaffinity policy for the scene.

        The five groups use independent bits: GROUND, ARM, GRIPPER, PICKUP,
        and NONPICKUP. A pair collides iff either geom's contype intersects the
        other's conaffinity. In reduced-collision mode the enabled pairs are
        gripper↔ground, arm/gripper↔pickup, pickup↔non-pickup,
        non-pickup↔non-pickup, and object↔ground; arm↔ground and
        arm/gripper↔non-pickup are disabled. If the corresponding filters are
        off, arm↔ground and arm/gripper↔non-pickup are added back.

        `collision_gripper_body_prefixes` is deliberately separate from the
        profile's grasp-contact prefixes. AIRBOT has collision geoms on fixed
        bodies such as `eef_connect_base_link` and `g2_base_link`, but adding
        those bodies to grasp contact classification would change inner/outer
        pad semantics.

        Pure visual geoms (both masks zero before this pass) are left untouched
        so applying a policy never turns a render-only mesh into a collider.
        """
        BIT_GROUND = 1 << 0
        BIT_ARM = 1 << 1
        BIT_GRIPPER = 1 << 2
        BIT_PICKUP = 1 << 3
        BIT_NONPICKUP = 1 << 4
        group_bits = {
            "ground": BIT_GROUND,
            "arm": BIT_ARM,
            "gripper": BIT_GRIPPER,
            "pickup": BIT_PICKUP,
            "nonpickup": BIT_NONPICKUP,
        }

        allowed = {
            "ground": {"gripper", "pickup", "nonpickup"},
            "arm": {"gripper", "pickup"},
            "gripper": {"ground", "arm", "pickup"},
            "pickup": {"ground", "arm", "gripper", "nonpickup"},
            "nonpickup": {"ground", "pickup", "nonpickup"},
        }
        if not self.profile.exclude_arm_ground_collision:
            allowed["ground"].add("arm")
            allowed["arm"].add("ground")
        if not self._exclude_robot_nonpickup:
            allowed["arm"].add("nonpickup")
            allowed["gripper"].add("nonpickup")
            allowed["nonpickup"].update({"arm", "gripper"})

        pickup_name = (
            self._exclude_robot_nonpickup[0]
            if self._exclude_robot_nonpickup
            else None
        )
        scene_obj_body_ids: dict[int, str] = {}
        for obj in self.config.objects:
            try:
                body_id = self.model.body(obj.name).id
            except KeyError:
                continue
            scene_obj_body_ids[body_id] = obj.name

        robot_body_ids = set(range(1, self.model.nbody)) - set(scene_obj_body_ids)
        collision_gripper_prefixes = (
            self.profile.collision_gripper_body_prefixes
            if self.profile.collision_gripper_body_prefixes is not None
            else self.profile.gripper_body_prefixes
        )
        ground_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        for gid in range(self.model.ngeom):
            if (
                self.model.geom_contype[gid] == 0
                and self.model.geom_conaffinity[gid] == 0
            ):
                continue

            if gid == ground_gid:
                group = "ground"
            else:
                body_id = int(self.model.geom_bodyid[gid])
                obj_name = scene_obj_body_ids.get(body_id)
                if obj_name is not None:
                    group = "pickup" if obj_name == pickup_name else "nonpickup"
                elif body_id in robot_body_ids:
                    body_name = (
                        mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                        )
                        or ""
                    )
                    is_gripper = any(
                        body_name == prefix or body_name.startswith(prefix)
                        for prefix in collision_gripper_prefixes
                    )
                    group = "gripper" if is_gripper else "arm"
                else:
                    # Keep any unrelated world-body geom at its authored mask.
                    continue

            self.model.geom_contype[gid] = group_bits[group]
            self.model.geom_conaffinity[gid] = sum(
                group_bits[name] for name in allowed[group]
            )

    def _apply_gripper_contact_params(self) -> None:
        """Set friction/solimp/solref/priority on the gripper's collision geoms
        from the profile (post-compile, since MjSpec.to_xml drops them). No-op
        when the profile leaves ``gripper_contact_friction`` None (e.g. Franka,
        whose Robotiq MJCF already tunes its pads)."""
        p = self.profile
        if p.gripper_contact_friction is None:
            return
        prefixes = p.gripper_body_prefixes
        for gid in range(self.model.ngeom):
            if self.model.geom_contype[gid] == 0 and self.model.geom_conaffinity[gid] == 0:
                continue  # visual-only
            bid = int(self.model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not any(bname == pf or bname.startswith(pf) for pf in prefixes):
                continue
            self.model.geom_friction[gid] = p.gripper_contact_friction
            self.model.geom_solimp[gid][:3] = p.gripper_contact_solimp
            self.model.geom_solref[gid][:2] = p.gripper_contact_solref
            self.model.geom_priority[gid] = p.gripper_contact_priority
            self.model.geom_condim[gid] = 6   # enable spin/roll friction

    # --------------------------------------------------------- Ground tuning
    def set_ground_offset(self, offset: float) -> None:
        """In-place update of ground geom z position. No recompile.

        Re-derives the ground pose via
        ``derive_ground_pose_world_xy`` so the sign convention matches the
        sim build path. Mostly used by the v1 ``ground_alignment`` loop —
        Stage 0 now fixes ``ground.offset`` ahead of sim build, so this is
        rarely called in production but kept for the alignment-loop API.
        """
        self.config.ground.offset = float(offset)
        pos, quat = derive_ground_pose_world_xy(
            self.config.ground, self.config.phase5_shift[2],
        )
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        if gid < 0:
            raise RuntimeError("ground geom not found in compiled model")
        self.model.geom_pos[gid] = pos
        self.model.geom_quat[gid] = quat
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------- TCP / FK helpers
    def get_ee_pos(self) -> np.ndarray:
        """Current world-frame position of the gripper TCP (`ee_frame` site).

        Reads `data.site_xpos` for the named site, which is maintained by
        forward kinematics. Caller is responsible for having stepped /
        forwarded the model into the desired joint configuration.
        """
        return np.array(self.data.site("ee_frame").xpos, dtype=np.float64)

    def predict_closest_ee_to_object(self, obj_name: str) -> dict:
        """Forward-kinematics scan of `q_target` to find the trajectory frame
        where the gripper TCP comes closest to a given object's CURRENT pose.

        Side-effect-free w.r.t. simulation state: saves and restores qpos/qvel.
        Only joints 1..7 are modulated (gripper opening + object freejoints
        are untouched).

        Returns dict with:
            frame: int            — argmin frame index
            ee_pos: np.ndarray(3,) — TCP world position at that frame
            obj_pos: np.ndarray(3,) — object world position used as target
            distance: float       — closest distance achieved (m)
            frac: float           — frame index normalized to [0, 1]
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        obj_adr = self.model.joint(f"{obj_name}_freejoint").qposadr.item()
        obj_pos = saved_qpos[obj_adr:obj_adr + 3].copy()
        n_frames = self.q_target.shape[0]
        best = {"frame": 0, "ee_pos": np.zeros(3), "obj_pos": obj_pos,
                "distance": float("inf"), "frac": 0.0}
        try:
            for f in range(n_frames):
                for i, joint_id in enumerate(self.joint_id_map):
                    self.data.qpos[joint_id] = self.q_target[f][i]
                mujoco.mj_forward(self.model, self.data)
                ee = np.array(self.data.site("ee_frame").xpos, dtype=np.float64)
                d = float(np.linalg.norm(ee - obj_pos))
                if d < best["distance"]:
                    best = {
                        "frame": int(f), "ee_pos": ee, "obj_pos": obj_pos,
                        "distance": d,
                        "frac": float(f) / max(n_frames - 1, 1),
                    }
            return best
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)

    def simulate_freefall(
        self,
        duration_seconds: float = 0.5,
    ) -> dict[str, dict]:
        """Reset to first-frame state, step physics with robot frozen.

        Used by ground alignment to see whether objects rest stably on
        the ground. Re-enables contacts even if `is_kinematic=True`.
        Returns per-object {initial_pos, final_pos, drift, vertical_drift,
        had_contact}.
        """
        prev_disableflags = self.model.opt.disableflags
        self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        try:
            mujoco.mj_resetData(self.model, self.data)

            for i, joint_id in enumerate(self.joint_id_map):
                self.data.qpos[joint_id] = self.q_target[0][i]
                self.data.ctrl[i] = self.q_target[0][i]
            self._seed_gripper_qpos(self.gripper_norm[0])
            self.data.ctrl[self.profile.gripper_ctrl_index] = self.q_target[0][self.arm_dof]

            for obj in self.config.objects:
                if obj.static:
                    continue
                adr = self.model.joint(f"{obj.name}_freejoint").qposadr.item()
                self.data.qpos[adr:adr + 3] = self._obj_pos_traj[obj.name][0]
                self.data.qpos[adr + 3:adr + 7] = self._obj_quat_traj[obj.name][0]

            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            initial_pos: dict[str, np.ndarray] = {}
            for obj in self.config.objects:
                if obj.static:
                    initial_pos[obj.name] = np.array(
                        self.data.body(obj.name).xpos, dtype=np.float64)
                    continue
                adr = self.model.joint(f"{obj.name}_freejoint").qposadr.item()
                initial_pos[obj.name] = self.data.qpos[adr:adr + 3].copy()

            had_contact = {obj.name: False for obj in self.config.objects}
            n_steps = int(duration_seconds / self.sim_dt)
            for _ in range(n_steps):
                mujoco.mj_step(self.model, self.data)
                if self.data.ncon > 0:
                    for c_idx in range(self.data.ncon):
                        geom1 = self.data.contact.geom1[c_idx]
                        geom2 = self.data.contact.geom2[c_idx]
                        for obj in self.config.objects:
                            if any(
                                self.model.geom(geom).bodyid
                                == self.model.body(obj.name).id
                                for geom in (geom1, geom2)
                            ):
                                had_contact[obj.name] = True

            result: dict[str, dict] = {}
            for obj in self.config.objects:
                if obj.static:
                    final = np.array(self.data.body(obj.name).xpos,
                                     dtype=np.float64)
                else:
                    adr = self.model.joint(f"{obj.name}_freejoint").qposadr.item()
                    final = self.data.qpos[adr:adr + 3].copy()
                delta = final - initial_pos[obj.name]
                result[obj.name] = {
                    "initial_pos": initial_pos[obj.name],
                    "final_pos": final,
                    "drift": float(np.linalg.norm(delta)),
                    "vertical_drift": float(delta[2]),
                    "had_contact": had_contact[obj.name],
                }
            return result
        finally:
            self.model.opt.disableflags = prev_disableflags
