"""Dataclasses for describing a scene consumed by RobotSceneSim.

See docs/agent_pipeline/architecture.md §3 for field semantics and §4 for
which fields are tuned by which agent.
"""
from dataclasses import dataclass, field


@dataclass
class FoundationPoseDir:
    """Per-frame object pose in camera frame, from FoundationPose outputs.

    `path` contains files named `frame_NNNNNN_transform.npy`, each a (4, 4)
    homogeneous transform from the object frame to the camera frame.

    `init_frame` is the earliest frame index for which a pose .npy exists
    in `path` (per-object keyframe — different objects may start at
    different frames if visual's keyframe_select picked a non-zero
    keyframe for them, e.g. snack_package starting at frame 2 because
    SAM3 couldn't segment it in frames 0-1). Frames 0..init_frame-1 are
    filled with the init_frame pose (object treated as static during
    that window). Default 0 = require frame_000000_transform.npy.
    """
    path: str
    init_frame: int = 0


@dataclass
class SceneObject:
    name: str                                                        # body name in MJCF, unique
    visual_mesh_path: str
    scale_path: str                                                  # .npy with (3,) per-axis scale
    pose_source: FoundationPoseDir
    collision_mesh_paths: list[str] = field(default_factory=list)    # empty => reuse visual as collision
    mass: float = 0.01
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)      # slide, spin, roll
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_offset: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)   # wxyz, identity
    scale_multiplier: tuple[float, float, float] = (1.0, 1.0, 1.0)   # agent-tunable delta on base scale
    static: bool = False                                             # no freejoint: fixed support furniture (e.g. table)
    rgba: tuple[float, float, float, float] | None = None            # flat colour for the visual geom;
                                                                     # None => MuJoCo's default grey


@dataclass
class GroundSpec:
    reference_object: str                                            # which SceneObject defines the floor
    local_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)   # in the reference object's local frame (default = MuJoCo z-up)
    offset: float = 0.0                                              # signed, along local_down_axis
    plane_size: tuple[float, float, float] = (5.0, 5.0, 0.1)


@dataclass
class CameraSpec:
    camera_ids: tuple[int, ...]
    object_camera_id: int                                            # which extrinsics to use for pose_source → world
    extrinsics_path: str                                             # .npz with cam_mat_{id}
    intrinsics_path: str                                             # .txt (shared K) or .npz with cam_K_{id}
    # Native sensor size (W, H) of the real cameras. Renders of a real camera
    # must be requested at this size: render_cameras crops a window around the
    # TRUE principal point, so asking for a smaller frame slides that window
    # off-centre instead of zooming out. K alone cannot supply it — cx is not
    # W/2 on a real sensor. None = unknown, callers fall back to 1280x720
    # (every DROID camera).
    image_size: tuple[int, int] | None = None


@dataclass
class SceneConfig:
    scene_name: str
    robot_traj_path: str                                             # .h5 with joint_position, gripper_position
    objects: list[SceneObject]
    ground: GroundSpec
    cameras: CameraSpec
    # Phase-5 group shift, accumulated across SceneShiftAgent rounds.
    # User-spec basis (NOT plain world frame):
    #   dx → projection of world +x onto ground plane (in-plane)
    #   dy → projection of world +y onto ground plane (in-plane)
    #   dz → world +z                                  (vertical)
    # Converted to world-frame displacement at sim build time; see
    # `ar2s.droid_sim.scene.shift.shift_to_world_disp_obj`.
    phase5_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
