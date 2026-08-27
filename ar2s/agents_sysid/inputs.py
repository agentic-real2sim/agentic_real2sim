"""User-facing input schema for the sysid pipeline."""
from dataclasses import dataclass, field

from ar2s.droid_sim.scene.config import CameraSpec


@dataclass
class ObjectInput:
    name: str
    visual_mesh_path: str
    scale_path: str
    pose_source_path: str
    collision_mesh_dir: str = ""
    mass: float = 0.01
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    init_frame: int = 0                              # earliest frame index with a FP pose .npy;
                                                      # frames 0..init_frame-1 hold the init pose
                                                      # static (per-object keyframe from visual)
    static: bool = False                             # no freejoint: fixed support
                                                      # furniture (e.g. table)
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)   # world-frame translation
                                                                 # added on top of FP-derived pose
                                                                 # at sim build time. Use for
                                                                 # post-hoc world-frame alignment
                                                                 # (filled by geometry_prior Stage 0).


@dataclass
class GroundInput:
    reference_object: str = ""
    local_down_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    offset: float = 0.0


@dataclass
class InputBundle:
    scene_name: str
    robot_traj_path: str
    objects: list[ObjectInput]
    cameras: CameraSpec
    ground: GroundInput = field(default_factory=GroundInput)
    robot_type: str = "franka_panda"     # manifest robot.type; selects RobotProfile
    robot_mask_path: str = ""
    first_frame_rgb_path: str = ""
    # ``pickup_object`` is ``pickup_objects[0]`` — the single object the sysid
    # grasp stages drive. The full list is kept for readers that need it.
    pickup_object: str = ""
    pickup_objects: list[str] = field(default_factory=list)


__all__ = ["GroundInput", "InputBundle", "ObjectInput"]
