"""RobotProfile — per-robot facts the sim needs, so RobotSceneSim /
RobotOnlySim stop hard-coding Franka Panda + Robotiq 2F-85.

A profile captures everything that differs between robots: arm DOF and joint
names, the incoming-trajectory zero offsets, how the MJCF is loaded (single
combined file vs an arm + gripper pair attached at runtime), the body that
hosts the TCP site, and how a normalised [0,1] gripper command maps onto the
gripper's joints/actuator.

Two profiles ship today:
  * FRANKA_PANDA   — 7-DOF Panda + Robotiq 2F-85 (the original hard-coded robot)
  * AIRBOT_PLAY_G2 — 6-DOF AIRBOT Play + G2 parallel gripper

`RobotSceneSim` defaults to FRANKA_PANDA, so existing behaviour is unchanged.

Gripper command convention: the sim works in a normalised gripper signal
`norm ∈ [0,1]` taken from `robot_traj.h5` `gripper_position`. Each profile says
how `norm` becomes (a) the per-joint qpos seeded in kinematic mode and (b) the
actuator ctrl value in dynamic mode. NOTE the two robots differ in polarity of
the underlying joint, but that polarity is already baked into how each dataset
normalises gripper_position, so both use the same `scale * norm` form:
  * Franka/Robotiq: norm 0=open→1=closed; driver-joint qpos = 0.8*norm;
    actuator ctrl (0-255 tendon) = 255*norm.
  * AIRBOT/G2:       norm already = g2_joint/0.072 (1=open→0=closed);
    g2_joint qpos/ctrl (metres) = 0.072*norm; fingers = 0.036*norm (mimic ×0.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotProfile:
    name: str                          # registry key; also manifest.robot.type
    gripper_name: str                  # manifest.robot.gripper (record only)
    arm_dof: int
    arm_joint_names: tuple[str, ...]

    # Per-arm-joint offset SUBTRACTED from incoming DROID joint_position
    # (Franka's joint7 has a −π/4 zero-offset vs the MuJoCo panda convention).
    # Length must equal arm_dof.
    arm_zero_offsets: tuple[float, ...]

    # --- MJCF loading -----------------------------------------------------
    # Either a single combined arm+gripper MJCF (combined_mjcf set, the attach
    # fields None), or an arm MJCF whose `arm_attach_body` frame receives the
    # gripper MJCF's `gripper_attach_body` at runtime (Franka style).
    combined_mjcf: str | None          # repo-relative .xml, or None
    arm_mjcf: str | None               # repo-relative .xml, or None
    gripper_mjcf: str | None           # repo-relative .xml, or None
    arm_mesh_dir: str | None           # repo-relative mesh dir for _absolutize, or None
    gripper_mesh_dir: str | None
    arm_root_body: str                 # body attached at the robot base frame
    arm_attach_body: str | None        # arm body the gripper mounts onto (split only)
    gripper_attach_body: str | None    # gripper root body (split only)

    # --- TCP site ---------------------------------------------------------
    ee_site_body: str                  # body that hosts the "ee_frame" site
    ee_site_pos: tuple[float, float, float]

    # --- Gripper actuation ------------------------------------------------
    # Joints whose qpos is seeded directly in kinematic mode, each with the
    # scale applied to norm∈[0,1]. In kinematic mode mj_forward does NOT solve
    # the mimic equality, so every finger joint must be listed explicitly.
    gripper_seed_joints: tuple[tuple[str, float], ...]
    gripper_ctrl_scale: float          # norm→actuator ctrl (Franka 255, AIRBOT 0.072)
    # Body-name prefixes identifying the gripper's bodies (grasp_probe contact
    # classification). A contact on a body matching any prefix is "gripper".
    gripper_body_prefixes: tuple[str, ...]
    # Substring marking an INNER-pad (grasping-surface) gripper body; anything
    # else under the gripper is OUTER (the failure mode grasp_probe detects).
    # For a single-body parallel jaw, match the finger link so every gripper
    # contact reads as inner (no outer-knuckle failure mode).
    gripper_inner_pad_key: str
    # Gripper finger contact tuning, applied to the gripper collision geoms in
    # RobotSceneSim AFTER compile (MjSpec.to_xml drops these on the build_mjcf
    # round-trip). None = leave the MJCF's own params (Franka's Robotiq xml
    # already tunes its pads). priority>0 makes the finger's params win over the
    # object's in the grasp contact pair.
    # Which end of the normalised gripper signal means CLOSED. robot_traj's
    # gripper_position is unit-scaled per dataset and the two robots disagree:
    # DROID/Franka records 0=open → 1=closed, while the AIRBOT G2's g2_joint is
    # an OPENING in metres, so its norm 1 = fully open. Actuation is unaffected
    # (both are scale*norm), but every "is the gripper closing?" test has to
    # know which end is which — use closedness() rather than reading the raw
    # norm. Getting this backwards makes the grasp detector fire on the
    # gripper OPENING.
    gripper_norm_one_is_closed: bool = True
    gripper_contact_friction: tuple[float, float, float] | None = None
    gripper_contact_solimp: tuple[float, float, float] = (0.95, 0.99, 0.001)
    gripper_contact_solref: tuple[float, float] = (0.004, 1.0)
    gripper_contact_priority: int = 1
    # Dynamic-mode close clamp as a fraction of full ctrl range
    # (Franka 0.78/0.8=0.975 keeps the driver just shy of the hard stop).
    gripper_close_cap_frac: float = 1.0
    # When True, disable collision between the arm's NON-gripper links and the
    # ground plane (gripper↔ground and object↔ground stay on). Temporary aid
    # while the ground height / FK reach are still being calibrated so stray
    # arm-link↔floor contacts don't corrupt the dynamics.
    exclude_arm_ground_collision: bool = False
    # Body-name prefixes used only by RobotSceneSim's collision policy to
    # distinguish gripper links from arm links. None preserves the historical
    # behaviour of reusing gripper_body_prefixes; grasp contact classification
    # and gripper friction continue to use gripper_body_prefixes directly.
    collision_gripper_body_prefixes: tuple[str, ...] | None = None

    def closedness(self, norm):
        """Normalised gripper signal → 0.0 = fully open, 1.0 = fully closed.

        The one place the per-robot polarity lives, so callers can compare
        against a single set of thresholds regardless of robot.
        """
        n = np.clip(norm, 0.0, 1.0)
        return n if self.gripper_norm_one_is_closed else 1.0 - n

    @property
    def gripper_ctrl_index(self) -> int:
        """Actuator/ctrl column of the gripper = arm_dof (arm actuators first)."""
        return self.arm_dof

    @property
    def is_combined(self) -> bool:
        return self.combined_mjcf is not None

    def __post_init__(self) -> None:
        if len(self.arm_zero_offsets) != self.arm_dof:
            raise ValueError(
                f"arm_zero_offsets len {len(self.arm_zero_offsets)} != arm_dof {self.arm_dof}")
        if len(self.arm_joint_names) != self.arm_dof:
            raise ValueError(
                f"arm_joint_names len {len(self.arm_joint_names)} != arm_dof {self.arm_dof}")
        if self.is_combined:
            if self.arm_attach_body or self.gripper_attach_body:
                raise ValueError("combined MJCF must not set attach bodies")
        else:
            if not (self.arm_mjcf and self.gripper_mjcf):
                raise ValueError("split robot needs both arm_mjcf and gripper_mjcf")


import numpy as np  # noqa: E402  (used only for the π/4 constant below)

FRANKA_PANDA = RobotProfile(
    name="franka_panda",
    gripper_name="robotiq_2f85",
    arm_dof=7,
    arm_joint_names=tuple(f"joint{i}" for i in range(1, 8)),
    arm_zero_offsets=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 4),
    combined_mjcf=None,
    arm_mjcf="assets/robot/franka_emika_panda/panda_nohand.xml",
    gripper_mjcf="assets/robot/robotiq_2f85/2f85.xml",
    arm_mesh_dir="assets/robot/franka_emika_panda/assets",
    gripper_mesh_dir="assets/robot/robotiq_2f85/assets",
    arm_root_body="link0",
    arm_attach_body="attachment",
    gripper_attach_body="base_mount",
    ee_site_body="base_mount",
    ee_site_pos=(0.0, 0.0, 0.160),
    gripper_seed_joints=(("right_driver_joint", 0.8), ("left_driver_joint", 0.8)),
    gripper_ctrl_scale=255.0,
    gripper_body_prefixes=(
        "base_mount", "base",
        "right_driver", "right_coupler", "right_spring_link",
        "right_follower", "right_pad", "right_silicone_pad",
        "left_driver", "left_coupler", "left_spring_link",
        "left_follower", "left_pad", "left_silicone_pad",
    ),
    gripper_inner_pad_key="pad",
    gripper_close_cap_frac=0.78 / 0.8,
)

AIRBOT_PLAY_G2 = RobotProfile(
    name="airbot_play",
    gripper_name="airbot_g2",
    arm_dof=6,
    arm_joint_names=tuple(f"joint{i}" for i in range(1, 7)),
    arm_zero_offsets=(0.0,) * 6,
    combined_mjcf="assets/robot/airbot_play_g2/airbot_play_g2.xml",
    arm_mjcf=None,
    gripper_mjcf=None,
    arm_mesh_dir="assets/robot/airbot_play_g2/meshes",
    gripper_mesh_dir=None,
    # base_link is the URDF root: MuJoCo merges the fixed root into worldbody,
    # so the combined robot is attached with MjSpec.attach (see _build_mjcf_model)
    # rather than attach_body(arm_root_body). The generated XML keeps the fixed
    # gripper chain as separate bodies, while the TCP site still rides link6
    # and is offset out along its +z toward the finger pinch line.
    arm_root_body="base_link",
    arm_attach_body=None,
    gripper_attach_body=None,
    ee_site_body="link6",
    ee_site_pos=(0.0, 0.0, 0.215),   # 2cm inside the previous pinch anchor (0.235); tip is 0.2515
    gripper_seed_joints=(
        ("g2_joint", 0.072),
        ("g2_left_joint", 0.036),
        ("g2_right_joint", 0.036),
    ),
    gripper_ctrl_scale=0.072,
    gripper_norm_one_is_closed=False,   # g2_joint is an opening: norm 1 = open
    # G2 fingers are the two finger links. Match both finger links; the inner
    # key matches them too so every G2 gripper contact reads as inner (a
    # parallel jaw has no outer-knuckle failure mode). Collision classification
    # additionally treats the fixed wrist/gripper links as gripper bodies, but
    # grasp contact classification intentionally remains unchanged.
    gripper_body_prefixes=("g2_left_link", "g2_right_link"),
    collision_gripper_body_prefixes=(
        "eef_connect_base_link",
        "g2_base_link",
        "g2_left_link",
        "g2_right_link",
    ),
    gripper_inner_pad_key="g2_",
    # Grippier than the Robotiq pad (0.7) + real torsional/rolling friction so a
    # thin round handle (spoon) doesn't pivot/roll out of the parallel jaw.
    gripper_contact_friction=(1.0, 0.05, 0.02),   # slide / spin / roll
    gripper_contact_solimp=(0.95, 0.99, 0.001),
    gripper_contact_solref=(0.004, 1.0),
    gripper_contact_priority=1,
    gripper_close_cap_frac=1.0,
    exclude_arm_ground_collision=True,
)

_REGISTRY = {p.name: p for p in (FRANKA_PANDA, AIRBOT_PLAY_G2)}


def get_robot_profile(name: str) -> RobotProfile:
    if name not in _REGISTRY:
        raise KeyError(f"unknown robot profile {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]
