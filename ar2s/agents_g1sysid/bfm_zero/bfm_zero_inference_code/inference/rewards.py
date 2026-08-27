import abc
import re
import typing as tp

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .utils import tolerance

COORD_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALIGNMENT_BOUNDS = {"x": (-0.1, 0.1), "z": (0.9, float("inf")), "y": (-0.1, 0.1)}

GRAVITY_VECTOR = np.array([0, 0, -1], dtype=np.float32)

REWARD_LIMITS = {
    "l": [0.6, 0.8, 0.2],
    "m": [1.0, float("inf"), 0.1],
}


class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model,
        data,
    ) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> tp.Optional["RewardFunction"]: ...

    def __call__(
        self,
        model,
        warp_model,
        data,
        qpos,
        qvel,
        ctrl,
    ):
        wp.copy(data.qpos, wp.array(qpos))
        wp.copy(data.qvel, wp.array(qvel))
        wp.copy(data.ctrl, wp.array(ctrl))
        mjw.forward(warp_model, data)
        return self.compute(model, data)

    def get_sensor_data(self, model, data, sensor_name) -> np.ndarray:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)  # in global coordinate
        start = model.sensor_adr[sensor_id]
        end = start + model.sensor_dim[sensor_id]
        sdata = wp.to_torch(data.sensordata)[:, start:end]
        return sdata


class MJWLocomotionReward(RewardFunction):
    move_speed: float = 5
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    stay_low: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        root_height = wp.to_torch(data.xpos[:, pelvis_id])[:, -1]
        center_of_mass_velocity = self.get_sensor_data(model, data, "torso_link_subtreelinvel")
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle) * torch.ones(root_height.shape[0], device=root_height.device, dtype=root_height.dtype)
        if self.egocentric_target:
            pelvis_xmat = wp.to_torch(data.xmat[:, pelvis_id])
            euler = self.rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[:, -1]

        if self.stay_low:
            standing = tolerance(
                root_height,
                bounds=(self.stand_height * 0.95, self.stand_height * 1.05),
                margin=self.stand_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        upvector_torso = self.get_sensor_data(model, data, "upvector_torso")
        cost_orientation = tolerance(
            torch.sum(
                torch.square(upvector_torso - torch.tensor([0.073, 0.0, 1.0], device=upvector_torso.device, dtype=upvector_torso.dtype)),
                dim=-1,
            ),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        small_control = 1.0
        if 0 <= self.move_speed <= 0.01:
            horizontal_velocity = center_of_mass_velocity[:, [0, 1]]
            dont_move = tolerance(horizontal_velocity, margin=0.2).mean(dim=-1)
            angular_velocity = self.get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = tolerance(angular_velocity, margin=0.1).mean(dim=-1)
            return small_control * stand_reward * dont_move * dont_rotate
        else:
            vel = center_of_mass_velocity[:, [0, 1]]
            com_velocity = torch.norm(vel, dim=-1)
            move = tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if move_angle is None:
                reward = small_control * stand_reward * move
            else:
                direction = vel / (com_velocity + 1e-6).reshape(-1, 1).repeat(1, 2)
                target_direction = torch.stack((torch.cos(move_angle), torch.sin(move_angle)), dim=-1)
                dot = (target_direction * direction).sum(dim=-1)
                angle_reward = (dot + 1.0) / 2.0
                angle_reward = torch.where(torch.isclose(com_velocity, torch.zeros_like(com_velocity)), 1.0, angle_reward)
                reward = small_control * stand_reward * move * angle_reward
            return reward

    def rot2eul(self, R: torch.Tensor) -> torch.Tensor:
        beta = -torch.arcsin(R[:, 2, 0])
        alpha = torch.atan2(R[:, 2, 1] / torch.cos(beta), R[:, 2, 2] / torch.cos(beta))
        gamma = torch.atan2(R[:, 1, 0] / torch.cos(beta), R[:, 0, 0] / torch.cos(beta))
        x = torch.concatenate((alpha.reshape(-1, 1), beta.reshape(-1, 1), gamma.reshape(-1, 1)), dim=-1)
        return x

    @staticmethod
    def reward_from_name(name: str) -> tp.Optional["RewardFunction"]:
        pattern = r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return MJWLocomotionReward(move_angle=move_angle, move_speed=move_speed)
        pattern = r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            stand_height, move_angle, move_speed = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return MJWLocomotionReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, stand_height=stand_height)
        return None


class MJWRotationReward(RewardFunction):
    axis: str = "z"
    target_ang_velocity: float = 5.0
    # Note: pelvis height is 0.8 exactly after reset with default pose
    stand_pelvis_height: float = 0.8

    def compute(
        self,
        model,
        data,
    ) -> float:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_height = wp.to_torch(data.xpos[:, pelvis_id])[:, -1]
        pelvis_xmat = wp.to_torch(data.xmat[:, pelvis_id])
        torso_rotation = pelvis_xmat[:, 2, :]
        angular_velocity = self.get_sensor_data(model, data, "imu-angular-velocity")

        height_reward = tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        small_control = tolerance(wp.to_torch(data.ctrl), margin=1.0, value_at_margin=0.0, sigmoid="quadratic").mean(-1)
        small_control = (4 + small_control) / 5
        small_control = 1

        targ_av = abs(self.target_ang_velocity)
        move = tolerance(
            direction * angular_velocity[:, COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        aligned = tolerance(
            torso_rotation[:, COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        reward = move * height_reward * small_control * aligned
        return reward

    @staticmethod
    def reward_from_name(name: str) -> tp.Optional["RewardFunction"]:
        pattern = r"^rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            axis, target_ang_velocity, stand_pelvis_height = (
                match.group(1),
                float(match.group(2)),
                float(match.group(3)),
            )
            return MJWRotationReward(
                axis=axis,
                target_ang_velocity=target_ang_velocity,
                stand_pelvis_height=stand_pelvis_height,
            )
        return None


class MJWArmsReward(RewardFunction):
    # head height of the character in T-pose, this is used as a reference to compute whether the character is standing
    left_pose: str = "m"
    right_pose: str = "m"
    # target height for standing pose
    stand_height: float = 0.5

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        left_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
        right_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        root_height = wp.to_torch(data.xpos[:, pelvis_id])[:, -1]
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        center_of_mass_velocity = self.get_sensor_data(model, data, "torso_link_subtreelinvel")
        left_height = wp.to_torch(data.xpos[:, left_wrist_id])[:, -1]
        right_height = wp.to_torch(data.xpos[:, right_wrist_id])[:, -1]
        standing = tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upvector_torso = self.get_sensor_data(model, data, "upvector_torso")
        cost_orientation = tolerance(
            torch.sum(
                torch.square(upvector_torso - torch.tensor([0.073, 0.0, 1.0], device=upvector_torso.device, dtype=upvector_torso.dtype)),
                dim=-1,
            ),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        dont_move = tolerance(center_of_mass_velocity, margin=0.2).mean(dim=-1)
        angular_velocity = self.get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = tolerance(angular_velocity, margin=0.1).mean(dim=-1)
        # dont_move = rewards.tolerance(data.qvel, margin=0.5).mean()
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1
        left_arm = tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        return small_control * stand_reward * dont_move * left_arm * right_arm * dont_rotate

    @staticmethod
    def reward_from_name(name: str) -> tp.Optional["RewardFunction"]:
        pattern = r"^raisearms-(l|m|h|x)-(l|m|h|x)"
        match = re.search(pattern, name)
        if match:
            left_pose, right_pose = match.group(1), match.group(2)
            return MJWArmsReward(left_pose=left_pose, right_pose=right_pose)
        return None


class MJWMoveArmsReward(MJWLocomotionReward):
    move_speed: float = 1
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    low_height: float = 0.5
    stay_low: bool = False  # TODO this currently doesn't work because arm heights are expressed in global coords
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(self, model, data) -> float:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        left_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
        right_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        root_height = wp.to_torch(data.xpos[:, pelvis_id])[:, -1]
        center_of_mass_velocity = self.get_sensor_data(model, data, "torso_link_subtreelinvel")
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle) * torch.ones(root_height.shape[0], device=root_height.device, dtype=root_height.dtype)
        if self.egocentric_target:
            pelvis_xmat = wp.to_torch(data.xmat[:, pelvis_id])
            euler = self.rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[:, -1]

        # STANDING HEIGHT
        if self.stay_low:
            standing = tolerance(
                root_height,
                bounds=(self.low_height / 2, self.low_height),
                margin=self.low_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )

        # STANDING STRAIGHT
        upvector_torso = self.get_sensor_data(model, data, "upvector_torso")
        cost_orientation = tolerance(
            torch.sum(
                torch.square(upvector_torso - torch.tensor([0.073, 0.0, 1.0], device=upvector_torso.device, dtype=upvector_torso.dtype)),
                dim=-1,
            ),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        small_control = 1.0

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = wp.to_torch(data.xpos[:, left_wrist_id])[:, -1]
        right_height = wp.to_torch(data.xpos[:, right_wrist_id])[:, -1]
        left_arm = tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        if self.move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[:, [0, 1]]
            dont_move = tolerance(horizontal_velocity, margin=0.2).mean(dim=-1)
            angular_velocity = self.get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = tolerance(angular_velocity, margin=0.1).mean(dim=-1)
            reward = small_control * stand_reward * dont_move * dont_rotate * left_arm * right_arm
            return reward
        else:
            vel = center_of_mass_velocity[:, [0, 1]]
            com_velocity = torch.norm(vel, dim=-1)
            move = tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6).reshape(-1, 1).repeat(1, 2)
                target_direction = torch.stack((torch.cos(move_angle), torch.sin(move_angle)), dim=-1)
                dot = (target_direction * direction).sum(dim=-1)
                angle_reward = (dot + 1.0) / 2.0
                angle_reward = torch.where(torch.isclose(com_velocity, torch.zeros_like(com_velocity)), 1.0, angle_reward)
            reward = small_control * stand_reward * move * angle_reward * left_arm * right_arm
            return reward

    @staticmethod
    def reward_from_name(name: str) -> tp.Optional["RewardFunction"]:
        pattern = r"^move-arms-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MJWMoveArmsReward(move_angle=move_angle, move_speed=move_speed, left_pose=left_pose, right_pose=right_pose)
        pattern = r"^move-ego-low-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MJWMoveArmsReward(
                move_angle=move_angle, move_speed=move_speed, stay_low=True, left_pose=left_pose, right_pose=right_pose
            )
        return None


class MJWSpinArmsReward(RewardFunction):
    axis: str = "z"
    target_ang_velocity: float = 5.0
    stand_pelvis_height: float = 0.5
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        left_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
        right_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        pelvis_height = wp.to_torch(data.xpos[:, pelvis_id])[:, -1]
        pelvis_xmat = wp.to_torch(data.xmat[:, pelvis_id])
        torso_rotation = pelvis_xmat[:, 2, :]
        angular_velocity = self.get_sensor_data(model, data, "imu-angular-velocity")

        # PELVIS HEIGHT
        height_reward = tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        # SMALL CONTROL
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1

        # SPINNING
        targ_av = np.abs(self.target_ang_velocity)
        move = tolerance(
            direction * angular_velocity[:, COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        # UPRIGHT
        aligned = tolerance(
            torso_rotation[:, COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = wp.to_torch(data.xpos[:, left_wrist_id])[:, -1]
        right_height = wp.to_torch(data.xpos[:, right_wrist_id])[:, -1]
        left_arm = tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        reward = move * height_reward * small_control * aligned * left_arm * right_arm
        return reward

    @staticmethod
    def reward_from_name(name: str) -> tp.Optional["RewardFunction"]:
        pattern = r"^spin-arms-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            target_ang_velocity, left_pose, right_pose = (
                float(match.group(1)),
                match.group(2),
                match.group(3),
            )
            return MJWSpinArmsReward(
                target_ang_velocity=target_ang_velocity,
                left_pose=left_pose,
                right_pose=right_pose,
            )
        return None


class MJWSitOnGroundReward(RewardFunction):
    pelvis_height_th: float = 0
    constrained_knees: bool = False
    knees_not_on_ground: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        left_knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
        right_knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_knee_link")
        pelvis_height = wp.to_torch(data.xpos[:, pelvis_id])[:, -1]
        left_knee_pos = wp.to_torch(data.xpos[:, left_knee_id])[:, -1]
        right_knee_pos = wp.to_torch(data.xpos[:, right_knee_id])[:, -1]
        center_of_mass_velocity = self.get_sensor_data(model, data, "torso_link_subtreelinvel")
        upvector_torso = self.get_sensor_data(model, data, "upvector_torso")
        cost_orientation = tolerance(
            torch.sum(
                torch.square(upvector_torso - torch.tensor([0.073, 0.0, 1.0], device=upvector_torso.device, dtype=upvector_torso.dtype)),
                dim=-1,
            ),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        dont_move = tolerance(center_of_mass_velocity, margin=0.5).mean(dim=-1)
        angular_velocity = self.get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = tolerance(angular_velocity, margin=0.1).mean(dim=-1)
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1
        pelvis_reward = tolerance(
            pelvis_height,
            bounds=(self.pelvis_height_th, self.pelvis_height_th + 0.1),
            sigmoid="linear",
            margin=0.7,
            value_at_margin=0,
        )
        knee_reward = 1
        if self.constrained_knees:
            knee_reward *= tolerance(
                left_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
            knee_reward *= tolerance(
                right_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
        if self.knees_not_on_ground:
            knee_reward *= tolerance(
                left_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
            knee_reward *= tolerance(
                right_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
        return small_control * cost_orientation * dont_move * dont_rotate * pelvis_reward * (2 * knee_reward + 1) / 3

    @staticmethod
    def reward_from_name(name: str) -> tp.Optional["RewardFunction"]:
        if name == "sitonground":
            pelvis_height_th = 0
            return MJWSitOnGroundReward(pelvis_height_th=pelvis_height_th, constrained_knees=True)
        pattern = r"^crouch-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            pelvis_height_th = float(match.group(1))
            return MJWSitOnGroundReward(pelvis_height_th=pelvis_height_th, knees_not_on_ground=True)
        return None
