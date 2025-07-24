# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
# from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still_without_cmd(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return (
        torch.sum(torch.abs(diff_angle), dim=1) * command  # * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 1)
    )


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return torch.where(
        torch.logical_or(cmd > 0.1, body_vel > velocity_threshold), reward, stand_still_scale * reward
    )  # * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 1)
    # reward = torch.square(
    #     asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # )
    # return torch.sum(reward, dim=1)


def arm_joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = stand_still_scale * torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return reward
    # * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 1)
    # reward = torch.square(
    #     asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # )
    # return torch.sum(reward, dim=1)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.1, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command(command_name), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # Penalize feet hitting vertical surfaces
    return torch.any(contact & (forces_xy > forces_z), dim=1)


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    return torch.exp(-torch.sum(stance_diff, dim=1) / std)


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    return torch.exp(-torch.sum(stance_diff, dim=1) / std)


def feet_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return torch.exp(-reward / std)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def wheel_spin_in_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(in_air * joint_vel, dim=1)
    return reward


def upward(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_error = asset.data.projected_gravity_b[:, 2] - 1
    return torch.exp(-gravity_error / std**2)


def track_lin_vel_world_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_w[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_world_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def feet_height_body_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        # footpos_in_body_frame[:, i, :] = quat_apply_inverse(
        #     asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        # )
        footvel_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
        # footvel_in_body_frame[:, i, :] = quat_apply_inverse(
        #     asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        # )
    height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(height_error * foot_leteral_vel, dim=1)
    return torch.exp(-reward / std**2)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)



def tracking_ee_sphere(
    env: ManagerBasedRLEnv,
    tracking_sigma: float,
    error_scale: torch.Tensor,
    target_spherical_pos: torch.Tensor,
    sphere_center_cfg: SceneEntityCfg = SceneEntityCfg("target_entity"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot.ee_link"),
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot.base_link"),
) -> torch.Tensor:
    """Reward for tracking end-effector position in spherical coordinates.
    
    Computes an exponential reward based on the difference between current EE
    position and target position in spherical coordinates relative to a center point.
    
    Args:
        env: The environment instance.
        tracking_sigma: Controls the decay rate of the exponential reward.
        error_scale: Scaling factors for [radial, polar, azimuthal] errors.
        target_spherical_pos: Target position in spherical coordinates (r, θ, φ).
        sphere_center_cfg: Configuration for the spherical coordinate center entity.
        ee_cfg: Configuration for the end-effector entity.
        base_cfg: Configuration for the base entity (for coordinate rotation).
        
    Returns:
        The computed exponential tracking reward.
    """
    # Extract scene entities
    ee_entity = env.scene[ee_cfg.name]
    base_entity = env.scene[base_cfg.name]
    center_entity = env.scene[sphere_center_cfg.name]
    
    # Get positions and orientations
    ee_pos = ee_entity.data.pos_w  # [num_envs, 3]
    center_pos = center_entity.data.pos_w  # [num_envs, 3]
    base_quat = base_entity.data.quat_w  # [num_envs, 4]
    
    # Compute vector from center to EE
    center_to_ee = ee_pos - center_pos  # [num_envs, 3]
    
    # Rotate vector to base frame (ignoring base yaw)
    center_to_ee_local = quat_rotate_inverse(base_quat, center_to_ee)
    
    # Convert Cartesian to spherical coordinates
    r, theta, phi = cartesian_to_spherical(center_to_ee_local)
    current_spherical = torch.stack([r, theta, phi], dim=-1)  # [num_envs, 3]
    
    # Compute absolute errors in each spherical component
    abs_errors = torch.abs(current_spherical - target_spherical_pos.to(ee_pos.device))
    
    # Scale errors by component-specific factors
    scaled_errors = abs_errors * error_scale.to(ee_pos.device)
    
    # Sum errors across components
    total_error = torch.sum(scaled_errors, dim=-1)  # [num_envs]
    
    # Exponential reward based on total error
    return torch.exp(-total_error / tracking_sigma)


def tracking_ee_world(
    env: ManagerBasedRLEnv,
    tracking_sigma: float,
    target_position: torch.Tensor,
    sigma_scale: float = 2.0,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot.ee_link"),
    return_error: bool = False,
) -> torch.Tensor:
    """Reward for tracking end-effector position in Cartesian world coordinates.
    
    Computes an exponential reward based on the L1 distance between current EE
    position and target position in world coordinates.
    
    Args:
        env: The environment instance.
        tracking_sigma: Controls the decay rate of the exponential reward.
        target_position: Target position in world coordinates [x, y, z].
        sigma_scale: Scaling factor applied to tracking_sigma (default=2.0).
        ee_cfg: Configuration for the end-effector entity.
        return_error: If True, returns a tuple (reward, error); otherwise only reward.
        
    Returns:
        The computed exponential tracking reward, or (reward, error) if return_error=True.
    """
    # Extract end-effector entity
    ee_entity: RigidObject = env.scene[ee_cfg.name]
    # ee_entity = env.scene[ee_cfg.name]
    
    # Get current end-effector position
    ee_pos = ee_entity.data.body_pos_w  # [num_envs, 3]
    
    # Ensure target position has correct shape
    if target_position.ndim == 1:
        target_position = target_position.unsqueeze(0).repeat(ee_pos.shape[0], 1)
    
    # Move target to same device as ee_pos
    target_position = target_position.to(ee_pos.device)
    
    # Compute L1 position error (absolute difference per axis, summed)
    ee_pos_error = torch.sum(torch.abs(ee_pos - target_position), dim=1)
    
    # Compute exponential reward
    adjusted_sigma = tracking_sigma * sigma_scale
    reward = torch.exp(-ee_pos_error / adjusted_sigma)
    
    # Return based on flag
    if return_error:
        return reward, ee_pos_error
    return reward


def tracking_ee_sphere_walking(
    env: ManagerBasedRLEnv,
    tracking_sigma: float,
    error_scale: torch.Tensor,
    target_spherical_pos: torch.Tensor,
    walking_cmd_entity_cfg: SceneEntityCfg = SceneEntityCfg("command"),
    sphere_center_cfg: SceneEntityCfg = SceneEntityCfg("target_entity"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot.ee_link"),
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot.base_link"),
    return_error: bool = False,
) -> torch.Tensor:
    """Reward for tracking end-effector position in spherical coordinates during walking.
    
    Computes an exponential reward based on EE position tracking, but only applied
    when the robot is in walking mode (as determined by the walking command entity).
    
    Args:
        env: The environment instance.
        tracking_sigma: Controls the decay rate of the exponential reward.
        error_scale: Scaling factors for [radial, polar, azimuthal] errors.
        target_spherical_pos: Target position in spherical coordinates (r, θ, φ).
        walking_cmd_entity_cfg: Entity that provides walking command mask.
        sphere_center_cfg: Configuration for the spherical coordinate center entity.
        ee_cfg: Configuration for the end-effector entity.
        base_cfg: Configuration for the base entity (for coordinate rotation).
        return_error: If True, returns a tuple (reward, error); otherwise only reward.
        
    Returns:
        The computed tracking reward (masked by walking state), or (reward, error) if return_error=True.
    """
    # Extract scene entities
    ee_entity = env.scene[ee_cfg.name]
    base_entity = env.scene[base_cfg.name]
    center_entity = env.scene[sphere_center_cfg.name]
    cmd_entity = env.scene[walking_cmd_entity_cfg.name]
    
    # Get positions and orientations
    ee_pos = ee_entity.data.pos_w
    center_pos = center_entity.data.pos_w
    base_quat = base_entity.data.quat_w
    
    # Compute vector from center to EE and rotate to base frame
    center_to_ee = ee_pos - center_pos
    center_to_ee_local = math_utils.quat_rotate_inverse(base_quat, center_to_ee)
    
    # Convert Cartesian to spherical coordinates
    r, theta, phi = math_utils.cartesian_to_spherical(center_to_ee_local)
    current_spherical = torch.stack([r, theta, phi], dim=-1)
    
    # Compute absolute errors and scale
    abs_errors = torch.abs(current_spherical - target_spherical_pos.to(ee_pos.device))
    scaled_errors = abs_errors * error_scale.to(ee_pos.device)
    total_error = torch.sum(scaled_errors, dim=-1)
    
    # Compute exponential reward
    reward = torch.exp(-total_error / tracking_sigma)
    
    # Apply walking mask - assume cmd_entity has a 'mask' attribute
    # This could be a binary mask or a probability mask
    if hasattr(cmd_entity.data, 'mask'):
        walking_mask = cmd_entity.data.mask
    else:
        # Fallback: check if the command is non-zero
        walking_mask = torch.norm(cmd_entity.data.command[:, :2], dim=1) > 0.1
    
    # Apply mask to reward and error
    reward = reward * walking_mask
    total_error = total_error * walking_mask
    
    # Return based on flag
    if return_error:
        return reward, total_error
    return reward

def tracking_ee_sphere_standing(
    env: ManagerBasedRLEnv,
    tracking_sigma: float,
    error_scale: torch.Tensor,
    target_spherical_pos: torch.Tensor,
    walking_cmd_entity_cfg: SceneEntityCfg = SceneEntityCfg("command"),
    sphere_center_cfg: SceneEntityCfg = SceneEntityCfg("target_entity"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot.ee_link"),
    base_cfg: SceneEntityCfg = SceneEntityCfg("robot.base_link"),
    return_error: bool = False,
) -> torch.Tensor:
    """Reward for tracking end-effector position in spherical coordinates during standing.
    
    Computes an exponential reward based on EE position tracking, but only applied
    when the robot is in standing mode (not walking).
    
    Args:
        env: The environment instance.
        tracking_sigma: Controls the decay rate of the exponential reward.
        error_scale: Scaling factors for [radial, polar, azimuthal] errors.
        target_spherical_pos: Target position in spherical coordinates (r, θ, φ).
        walking_cmd_entity_cfg: Entity that provides walking command mask.
        sphere_center_cfg: Configuration for the spherical coordinate center entity.
        ee_cfg: Configuration for the end-effector entity.
        base_cfg: Configuration for the base entity (for coordinate rotation).
        return_error: If True, returns a tuple (reward, error); otherwise only reward.
        
    Returns:
        The computed tracking reward (masked by standing state), or (reward, error) if return_error=True.
    """
    # Extract scene entities
    ee_entity = env.scene[ee_cfg.name]
    base_entity = env.scene[base_cfg.name]
    center_entity = env.scene[sphere_center_cfg.name]
    cmd_entity = env.scene[walking_cmd_entity_cfg.name]
    
    # Get positions and orientations
    ee_pos = ee_entity.data.pos_w
    center_pos = center_entity.data.pos_w
    base_quat = base_entity.data.quat_w
    
    # Compute vector from center to EE and rotate to base frame
    center_to_ee = ee_pos - center_pos
    center_to_ee_local = math_utils.quat_rotate_inverse(base_quat, center_to_ee)
    
    # Convert Cartesian to spherical coordinates
    r, theta, phi = math_utils.cartesian_to_spherical(center_to_ee_local)
    current_spherical = torch.stack([r, theta, phi], dim=-1)
    
    # Compute absolute errors and scale
    abs_errors = torch.abs(current_spherical - target_spherical_pos.to(ee_pos.device))
    scaled_errors = abs_errors * error_scale.to(ee_pos.device)
    total_error = torch.sum(scaled_errors, dim=-1)
    
    # Compute exponential reward
    reward = torch.exp(-total_error / tracking_sigma)
    
    # Get walking mask and create standing mask (inverse)
    if hasattr(cmd_entity.data, 'mask'):
        walking_mask = cmd_entity.data.mask
    else:
        # Fallback: check if the command is non-zero
        walking_mask = torch.norm(cmd_entity.data.command[:, :2], dim=1) > 0.1
    
    standing_mask = ~walking_mask
    
    # Apply standing mask to reward and error
    reward = reward * standing_mask
    total_error = total_error * standing_mask
    
    # Return based on flag
    if return_error:
        return reward, total_error
    return reward


def tracking_ee_cartesian(
    env: ManagerBasedRLEnv,
    tracking_sigma: float,
    base_quat: torch.Tensor,  # 基座朝向四元数
    spherical_center: torch.Tensor,  # 球坐标中心点
    cartesian_offset: torch.Tensor,  # 局部笛卡尔偏移量
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot.ee_link"),
    return_error: bool = False,
) -> torch.Tensor:
    """Reward for tracking end-effector position in Cartesian coordinates relative to a base frame.
    
    Computes the target position by:
        1. Applying base rotation to the cartesian offset
        2. Adding to the spherical center position
    Then computes L1 error between current EE position and this target.
    
    Args:
        env: The environment instance.
        tracking_sigma: Controls the decay rate of the exponential reward.
        base_quat: Base orientation quaternion [w, x, y, z] - [num_envs, 4]
        spherical_center: Center point in world coordinates - [num_envs, 3]
        cartesian_offset: Local Cartesian offset relative to base frame - [num_envs, 3]
        ee_cfg: Configuration for the end-effector entity.
        return_error: If True, returns a tuple (reward, error); otherwise only reward.
        
    Returns:
        The computed exponential tracking reward, or (reward, error) if return_error=True.
    """
    # Extract end-effector entity
    ee_entity = env.scene[ee_cfg.name]
    ee_pos = ee_entity.data.pos_w  # [num_envs, 3]
    
    # Ensure inputs have correct shape and device
    base_quat = base_quat.to(ee_pos.device)
    spherical_center = spherical_center.to(ee_pos.device)
    cartesian_offset = cartesian_offset.to(ee_pos.device)
    
    # Apply base rotation to cartesian offset
    rotated_offset = math_utils.quat_apply(base_quat, cartesian_offset)
    
    # Compute target position in world frame
    target_ee = spherical_center + rotated_offset
    
    # Compute L1 position error
    ee_pos_error = torch.sum(torch.abs(ee_pos - target_ee), dim=1)
    
    # Compute exponential reward
    reward = torch.exp(-ee_pos_error / tracking_sigma)
    
    # Return based on flag
    if return_error:
        return reward, ee_pos_error
    return reward


def tracking_ee_orientation(
    env: ManagerBasedRLEnv,
    tracking_sigma: float,
    target_euler: torch.Tensor,
    error_scale: torch.Tensor = None,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot.ee_link"),
    return_error: bool = False,
) -> torch.Tensor:
    """Reward for tracking end-effector orientation.
    
    Computes an exponential reward based on the difference between current EE
    orientation and target orientation in Euler angles.
    
    Args:
        env: The environment instance.
        tracking_sigma: Controls the decay rate of the exponential reward.
        target_euler: Target orientation in Euler angles (roll, pitch, yaw) [radians].
        error_scale: Scaling factors for each Euler angle component (default=[1.0, 1.0, 1.0]).
        ee_cfg: Configuration for the end-effector entity.
        return_error: If True, returns a tuple (reward, error); otherwise only reward.
        
    Returns:
        The computed exponential tracking reward, or (reward, error) if return_error=True.
    """
    # Extract end-effector entity
    ee_entity = env.scene[ee_cfg.name]
    
    # Get current end-effector orientation (quaternion)
    ee_quat = ee_entity.data.quat_w  # [num_envs, 4] (w, x, y, z)
    
    # Convert current orientation to Euler angles
    ee_euler = math_utils.quat_to_euler_angles(ee_quat, convention="XYZ")  # [num_envs, 3]
    
    # Ensure target_euler has correct shape
    if target_euler.ndim == 1:
        target_euler = target_euler.unsqueeze(0).repeat(ee_euler.shape[0], 1)
    target_euler = target_euler.to(ee_euler.device)
    
    # Set default error scaling
    if error_scale is None:
        error_scale = torch.tensor([1.0, 1.0, 1.0])
    if error_scale.ndim == 1:
        error_scale = error_scale.unsqueeze(0).repeat(ee_euler.shape[0], 1)
    error_scale = error_scale.to(ee_euler.device)
    
    # Compute angular differences with wrapping to [-π, π]
    angular_diff = ee_euler - target_euler
    
    # Wrap angles to [-π, π]
    angular_diff_wrapped = math_utils.wrap_to_pi_minuspi(angular_diff)
    
    # Compute absolute errors and scale
    abs_errors = torch.abs(angular_diff_wrapped)
    scaled_errors = abs_errors * error_scale
    total_error = torch.sum(scaled_errors, dim=1)
    
    # Compute exponential reward
    reward = torch.exp(-total_error / tracking_sigma)
    
    # Return based on flag
    if return_error:
        return reward, total_error
    return reward


def arm_energy_abs_sum(
    env: ManagerBasedRLEnv,
    arm_joint_cfg: SceneEntityCfg = SceneEntityCfg("robot.arm_joints"),
    gripper_joint_cfg: SceneEntityCfg = SceneEntityCfg("robot.gripper_joints"),
    return_energy: bool = False,
) -> torch.Tensor:
    """Penalty for absolute mechanical energy consumption of the arm joints.
    
    Computes the instantaneous absolute power consumption for each arm joint as:
        power = |torque * velocity|
    Then sums over all arm joints.
    
    Note: This excludes gripper joints and any other non-arm joints.
    
    Args:
        env: The environment instance.
        arm_joint_cfg: Configuration for the arm joints.
        gripper_joint_cfg: Configuration for the gripper joints (to exclude).
        return_energy: If True, returns a tuple (penalty, energy); otherwise only penalty.
        
    Returns:
        The computed energy penalty (sum of absolute joint powers), 
        or (penalty, energy) if return_energy=True.
    """
    # Extract joint entities
    arm_joints = env.scene[arm_joint_cfg.name]
    gripper_joints = env.scene[gripper_joint_cfg.name]
    
    # Get joint torques and velocities for ARM joints only
    # Note: We assume arm_joints contains ONLY the joints we want to consider
    arm_torques = arm_joints.data.applied_torque
    arm_velocities = arm_joints.data.joint_vel
    
    # Compute instantaneous power for each joint: |torque * velocity|
    joint_power = torch.abs(arm_torques * arm_velocities)
    
    # Sum power across all arm joints
    total_power = torch.sum(joint_power, dim=1)
    
    # Return based on flag
    if return_energy:
        return total_power, total_power
    return total_power

def tracking_ee_orientation_roll_yaw(
    env: ManagerBasedRLEnv,
    tracking_sigma: float,
    target_euler: torch.Tensor,
    error_scale: torch.Tensor = None,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot.ee_link"),
    return_error: bool = False,
) -> torch.Tensor:
    """Reward for tracking end-effector roll and yaw orientation.
    
    Computes an exponential reward based on the difference between current EE
    orientation and target orientation in Euler angles, considering only roll and yaw.
    
    Args:
        env: The environment instance.
        tracking_sigma: Controls the decay rate of the exponential reward.
        target_euler: Target orientation in Euler angles (roll, pitch, yaw) [radians].
        error_scale: Scaling factors for each Euler angle component (default=[1.0, 0.0, 1.0]).
        ee_cfg: Configuration for the end-effector entity.
        return_error: If True, returns a tuple (reward, error); otherwise only reward.
        
    Returns:
        The computed exponential tracking reward, or (reward, error) if return_error=True.
    """
    # Extract end-effector entity
    ee_entity = env.scene[ee_cfg.name]
    
    # Get current end-effector orientation (quaternion)
    ee_quat = ee_entity.data.quat_w  # [num_envs, 4] (w, x, y, z)
    
    # Convert current orientation to Euler angles
    ee_euler = math_utils.quat_to_euler_angles(ee_quat, convention="XYZ")  # [num_envs, 3]
    
    # Ensure target_euler has correct shape
    if target_euler.ndim == 1:
        target_euler = target_euler.unsqueeze(0).repeat(ee_euler.shape[0], 1)
    target_euler = target_euler.to(ee_euler.device)
    
    # Set default error scaling (only roll and yaw)
    if error_scale is None:
        error_scale = torch.tensor([1.0, 0.0, 1.0])  # 默认只考虑滚转和偏航
    if error_scale.ndim == 1:
        error_scale = error_scale.unsqueeze(0).repeat(ee_euler.shape[0], 1)
    error_scale = error_scale.to(ee_euler.device)
    
    # Compute angular differences with wrapping to [-π, π]
    angular_diff = ee_euler - target_euler
    angular_diff_wrapped = math_utils.wrap_to_pi_minuspi(angular_diff)
    
    # Apply error scaling and select only roll (index 0) and yaw (index 2)
    scaled_errors = torch.abs(angular_diff_wrapped) * error_scale
    roll_yaw_errors = scaled_errors[:, [0, 2]]  # 只取滚转和偏航分量
    
    # Compute total error for roll and yaw
    total_error = torch.sum(roll_yaw_errors, dim=1)
    
    # Compute exponential reward
    reward = torch.exp(-total_error / tracking_sigma)
    
    # Return based on flag
    if return_error:
        return reward, total_error
    return reward