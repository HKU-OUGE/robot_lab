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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from typing import Optional
import robot_lab.tasks.locomotion.velocity.mdp as mdp
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ==============================================================================
# Existing Functions (Preserved)
# ==============================================================================

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_lin_vel_x_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd_b = env.command_manager.get_command(command_name)[:, :2]
    cmd_b3 = torch.cat([cmd_b, torch.zeros_like(cmd_b[:, :1])], dim=1)
    quat_w = asset.data.root_link_quat_w
    cmd_w3 = math_utils.quat_apply_yaw(quat_w, cmd_b3)
    v_cmd_x_w = cmd_w3[:, 0]
    v_x_w = asset.data.root_com_lin_vel_w[:, 0]
    err = (v_cmd_x_w - v_x_w).pow(2)
    rew = torch.exp(-err / (std ** 2))
    return rew

def track_ang_vel_z_base_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    return reward

def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_action_l2(env: ManagerBasedRLEnv, wheel_ids: list[int]) -> torch.Tensor:
    actions = env.action_manager.action[:, wheel_ids]
    return torch.sum(actions**2, dim=1)

def wheel_slip_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.1,
    epsilon: float = 0.10,
    vel_body_frame: bool = True,
    command_name: Optional[str] = None,
    no_cmd_lin_thresh: float = 0.05,
    no_cmd_ang_thresh: float = 0.05,
    gate_on_no_command: bool = False,
    contact_sensor_cfg: Optional[SceneEntityCfg] = None,
    contact_threshold: float = 1.0,
    gate_on_contact: bool = False
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    omega = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if vel_body_frame and hasattr(asset.data, "root_lin_vel_b"):
        v = asset.data.root_lin_vel_b
        v_x = v[:, 0]
    elif hasattr(asset.data, "root_lin_vel_w"):
        v_x = asset.data.root_lin_vel_w[:, 0]
    else:
        return torch.zeros(omega.shape[0], device=omega.device, dtype=omega.dtype)

    slip = (omega * wheel_radius - v_x.unsqueeze(-1)) / (torch.abs(v_x).unsqueeze(-1) + epsilon)
    penalty = torch.mean(torch.abs(slip), dim=1)

    if gate_on_no_command and (command_name is not None):
        cmd = env.command_manager.get_command(command_name)
        lin_cmd = cmd[:, :2]
        ang_cmd = cmd[:, 2] if cmd.shape[1] >= 3 else torch.zeros_like(lin_cmd[:, 0])
        no_lin = torch.norm(lin_cmd, dim=1) < no_cmd_lin_thresh
        no_ang = torch.abs(ang_cmd) < no_cmd_ang_thresh
        no_cmd_mask = (no_lin & no_ang).to(penalty.dtype)
        penalty = penalty * no_cmd_mask

    if gate_on_contact and (contact_sensor_cfg is not None) and (contact_sensor_cfg.name in env.scene):
        cf = env.scene[contact_sensor_cfg.name].data.net_forces_w
        if cf.ndim == 3:
            contact_mask = (cf[..., 2].abs().max(dim=1).values > contact_threshold)
        else:
            contact_mask = (cf.abs().max(dim=1).values > contact_threshold)
        penalty = penalty * contact_mask.to(penalty.dtype)

    return penalty

def joint_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return torch.where(
        torch.logical_or(cmd > 0.1, body_vel > velocity_threshold), reward, stand_still_scale * reward
    )

def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward

class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
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
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.1, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7) * torch.where(torch.abs(torch.atan2(env.scene["robot"].data.projected_gravity_b[:, 0], -env.scene["robot"].data.projected_gravity_b[:, 2])) > 0.3490658503988659, 0.1*torch.ones_like(reward), torch.ones_like(reward))
    return reward

def wheel_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "wheel_mirror_joints_cache") or env.wheel_mirror_joints_cache is None:
        env.wheel_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    tau = 0.3
    small_scale = 0.25
    reward = torch.zeros(env.num_envs, device=env.device)
    per_pair_terms = []
    for joint_pair in env.wheel_mirror_joints_cache:
        left  = asset.data.joint_vel[:, joint_pair[0][0]]
        right = asset.data.joint_vel[:, joint_pair[1][0]]
        d = torch.abs(left - right)
        d_norm = d / 30.0
        scale = 50.0
        diff = -(scale * (d_norm ** 2))
        if diff.ndim > 1:
            diff = diff.sum(dim=-1)
        per_pair_terms.append(diff)
    if len(per_pair_terms) > 0:
        terms = torch.stack(per_pair_terms, dim=-1)
        reward = terms.sum(dim=-1)
    else:
        reward = torch.zeros(env.num_envs, device=env.device)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def action_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]
    penalty = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.action_mirror_joints_cache:
        diff = torch.abs(env.action_manager.action[:, joint_pair[0][0]]) - \
               torch.abs(env.action_manager.action[:, joint_pair[1][0]])
        penalty += torch.mean(torch.square(diff), dim=-1)
    penalty = penalty / (len(mirror_joints) + 1e-6)
    reward = torch.exp(-5.0 * penalty)
    upright_weight = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    reward *= upright_weight
    return reward

def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )
        mean_actions = torch.mean(actions, dim=1, keepdim=True)
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_continue_contact(env, command_name, expect_contact_num, sensor_cfg) -> torch.Tensor:
    s = env.scene.sensors[sensor_cfg.name]
    forces = s.data.net_forces_w
    contact = (forces[:, sensor_cfg.body_ids, 2].abs() > 20).float()
    if not hasattr(env, "contact_ema"):
        env.contact_ema = torch.zeros_like(contact)
    alpha = 0.1
    env.contact_ema = (1.0 - alpha) * env.contact_ema + alpha * contact
    target_duty = 0.85
    slack = 0.05
    duty_ok = (env.contact_ema >= (target_duty - slack)).float()
    reward = duty_ok.mean(dim=1)
    cmd = env.command_manager.get_command(command_name)
    v_ref = 1.0
    w = (cmd[:, 0:2].norm(dim=1) / (v_ref + 1e-6)).clamp(0.2, 1.0)
    return reward * w

def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    return torch.any(contact & (forces_xy > forces_z), dim=1)

def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)
    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
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
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return torch.exp(-reward / std)

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward

def track_lin_vel_world_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_w[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_world_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

def feet_height_body_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty_height_gated(
    env,
    command_name: str,
    asset_cfg,
    sensor_cfg,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
    h_free_min: float = 0.10,
    h_free_max: float = 0.45,
    offset: float = 0.5,
    alpha: float = 0.2,
    tilt_floor: float = 0.1,
):
    import math
    asset = env.scene[asset_cfg.name]
    pos_err = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1
    )
    H = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    terrain_delta = -H
    if terrain_delta.ndim == 2 and terrain_delta.shape[1] > 1:
        mag = terrain_delta.abs().max(dim=1).values
    else:
        mag = terrain_delta.abs().squeeze(-1)
    gate = torch.clamp((mag - h_free_min) / max(1e-6, (h_free_max - h_free_min)), 0.0, 1.0)
    if not hasattr(env, "pose_gate_ema"):
        env.pose_gate_ema = gate
    env.pose_gate_ema = (1.0 - alpha) * env.pose_gate_ema + alpha * gate
    g = env.pose_gate_ema
    s = stand_still_scale - (stand_still_scale - 1.0) * g
    g_b = asset.data.projected_gravity_b
    pitch = torch.atan2(g_b[:, 0].abs(), (-g_b[:, 2]).clamp_min(1e-6))
    lo = math.radians(10.0)
    hi = math.radians(30.0)
    t = ((pitch - lo) / (hi - lo)).clamp(0.0, 1.0)
    grav = 1.0 - t * (1.0 - tilt_floor)
    scale = s * grav
    return pos_err * scale

def flat_orientation_height_gated(
    env,
    sensor_cfg=None,
    h_low: float = 0.10,
    h_high: float = 0.25,
    encourage_scale: float = 0.5,
    use_disc: bool = True,
    offset: float = 0.5,
    alpha: float = 0.2,
    tilt_cap_rad: float = 0.35,
):
    import math
    robot = env.scene["robot"]
    g = robot.data.projected_gravity_b
    tilt_xy = torch.sqrt(torch.clamp(g[:, 0]**2 + g[:, 1]**2, min=1e-9))
    upright_pen = tilt_xy**2
    hazard = None
    if isinstance(getattr(sensor_cfg, "name", None), str):
        try:
            _ = env.scene[sensor_cfg.name]
            if use_disc:
                bins = mdp.height_scan_disc(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)
                hazard = bins.abs().clamp(0.0, 1.0)
            else:
                h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)
                hazard = h.abs()
        except KeyError:
            pass
    if hazard is None:
        gate = torch.zeros_like(upright_pen)
    else:
        gate_lin = torch.clamp((hazard - h_low) / max(1e-6, (h_high - h_low)), 0.0, 1.0)
        gate = gate_lin * gate_lin * (3.0 - 2.0 * gate_lin)
    if not hasattr(env, "_tilt_gate_ema"):
        env._tilt_gate_ema = gate
    else:
        env._tilt_gate_ema = (1.0 - alpha) * env._tilt_gate_ema + alpha * gate
    gate_smooth = env._tilt_gate_ema
    tilt_cap = math.sin(tilt_cap_rad)
    encouraged_tilt = torch.clamp(tilt_xy, max=tilt_cap)
    return (1.0 - gate_smooth) * upright_pen - encourage_scale * gate_smooth * encouraged_tilt

def upright_gate(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return upright

def obstacle_gate(env, sensor_cfg, h_low=0.10, h_high=0.25, offset=0.5):
    h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    h = h.abs().max(dim=1).values
    gate = torch.clamp((h - h_low) / (h_high - h_low + 1e-6), 0.0, 1.0)
    return gate

def upward_for_climb(env, asset_cfg=SceneEntityCfg("robot"),
                     sensor_cfg=None, k_progress=0.5, relax_scale=0.3):
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        gate = obstacle_gate(env, sensor_cfg)
    else:
        gate = torch.zeros(asset.data.root_pos_w.shape[0], device=asset.device)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    pen_tilt = (1.0 - upright)**2
    pen_tilt *= (1.0 - gate + gate * relax_scale)
    z = asset.data.root_pos_w[:, 2]
    if not hasattr(env, "_last_z"): env._last_z = z.clone()
    dz = torch.clamp(z - env._last_z, min=0.0)
    env._last_z = z
    r_progress = k_progress * dz * gate
    return r_progress - pen_tilt

def climb_progress_dyn_pbrs(env, asset_cfg=SceneEntityCfg("robot"),
                            sensor_cfg=None, k: float = 2.0, gamma_shape: float = 0.99,
                            h_low: float = 0.10, h_high: float = 0.25, offset: float = 0.5):
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]
    if sensor_cfg is not None:
        h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
        h = h.abs().max(dim=1).values
        alpha = torch.clamp((h - h_low) / (h_high - h_low + 1e-6), 0.0, 1.0)
        g = alpha
    else:
        g = torch.ones_like(z)
    phi = k * g * z
    if not hasattr(env, "_phi_prev"):
        env._phi_prev = phi.clone()
    if hasattr(env, "reset_buf"):
        env._phi_prev = torch.where(env.reset_buf.bool(), phi, env._phi_prev)
    rew = gamma_shape * phi - env._phi_prev
    env._phi_prev = phi
    return rew

def _get_gait(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "gait_mode"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.gait_mode

def _get_gait_mask(env: ManagerBasedRLEnv, gait_mode: int) -> torch.Tensor:
    g = _get_gait(env).squeeze(-1)
    return g if gait_mode == 1 else (1.0 - g)

def is_alive_gated(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def track_lin_vel_xy_gated(env: ManagerBasedRLEnv, std: float, command_name: str, gait_mode: int) -> torch.Tensor:
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(vel_cmd - lin_vel), dim=1)
    return torch.exp(-lin_vel_error / (std**2)) * _get_gait_mask(env, gait_mode)

def track_ang_vel_z_gated(env: ManagerBasedRLEnv, std: float, command_name: str, gait_mode: int) -> torch.Tensor:
    vel_cmd = env.command_manager.get_command(command_name)[:, 2]
    ang_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    ang_vel_error = torch.square(vel_cmd - ang_vel)
    return torch.exp(-ang_vel_error / (std**2)) * _get_gait_mask(env, gait_mode)

def joint_pos_penalty_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    robot = env.scene["robot"]
    diff = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff), dim=1)
    return reward * _get_gait_mask(env, gait_mode)

def joint_vel_penalty_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    robot = env.scene["robot"]
    vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    reward = torch.norm(vel, dim=1)
    return reward * _get_gait_mask(env, gait_mode)

def feet_in_air_quad(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2]
    in_contact = torch.abs(forces) > 1.0
    feet_contact = in_contact[:, :4]
    knee_contact = in_contact[:, 4:]
    feet_in_air = ~feet_contact
    all_feet_air = torch.all(feet_in_air, dim=1).float()
    kneeling = torch.sum(feet_in_air.float() * knee_contact.float(), dim=1)
    reward = all_feet_air + kneeling
    return reward * _get_gait_mask(env, 0)

def joint_deviation_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    robot = env.scene["robot"]
    diff = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(diff), dim=1) * _get_gait_mask(env, gait_mode)

def base_height_quad(
    env: ManagerBasedRLEnv, 
    target_height: float, 
    asset_cfg: SceneEntityCfg, 
    sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        ground_height = torch.mean(ray_hits_z, dim=1)
        current_height = robot.data.root_pos_w[:, 2] - ground_height
    else:
        current_height = robot.data.root_pos_w[:, 2]
    error = torch.square(current_height - target_height)
    return error * _get_gait_mask(env, 0)

def balance_quad(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    diagonal_1 = forces[:, 0] + forces[:, 3]
    diagonal_2 = forces[:, 2] + forces[:, 1]
    return torch.abs(diagonal_1 - diagonal_2) * _get_gait_mask(env, 0)

def joint_limit_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    out_of_limits = (joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])
    return torch.sum(out_of_limits.float(), dim=1) * _get_gait_mask(env, gait_mode)

def rear_air_biped(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2]
    in_contact = torch.abs(forces) > 1.0
    rear_feet_air = ~in_contact[:, :2]
    rear_knee_contact = in_contact[:, 2:]
    all_air = torch.all(rear_feet_air, dim=1).float()
    kneeling = torch.sum(rear_feet_air.float() * rear_knee_contact.float(), dim=1)
    return (all_air + kneeling) * _get_gait_mask(env, 1)

def rear_pos_balance_biped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene["robot"]
    n = len(asset_cfg.joint_ids) // 2
    left = robot.data.joint_pos[:, asset_cfg.joint_ids[:n]]
    right = robot.data.joint_pos[:, asset_cfg.joint_ids[n:]]
    return torch.norm(left - right, dim=1) * _get_gait_mask(env, 1)

def joint_power_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    power = torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids] * robot.data.applied_torque[:, asset_cfg.joint_ids])
    return torch.sum(power, dim=1) * _get_gait_mask(env, gait_mode)

def biped_stand_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity_vec)
    cos_theta = -projected_gravity[:, 2]
    r = torch.square(0.5 * cos_theta + 0.5)
    return r * _get_gait_mask(env, 1)

def biped_stand_height_linear(
    env: ManagerBasedRLEnv, 
    target_height: float = 0.55, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        ground_height = torch.mean(ray_hits_z, dim=1)
        current_height = robot.data.root_pos_w[:, 2] - ground_height
    else:
        current_height = robot.data.root_pos_w[:, 2]
    z_min = target_height - 0.15
    z_max = target_height + 0.05
    r = (current_height - z_min) / (z_max - z_min)
    r = torch.clamp(r, 0.0, 1.0)
    return r * _get_gait_mask(env, 1)

def biped_front_legs_lift(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    time_threshold_s: float = 0.0
) -> torch.Tensor:
    """
    Penalize joint position error for front legs (arms) in biped mode.
    Includes optional time gating: 1{t > T_allow}
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    target_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(joint_pos - target_pos), dim=1)
    
    # Time gating
    if time_threshold_s > 0.0 and hasattr(env, "episode_length_buf"):
        current_time = env.episode_length_buf * env.step_dt
        time_gate = (current_time > time_threshold_s).float()
        reward *= time_gate

    return reward * _get_gait_mask(env, 1)

def track_lin_vel_xy_biped_gated(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    target_height: float = 0.55
) -> torch.Tensor:
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(vel_cmd - lin_vel), dim=1)
    base_reward = torch.exp(-lin_vel_error / (std**2))
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    proj_g = quat_apply_inverse(env.scene["robot"].data.root_quat_w, gravity_vec)
    upright = -proj_g[:, 2]
    posture_gate = (upright > 0.95).float()
    root_z = env.scene["robot"].data.root_pos_w[:, 2]
    z_min = target_height - 0.15
    z_max = target_height + 0.05
    height_coef = (root_z - z_min) / (z_max - z_min)
    height_coef = torch.clamp(height_coef, 0.0, 1.0)
    return base_reward * posture_gate * height_coef * _get_gait_mask(env, 1)

def torque_exceed_limit(env: ManagerBasedRLEnv, limit_ratio: float = 0.9) -> torch.Tensor:
    robot = env.scene["robot"]
    torques = torch.abs(robot.data.applied_torque)
    if not hasattr(env, "_max_torque_tensor"):
        max_limits = torch.full((16,), 40.0, device=env.device)
        knee_indices = [2, 5, 8, 11]
        max_limits[knee_indices] = 100.0
        env._max_torque_tensor = max_limits
    threshold = env._max_torque_tensor * limit_ratio
    excess = torch.relu(torques - threshold)
    return torch.sum(excess, dim=1)

def ang_vel_xy_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    ang_vel = env.scene["robot"].data.root_ang_vel_b
    return (torch.abs(ang_vel[:, 0]) + torch.abs(ang_vel[:, 1]))

def set_gait_mode_fixed(env: ManagerBasedRLEnv, env_ids: torch.Tensor, mode_val: float = 0.0):
    if not hasattr(env, "gait_mode"):
        env.gait_mode = torch.zeros(env.num_envs, 1, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env.gait_mode[env_ids] = mode_val

def set_gait_mode_random(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    if not hasattr(env, "gait_mode"):
        env.gait_mode = torch.zeros(env.num_envs, 1, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_selected = env_ids.shape[0]
    mode = torch.randint(0, 2, (num_selected, 1), device=env.device).float()
    env.gait_mode[env_ids] = mode

def set_gait_mode_flip(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    if not hasattr(env, "gait_mode"):
        env.gait_mode = torch.zeros(env.num_envs, 1, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env.gait_mode[env_ids] = 1.0 - env.gait_mode[env_ids]

def gait_mode_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "gait_mode"):
        env.gait_mode = torch.zeros(env.num_envs, 1, device=env.device)
    return env.gait_mode

# ==============================================================================
# New / Modified Functions for Alignment
# ==============================================================================

def track_ang_vel_z_biped_gated(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    target_height: float = 0.55
) -> torch.Tensor:
    """
    Tracking angular velocity for biped with posture and height gating.
    Matches formula: exp(...) * I(cos>0.95) * (z - low)/(high - low)
    """
    # 1. Base tracking
    vel_cmd = env.command_manager.get_command(command_name)[:, 2]
    ang_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    ang_vel_error = torch.square(vel_cmd - ang_vel)
    base_reward = torch.exp(-ang_vel_error / (std**2))
    
    # 2. Posture gate (cos_theta > 0.95)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    proj_g = quat_apply_inverse(env.scene["robot"].data.root_quat_w, gravity_vec)
    upright = -proj_g[:, 2]
    posture_gate = (upright > 0.95).float()
    
    # 3. Height gate
    root_z = env.scene["robot"].data.root_pos_w[:, 2]
    z_min = target_height - 0.15 
    z_max = target_height + 0.05
    height_coef = (root_z - z_min) / (z_max - z_min)
    height_coef = torch.clamp(height_coef, 0.0, 1.0)
    
    return base_reward * posture_gate * height_coef * _get_gait_mask(env, 1)

def joint_acc_l2_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """
    Gated joint acceleration penalty (L2 squared). Kept for legacy compatibility if needed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1) * _get_gait_mask(env, gait_mode)

def joint_acc_norm_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """
    Gated joint acceleration penalty (L2 Norm).
    Matches formula: || q_ddot ||_2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    # L2 Norm: sqrt(sum(acc^2))
    return torch.norm(acc, dim=1) * _get_gait_mask(env, gait_mode)

def joint_limit_l1_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """
    Joint limit penalty (L1 Norm of violation).
    Matches formula: || out_of_limits ||_1 (sum of magnitudes)
    """
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    
    # Compute excess (L1)
    lower_exceed = torch.relu(limits[..., 0] - joint_pos)
    upper_exceed = torch.relu(joint_pos - limits[..., 1])
    
    violation_magnitude = torch.sum(lower_exceed + upper_exceed, dim=1)
    
    return violation_magnitude * _get_gait_mask(env, gait_mode)

def action_rate_norm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Action rate penalty (L2 Norm).
    Matches formula: || a_t - a_{t-1} ||_2
    """
    # Requires action manager to have prev_action. Standard Isaac Lab managers usually track this.
    curr_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    if prev_action is None:
        return torch.zeros(env.num_envs, device=env.device)
        
    diff = curr_action - prev_action
    return torch.norm(diff, dim=1)

def undesired_contacts_count(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """
    Count of undesired contacts exceeding threshold.
    Matches formula: Sum( Indicator( ||f|| > 1 ) )
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Net forces in world frame: [env, body, 3]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_magnitudes = torch.norm(forces, dim=-1) # [env, body]
    
    # Indicator function
    is_contact = (force_magnitudes > threshold).float()
    
    # Sum over bodies (dim=1)
    return torch.sum(is_contact, dim=1)

def biped_front_joint_vel(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    gait_mode: int,
    time_threshold_s: float = 0.0
) -> torch.Tensor:
    """
    Penalize joint velocity for front legs in biped mode (Squared L2).
    Includes optional time gating: 1{t > T_allow}
    """
    robot = env.scene[asset_cfg.name]
    # Formula: sum(q_dot^2)
    vel_sq = torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids])
    reward = torch.sum(vel_sq, dim=1)
    
    # Time gating
    if time_threshold_s > 0.0 and hasattr(env, "episode_length_buf"):
        current_time = env.episode_length_buf * env.step_dt
        time_gate = (current_time > time_threshold_s).float()
        reward *= time_gate
        
    return reward * _get_gait_mask(env, gait_mode)

def biped_front_joint_acc(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    gait_mode: int,
    time_threshold_s: float = 0.0
) -> torch.Tensor:
    """
    Penalize joint acceleration for front legs in biped mode (Squared L2).
    Includes optional time gating: 1{t > T_allow}
    """
    robot = env.scene[asset_cfg.name]
    # Formula: sum(q_ddot^2)
    acc_sq = torch.square(robot.data.joint_acc[:, asset_cfg.joint_ids])
    reward = torch.sum(acc_sq, dim=1)
    
    # Time gating
    if time_threshold_s > 0.0 and hasattr(env, "episode_length_buf"):
        current_time = env.episode_length_buf * env.step_dt
        time_gate = (current_time > time_threshold_s).float()
        reward *= time_gate
        
    return reward * _get_gait_mask(env, gait_mode)

# ==============================================================================
# NEW: Rough Terrain Imitation Rewards (Gated)
# ==============================================================================

def lin_vel_z_l2_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """Penalize z-axis base linear velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * _get_gait_mask(env, gait_mode)

def ang_vel_xy_l2_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """Penalize xy-axis base angular velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * _get_gait_mask(env, gait_mode)

def flat_orientation_l2_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * _get_gait_mask(env, gait_mode)

def base_height_l2_gated(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    gait_mode: int,
    sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Penalize base height error."""
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    
    error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return error * _get_gait_mask(env, gait_mode)

def joint_torques_l2_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """Penalize joint torques."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1) * _get_gait_mask(env, gait_mode)

def joint_vel_l2_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """Penalize joint velocities."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * _get_gait_mask(env, gait_mode)

def joint_pos_limits_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, gait_mode: int) -> torch.Tensor:
    """Penalize joint positions outside limits."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    out_of_limits = (joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])
    return torch.sum(out_of_limits.float(), dim=1) * _get_gait_mask(env, gait_mode)

def joint_pos_penalty_complex_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    gait_mode: int,
    stand_still_scale: float = 1.0,
    velocity_threshold: float = 0.1,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint position deviation with stand still scaling."""
    asset = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    # Note: accessing root_lin_vel_b directly if available
    if hasattr(asset.data, "root_lin_vel_b"):
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    else:
        body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
        
    deviation = torch.sum(torch.square(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ), dim=1)
    
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        deviation,
        stand_still_scale * deviation,
    )
    return reward * _get_gait_mask(env, gait_mode)

def action_rate_l2_gated(env: ManagerBasedRLEnv, gait_mode: int) -> torch.Tensor:
    """Penalize action rate."""
    if env.action_manager.prev_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(diff), dim=1) * _get_gait_mask(env, gait_mode)

def undesired_contacts_gated(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    gait_mode: int, 
    threshold: float = 1.0
) -> torch.Tensor:
    """Penalize undesired contacts."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Check if any force on body parts exceeds threshold
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    is_contact = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > threshold
    return is_contact.float() * _get_gait_mask(env, gait_mode)

def contact_forces_gated(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    gait_mode: int, 
    threshold: float = 1.0
) -> torch.Tensor:
    """Penalize high contact forces."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    # Sum of forces exceeding threshold
    penalty = torch.sum(torch.relu(forces - threshold), dim=1)
    return penalty * _get_gait_mask(env, gait_mode)

def feet_air_time_gated(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    gait_mode: int, 
    threshold: float
) -> torch.Tensor:
    """Reward feet air time."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # Only reward when moving
    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward * cmd.float() * _get_gait_mask(env, gait_mode)

def feet_slide_gated(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg, 
    gait_mode: int
) -> torch.Tensor:
    """Penalize feet sliding."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Check contacts
    contacts = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    
    asset = env.scene[asset_cfg.name]
    # Calculate feet velocity in world frame
    body_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_vel_xy = torch.norm(body_vel_w, dim=-1)
    
    reward = torch.sum(foot_vel_xy * contacts.float(), dim=1)
    return reward * _get_gait_mask(env, gait_mode)

def track_lin_vel_xy_exp_gated(
    env: ManagerBasedRLEnv, std: float, command_name: str, gait_mode: int
) -> torch.Tensor:
    """Exponential tracking of linear velocity xy."""
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error = torch.sum(torch.square(vel_cmd - lin_vel), dim=1)
    return torch.exp(-error / (std**2)) * _get_gait_mask(env, gait_mode)

def track_ang_vel_z_exp_gated(
    env: ManagerBasedRLEnv, std: float, command_name: str, gait_mode: int
) -> torch.Tensor:
    """Exponential tracking of angular velocity z."""
    vel_cmd = env.command_manager.get_command(command_name)[:, 2]
    ang_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error = torch.square(vel_cmd - ang_vel)
    return torch.exp(-error / (std**2)) * _get_gait_mask(env, gait_mode)

def feet_contact_without_cmd_gated(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    gait_mode: int
) -> torch.Tensor:
    """Penalize feet contact when no command is given (standing still)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Count contacts
    contact = torch.sum(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs() > 1.0, dim=1).float()
    # Check if command is zero
    no_cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    return contact * no_cmd.float() * _get_gait_mask(env, gait_mode)

def joint_mirror_gated(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    mirror_joints: list[list[str]], 
    gait_mode: int
) -> torch.Tensor:
    """
    Penalize asymmetry in joint positions (Gated).
    Matches the logic in 'rough_env_cfg.py' / 'rewards.py' joint_mirror function,
    including the projected gravity scaling.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Cache the joint indices lookup
    if not hasattr(env, "joint_mirror_joints_cache_gated") or env.joint_mirror_joints_cache_gated is None:
        env.joint_mirror_joints_cache_gated = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
        
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache_gated:
        # Sum of squared differences between left and right joint positions
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
        
    # Normalize by number of pairs
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    
    # Apply gravity scaling (from original implementation)
    # Scaling down reward if robot is tilted heavily or upside down
    proj_g = env.scene["robot"].data.projected_gravity_b
    upright_factor = torch.clamp(-proj_g[:, 2], 0, 0.7) / 0.7
    
    # Additional orientation check (pitch/roll logic from original code)
    # atan2(x, -z) roughly corresponds to pitch if y is small
    angle_check = torch.abs(torch.atan2(proj_g[:, 0], -proj_g[:, 2])) > 0.34906585  # ~20 degrees
    orientation_scale = torch.where(angle_check, 0.1 * torch.ones_like(reward), torch.ones_like(reward))
    
    reward *= upright_factor * orientation_scale
    
    return reward * _get_gait_mask(env, gait_mode)

# ==============================================================================
# Wheel Constraints & Standing Still Rewards
# ==============================================================================

def wheels_stop_without_cmd(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1
) -> torch.Tensor:
    """
    Penalize wheel velocity when the command is small (standing still).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # Check linear velocity commands (xy)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_standing = cmd_norm < command_threshold
    
    # Penalty is sum of absolute velocities
    penalty = torch.sum(torch.abs(wheel_vel), dim=1)
    return penalty * is_standing.float()

def wheels_stop_without_cmd_gated(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    gait_mode: int,
    command_threshold: float = 0.1
) -> torch.Tensor:
    """
    Gated version of wheels_stop_without_cmd. 
    Useful to apply only in Quad mode (gait_mode=0) and NOT in Biped mode (where wheels must move to balance).
    """
    base_penalty = wheels_stop_without_cmd(env, command_name, asset_cfg, command_threshold)
    return base_penalty * _get_gait_mask(env, gait_mode)

def wheel_spin_in_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize wheel rotation if the wheel is in the air.
    Requires a ContactSensor defined on the wheel links.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Check if wheels are in air (no contact)
    # Note: ensure sensor_cfg.body_ids matches the order of asset_cfg.joint_ids logic if mapped 1:1, 
    # but usually we just sum up all "air * vel"
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    
    reward = torch.sum(in_air * joint_vel, dim=1)
    return reward

def wheel_spin_in_air_penalty_gated(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    gait_mode: int
) -> torch.Tensor:
    """Gated version of wheel_spin_in_air_penalty."""
    base_penalty = wheel_spin_in_air_penalty(env, sensor_cfg, asset_cfg)
    return base_penalty * _get_gait_mask(env, gait_mode)