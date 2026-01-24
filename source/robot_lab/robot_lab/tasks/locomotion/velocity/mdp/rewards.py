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
    # 改为 sum，让所有轮子都受到约束
    return torch.sum(torch.square(actions), dim=1)

def wheel_sync_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize the variance of PHYSICAL wheel velocities.
    Robust to Action Scale changes.
    """
    # 1. 获取物理速度 (rad/s)
    # asset_cfg 需在 params 中传入
    wheel_vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    
    # 2. 计算平均物理速度
    mean_vel = torch.mean(wheel_vel, dim=1, keepdim=True)
    
    # 3. 计算方差 (Sum((v_i - v_mean)^2))
    variance = torch.sum(torch.square(wheel_vel - mean_vel), dim=1)
    
    return variance

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
    # gait_mode 1: Biped/Stand, gait_mode 0: Quadruped
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

def joint_pos_penalty_gated(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    gait_mode: int,
    sigma: float = 0.5
) -> torch.Tensor:
    """
    Penalize deviation from default joint positions ONLY in specific gait mode.
    """
    robot = env.scene["robot"]
    diff = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(diff), dim=1)
    return torch.exp(-error / sigma) * _get_gait_mask(env, gait_mode)

def joint_deviation_l1_gated(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    gait_mode: int
) -> torch.Tensor:
    """
    Penalize joint positions that deviate from the default one (L1 Norm), 
    masked by the specific gait mode.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Calculate deviation from default positions
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    # Compute L1 norm (sum of absolute errors)
    l1_error = torch.sum(torch.abs(diff), dim=1)
    
    # Apply gait mask
    return l1_error * _get_gait_mask(env, gait_mode)

def joint_pos_target_l1_gated(
    env: ManagerBasedRLEnv, 
    target_pos_list: list[float], 
    gait_mode: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize deviation from a specific target joint position list, gated by gait mode.
    
    Args:
        target_pos_list: A list of target angles (rad) matching the order of joint_names in asset_cfg.
    """
    # 1. 获取当前关节位置
    asset = env.scene[asset_cfg.name]
    # 注意：这里获取的是 asset_cfg 中指定的 joint_names 的数据
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # 2. 将传入的列表转换为 Tensor
    # 形状扩展为 (num_envs, num_selected_joints)
    target_tensor = torch.tensor(target_pos_list, device=env.device, dtype=torch.float32)
    target_tensor = target_tensor.view(1, -1).repeat(env.num_envs, 1)
    
    # 3. 计算 L1 误差 (绝对值之和 或 平均)
    # 使用 mean 可以避免关节数量不同导致权重需要重新调整，使用 sum 梯度更强
    deviation = torch.sum(torch.abs(current_joint_pos - target_tensor), dim=1)
    
    # 4. 门控输出
    return deviation * _get_gait_mask(env, gait_mode)

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
    """
    Reward for upright orientation in Biped mode.
    Encourages the robot's projected gravity vector to align with the body's negative z-axis.
    """
    robot = env.scene["robot"]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity_vec)
    
    # projected_gravity[:, 2] should be -1.0 (gravity pointing down in body frame)
    # We use -projected_gravity[:, 2] which should be 1.0
    upright_alignment = -projected_gravity[:, 2]
    
    # Use a sharper curve to punish deviation stronger: x^4 or similar, or just high weight
    r = torch.clamp(upright_alignment, min=0.0)
    r = torch.square(r) 
    
    return r * _get_gait_mask(env, 1)

def biped_stand_height_linear(
    env: ManagerBasedRLEnv, 
    target_height: float = 0.55, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """
    Reward for reaching a specific base height in Biped mode.
    """
    robot = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        ground_height = torch.mean(ray_hits_z, dim=1)
        # Handle invalid ray casts
        ground_height = torch.where(torch.abs(ground_height) > 10.0, torch.zeros_like(ground_height), ground_height)
        current_height = robot.data.root_pos_w[:, 2] - ground_height
    else:
        current_height = robot.data.root_pos_w[:, 2]
    
    # Linear reward: 1.0 at target, decreasing to 0.0 at distance
    error = torch.abs(current_height - target_height)
    # Tolerance range
    reward = 1.0 - torch.clamp(error / 0.25, 0.0, 1.0) # Within 25cm
    
    return reward * _get_gait_mask(env, 1)

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
    Includes optional time gating: 1{t > T_allow}from isaaclab.utils.math import yaw_quat, quat_apply_inverse
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
    asset = env.scene[asset_cfg.name]
    
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        
        # 1. 获取机器人基座高度
        robot_z = asset.data.root_pos_w[:, 2]
        
        # 2. 计算 射线点 相对于 基座 的垂直距离
        # dist = RobotZ - RayZ
        # 正常地面：dist ≈ 0.55 (正数)
        # 天花板(梁)：dist ≈ 0.55 - 2.0 = -1.45 (负数)
        # 深坑(Gap)：dist ≈ 0.55 - (-10) = 10.55 (大正数)
        dist_to_base = robot_z.unsqueeze(-1) - ray_hits_z
        
        # 3. 创建过滤器 Mask
        # 条件 A: 不是深坑 (距离基座 < 1.5米) -> 过滤 Gap
        # 条件 B: 不是天花板 (距离基座 > -0.2米) -> 过滤 Beam
        #         (允许地面稍微比基座高一点点，比如上坡，但不能高出 20cm 以上)
        valid_mask = (dist_to_base < 1.5) & (dist_to_base > -0.2)
        
        # 还要过滤无效值 (NaN/Inf)
        valid_mask &= (~torch.isnan(ray_hits_z)) & (~torch.isinf(ray_hits_z))
        
        # 4. 替换无效值
        # 如果射线击中了天花板或深坑，我们就认为该方向“没有有效地面数据”
        # 此时使用默认策略：假设地面在基座下方 target_height 处
        # 这样 Error = 0，不会产生错误的惩罚
        default_ground_z = robot_z - target_height
        
        hits_safe = torch.where(valid_mask, ray_hits_z, default_ground_z.unsqueeze(-1))
        
        # 5. 计算平均地面高度
        ground_height = torch.mean(hits_safe, dim=1)
        
        # 6. 计算相对高度
        current_height = robot_z - ground_height
    else:
        # 无 Sensor 模式 (假设 Z=0 是地面)
        current_height = asset.data.root_pos_w[:, 2]
    
    error = torch.square(current_height - target_height)
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
    """
    Penalize contacts on specified bodies (e.g., front feet/knees) in a specific gait mode.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
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
    # [修复] 使用关键字参数指定 asset_cfg 和 command_threshold，防止位置传参错误
    base_penalty = wheels_stop_without_cmd(
        env, 
        command_name=command_name, 
        asset_cfg=asset_cfg,
        command_threshold=command_threshold      
    )
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

def wheel_vel_smoothness_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    gait_mode: int
) -> torch.Tensor:
    """
    Penalize large changes in wheel velocity to encourage smooth rolling.
    """
    if env.action_manager.prev_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Assuming actions map to velocities for wheels roughly
    # Or directly check joint accel
    robot = env.scene[asset_cfg.name]
    acc = robot.data.joint_acc[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(acc), dim=1) * _get_gait_mask(env, gait_mode)

def feet_contact_gated(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    gait_mode: int,
    threshold: float = 1.0
) -> torch.Tensor:
    """
    Reward having contact on specific feet/wheels in a specific gait mode.
    Useful for ensuring rear wheels stay on the ground in Biped mode.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Check if force magnitude > threshold
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    has_contact = (forces > threshold).float()
    
    # Reward is mean contact of all specified bodies
    return torch.mean(has_contact, dim=1) * _get_gait_mask(env, gait_mode)


def track_lin_vel_xy_biped_soft(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    # target_height: float = 1.0 # 移除高度耦合，让速度奖励专注于速度
) -> torch.Tensor:
    """
    修改版：移除硬截断 Gate。即使倒在地上，只要尝试移动也能获得部分奖励，
    配合 Orientation 奖励共同作用。
    """
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(vel_cmd - lin_vel), dim=1)
    
    # 基础速度奖励
    base_reward = torch.exp(-lin_vel_error / (std**2))
    
    # 可选：加上一个软性的姿态系数，而不是硬截断 (0 or 1)
    # 这样当它趴着时 (upright~0)，奖励很小但不是0；站起来时奖励变大
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    proj_g = quat_apply_inverse(env.scene["robot"].data.root_quat_w, gravity_vec)
    upright = -proj_g[:, 2] # 1.0 is standing
    
    # 使用 ReLU 或软映射，保证趴着时也有微弱信号
    soft_posture = torch.clamp(upright, min=0.0) 
    
    return base_reward * soft_posture * _get_gait_mask(env, 1)

def track_ang_vel_z_biped_soft(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
) -> torch.Tensor:
    """
    修改版：移除硬截断 Gate。
    """
    vel_cmd = env.command_manager.get_command(command_name)[:, 2]
    ang_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    ang_vel_error = torch.square(vel_cmd - ang_vel)
    
    base_reward = torch.exp(-ang_vel_error / (std**2))
    
    # 同样使用软姿态系数
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    proj_g = quat_apply_inverse(env.scene["robot"].data.root_quat_w, gravity_vec)
    upright = torch.clamp(-proj_g[:, 2], min=0.0)

    return base_reward * upright * _get_gait_mask(env, 1)

def biped_stand_height_exp(
    env: ManagerBasedRLEnv, 
    target_height: float, 
    std: float = 0.5, # 使用较大的 std 确保长距离也有梯度
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """
    修改版：使用指数奖励代替线性截断。
    即使误差很大（如 0.5m），指数函数也会返回非零值和梯度。
    """
    robot = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        ground_height = torch.mean(ray_hits_z, dim=1)
        # 过滤无效数据
        ground_height = torch.where(torch.abs(ground_height) > 10.0, torch.zeros_like(ground_height), ground_height)
        current_height = robot.data.root_pos_w[:, 2] - ground_height
    else:
        current_height = robot.data.root_pos_w[:, 2]
    
    error = torch.square(current_height - target_height)
    
    # 指数奖励：exp(-error / std^2)
    reward = torch.exp(-error / (std**2))
    
    return reward * _get_gait_mask(env, 1)

def biped_upright_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    更直接的姿态惩罚：最小化重力在 X/Y 轴的分量。
    这比最大化 Z 分量在数值优化上通常更平滑。
    """
    robot = env.scene["robot"]
    # 投影重力到机身坐标系
    g = robot.data.projected_gravity_b 
    # 我们希望 g 接近 [0, 0, -1]。所以 punish x 和 y 的分量
    error = torch.sum(torch.square(g[:, :2]), dim=1)
    return torch.exp(-error / 0.5) * _get_gait_mask(env, 1)

def track_lin_vel_xy_yaw_frame_biped(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
) -> torch.Tensor:
    """
    修正版：在 Heading Frame (航向坐标系) 下计算速度误差。
    无论机器人是趴着还是站着，Heading Frame 的 X 轴永远指向它面朝的“水平前方”。
    """
    # 1. 获取指令 (通常是在水平面上的 x, y)
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    
    # 2. 获取世界坐标系下的绝对速度
    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :3]
    
    # 3. 计算机器人的 Yaw (航向)
    # 提取四元数中的 Yaw 分量 (忽略 Pitch 和 Roll)
    # 注意：这里假设机器人的 root_quat_w 能够反映整体朝向
    # 如果站立时发生 Gimbal Lock 导致 Yaw 乱跳，可能需要用 IMU 或特定 Link 辅助，
    # 但一般 IsaacLab 的 yaw_quat 工具对 standard Z-up 也是适用的。
    yaw_q = yaw_quat(env.scene["robot"].data.root_quat_w)
    
    # 4. 将世界速度投影回航向坐标系 (Heading Frame)
    # 结果: [v_forward, v_left, v_up]
    vel_heading = quat_apply_inverse(yaw_q, vel_w)
    
    # 5. 只取前两维 (Forward, Left) 与指令比较
    lin_vel_error = torch.sum(torch.square(vel_cmd - vel_heading[:, :2]), dim=1)
    
    # 6. 计算奖励
    base_reward = torch.exp(-lin_vel_error / (std**2))
    
    # 加上之前讨论的软姿态门控
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    proj_g = quat_apply_inverse(env.scene["robot"].data.root_quat_w, gravity_vec)
    soft_posture = torch.clamp(-proj_g[:, 2], min=0.0) # 修正这里：-proj_g[:, 2] 代表 Up 轴分量
    
    return base_reward * soft_posture * _get_gait_mask(env, 1)

def track_ang_vel_z_world_biped(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
) -> torch.Tensor:
    """
    修正版：追踪世界坐标系下的 Z 轴角速度。
    不管机器人怎么倾斜，我们只关心它绕着世界垂直轴旋转的速度。
    """
    vel_cmd = env.command_manager.get_command(command_name)[:, 2]
    
    # 直接取世界坐标系的角速度 Z 分量
    # 这样只有绕地面的垂直轴旋转才算数
    ang_vel_w_z = env.scene["robot"].data.root_ang_vel_w[:, 2]
    
    ang_vel_error = torch.square(vel_cmd - ang_vel_w_z)
    
    base_reward = torch.exp(-ang_vel_error / (std**2))
    
    # 姿态门控
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    proj_g = quat_apply_inverse(env.scene["robot"].data.root_quat_w, gravity_vec)
    soft_posture = torch.clamp(-proj_g[:, 2], min=0.0)

    return base_reward * soft_posture * _get_gait_mask(env, 1)
def lin_vel_z_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize vertical velocity to prevent hopping/bouncing."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2]) * _get_gait_mask(env, 1)
# ==============================================================================
# Improved Bipedal Rewards (Focus on Swing-up)
# ==============================================================================

def biped_upright_linear(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for upright orientation (Alignment with Gravity).
    Uses a linear reward structure to provide strong gradients even when the robot is flat.
    range: [0, 1]
    """
    robot = env.scene["robot"]
    # Projected gravity in body frame
    # For a standing robot, gravity vector in body frame should be [0, 0, -1]
    g_body = robot.data.projected_gravity_b 
    
    # We want g_body.z to be -1.
    # If robot is flat, g_body.z is approx 0.
    # Reward = -g_body.z (Since g_z is negative when upright)
    # Clamp to ensure it's positive [0, 1]
    
    # Note: This gives a linear gradient. 
    # Flat (0) -> 45deg (0.7) -> Upright (1.0)
    reward = torch.clamp(-g_body[:, 2], min=0.0)
    
    # Optional: Enhance gradient at the beginning (swing-up phase)
    # reward = torch.pow(reward, 0.5) # Square root makes the gradient steeper near 0
    
    return reward * _get_gait_mask(env, 1)

def biped_stand_orientation_corrected(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Corrected Orientation Reward for Wheel-Bipedal Balance.
    Target: The robot should pitch up ~90 degrees.
    
    Logic:
    - Flat (Quadruped): Gravity projects to Z-axis (approx -1).
    - Upright (Biped): Gravity projects to X-axis (approx -1 if nose up, +1 if nose down).
    
    We want to maximize the magnitude of gravity on the X-axis.
    """
    robot = env.scene[asset_cfg.name]
    # Projected gravity in body frame
    g_body = robot.data.projected_gravity_b 
    
    # We want g_body.x to be large (close to 1 or -1).
    # We want g_body.z to be small (close to 0).
    
    # Method 1: Penalize Z-alignment (Force it to be NOT flat)
    # If flat, g_z ~ -1, reward ~ 0.
    # If upright, g_z ~ 0, reward ~ 1.
    reward_z = 1.0 - torch.abs(g_body[:, 2])
    
    # Method 2: Encourage X-alignment (Stronger gradient for lifting nose)
    # Assuming standing on rear wheels means 'nose up', X-axis points roughly UP.
    # Gravity points DOWN. So g_body.x should be approx -1.
    # reward_x = torch.relu(-g_body[:, 0]) 
    
    # Combined: Penalize Z is safer to avoid confusion about "which way is up" initially
    # Use a squared penalty to make the gradient sharper near flat
    reward = torch.square(1.0 - torch.abs(g_body[:, 2]))
    
    return reward * _get_gait_mask(env, 1)

def biped_stand_height_posture_gated(
    env: ManagerBasedRLEnv, 
    target_height: float, 
    std: float = 0.5,
    gate_threshold: float = 0.3, # Approx 17 degrees pitch required to start getting height reward
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """
    Reward base height ONLY if the robot is sufficiently upright.
    Prevents the 'tiptoeing' local minimum where robot extends legs while lying flat.
    """
    robot = env.scene[asset_cfg.name]
    
    # 1. Calculate Height Error
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        ground_height = torch.mean(ray_hits_z, dim=1)
        ground_height = torch.where(torch.abs(ground_height) > 10.0, torch.zeros_like(ground_height), ground_height)
        current_height = robot.data.root_pos_w[:, 2] - ground_height
    else:
        current_height = robot.data.root_pos_w[:, 2]
    
    height_error = torch.square(current_height - target_height)
    height_reward = torch.exp(-height_error / (std**2))
    
    # 2. Calculate Orientation Gate
    g_body = robot.data.projected_gravity_b
    uprightness = -g_body[:, 2] # 1.0 is upright, 0.0 is flat
    
    # Hard Gate or Soft Sigmoid Gate
    # If uprightness < threshold, reward is 0 (or very small)
    # This forces the agent to fix orientation FIRST.
    gate = torch.where(uprightness > gate_threshold, torch.ones_like(uprightness), torch.zeros_like(uprightness))
    
    # Alternative: Soft multiplication (uprightness * height)
    # But a hard gate is often better to break the local minimum.
    
    return height_reward * gate * _get_gait_mask(env, 1)

def biped_height_strict_gated(
    env: ManagerBasedRLEnv, 
    target_height: float, 
    gate_pitch_threshold: float = 0.5, # ~30 degrees
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """
    Height reward that ONLY activates if the robot is actually pitching up.
    Stops the robot from 'jumping' while flat.
    """
    robot = env.scene[asset_cfg.name]
    
    # 1. Height Error
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        ground_height = torch.mean(ray_hits_z, dim=1)
        # Filter invalid
        ground_height = torch.where(torch.abs(ground_height) > 10.0, torch.zeros_like(ground_height), ground_height)
        current_height = robot.data.root_pos_w[:, 2] - ground_height
    else:
        current_height = robot.data.root_pos_w[:, 2]
    
    # Exponential reward for height
    height_reward = torch.exp(-torch.square(current_height - target_height) / 0.25)
    
    # 2. Strict Pitch Gate
    # Calculate Pitch from Projected Gravity
    # If flat, g_x ~ 0, g_z ~ -1.
    # If pitched up, g_x becomes non-zero.
    g_body = robot.data.projected_gravity_b
    
    # We use the absolute value of X projection as a proxy for pitch angle magnitude
    # If g_body.x > 0.5, it means we are tilted at least 30 degrees (sin(30)=0.5)
    tilt_magnitude = torch.abs(g_body[:, 0])
    
    # Sigmoid gate: 0 if flat, 1 if tilted
    gate = torch.sigmoid((tilt_magnitude - 0.3) * 10.0)
    
    return height_reward * gate * _get_gait_mask(env, 1)

def biped_swingup_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Incentivize Pitch Velocity explicitly when the robot is NOT yet upright.
    Helping the robot discover the 'jerk' motion needed to stand up.
    """
    robot = env.scene[asset_cfg.name]
    
    # Check orientation
    g_body = robot.data.projected_gravity_b
    uprightness = -g_body[:, 2]
    
    # Only reward pitch velocity if we are lying down (uprightness < 0.8)
    # Assuming standard coordinate system where +Pitch (y-axis angular vel) raises the front
    # Check your robot's coordinate frame! Usually Pitch is Y-axis.
    # If robot faces +X, rotating around +Y lifts the nose (if using Right Hand Rule).
    # You might need to check if it needs +Y or -Y velocity.
    
    ang_vel_pitch = robot.data.root_ang_vel_b[:, 1]
    
    # Bonus for pitching UP when flat
    # We use a tanh to cap the bonus so it doesn't explode
    bonus = torch.tanh(ang_vel_pitch) 
    
    mask = (uprightness < 0.7).float() # Only active during transition
    
    return bonus * mask * _get_gait_mask(env, 1)

def biped_stand_orientation_strict(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Strict orientation reward for Rear-Wheel Standing.
    Target: The robot's nose should point UP.
    
    Physics:
    - Gravity vector in World is [0, 0, -1].
    - If Robot Nose (+X) points UP, Gravity in Body frame is [-1, 0, 0].
    
    We reward -g_body_x.
    """
    robot = env.scene[asset_cfg.name]
    g_body = robot.data.projected_gravity_b 
    
    # g_body[:, 0] 是重力在 X 轴的分量。
    # 我们希望它是 -1.0。
    # 奖励公式： (-g_x + 1) / 2  --> 范围 [0, 1]
    # 或者简单粗暴的ReLU： clamp(-g_x, 0, 1)
    
    # 采用平方误差形式，引导梯度更平滑
    # error = (g_x - (-1))^2 + g_y^2 + g_z^2
    # 但我们只关心把头抬起来，所以主要奖励 -g_x
    
    # 线性奖励：越接近 -1 越高
    reward = torch.clamp(-g_body[:, 0], min=0.0) 
    
    # 增加指数锐化，迫使它必须非常直才能拿高分
    reward = torch.pow(reward, 2)
    
    return reward * _get_gait_mask(env, 1)

def biped_swing_pitch_bonus(
    env: ManagerBasedRLEnv, 
    threshold_rad: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward giving a 'kick' of Pitch Velocity ONLY when the robot is lying flat.
    Helps initiate the stand-up motion.
    """
    robot = env.scene[asset_cfg.name]
    g_body = robot.data.projected_gravity_b
    
    # If g_x is near 0, we are flat. 
    is_flat = torch.abs(g_body[:, 0]) < 0.5 # < 30 degrees tilt
    
    # We want +Pitch Velocity (Nose goes up)
    # Check your axis! Usually Y-axis angular velocity.
    pitch_vel = robot.data.root_ang_vel_b[:, 1]
    
    reward = torch.relu(pitch_vel) 
    
    return reward * is_flat.float() * _get_gait_mask(env, 1)


# ==============================================================================
# Ported Handstand/Stand-Env Rewards (with Gait Gating)
# ==============================================================================

def biped_stand_orientation_l2(
    env: ManagerBasedRLEnv, 
    target_gravity: list[float], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    L2 regularization for orientation based on projected gravity.
    Ported from stand_env's handstand_orientation_l2.
    
    Arguments:
        target_gravity: Expected gravity vector in BODY frame (e.g. [-1, 0, 0] for nose up).
    """
    robot = env.scene[asset_cfg.name]
    
    # 当前在机身坐标系下的重力向量
    current_gravity = robot.data.projected_gravity_b
    
    # 目标重力向量
    target = torch.tensor(target_gravity, device=env.device)
    
    # 计算 L2 误差 (Sum of Squared Errors)
    error = torch.sum(torch.square(current_gravity - target), dim=1)
    
    # 转化为奖励：exp(-error) 或 直接用负 error (stand_env 似乎直接用了 L2 作为惩罚项，权重为负)
    # 这里我们返回 Error 本身，配置里权重设为负数 (weight < 0)
    return error * _get_gait_mask(env, 1)


def biped_feet_air_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for keeping specific feet (wheels) at a target height.
    Used for the "Air Feet" (Front wheels).
    """
    asset = env.scene[asset_cfg.name]
    
    # 获取指定脚（body_ids）的 Z 坐标
    # asset_cfg.body_ids 应该在 config 里指定为前轮/前脚
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # 计算误差
    error = torch.square(foot_pos_z - target_height)
    
    # 对所有指定的脚取平均或求和
    mean_error = torch.mean(error, dim=1)
    
    reward = torch.exp(-mean_error / (std**2))
    
    return reward * _get_gait_mask(env, 1)


def biped_feet_air_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize if the specified feet (Front feet) touch the ground.
    Ported from handstand_feet_on_air.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取接触力 (Net Forces)
    # shape: (num_envs, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # 计算合力大小
    force_norm = torch.norm(forces, dim=-1)
    
    # 如果力 > 1.0，则认为接触
    is_contact = (force_norm > 1.0).float()
    
    # 对所有指定的脚求和 (接触脚的数量)
    # 我们希望这个值为 0，所以配置里权重设为负数
    return torch.sum(is_contact, dim=1) * _get_gait_mask(env, 1)

def feet_contact_gated(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    gait_mode: int,
    threshold: float = 1.0
) -> torch.Tensor:
    """
    Reward having contact on specific feet/wheels in a specific gait mode.
    Encourages maintaining contact (avoiding stepping/trotting) for wheeled locomotion.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Check if force magnitude > threshold
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    has_contact = (forces > threshold).float()
    
    # Reward is mean contact ratio of all specified bodies (1.0 = all commanded feet on ground)
    return torch.mean(has_contact, dim=1) * _get_gait_mask(env, gait_mode)

def joint_pos_penalty_adaptive_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    gait_mode: int,
    # --- 基础参数 ---
    velocity_threshold: float = 0.5,
    # --- 高度门控 (Floating Ring/Gap) ---
    h_free_min: float = 0.10,   # 当障碍物距离/高度差小于此值（比如头顶空间紧张），完全放松限制
    h_free_max: float = 0.50,   # 当空间正常时，施加完全限制
    offset: float = 0.5,
    # --- 平滑参数 ---
    alpha: float = 0.05,        # [重要] 降低此值以减缓 Gate 变化，防止闪烁 (原 0.1)
) -> torch.Tensor:
    """
    自适应关节位置惩罚（稳定版）。
    1. 移除了 Pitch Gate 以消除高频振荡。
    2. 仅保留 Height Gate 以应对 Floating Ring 的蹲下需求。
    3. 增强了 EMA 平滑。
    """
    # 1. 基础 L2 偏差计算
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    pos_err = torch.sum(torch.square(diff), dim=1) # L2 Error

    # 2. Gate: 高度/障碍物门控 (Height Gate)
    # 调用 mdp 中的 height_scan (确保你的 rewards.py 顶部导入了 mdp)
    # H 通常代表 "扫描到的地形高度 z" 或 "射线击中点的 z 坐标"
    # 假设你的 sensor_cfg 指向一个 Height Scanner
    # 如果是向上看的 scanner (测天花板): H 越小，说明空间越局促 -> 需要放松限制 (Gate -> 1)
    # 如果是向下看的 scanner (测地形): H 代表地形高度。
    
    # 这里的逻辑假设: 高度扫描器检测到某种"异常"或"障碍"
    # 我们用 magnitude 来衡量"非平坦程度"或"障碍物接近程度"
    H = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    
    if H.ndim == 2 and H.shape[1] > 1:
        mag = H.abs().max(dim=1).values
    else:
        mag = H.abs().squeeze(-1)
        
    # 计算 Gate: 
    # mag < h_free_min (遇到障碍/高度差大): Gate -> 1.0 (Relax)
    # mag > h_free_max (正常空间):         Gate -> 0.0 (Strict)
    # 注意：这里需要根据你的 scanner 具体是"测距"还是"测高"来调整符号。
    # 假设 mag 代表"自由空间高度"或"与障碍物的距离": 距离越小，Gate越大(放松)
    # 如果 mag 代表"地形高度差(terrain delta)": 差值越大，Gate越大(放松)
    
    # 下面逻辑假设 mag 是 "地形起伏/障碍程度" (越大约危险 -> 放松限制允许调整):
    gate = torch.clamp((mag - h_free_min) / max(1e-6, (h_free_max - h_free_min)), 0.0, 1.0)

    # 3. 平滑 (EMA)
    # 即使移除了 Pitch，Height Scan 也可能有噪声，必须平滑
    if not hasattr(env, "adaptive_pos_gate_ema"):
        env.adaptive_pos_gate_ema = torch.zeros_like(gate)
    # 使用更小的 alpha (如 0.05) 让变化更迟钝，过滤高频噪声
    env.adaptive_pos_gate_ema = (1.0 - alpha) * env.adaptive_pos_gate_ema + alpha * gate
    g_smooth = env.adaptive_pos_gate_ema

    # 4. 计算最终缩放系数
    # g=0 (Normal) -> scale=1.0 (全额惩罚)
    # g=1 (Obstructed) -> scale=0.1 (保留微弱引导)
    min_scale = 0.1
    scale = (1.0 - g_smooth) + (min_scale * g_smooth)

    # 5. 应用 Gait Mask
    gait_mask = _get_gait_mask(env, gait_mode)

    return pos_err * scale * gait_mask


def wheel_sync_when_straight_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    gait_mode: int,
    rot_threshold: float = 0.1
) -> torch.Tensor:
    """
    Penalize variance in wheel velocities when the rotational command is small (moving straight).
    Formula: Sum((v_i - v_mean)^2) * Mask
    """
    # 1. 获取指令
    cmd = env.command_manager.get_command(command_name)
    ang_cmd = cmd[:, 2] # Yaw rate command

    # 2. 计算 Mask：只有当转向指令绝对值小于阈值时才激活
    is_straight = (torch.abs(ang_cmd) < rot_threshold).float()

    # 3. 获取轮子速度
    asset = env.scene[asset_cfg.name]
    # 注意：这里假设 asset_cfg 传入的是 4 个轮子的 joint_names
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 4. 计算方差惩罚
    # 计算当前时刻 4 个轮子的平均速度 [env_num, 1]
    mean_vel = torch.mean(wheel_vel, dim=1, keepdim=True)
    # 计算每个轮子偏离平均值的平方和 [env_num]
    sync_penalty = torch.sum(torch.square(wheel_vel - mean_vel), dim=1)

    # 5. 应用 Gate 和 Gait Mask
    # 如果处于四足模式(gait_mode=0) 且 处于直线行驶指令，则施加惩罚
    return sync_penalty * is_straight * _get_gait_mask(env, gait_mode)



def joint_deviation_abad_straight_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    gait_mode: int,
    rot_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Penalize deviation of specific joints (usually HAA/ABAD) ONLY when moving straight.
    This prevents the penalty from fighting against necessary leg movements during turning.
    """
    # 1. 获取指令
    cmd = env.command_manager.get_command(command_name)
    ang_cmd = cmd[:, 2]  # Yaw rate command

    # 2. 计算 Mask：只有当转向指令绝对值小于阈值时才激活 (直线行驶)
    is_straight = (torch.abs(ang_cmd) < rot_threshold).float()

    # 3. 计算关节偏差 (L2 square)
    asset: Articulation = env.scene[asset_cfg.name]
    # 注意：这里只计算 asset_cfg 中选定的关节 (即 HAA)
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.square(diff), dim=1)

    # 4. 组合 Mask 和 Gait Mode
    # 逻辑：(是直线行驶) AND (是指定步态)
    return deviation * is_straight * _get_gait_mask(env, gait_mode)

def action_rate_l2_subset(
    env: ManagerBasedRLEnv, 
    action_ids: list[int], 
    gait_mode: int | None = None  # 可选：如果不需要步态门控，可以不传
) -> torch.Tensor:
    """
    Penalize the rate of change of specific actions (L2 squared).
    Useful for penalizing leg jitter while allowing wheel velocity changes.
    """
    # 1. 处理第一帧没有 prev_action 的情况
    if env.action_manager.prev_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 2. 提取指定维度的动作
    # [env_num, selected_dim]
    curr_action = env.action_manager.action[:, action_ids]
    prev_action = env.action_manager.prev_action[:, action_ids]
    
    # 3. 计算变化率平方 (curr - prev)^2
    diff_sq = torch.square(curr_action - prev_action)
    
    # 4. 求和得到惩罚值
    penalty = torch.sum(diff_sq, dim=1)
    
    # 5. (可选) 应用步态门控
    if gait_mode is not None:
        # 假设你在 rewards.py 里有这个辅助函数 _get_gait_mask
        # 或者直接复制逻辑: mask = env.gait_mode if gait_mode==1 else (1 - env.gait_mode)
        mask = _get_gait_mask(env, gait_mode)
        penalty = penalty * mask
        
    return penalty


def wheels_stop_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,  # 把 asset_cfg 放在非默认参数位置，或者统一参数顺序
    command_threshold: float = 0.1,
) -> torch.Tensor:
    # 1. 检查命令：是否要求静止
    cmd = env.command_manager.get_command(command_name)
    # 取前两维 (lin_vel_x, lin_vel_y)
    cmd_norm = torch.norm(cmd[:, :2], dim=1) 
    is_still = cmd_norm < command_threshold

    # 2. 获取【物理速度】(rad/s)
    # 确保 asset_cfg 是正确的配置对象
    wheel_vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    
    # 3. 计算惩罚 (建议改用 L1 abs，锁车更紧)
    cost = torch.sum(torch.abs(wheel_vel), dim=1)
    
    return cost * is_still.float()

def wheel_freeze_during_turn_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    gait_mode: int,
    rot_threshold: float = 0.1,  # 当转向命令大于此值
    lin_threshold: float = 0.1,  # 且直线命令小于此值（原地转向）
) -> torch.Tensor:
    """
    当处于原地转向状态时，惩罚轮子的转动。
    强迫机器人使用腿部动作（踏步）来完成转向。
    """
    # 1. 获取指令
    cmd = env.command_manager.get_command(command_name)
    ang_cmd = torch.abs(cmd[:, 2])      # Z轴转向
    lin_cmd = torch.norm(cmd[:, :2], dim=1) # XY平面速度

    # 2. 判断是否是“原地转向”意图
    # 逻辑：用户想转 (rot > 0.1) 且 不想走 (lin < 0.1)
    is_turning_in_place = (ang_cmd > rot_threshold) & (lin_cmd < lin_threshold)

    # 3. 获取轮子速度
    asset = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # 4. 惩罚：如果满足转向意图，轮子速度越快惩罚越大
    penalty = torch.sum(torch.square(wheel_vel), dim=1)

    # 5. 应用 Mask
    return penalty * is_turning_in_place.float() * _get_gait_mask(env, gait_mode)