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
import robot_lab.tasks.locomotion.velocity.mdp as mdp
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
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
    # # 计算竖直方向分量
    # uprightness = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], -1.0, 1.0)
    # # 15°阈值，对应 cos(15°)≈0.966
    # threshold = torch.cos(torch.deg2rad(torch.tensor(15.0, device=uprightness.device)))
    # # 当姿态比15°更正时，直接取满额；超过15°才进入衰减
    # scale = torch.clamp((uprightness - threshold) / (1 - threshold), 0.0, 1.0)

    # reward *= scale
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_lin_vel_x_world_exp(
    env, command_name: str, std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """世界系 x 方向线速度跟踪（指数核）"""
    asset = env.scene[asset_cfg.name]
    cmd_b = env.command_manager.get_command("base_velocity")[:, :2]          # [N,2]  (vx^b, vy^b)
    quat_w = asset.data.root_link_quat_w                                     # [N,4], wxyz
    cmd_b3 = torch.cat([cmd_b, torch.zeros_like(cmd_b[:, :1])], dim=1)       # [N,3]
    cmd_w3 = math_utils.quat_apply_yaw(quat_w, cmd_b3)                 # 旋到世界
    v_cmd_x_world = cmd_w3[:, 0]
    v_x_w   = asset.data.root_com_lin_vel_w[:, 0]                               # 实际 vx（世界系）
    err = (v_cmd_x_world - v_x_w).pow(2)
    rew = torch.exp(-err / (std**2))
    # 可选：只有机器人“站直/直立”时才给这项奖励（用重力在机体系 z 轴上的投影做 gating）
    rew *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7  # projected_gravity_b 定义见文档
    return rew

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

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
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheels_stop_without_cmd(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    当没有速度命令时，惩罚轮子转动（基于关节速度）。
    """
    # 提取机器人 articulation
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取这些关节的速度
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [num_envs, num_wheel_joints]

    # 判断命令是否为 "静止" （这里只看 base 线速度/角速度是否接近 0）
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1

    # 计算惩罚：轮子速度越大，惩罚越大
    penalty = torch.sum(torch.abs(wheel_vel), dim=1)

    return penalty * command


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

def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7) * torch.where(torch.abs(torch.atan2(env.scene["robot"].data.projected_gravity_b[:, 0], -env.scene["robot"].data.projected_gravity_b[:, 2])) > 0.3490658503988659, 0.1*torch.ones_like(reward), torch.ones_like(reward))
    return reward

def wheel_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "wheel_mirror_joints_cache") or env.wheel_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.wheel_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]

    # ---- 新增：本地超参（不改函数签名）----
    tau = 0.3           # 小差距阈值（rad/s）
    small_scale = 0.25  # 小差距区惩罚缩放
    exp_cap = 60.0      # 指数上限防爆
    eps = 1e-12
    beta = 2.0 * small_scale / (tau + eps)  # 保证在 d=tau 处函数值与一阶导连续
    # -----------------------------------

    # ... 前面保持不变（含 tau/d_max/scale 等） ...
    reward = torch.zeros(env.num_envs, device=env.device)

    per_pair_terms = []  # 收集每一对镜像轮的惩罚（负值）

    for joint_pair in env.wheel_mirror_joints_cache:
        left  = asset.data.joint_vel[:, joint_pair[0][0]]
        right = asset.data.joint_vel[:, joint_pair[1][0]]

        d = torch.abs(left - right)        # [N] or [N,K]
        d_norm = d / 30.0                  # 30 = 最大速度差
        scale = 50.0                       # d=3 → -0.5（单对）
        diff = -(scale * (d_norm ** 2))    # 负值=惩罚

        if diff.ndim > 1:
            diff = diff.sum(dim=-1)        # [N]
        per_pair_terms.append(diff)        # 记录每一对

    if len(per_pair_terms) > 0:
        terms = torch.stack(per_pair_terms, dim=-1)  # [N, P]

        # ===== 选择一种聚合方式（任选其一）=====

        # 1) 求和（不平均）：多个异常叠加更痛
        reward = terms.sum(dim=-1)

        # 2) 取“最差一对”（最负的那一列）：任意一对异常就很痛
        # reward, _ = terms.min(dim=-1)

        # 3) Top-k 平均（例如最差的2对）
        # k = min(2, terms.shape[-1])
        # reward = terms.topk(k, dim=-1, largest=False).values.mean(dim=-1)

        # 4) 平滑最小（smooth-min），兼顾可导与“抓最差”
        # tau_aggr = 0.5
        # reward = -tau_aggr * torch.logsumexp(-terms / tau_aggr, dim=-1)
    else:
        reward = torch.zeros(env.num_envs, device=env.device)

    # 姿态缩放保持不变
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
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


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


def feet_continue_contact(env, command_name, expect_contact_num, sensor_cfg) -> torch.Tensor:
    s = env.scene.sensors[sensor_cfg.name]
    forces = s.data.net_forces_w  # [N, num_bodies, 3]
    # 每脚是否接触（法向力阈值可按需要调）
    contact = (forces[:, sensor_cfg.body_ids, 2].abs() > 9.8).float()  # [N, num_feet]

    # —— 惰性初始化 & 指数滑动平均占空比（强调“持续贴地”）——
    if not hasattr(env, "contact_ema"):
        env.contact_ema = torch.zeros_like(contact)
    alpha = 0.1  # 越小越强调“持续”；0.05~0.2 常用
    env.contact_ema = (1.0 - alpha) * env.contact_ema + alpha * contact  # [N, num_feet]

    # —— 占空比阈值：每脚是否达到“贴地占空比”要求 —— 
    target_duty = 0.85  # 平地建议 0.75~0.85；爬箱子可放宽到 ~0.65
    slack = 0.05
    duty_ok = (env.contact_ema >= (target_duty - slack)).float()        # [N, num_feet]
    reward = duty_ok.mean(dim=1)                                        # [N], 0~1

    # —— 速度权重（替代硬门控）：慢速也能有奖励，但快一点更赚 —— 
    cmd = env.command_manager.get_command(command_name)                 # [N, D]
    v_ref = 1.0  # 参考最大期望线速度，按你的命令分布调整
    w = (cmd[:, 0:2].norm(dim=1) / (v_ref + 1e-6)).clamp(0.2, 1.0)     # 避免静止时全没奖励
    return reward * w


# def feet_continue_contact(env, command_name, expect_contact_num, sensor_cfg) -> torch.Tensor:
#     s = env.scene.sensors[sensor_cfg.name]
#     forces = s.data.net_forces_w  # [N, num_bodies, 3]
#     # 每脚是否接触（法向力阈值可按需要调）
#     contact = (forces[:, sensor_cfg.body_ids, 2].abs() > 9.8).float()  # [N, num_feet]

#     # —— 惰性初始化 & 指数滑动平均占空比（强调“持续贴地”）——
#     if not hasattr(env, "contact_ema"):
#         env.contact_ema = torch.zeros_like(contact)
#     alpha = 0.05  # 越小越强调“持续”；0.05~0.2 常用
#     env.contact_ema = (1.0 - alpha) * env.contact_ema + alpha * contact  # [N, num_feet]

#     # —— 占空比阈值：每脚是否达到“贴地占空比”要求 —— 
#     target_duty = 0.65  # 平地建议 0.75~0.85；爬箱子可放宽到 ~0.65
#     slack = 0.05
#     duty_ok = (env.contact_ema >= (target_duty - slack)).float()        # [N, num_feet]
#     reward = duty_ok.mean(dim=1)                                        # [N], 0~1

#     # —— 速度权重（替代硬门控）：慢速也能有奖励，但快一点更赚 —— 
#     cmd = env.command_manager.get_command(command_name)                 # [N, D]
#     v_ref = 0.6  # 参考最大期望线速度，按你的命令分布调整
#     w = (cmd[:, 0].abs() / (v_ref + 1e-6)).clamp(0.2, 1.0)     # 避免静止时全没奖励
#     return reward * w


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
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


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

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


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


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
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
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
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty_height_gated(
    env,
    command_name: str,
    asset_cfg,                  # SceneEntityCfg("robot", joint_names=腿部关节)
    sensor_cfg,                 # SceneEntityCfg("front_height") 或你的高度传感器名
    stand_still_scale: float = 5.0,   # 平地/小起伏时的最强制动倍数
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
    h_free_min: float = 0.10,          # m：低于它强制动
    h_free_max: float = 0.45,          # m：高于它几乎不制动
    offset: float = 0.5,               # 传给 mdp.height_scan 的 offset
    alpha: float = 0.2,                # EMA 平滑系数，0.1~0.3
):
    """
    连续高度扫描做门控：|前方高度变化| 越大，越“放开”；|变化|小则强制动。
    正、负高度（上台阶/坑）都会放开。
    """
    import torch
    from isaaclab.envs.mdp import observations as mdp_obs  # 若你的导入是 mdp.height_scan，就沿用原写法

    # 1) 基础量
    asset = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    pos_err = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1
    )

    # 2) 连续高度扫描（不离散）
    H = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)  # [N, K] 或 [N, 1]
    # Isaac Lab 的 height_scan 是“传感器高度 - 命中点z - offset”，
    # 通常：前方“上台阶”→返回值偏负；“坑”→返回值偏正（见官方API与教程说明）。为了统一为“高度变化量”，用 -H。
    terrain_delta = -H

    # 3) 取“最需要动作”的幅值（你也可改成前方扇区的percentile/mean）
    if terrain_delta.ndim == 2 and terrain_delta.shape[1] > 1:
        mag = terrain_delta.abs().max(dim=1).values  # |上台阶| 或 |坑| 的最大幅值
    else:
        mag = terrain_delta.abs().squeeze(-1)

    # 4) 连续门控：|变化| ≤ h_free_min 强制动；≥ h_free_max 基本放开
    gate = torch.clamp((mag - h_free_min) / max(1e-6, (h_free_max - h_free_min)), 0.0, 1.0)  # [0,1]

    # 5) EMA 平滑，避免门控抖动
    if not hasattr(env, "pose_gate_ema"):
        env.pose_gate_ema = gate
    env.pose_gate_ema = (1.0 - alpha) * env.pose_gate_ema + alpha * gate
    g = env.pose_gate_ema

    # 6) 制动倍数：g=0 → stand_still_scale；g=1 → 1
    s = stand_still_scale - (stand_still_scale - 1.0) * g

    # 7) 应用策略（两种模式见下文）
    use_standstill = torch.logical_and(cmd <= command_threshold, body_vel <= velocity_threshold)
    scale = torch.where(use_standstill, s, torch.ones_like(s))  # “仅静止/慢行时制动”的版本

    return pos_err * scale  # 外面配 weight 为负，使其成为惩罚项


# 安全的高度门控版平姿态损失：近处低障强烈压制倾斜；高障主动鼓励倾斜
def flat_orientation_height_gated(
    env,
    sensor_cfg=None,             # SceneEntityCfg("height_scanner")
    h_low: float = 0.10,         # 低于此高度→强烈抑制倾斜
    h_high: float = 0.25,        # 高于此高度→开始鼓励倾斜（线性过渡）
    encourage_scale: float = 0.5,# 鼓励倾斜强度(系数)
    use_disc: bool = True,       # 复用你提供的 height_scan_disc
    offset: float = 0.5,         # 与你的扫描一致
):
    # 1) 基础“平姿态”误差：proj_g_b 应该接近 [0,0,-1]，所以用 x/y 两分量的平方和
    proj_g = env.scene["robot"].data.projected_gravity_b  # [N,3]
    upright_err = (proj_g[:, 0] ** 2 + proj_g[:, 1] ** 2)  # [N]

    # 2) 读取前向高度，做鲁棒的“实体存在性”检查（不要直接 `name in env.scene`）
    front_h = None
    sensor_name = getattr(sensor_cfg, "name", None) if (sensor_cfg is not None) else None
    if isinstance(sensor_name, str) and sensor_name:
        try:
            _ = env.scene[sensor_name]  # 若不存在会抛 KeyError
            if use_disc:
                # 你给的离散化接口：返回 [-1,1] 的 bin。这里把它转成一个“高障指标” in [0,1]
                bins = mdp.height_scan_disc(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)  # [N]
                # 经验约定：bins 越小（更负）意味着前方越高/越近的台阶或障碍
                high_obs_score = (-bins).clamp(0.0, 1.0)  # [0,1]：0=平地/低障，1=高障
                # 用一个“等效高度”表达门控（不必是物理米值，只要单调即可）
                front_h = high_obs_score
            else:
                # 若你使用未离散化的真实高度（米），就直接 squeeze
                heights = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)  # [N]
                # 这里假设 heights 越大代表障碍越高；如含义相反，请对号入座取负
                front_h = heights
        except KeyError:
            pass

    # 3) 计算门控系数 gate ∈ [0,1]：低障→0（强抑制倾斜），高障→1（鼓励倾斜）
    if front_h is None:
        gate = torch.zeros_like(upright_err)  # 没有传感器就当低障/平地处理
    else:
        # 如果用 disc：front_h 已在 [0,1]，可以直接把 h_low/h_high 视作阈值分位
        # 如果用真实米值：h_low/h_high 请用米。两者只要保证单调映射即可。
        gate = (front_h - h_low) / max(1e-6, (h_high - h_low))
        gate = torch.clamp(gate, 0.0, 1.0)

    # 4) 组合：低障时 (1-gate)*upright_err 强力压制倾斜；高障时 -encourage_scale*gate*tilt_reward 鼓励倾斜
    # 这里以“前后俯仰倾斜”为例，鼓励 |gx|（有需要你也可以加 |gy|）
    tilt_magnitude = torch.abs(proj_g[:, 0])  # 主要鼓励 pitch 方向的倾斜
    reward = (1.0 - gate) * upright_err - encourage_scale * gate * tilt_magnitude
    return reward


