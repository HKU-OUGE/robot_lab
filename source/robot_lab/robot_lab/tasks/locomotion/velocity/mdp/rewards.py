# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import math
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
from .highstep_schedule import global_update as _global_update
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
# def track_lin_vel_xy_exp(
#     env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     # compute the error
#     lin_vel_error = torch.sum(
#         torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
#         dim=1,
#     )
#     reward = torch.exp(-lin_vel_error / std**2)
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


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
    env: ManagerBasedRLEnv,
    command_name: str,  # 通常是 "base_velocity"
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    # 1) 取命令的 (vx^b, vy^b)，扩到 3D 向量，z=0
    cmd_b = env.command_manager.get_command(command_name)[:, :2]           # [N,2]
    cmd_b3 = torch.cat([cmd_b, torch.zeros_like(cmd_b[:, :1])], dim=1)     # [N,3]

    # 2) 仅用 yaw 把命令旋到世界系（忽略 pitch/roll）
    #    quat_apply_yaw 文档：只绕航向旋转向量
    quat_w = asset.data.root_link_quat_w                                   # [N,4], wxyz
    cmd_w3 = math_utils.quat_apply_yaw(quat_w, cmd_b3)                     # [N,3]
    v_cmd_x_w = cmd_w3[:, 0]                                               # 目标世界 x 速度

    # 3) 实际世界 x 速度（用 root_com_lin_vel_w）
    v_x_w = asset.data.root_com_lin_vel_w[:, 0]

    # 4) 指数核
    err = (v_cmd_x_w - v_x_w).pow(2)
    rew = torch.exp(-err / (std ** 2))

    # 5) 不要再用 “-projected_gravity_b[:,2]” 去关停奖励（会在站立时 → 0）
    #    如需门控，可改用与“竖直更友好”的 gate（见上文）
    return rew

def track_ang_vel_z_base_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
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

def wheel_action_l2(env: ManagerBasedRLEnv, wheel_ids: list[int]) -> torch.Tensor:
    # env.action_manager.action: (num_envs, action_dim)
    actions = env.action_manager.action[:, wheel_ids]
    return torch.sum(actions**2, dim=1)


def wheel_slip_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.1,
    epsilon: float = 0.10,
    vel_body_frame: bool = True,
    # ---- 可选门控：与你的 wheels_stop_without_cmd 对齐 ----
    command_name: Optional[str] = None,   # 传入则可按“无命令”门控
    no_cmd_lin_thresh: float = 0.05,      # m/s
    no_cmd_ang_thresh: float = 0.05,      # rad/s
    gate_on_no_command: bool = False,     # True: 仅在“无命令”时启用惩罚
    # ---- 可选门控：接触（需要你在 scene 里有 contact_forces） ----
    contact_sensor_cfg: Optional[SceneEntityCfg] = None,
    contact_threshold: float = 1.0,
    gate_on_contact: bool = False         # True: 仅在“接地”时启用惩罚
) -> torch.Tensor:
    """
    轮滑率（longitudinal slip ratio）的 L1-mean 惩罚。
    s_i = (ω_i * R - v_x) / (|v_x| + eps)
    - 理想纯滚: v_x = ω * R -> s -> 0
    - 返回: 每个 env 的标量惩罚 [N]
    """
    # 取机器人与轮关节角速度 ω: [N, n_wheels]
    asset: Articulation = env.scene[asset_cfg.name]
    omega = asset.data.joint_vel[:, asset_cfg.joint_ids]  # rad/s, shape [N, n_wheels]

    # 取得机体前向线速度 v_x: 优先机体系，其次世界系
    if vel_body_frame and hasattr(asset.data, "root_lin_vel_b"):
        v = asset.data.root_lin_vel_b  # [N, 3]
        v_x = v[:, 0]
    elif hasattr(asset.data, "root_lin_vel_w"):
        # 若只有世界系速度，这里保守取 world x；更严谨可将速度投影到机体前向
        v_x = asset.data.root_lin_vel_w[:, 0]
    else:
        # 兜底：若无速度可用，则不产生惩罚
        return torch.zeros(omega.shape[0], device=omega.device, dtype=omega.dtype)

    # 纵向轮滑率 s: [N, n_wheels]
    slip = (omega * wheel_radius - v_x.unsqueeze(-1)) / (torch.abs(v_x).unsqueeze(-1) + epsilon)

    # L1-mean（对大 slip 更敏感，同时与轮数无关）
    penalty = torch.mean(torch.abs(slip), dim=1)  # [N]

    # ----- 可选门控 1：仅在“无命令”时启用 -----
    if gate_on_no_command and (command_name is not None):
        cmd = env.command_manager.get_command(command_name)  # [N, k]
        lin_cmd = cmd[:, :2]
        ang_cmd = cmd[:, 2] if cmd.shape[1] >= 3 else torch.zeros_like(lin_cmd[:, 0])
        no_lin = torch.norm(lin_cmd, dim=1) < no_cmd_lin_thresh
        no_ang = torch.abs(ang_cmd) < no_cmd_ang_thresh
        no_cmd_mask = (no_lin & no_ang).to(penalty.dtype)
        penalty = penalty * no_cmd_mask

    # ----- 可选门控 2：仅在接地时启用 -----
    if gate_on_contact and (contact_sensor_cfg is not None) and (contact_sensor_cfg.name in env.scene):
        cf = env.scene[contact_sensor_cfg.name].data.net_forces_w  # [N, B, 3] or [N,B,6]
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


def lateral_step_scaled_joint_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    foot_body_names: dict[str, str],
    height_threshold: float,
    gate_width: float,
    relief_scale: float,
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 5.0,
    min_contacts_per_side: int = 1,
) -> torch.Tensor:
    """Reduce default-position penalty when the robot is straddling a lateral step."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = joint_position_penalty(env, command_name, asset_cfg, stand_still_scale, velocity_threshold)
    relief = _lateral_step_relief_scale(
        env,
        asset,
        foot_body_names,
        height_threshold,
        gate_width,
        relief_scale,
        contact_sensor_cfg,
        contact_threshold,
        min_contacts_per_side,
    )
    return penalty * relief


def joint_position_penalty_flat_terrain_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    stand_still_scale: float,
    flat_moving_scale: float,
    obstacle_scale: float,
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.5,
    h_low: float = 0.06,
    h_high: float = 0.18,
    offset: float = 0.5,
    ignore_yaw_command: bool = False,
) -> torch.Tensor:
    """Penalize default-joint deviation strongly on flat ground and weakly near obstacles."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1
    )

    command = env.command_manager.get_command(command_name)
    command_norm = _stand_still_command_norm(command, ignore_yaw_command)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
    is_still = torch.logical_and(command_norm < command_threshold, body_vel < velocity_threshold)

    flat_scale = torch.where(
        is_still,
        torch.full_like(joint_error, stand_still_scale),
        torch.full_like(joint_error, flat_moving_scale),
    )

    try:
        height = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
        if height.ndim > 1:
            obstacle_height = height.abs().max(dim=1).values
        else:
            obstacle_height = height.abs()
        obstacle_gate = torch.clamp((obstacle_height - h_low) / (h_high - h_low + 1e-6), 0.0, 1.0)
        obstacle_gate = obstacle_gate * obstacle_gate * (3.0 - 2.0 * obstacle_gate)
    except (KeyError, AttributeError):
        obstacle_gate = torch.zeros_like(joint_error)

    scale = flat_scale * (1.0 - obstacle_gate) + obstacle_scale * obstacle_gate
    return joint_error * scale


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
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, command_threshold: float = 0.25 # 新增：指令速度阈值
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
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
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


def anti_pronk_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_foot_names: tuple[str, str] = ("FL_foot", "FR_foot"),
    rear_foot_names: tuple[str, str] = ("RL_foot", "RR_foot"),
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.1,
    contact_threshold: float = 1.0,
    all_air_weight: float = 1.0,
    all_contact_weight: float = 0.5,
    same_side_pair_weight: float = 0.5,
    include_yaw_command: bool = True,
) -> torch.Tensor:
    """Penalize contact patterns that lead to pronking/bounding instead of trotting."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w
    feet_contact = torch.linalg.norm(contact_forces[:, sensor_cfg.body_ids, :], dim=-1) > contact_threshold
    contact_count = torch.sum(feet_contact.float(), dim=1)
    num_feet = feet_contact.shape[1]

    all_air = contact_count <= 0.5
    all_contact = contact_count >= num_feet - 0.5
    penalty = all_air_weight * all_air.float() + all_contact_weight * all_contact.float()

    if not hasattr(env, "_anti_pronk_front_foot_ids") or not hasattr(env, "_anti_pronk_rear_foot_ids"):
        env._anti_pronk_front_foot_ids = contact_sensor.find_bodies(front_foot_names)[0]
        env._anti_pronk_rear_foot_ids = contact_sensor.find_bodies(rear_foot_names)[0]

    body_contact = torch.linalg.norm(contact_forces, dim=-1) > contact_threshold
    front_contact = body_contact[:, env._anti_pronk_front_foot_ids]
    rear_contact = body_contact[:, env._anti_pronk_rear_foot_ids]
    front_same_phase = front_contact[:, 0] == front_contact[:, 1]
    rear_same_phase = rear_contact[:, 0] == rear_contact[:, 1]
    same_side_pair = 0.5 * (front_same_phase.float() + rear_same_phase.float())
    penalty = penalty + same_side_pair_weight * same_side_pair

    command = env.command_manager.get_command(command_name)
    command_slice = command if include_yaw_command else command[:, :2]
    command_norm = torch.linalg.norm(command_slice, dim=1)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
    is_moving = torch.logical_or(command_norm > command_threshold, body_vel > velocity_threshold)
    return penalty * is_moving.float()


def feet_continue_contact(env, command_name, expect_contact_num, sensor_cfg) -> torch.Tensor:
    s = env.scene.sensors[sensor_cfg.name]
    forces = s.data.net_forces_w  # [N, num_bodies, 3]
    # 每脚是否接触（法向力阈值可按需要调）
    contact = (forces[:, sensor_cfg.body_ids, 2].abs() > 20).float()  # [N, num_feet]

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
    command_name: str,          # 不再使用速度门槛，但保留签名兼容 RewardTermCfg
    asset_cfg,                  # SceneEntityCfg("robot", joint_names=腿部关节)
    sensor_cfg,                 # SceneEntityCfg("front_height")
    stand_still_scale: float = 5.0,
    # velocity_threshold / command_threshold 不再使用，但保留参数以兼容旧配置
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
    h_free_min: float = 0.10,
    h_free_max: float = 0.45,
    offset: float = 0.5,
    alpha: float = 0.2,
    # 可选：倾斜缩放的下限，避免被乘成 0 完全没梯度
    tilt_floor: float = 0.1,    # ∈[0,1]；0.1 表示至少保留 10%
):
    """
    连续高度扫描做门控：|前方高度变化| 越大，越“放开”；|变化|小则强制动。
    与机器人速度无关；全时生效。倾斜越大，惩罚越小（鼓励在倾斜时调整姿态）。
    """
    import torch
    from isaaclab.envs.mdp import observations as mdp  # 这里用 mdp.height_scan
    import math
    # 1) 基础量
    asset = env.scene[asset_cfg.name]
    pos_err = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1
    )

    # 2) 连续高度扫描（不离散）
    H = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)  # [N, K] or [N, 1]
    # height_scan 返回(传感器高度 - 命中点z - offset)；常见设置下上台阶为负、坑为正
    terrain_delta = -H

    # 3) 取代表性幅值（可换成 quantile 更稳）
    if terrain_delta.ndim == 2 and terrain_delta.shape[1] > 1:
        mag = terrain_delta.abs().max(dim=1).values
        # 也可用更鲁棒的分位数：mag = torch.quantile(terrain_delta.abs(), q=0.8, dim=1)
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
    s = stand_still_scale - (stand_still_scale - 1.0) * g   # 与速度无关

    # 7) 倾斜缩放（直立时因子≈1，越倾斜越小；带 floor 避免缩放为 0）
    # proj_gz = env.scene["robot"].data.projected_gravity_b[:, 2]  # 直立≈-1
    # grav = torch.clamp(-proj_gz, 0.0, 0.7) / 0.7                # [0,1]
    # grav = tilt_floor + (1.0 - tilt_floor) * grav               # [tilt_floor, 1]
    g_b = asset.data.projected_gravity_b  # [N,3], 直立≈[0,0,-1]
    # 用重力在机体系的 x、z 分量估计俯仰角：pitch = atan2(|gx|, -gz)
    pitch = torch.atan2(g_b[:, 0].abs(), (-g_b[:, 2]).clamp_min(1e-6))  # [rad]

    lo = math.radians(10.0)
    hi = math.radians(30.0)
    t = ((pitch - lo) / (hi - lo)).clamp(0.0, 1.0)          # [0,1]
    grav = 1.0 - t * (1.0 - tilt_floor)                     # 1 → tilt_floor
    scale = s * grav
    return pos_err * scale   # 外面配 weight 为负，使其成为惩罚项



# 安全的“高台/坑”双向门控版：平地强压制，遇高台或坑逐步放开并适度鼓励俯仰
def flat_orientation_height_gated(
    env,
    sensor_cfg=None,              # SceneEntityCfg("height_scanner")
    h_low: float = 0.10,          # 低于此(米/分位)→视作低风险，强压制倾斜
    h_high: float = 0.25,         # 高于此开始完全放开(线性-光滑过渡)
    encourage_scale: float = 0.5, # 鼓励俯仰强度
    use_disc: bool = True,        # 复用你的 height_scan_disc
    offset: float = 0.5,          # 与你的扫描一致
    alpha: float = 0.2,           # 门控EMA，抑制抖动(0.1~0.3)
    tilt_cap_rad: float = 0.35,   # 最多鼓励到 ~20° 的俯仰，防止过大倾斜
):
    import torch, math
    robot = env.scene["robot"]

    # 1) 倾斜度：XY分量的模（≈ sin(倾角)）；以及原始“平姿态”惩罚
    g = robot.data.projected_gravity_b  # [N,3]
    tilt_xy = torch.sqrt(torch.clamp(g[:, 0]**2 + g[:, 1]**2, min=1e-9))     # [N]
    upright_pen = tilt_xy**2                                                 # 与 isaaclab flat_orientation_l2 对齐

    # 2) 读取前向“高度/坑深”并做成对称的“危险度”hazard ∈ [0,1]
    hazard = None
    if isinstance(getattr(sensor_cfg, "name", None), str):
        try:
            _ = env.scene[sensor_cfg.name]
            if use_disc:
                # 约定：bins ∈ [-1,1]；绝对值越大说明越“极端”(高台 or 坑更明显/更近)
                bins = mdp.height_scan_disc(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)  # [N]
                hazard = bins.abs().clamp(0.0, 1.0)   # 同时覆盖“台阶高”和“坑深”
            else:
                # 连续高度：正=台阶高、负=坑深（若你语义相反，可取负号）
                h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)          # [N]
                # 统一成“危险度”：取绝对值，并按阈值映射
                hazard = h.abs()
        except KeyError:
            pass

    if hazard is None:
        gate = torch.zeros_like(upright_pen)  # 没传感器→当平地处理
    else:
        # 3) hazard→[0,1] 门控；支持“米值”或“分位值”，并用 smoothstep 让过渡更平滑
        gate_lin = torch.clamp((hazard - h_low) / max(1e-6, (h_high - h_low)), 0.0, 1.0)
        gate = gate_lin * gate_lin * (3.0 - 2.0 * gate_lin)  # smoothstep

    # 4) EMA平滑，避免相机/射线抖动引起奖励震荡
    if not hasattr(env, "_tilt_gate_ema"):
        env._tilt_gate_ema = gate
    else:
        env._tilt_gate_ema = (1.0 - alpha) * env._tilt_gate_ema + alpha * gate
    gate_smooth = env._tilt_gate_ema

    # 5) 只鼓励到一个安全上限，防止把机器人“教”到大仰角翻车
    tilt_cap = math.sin(tilt_cap_rad)
    encouraged_tilt = torch.clamp(tilt_xy, max=tilt_cap)

    # 6) 组合：平地(门控小)→强压制；遇高台/坑(门控大)→减惩罚并适度鼓励俯仰
    #    返回“代价”型项：权重大于0时，就是惩罚 − 奖励 的形式
    return (1.0 - gate_smooth) * upright_pen - encourage_scale * gate_smooth * encouraged_tilt


def upright_gate(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    # 直立度：-gz ∈ [0,1]；直立≈1，倾倒≈0
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return upright

# 利用高度扫描做“障碍门控”：有台阶/平台时缩小这项的权重
def obstacle_gate(env, sensor_cfg, h_low=0.10, h_high=0.25, offset=0.5):
    # height_scan: 传感器高度 - 击中点z - offset，越大越“有台阶”
    h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)  # [N, M] 或 [N,1]
    h = h.abs().max(dim=1).values  # 取最大幅度
    gate = torch.clamp((h - h_low) / (h_high - h_low + 1e-6), 0.0, 1.0)
    return gate

def upward_for_climb(env, asset_cfg=SceneEntityCfg("robot"),
                     sensor_cfg=None, k_progress=0.5, relax_scale=0.3):
    asset = env.scene[asset_cfg.name]

    # 1) 门控：有明显台阶/平台时 gate→1，否则→0
    if sensor_cfg is not None:
        gate = obstacle_gate(env, sensor_cfg)  # 见上面实现
    else:
        gate = torch.zeros(asset.data.root_pos_w.shape[0], device=asset.device)

    # 2) 直立度惩罚（平地强，遇障放松）
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)  # 直立≈1
    pen_tilt = (1.0 - upright)**2
    pen_tilt *= (1.0 - gate + gate * relax_scale)

    # 3) 向上“进度”奖励（PBRS），只在 gate>0 时起作用
    z = asset.data.root_pos_w[:, 2]
    if not hasattr(env, "_last_z"): env._last_z = z.clone()
    dz = torch.clamp(z - env._last_z, min=0.0)
    env._last_z = z
    r_progress = k_progress * dz * gate

    # 4) 汇总（注意：惩罚在总奖励里给负权重）
    return r_progress - pen_tilt


def climb_progress_dyn_pbrs(env, asset_cfg=SceneEntityCfg("robot"),
                            sensor_cfg=None, k: float = 2.0, gamma_shape: float = 0.99,
                            h_low: float = 0.10, h_high: float = 0.25, offset: float = 0.5):
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]

    # 门控：这里以“增强门控”为例（有障碍→更强调爬高），若要抑制，把  alpha 换成 (1-alpha)
    if sensor_cfg is not None:
        h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
        h = h.abs().max(dim=1).values
        alpha = torch.clamp((h - h_low) / (h_high - h_low + 1e-6), 0.0, 1.0)
        g = alpha  # 有障碍→g↑
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

def stand_still_flat_orientation_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    command_threshold: float = 0.1,
    ignore_yaw_command: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Provides an extra reward for keeping the body flat specifically when the robot
    is commanded to stand still. This helps stabilization on slopes.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 1. Check if the command is "stand still"
    command = env.command_manager.get_command(command_name)
    cmd_norm = _stand_still_command_norm(command, ignore_yaw_command)
    is_static_cmd = cmd_norm < command_threshold

    # 2. Calculate flatness reward (same as above)
    gravity_b = asset.data.projected_gravity_b
    flat_error = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    flat_reward = torch.exp(-flat_error / std**2)

    # 3. Apply reward only when static
    reward = flat_reward * is_static_cmd.float()

    # Survival gating
    reward *= torch.clamp(-gravity_b[:, 2], 0.0, 0.7) / 0.7

    return reward

def moving_flat_orientation_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    command_threshold: float = 0.12,
    command_max: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward a flat body when the command asks the robot to move.

    This complements stand_still_flat_orientation_bonus: static commands get the
    stand-still reward, while walking/turning commands get this term.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    cmd_norm = torch.norm(command[:, :3], dim=1)
    motion_scale = torch.clamp(
        (cmd_norm - command_threshold) / max(command_max - command_threshold, 1e-6),
        min=0.0,
        max=1.0,
    )

    gravity_b = asset.data.projected_gravity_b
    flat_error = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    flat_reward = torch.exp(-flat_error / std**2)

    reward = flat_reward * motion_scale
    reward *= torch.clamp(-gravity_b[:, 2], 0.0, 0.7) / 0.7
    return reward


def _highstep_training_progress_gate(
    env: ManagerBasedRLEnv,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Scalar gate staged by the checkpoint-continuous high-step update."""
    update_count = _global_update(env, num_steps_per_update)
    progress = (update_count - float(stage_start_update)) / max(float(stage_ramp_updates), 1.0)
    progress = max(0.0, min(1.0, progress))
    return torch.as_tensor(progress, device=env.device)


def blind_climbing_vel_z_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    pitch_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """
    盲走爬台阶奖励：当机器人被指令向前移动，且车头抬起时，奖励其向上的 Z 轴速度。
    """
    # 获取机器人实体数据
    robot = env.scene[asset_cfg.name]

    # 1. 获取向前的速度指令 (X轴)
    commands = env.command_manager.get_command(command_name)
    cmd_vel_x = commands[:, 0]

    # 2. 估算机身仰角 (Pitch)
    # projected_gravity_b 是世界坐标系的重力向量 [0, 0, -1] 在机身局部坐标系下的投影。
    # 当车头抬起 (Nose up) 时，重力在机身局部坐标系下会指向斜后方，即局部 X 轴分量为负。
    # 因此，-projected_gravity_b[:, 0] 是一个正值，近似代表仰角的正弦值 (sin(pitch))。
    projected_gravity = robot.data.projected_gravity_b
    pitch_approx = -projected_gravity[:, 0]

    # 3. 获取世界坐标系下的实际 Z 轴线速度
    vel_z = robot.data.root_lin_vel_w[:, 2]

    # 4. 判断是否处于“爬台阶”状态：
    # 条件 A: 接收到足够大的向前指令 (例如 > 0.2 m/s)
    # 条件 B: 车头抬起超过一定阈值 (pitch_threshold 0.05 大约是 3 度)
    is_climbing = torch.logical_and(cmd_vel_x > 0.2, pitch_approx > pitch_threshold)

    # 5. 计算奖励：只奖励向上的速度 (vel_z > 0)，且只有在爬台阶状态下才给奖励
    climbing_vel_z = torch.clamp(vel_z, min=0.0)

    # 返回奖励值 (非爬台阶状态下，此项奖励为 0)
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return climbing_vel_z * is_climbing.float() * stage_gate

def climbing_pitch_up_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_cmd_x: float = 0.4,
    min_actual_vel_x: float = 0.05,
    max_actual_vel_x: float = 0.3,
    min_pitch_metric: float = 0.05,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """
    【改进版：扬身奖励】
    鼓励机器人在遇到障碍物受阻时抬起前身，但防止其在平地起步时原地“翘头”作弊。
    """
    robot = env.scene[asset_cfg.name]

    # 1. 获取指令和实际速度
    cmd_vel_x = env.command_manager.get_command(command_name)[:, 0]
    actual_vel_x = robot.data.root_lin_vel_b[:, 0]
    actual_vel_z = robot.data.root_lin_vel_b[:, 2]  # 获取 Z 轴（上下）速度

    # 2. 计算 Pitch 角的替代指标 (抬头时为正)
    pitch_metric = -robot.data.projected_gravity_b[:, 0]

    # 3. 核心逻辑修改：增加多重限制条件

    # 条件 A: 强烈的向前意图
    intent_forward = cmd_vel_x > min_cmd_x

    # 条件 B: 实际速度受阻，但必须大于一个下限！(防起步作弊核心)
    # actual_vel_x > 0.05: 确保机器人已经“动起来了”，而不是刚出生在原地静止。
    # actual_vel_x < 0.3: 速度明显低于预期，说明被障碍物挡住了。
    is_resisted = (actual_vel_x > min_actual_vel_x) & (actual_vel_x < max_actual_vel_x)

    # 条件 C: 确保机器人没有在往下掉 (防止下坡或下台阶时误触发抬头)
    not_falling = actual_vel_z > -0.1

    # 条件 D: 限制最大奖励值 (防后空翻核心)
    # 如果不限制，机器人会为了追求无限大的奖励而直接向后翻倒。
    # 限制最大值为 0.4 (大约对应 pitch 角 23.5 度，sin(23.5°) ≈ 0.4)
    # 将上限提高到 0.85，允许机器人仰角达到约 60 度时获得最大奖励
    capped_pitch = torch.clamp(pitch_metric, min=0.0, max=0.85)

    # 【新增】防翻车熔断锁：如果仰角超过约 75 度 (sin(75°) ≈ 0.96)，说明快要后空翻了，直接判定为不安全
    is_safe_pitch = pitch_metric < 0.95

    # 4. 组合所有条件：必须同时满足才给奖励，且要求已经有轻微的抬头趋势 (>0.05)
    # 必须满足：想往前走 + 速度受阻 + 没在下落 + 仰角大于0.05 + 仰角在安全范围内
    valid_climbing_state = intent_forward & is_resisted & not_falling & (pitch_metric > min_pitch_metric) & is_safe_pitch

    # 5. 发放奖励
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    reward = torch.where(valid_climbing_state, capped_pitch, torch.zeros_like(pitch_metric)) * stage_gate

    return reward

def front_legs_reach_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【前腿搭台奖励】
    当机器人处于抬头状态时，奖励前腿（Z轴）抬高。
    鼓励它把前脚尽可能举高，去够高台的台面。
    """
    robot = env.scene[asset_cfg.name]

    # 获取前脚在世界坐标系下的 Z 轴高度
    # 注意：这里依赖于在 params 中传入 asset_cfg=SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"])
    front_feet_indices = asset_cfg.body_ids
    front_feet_z = robot.data.body_pos_w[:, front_feet_indices, 2] # shape: (num_envs, 2)
    mean_front_z = torch.mean(front_feet_z, dim=1)

    # 获取机身高度
    root_z = robot.data.root_pos_w[:, 2]

    # 计算前脚相对于机身的高度差 (鼓励前脚比机身抬得更高)
    relative_z = mean_front_z - root_z

    # 触发条件：机身必须处于抬头状态 (pitch_metric > 0.05，约 3度以上)
    pitch_metric = -robot.data.projected_gravity_b[:, 0]
    is_pitching = pitch_metric > 0.05

    # 当抬头且前脚抬起时，奖励其相对高度
    reward = torch.where(is_pitching & (relative_z > -0.1), relative_z + 0.1, torch.zeros_like(relative_z))
    return reward


def front_legs_highstep_reach_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    sensor_cfg: SceneEntityCfg | None = None,
    min_cmd_x: float = 0.08,
    min_pitch_metric: float = 0.02,
    min_height_diff: float = 0.05,
    target_height_diff: float = 0.30,
    relative_lift_min: float = -0.34,
    relative_lift_target: float = -0.08,
    min_front_x: float = 0.20,
    target_front_x: float = 0.45,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    terrain_gate_floor: float = 0.25,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward front-leg lift/reach for high-step climbing before forward motion exists."""
    robot = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]

    front_foot_ids = robot.find_bodies(front_foot_names)[0]
    rear_foot_ids = robot.find_bodies(rear_foot_names)[0]
    front_feet_z = robot.data.body_pos_w[:, front_foot_ids, 2].mean(dim=1)
    rear_feet_z = robot.data.body_pos_w[:, rear_foot_ids, 2].mean(dim=1)
    height_diff = front_feet_z - rear_feet_z

    pitch_metric = -robot.data.projected_gravity_b[:, 0]
    safe_pitch = pitch_metric < 0.95

    denom = max(target_height_diff - min_height_diff, 1.0e-6)
    height_score = torch.clamp((height_diff - min_height_diff) / denom, min=0.0, max=1.0)
    root_z = robot.data.root_pos_w[:, 2]
    relative_front_lift = front_feet_z - root_z
    lift_score = torch.clamp(
        (relative_front_lift - relative_lift_min) / max(relative_lift_target - relative_lift_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    front_rel_w = robot.data.body_pos_w[:, front_foot_ids, :].mean(dim=1) - robot.data.root_pos_w
    front_rel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), front_rel_w)
    front_x_score = torch.clamp(
        (front_rel_b[:, 0] - min_front_x) / max(target_front_x - min_front_x, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    pitch_gate = torch.clamp((pitch_metric - min_pitch_metric) / 0.25, min=0.0, max=1.0)
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, min=0.0, max=1.0)
    terrain_gate = torch.ones_like(cmd_gate)
    if sensor_cfg is not None:
        terrain_gate, _, _ = _forward_highstep_terrain_gate(
            env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
        )
    task_gate = torch.clamp(
        terrain_gate_floor + (1.0 - terrain_gate_floor) * terrain_gate,
        min=0.0,
        max=1.0,
    )
    reach_score = 0.40 * lift_score * pitch_gate + 0.35 * height_score + 0.25 * front_x_score
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return task_gate * cmd_gate * safe_pitch * reach_score * stage_gate


def left_front_highstep_precontact_retraction_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    left_front_foot_name: str,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    retraction_start_x: float = 0.36,
    retraction_target_x: float = 0.30,
    relative_lift_min: float = -0.30,
    relative_lift_target: float = -0.12,
    clearance_release: float = 0.04,
    clearance_window: float = 0.12,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Retract the left-front foot only during the pre-contact high-step swing.

    The v1.12 rear-support Teacher has the desired rear-leg motion, but the
    frozen 00155 review showed one repeated residual failure: the left-front
    toe catches the vertical riser (8 of 14 completed attempts).  This term asks
    the lifted FL foot to stay 6 cm closer to the body until it clears the
    detected platform top.  It then fades out, leaving the unchanged symmetric
    reach/support objectives to place and load the foot on the platform.

    The narrow terrain, command, clearance, and pre-commit gates deliberately
    prevent this term from changing flat walking, the right-front leg, or any
    rear-leg phase.
    """
    asset = env.scene[asset_cfg.name]
    terrain_gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env,
        sensor_cfg,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, min=0.0, max=1.0)
    commit_gate = torch.clamp(
        _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names),
        min=0.0,
        max=1.0,
    )
    precommit_gate = torch.square(1.0 - commit_gate)

    fl_id = asset.find_bodies([left_front_foot_name])[0][0]
    fl_pos_w = asset.data.body_pos_w[:, fl_id, :]
    fl_rel_w = fl_pos_w - asset.data.root_pos_w
    fl_rel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), fl_rel_w)
    retraction_score = torch.clamp(
        (retraction_start_x - fl_rel_b[:, 0])
        / max(retraction_start_x - retraction_target_x, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    fl_rel_z = fl_pos_w[:, 2] - asset.data.root_pos_w[:, 2]
    lift_score = torch.clamp(
        (fl_rel_z - relative_lift_min)
        / max(relative_lift_target - relative_lift_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    fl_clearance = fl_pos_w[:, 2] - front_terrain_z
    below_release_gate = torch.clamp(
        (clearance_release - fl_clearance) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    stage_gate = _highstep_training_progress_gate(
        env, stage_start_update, stage_ramp_updates, num_steps_per_update
    )
    product = (
        terrain_gate
        * cmd_gate
        * precommit_gate
        * below_release_gate
        * lift_score
        * retraction_score
        * stage_gate
    )
    return product


def _highstep_monotonic_reward_release_latch(
    env: ManagerBasedRLEnv,
    crossing: torch.Tensor,
) -> torch.Tensor:
    """Return the per-env active mask for a reward-only monotonic release latch.

    This state is deliberately attached only to the training environment and is
    neither observed by the policy nor serialized by the checkpoint/runtime
    sidecar.  An episode-length decrease identifies an individual environment
    reset even when the first reward evaluation occurs after the counter has
    already advanced from zero.
    """
    current_length = env.episode_length_buf.detach()
    released = getattr(env, "_highstep_fl_precontact_reward_released", None)
    previous_length = getattr(env, "_highstep_fl_precontact_reward_previous_episode_length", None)
    if (
        not isinstance(released, torch.Tensor)
        or released.shape != crossing.shape
        or released.device != crossing.device
    ):
        released = torch.zeros_like(crossing, dtype=torch.bool)
        previous_length = current_length.clone()
    elif not isinstance(previous_length, torch.Tensor) or previous_length.shape != current_length.shape:
        raise RuntimeError("FL pre-contact reward latch lost its per-env episode-length state")
    else:
        reset_mask = current_length < previous_length.to(device=current_length.device)
        released = released.to(device=crossing.device)
        released = torch.where(reset_mask.to(device=crossing.device), False, released)

    released = released | crossing.detach().to(dtype=torch.bool)
    env._highstep_fl_precontact_reward_released = released
    env._highstep_fl_precontact_reward_previous_episode_length = current_length.clone()
    return (~released).to(dtype=torch.float32)


def left_front_highstep_precontact_reward_geometry_contract_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    left_front_foot_name: str,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    low_lift_min: float = 0.02,
    low_lift_target: float = 0.18,
    retraction_target_x: float = 0.30,
    retraction_sigma: float = 0.06,
    clearance_release: float = 0.04,
    clearance_window: float = 0.12,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """v1.12.3 FL-only pre-contact geometry contract with reward-only release state."""
    asset = env.scene[asset_cfg.name]
    terrain_gate, detected_step_height, front_terrain_z = _forward_highstep_terrain_gate(
        env,
        sensor_cfg,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, min=0.0, max=1.0)
    commit_gate = torch.clamp(
        _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names),
        min=0.0,
        max=1.0,
    )
    precommit_gate = torch.square(1.0 - commit_gate)

    fl_id = asset.find_bodies([left_front_foot_name])[0][0]
    fl_pos_w = asset.data.body_pos_w[:, fl_id, :]
    fl_rel_w = fl_pos_w - asset.data.root_pos_w
    fl_rel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), fl_rel_w)

    low_plane_z = front_terrain_z - detected_step_height
    low_plane_clearance = fl_pos_w[:, 2] - low_plane_z
    lift_score = torch.clamp(
        (low_plane_clearance - low_lift_min) / max(low_lift_target - low_lift_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    retraction_score = torch.exp(
        -0.5 - torch.square((fl_rel_b[:, 0] - retraction_target_x) / max(retraction_sigma, 1.0e-6))
    )

    fl_clearance = fl_pos_w[:, 2] - front_terrain_z
    below_release_gate = torch.clamp(
        (clearance_release - fl_clearance) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    active_latch = _highstep_monotonic_reward_release_latch(
        env, fl_clearance >= clearance_release
    )
    stage_gate = _highstep_training_progress_gate(
        env, stage_start_update, stage_ramp_updates, num_steps_per_update
    )
    product = (
        terrain_gate
        * cmd_gate
        * precommit_gate
        * below_release_gate
        * lift_score
        * retraction_score
        * stage_gate
        * active_latch
    )
    if getattr(env, "_highstep_v1123_preflight_capture", False):
        env._highstep_v1123_preflight_frame = {
            "terrain_gate": terrain_gate.detach(),
            "cmd_gate": cmd_gate.detach(),
            "precommit_gate": precommit_gate.detach(),
            "below_release_gate": below_release_gate.detach(),
            "lift_score": lift_score.detach(),
            "retraction_score": retraction_score.detach(),
            "active_latch": active_latch.detach(),
            "final_product": product.detach(),
            "fl_world_z": fl_pos_w[:, 2].detach(),
            "fl_body_x": fl_rel_b[:, 0].detach(),
            "front_terrain_z": front_terrain_z.detach(),
            "low_plane_z": low_plane_z.detach(),
            "fl_clearance": fl_clearance.detach(),
        }
    return product


def _highstep_masked_ray_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if mask is None or not bool(torch.any(mask).item()):
        return fallback
    selected = values[:, mask]
    valid = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
    valid_count = valid.float().sum(dim=1)
    selected_sum = torch.where(valid, selected, torch.zeros_like(selected)).sum(dim=1)
    mean = selected_sum / torch.clamp(valid_count, min=1.0)
    return torch.where(valid_count > 0.0, mean, fallback)


def _forward_highstep_terrain_gate(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    front_x_min: float,
    rear_x_max: float,
    max_abs_y: float,
    height_threshold: float,
    height_gate_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_starts = sensor.ray_starts[0]
    ray_x = ray_starts[:, 0]
    ray_y = ray_starts[:, 1]
    side_mask = torch.abs(ray_y) <= max_abs_y
    front_mask = (ray_x >= front_x_min) & side_mask
    rear_mask = (ray_x <= rear_x_max) & side_mask

    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    sensor_z = sensor.data.pos_w[:, 2]
    front_z = _highstep_masked_ray_mean(ray_hits_z, front_mask, sensor_z)
    rear_z = _highstep_masked_ray_mean(ray_hits_z, rear_mask, front_z)
    height_delta = front_z - rear_z
    gate = torch.clamp((height_delta - height_threshold) / (height_gate_width + 1.0e-6), 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    return gate, height_delta, front_z


def highstep_box_default_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_cmd_x: float = 0.10,
    command_gate_width: float = 0.25,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    hold_scale: float = 16.0,
    highstep_scale: float = 0.08,
) -> torch.Tensor:
    """Keep box joints at default except during an active forward high-step climb."""
    asset: Articulation = env.scene[asset_cfg.name]
    box_error = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )

    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / max(command_gate_width, 1.0e-6), 0.0, 1.0)
    highstep_gate = terrain_gate * cmd_gate

    scale = hold_scale * (1.0 - highstep_gate) + highstep_scale * highstep_gate
    return box_error * scale


def _front_feet_highstep_commit_gate(
    asset: RigidObject,
    front_foot_names: list[str] | None,
    rear_foot_names: list[str] | None,
    min_height_diff: float = 0.08,
    target_height_diff: float = 0.26,
    min_front_x: float = 0.12,
    target_front_x: float = 0.36,
    min_pitch_metric: float = 0.02,
    target_pitch_metric: float = 0.28,
) -> torch.Tensor:
    """Behavior gate for the moment when the front feet have committed onto a high step."""
    if front_foot_names is None or rear_foot_names is None:
        return torch.zeros(asset.data.root_pos_w.shape[0], device=asset.data.root_pos_w.device)

    front_ids = asset.find_bodies(front_foot_names)[0]
    rear_ids = asset.find_bodies(rear_foot_names)[0]

    front_pos_w = asset.data.body_pos_w[:, front_ids, :].mean(dim=1)
    rear_pos_w = asset.data.body_pos_w[:, rear_ids, :].mean(dim=1)
    height_diff = front_pos_w[:, 2] - rear_pos_w[:, 2]
    height_score = torch.clamp(
        (height_diff - min_height_diff) / max(target_height_diff - min_height_diff, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    front_rel_w = front_pos_w - asset.data.root_pos_w
    front_rel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), front_rel_w)
    front_x_score = torch.clamp(
        (front_rel_b[:, 0] - min_front_x) / max(target_front_x - min_front_x, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    pitch_metric = -asset.data.projected_gravity_b[:, 0]
    pitch_score = torch.clamp(
        (pitch_metric - min_pitch_metric) / max(target_pitch_metric - min_pitch_metric, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    return height_score * (0.45 + 0.55 * front_x_score) * (0.35 + 0.65 * pitch_score)


def _highstep_on_top_phase_gates(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str] | None,
    rear_foot_names: list[str] | None,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    second_clear_min: float = 0.72,
    min_base_clearance: float = 0.35,
    target_base_clearance: float = 0.43,
    commit_gate_scale: float = 0.85,
) -> dict[str, torch.Tensor]:
    """Detect when the high-step task has finished so entry/prep rewards can shut off."""
    terrain_gate, height_delta, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    task_gate = torch.maximum(terrain_gate, commit_gate_scale * commit_gate)

    if rear_foot_names is None:
        zero = torch.zeros_like(task_gate)
        return {
            "terrain_gate": terrain_gate,
            "height_delta": height_delta,
            "front_terrain_z": front_terrain_z,
            "commit_gate": commit_gate,
            "task_gate": task_gate,
            "second_ready_gate": zero,
            "base_clearance_score": zero,
            "on_top_gate": zero,
            "entry_allowed_gate": torch.ones_like(task_gate),
        }

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_z = asset.data.body_pos_w[:, rear_foot_ids, 2]
    rear_clearance = rear_z - (front_terrain_z[:, None] + clearance_margin)
    rear_clearance_score = torch.clamp(
        (rear_clearance + clearance_window) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_score = torch.min(rear_clearance_score, dim=1).values
    second_ready_gate = torch.clamp(
        (second_score - second_clear_min) / max(1.0 - second_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    base_clearance = asset.data.root_pos_w[:, 2] - front_terrain_z
    base_clearance_score = torch.clamp(
        (base_clearance - min_base_clearance) / max(target_base_clearance - min_base_clearance, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    on_top_gate = torch.clamp(task_gate * second_ready_gate * base_clearance_score, min=0.0, max=1.0)
    return {
        "terrain_gate": terrain_gate,
        "height_delta": height_delta,
        "front_terrain_z": front_terrain_z,
        "commit_gate": commit_gate,
        "task_gate": task_gate,
        "second_ready_gate": second_ready_gate,
        "base_clearance_score": base_clearance_score,
        "on_top_gate": on_top_gate,
        "entry_allowed_gate": 1.0 - on_top_gate,
    }


def _ensure_highstep_rear_branch_buffers(env: ManagerBasedRLEnv):
    """Create per-env buffers used to diagnose rear-foot branch choice."""
    device = torch.device(env.device)
    needs_init = (
        not hasattr(env, "_highstep_rear_branch_lead")
        or not hasattr(env, "_highstep_rear_branch_lead_steps")
        or not hasattr(env, "_highstep_rear_branch_sample_steps")
        or not hasattr(env, "_highstep_lead_support_drive_sample_steps")
        or not hasattr(env, "_highstep_lead_support_drive_lift_velocity_sum")
        or not hasattr(env, "_highstep_post_lead_drive_lift_velocity_sum")
        or not hasattr(env, "_highstep_rear_approach_width_sample_steps")
        or not hasattr(env, "_highstep_rear_approach_center_deficit_max")
        or not hasattr(env, "_highstep_rear_motion_center_deficit_max")
        or not hasattr(env, "_highstep_support_stability_sample_steps")
        or not hasattr(env, "_highstep_front_lift_guard_sample_steps")
        or not hasattr(env, "_highstep_fl_forward_flat_sample_steps")
        or not hasattr(env, "_highstep_post_clear_origin_valid")
        or not hasattr(env, "_highstep_post_clear_origin_rear_pos_w")
        or not hasattr(env, "_highstep_post_clear_heading_w")
        or not hasattr(env, "_highstep_post_clear_origin_step")
        or not hasattr(env, "_highstep_post_clear_stall_near_edge_counter")
        or env._highstep_rear_branch_lead.shape[0] != env.num_envs
        or env._highstep_rear_branch_lead.device != device
    )
    if needs_init:
        env._highstep_rear_branch_lead = torch.full(
            (env.num_envs,), -1, dtype=torch.int64, device=device
        )
        env._highstep_rear_branch_commit_steps = torch.zeros(env.num_envs, device=device)
        env._highstep_rear_branch_lead_steps = torch.zeros(env.num_envs, device=device)
        env._highstep_rear_branch_one_sided_steps = torch.zeros(env.num_envs, device=device)
        env._highstep_rear_branch_second_clear_steps = torch.zeros(env.num_envs, device=device)
        env._highstep_post_clear_origin_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device
        )
        env._highstep_post_clear_origin_rear_pos_w = torch.zeros(
            (env.num_envs, 2, 2), device=device
        )
        env._highstep_post_clear_heading_w = torch.zeros((env.num_envs, 2), device=device)
        env._highstep_post_clear_origin_step = torch.zeros(env.num_envs, device=device)
        for buffer_name in (
            "_highstep_rear_branch_sample_steps",
            "_highstep_rear_branch_active_steps",
            "_highstep_rear_branch_soft_active_steps",
            "_highstep_rear_branch_lead_candidate_steps",
            "_highstep_rear_branch_active_gate_sum",
            "_highstep_rear_branch_active_gate_max",
            "_highstep_rear_branch_task_gate_sum",
            "_highstep_rear_branch_task_gate_max",
            "_highstep_rear_branch_terrain_gate_sum",
            "_highstep_rear_branch_terrain_gate_max",
            "_highstep_rear_branch_commit_gate_sum",
            "_highstep_rear_branch_commit_gate_max",
            "_highstep_rear_branch_cmd_gate_sum",
            "_highstep_rear_branch_cmd_gate_max",
            "_highstep_rear_branch_first_score_sum",
            "_highstep_rear_branch_first_score_max",
            "_highstep_rear_branch_second_score_sum",
            "_highstep_rear_branch_second_score_max",
            "_highstep_post_lead_stall_sample_steps",
            "_highstep_post_lead_stall_base_sum",
            "_highstep_post_lead_stall_base_max",
            "_highstep_post_lead_stall_second_low_sum",
            "_highstep_post_lead_stall_progress_low_sum",
            "_highstep_post_lead_stall_modifier_sum",
            "_highstep_post_lead_stall_signal_sum",
            "_highstep_post_lead_stall_signal_max",
            "_highstep_lead_support_drive_sample_steps",
            "_highstep_lead_support_drive_gate_sum",
            "_highstep_lead_support_drive_lead_sum",
            "_highstep_lead_support_drive_height_sum",
            "_highstep_lead_support_drive_height_active_sum",
            "_highstep_lead_support_drive_progress_sum",
            "_highstep_lead_support_drive_lift_velocity_sum",
            "_highstep_lead_support_drive_lift_velocity_active_sum",
            "_highstep_lead_support_drive_body_sum",
            "_highstep_lead_support_drive_body_active_sum",
            "_highstep_lead_support_drive_signal_sum",
            "_highstep_lead_support_drive_signal_max",
            "_highstep_lead_support_drive_height_max",
            "_highstep_lead_support_drive_lift_velocity_max",
            "_highstep_lead_support_drive_body_max",
            "_highstep_post_lead_drive_sample_steps",
            "_highstep_post_lead_drive_gate_sum",
            "_highstep_post_lead_drive_lead_sum",
            "_highstep_post_lead_drive_height_sum",
            "_highstep_post_lead_drive_lift_velocity_sum",
            "_highstep_post_lead_drive_progress_sum",
            "_highstep_post_lead_drive_second_sum",
            "_highstep_post_lead_drive_stage_gate_sum",
            "_highstep_post_lead_drive_signal_sum",
            "_highstep_post_lead_drive_signal_max",
            "_highstep_post_clear_recovery_sample_steps",
            "_highstep_post_clear_recovery_gate_sum",
            "_highstep_post_clear_recovery_posture_sum",
            "_highstep_post_clear_recovery_base_clearance_sum",
            "_highstep_post_clear_recovery_forward_sum",
            "_highstep_post_clear_recovery_rear_advance_sum",
            "_highstep_post_clear_recovery_signal_sum",
            "_highstep_post_clear_recovery_signal_max",
            "_highstep_post_clear_stall_near_edge_counter",
            "_highstep_post_clear_stall_sample_steps",
            "_highstep_post_clear_stall_margin_sum",
            "_highstep_post_clear_stall_counter_max",
            "_highstep_post_clear_stall_signal_sum",
            "_highstep_scanner_pretrigger_sample_steps",
            "_highstep_scanner_pretrigger_gate_sum",
            "_highstep_scanner_pretrigger_terrain_gate_sum",
            "_highstep_scanner_pretrigger_commit_gate_sum",
            "_highstep_scanner_pretrigger_low_cmd_gate_sum",
            "_highstep_scanner_pretrigger_front_lift_sum",
            "_highstep_scanner_pretrigger_uncommanded_vel_sum",
            "_highstep_scanner_pretrigger_signal_sum",
            "_highstep_scanner_pretrigger_signal_max",
            "_highstep_rear_approach_width_sample_steps",
            "_highstep_rear_approach_width_active_steps",
            "_highstep_rear_approach_width_gate_sum",
            "_highstep_rear_approach_width_terrain_gate_sum",
            "_highstep_rear_approach_width_commit_gate_sum",
            "_highstep_rear_approach_width_cmd_gate_sum",
            "_highstep_rear_approach_width_sum",
            "_highstep_rear_approach_width_active_sum",
            "_highstep_rear_approach_min_abs_y_sum",
            "_highstep_rear_approach_min_abs_y_active_sum",
            "_highstep_rear_approach_width_violation_steps",
            "_highstep_rear_approach_center_violation_steps",
            "_highstep_rear_approach_center_deficit_max",
            "_highstep_rear_approach_signal_sum",
            "_highstep_rear_approach_signal_max",
            "_highstep_rear_motion_width_sample_steps",
            "_highstep_rear_motion_width_active_steps",
            "_highstep_rear_motion_width_gate_sum",
            "_highstep_rear_motion_width_terrain_gate_sum",
            "_highstep_rear_motion_width_commit_gate_sum",
            "_highstep_rear_motion_width_cmd_gate_sum",
            "_highstep_rear_motion_width_second_sum",
            "_highstep_rear_motion_width_sum",
            "_highstep_rear_motion_width_active_sum",
            "_highstep_rear_motion_min_abs_y_sum",
            "_highstep_rear_motion_min_abs_y_active_sum",
            "_highstep_rear_motion_width_violation_steps",
            "_highstep_rear_motion_center_violation_steps",
            "_highstep_rear_motion_center_deficit_max",
            "_highstep_rear_motion_signal_sum",
            "_highstep_rear_motion_signal_max",
            "_highstep_support_stability_sample_steps",
            "_highstep_support_stability_active_steps",
            "_highstep_support_stability_gate_sum",
            "_highstep_support_stability_front_contact_sum",
            "_highstep_support_stability_front_slip_sum",
            "_highstep_support_stability_posture_rate_sum",
            "_highstep_support_stability_rear_lag_sum",
            "_highstep_support_stability_support_width_sum",
            "_highstep_support_stability_signal_sum",
            "_highstep_support_stability_signal_max",
            "_highstep_front_lift_guard_sample_steps",
            "_highstep_front_lift_guard_gate_sum",
            "_highstep_front_lift_guard_signal_sum",
            "_highstep_front_lift_guard_signal_max",
            "_highstep_front_lift_guard_max_lift_sum",
            "_highstep_front_lift_guard_asym_sum",
            "_highstep_front_lift_guard_height_excess_sum",
            "_highstep_front_lift_guard_asym_excess_sum",
            "_highstep_front_lift_guard_terrain_gate_sum",
            "_highstep_front_lift_guard_commit_gate_sum",
            "_highstep_fl_forward_flat_sample_steps",
            "_highstep_fl_forward_flat_active_gate_sum",
            "_highstep_fl_forward_flat_forward_gate_sum",
            "_highstep_fl_forward_flat_backward_gate_sum",
            "_highstep_fl_forward_flat_lateral_gate_sum",
            "_highstep_fl_forward_flat_yaw_gate_sum",
            "_highstep_fl_forward_flat_flat_gate_sum",
            "_highstep_fl_forward_flat_highstep_relief_sum",
            "_highstep_fl_forward_flat_fl_lift_sum",
            "_highstep_fl_forward_flat_fr_lift_sum",
            "_highstep_fl_forward_flat_fl_minus_fr_sum",
            "_highstep_fl_forward_flat_height_excess_sum",
            "_highstep_fl_forward_flat_asym_excess_sum",
            "_highstep_fl_forward_flat_overlift_sum",
            "_highstep_fl_forward_flat_penalty_sum",
            "_highstep_fl_forward_flat_penalty_max",
        ):
            setattr(env, buffer_name, torch.zeros(env.num_envs, device=device))
        for buffer_name in (
            "_highstep_rear_approach_min_abs_y_active_min",
            "_highstep_rear_motion_min_abs_y_active_min",
            "_highstep_post_clear_stall_margin_min",
        ):
            setattr(env, buffer_name, torch.full((env.num_envs,), 10.0, device=device))


def _update_highstep_rear_branch_state(
    env: ManagerBasedRLEnv,
    active_gate: torch.Tensor,
    rear_clearance_score: torch.Tensor,
    lead_threshold: float = 0.62,
    second_clear_threshold: float = 0.72,
    one_sided_gap: float = 0.22,
    active_threshold: float = 0.15,
    task_gate: torch.Tensor | None = None,
    terrain_gate: torch.Tensor | None = None,
    commit_gate: torch.Tensor | None = None,
    cmd_gate: torch.Tensor | None = None,
) -> None:
    """Track which rear foot leads and whether the other rear foot is left behind."""
    _ensure_highstep_rear_branch_buffers(env)
    active_gate = torch.clamp(active_gate.detach(), min=0.0, max=1.0)
    active = active_gate > active_threshold
    if rear_clearance_score.shape[1] < 2:
        return

    rear_clearance_score = torch.clamp(rear_clearance_score.detach(), min=0.0, max=1.0)
    rl_score = rear_clearance_score[:, 0]
    rr_score = rear_clearance_score[:, 1]
    rl_high = rl_score > lead_threshold
    rr_high = rr_score > lead_threshold
    lead_gap = 0.5 * one_sided_gap
    no_lead = env._highstep_rear_branch_lead < 0

    rl_first = active & no_lead & rl_high & ((rl_score - rr_score) > lead_gap)
    rr_first = active & no_lead & rr_high & ((rr_score - rl_score) > lead_gap)
    both_first = active & no_lead & (rl_high | rr_high) & (~rl_first) & (~rr_first)
    new_lead = rl_first | rr_first | both_first
    env._highstep_rear_branch_lead = torch.where(
        rl_first,
        torch.zeros_like(env._highstep_rear_branch_lead),
        env._highstep_rear_branch_lead,
    )
    env._highstep_rear_branch_lead = torch.where(
        rr_first,
        torch.ones_like(env._highstep_rear_branch_lead),
        env._highstep_rear_branch_lead,
    )
    env._highstep_rear_branch_lead = torch.where(
        both_first,
        torch.full_like(env._highstep_rear_branch_lead, 2),
        env._highstep_rear_branch_lead,
    )

    def _gate_or_zero(gate: torch.Tensor | None) -> torch.Tensor:
        if gate is None:
            return torch.zeros_like(active_gate)
        return torch.clamp(gate.detach(), min=0.0, max=1.0)

    task_gate = _gate_or_zero(task_gate)
    terrain_gate = _gate_or_zero(terrain_gate)
    commit_gate = _gate_or_zero(commit_gate)
    cmd_gate = _gate_or_zero(cmd_gate)
    max_score = torch.maximum(rl_score, rr_score)
    min_score = torch.minimum(rl_score, rr_score)
    lead_candidate = active & (max_score > lead_threshold)

    env._highstep_rear_branch_sample_steps += torch.ones_like(active_gate)
    env._highstep_rear_branch_active_steps += active.float()
    env._highstep_rear_branch_soft_active_steps += (active_gate > 0.01).float()
    env._highstep_rear_branch_lead_candidate_steps += lead_candidate.float()
    env._highstep_rear_branch_active_gate_sum += active_gate
    env._highstep_rear_branch_active_gate_max = torch.maximum(env._highstep_rear_branch_active_gate_max, active_gate)
    env._highstep_rear_branch_task_gate_sum += task_gate
    env._highstep_rear_branch_task_gate_max = torch.maximum(env._highstep_rear_branch_task_gate_max, task_gate)
    env._highstep_rear_branch_terrain_gate_sum += terrain_gate
    env._highstep_rear_branch_terrain_gate_max = torch.maximum(env._highstep_rear_branch_terrain_gate_max, terrain_gate)
    env._highstep_rear_branch_commit_gate_sum += commit_gate
    env._highstep_rear_branch_commit_gate_max = torch.maximum(env._highstep_rear_branch_commit_gate_max, commit_gate)
    env._highstep_rear_branch_cmd_gate_sum += cmd_gate
    env._highstep_rear_branch_cmd_gate_max = torch.maximum(env._highstep_rear_branch_cmd_gate_max, cmd_gate)
    env._highstep_rear_branch_first_score_sum += max_score
    env._highstep_rear_branch_first_score_max = torch.maximum(env._highstep_rear_branch_first_score_max, max_score)
    env._highstep_rear_branch_second_score_sum += min_score
    env._highstep_rear_branch_second_score_max = torch.maximum(env._highstep_rear_branch_second_score_max, min_score)

    env._highstep_rear_branch_commit_steps += active.float()
    env._highstep_rear_branch_lead_steps = torch.where(
        new_lead,
        env._highstep_rear_branch_commit_steps,
        env._highstep_rear_branch_lead_steps,
    )
    one_sided = active & (max_score > lead_threshold) & ((max_score - min_score) > one_sided_gap)
    second_clear = active & (min_score > second_clear_threshold)
    env._highstep_rear_branch_one_sided_steps += one_sided.float()
    env._highstep_rear_branch_second_clear_steps += second_clear.float()


def _highstep_rear_stage_context(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    commit_gate_scale: float = 0.90,
) -> dict[str, torch.Tensor]:
    """Shared high-step rear-branch signals after the branch tracker has run."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    task_gate = torch.maximum(terrain_gate, commit_gate_scale * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_feet_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    rear_clearance = rear_feet_pos_w[..., 2] - (front_terrain_z[:, None] + clearance_margin)
    rear_clearance_score = torch.clamp(
        (rear_clearance + clearance_window) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    _ensure_highstep_rear_branch_buffers(env)
    lead = env._highstep_rear_branch_lead
    lead_known = lead >= 0
    lead_index = torch.clamp(lead, min=0, max=1)
    lead_score = torch.gather(rear_clearance_score, 1, lead_index.unsqueeze(1)).squeeze(1)
    lead_score = torch.where(lead == 2, torch.max(rear_clearance_score, dim=1).values, lead_score)
    second_score = torch.min(rear_clearance_score, dim=1).values
    lead_elapsed_steps = torch.clamp(
        env._highstep_rear_branch_commit_steps - env._highstep_rear_branch_lead_steps,
        min=0.0,
    )

    return {
        "asset": asset,
        "task_gate": task_gate,
        "terrain_gate": terrain_gate,
        "commit_gate": commit_gate,
        "front_terrain_z": front_terrain_z,
        "cmd_gate": cmd_gate,
        "rear_clearance_score": rear_clearance_score,
        "lead": lead,
        "lead_known": lead_known.float(),
        "lead_index": lead_index,
        "lead_score": lead_score,
        "second_score": second_score,
        "lead_elapsed_steps": lead_elapsed_steps,
        "first_score": torch.max(rear_clearance_score, dim=1).values,
    }


def front_feet_highstep_clearance_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str] | None = None,
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.03,
    clearance_window: float = 0.12,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward front feet clearing the high terrain detected in front of the robot."""
    robot = env.scene[asset_cfg.name]
    gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    front_foot_ids = robot.find_bodies(front_foot_names)[0]
    front_feet_z = robot.data.body_pos_w[:, front_foot_ids, 2].mean(dim=1)
    clearance = front_feet_z - (front_terrain_z + clearance_margin)
    clearance_score = torch.clamp((clearance + clearance_window) / max(clearance_window, 1.0e-6), 0.0, 1.0)

    pitch_metric = -robot.data.projected_gravity_b[:, 0]
    safe_pitch = pitch_metric < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * clearance_score * safe_pitch * stage_gate


def rear_feet_highstep_clearance_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = -0.03,
    clearance_window: float = 0.22,
    commit_gate_scale: float = 0.80,
    single_rear_weight: float = 0.65,
    min_rear_weight: float = 0.35,
    single_rear_temperature: float = 0.12,
    rear_x_min: float = -0.48,
    rear_x_target: float = -0.08,
    rear_x_weight: float = 0.25,
    rear_x_single_weight: float = 0.35,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward rear feet beginning to climb after the front feet have committed to a high step."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    gate = torch.maximum(terrain_gate, commit_gate_scale * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_feet_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    rear_clearance = rear_feet_pos_w[..., 2] - (front_terrain_z[:, None] + clearance_margin)
    rear_clearance_score = torch.clamp(
        (rear_clearance + clearance_window) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    single_rear_score = torch.max(rear_clearance_score, dim=1).values
    min_rear_score = torch.min(rear_clearance_score, dim=1).values
    both_rear_score = torch.mean(rear_clearance_score, dim=1)
    single_w = max(0.0, min(single_rear_weight, 1.0))
    min_w = max(0.0, min(min_rear_weight, 1.0 - single_w))
    mean_w = 1.0 - single_w - min_w
    rear_height_score = single_w * single_rear_score + min_w * min_rear_score + mean_w * both_rear_score

    rear_rel_w = rear_feet_pos_w - asset.data.root_pos_w[:, None, :]
    num_rear_feet = rear_rel_w.shape[1]
    heading_quat = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, num_rear_feet, -1)
    rear_rel_b = quat_apply_inverse(
        heading_quat.reshape(-1, 4),
        rear_rel_w.reshape(-1, 3),
    ).reshape(rear_rel_w.shape)
    rear_x_each_score = torch.clamp(
        (rear_rel_b[..., 0] - rear_x_min) / max(rear_x_target - rear_x_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    rear_x_single_score = torch.max(rear_x_each_score, dim=1).values
    rear_x_mean_score = torch.mean(rear_x_each_score, dim=1)
    rear_x_single_w = max(0.0, min(rear_x_single_weight, 1.0))
    rear_x_score = rear_x_single_w * rear_x_single_score + (1.0 - rear_x_single_w) * rear_x_mean_score

    score = (1.0 - rear_x_weight) * rear_height_score + rear_x_weight * rear_x_score
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * score * safe_pitch * stage_gate


def rear_first_foot_highstep_preclearance_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.06,
    clearance_window: float = 0.16,
    first_clear_min: float = 0.55,
    second_clear_suppress_min: float = 0.72,
    commit_gate_min: float = 0.25,
    commit_gate_scale: float = 0.90,
    terrain_commit_min: float = 0.22,
    terrain_commit_floor: float = 0.0,
    max_roll_metric: float = 0.28,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward the first rear foot lifting above the step before a one-sided stall forms."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    task_gate = torch.maximum(terrain_gate, commit_gate_scale * commit_gate)
    front_commit_ready = torch.clamp(
        (commit_gate - commit_gate_min) / max(1.0 - commit_gate_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    terrain_commit_ready = torch.clamp(
        (terrain_gate - terrain_commit_min) / max(1.0 - terrain_commit_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    terrain_floor = max(0.0, min(float(terrain_commit_floor), 1.0))
    commit_ready = torch.maximum(front_commit_ready, terrain_floor * terrain_commit_ready)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_feet_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    rear_clearance = rear_feet_pos_w[..., 2] - (front_terrain_z[:, None] + clearance_margin)
    rear_clearance_score = torch.clamp(
        (rear_clearance + clearance_window) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    first_score = torch.max(rear_clearance_score, dim=1).values
    second_score = torch.min(rear_clearance_score, dim=1).values
    first_clear_score = torch.clamp(
        (first_score - first_clear_min) / max(1.0 - first_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_clear_score = torch.clamp(
        (second_score - second_clear_suppress_min) / max(1.0 - second_clear_suppress_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    first_needed_gate = 1.0 - second_clear_score

    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_gate = torch.clamp((max_roll_metric - roll_metric) / max(max_roll_metric, 1.0e-6), min=0.0, max=1.0)
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return (
        task_gate
        * commit_ready
        * cmd_gate
        * first_clear_score
        * first_needed_gate
        * roll_gate
        * safe_pitch
        * stage_gate
    )


def rear_feet_under_step_after_commit_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.02,
    under_window: float = 0.16,
    commit_gate_floor: float = 0.25,
    nominal_base_height: float = 0.44,
    min_distance: float = 0.16,
    target_distance: float = 0.56,
    min_height_gain: float = -0.02,
    target_height_gain: float = 0.08,
    min_forward_vel: float = 0.06,
    relief_forward_vel: float = 0.22,
    stall_floor: float = 0.0,
    worst_rear_weight: float = 0.75,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize rear feet staying below the step only when the climb is stalled."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    gate = torch.maximum(terrain_gate * commit_gate, commit_gate_floor * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_z = asset.data.body_pos_w[:, rear_foot_ids, 2]
    target_z = front_terrain_z[:, None] + clearance_margin
    rear_under_score = torch.clamp((target_z - rear_z) / max(under_window, 1.0e-6), min=0.0, max=1.0)
    mean_under_score = torch.mean(rear_under_score, dim=1)
    worst_under_score = torch.max(rear_under_score, dim=1).values
    worst_w = max(0.0, min(worst_rear_weight, 1.0))
    under_score = worst_w * worst_under_score + (1.0 - worst_w) * mean_under_score

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_gain_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    progress_score = distance_score * (0.40 + 0.60 * height_gain_score)
    position_stall_gate = torch.clamp(1.0 - progress_score, 0.0, 1.0)

    forward_vel = asset.data.root_lin_vel_b[:, 0]
    velocity_stall_gate = torch.clamp(
        (relief_forward_vel - forward_vel) / max(relief_forward_vel - min_forward_vel, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    stall_gate = torch.clamp(
        stall_floor + (1.0 - stall_floor) * position_stall_gate * velocity_stall_gate,
        min=0.0,
        max=1.0,
    )

    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * under_score * stall_gate * safe_pitch * stage_gate


def rear_second_foot_highstep_clearance_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    first_rear_gate_min: float = 0.55,
    commit_gate_scale: float = 0.90,
    nominal_base_height: float = 0.44,
    min_distance: float = 0.18,
    target_distance: float = 0.60,
    min_height_gain: float = -0.02,
    target_height_gain: float = 0.08,
    progress_floor: float = 0.20,
    max_roll_metric: float = 0.24,
    branch_lead_threshold: float = 0.62,
    branch_second_clear_threshold: float = 0.72,
    branch_one_sided_gap: float = 0.22,
    branch_active_threshold: float = 0.12,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward the second rear foot clearing the step after either rear foot has reached the platform."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    gate = torch.maximum(terrain_gate, commit_gate_scale * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_feet_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    rear_clearance = rear_feet_pos_w[..., 2] - (front_terrain_z[:, None] + clearance_margin)
    rear_clearance_score = torch.clamp(
        (rear_clearance + clearance_window) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    _update_highstep_rear_branch_state(
        env,
        cmd_gate * gate,
        rear_clearance_score,
        lead_threshold=branch_lead_threshold,
        second_clear_threshold=branch_second_clear_threshold,
        one_sided_gap=branch_one_sided_gap,
        active_threshold=branch_active_threshold,
        task_gate=gate,
        terrain_gate=terrain_gate,
        commit_gate=commit_gate,
        cmd_gate=cmd_gate,
    )

    first_rear_score = torch.max(rear_clearance_score, dim=1).values
    second_rear_score = torch.min(rear_clearance_score, dim=1).values
    first_rear_gate = torch.clamp(
        (first_rear_score - first_rear_gate_min) / max(1.0 - first_rear_gate_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_gain_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    progress_score = distance_score * (0.35 + 0.65 * height_gain_score)
    progress_gate = torch.clamp(progress_floor + (1.0 - progress_floor) * progress_score, 0.0, 1.0)

    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_gate = torch.clamp((max_roll_metric - roll_metric) / max(max_roll_metric, 1.0e-6), min=0.0, max=1.0)
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * first_rear_gate * second_rear_score * progress_gate * roll_gate * safe_pitch * stage_gate


def second_rear_clear_deadline_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    second_clear_min: float = 0.62,
    deadline_steps: float = 32.0,
    deadline_grace_steps: float = 6.0,
    commit_gate_scale: float = 0.90,
    max_roll_metric: float = 0.28,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward the second rear foot clearing soon after the first rear foot leads."""
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale,
    )
    asset = ctx["asset"]
    second_clear_score = torch.clamp(
        (ctx["second_score"] - second_clear_min) / max(1.0 - second_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    deadline_score = torch.clamp(
        (deadline_steps - ctx["lead_elapsed_steps"]) / max(deadline_steps - deadline_grace_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_gate = torch.clamp((max_roll_metric - roll_metric) / max(max_roll_metric, 1.0e-6), min=0.0, max=1.0)
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return (
        ctx["task_gate"]
        * ctx["cmd_gate"]
        * ctx["lead_known"]
        * second_clear_score
        * deadline_score
        * roll_gate
        * safe_pitch
        * stage_gate
    )


def one_sided_rear_stall_time_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    lead_threshold: float = 0.45,
    one_sided_gap: float = 0.16,
    grace_steps: float = 12.0,
    ramp_steps: float = 24.0,
    commit_gate_scale: float = 0.90,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize long one-rear-foot hanging after a rear branch has been chosen."""
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale,
    )
    asset = ctx["asset"]
    first_score = ctx["first_score"]
    second_score = ctx["second_score"]
    one_sided_now = (first_score > lead_threshold) & ((first_score - second_score) > one_sided_gap)
    stall_time_score = torch.clamp(
        (env._highstep_rear_branch_one_sided_steps - grace_steps) / max(ramp_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return (
        ctx["task_gate"]
        * ctx["cmd_gate"]
        * ctx["lead_known"]
        * one_sided_now.float()
        * stall_time_score
        * safe_pitch
        * stage_gate
    )


def lead_rear_support_drive_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    box_joint_names: dict[str, str],
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    rear_push_target: float = 0.003,
    target_std: float = 0.016,
    lead_box_floor: float = 0.25,
    nominal_base_height: float = 0.44,
    min_distance: float = 0.20,
    target_distance: float = 0.62,
    min_height_gain: float = -0.02,
    target_height_gain: float = 0.09,
    target_forward_vel: float = 0.28,
    body_drive_floor: float = 0.10,
    body_height_weight: float = 0.55,
    body_progress_weight: float = 0.45,
    body_lift_velocity_weight: float = 0.0,
    lift_velocity_target: float = 0.16,
    progress_lift_gate_start: float = 0.0,
    progress_lift_gate_end: float = 0.0,
    second_clear_relief_min: float = 0.70,
    second_pending_floor: float = 0.35,
    commit_gate_scale: float = 0.90,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward the first-cleared rear leg supporting and driving the body upward/forward."""
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale,
    )
    asset = ctx["asset"]

    rear_joint_names = [box_joint_names[k] for k in ("RL", "RR")]
    joint_ids = []
    for joint_name in rear_joint_names:
        ids = asset.find_joints(joint_name)[0]
        joint_ids.append(ids[0])
    joint_ids = torch.as_tensor(joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)
    rear_box_pos = asset.data.joint_pos[:, joint_ids]
    lead_box_pos = torch.gather(rear_box_pos, 1, ctx["lead_index"].unsqueeze(1)).squeeze(1)
    both_box_pos = torch.mean(rear_box_pos, dim=1)
    lead_box_pos = torch.where(ctx["lead"] == 2, both_box_pos, lead_box_pos)
    box_err = torch.square((lead_box_pos - rear_push_target) / max(target_std, 1.0e-6))
    lead_box_score = lead_box_floor + (1.0 - lead_box_floor) * torch.exp(-box_err)

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_gain_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    forward_vel_score = torch.clamp(asset.data.root_lin_vel_b[:, 0] / max(target_forward_vel, 1.0e-6), 0.0, 1.0)
    progress_score = torch.maximum(distance_score, forward_vel_score)
    lift_velocity_score = torch.clamp(
        asset.data.root_lin_vel_w[:, 2] / max(lift_velocity_target, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    lift_velocity_component = lift_velocity_score * torch.clamp(1.0 - height_gain_score, min=0.0, max=1.0)
    if progress_lift_gate_end > progress_lift_gate_start:
        progress_lift_gate = torch.clamp(
            (height_gain_score - progress_lift_gate_start)
            / max(progress_lift_gate_end - progress_lift_gate_start, 1.0e-6),
            min=0.0,
            max=1.0,
        )
    else:
        progress_lift_gate = torch.ones_like(progress_score)
    # Progress should assist only after the lead rear leg starts lifting the body.
    body_drive_score = torch.clamp(
        body_height_weight * height_gain_score
        + body_progress_weight * progress_score * progress_lift_gate
        + body_lift_velocity_weight * lift_velocity_component,
        min=0.0,
        max=1.0,
    )
    body_drive_score = body_drive_floor + (1.0 - body_drive_floor) * body_drive_score
    second_clear_relief = torch.clamp(
        (ctx["second_score"] - second_clear_relief_min) / max(1.0 - second_clear_relief_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_pending_gate = torch.clamp(
        second_pending_floor + (1.0 - second_pending_floor) * (1.0 - second_clear_relief),
        min=0.0,
        max=1.0,
    )

    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    support_gate = (
        ctx["task_gate"]
        * ctx["cmd_gate"]
        * ctx["lead_known"]
        * ctx["lead_score"]
        * safe_pitch
        * stage_gate
    )
    signal = (
        support_gate
        * lead_box_score
        * body_drive_score
        * second_pending_gate
    )

    _ensure_highstep_rear_branch_buffers(env)
    gate_detached = support_gate.detach()
    env._highstep_lead_support_drive_sample_steps += torch.ones_like(signal)
    env._highstep_lead_support_drive_gate_sum += gate_detached
    env._highstep_lead_support_drive_lead_sum += ctx["lead_score"].detach()
    env._highstep_lead_support_drive_height_sum += height_gain_score.detach()
    env._highstep_lead_support_drive_height_active_sum += height_gain_score.detach() * gate_detached
    env._highstep_lead_support_drive_progress_sum += progress_score.detach()
    env._highstep_lead_support_drive_lift_velocity_sum += lift_velocity_component.detach()
    env._highstep_lead_support_drive_lift_velocity_active_sum += lift_velocity_component.detach() * gate_detached
    env._highstep_lead_support_drive_body_sum += body_drive_score.detach()
    env._highstep_lead_support_drive_body_active_sum += body_drive_score.detach() * gate_detached
    env._highstep_lead_support_drive_signal_sum += signal.detach()
    env._highstep_lead_support_drive_signal_max = torch.maximum(
        env._highstep_lead_support_drive_signal_max, signal.detach()
    )
    env._highstep_lead_support_drive_height_max = torch.maximum(
        env._highstep_lead_support_drive_height_max, height_gain_score.detach()
    )
    env._highstep_lead_support_drive_lift_velocity_max = torch.maximum(
        env._highstep_lead_support_drive_lift_velocity_max, lift_velocity_component.detach()
    )
    env._highstep_lead_support_drive_body_max = torch.maximum(
        env._highstep_lead_support_drive_body_max, body_drive_score.detach()
    )

    return signal


def post_lead_body_drive_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    lead_score_min: float = 0.58,
    lead_ready_floor: float = 0.0,
    second_clear_min: float = 0.58,
    min_elapsed_steps: float = 1.0,
    elapsed_ramp_steps: float = 6.0,
    elapsed_gate_floor: float = 0.0,
    deadline_steps: float = 38.0,
    deadline_grace_steps: float = 8.0,
    nominal_base_height: float = 0.44,
    min_distance: float = 0.20,
    target_distance: float = 0.64,
    min_height_gain: float = -0.01,
    target_height_gain: float = 0.10,
    target_forward_vel: float = 0.28,
    progress_floor: float = 0.25,
    second_floor: float = 0.35,
    body_height_weight: float = 0.50,
    body_progress_weight: float = 0.35,
    second_clear_weight: float = 0.15,
    body_lift_velocity_weight: float = 0.0,
    lift_velocity_target: float = 0.16,
    progress_lift_gate_start: float = 0.0,
    progress_lift_gate_end: float = 0.0,
    second_lift_gate_start: float = 0.0,
    second_lift_gate_end: float = 0.0,
    max_roll_metric: float = 0.30,
    roll_gate_floor: float = 0.0,
    commit_gate_scale: float = 0.85,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward body lift/advance immediately after the first rear foot has reached the step."""
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale,
    )
    asset = ctx["asset"]

    lead_ready = torch.clamp(
        (ctx["lead_score"] - lead_score_min) / max(1.0 - lead_score_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    elapsed_gate = torch.clamp(
        (ctx["lead_elapsed_steps"] - min_elapsed_steps) / max(elapsed_ramp_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    deadline_score = torch.clamp(
        (deadline_steps - ctx["lead_elapsed_steps"]) / max(deadline_steps - deadline_grace_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    lift_velocity_score = torch.clamp(
        asset.data.root_lin_vel_w[:, 2] / max(lift_velocity_target, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    lift_velocity_component = lift_velocity_score * torch.clamp(1.0 - height_score, min=0.0, max=1.0)
    forward_vel_score = torch.clamp(asset.data.root_lin_vel_b[:, 0] / max(target_forward_vel, 1.0e-6), 0.0, 1.0)
    progress_score = torch.maximum(distance_score, forward_vel_score)
    second_score = torch.clamp(
        (ctx["second_score"] - second_clear_min) / max(1.0 - second_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_component = second_floor + (1.0 - second_floor) * second_score
    if progress_lift_gate_end > progress_lift_gate_start:
        progress_lift_gate = torch.clamp(
            (height_score - progress_lift_gate_start)
            / max(progress_lift_gate_end - progress_lift_gate_start, 1.0e-6),
            min=0.0,
            max=1.0,
        )
    else:
        progress_lift_gate = torch.ones_like(progress_score)
    if second_lift_gate_end > second_lift_gate_start:
        second_lift_gate = torch.clamp(
            (height_score - second_lift_gate_start)
            / max(second_lift_gate_end - second_lift_gate_start, 1.0e-6),
            min=0.0,
            max=1.0,
        )
    else:
        second_lift_gate = torch.ones_like(second_component)
    # Progress/second-clear can polish support, but body lift must open that path.
    drive_score = torch.clamp(
        body_height_weight * height_score
        + body_progress_weight * (progress_floor + (1.0 - progress_floor) * progress_score) * progress_lift_gate
        + second_clear_weight * second_component * second_lift_gate
        + body_lift_velocity_weight * lift_velocity_component,
        min=0.0,
        max=1.0,
    )

    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_gate = torch.clamp((max_roll_metric - roll_metric) / max(max_roll_metric, 1.0e-6), min=0.0, max=1.0)
    lead_factor = torch.clamp(
        lead_ready_floor + (1.0 - lead_ready_floor) * lead_ready,
        min=0.0,
        max=1.0,
    )
    elapsed_factor = torch.clamp(
        elapsed_gate_floor + (1.0 - elapsed_gate_floor) * elapsed_gate,
        min=0.0,
        max=1.0,
    )
    roll_factor = torch.clamp(
        roll_gate_floor + (1.0 - roll_gate_floor) * roll_gate,
        min=0.0,
        max=1.0,
    )
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    post_lead_gate = (
        ctx["task_gate"]
        * ctx["cmd_gate"]
        * ctx["lead_known"]
        * lead_factor
        * elapsed_factor
        * deadline_score
        * roll_factor
        * safe_pitch
        * stage_gate
    )
    signal = post_lead_gate * drive_score

    _ensure_highstep_rear_branch_buffers(env)
    env._highstep_post_lead_drive_sample_steps += torch.ones_like(signal)
    env._highstep_post_lead_drive_gate_sum += post_lead_gate.detach()
    env._highstep_post_lead_drive_lead_sum += lead_ready.detach()
    env._highstep_post_lead_drive_height_sum += height_score.detach()
    env._highstep_post_lead_drive_lift_velocity_sum += lift_velocity_component.detach()
    env._highstep_post_lead_drive_progress_sum += progress_score.detach()
    env._highstep_post_lead_drive_second_sum += second_score.detach()
    env._highstep_post_lead_drive_stage_gate_sum += torch.ones_like(signal) * stage_gate.detach()
    env._highstep_post_lead_drive_signal_sum += signal.detach()
    env._highstep_post_lead_drive_signal_max = torch.maximum(
        env._highstep_post_lead_drive_signal_max, signal.detach()
    )

    return signal


def post_lead_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    lead_score_min: float = 0.55,
    second_low_threshold: float = 0.58,
    one_sided_gap: float = 0.14,
    one_sided_floor: float = 0.35,
    grace_steps: float = 10.0,
    ramp_steps: float = 26.0,
    target_progress: float = 0.45,
    base_floor: float = 0.35,
    second_low_weight: float = 0.40,
    progress_low_weight: float = 0.25,
    nominal_base_height: float = 0.44,
    min_distance: float = 0.20,
    target_distance: float = 0.62,
    min_height_gain: float = -0.01,
    target_height_gain: float = 0.09,
    target_forward_vel: float = 0.24,
    max_roll_metric: float = 0.35,
    commit_gate_scale: float = 0.85,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize post-lead stalls when one rear foot is up but the body/other rear foot do not follow."""
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale,
    )
    asset = ctx["asset"]

    lead_ready = torch.clamp(
        (ctx["lead_score"] - lead_score_min) / max(1.0 - lead_score_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    elapsed_pressure = torch.clamp(
        (ctx["lead_elapsed_steps"] - grace_steps) / max(ramp_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    first_minus_second = ctx["first_score"] - ctx["second_score"]
    one_sided_hard = ((ctx["first_score"] > lead_score_min) & (first_minus_second > one_sided_gap)).float()
    one_sided_soft = torch.clamp(
        (first_minus_second - one_sided_gap) / max(1.0 - one_sided_gap, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    one_sided_score = one_sided_hard * (one_sided_floor + (1.0 - one_sided_floor) * one_sided_soft)
    second_low_score = torch.clamp(
        (second_low_threshold - ctx["second_score"]) / max(second_low_threshold, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    forward_vel_score = torch.clamp(asset.data.root_lin_vel_b[:, 0] / max(target_forward_vel, 1.0e-6), 0.0, 1.0)
    progress_score = torch.maximum(torch.maximum(height_score, distance_score), forward_vel_score)
    progress_low_score = torch.clamp(
        (target_progress - progress_score) / max(target_progress, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_gate = torch.clamp((max_roll_metric - roll_metric) / max(max_roll_metric, 1.0e-6), min=0.0, max=1.0)
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    stall_condition_score = torch.maximum(one_sided_score, second_low_score * progress_low_score)
    base_signal = (
        ctx["task_gate"]
        * ctx["cmd_gate"]
        * ctx["lead_known"]
        * lead_ready
        * elapsed_pressure
        * stall_condition_score
        * roll_gate
        * safe_pitch
        * stage_gate
    )
    stall_modifier = torch.clamp(
        base_floor + second_low_weight * second_low_score + progress_low_weight * progress_low_score,
        min=0.0,
        max=1.0,
    )
    penalty_signal = base_signal * stall_modifier

    _ensure_highstep_rear_branch_buffers(env)
    env._highstep_post_lead_stall_sample_steps += torch.ones_like(penalty_signal)
    env._highstep_post_lead_stall_base_sum += base_signal.detach()
    env._highstep_post_lead_stall_base_max = torch.maximum(
        env._highstep_post_lead_stall_base_max, base_signal.detach()
    )
    env._highstep_post_lead_stall_second_low_sum += second_low_score.detach()
    env._highstep_post_lead_stall_progress_low_sum += progress_low_score.detach()
    env._highstep_post_lead_stall_modifier_sum += stall_modifier.detach()
    env._highstep_post_lead_stall_signal_sum += penalty_signal.detach()
    env._highstep_post_lead_stall_signal_max = torch.maximum(
        env._highstep_post_lead_stall_signal_max, penalty_signal.detach()
    )

    return penalty_signal


def highstep_action_score_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    first_clear_min: float = 0.54,
    second_clear_min: float = 0.58,
    lead_score_min: float = 0.52,
    one_sided_gap: float = 0.16,
    nominal_base_height: float = 0.44,
    min_distance: float = 0.20,
    target_distance: float = 0.64,
    min_height_gain: float = -0.01,
    target_height_gain: float = 0.11,
    target_forward_vel: float = 0.24,
    entry_weight: float = 0.35,
    support_weight: float = 0.50,
    safety_weight: float = 0.15,
    entry_floor: float = 0.55,
    support_floor: float = 0.45,
    support_gate_floor: float = 0.22,
    second_floor: float = 0.45,
    pre_support_cap: float = 0.42,
    support_cap_gain: float = 0.40,
    second_cap_gain: float = 0.18,
    max_roll_metric: float = 0.34,
    centerline_min_rear_width: float = 0.30,
    centerline_width_window: float = 0.12,
    centerline_min_rear_abs_y: float = 0.15,
    centerline_center_window: float = 0.10,
    commit_gate_scale: float = 0.85,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    support_bottleneck_start_update: int = 0,
    support_bottleneck_ramp_updates: int = 1,
    support_bottleneck_warmup_min_gate: float = 0.0,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Composite reward aligned with the hand-scored high-step behavior.

    This term intentionally keeps the score close to the behavior we inspect in
    play: rear entry, post-lead body support, and low one-sided stall risk.  It
    is used only by the rebuilt high-step task so the old task remains a
    comparable baseline.
    """
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale,
    )
    asset = ctx["asset"]

    first_entry = torch.clamp(
        (ctx["first_score"] - first_clear_min) / max(1.0 - first_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_entry = torch.clamp(
        (ctx["second_score"] - second_clear_min) / max(1.0 - second_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    entry_score = torch.clamp(0.62 * first_entry + 0.28 * second_entry + 0.10 * ctx["lead_known"], 0.0, 1.0)

    lead_ready = torch.clamp(
        (ctx["lead_score"] - lead_score_min) / max(1.0 - lead_score_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    # Support must be earned by body lift after lead entry; progress is scored by separate terms.
    height_gate = torch.clamp(height_score / 0.35, min=0.0, max=1.0)
    support_core = torch.clamp(0.84 * height_score + 0.16 * second_entry, min=0.0, max=1.0)
    support_score = torch.clamp(
        ctx["lead_known"] * lead_ready * height_gate * support_core,
        min=0.0,
        max=1.0,
    )

    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_gate = torch.clamp((max_roll_metric - roll_metric) / max(max_roll_metric, 1.0e-6), min=0.0, max=1.0)
    one_sided_score = torch.clamp(
        (ctx["first_score"] - ctx["second_score"] - one_sided_gap) / max(1.0 - one_sided_gap, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    safe_pitch = ((-asset.data.projected_gravity_b[:, 0]) < 0.98).float()
    safety_score = torch.clamp(roll_gate * safe_pitch * (1.0 - 0.70 * one_sided_score), min=0.0, max=1.0)

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    base_pos_rep = asset.data.root_pos_w.unsqueeze(1).repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 3)
    heading_rep = yaw_quat(asset.data.root_quat_w).unsqueeze(1).repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 4)
    rear_pos_b = quat_apply_inverse(heading_rep, rear_pos_w.reshape(-1, 3) - base_pos_rep)
    rear_y = rear_pos_b.reshape(env.num_envs, rear_pos_w.shape[1], 3)[:, :, 1]
    rear_width = torch.abs(rear_y[:, 0] - rear_y[:, 1])
    rear_min_abs_y = torch.min(torch.abs(rear_y), dim=1).values
    centerline_width_risk = torch.clamp(
        (centerline_min_rear_width - rear_width) / max(centerline_width_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    centerline_abs_y_risk = torch.clamp(
        (centerline_min_rear_abs_y - rear_min_abs_y) / max(centerline_center_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    centerline_gate = torch.clamp(1.0 - torch.maximum(centerline_width_risk, centerline_abs_y_risk), min=0.0, max=1.0)

    raw_score = torch.clamp(
        entry_weight * entry_score + support_weight * support_score + safety_weight * safety_score,
        min=0.0,
        max=1.0,
    )
    entry_gate = torch.clamp(entry_score / max(entry_floor, 1.0e-6), min=0.0, max=1.0)
    support_gate = torch.clamp(support_score / max(support_floor, 1.0e-6), min=0.0, max=1.0)
    second_gate = torch.clamp(second_entry / max(second_floor, 1.0e-6), min=0.0, max=1.0)
    support_bottleneck_gate = torch.clamp(
        (support_score - support_gate_floor) / max(support_floor - support_gate_floor, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    bottleneck_blend = _highstep_training_progress_gate(
        env,
        support_bottleneck_start_update,
        support_bottleneck_ramp_updates,
        num_steps_per_update,
    )
    relaxed_bottleneck_gate = torch.maximum(
        support_bottleneck_gate,
        torch.full_like(support_bottleneck_gate, float(support_bottleneck_warmup_min_gate)),
    )
    effective_bottleneck_gate = relaxed_bottleneck_gate + bottleneck_blend * (
        support_bottleneck_gate - relaxed_bottleneck_gate
    )

    # Stage bottleneck: entry and safety may teach useful preparation, but they
    # must not compensate for a missing post-lead support phase.  Without this
    # cap, the policy can keep a high score while the first rear foot never
    # really pushes the body onto the step.
    stage_cap = torch.clamp(
        pre_support_cap + support_cap_gain * support_gate + second_cap_gain * second_gate,
        min=0.0,
        max=1.0,
    )
    # The cap alone is intentionally not trusted: near the support floor it can
    # stay high enough that the policy ignores the support phase.  During
    # bodyflat -> highstep migration, however, applying the hard support gate too
    # early erases the entry/first-rear-clear signal before support can emerge.
    # The warmup only relaxes that multiplier; after the ramp, the hard
    # bottleneck again says "entry without support is incomplete".
    total_score = torch.minimum(raw_score, stage_cap) * entry_gate * effective_bottleneck_gate * centerline_gate
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return ctx["task_gate"] * ctx["cmd_gate"] * total_score * stage_gate


def _post_clear_absolute_rear_margin(
    rear_pos_w: torch.Tensor,
    platform_origin_w: torch.Tensor,
    heading_w: torch.Tensor,
    platform_half_width: float,
) -> torch.Tensor:
    """Return the slower rear foot's inward margin from a square platform entry edge.

    ``heading_w`` points into the platform.  The ray from the platform center in
    the opposite direction reaches the axis-aligned square at
    ``half_width / max(abs(hx), abs(hy))``.  This keeps the margin exact when the
    robot approaches with a small yaw instead of treating every entry edge as a
    plane at a fixed 1.5 m projection.
    """
    heading_norm = torch.linalg.vector_norm(heading_w, dim=-1, keepdim=True)
    unit_heading_w = heading_w / torch.clamp(heading_norm, min=1.0e-6)
    edge_distance = float(platform_half_width) / torch.clamp(
        torch.amax(torch.abs(unit_heading_w), dim=-1), min=1.0e-6
    )
    outward_distance = torch.sum(
        (rear_pos_w - platform_origin_w[:, None, :])
        * (-unit_heading_w[:, None, :]),
        dim=-1,
    )
    rear_margin = edge_distance[:, None] - outward_distance
    return torch.min(rear_margin, dim=1).values


def _update_post_clear_near_edge_counter(
    counter: torch.Tensor,
    post_clear_valid: torch.Tensor,
    command_relevant: torch.Tensor,
    slow_rear_margin: torch.Tensor,
    near_edge_margin: float,
) -> torch.Tensor:
    """Increment only a consecutive, commanded post-clear near-edge streak."""
    near_edge = (
        post_clear_valid.to(dtype=torch.bool)
        & command_relevant.to(dtype=torch.bool)
        & (slow_rear_margin < float(near_edge_margin))
    )
    return torch.where(near_edge, counter + 1.0, torch.zeros_like(counter))


def post_clear_recovery_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    second_clear_min: float = 0.68,
    min_elapsed_steps: float = 8.0,
    elapsed_ramp_steps: float = 10.0,
    min_base_clearance: float = 0.34,
    target_base_clearance: float = 0.43,
    target_forward_vel: float = 0.22,
    platform_half_width: float = 1.5,
    min_rear_margin: float = 0.04,
    target_rear_margin: float = 0.18,
    max_abs_pitch_metric: float = 0.24,
    max_abs_roll_metric: float = 0.24,
    posture_weight: float = 0.45,
    base_clearance_weight: float = 0.35,
    forward_weight: float = 0.20,
    rear_advance_weight: float = 0.35,
    post_clear_latch_scale: float = 1.0,
    commit_gate_scale: float = 0.85,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward returning to a usable walking posture after both rear feet clear the step."""
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale,
    )
    asset = ctx["asset"]

    second_ready = torch.clamp(
        (ctx["second_score"] - second_clear_min) / max(1.0 - second_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    elapsed_gate = torch.clamp(
        (ctx["lead_elapsed_steps"] - min_elapsed_steps) / max(elapsed_ramp_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    pitch_metric = torch.abs(asset.data.projected_gravity_b[:, 0])
    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    pitch_score = torch.clamp(
        (max_abs_pitch_metric - pitch_metric) / max(max_abs_pitch_metric, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    roll_score = torch.clamp(
        (max_abs_roll_metric - roll_metric) / max(max_abs_roll_metric, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    posture_score = pitch_score * roll_score

    base_clearance = asset.data.root_pos_w[:, 2] - ctx["front_terrain_z"]
    base_clearance_score = torch.clamp(
        (base_clearance - min_base_clearance) / max(target_base_clearance - min_base_clearance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_target = torch.minimum(
        torch.clamp(cmd_x, min=min_cmd_x),
        torch.full_like(cmd_x, float(target_forward_vel)),
    )
    forward_score = torch.clamp(
        asset.data.root_lin_vel_b[:, 0] / torch.clamp(forward_target, min=1.0e-6),
        0.0,
        1.0,
    )

    # Latch heading when the second rear foot first clears the entry edge.  The
    # rear component now rewards an absolute inward safety margin from that edge;
    # the old relative origin is retained only for checkpoint/runtime compatibility.
    post_clear_latch_bool = env._highstep_rear_branch_second_clear_steps > 0.0
    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :2]
    unit_forward_b = torch.zeros_like(asset.data.root_pos_w)
    unit_forward_b[:, 0] = 1.0
    heading_w = math_utils.quat_apply_yaw(asset.data.root_quat_w, unit_forward_b)[:, :2]
    new_latch = post_clear_latch_bool & (~env._highstep_post_clear_origin_valid)
    env._highstep_post_clear_origin_rear_pos_w = torch.where(
        new_latch[:, None, None],
        rear_pos_w.detach(),
        env._highstep_post_clear_origin_rear_pos_w,
    )
    env._highstep_post_clear_heading_w = torch.where(
        new_latch[:, None],
        heading_w.detach(),
        env._highstep_post_clear_heading_w,
    )
    env._highstep_post_clear_origin_step = torch.where(
        new_latch,
        env.episode_length_buf.to(dtype=rear_pos_w.dtype),
        env._highstep_post_clear_origin_step,
    )
    env._highstep_post_clear_origin_valid |= post_clear_latch_bool
    rear_margin = _post_clear_absolute_rear_margin(
        rear_pos_w,
        env.scene.env_origins[:, :2],
        env._highstep_post_clear_heading_w,
        platform_half_width,
    )
    rear_advance_score = torch.clamp(
        (rear_margin - min_rear_margin)
        / max(target_rear_margin - min_rear_margin, 1.0e-6),
        min=0.0,
        max=1.0,
    ) * env._highstep_post_clear_origin_valid.float()

    total_weight = max(
        posture_weight + base_clearance_weight + forward_weight + rear_advance_weight,
        1.0e-6,
    )
    recovery_score = torch.clamp(
        (
            posture_weight * posture_score
            + base_clearance_weight * base_clearance_score
            + forward_weight * forward_score
            + rear_advance_weight * rear_advance_score
        )
        / total_weight,
        min=0.0,
        max=1.0,
    )
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    # Once both rear feet have cleared, the forward scanner becomes flat and the
    # instantaneous high-step task gate can vanish. Keep this phase latched for
    # the rest of the episode so walking away from the entry edge remains rewarded.
    post_clear_latch = post_clear_latch_bool.to(dtype=base_clearance_score.dtype)
    completion_gate = torch.maximum(
        ctx["task_gate"] * second_ready,
        torch.clamp(float(post_clear_latch_scale) * post_clear_latch, min=0.0, max=1.0),
    )
    done_gate = completion_gate * base_clearance_score
    recovery_gate = (
        done_gate
        * ctx["cmd_gate"]
        * ctx["lead_known"]
        * elapsed_gate
        * stage_gate
    )
    signal = recovery_gate * recovery_score

    _ensure_highstep_rear_branch_buffers(env)
    env._highstep_post_clear_recovery_sample_steps += torch.ones_like(signal)
    env._highstep_post_clear_recovery_gate_sum += recovery_gate.detach()
    env._highstep_post_clear_recovery_posture_sum += posture_score.detach()
    env._highstep_post_clear_recovery_base_clearance_sum += base_clearance_score.detach()
    env._highstep_post_clear_recovery_forward_sum += forward_score.detach()
    env._highstep_post_clear_recovery_rear_advance_sum += rear_advance_score.detach()
    env._highstep_post_clear_recovery_signal_sum += signal.detach()
    env._highstep_post_clear_recovery_signal_max = torch.maximum(
        env._highstep_post_clear_recovery_signal_max, signal.detach()
    )

    return signal


def post_clear_rear_advance_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    platform_half_width: float = 1.5,
    min_rear_margin: float = 0.04,
    near_edge_margin: float = 0.12,
    target_rear_margin: float = 0.18,
    grace_steps: float = 8.0,
    ramp_steps: float = 32.0,
) -> torch.Tensor:
    """Penalize a consecutive commanded post-clear stay near the platform edge."""
    _ensure_highstep_rear_branch_buffers(env)
    asset: RigidObject = env.scene[asset_cfg.name]
    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :2]
    slow_rear_margin = _post_clear_absolute_rear_margin(
        rear_pos_w,
        env.scene.env_origins[:, :2],
        env._highstep_post_clear_heading_w,
        platform_half_width,
    )
    margin_deficit = torch.clamp(
        (target_rear_margin - slow_rear_margin)
        / max(target_rear_margin - min_rear_margin, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)
    post_clear_valid = env._highstep_post_clear_origin_valid
    command_relevant = cmd_gate > 0.0
    env._highstep_post_clear_stall_near_edge_counter = _update_post_clear_near_edge_counter(
        env._highstep_post_clear_stall_near_edge_counter,
        post_clear_valid,
        command_relevant,
        slow_rear_margin,
        near_edge_margin,
    ).detach()
    consecutive_gate = torch.clamp(
        (env._highstep_post_clear_stall_near_edge_counter - grace_steps)
        / max(ramp_steps, 1.0e-6),
        0.0,
        1.0,
    )
    signal = (
        post_clear_valid.to(dtype=rear_pos_w.dtype)
        * cmd_gate
        * consecutive_gate
        * margin_deficit
    )

    metric_active = post_clear_valid & command_relevant
    metric_active_f = metric_active.to(dtype=rear_pos_w.dtype)
    env._highstep_post_clear_stall_sample_steps += metric_active_f
    env._highstep_post_clear_stall_margin_sum += metric_active_f * slow_rear_margin.detach()
    env._highstep_post_clear_stall_margin_min = torch.where(
        metric_active,
        torch.minimum(
            env._highstep_post_clear_stall_margin_min,
            slow_rear_margin.detach(),
        ),
        env._highstep_post_clear_stall_margin_min,
    )
    env._highstep_post_clear_stall_counter_max = torch.maximum(
        env._highstep_post_clear_stall_counter_max,
        env._highstep_post_clear_stall_near_edge_counter,
    )
    env._highstep_post_clear_stall_signal_sum += signal.detach()
    return signal


def scanner_pretrigger_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    terrain_gate_min: float = 0.20,
    commit_gate_cutoff: float = 0.22,
    low_cmd_threshold: float = 0.12,
    front_lift_limit: float = -0.14,
    front_lift_window: float = 0.14,
    forward_vel_margin: float = 0.10,
    forward_vel_window: float = 0.25,
    front_lift_weight: float = 0.55,
    uncommanded_vel_weight: float = 0.45,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize scanner-only pre-trigger motion without touching committed climbs."""
    asset = env.scene[asset_cfg.name]
    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    phase = _highstep_on_top_phase_gates(
        env,
        asset,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
    )

    terrain_active = torch.clamp(
        (terrain_gate - terrain_gate_min) / max(1.0 - terrain_gate_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    precommit_gate = torch.clamp(
        (commit_gate_cutoff - commit_gate) / max(commit_gate_cutoff, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    low_cmd_gate = torch.clamp(
        (low_cmd_threshold - cmd_x) / max(low_cmd_threshold, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    front_foot_ids = asset.find_bodies(front_foot_names)[0]
    front_rel_z = asset.data.body_pos_w[:, front_foot_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
    max_front_lift = torch.max(front_rel_z, dim=1).values
    front_lift_score = torch.clamp(
        (max_front_lift - front_lift_limit) / max(front_lift_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    uncommanded_forward_score = torch.clamp(
        (asset.data.root_lin_vel_b[:, 0] - cmd_x - forward_vel_margin) / max(forward_vel_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    total_weight = max(front_lift_weight + uncommanded_vel_weight, 1.0e-6)
    raw_score = torch.clamp(
        (
            front_lift_weight * front_lift_score
            + uncommanded_vel_weight * low_cmd_gate * uncommanded_forward_score
        )
        / total_weight,
        min=0.0,
        max=1.0,
    )
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    penalty_gate = terrain_active * precommit_gate * phase["entry_allowed_gate"] * stage_gate
    signal = penalty_gate * raw_score

    _ensure_highstep_rear_branch_buffers(env)
    env._highstep_scanner_pretrigger_sample_steps += torch.ones_like(signal)
    env._highstep_scanner_pretrigger_gate_sum += penalty_gate.detach()
    env._highstep_scanner_pretrigger_terrain_gate_sum += terrain_gate.detach()
    env._highstep_scanner_pretrigger_commit_gate_sum += commit_gate.detach()
    env._highstep_scanner_pretrigger_low_cmd_gate_sum += low_cmd_gate.detach()
    env._highstep_scanner_pretrigger_front_lift_sum += front_lift_score.detach()
    env._highstep_scanner_pretrigger_uncommanded_vel_sum += uncommanded_forward_score.detach()
    env._highstep_scanner_pretrigger_signal_sum += signal.detach()
    env._highstep_scanner_pretrigger_signal_max = torch.maximum(
        env._highstep_scanner_pretrigger_signal_max, signal.detach()
    )

    return signal


def rear_approach_width_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.06,
    cmd_gate_width: float = 0.18,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    terrain_gate_min: float = 0.08,
    commit_gate_cutoff: float = 0.28,
    min_rear_width: float = 0.26,
    width_window: float = 0.08,
    min_rear_abs_y: float = 0.11,
    center_window: float = 0.05,
    width_weight: float = 0.65,
    center_weight: float = 0.35,
    hard_center_weight: float = 0.0,
    critical_center_boost: float = 0.0,
    critical_min_rear_abs_y: float = 0.075,
    critical_center_window: float = 0.04,
    terrain_gate_floor: float = 0.0,
    precommit_gate_floor: float = 0.0,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize rear-foot narrowing during high-step approach before front commit.

    This targets the sim-to-real failure where the rear feet collapse toward the
    centerline immediately after switching into the high-step policy, before the
    robot has actually climbed onto the platform.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    terrain_active = torch.clamp(
        (terrain_gate - terrain_gate_min) / max(1.0 - terrain_gate_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    if terrain_gate_floor > 0.0:
        terrain_active = torch.maximum(
            terrain_active,
            torch.full_like(terrain_active, min(max(terrain_gate_floor, 0.0), 1.0)),
        )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    precommit_gate = torch.clamp(
        (commit_gate_cutoff - commit_gate) / max(commit_gate_cutoff, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    if precommit_gate_floor > 0.0:
        precommit_gate = torch.maximum(
            precommit_gate,
            torch.full_like(precommit_gate, min(max(precommit_gate_floor, 0.0), 1.0)),
        )
    phase = _highstep_on_top_phase_gates(
        env,
        asset,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
    )

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / max(cmd_gate_width, 1.0e-6), min=0.0, max=1.0)
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    approach_gate = terrain_active * cmd_gate * precommit_gate * phase["entry_allowed_gate"] * stage_gate

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    base_pos_w = asset.data.root_pos_w
    heading_quat = yaw_quat(asset.data.root_quat_w)
    base_pos_rep = base_pos_w.unsqueeze(1).repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 3)
    heading_rep = heading_quat.unsqueeze(1).repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 4)
    rear_pos_b = quat_apply_inverse(heading_rep, rear_pos_w.reshape(-1, 3) - base_pos_rep)
    rear_pos_b = rear_pos_b.reshape(env.num_envs, rear_pos_w.shape[1], 3)
    rear_y = rear_pos_b[:, :, 1]

    rear_width = torch.abs(rear_y[:, 0] - rear_y[:, 1])
    rear_min_abs_y = torch.min(torch.abs(rear_y), dim=1).values
    width_deficit = torch.clamp((min_rear_width - rear_width) / max(width_window, 1.0e-6), min=0.0, max=1.0)
    center_deficit = torch.clamp(
        (min_rear_abs_y - rear_min_abs_y) / max(center_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    hard_center_deficit = center_deficit.square()
    critical_center_deficit = torch.clamp(
        (critical_min_rear_abs_y - rear_min_abs_y) / max(critical_center_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    total_weight = max(width_weight + center_weight + hard_center_weight, 1.0e-6)
    raw_score = torch.clamp(
        (
            width_weight * width_deficit
            + center_weight * center_deficit
            + hard_center_weight * hard_center_deficit
        )
        / total_weight,
        min=0.0,
        max=1.0,
    )
    raw_score = torch.clamp(raw_score + critical_center_boost * critical_center_deficit, min=0.0, max=1.0)
    signal = approach_gate * raw_score

    _ensure_highstep_rear_branch_buffers(env)
    active = approach_gate > 0.01
    env._highstep_rear_approach_width_sample_steps += torch.ones_like(signal)
    env._highstep_rear_approach_width_active_steps += active.float()
    env._highstep_rear_approach_width_gate_sum += approach_gate.detach()
    env._highstep_rear_approach_width_terrain_gate_sum += terrain_gate.detach()
    env._highstep_rear_approach_width_commit_gate_sum += commit_gate.detach()
    env._highstep_rear_approach_width_cmd_gate_sum += cmd_gate.detach()
    env._highstep_rear_approach_width_sum += rear_width.detach()
    env._highstep_rear_approach_width_active_sum += rear_width.detach() * approach_gate.detach()
    env._highstep_rear_approach_min_abs_y_sum += rear_min_abs_y.detach()
    env._highstep_rear_approach_min_abs_y_active_sum += rear_min_abs_y.detach() * approach_gate.detach()
    env._highstep_rear_approach_width_violation_steps += (active & (width_deficit > 0.0)).float()
    env._highstep_rear_approach_center_violation_steps += (active & (center_deficit > 0.0)).float()
    env._highstep_rear_approach_min_abs_y_active_min = torch.minimum(
        env._highstep_rear_approach_min_abs_y_active_min,
        torch.where(active, rear_min_abs_y.detach(), env._highstep_rear_approach_min_abs_y_active_min),
    )
    env._highstep_rear_approach_center_deficit_max = torch.maximum(
        env._highstep_rear_approach_center_deficit_max,
        torch.where(active, center_deficit.detach(), torch.zeros_like(center_deficit)),
    )
    env._highstep_rear_approach_signal_sum += signal.detach()
    env._highstep_rear_approach_signal_max = torch.maximum(
        env._highstep_rear_approach_signal_max,
        signal.detach(),
    )

    return signal


def rear_highstep_motion_width_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.04,
    cmd_gate_width: float = 0.18,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    terrain_gate_min: float = 0.05,
    commit_gate_floor: float = 0.25,
    min_rear_width: float = 0.255,
    width_window: float = 0.08,
    min_rear_abs_y: float = 0.105,
    center_window: float = 0.05,
    width_weight: float = 0.70,
    center_weight: float = 0.30,
    hard_center_weight: float = 0.0,
    critical_center_boost: float = 0.0,
    critical_min_rear_abs_y: float = 0.075,
    critical_center_window: float = 0.04,
    post_clear_relief: float = 0.45,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize rear-foot inward collapse through highstep approach and support."""

    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    terrain_active = torch.clamp(
        (terrain_gate - terrain_gate_min) / max(1.0 - terrain_gate_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    phase = _highstep_on_top_phase_gates(
        env,
        asset,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
    )

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / max(cmd_gate_width, 1.0e-6), min=0.0, max=1.0)
    precommit_gate = terrain_active * (1.0 - torch.clamp(commit_gate, min=0.0, max=1.0))
    postcommit_gate = torch.maximum(terrain_active * commit_gate, commit_gate_floor * commit_gate)
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    base_pos_w = asset.data.root_pos_w
    heading_quat = yaw_quat(asset.data.root_quat_w)
    base_pos_rep = base_pos_w.unsqueeze(1).repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 3)
    heading_rep = heading_quat.unsqueeze(1).repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 4)
    rear_pos_b = quat_apply_inverse(heading_rep, rear_pos_w.reshape(-1, 3) - base_pos_rep)
    rear_pos_b = rear_pos_b.reshape(env.num_envs, rear_pos_w.shape[1], 3)
    rear_y = rear_pos_b[:, :, 1]

    rear_clearance = rear_pos_w[..., 2] - (front_terrain_z[:, None] + clearance_margin)
    rear_clearance_score = torch.clamp(
        (rear_clearance + clearance_window) / max(clearance_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_score = torch.min(rear_clearance_score, dim=1).values
    post_clear_scale = 1.0 - post_clear_relief * second_score

    rear_width = torch.abs(rear_y[:, 0] - rear_y[:, 1])
    rear_min_abs_y = torch.min(torch.abs(rear_y), dim=1).values
    width_deficit = torch.clamp((min_rear_width - rear_width) / max(width_window, 1.0e-6), min=0.0, max=1.0)
    center_deficit = torch.clamp(
        (min_rear_abs_y - rear_min_abs_y) / max(center_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    hard_center_deficit = center_deficit.square()
    critical_center_deficit = torch.clamp(
        (critical_min_rear_abs_y - rear_min_abs_y) / max(critical_center_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    total_weight = max(width_weight + center_weight + hard_center_weight, 1.0e-6)
    raw_score = torch.clamp(
        (
            width_weight * width_deficit
            + center_weight * center_deficit
            + hard_center_weight * hard_center_deficit
        )
        / total_weight,
        min=0.0,
        max=1.0,
    )
    raw_score = torch.clamp(raw_score + critical_center_boost * critical_center_deficit, min=0.0, max=1.0)

    motion_gate = torch.maximum(precommit_gate, postcommit_gate) * phase["entry_allowed_gate"] * cmd_gate * stage_gate
    signal = motion_gate * raw_score * post_clear_scale

    _ensure_highstep_rear_branch_buffers(env)
    active = motion_gate > 0.01
    env._highstep_rear_motion_width_sample_steps += torch.ones_like(signal)
    env._highstep_rear_motion_width_active_steps += active.float()
    env._highstep_rear_motion_width_gate_sum += motion_gate.detach()
    env._highstep_rear_motion_width_terrain_gate_sum += terrain_gate.detach()
    env._highstep_rear_motion_width_commit_gate_sum += commit_gate.detach()
    env._highstep_rear_motion_width_cmd_gate_sum += cmd_gate.detach()
    env._highstep_rear_motion_width_second_sum += second_score.detach()
    env._highstep_rear_motion_width_sum += rear_width.detach()
    env._highstep_rear_motion_width_active_sum += rear_width.detach() * motion_gate.detach()
    env._highstep_rear_motion_min_abs_y_sum += rear_min_abs_y.detach()
    env._highstep_rear_motion_min_abs_y_active_sum += rear_min_abs_y.detach() * motion_gate.detach()
    env._highstep_rear_motion_width_violation_steps += (active & (width_deficit > 0.0)).float()
    env._highstep_rear_motion_center_violation_steps += (active & (center_deficit > 0.0)).float()
    env._highstep_rear_motion_min_abs_y_active_min = torch.minimum(
        env._highstep_rear_motion_min_abs_y_active_min,
        torch.where(active, rear_min_abs_y.detach(), env._highstep_rear_motion_min_abs_y_active_min),
    )
    env._highstep_rear_motion_center_deficit_max = torch.maximum(
        env._highstep_rear_motion_center_deficit_max,
        torch.where(active, center_deficit.detach(), torch.zeros_like(center_deficit)),
    )
    env._highstep_rear_motion_signal_sum += signal.detach()
    env._highstep_rear_motion_signal_max = torch.maximum(
        env._highstep_rear_motion_signal_max,
        signal.detach(),
    )

    return signal


def highstep_support_stability_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.06,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    clearance_margin: float = 0.04,
    clearance_window: float = 0.18,
    commit_gate_floor: float = 0.18,
    contact_threshold: float = 4.0,
    contact_force_window: float = 32.0,
    front_slip_deadband: float = 0.08,
    front_slip_window: float = 0.24,
    roll_deadband: float = 0.16,
    roll_window: float = 0.20,
    roll_rate_deadband: float = 0.45,
    roll_rate_window: float = 1.10,
    yaw_rate_deadband: float = 0.18,
    yaw_rate_window: float = 0.65,
    lead_score_min: float = 0.44,
    second_clear_min: float = 0.62,
    rear_lag_grace_steps: float = 10.0,
    rear_lag_ramp_steps: float = 30.0,
    one_sided_gap: float = 0.18,
    min_support_width: float = 0.25,
    support_width_window: float = 0.10,
    min_support_abs_y: float = 0.09,
    support_center_window: float = 0.06,
    front_support_grace_steps: float = 8.0,
    front_support_ramp_steps: float = 16.0,
    front_contact_weight: float = 0.25,
    front_slip_weight: float = 0.18,
    posture_rate_weight: float = 0.32,
    rear_lag_weight: float = 0.18,
    support_width_weight: float = 0.07,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize risky high-step support patterns seen in sim-to-real tests.

    This is a SWAP-inspired guard: once the front feet have committed to the
    step, discourage single-front-foot bracing, front-foot slip, roll/yaw rate,
    rear-foot edge dwell, and a narrow lateral support polygon.
    """
    ctx = _highstep_rear_stage_context(
        env,
        command_name,
        asset_cfg,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        min_cmd_x,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
        commit_gate_scale=0.90,
    )
    asset = ctx["asset"]
    phase = _highstep_on_top_phase_gates(
        env,
        asset,
        sensor_cfg,
        front_foot_names,
        rear_foot_names,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
        clearance_margin,
        clearance_window,
    )
    commit_support_gate = torch.clamp(
        (ctx["commit_gate"] - commit_gate_floor) / max(1.0 - commit_gate_floor, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    support_gate = ctx["task_gate"] * ctx["cmd_gate"] * commit_support_gate * phase["entry_allowed_gate"] * stage_gate
    # Sequential first contact is part of a normal climb.  Delay the
    # bilateral-contact/slip/posture pressure briefly, then ramp it in; the real
    # failure lasts far beyond this grace window and remains penalized.
    front_support_pressure = torch.clamp(
        (env._highstep_rear_branch_commit_steps - front_support_grace_steps)
        / max(front_support_ramp_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    front_contact_ids = contact_sensor.find_bodies(front_foot_names)[0]
    if hasattr(contact_sensor.data, "net_forces_w_history"):
        front_force_norm = torch.linalg.norm(
            contact_sensor.data.net_forces_w_history[:, :, front_contact_ids, :],
            dim=-1,
        ).max(dim=1).values
    else:
        front_force_norm = torch.linalg.norm(
            contact_sensor.data.net_forces_w[:, front_contact_ids, :],
            dim=-1,
        )
    front_contact_score = torch.clamp(
        (front_force_norm - contact_threshold) / max(contact_force_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    front_bilateral_contact = torch.min(front_contact_score, dim=1).values
    front_contact_imbalance = torch.abs(front_contact_score[:, 0] - front_contact_score[:, 1])
    front_contact_penalty = torch.clamp(
        0.65 * (1.0 - front_bilateral_contact) + 0.35 * front_contact_imbalance,
        min=0.0,
        max=1.0,
    ) * front_support_pressure

    front_asset_ids = asset.find_bodies(front_foot_names)[0]
    front_foot_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, front_asset_ids, :2], dim=-1)
    front_contact_weighted = torch.clamp(front_contact_score.detach(), min=0.0, max=1.0)
    front_slip = torch.sum(front_foot_vel * front_contact_weighted, dim=1) / torch.clamp(
        torch.sum(front_contact_weighted, dim=1),
        min=1.0,
    )
    front_slip_score = torch.clamp(
        (front_slip - front_slip_deadband) / max(front_slip_window, 1.0e-6),
        min=0.0,
        max=1.0,
    ) * front_support_pressure

    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_score = torch.clamp((roll_metric - roll_deadband) / max(roll_window, 1.0e-6), min=0.0, max=1.0)
    roll_rate = torch.abs(asset.data.root_ang_vel_b[:, 0])
    roll_rate_score = torch.clamp(
        (roll_rate - roll_rate_deadband) / max(roll_rate_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
    yaw_rate_error = torch.abs(asset.data.root_ang_vel_b[:, 2] - cmd_yaw)
    yaw_rate_score = torch.clamp(
        (yaw_rate_error - yaw_rate_deadband) / max(yaw_rate_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    posture_rate_score = torch.clamp(
        0.35 * roll_score + 0.35 * roll_rate_score + 0.30 * yaw_rate_score,
        min=0.0,
        max=1.0,
    ) * front_support_pressure

    lead_ready = torch.clamp(
        (ctx["lead_score"] - lead_score_min) / max(1.0 - lead_score_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_low = torch.clamp(
        (second_clear_min - ctx["second_score"]) / max(second_clear_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    elapsed_pressure = torch.clamp(
        (ctx["lead_elapsed_steps"] - rear_lag_grace_steps) / max(rear_lag_ramp_steps, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    one_sided_soft = torch.clamp(
        (ctx["first_score"] - ctx["second_score"] - one_sided_gap) / max(1.0 - one_sided_gap, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    rear_lag_score = ctx["lead_known"] * lead_ready * torch.maximum(second_low * elapsed_pressure, one_sided_soft)

    support_body_ids = asset.find_bodies(front_foot_names + rear_foot_names)[0]
    support_pos_w = asset.data.body_pos_w[:, support_body_ids, :]
    support_rel_w = support_pos_w - asset.data.root_pos_w[:, None, :]
    num_support_bodies = support_rel_w.shape[1]
    heading_quat = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, num_support_bodies, -1)
    support_rel_b = quat_apply_inverse(
        heading_quat.reshape(-1, 4),
        support_rel_w.reshape(-1, 3),
    ).reshape(support_rel_w.shape)
    support_y = support_rel_b[..., 1]
    front_width = torch.abs(support_y[:, 0] - support_y[:, 1])
    rear_width = torch.abs(support_y[:, 2] - support_y[:, 3])
    min_pair_width = torch.minimum(front_width, rear_width)
    support_min_abs_y = torch.min(torch.abs(support_y), dim=1).values
    support_width_deficit = torch.clamp(
        (min_support_width - min_pair_width) / max(support_width_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    support_center_deficit = torch.clamp(
        (min_support_abs_y - support_min_abs_y) / max(support_center_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    support_width_score = torch.clamp(0.65 * support_width_deficit + 0.35 * support_center_deficit, 0.0, 1.0)

    total_weight = max(
        front_contact_weight + front_slip_weight + posture_rate_weight + rear_lag_weight + support_width_weight,
        1.0e-6,
    )
    raw_score = torch.clamp(
        (
            front_contact_weight * front_contact_penalty
            + front_slip_weight * front_slip_score
            + posture_rate_weight * posture_rate_score
            + rear_lag_weight * rear_lag_score
            + support_width_weight * support_width_score
        )
        / total_weight,
        min=0.0,
        max=1.0,
    )
    signal = support_gate * raw_score

    _ensure_highstep_rear_branch_buffers(env)
    gate_detached = support_gate.detach()
    active = support_gate > 0.01
    env._highstep_support_stability_sample_steps += torch.ones_like(signal)
    env._highstep_support_stability_active_steps += active.float()
    env._highstep_support_stability_gate_sum += gate_detached
    env._highstep_support_stability_front_contact_sum += front_bilateral_contact.detach() * gate_detached
    env._highstep_support_stability_front_slip_sum += front_slip_score.detach() * gate_detached
    env._highstep_support_stability_posture_rate_sum += posture_rate_score.detach() * gate_detached
    env._highstep_support_stability_rear_lag_sum += rear_lag_score.detach() * gate_detached
    env._highstep_support_stability_support_width_sum += support_width_score.detach() * gate_detached
    env._highstep_support_stability_signal_sum += signal.detach()
    env._highstep_support_stability_signal_max = torch.maximum(
        env._highstep_support_stability_signal_max,
        signal.detach(),
    )

    return signal


def highstep_forward_progress_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_cmd_x: float = 0.08,
    target_forward_vel: float = 0.35,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    front_foot_names: list[str] | None = None,
    rear_foot_names: list[str] | None = None,
    commit_gate_scale: float = 0.70,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward forward progress while a high step is detected in front."""
    robot = env.scene[asset_cfg.name]
    gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(robot, front_foot_names, rear_foot_names)
    gate = torch.maximum(gate, commit_gate_scale * commit_gate)
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)
    forward_progress = torch.clamp(robot.data.root_lin_vel_b[:, 0] / max(target_forward_vel, 1.0e-6), 0.0, 1.0)
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * forward_progress * stage_gate


def highstep_body_lift_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    nominal_base_height: float = 0.44,
    min_lift: float = 0.02,
    target_lift: float = 0.16,
    front_foot_names: list[str] | None = None,
    rear_foot_names: list[str] | None = None,
    commit_gate_scale: float = 0.70,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward lifting the body relative to the lower terrain when a forward high step is present."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gate, height_delta, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    gate = torch.maximum(gate, commit_gate_scale * commit_gate)
    rear_terrain_z = front_terrain_z - height_delta

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    body_lift = asset.data.root_pos_w[:, 2] - (rear_terrain_z + nominal_base_height)
    lift_score = torch.clamp(
        (body_lift - min_lift) / max(target_lift - min_lift, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * lift_score * safe_pitch * stage_gate


def highstep_base_advance_lift_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    nominal_base_height: float = 0.44,
    min_distance: float = 0.25,
    target_distance: float = 0.80,
    min_height_gain: float = -0.02,
    target_height_gain: float = 0.10,
    distance_floor: float = 0.0,
    front_foot_names: list[str] | None = None,
    rear_foot_names: list[str] | None = None,
    commit_gate_scale: float = 0.80,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward the body actually advancing and lifting after committing to a high step."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    gate = torch.maximum(terrain_gate, commit_gate_scale * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    distance_floor_value = torch.clamp(torch.as_tensor(distance_floor, device=distance_score.device), 0.0, 1.0)
    lift_gated_distance = distance_score * (distance_floor_value + (1.0 - distance_floor_value) * height_score)
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * lift_gated_distance * safe_pitch * stage_gate


def base_height_floor_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_height: float = 0.32,
) -> torch.Tensor:
    """Penalize crawling below a terrain-relative minimum body height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(ray_hits_z) & (torch.abs(ray_hits_z) < 1.0e6)
    valid_count = valid.float().sum(dim=1)
    lowest_hits = torch.where(valid, ray_hits_z, torch.full_like(ray_hits_z, float("inf"))).min(dim=1).values
    ground_z = torch.where(valid_count > 0.0, lowest_hits, asset.data.root_pos_w[:, 2] - min_height)

    body_height = asset.data.root_pos_w[:, 2] - ground_z
    return torch.square(torch.clamp(min_height - body_height, min=0.0))


def highstep_leg_support_contact_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    contact_threshold: float = 8.0,
    contact_force_window: float = 60.0,
    max_contact_score: float = 2.0,
    front_foot_names: list[str] | None = None,
    rear_foot_names: list[str] | None = None,
    commit_gate_scale: float = 0.75,
    support_body_names: list[str] | None = None,
    support_pose_scale: float = 0.0,
    support_x_min: float = 0.02,
    support_x_target: float = 0.28,
    support_height_margin: float = 0.10,
    support_height_window: float = 0.18,
    support_phase_floor: float = 0.40,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward useful thigh/calf support contacts or the posture that leads to them."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    gate = torch.maximum(gate, commit_gate_scale * commit_gate)
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    force_norm = torch.linalg.norm(forces, dim=-1)
    contact_score = torch.clamp(
        (force_norm - contact_threshold) / max(contact_force_window, 1.0e-6),
        min=0.0,
        max=1.0,
    ).sum(dim=1)
    contact_score = torch.clamp(contact_score, max=max_contact_score) / max(max_contact_score, 1.0e-6)

    if support_body_names is not None and support_pose_scale > 0.0:
        support_body_ids = asset.find_bodies(support_body_names)[0]
        support_pos_w = asset.data.body_pos_w[:, support_body_ids, :]
        support_rel_w = support_pos_w - asset.data.root_pos_w[:, None, :]
        num_support_bodies = support_rel_w.shape[1]
        heading_quat = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, num_support_bodies, -1)
        support_rel_b = quat_apply_inverse(
            heading_quat.reshape(-1, 4),
            support_rel_w.reshape(-1, 3),
        ).reshape(support_rel_w.shape)
        support_x_score = torch.clamp(
            (support_rel_b[..., 0] - support_x_min) / max(support_x_target - support_x_min, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        _, _, front_terrain_z = _forward_highstep_terrain_gate(
            env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
        )
        support_height_score = torch.clamp(
            (support_pos_w[..., 2] - (front_terrain_z[:, None] - support_height_margin))
            / max(support_height_window, 1.0e-6),
            min=0.0,
            max=1.0,
        )
        support_pose_score = torch.max(support_x_score * support_height_score, dim=1).values
        contact_score = torch.maximum(contact_score, support_pose_scale * support_pose_score)

    forward_progress = torch.clamp(asset.data.root_lin_vel_b[:, 0] / 0.25, min=0.0, max=1.0)
    support_phase = torch.clamp(
        support_phase_floor + (1.0 - support_phase_floor) * forward_progress,
        min=0.0,
        max=1.0,
    )
    safe_pitch = (-asset.data.projected_gravity_b[:, 0]) < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * contact_score * support_phase * safe_pitch * stage_gate


def non_forward_highstep_pitch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    forward_cmd_threshold: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    pitch_deadband: float = 0.12,
) -> torch.Tensor:
    """Penalize pitch when the command is not a forward high-step climb."""
    robot = env.scene[asset_cfg.name]
    gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_gate = torch.clamp((cmd_x - forward_cmd_threshold) / 0.25, 0.0, 1.0)
    climb_gate = gate * forward_gate
    pitch_metric = torch.abs(robot.data.projected_gravity_b[:, 0])
    pitch_error = torch.clamp(pitch_metric - pitch_deadband, min=0.0)
    return torch.square(pitch_error) * (1.0 - climb_gate)


def backward_motion_pitch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    min_backward_cmd: float = 0.05,
    full_backward_cmd: float = 0.22,
    pitch_deadband: float = 0.08,
    pitch_limit: float = 0.32,
    height_soft_limit: float = 0.56,
    height_limit: float = 0.72,
    height_weight: float = 0.40,
) -> torch.Tensor:
    """Penalize pitching and body lift when tracking a backward velocity command."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]

    backward_gate = torch.clamp(
        (-cmd_x - min_backward_cmd) / max(full_backward_cmd - min_backward_cmd, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    pitch_metric = torch.abs(asset.data.projected_gravity_b[:, 0])
    pitch_score = torch.clamp(
        (pitch_metric - pitch_deadband) / max(pitch_limit - pitch_deadband, 1.0e-6),
        min=0.0,
        max=2.0,
    )

    base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height_score = torch.clamp(
        (base_height - height_soft_limit) / max(height_limit - height_soft_limit, 1.0e-6),
        min=0.0,
        max=2.0,
    )
    return backward_gate * (torch.square(pitch_score) + height_weight * torch.square(height_score))

# ==========================================
# 1. 像马一样扬身 (Horse Rearing Posture) - 防破解版
# ==========================================
def horse_rearing_posture_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    front_foot_names: list,
    rear_foot_names: list,
    sensor_cfg: SceneEntityCfg | None = None,
    target_pitch_deg: float = 35.0,  # 目标仰角：35度 (不要让它竖直)
    target_height_diff: float = 0.35, # 目标高度差：0.35米 (根据你的台阶高度调整)
    min_front_x: float = 0.18,
    target_front_x: float = 0.42,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    terrain_gate_floor: float = 0.25,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """
    鼓励机器人在前方高台前扬起并向前够台，而不是原地仰头刷姿态奖励。
    """
    asset = env.scene[asset_cfg.name]

    velocity_command = env.command_manager.get_command(command_name)
    cmd_x = velocity_command[:, 0]

    projected_gravity = asset.data.projected_gravity_b
    current_pitch_sin = -projected_gravity[:, 0]

    target_pitch_rad = math.radians(target_pitch_deg)
    target_pitch_sin = math.sin(target_pitch_rad)

    front_foot_ids = asset.find_bodies(front_foot_names)[0]
    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    front_feet_z = asset.data.body_pos_w[:, front_foot_ids, 2].mean(dim=1)
    rear_feet_z = asset.data.body_pos_w[:, rear_foot_ids, 2].mean(dim=1)
    height_diff = front_feet_z - rear_feet_z
    front_rel_w = asset.data.body_pos_w[:, front_foot_ids, :].mean(dim=1) - asset.data.root_pos_w
    front_rel_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), front_rel_w)
    front_x_score = torch.clamp(
        (front_rel_b[:, 0] - min_front_x) / max(target_front_x - min_front_x, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    cmd_gate = torch.clamp((cmd_x - 0.08) / 0.25, min=0.0, max=1.0)
    pitch_score = torch.clamp((current_pitch_sin - 0.05) / max(target_pitch_sin - 0.05, 1.0e-6), 0.0, 1.0)
    height_score = torch.clamp(height_diff / max(target_height_diff, 1.0e-6), 0.0, 1.0)
    terrain_gate = torch.ones_like(cmd_gate)
    if sensor_cfg is not None:
        terrain_gate, _, _ = _forward_highstep_terrain_gate(
            env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
        )
    task_gate = torch.clamp(
        terrain_gate_floor + (1.0 - terrain_gate_floor) * terrain_gate,
        min=0.0,
        max=1.0,
    )
    safe_pitch = current_pitch_sin < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return (
        task_gate
        * cmd_gate
        * safe_pitch
        * (0.45 * pitch_score + 0.35 * height_score + 0.20 * front_x_score)
        * stage_gate
    )

# ==========================================
# 2. 前腿搭台后锁死 (Front Legs Quiet on Step) - 修改版
# ==========================================
def front_legs_quiet_on_step_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    front_foot_names: list,
    front_knee_names: list,
    step_height_threshold: float = 0.30
) -> torch.Tensor:
    """
    当【两只前脚同时】接触到高于地面的平台时，严厉惩罚前腿膝盖（calf）的运动，迫使其“锁死”或保持稳定。
    """
    asset = env.scene[asset_cfg.name]

    # 找到前脚和前膝盖的索引
    front_foot_ids = asset.find_bodies(front_foot_names)[0]
    front_knee_ids = asset.find_joints(front_knee_names)[0]

    # 获取前脚的高度 (Z坐标)，形状为 (num_envs, 2)
    front_foot_z = asset.data.body_pos_w[:, front_foot_ids, 2]

    # 判定条件：两只前脚的高度【同时】大于绝对值 step_height_threshold
    # 使用 .all(dim=1) 确保两个脚都满足条件
    both_feet_on_step = (front_foot_z > step_height_threshold).all(dim=1)

    # 获取前膝盖的速度
    front_knee_vel = asset.data.joint_vel[:, front_knee_ids]

    # 如果双腿都搭上了高台，惩罚前膝盖的速度平方
    penalty = torch.sum(torch.square(front_knee_vel), dim=1) * both_feet_on_step
    return penalty


def front_legs_lift_guard_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    relative_lift_limit: float = -0.14,
    relative_lift_window: float = 0.14,
    asymmetry_limit: float = 0.16,
    asymmetry_window: float = 0.16,
    terrain_relief_scale: float = 1.0,
    commit_relief_scale: float = 0.90,
    height_weight: float = 0.60,
    asymmetry_weight: float = 0.40,
    min_command_norm: float = 0.02,
    command_gate_width: float = 0.18,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize front-leg high-lift leakage when no real high-step commitment exists."""
    asset = env.scene[asset_cfg.name]
    front_foot_ids = asset.find_bodies(front_foot_names)[0]
    front_pos_w = asset.data.body_pos_w[:, front_foot_ids, :]
    front_rel_z = front_pos_w[:, :, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)

    max_lift = torch.max(front_rel_z, dim=1).values
    height_excess = torch.clamp(
        (max_lift - relative_lift_limit) / max(relative_lift_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    if front_rel_z.shape[1] >= 2:
        asymmetry = torch.abs(front_rel_z[:, 0] - front_rel_z[:, 1])
    else:
        asymmetry = torch.zeros_like(max_lift)
    asymmetry_excess = torch.clamp(
        (asymmetry - asymmetry_limit) / max(asymmetry_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    highstep_relief = torch.clamp(
        torch.maximum(terrain_relief_scale * terrain_gate, commit_relief_scale * commit_gate),
        min=0.0,
        max=1.0,
    )
    guard_gate = torch.square(1.0 - highstep_relief)

    command = env.command_manager.get_command(command_name)
    command_norm = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    command_gate = torch.clamp(
        (command_norm - min_command_norm) / max(command_gate_width, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)

    raw_score = torch.clamp(
        height_weight * height_excess + asymmetry_weight * asymmetry_excess,
        min=0.0,
        max=1.0,
    )
    signal = command_gate * guard_gate * raw_score * stage_gate

    _ensure_highstep_rear_branch_buffers(env)
    env._highstep_front_lift_guard_sample_steps += torch.ones_like(signal)
    env._highstep_front_lift_guard_gate_sum += guard_gate.detach()
    env._highstep_front_lift_guard_signal_sum += signal.detach()
    env._highstep_front_lift_guard_signal_max = torch.maximum(
        env._highstep_front_lift_guard_signal_max, signal.detach()
    )
    env._highstep_front_lift_guard_max_lift_sum += max_lift.detach()
    env._highstep_front_lift_guard_asym_sum += asymmetry.detach()
    env._highstep_front_lift_guard_height_excess_sum += height_excess.detach()
    env._highstep_front_lift_guard_asym_excess_sum += asymmetry_excess.detach()
    env._highstep_front_lift_guard_terrain_gate_sum += terrain_gate.detach()
    env._highstep_front_lift_guard_commit_gate_sum += commit_gate.detach()

    return signal


def forward_flat_left_front_lift_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    left_front_foot_name: str,
    right_front_foot_name: str,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    min_forward_cmd: float = 0.08,
    forward_gate_width: float = 0.18,
    max_side_cmd: float = 0.12,
    side_gate_width: float = 0.18,
    max_yaw_cmd: float = 0.16,
    yaw_gate_width: float = 0.22,
    terrain_gate_cutoff: float = 0.08,
    commit_gate_cutoff: float = 0.06,
    fl_lift_limit: float = -0.18,
    fl_lift_window: float = 0.10,
    fl_over_fr_limit: float = 0.08,
    fl_over_fr_window: float = 0.12,
    height_weight: float = 0.45,
    asymmetry_weight: float = 0.55,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize only left-front high-lift leakage during straight forward flat walking.

    This is intentionally narrower than ``front_legs_lift_guard_penalty``: it is
    FL-only, forward-command-only, and hard-disabled once terrain/commit gates
    indicate a real high-step approach.
    """
    asset = env.scene[asset_cfg.name]
    fl_id = asset.find_bodies([left_front_foot_name])[0][0]
    fr_id = asset.find_bodies([right_front_foot_name])[0][0]

    root_z = asset.data.root_pos_w[:, 2]
    fl_rel_z = asset.data.body_pos_w[:, fl_id, 2] - root_z
    fr_rel_z = asset.data.body_pos_w[:, fr_id, 2] - root_z
    fl_minus_fr = fl_rel_z - fr_rel_z

    height_excess = torch.clamp(
        (fl_rel_z - fl_lift_limit) / max(fl_lift_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    asymmetry_excess = torch.clamp(
        (fl_minus_fr - fl_over_fr_limit) / max(fl_over_fr_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    command = env.command_manager.get_command(command_name)
    cmd_x = command[:, 0]
    cmd_y = command[:, 1]
    cmd_yaw = command[:, 2]
    forward_gate = torch.clamp((cmd_x - min_forward_cmd) / max(forward_gate_width, 1.0e-6), 0.0, 1.0)
    backward_gate = torch.clamp((-cmd_x - min_forward_cmd) / max(forward_gate_width, 1.0e-6), 0.0, 1.0)
    lateral_cmd = torch.abs(cmd_y)
    yaw_cmd = torch.abs(cmd_yaw)
    straight_side_gate = torch.clamp(
        (max_side_cmd + side_gate_width - lateral_cmd) / max(side_gate_width, 1.0e-6),
        0.0,
        1.0,
    )
    straight_yaw_gate = torch.clamp(
        (max_yaw_cmd + yaw_gate_width - yaw_cmd) / max(yaw_gate_width, 1.0e-6),
        0.0,
        1.0,
    )
    lateral_gate = torch.clamp((lateral_cmd - max_side_cmd) / max(side_gate_width, 1.0e-6), 0.0, 1.0)
    yaw_gate = torch.clamp((yaw_cmd - max_yaw_cmd) / max(yaw_gate_width, 1.0e-6), 0.0, 1.0)

    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    highstep_relief = torch.maximum(terrain_gate, commit_gate)
    flat_gate = ((terrain_gate <= terrain_gate_cutoff) & (commit_gate <= commit_gate_cutoff)).float()

    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    active_gate = forward_gate * straight_side_gate * straight_yaw_gate * flat_gate * stage_gate
    raw_score = torch.clamp(height_weight * height_excess + asymmetry_weight * asymmetry_excess, 0.0, 1.0)
    signal = active_gate * raw_score

    _ensure_highstep_rear_branch_buffers(env)
    env._highstep_fl_forward_flat_sample_steps += torch.ones_like(signal)
    env._highstep_fl_forward_flat_active_gate_sum += active_gate.detach()
    env._highstep_fl_forward_flat_forward_gate_sum += forward_gate.detach()
    env._highstep_fl_forward_flat_backward_gate_sum += backward_gate.detach()
    env._highstep_fl_forward_flat_lateral_gate_sum += lateral_gate.detach()
    env._highstep_fl_forward_flat_yaw_gate_sum += yaw_gate.detach()
    env._highstep_fl_forward_flat_flat_gate_sum += flat_gate.detach()
    env._highstep_fl_forward_flat_highstep_relief_sum += highstep_relief.detach()
    env._highstep_fl_forward_flat_fl_lift_sum += fl_rel_z.detach()
    env._highstep_fl_forward_flat_fr_lift_sum += fr_rel_z.detach()
    env._highstep_fl_forward_flat_fl_minus_fr_sum += fl_minus_fr.detach()
    env._highstep_fl_forward_flat_height_excess_sum += height_excess.detach()
    env._highstep_fl_forward_flat_asym_excess_sum += asymmetry_excess.detach()
    env._highstep_fl_forward_flat_overlift_sum += ((raw_score > 0.0) & (active_gate > 0.01)).float()
    env._highstep_fl_forward_flat_penalty_sum += signal.detach()
    env._highstep_fl_forward_flat_penalty_max = torch.maximum(
        env._highstep_fl_forward_flat_penalty_max, signal.detach()
    )

    return signal


# ==========================================
# 3. 后腿发力蹬踏 (Rear Legs Power Drive) - 修改版
# ==========================================
def rear_legs_power_drive_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    rear_drive_joint_names: list,
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    nominal_base_height: float = 0.44,
    min_body_lift: float = 0.02,
    target_positive_power: float = 120.0,
    max_power_score: float = 1.0,
    front_foot_names: list[str] | None = None,
    rear_foot_names: list[str] | None = None,
    commit_gate_scale: float = 0.75,
    lift_gate_floor: float = 0.30,
    use_abs_power: bool = False,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """
    Reward rear-leg positive power only when it helps a detected forward high-step climb.

    The reward is bounded so rear-leg flailing cannot dominate the real high-step progress terms.
    """
    asset = env.scene[asset_cfg.name]
    gate, height_delta, front_terrain_z = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(asset, front_foot_names, rear_foot_names)
    gate = torch.maximum(gate, commit_gate_scale * commit_gate)
    rear_terrain_z = front_terrain_z - height_delta
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    # 找到后腿发力关节（thigh 和 calf）的索引
    rear_joint_ids = asset.find_joints(rear_drive_joint_names)[0]

    # 获取当前 Pitch 角
    projected_gravity = asset.data.projected_gravity_b
    pitch_gate = torch.clamp((-projected_gravity[:, 0] - 0.05) / 0.25, 0.0, 1.0)
    body_lift = asset.data.root_pos_w[:, 2] - (rear_terrain_z + nominal_base_height)
    lift_gate = torch.clamp((body_lift - min_body_lift) / 0.10, 0.0, 1.0)

    # 获取后腿指定关节的输出扭矩和速度
    rear_torques = asset.data.applied_torque[:, rear_joint_ids]
    rear_vel = asset.data.joint_vel[:, rear_joint_ids]

    # Mechanical power = torque * velocity.  Joint sign conventions can differ
    # between models, so high-step training can optionally use bounded absolute
    # power after the high-step gates have already selected the relevant state.
    power = rear_torques * rear_vel
    drive_power = torch.abs(power) if use_abs_power else torch.clamp(power, min=0.0)
    drive_lift_gate = torch.clamp(
        lift_gate_floor + (1.0 - lift_gate_floor) * lift_gate,
        min=0.0,
        max=1.0,
    )

    # 只有在高台、向前、抬头、身体开始上沿时，才奖励后腿发力。
    power_score = torch.clamp(
        torch.sum(drive_power, dim=1) / max(target_positive_power, 1.0e-6),
        min=0.0,
        max=max_power_score,
    )
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    bonus = power_score * gate * cmd_gate * pitch_gate * drive_lift_gate * stage_gate
    return bonus


def highstep_rear_support_motion_contract(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    rear_mirror_joint_pairs: list[list[str]],
    rear_mirror_joint_signs: list[float],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    nominal_base_height: float = 0.44,
    min_rear_width: float = 0.34,
    rear_width_window: float = 0.10,
    min_rear_abs_y: float = 0.15,
    rear_center_window: float = 0.08,
    fore_aft_symmetry_scale: float = 0.10,
    lateral_center_scale: float = 0.08,
    rear_slip_scale: float = 0.18,
    action_symmetry_scale: float = 0.35,
    joint_symmetry_scale: float = 0.35,
    contact_threshold: float = 5.0,
    contact_force_window: float = 40.0,
    roll_scale: float = 0.20,
    roll_rate_scale: float = 0.90,
    yaw_rate_scale: float = 0.55,
    body_lift_target: float = 0.10,
    body_lift_velocity_target: float = 0.18,
    first_rear_clear_start: float = 0.20,
    first_rear_clear_full: float = 0.52,
    second_rear_clear_start: float = 0.52,
    second_rear_clear_full: float = 0.82,
    front_commit_ready_start: float = 0.28,
    front_commit_ready_full: float = 0.58,
    inward_penalty_scale: float = 1.0,
    premature_rear_penalty_scale: float = 0.8,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Rear-only motion contract for a stable high-step entry and recovery.

    Before either rear foot leaves the lower surface, both rear feet are the
    anchor: they should remain in bilateral contact, move little in the world
    frame, stay symmetric about the body centerline, and drive a stable body
    lift with mirrored rear revolute-joint actions.  Once the first rear foot
    starts to clear the platform edge, same-trajectory action symmetry is no
    longer appropriate; the contract then protects the stance rear foot,
    centerline margin, and body attitude while the two rear feet climb in
    sequence.  Front-foot geometry is used only as a phase-transition signal.

    The term intentionally excludes box-response tuning, front-leg shaping,
    domain-randomization changes, and any deployment-time guard.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain_gate, height_delta, front_terrain_z = _forward_highstep_terrain_gate(
        env,
        sensor_cfg,
        front_x_min,
        rear_x_max,
        max_abs_y,
        height_threshold,
        height_gate_width,
    )
    commit_gate = _front_feet_highstep_commit_gate(
        asset, front_foot_names, rear_foot_names
    )
    task_gate = torch.maximum(terrain_gate, 0.85 * commit_gate)
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, min=0.0, max=1.0)
    stage_gate = _highstep_training_progress_gate(
        env, stage_start_update, stage_ramp_updates, num_steps_per_update
    )
    active_gate = task_gate * cmd_gate * stage_gate

    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    rear_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    rear_rel_w = rear_pos_w - asset.data.root_pos_w[:, None, :]
    heading = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, len(rear_foot_ids), -1)
    rear_pos_b = quat_apply_inverse(
        heading.reshape(-1, 4), rear_rel_w.reshape(-1, 3)
    ).reshape(rear_rel_w.shape)

    rear_x = rear_pos_b[..., 0]
    rear_y = rear_pos_b[..., 1]
    rear_width = torch.abs(rear_y[:, 0] - rear_y[:, 1])
    rear_min_abs_y = torch.min(torch.abs(rear_y), dim=1).values
    width_deficit = torch.clamp(
        (min_rear_width - rear_width) / max(rear_width_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    center_deficit = torch.clamp(
        (min_rear_abs_y - rear_min_abs_y) / max(rear_center_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    width_score = 1.0 - width_deficit
    fore_aft_score = torch.exp(
        -torch.square(torch.abs(rear_x[:, 0] - rear_x[:, 1]) / max(fore_aft_symmetry_scale, 1.0e-6))
    )
    lateral_center_score = torch.exp(
        -torch.square(torch.abs(rear_y[:, 0] + rear_y[:, 1]) / max(lateral_center_scale, 1.0e-6))
    )

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    rear_contact_ids = contact_sensor.find_bodies(rear_foot_names)[0]
    rear_force_z = torch.abs(
        contact_sensor.data.net_forces_w[:, rear_contact_ids, 2]
    )
    rear_contact_score = torch.clamp(
        (rear_force_z - contact_threshold) / max(contact_force_window, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    bilateral_contact = torch.min(rear_contact_score, dim=1).values
    stance_contact = torch.max(rear_contact_score, dim=1).values
    force_balance = 1.0 - torch.clamp(
        torch.abs(rear_force_z[:, 0] - rear_force_z[:, 1])
        / torch.clamp(rear_force_z[:, 0] + rear_force_z[:, 1], min=contact_threshold),
        min=0.0,
        max=1.0,
    )

    rear_foot_speed_xy = torch.linalg.norm(
        asset.data.body_lin_vel_w[:, rear_foot_ids, :2], dim=-1
    )
    contact_weighted_slip = torch.max(
        rear_foot_speed_xy * rear_contact_score.detach(), dim=1
    ).values
    anchor_score = torch.exp(
        -torch.square(contact_weighted_slip / max(rear_slip_scale, 1.0e-6))
    )

    if len(rear_mirror_joint_pairs) != len(rear_mirror_joint_signs):
        raise ValueError(
            "rear_mirror_joint_pairs and rear_mirror_joint_signs must have equal length"
        )
    if not hasattr(env, "_highstep_rear_support_contract_joint_pairs"):
        env._highstep_rear_support_contract_joint_pairs = [
            (
                int(asset.find_joints(pair[0])[0][0]),
                int(asset.find_joints(pair[1])[0][0]),
            )
            for pair in rear_mirror_joint_pairs
        ]
    pair_ids = env._highstep_rear_support_contract_joint_pairs
    action_diffs = torch.stack(
        [
            torch.abs(
                env.action_manager.action[:, left_id]
                - float(mirror_sign) * env.action_manager.action[:, right_id]
            )
            for (left_id, right_id), mirror_sign in zip(
                pair_ids, rear_mirror_joint_signs, strict=True
            )
        ],
        dim=1,
    )
    joint_diffs = torch.stack(
        [
            torch.abs(
                asset.data.joint_pos[:, left_id]
                - float(mirror_sign) * asset.data.joint_pos[:, right_id]
            )
            for (left_id, right_id), mirror_sign in zip(
                pair_ids, rear_mirror_joint_signs, strict=True
            )
        ],
        dim=1,
    )
    action_symmetry_score = torch.exp(
        -torch.square(
            torch.mean(action_diffs, dim=1) / max(action_symmetry_scale, 1.0e-6)
        )
    )
    joint_symmetry_score = torch.exp(
        -torch.square(
            torch.mean(joint_diffs, dim=1) / max(joint_symmetry_scale, 1.0e-6)
        )
    )

    roll_metric = torch.abs(asset.data.projected_gravity_b[:, 1])
    roll_rate = torch.abs(asset.data.root_ang_vel_b[:, 0])
    yaw_rate_error = torch.abs(
        asset.data.root_ang_vel_b[:, 2]
        - env.command_manager.get_command(command_name)[:, 2]
    )
    posture_score = torch.exp(
        -torch.square(roll_metric / max(roll_scale, 1.0e-6))
        -torch.square(roll_rate / max(roll_rate_scale, 1.0e-6))
        -torch.square(yaw_rate_error / max(yaw_rate_scale, 1.0e-6))
    )

    rear_clearance = rear_pos_w[..., 2] - front_terrain_z[:, None]
    rear_clearance_score = torch.clamp(
        (rear_clearance + 0.18) / 0.18, min=0.0, max=1.0
    )
    first_rear_score = torch.max(rear_clearance_score, dim=1).values
    second_rear_score = torch.min(rear_clearance_score, dim=1).values
    first_rear_gate = torch.clamp(
        (first_rear_score - first_rear_clear_start)
        / max(first_rear_clear_full - first_rear_clear_start, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    second_rear_gate = torch.clamp(
        (second_rear_score - second_rear_clear_start)
        / max(second_rear_clear_full - second_rear_clear_start, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    front_commit_ready = torch.clamp(
        (commit_gate - front_commit_ready_start)
        / max(front_commit_ready_full - front_commit_ready_start, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    rear_ground_z = front_terrain_z - height_delta
    body_lift = asset.data.root_pos_w[:, 2] - (
        rear_ground_z + nominal_base_height
    )
    body_lift_score = torch.clamp(
        body_lift / max(body_lift_target, 1.0e-6), min=0.0, max=1.0
    )
    body_lift_velocity_score = torch.clamp(
        asset.data.root_lin_vel_w[:, 2]
        / max(body_lift_velocity_target, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    lift_progress = 0.70 * body_lift_score + 0.30 * body_lift_velocity_score

    double_support_phase = 1.0 - first_rear_gate
    sequential_phase = first_rear_gate * (1.0 - second_rear_gate)
    hold_phase = second_rear_gate
    symmetry_score = (
        fore_aft_score
        * lateral_center_score
        * width_score
        * action_symmetry_score
        * joint_symmetry_score
    )
    double_support_contract = (
        bilateral_contact
        * anchor_score
        * force_balance
        * posture_score
        * symmetry_score
    )
    double_support_reward = (
        double_support_phase
        * double_support_contract
        * (0.35 + 0.65 * lift_progress)
    )

    lateral_safety = width_score * (1.0 - center_deficit)
    sequential_reward = (
        sequential_phase
        * stance_contact
        * lateral_safety
        * posture_score
        * (0.45 + 0.55 * first_rear_gate)
    )
    hold_reward = hold_phase * lateral_safety * posture_score

    inward_penalty = torch.clamp(
        0.55 * width_deficit + 0.45 * center_deficit, min=0.0, max=1.0
    )
    premature_rear_penalty = first_rear_gate * (1.0 - front_commit_ready)
    signal = active_gate * (
        double_support_reward
        + 0.70 * sequential_reward
        + 0.55 * hold_reward
        - inward_penalty_scale * inward_penalty
        - premature_rear_penalty_scale * premature_rear_penalty
    )

    # Expose detached diagnostics without introducing any extra observation or
    # deployment-time state.  The supervisor/evaluator reads these only after
    # rollout; training gradients flow exclusively through ``signal``.
    env._highstep_rear_support_contract_metrics = {
        "active_gate": active_gate.detach(),
        "double_support_phase": double_support_phase.detach(),
        "sequential_phase": sequential_phase.detach(),
        "hold_phase": hold_phase.detach(),
        "rear_width": rear_width.detach(),
        "rear_min_abs_y": rear_min_abs_y.detach(),
        "fore_aft_error": torch.abs(rear_x[:, 0] - rear_x[:, 1]).detach(),
        "lateral_center_error": torch.abs(rear_y[:, 0] + rear_y[:, 1]).detach(),
        "rear_slip": contact_weighted_slip.detach(),
        "force_balance": force_balance.detach(),
        "action_symmetry_score": action_symmetry_score.detach(),
        "joint_symmetry_score": joint_symmetry_score.detach(),
        "posture_score": posture_score.detach(),
        "front_commit_ready": front_commit_ready.detach(),
        "first_rear_gate": first_rear_gate.detach(),
        "second_rear_gate": second_rear_gate.detach(),
    }
    return signal


def highstep_rear_push_posture_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    nominal_base_height: float = 0.44,
    rear_back_min: float = 0.18,
    rear_back_target: float = 0.46,
    min_front_rear_height_diff: float = 0.08,
    target_front_rear_height_diff: float = 0.28,
    min_distance: float = 0.22,
    target_distance: float = 0.65,
    min_height_gain: float = -0.02,
    target_height_gain: float = 0.08,
    commit_gate_floor: float = 0.30,
    progress_floor: float = 0.20,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward the 00059-style rear-leg push after the front feet have committed.

    This term intentionally does not reward a rear-stretched posture by itself.
    The posture is useful only when it happens with body advance/lift, which
    prevents the high-step policy from getting paid for a static bridge.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(
        asset,
        front_foot_names,
        rear_foot_names,
        min_height_diff=0.04,
        target_height_diff=0.18,
        min_front_x=0.08,
        target_front_x=0.34,
        min_pitch_metric=0.01,
        target_pitch_metric=0.24,
    )
    gate = torch.maximum(terrain_gate * commit_gate, commit_gate_floor * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    front_ids = asset.find_bodies(front_foot_names)[0]
    rear_ids = asset.find_bodies(rear_foot_names)[0]
    front_z = asset.data.body_pos_w[:, front_ids, 2].mean(dim=1)
    rear_pos_w = asset.data.body_pos_w[:, rear_ids, :]
    rear_z = rear_pos_w[..., 2].mean(dim=1)

    rear_rel_w = rear_pos_w - asset.data.root_pos_w[:, None, :]
    num_rear_feet = rear_rel_w.shape[1]
    heading_quat = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, num_rear_feet, -1)
    rear_rel_b = quat_apply_inverse(
        heading_quat.reshape(-1, 4),
        rear_rel_w.reshape(-1, 3),
    ).reshape(rear_rel_w.shape)
    rear_back = torch.clamp(-rear_rel_b[..., 0].mean(dim=1), min=0.0)
    rear_back_score = torch.clamp(
        (rear_back - rear_back_min) / max(rear_back_target - rear_back_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    height_diff = front_z - rear_z
    height_diff_score = torch.clamp(
        (height_diff - min_front_rear_height_diff)
        / max(target_front_rear_height_diff - min_front_rear_height_diff, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_gain_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    progress_score = distance_score * (0.35 + 0.65 * height_gain_score)
    progress_gate = torch.clamp(progress_floor + (1.0 - progress_floor) * progress_score, 0.0, 1.0)

    pitch_metric = -asset.data.projected_gravity_b[:, 0]
    safe_pitch = pitch_metric < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * rear_back_score * height_diff_score * progress_gate * safe_pitch * stage_gate


def highstep_rear_box_push_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    box_joint_names: dict[str, str],
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.05,
    front_x_min: float = 0.15,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.35,
    height_threshold: float = 0.035,
    height_gate_width: float = 0.12,
    nominal_base_height: float = 0.44,
    rear_push_target: float = 0.004,
    target_std: float = 0.012,
    min_distance: float = 0.18,
    target_distance: float = 0.60,
    min_height_gain: float = -0.02,
    target_height_gain: float = 0.08,
    commit_gate_floor: float = 0.35,
    progress_floor: float = 0.30,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward rear box-joint extension only in the committed push phase."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(
        asset,
        front_foot_names,
        rear_foot_names,
        min_height_diff=0.04,
        target_height_diff=0.18,
        min_front_x=0.08,
        target_front_x=0.34,
        min_pitch_metric=0.01,
        target_pitch_metric=0.24,
    )
    gate = torch.maximum(terrain_gate * commit_gate, commit_gate_floor * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.20, 0.0, 1.0)

    rear_joint_names = [box_joint_names[k] for k in ("RL", "RR")]
    joint_ids = []
    for joint_name in rear_joint_names:
        ids = asset.find_joints(joint_name)[0]
        joint_ids.append(ids[0])
    joint_ids = torch.as_tensor(joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)
    rear_box_pos = asset.data.joint_pos[:, joint_ids]
    err = torch.mean(torch.square((rear_box_pos - rear_push_target) / max(target_std, 1.0e-6)), dim=1)
    box_score = torch.exp(-err)

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_gain_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    progress_score = distance_score * (0.35 + 0.65 * height_gain_score)
    progress_gate = torch.clamp(progress_floor + (1.0 - progress_floor) * progress_score, 0.0, 1.0)

    pitch_metric = -asset.data.projected_gravity_b[:, 0]
    safe_pitch = pitch_metric < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * box_score * progress_gate * safe_pitch * stage_gate


def highstep_bridge_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.08,
    front_x_min: float = 0.25,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.30,
    height_threshold: float = 0.06,
    height_gate_width: float = 0.14,
    nominal_base_height: float = 0.44,
    rear_back_min: float = 0.18,
    rear_back_target: float = 0.46,
    min_distance: float = 0.20,
    target_distance: float = 0.58,
    min_height_gain: float = -0.02,
    target_height_gain: float = 0.07,
    commit_gate_floor: float = 0.30,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize the stuck bridge state: committed front feet, stretched rear legs, no climb progress."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(
        asset,
        front_foot_names,
        rear_foot_names,
        min_height_diff=0.04,
        target_height_diff=0.18,
        min_front_x=0.08,
        target_front_x=0.34,
        min_pitch_metric=0.01,
        target_pitch_metric=0.24,
    )
    gate = torch.maximum(terrain_gate * commit_gate, commit_gate_floor * commit_gate)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.25, 0.0, 1.0)

    rear_ids = asset.find_bodies(rear_foot_names)[0]
    rear_rel_w = asset.data.body_pos_w[:, rear_ids, :] - asset.data.root_pos_w[:, None, :]
    num_rear_feet = rear_rel_w.shape[1]
    heading_quat = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, num_rear_feet, -1)
    rear_rel_b = quat_apply_inverse(
        heading_quat.reshape(-1, 4),
        rear_rel_w.reshape(-1, 3),
    ).reshape(rear_rel_w.shape)
    rear_back = torch.clamp(-rear_rel_b[..., 0].mean(dim=1), min=0.0)
    rear_back_score = torch.clamp(
        (rear_back - rear_back_min) / max(rear_back_target - rear_back_min, 1.0e-6),
        min=0.0,
        max=1.0,
    )

    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    distance_score = torch.clamp(
        (distance - min_distance) / max(target_distance - min_distance, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    height_gain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - nominal_base_height
    height_gain_score = torch.clamp(
        (height_gain - min_height_gain) / max(target_height_gain - min_height_gain, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    progress_score = torch.maximum(distance_score, height_gain_score)
    stall_score = torch.clamp(1.0 - progress_score, 0.0, 1.0)

    pitch_metric = -asset.data.projected_gravity_b[:, 0]
    safe_pitch = pitch_metric < 0.95
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return gate * cmd_gate * rear_back_score * stall_score * safe_pitch * stage_gate


def highstep_box_phase_prior_alignment_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    box_joint_names: dict[str, str],
    front_foot_names: list[str],
    rear_foot_names: list[str],
    min_cmd_x: float = 0.05,
    front_x_min: float = 0.15,
    rear_x_max: float = -0.20,
    max_abs_y: float = 0.35,
    height_threshold: float = 0.035,
    height_gate_width: float = 0.12,
    commit_gate_floor: float = 0.25,
    front_reach_target: float = 0.014,
    rear_approach_target: float = 0.034,
    front_support_target: float = 0.034,
    rear_push_target: float = 0.012,
    target_std: float = 0.015,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Reward the phased high-step box posture used by the teacher action prior.

    Smaller box-joint position means a longer telescopic leg.  Before the
    front feet commit, this rewards front-leg extension for reaching the step;
    after commitment, it rewards rear-leg extension for pushing the body up.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_gate, _, _ = _forward_highstep_terrain_gate(
        env, sensor_cfg, front_x_min, rear_x_max, max_abs_y, height_threshold, height_gate_width
    )
    commit_gate = _front_feet_highstep_commit_gate(
        asset,
        front_foot_names,
        rear_foot_names,
        min_height_diff=0.04,
        target_height_diff=0.18,
        min_front_x=0.08,
        target_front_x=0.34,
        min_pitch_metric=0.01,
        target_pitch_metric=0.24,
    )
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    cmd_gate = torch.clamp((cmd_x - min_cmd_x) / 0.20, 0.0, 1.0)

    reach_gate = terrain_gate * (1.0 - commit_gate) * cmd_gate
    push_gate = (
        torch.maximum(terrain_gate * commit_gate, commit_gate_floor * commit_gate)
        * cmd_gate
    )
    phase_gate = torch.maximum(reach_gate, push_gate)

    joint_names = [box_joint_names[k] for k in ("FL", "FR", "RL", "RR")]
    joint_ids = []
    for joint_name in joint_names:
        ids = asset.find_joints(joint_name)[0]
        joint_ids.append(ids[0])
    joint_ids = torch.as_tensor(joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)
    box_pos = asset.data.joint_pos[:, joint_ids]

    reach_target = torch.tensor(
        [front_reach_target, front_reach_target, rear_approach_target, rear_approach_target],
        device=box_pos.device,
        dtype=box_pos.dtype,
    )
    push_target = torch.tensor(
        [front_support_target, front_support_target, rear_push_target, rear_push_target],
        device=box_pos.device,
        dtype=box_pos.dtype,
    )
    target_weight = torch.clamp(reach_gate + push_gate, min=1.0e-6).unsqueeze(1)
    target = (reach_gate.unsqueeze(1) * reach_target + push_gate.unsqueeze(1) * push_target) / target_weight

    err = torch.mean(torch.square((box_pos - target) / max(target_std, 1.0e-6)), dim=1)
    stage_gate = _highstep_training_progress_gate(env, stage_start_update, stage_ramp_updates, num_steps_per_update)
    return phase_gate * torch.exp(-err) * stage_gate

# ==========================================
# 4. 仅惩罚 Roll 和 Yaw，放开 Pitch (Roll-Yaw Only Penalty)
# ==========================================
def roll_yaw_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    替代原来的 flat_orientation_l2。
    允许机器人抬头(Pitch)，但严厉惩罚左右侧翻(Roll)和不必要的偏航(Yaw)。
    """
    asset = env.scene[asset_cfg.name]
    # projected_gravity_b = [sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)]
    # 我们只惩罚 Y 轴分量 (对应 Roll)
    roll_penalty = torch.square(asset.data.projected_gravity_b[:, 1])
    return roll_penalty


def action_rate_l2_by_name(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    通过 SceneEntityCfg 指定关节名称，只惩罚这些关节的动作变化率。
    """
    # 获取机器人资产
    asset = env.scene[asset_cfg.name]

    # 根据传入的正则表达式解析出具体的关节索引
    # 注意：这里获取的是在驱动关节列表中的索引，通常与 action 的索引一一对应
    joint_indices, _ = asset.find_joints(asset_cfg.joint_names)

    # 切片提取特定关节的动作
    current_action = env.action_manager.action[:, joint_indices]
    prev_action = env.action_manager.prev_action[:, joint_indices]

    return torch.sum(torch.square(current_action - prev_action), dim=1)


def lateral_step_scaled_action_rate_l2_by_name(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_body_names: dict[str, str],
    height_threshold: float,
    gate_width: float,
    relief_scale: float,
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 5.0,
    min_contacts_per_side: int = 1,
) -> torch.Tensor:
    """Reduce named-joint action-rate penalty when lateral-step posture correction is needed."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = action_rate_l2_by_name(env, asset_cfg)
    relief = _lateral_step_relief_scale(
        env,
        asset,
        foot_body_names,
        height_threshold,
        gate_width,
        relief_scale,
        contact_sensor_cfg,
        contact_threshold,
        min_contacts_per_side,
    )
    return penalty * relief


def action_rate_l2_by_name_command_scale(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 1.0,
    moving_scale: float = 0.5,
    command_threshold: float = 0.1,
    ignore_yaw_command: bool = False,
) -> torch.Tensor:
    """Scale named-joint action-rate penalty by whether the velocity command is still or moving."""
    penalty = action_rate_l2_by_name(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    command_norm = _stand_still_command_norm(command, ignore_yaw_command)
    scale = torch.where(
        command_norm < command_threshold,
        torch.full_like(penalty, stand_still_scale),
        torch.full_like(penalty, moving_scale),
    )
    return penalty * scale


def _stand_still_command_norm(command: torch.Tensor, ignore_yaw_command: bool) -> torch.Tensor:
    if ignore_yaw_command:
        return torch.norm(command[:, :2], dim=1)
    return torch.norm(command[:, :3], dim=1)


def stand_still_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ignore_yaw_command: bool = False,
) -> torch.Tensor:
    """
    自定义惩罚：当机器人收到静止指令时，严厉惩罚任何关节的速度（即摇晃和抽搐）。
    不影响移动时的步态，也不限制静止时的具体关节角度。
    """
    # 获取机器人实体
    robot = env.scene[asset_cfg.name]

    # 获取速度指令 (通常是 [lin_x, lin_y, ang_z])
    command = env.command_manager.get_command(command_name)

    command_norm = _stand_still_command_norm(command, ignore_yaw_command)

    # 判断是否处于“静止状态” (指令速度小于阈值)
    is_standing_still = command_norm < command_threshold

    # 【关键修改】只获取 asset_cfg 中指定的关节速度
    # Isaac Lab 的 Reward Manager 会自动解析 joint_names 并将其转换为 joint_ids
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]

    # 计算所有关节速度的平方和 (dof_vel^2)
    # 速度越大，平方后的惩罚越重
    # joint_vel_sq = torch.sum(torch.square(robot.data.joint_vel), dim=1) #非解耦时的写法
    joint_vel_sq = torch.sum(torch.square(joint_vel), dim=1)

    # 只有在静止时才输出惩罚值，移动时输出 0
    return is_standing_still.float() * joint_vel_sq


def lateral_step_scaled_stand_still_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_body_names: dict[str, str],
    height_threshold: float,
    gate_width: float,
    relief_scale: float,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ignore_yaw_command: bool = False,
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 5.0,
    min_contacts_per_side: int = 1,
) -> torch.Tensor:
    """Reduce stillness joint-velocity penalty while correcting lateral-step posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = stand_still_joint_vel_penalty(
        env,
        command_name,
        command_threshold=command_threshold,
        asset_cfg=asset_cfg,
        ignore_yaw_command=ignore_yaw_command,
    )
    relief = _lateral_step_relief_scale(
        env,
        asset,
        foot_body_names,
        height_threshold,
        gate_width,
        relief_scale,
        contact_sensor_cfg,
        contact_threshold,
        min_contacts_per_side,
    )
    return penalty * relief

def stand_still_base_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ignore_yaw_command: bool = False,
) -> torch.Tensor:
    """
    惩罚静止时的机身角速度。
    允许机器人以任何姿态站立，但严厉惩罚机身的晃动（Roll, Pitch, Yaw 的变化率）。
    这能有效迫使策略学会“柔和纠正”，增加系统的阻尼，防止真机震荡发散。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 判断是否处于静止指令
    command_norm = _stand_still_command_norm(command, ignore_yaw_command)
    is_standing_still = command_norm < command_threshold

    # 获取机身在世界坐标系下的角速度 (root_ang_vel_w)
    # 也可以使用相对于机身坐标系的角速度 (root_ang_vel_b)，效果类似
    base_ang_vel_sq = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)

    return is_standing_still.float() * base_ang_vel_sq


def stand_still_base_lin_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ignore_yaw_command: bool = False,
) -> torch.Tensor:
    """
    惩罚静止时的机身线速度。
    抑制机身的 X, Y, Z 平移晃动（例如前后左右平移或上下起伏）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 判断是否处于静止指令
    command_norm = _stand_still_command_norm(command, ignore_yaw_command)
    is_standing_still = command_norm < command_threshold

    # 获取机身在世界坐标系下的线速度的平方和
    base_lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)

    return is_standing_still.float() * base_lin_vel_sq

def base_height_l2_relaxed_on_tilt(
    env: ManagerBasedRLEnv,
    target_height: float,
    tilt_sensitivity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize base height error, but relax the penalty when the robot is tilted.

    当机器人水平时，施加正常的 L2 高度惩罚。
    当机器人倾斜时（例如爬坡、上下台阶），自动减小高度惩罚的力度。
    """
    # 获取机器人资产
    asset: RigidObject = env.scene[asset_cfg.name]

    # 1. 计算标准的高度误差平方: (z - z_target)^2
    base_height = asset.data.root_pos_w[:, 2]
    height_error_sq = torch.square(base_height - target_height)

    # 2. 计算倾斜系数 (基于投影重力向量)
    # projected_gravity_b 是世界坐标系的重力向量 [0, 0, -1] 投影到机器人机身坐标系下的结果。
    # 当机器人完全水平时，它的 Z 分量 g_z 接近 -1.0。
    # 当机器人倾斜 90 度时，它的 Z 分量 g_z 接近 0.0。
    g_z = asset.data.projected_gravity_b[:, 2]

    # 取绝对值得到 flatness (平坦度): 水平时为 1.0，越倾斜越接近 0.0
    flatness = torch.abs(g_z)

    # 3. 应用敏感度调节
    # tilt_sensitivity 控制惩罚衰减的速度：
    # = 1.0: 线性衰减
    # > 1.0 (例如 2.0): 稍微一倾斜，惩罚就迅速减小 (对倾斜更宽容)
    # < 1.0 (例如 0.5): 倾斜很多时，惩罚才明显减小 (依然比较严格)
    relaxation_factor = torch.pow(flatness, tilt_sensitivity)

    # 最终惩罚 = 原始误差 * 放宽系数
    return height_error_sq * relaxation_factor

def feet_stance_width_penalty(env: ManagerBasedRLEnv, min_width: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    惩罚足端间距过小（防止内八字/走钢丝步态）。
    计算每只脚在机身坐标系下的 Y 轴绝对距离，如果小于 min_width / 2 则产生惩罚。
    """
    # 获取机器人资产
    robot: RigidObject = env.scene[asset_cfg.name]

    # 获取足端和机身在世界坐标系下的位置与姿态
    feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :] # (num_envs, num_feet, 3)
    base_pos_w = robot.data.root_pos_w # (num_envs, 3)
    base_quat_w = robot.data.root_quat_w # (num_envs, 4)

    num_envs = env.num_envs
    num_feet = feet_pos_w.shape[1]

    # 扩展 base 的位置和姿态，以便与所有脚进行批量计算
    base_pos_w_rep = base_pos_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 3)
    base_quat_w_rep = base_quat_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 4)
    feet_pos_w_flat = feet_pos_w.view(-1, 3)

    # 将足端坐标从世界坐标系转换到机身坐标系 (Base Frame)
    feet_pos_b = quat_apply_inverse(base_quat_w_rep, feet_pos_w_flat - base_pos_w_rep)
    feet_pos_b = feet_pos_b.view(num_envs, num_feet, 3)

    # 提取在机身坐标系下的 Y 轴坐标（左右方向）
    feet_y = feet_pos_b[:, :, 1]

    # 计算惩罚：如果单侧脚距离中心线小于 min_width / 2，则惩罚差值
    penalty = torch.clamp(min_width / 2.0 - torch.abs(feet_y), min=0.0)

    # 将所有脚的惩罚相加
    return torch.sum(penalty, dim=1)

# def feet_stance_width_advanced_adaptive_penalty(
#     env: ManagerBasedRLEnv,
#     flat_min_width: float,
#     rough_stationary_min_width: float,
#     rough_moving_min_width: float,
#     stationary_speed_threshold: float,
#     moving_speed_threshold: float,
#     asset_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """
#     进阶自适应足端间距惩罚。
#     - 平地：使用 flat_min_width (最严格)
#     - 复杂地形 + 静止：使用 rough_stationary_min_width (保证站立稳定)
#     - 复杂地形 + 运动：使用 rough_moving_min_width (最宽松，允许灵活跨越)

#     参数:
#         stationary_speed_threshold: 低于此速度完全视为静止状态
#         moving_speed_threshold: 高于此速度完全视为正常运动状态
#     """
#     robot = env.scene[asset_cfg.name]

#     # 获取状态数据
#     feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :] # (num_envs, num_feet, 3)
#     base_pos_w = robot.data.root_pos_w # (num_envs, 3)
#     base_quat_w = robot.data.root_quat_w # (num_envs, 4)
#     base_lin_vel_w = robot.data.root_lin_vel_w # (num_envs, 3) 机身线速度

#     num_envs = env.num_envs
#     num_feet = feet_pos_w.shape[1]

#     # ==========================================
#     # 1. 计算地形崎岖度 (Roughness Factor) 0.0 ~ 1.0
#     # ==========================================
#     feet_z = feet_pos_w[:, :, 2]
#     z_diff = torch.max(feet_z, dim=1)[0] - torch.min(feet_z, dim=1)[0]
#     z_factor = torch.clamp((z_diff - 0.12) / 0.13, min=0.0, max=1.0)

#     gravity_w = torch.tensor([0.0, 0.0, -1.0], device=robot.device).repeat(num_envs, 1)
#     gravity_b = quat_apply_inverse(base_quat_w, gravity_w)
#     tilt = torch.norm(gravity_b[:, :2], dim=1)
#     tilt_factor = torch.clamp((tilt - 0.1) / 0.2, min=0.0, max=1.0)

#     roughness = torch.max(z_factor, tilt_factor) # (num_envs,) 0表示平地，1表示复杂地形

#     # ==========================================
#     # 2. 计算运动状态因子 (Motion Factor) 0.0 ~ 1.0
#     # ==========================================
#     # 计算水平方向的实际移动速度
#     speed = torch.norm(base_lin_vel_w[:, :2], dim=1) # (num_envs,)

#     # 计算速度区间差值 (加入 1e-5 防止除以 0 的异常)
#     speed_diff = max(moving_speed_threshold - stationary_speed_threshold, 1e-5)

#     # 速度 < stationary_speed_threshold 视为静止 (factor=0)
#     # 速度 > moving_speed_threshold 视为正常运动 (factor=1)
#     # 中间状态平滑过渡
#     motion_factor = torch.clamp((speed - stationary_speed_threshold) / speed_diff, min=0.0, max=1.0)

#     # ==========================================
#     # 3. 动态计算目标间距 (Dynamic Min Width)
#     # ==========================================
#     # 步骤 A：先计算如果在复杂地形上，当前速度应该对应的目标间距
#     # 静止时为 rough_stationary_min_width，运动时放宽至 rough_moving_min_width
#     rough_target_width = rough_stationary_min_width + motion_factor * (rough_moving_min_width - rough_stationary_min_width)

#     # 步骤 B：结合地形崎岖度，在平地间距和复杂地形目标间距之间插值
#     current_min_width = flat_min_width + roughness * (rough_target_width - flat_min_width)

#     # 扩展维度以便与 4 条腿广播计算
#     current_min_width = current_min_width.unsqueeze(1).repeat(1, num_feet) # (num_envs, num_feet)

#     # ==========================================
#     # 4. 计算 Y 轴距离并施加惩罚
#     # ==========================================
#     base_pos_w_rep = base_pos_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 3)
#     base_quat_w_rep = base_quat_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 4)
#     feet_pos_w_flat = feet_pos_w.view(-1, 3)

#     feet_pos_b = quat_apply_inverse(base_quat_w_rep, feet_pos_w_flat - base_pos_w_rep)
#     feet_pos_b = feet_pos_b.view(num_envs, num_feet, 3)

#     feet_y = feet_pos_b[:, :, 1]

#     # 只有当实际间距小于动态计算的 current_min_width 时，才产生惩罚
#     penalty = torch.clamp(current_min_width / 2.0 - torch.abs(feet_y), min=0.0)

#     return torch.sum(penalty, dim=1)

def feet_stance_width_adaptive_penalty(
    env: ManagerBasedRLEnv,
    min_width: float,
    command_speed_threshold: float,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    智能自适应足端间距惩罚 (V4 终极版)
    核心逻辑：仅在【复杂地形】且【正在运动】时释放腿距限制以利于攀爬。
    只要机器人停下（无论在平地还是楼梯），或者在平地行走，都会强制要求宽站距以保证稳定性。
    """
    robot = env.scene[asset_cfg.name]

    # 1. 获取状态数据
    feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    num_envs = env.num_envs
    num_feet = feet_pos_w.shape[1]

    # 2. 获取指令速度 (Command Velocity) - 代表机器人的运动意图
    try:
        commands = env.command_manager.get_command("base_velocity")
        cmd_speed = torch.norm(commands[:, :2], dim=1) # (num_envs,)
    except:
        base_lin_vel_w = robot.data.root_lin_vel_w
        cmd_speed = torch.norm(base_lin_vel_w[:, :2], dim=1)

    # 3. 计算运动因子 (Speed Factor): 0.0 表示完全静止，1.0 表示正在运动
    # 阈值可以设小一点，比如 0.1m/s，意味着只要摇杆回中，立刻开始要求宽站距
    speed_factor = torch.clamp(cmd_speed / command_speed_threshold, min=0.0, max=1.0)

    # 4. 计算地形崎岖度 (Roughness Factor): 0.0 表示平地，1.0 表示复杂地形
    feet_z = feet_pos_w[:, :, 2]
    z_diff = torch.max(feet_z, dim=1)[0] - torch.min(feet_z, dim=1)[0]
    z_factor = torch.clamp((z_diff - 0.05) / 0.10, min=0.0, max=1.0)

    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=robot.device).repeat(num_envs, 1)
    gravity_b = quat_apply_inverse(base_quat_w, gravity_w)
    tilt = torch.norm(gravity_b[:, :2], dim=1)
    tilt_factor = torch.clamp((tilt - 0.08) / 0.17, min=0.0, max=1.0)

    roughness = torch.max(z_factor, tilt_factor)

    # =====================================================================
    # 5. 核心逻辑反转：计算惩罚乘子 (0.0 表示不惩罚，1.0 表示满额惩罚)
    # 公式: multiplier = 1.0 - (运动因子 * 地形因子)
    # - 运动(1) * 复杂(1) = 1 -> 乘子 0.0 (不惩罚，自由攀爬)
    # - 静止(0) * 复杂(1) = 0 -> 乘子 1.0 (满额惩罚，楼梯上强制张开腿)
    # - 运动(1) * 平地(0) = 0 -> 乘子 1.0 (满额惩罚，平地宽步态)
    # - 静止(0) * 平地(0) = 0 -> 乘子 1.0 (满额惩罚，平地宽站距)
    # =====================================================================
    penalty_multiplier = 1.0 - (speed_factor * roughness)

    penalty_multiplier = penalty_multiplier.unsqueeze(1).repeat(1, num_feet) # (num_envs, num_feet)

    # 6. 计算 Y 轴距离并施加惩罚
    base_pos_w_rep = base_pos_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 3)
    base_quat_w_rep = base_quat_w.unsqueeze(1).repeat(1, num_feet, 1).view(-1, 4)
    feet_pos_w_flat = feet_pos_w.view(-1, 3)

    feet_pos_b = quat_apply_inverse(base_quat_w_rep, feet_pos_w_flat - base_pos_w_rep)
    feet_pos_b = feet_pos_b.view(num_envs, num_feet, 3)

    feet_y = feet_pos_b[:, :, 1]

    # 基础惩罚：实际间距小于 min_width 时产生
    base_penalty = torch.clamp(min_width / 2.0 - torch.abs(feet_y), min=0.0)

    # 应用智能乘子
    final_penalty = base_penalty * penalty_multiplier

    return torch.sum(final_penalty, dim=1)


def hip_joint_abduction_target_l2(
    env: ManagerBasedRLEnv,
    target_positions: dict[str, float],
    deadband: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize hip joints for moving away from a small abduction target."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(asset.data.joint_pos.shape[1]))[joint_ids]
    elif isinstance(joint_ids, torch.Tensor):
        joint_ids = joint_ids.tolist()
    joint_names = [asset.data.joint_names[joint_id] for joint_id in joint_ids]
    target = torch.tensor(
        [target_positions[joint_name] for joint_name in joint_names],
        device=asset.data.joint_pos.device,
        dtype=asset.data.joint_pos.dtype,
    )
    error = torch.abs(asset.data.joint_pos[:, joint_ids] - target.unsqueeze(0))
    error = torch.clamp(error - deadband, min=0.0)
    return torch.sum(torch.square(error), dim=1)


def hip_joint_abduction_min_l2(
    env: ManagerBasedRLEnv,
    min_positions: dict[str, float],
    deadband: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize hip joints only when they are less abducted than a signed minimum."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(asset.data.joint_pos.shape[1]))[joint_ids]
    elif isinstance(joint_ids, torch.Tensor):
        joint_ids = joint_ids.tolist()
    joint_names = [asset.data.joint_names[joint_id] for joint_id in joint_ids]
    min_target = torch.tensor(
        [min_positions[joint_name] for joint_name in joint_names],
        device=asset.data.joint_pos.device,
        dtype=asset.data.joint_pos.dtype,
    )
    direction = torch.sign(min_target).clamp(min=-1.0, max=1.0)
    signed_pos = asset.data.joint_pos[:, joint_ids] * direction.unsqueeze(0)
    min_abs = torch.abs(min_target).unsqueeze(0)
    error = torch.clamp(min_abs - signed_pos - deadband, min=0.0)
    return torch.sum(torch.square(error), dim=1)


def _ids_from_names(all_names: list[str], names: tuple[str, ...] | list[str]) -> list[int]:
    return [all_names.index(name) for name in names]


def _lateral_step_contact_gate(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg | None,
    contact_threshold: float,
    min_contacts_per_side: int,
) -> torch.Tensor:
    if contact_sensor_cfg is None:
        return torch.ones(env.num_envs, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    contact = torch.linalg.norm(contact_forces, dim=-1) > contact_threshold
    left_contacts = contact[:, 0].float() + contact[:, 2].float()
    right_contacts = contact[:, 1].float() + contact[:, 3].float()
    return ((left_contacts >= min_contacts_per_side) & (right_contacts >= min_contacts_per_side)).float()


def _lateral_step_gate(
    asset: Articulation,
    foot_body_names: dict[str, str],
    height_threshold: float,
    gate_width: float,
    invert_side_height_delta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    foot_ids = _ids_from_names(
        asset.data.body_names,
        [
            foot_body_names["FL"],
            foot_body_names["FR"],
            foot_body_names["RL"],
            foot_body_names["RR"],
        ],
    )
    foot_z = asset.data.body_pos_w[:, foot_ids, 2]
    left_z = 0.5 * (foot_z[:, 0] + foot_z[:, 2])
    right_z = 0.5 * (foot_z[:, 1] + foot_z[:, 3])
    side_height_delta = left_z - right_z
    if invert_side_height_delta:
        side_height_delta = -side_height_delta
    gate = torch.clamp((torch.abs(side_height_delta) - height_threshold) / gate_width, 0.0, 1.0)
    return side_height_delta, gate


def _lateral_step_relief_scale(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    foot_body_names: dict[str, str],
    height_threshold: float,
    gate_width: float,
    relief_scale: float,
    contact_sensor_cfg: SceneEntityCfg | None,
    contact_threshold: float,
    min_contacts_per_side: int,
) -> torch.Tensor:
    _, gate = _lateral_step_gate(asset, foot_body_names, height_threshold, gate_width)
    gate *= _lateral_step_contact_gate(env, contact_sensor_cfg, contact_threshold, min_contacts_per_side)
    relief_scale_tensor = torch.full_like(gate, relief_scale)
    return torch.lerp(torch.ones_like(gate), relief_scale_tensor, gate)


def lateral_step_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    foot_body_names: dict[str, str],
    height_threshold: float,
    gate_width: float,
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 5.0,
    min_contacts_per_side: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body tilt only when the robot straddles a left-right height step."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, gate = _lateral_step_gate(asset, foot_body_names, height_threshold, gate_width)
    gate *= _lateral_step_contact_gate(env, contact_sensor_cfg, contact_threshold, min_contacts_per_side)
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * gate


def lateral_step_hip_abduction_l2(
    env: ManagerBasedRLEnv,
    foot_body_names: dict[str, str],
    hip_joint_names: dict[str, str],
    lower_side_target: float,
    upper_side_target: float,
    deadband: float,
    height_threshold: float,
    gate_width: float,
    invert_side_height_delta: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage the lower-side legs to abduct when left and right feet are at different heights."""
    asset: Articulation = env.scene[asset_cfg.name]
    side_height_delta, gate = _lateral_step_gate(
        asset, foot_body_names, height_threshold, gate_width, invert_side_height_delta
    )
    joint_ids = _ids_from_names(
        asset.data.joint_names,
        [
            hip_joint_names["FL"],
            hip_joint_names["FR"],
            hip_joint_names["RL"],
            hip_joint_names["RR"],
        ],
    )
    left_is_lower = side_height_delta < 0.0
    left_target = torch.where(
        left_is_lower,
        torch.full_like(side_height_delta, lower_side_target),
        torch.full_like(side_height_delta, upper_side_target),
    )
    right_target = torch.where(
        left_is_lower,
        torch.full_like(side_height_delta, -upper_side_target),
        torch.full_like(side_height_delta, -lower_side_target),
    )
    target = torch.stack([left_target, right_target, left_target, right_target], dim=1)
    error = torch.abs(asset.data.joint_pos[:, joint_ids] - target)
    error = torch.clamp(error - deadband, min=0.0)
    return torch.sum(torch.square(error), dim=1) * gate


def lateral_step_hip_abduction_min_l2(
    env: ManagerBasedRLEnv,
    foot_body_names: dict[str, str],
    hip_joint_names: dict[str, str],
    lower_side_min: float,
    upper_side_min: float,
    deadband: float,
    height_threshold: float,
    gate_width: float,
    lower_side_weight: float = 1.0,
    upper_side_weight: float = 1.0,
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 5.0,
    min_contacts_per_side: int = 1,
    invert_side_height_delta: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage the lower side to abduct more without penalizing extra abduction."""
    asset: Articulation = env.scene[asset_cfg.name]
    side_height_delta, gate = _lateral_step_gate(
        asset, foot_body_names, height_threshold, gate_width, invert_side_height_delta
    )
    gate *= _lateral_step_contact_gate(env, contact_sensor_cfg, contact_threshold, min_contacts_per_side)
    joint_ids = _ids_from_names(
        asset.data.joint_names,
        [
            hip_joint_names["FL"],
            hip_joint_names["FR"],
            hip_joint_names["RL"],
            hip_joint_names["RR"],
        ],
    )
    left_is_lower = side_height_delta < 0.0
    left_min = torch.where(
        left_is_lower,
        torch.full_like(side_height_delta, lower_side_min),
        torch.full_like(side_height_delta, upper_side_min),
    )
    right_min = torch.where(
        left_is_lower,
        torch.full_like(side_height_delta, upper_side_min),
        torch.full_like(side_height_delta, lower_side_min),
    )
    min_target = torch.stack([left_min, right_min, left_min, right_min], dim=1)
    direction = torch.tensor(
        [1.0, -1.0, 1.0, -1.0],
        device=asset.data.joint_pos.device,
        dtype=asset.data.joint_pos.dtype,
    ).unsqueeze(0)
    signed_pos = asset.data.joint_pos[:, joint_ids] * direction
    error = torch.clamp(min_target - signed_pos - deadband, min=0.0)
    left_weight = torch.where(
        left_is_lower,
        torch.full_like(side_height_delta, lower_side_weight),
        torch.full_like(side_height_delta, upper_side_weight),
    )
    right_weight = torch.where(
        left_is_lower,
        torch.full_like(side_height_delta, upper_side_weight),
        torch.full_like(side_height_delta, lower_side_weight),
    )
    weights = torch.stack([left_weight, right_weight, left_weight, right_weight], dim=1)
    return torch.sum(torch.square(error) * weights, dim=1) * gate


def lateral_step_box_length_difference_l2(
    env: ManagerBasedRLEnv,
    foot_body_names: dict[str, str],
    box_joint_names: dict[str, str],
    min_lower_upper_delta: float,
    deadband: float,
    height_threshold: float,
    gate_width: float,
    pairwise: bool = False,
    command_name: str | None = None,
    command_threshold: float = 0.12,
    body_velocity_threshold: float = 0.18,
    low_speed_gate_width: float = 0.20,
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 5.0,
    min_contacts_per_side: int = 1,
    invert_side_height_delta: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage lower-side box joints to be longer than upper-side joints by a small margin."""
    asset: Articulation = env.scene[asset_cfg.name]
    side_height_delta, gate = _lateral_step_gate(
        asset, foot_body_names, height_threshold, gate_width, invert_side_height_delta
    )
    gate *= _lateral_step_contact_gate(env, contact_sensor_cfg, contact_threshold, min_contacts_per_side)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)
        body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        command_gate = torch.clamp(
            (command_threshold + low_speed_gate_width - command_speed) / low_speed_gate_width, 0.0, 1.0
        )
        body_gate = torch.clamp(
            (body_velocity_threshold + low_speed_gate_width - body_speed) / low_speed_gate_width, 0.0, 1.0
        )
        gate *= command_gate * body_gate
    joint_ids = _ids_from_names(
        asset.data.joint_names,
        [
            box_joint_names["FL"],
            box_joint_names["FR"],
            box_joint_names["RL"],
            box_joint_names["RR"],
        ],
    )
    box_pos = asset.data.joint_pos[:, joint_ids]
    if pairwise:
        left_is_lower = side_height_delta < 0.0
        left_lengths = torch.stack([box_pos[:, 0], box_pos[:, 2]], dim=1)
        right_lengths = torch.stack([box_pos[:, 1], box_pos[:, 3]], dim=1)
        lower_lengths = torch.where(left_is_lower.unsqueeze(1), left_lengths, right_lengths)
        upper_lengths = torch.where(left_is_lower.unsqueeze(1), right_lengths, left_lengths)
        error = torch.clamp(min_lower_upper_delta - (lower_lengths - upper_lengths) - deadband, min=0.0)
        return torch.mean(torch.square(error), dim=1) * gate

    left_length = 0.5 * (box_pos[:, 0] + box_pos[:, 2])
    right_length = 0.5 * (box_pos[:, 1] + box_pos[:, 3])
    left_is_lower = side_height_delta < 0.0
    lower_length = torch.where(left_is_lower, left_length, right_length)
    upper_length = torch.where(left_is_lower, right_length, left_length)
    error = torch.clamp(min_lower_upper_delta - (lower_length - upper_length) - deadband, min=0.0)
    return torch.square(error) * gate


def diagonal_gait_symmetry_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    对角腿对称性惩罚 (Diagonal Gait Symmetry Penalty)
    强制 Trot 步态下对角腿（左前-右后，右前-左后）的足端高度保持一致。
    """
    # 获取机器人资产
    robot: Articulation = env.scene[asset_cfg.name]

    # ==========================================
    # 调试代码：检查匹配到的足端顺序（只打印一次）
    # 使用 hasattr 检查 env 对象，避免使用 global 变量
    # ==========================================
    if not hasattr(env, "_foot_order_printed"):
        # 根据 asset_cfg.body_ids 获取对应的刚体名称
        matched_foot_names = [robot.data.body_names[i] for i in asset_cfg.body_ids]
        print("\n" + "="*50)
        print(f"🚨 [DEBUG] 正则表达式 '.*_foot' 匹配到的足端顺序为:")
        for idx, name in enumerate(matched_foot_names):
            print(f"   索引 {idx}: {name}")
        print("="*50 + "\n")

        # 给 env 对象打上标记，这样下一帧就不会再打印了
        env._foot_order_printed = True
    # ==========================================


    # 获取由 asset_cfg 过滤出的足端刚体在世界坐标系下的 Z 轴高度
    # shape: (num_envs, num_feet) -> 通常 num_feet 为 4
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 【重要警告】这里假设你的 URDF/刚体解析顺序是标准的：
    # 0: FL (左前), 1: FR (右前), 2: RL (左后), 3: RR (右后)
    # 如果你的机器人按左右划分 (例如 FL, RL, FR, RR)，请务必修改下面的索引！

    # 计算对角腿的高度差
    diff_FL_RR = foot_z[:, 0] - foot_z[:, 3]  # 左前和右后
    diff_FR_RL = foot_z[:, 1] - foot_z[:, 2]  # 右前和左后

    # 返回惩罚值：高度差的平方和
    return torch.square(diff_FL_RR) + torch.square(diff_FR_RL)


def action_acceleration_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    惩罚动作的二阶导数 (动作加速度) 使用 L2 范数。
    计算公式: || (a_t - a_{t-1}) - (a_{t-1} - a_{t-2}) ||^2
    """
    # 1. 初始化缓存：如果 env 中还没有这个变量，就创建一个全零的 Tensor
    if not hasattr(env, "_prev_action_diff"):
        env._prev_action_diff = torch.zeros_like(env.action_manager.action)

    # 2. 处理环境重置 (Reset)：
    # 强化学习中环境会不断重置，重置的环境必须把历史记录清零，否则会产生错误的巨大惩罚
    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        env._prev_action_diff[reset_ids] = 0.0

    # 3. 计算当前的动作一阶导数 (a_t - a_{t-1})
    current_action_diff = env.action_manager.action - env.action_manager.prev_action

    # 4. 计算动作二阶导数 (当前一阶导 - 上一步一阶导)
    action_acc = current_action_diff - env._prev_action_diff

    # 5. 更新缓存，供下一个 Step 使用
    env._prev_action_diff = current_action_diff.clone()

    # 6. 返回 L2 范数 (对每个环境的动作加速度求平方和)
    return torch.sum(torch.square(action_acc), dim=1)


def joint_action_target_limit_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    safety_margin_fraction: float = 0.0,
    stage_start_update: int = 0,
    stage_ramp_updates: int = 1,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Penalize policy-mapped joint targets before the deployment limit clamp.

    Isaac Lab's joint action term clips the affine-mapped target before it is
    applied to the articulation.  Looking only at the simulated joint position
    therefore hides a policy that continuously asks for an impossible target.
    The 0707 real-robot bags exposed exactly that failure on the FL hip.  This
    term reconstructs the unclipped target from the action term's raw action,
    scale and offset, and records both normalized loss and physical violations.
    """
    terms = getattr(env.action_manager, "_terms", {})
    term = terms.get(action_name) if isinstance(terms, dict) else None
    if term is None:
        raise RuntimeError(f"Action term '{action_name}' is required for target-limit safety.")
    clip = getattr(term, "_clip", None)
    if not isinstance(clip, torch.Tensor) or clip.ndim != 3 or clip.shape[-1] != 2:
        raise RuntimeError(f"Action term '{action_name}' has no resolved finite target-limit tensor.")

    raw_actions = term.raw_actions
    target = raw_actions * term._scale + term._offset
    lower = clip[:, :, 0]
    upper = clip[:, :, 1]
    finite = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
    if not bool(torch.all(finite).item()):
        raise RuntimeError(f"Action term '{action_name}' contains non-finite or inverted target limits.")

    span = torch.clamp(upper - lower, min=1.0e-6)
    actual_violation = torch.maximum(lower - target, target - upper).clamp(min=0.0)
    margin_fraction = max(0.0, min(float(safety_margin_fraction), 0.49))
    safe_lower = lower + span * margin_fraction
    safe_upper = upper - span * margin_fraction
    margin_violation = torch.maximum(safe_lower - target, target - safe_upper).clamp(min=0.0)
    normalized_margin_violation = margin_violation / span

    actual_rate = torch.mean((actual_violation > 1.0e-6).float(), dim=1)
    actual_step = torch.any(actual_violation > 1.0e-6, dim=1).float()
    actual_max = torch.max(actual_violation, dim=1).values
    margin_rate = torch.mean((margin_violation > 1.0e-6).float(), dim=1)
    margin_max = torch.max(margin_violation, dim=1).values
    penalty = torch.mean(torch.square(normalized_margin_violation), dim=1)

    # Keep the physical-limit audit separate from the optional soft safety
    # margin.  A margin hit may be useful training pressure, but it must never
    # be reported as a real joint-limit violation.
    env._highstep_target_limit_violation_rate = actual_rate.detach()
    env._highstep_target_limit_max_delta = actual_max.detach()
    env._highstep_target_margin_violation_rate = margin_rate.detach()
    env._highstep_target_margin_max_delta = margin_max.detach()
    env._highstep_target_limit_penalty = penalty.detach()

    buffer_defaults = {
        "_highstep_target_limit_sample_steps": 0.0,
        "_highstep_target_limit_violation_sum": 0.0,
        "_highstep_target_limit_step_violation_sum": 0.0,
        "_highstep_target_limit_max_delta_episode": 0.0,
        "_highstep_target_limit_violation_streak": 0.0,
        "_highstep_target_limit_violation_max_consecutive": 0.0,
        "_highstep_target_margin_violation_sum": 0.0,
        "_highstep_target_margin_max_delta_episode": 0.0,
        "_highstep_target_limit_penalty_sum": 0.0,
    }
    for buffer_name, initial_value in buffer_defaults.items():
        if not hasattr(env, buffer_name):
            setattr(
                env,
                buffer_name,
                torch.full((env.num_envs,), initial_value, device=env.device, dtype=penalty.dtype),
            )

    env._highstep_target_limit_sample_steps += 1.0
    env._highstep_target_limit_violation_sum += actual_rate.detach()
    env._highstep_target_limit_step_violation_sum += actual_step.detach()
    env._highstep_target_limit_max_delta_episode = torch.maximum(
        env._highstep_target_limit_max_delta_episode,
        actual_max.detach(),
    )
    env._highstep_target_limit_violation_streak = torch.where(
        actual_step > 0.0,
        env._highstep_target_limit_violation_streak + 1.0,
        torch.zeros_like(env._highstep_target_limit_violation_streak),
    )
    env._highstep_target_limit_violation_max_consecutive = torch.maximum(
        env._highstep_target_limit_violation_max_consecutive,
        env._highstep_target_limit_violation_streak,
    )
    env._highstep_target_margin_violation_sum += margin_rate.detach()
    env._highstep_target_margin_max_delta_episode = torch.maximum(
        env._highstep_target_margin_max_delta_episode,
        margin_max.detach(),
    )
    env._highstep_target_limit_penalty_sum += penalty.detach()
    stage_gate = _highstep_training_progress_gate(
        env,
        stage_start_update=stage_start_update,
        stage_ramp_updates=stage_ramp_updates,
        num_steps_per_update=num_steps_per_update,
    )
    return penalty * stage_gate
