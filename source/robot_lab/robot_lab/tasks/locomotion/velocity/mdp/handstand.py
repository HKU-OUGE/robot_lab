# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils 
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
def _get_gait(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "gait_mode"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.gait_mode

def _get_gait_mask(env: ManagerBasedRLEnv, gait_mode: int) -> torch.Tensor:
    g = _get_gait(env).squeeze(-1)
    # gait_mode 1: Biped/Stand, gait_mode 0: Quadruped
    return g if gait_mode == 1 else (1.0 - g)
# def handstand_feet_height_exp(
#     env: ManagerBasedRLEnv,
#     std: float,
#     target_height: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
#     feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
#     return torch.exp(-feet_height_error / std**2)


# def handstand_feet_on_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
#     reward = torch.all(first_air, dim=1).float()
#     return reward


# def handstand_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     return reward


# def handstand_orientation_l2(
#     env: ManagerBasedRLEnv, target_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     # Define the target gravity direction for an upright posture in the base frame
#     target_gravity_tensor = torch.tensor(target_gravity, device=env.device)
#     # Penalize deviation of the projected gravity vector from the target
#     return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity_tensor), dim=1)
def _posture_from_command(env, command_name: str = "posture"):
    """从姿态命令的四元数判定当前目标姿态: flat / front / back / left / right。
    做法：将世界重力 [0,0,-1] 旋回到“命令姿态”的 base 帧，取主导轴与符号分类。"""
    cmd = env.command_manager.get_command(command_name)  # [N, 7] = [x,y,z, qw,qx,qy,qz]
    quat_w = cmd[:, 3:7]  # (w,x,y,z) 约定见官方文档:contentReference[oaicite:2]{index=2}
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    g_b_tgt = math_utils.quat_apply_inverse(quat_w, g_w)  # R(q)^T * g_w
    ax = torch.argmax(g_b_tgt.abs(), dim=1)                # 0:x 1:y 2:z
    sgn = torch.sign(g_b_tgt.gather(1, ax.unsqueeze(1))).squeeze(1)
    # 规则：x主导→front/back；y主导→left/right；z主导→flat（把倒立也合并到 flat）
    # （如需区分 upside，可在 ax==2 且 sgn>0 时返回 "upside"）
    states = torch.full((env.num_envs,), fill_value=4, dtype=torch.long, device=env.device)  # 先置为 right=4
    states = torch.where((ax == 2), torch.full_like(states, 0), states)                      # flat=0
    states = torch.where((ax == 0) & (sgn < 0), torch.full_like(states, 1), states)          # front=1
    states = torch.where((ax == 0) & (sgn > 0), torch.full_like(states, 2), states)          # back =2
    states = torch.where((ax == 1) & (sgn < 0), torch.full_like(states, 3), states)          # left =3
    # right=4 已默认
    return states  # [N]，0:flat 1:front 2:back 3:left 4:right

def _foot_ids(env: "ManagerBasedRLEnv", asset: RigidObject):
    """缓存四类脚的 body 索引，避免每步正则检索。"""
    if not hasattr(env, "_handstand_foot_ids"):
        fF, _ = asset.find_bodies(".*F_FOOT_link")
        fH, _ = asset.find_bodies(".*H_FOOT_link")
        fL, _ = asset.find_bodies("L.*_FOOT_link")
        fR, _ = asset.find_bodies("R.*_FOOT_link")
        env._handstand_foot_ids = {"front": fF, "back": fH, "left": fL, "right": fR}
    return env._handstand_foot_ids  # dict[str, list[int]]

def handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
    return torch.exp(-feet_height_error / std**2)

def handstand_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward


def handstand_orientation_l2(
    env: ManagerBasedRLEnv, target_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Define the target gravity direction for an upright posture in the base frame
    target_gravity_tensor = torch.tensor(target_gravity, device=env.device)
    # Penalize deviation of the projected gravity vector from the target
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity_tensor), dim=1)


def orientation_align_gravity_gated(
    env: ManagerBasedRLEnv, 
    target_gravity: list[float], 
    gait_mode: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize deviation from a target gravity vector in the base frame.
    
    Args:
        target_gravity: Expected gravity vector in base frame (e.g., [-1.0, 0.0, 0.0] for standing upright).
    """
    # 1. 获取当前重力在基座下的投影
    asset = env.scene[asset_cfg.name]
    current_gravity_b = asset.data.projected_gravity_b
    
    # 2. 创建目标向量 (Batch size 扩展)
    target_g_b = torch.tensor(target_gravity, device=env.device).repeat(current_gravity_b.shape[0], 1)
    
    # 3. 计算欧氏距离平方误差 ||g_curr - g_tgt||^2
    error = torch.sum(torch.square(current_gravity_b - target_g_b), dim=1)
    
    # 4. 门控输出
    return error * _get_gait_mask(env, gait_mode)

def track_lin_vel_xy_heading_gated(
    env: ManagerBasedRLEnv, std: float, command_name: str, gait_mode: int
) -> torch.Tensor:
    """
    追踪航向坐标系（Heading Frame）下的 XY 线速度。
    适用于机身 Pitch/Roll 发生剧烈变化的机器人（如轮足站立）。
    """
    # 1. 获取指令 (vx, vy)
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    
    # 2. 获取世界坐标系下的线速度
    # 注意：这里用 root_lin_vel_w 而不是 _b
    lin_vel_w = env.scene["robot"].data.root_lin_vel_w[:, :3]
    
    # 3. 获取机器人的 Yaw 方向（排除 Pitch 和 Roll 的干扰）
    root_quat = env.scene["robot"].data.root_quat_w
    heading_quat = yaw_quat(root_quat) # 只提取 Yaw 分量的四元数
    
    # 4. 将世界速度旋转到航向坐标系 (Heading Frame)
    # 相当于把世界速度投影到机器人当前的“前后左右”平面
    lin_vel_heading = quat_apply_inverse(heading_quat, lin_vel_w)[:, :2]
    
    # 5. 计算误差
    error = torch.sum(torch.square(vel_cmd - lin_vel_heading), dim=1)
    
    # 6.通过门控应用奖励
    return torch.exp(-error / (std**2)) * _get_gait_mask(env, gait_mode)

def track_ang_vel_z_world_gated(
    env: ManagerBasedRLEnv, std: float, command_name: str, gait_mode: int
) -> torch.Tensor:
    """
    追踪世界坐标系 Z 轴的角速度。
    确保无论机身姿态如何，指令都是控制“绕着垂直轴转向”。
    """
    # 1. 获取指令 (yaw_rate)
    cmd_ang_vel_z = env.command_manager.get_command(command_name)[:, 2]
    
    # 2. 获取世界坐标系下的角速度 Z 分量
    # 无论机身怎么歪，我们只关心它绕地轴的旋转
    current_ang_vel_z = env.scene["robot"].data.root_ang_vel_w[:, 2]
    
    error = torch.square(cmd_ang_vel_z - current_ang_vel_z)
    
    return torch.exp(-error / (std**2)) * _get_gait_mask(env, gait_mode)


def undesired_contacts_gated(
    env: ManagerBasedRLEnv,
    threshold: float,
    gait_mode: int,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """
    惩罚特定模式下的接触。
    返回的是接触部位的【总数量】，而不是简单的 0/1。
    """
    # 1. 获取传感器数据
    asset = env.scene.sensors[sensor_cfg.name]
    
    # 2. 获取接触力大小 (Batch, History, Bodies)
    contact_forces = asset.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    forces_norm = torch.norm(contact_forces, dim=-1)
    
    # 3. 取历史最大值 (Batch, Bodies)
    max_forces = torch.max(forces_norm, dim=1)[0]
    
    # 4. 关键点：使用 sum() 计算有多少个部位接触了，而不是 max() 判断是否有接触
    # (max_forces > threshold) 得到一个 bool 矩阵 (Batch, Bodies)
    # .sum(dim=1) 得到每个环境的接触总数 (Batch,)
    contact_count = (max_forces > threshold).float().sum(dim=1)
    
    # 5. 门控输出
    return contact_count * _get_gait_mask(env, gait_mode)

def track_lin_vel_xy_heading_posture_gated(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gait_mode: int,
    target_gravity: list[float] = [-1.0, 0.0, 0.0], # 默认机身竖直向上
    gravity_threshold: float = 0.5 # 允许一定的倾斜
) -> torch.Tensor:
    """
    追踪航向坐标系下的 XY 线速度，但仅当机器人姿态接近目标重力方向时才给予奖励。
    """
    # 1. 基础速度追踪计算 (同原函数)
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_w = env.scene["robot"].data.root_lin_vel_w[:, :3]
    root_quat = env.scene["robot"].data.root_quat_w
    heading_quat = yaw_quat(root_quat)
    lin_vel_heading = quat_apply_inverse(heading_quat, lin_vel_w)[:, :2]
    error = torch.sum(torch.square(vel_cmd - lin_vel_heading), dim=1)
    speed_reward = torch.exp(-error / (std**2))
    
    # 2. [新增] 姿态门控计算
    # 获取当前重力在基座下的投影
    current_gravity_b = env.scene["robot"].data.projected_gravity_b
    target_g_b = torch.tensor(target_gravity, device=env.device).view(1, 3)
    
    # 计算当前重力与目标重力的点积 (Cosine Similarity)
    # 如果完全重合，dot = 1.0; 如果垂直，dot = 0.0; 如果反向，dot = -1.0
    # 我们希望 dot 接近 1.0
    gravity_dot = torch.sum(current_gravity_b * target_g_b, dim=1)
    
    # 创建一个软掩码 (Soft Mask)
    # 当 dot > threshold 时，mask 接近 1；当 dot < threshold 时，mask 迅速衰减到 0
    # 这里使用 ReLU 简单截断，或者使用 Sigmoid
    # 简单做法：线性衰减。如果 dot < 0.7 (约45度倾斜)，则奖励开始打折
    posture_mask = torch.clamp((gravity_dot - gravity_threshold) / (1.0 - gravity_threshold), 0.0, 1.0)
    
    # 3. 组合奖励
    return speed_reward * posture_mask * _get_gait_mask(env, gait_mode)

def track_ang_vel_z_world_posture_gated(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gait_mode: int,
    target_gravity: list[float] = [-1.0, 0.0, 0.0],
    gravity_threshold: float = 0.5
) -> torch.Tensor:
    """
    追踪世界 Z 角速度，带姿态门控。
    """
    # 1. 基础角速度追踪
    cmd_ang_vel_z = env.command_manager.get_command(command_name)[:, 2]
    current_ang_vel_z = env.scene["robot"].data.root_ang_vel_w[:, 2]
    error = torch.square(cmd_ang_vel_z - current_ang_vel_z)
    ang_reward = torch.exp(-error / (std**2))
    
    # 2. [新增] 姿态门控 (逻辑同上)
    current_gravity_b = env.scene["robot"].data.projected_gravity_b
    target_g_b = torch.tensor(target_gravity, device=env.device).view(1, 3)
    gravity_dot = torch.sum(current_gravity_b * target_g_b, dim=1)
    
    posture_mask = torch.clamp((gravity_dot - gravity_threshold) / (1.0 - gravity_threshold), 0.0, 1.0)
    
    return ang_reward * posture_mask * _get_gait_mask(env, gait_mode)

def joint_pos_target_l1_posture_gated(
    env: ManagerBasedRLEnv, 
    target_pos_list: list[float], 
    gait_mode: int,
    target_gravity: list[float] = [-1.0, 0.0, 0.0],
    gravity_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize deviation from a specific target joint position list, 
    BUT only when the robot's posture is close to the target gravity vector.
    
    This allows the robot to use arbitrary joint configurations to stand up 
    without being penalized, enforcing the pose only when upright.
    """
    # 1. 获取当前关节位置
    asset = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # 2. 目标关节 Tensor 处理
    target_tensor = torch.tensor(target_pos_list, device=env.device, dtype=torch.float32)
    target_tensor = target_tensor.view(1, -1).repeat(env.num_envs, 1)
    
    # 3. 计算 L1 误差
    deviation = torch.sum(torch.abs(current_joint_pos - target_tensor), dim=1)
    
    # 4. [新增] 姿态门控计算
    # 获取重力投影
    current_gravity_b = asset.data.projected_gravity_b
    target_g_b = torch.tensor(target_gravity, device=env.device).view(1, 3)
    
    # 计算相似度 (Dot Product)
    gravity_dot = torch.sum(current_gravity_b * target_g_b, dim=1)
    
    # [Modify] Harder Gating Logic
    # 之前的线性衰减可能导致机器人故意歪一点来逃避惩罚。
    # 这里我们使用更陡峭的曲线：
    # 只要 dot > threshold，mask 就迅速接近 1。
    # 例如使用 Sigmoid 变体，或者简单的 power 函数让它更陡。
    
    # 归一化到 [0, 1] 区间 (x: 0 when dot=threshold, 1 when dot=1)
    normalized_dot = torch.clamp((gravity_dot - gravity_threshold) / (1.0 - gravity_threshold), 0.0, 1.0)
    
    # 使用 power(0.1) 让曲线在接近 0 的时候迅速上升到 1 (即：只要有一点点直，就全额惩罚)
    # 这样机器人就无法通过“微倾斜”来获得显著的惩罚减免了。
    posture_mask = torch.pow(normalized_dot, 0.1)
    
    # 5. 综合门控输出
    return deviation * posture_mask * _get_gait_mask(env, gait_mode)


def joint_pos_target_l1_posture_robust(
    env: ManagerBasedRLEnv, 
    target_pos_list: list[float], 
    gait_mode: int,
    target_gravity: list[float] = [-1.0, 0.0, 0.0],
    gravity_threshold: float = 0.5,
    base_penalty_scale: float = 0.2, # 新增：基础惩罚比例
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize deviation from target joint positions.
    
    Improvements:
    - Adds a 'base_penalty_scale': Even if the robot is leaning (mask is low), 
      it still receives partial penalty (e.g. 20%). This prevents it from 
      intentionally leaning to completely avoid joint penalties.
    """
    # 1. Get Joint Positions
    asset = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # 2. Target Tensor
    target_tensor = torch.tensor(target_pos_list, device=env.device, dtype=torch.float32)
    target_tensor = target_tensor.view(1, -1).repeat(env.num_envs, 1)
    
    # 3. Calculate L1 Deviation (Average per joint is easier to tune than Sum)
    # Using mean makes the weight independent of the number of joints
    deviation = torch.mean(torch.abs(current_joint_pos - target_tensor), dim=1)
    
    # 4. Posture Gating
    current_gravity_b = asset.data.projected_gravity_b
    target_g_b = torch.tensor(target_gravity, device=env.device).view(1, 3)
    
    # Dot product: 1.0 is aligned, < 0.8 is tilted
    gravity_dot = torch.sum(current_gravity_b * target_g_b, dim=1)
    
    # Calculate Mask: 0.0 (tilted) to 1.0 (upright)
    # Using a smoother transition than hard clamp might help gradients
    normalized_dot = torch.clamp((gravity_dot - gravity_threshold) / (1.0 - gravity_threshold), 0.0, 1.0)
    posture_mask = torch.pow(normalized_dot, 0.1) # Aggressive mask
    
    # 5. Combined Logic [CRITICAL FIX]
    # Old logic: deviation * posture_mask
    # New logic: deviation * (base_scale + (1 - base_scale) * posture_mask)
    # Meaning: If posture_mask is 0 (leaning), we still apply 'base_scale' (e.g., 0.2) of the penalty.
    # If posture_mask is 1 (upright), we apply 1.0 * penalty.
    effective_scale = base_penalty_scale + (1.0 - base_penalty_scale) * posture_mask
    
    # 6. Apply Gait Mask
    gait_mask = _get_gait_mask(env, gait_mode)
    
    return deviation * effective_scale * gait_mask
# ==============================================================================
# [New] Task Space Rewards
# ==============================================================================

def feet_height_exp_gated(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg,
    gait_mode: int
) -> torch.Tensor:
    """
    Reward specific feet (e.g. hands/front legs) to reach target height.
    Uses exponential kernel: exp(-error^2 / std^2)
    """
    asset = env.scene[asset_cfg.name]
    # body_pos_w: [num_envs, num_bodies, 3]
    # 获取指定body的Z坐标 (Index 2)
    feet_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # 计算每个脚与目标高度的误差平方
    error_sq = torch.square(feet_heights - target_height)
    
    # 对所有指定的脚求平均误差，或者求和
    # 这里我们希望所有指定的脚都达标，所以取 mean
    mean_error_sq = torch.mean(error_sq, dim=1)
    
    reward = torch.exp(-mean_error_sq / (std**2))
    return reward * _get_gait_mask(env, gait_mode)

# ==============================================================================
# [New] Regularization for Stability
# ==============================================================================

def joint_vel_penalty_gated(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    gait_mode: int
) -> torch.Tensor:
    """Penalize joint velocities in specific gait mode."""
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=1) * _get_gait_mask(env, gait_mode)

def action_rate_l2_gated(
    env: ManagerBasedRLEnv,
    gait_mode: int
) -> torch.Tensor:
    """Penalize rate of change of actions (smoothness) in specific gait mode."""
    # 简单的 action diff
    diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(diff), dim=1) * _get_gait_mask(env, gait_mode)

def track_rear_wheel_velocity_exp_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    track_width: float,
    std: float,
    gait_mode: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Track desired wheel velocities for Biped mode based on base velocity commands.
    This bypasses Base Frame confusion by controlling wheels directly.
    """
    # 1. Get Commands (v_x: linear, w_z: yaw rate)
    cmd = env.command_manager.get_command(command_name)
    v_x_cmd = cmd[:, 0]
    w_z_cmd = cmd[:, 2]

    # 2. Calculate Ideal Wheel Angular Velocities (Differential Drive Kinematics)
    # left_vel = (v - w * width / 2) / r
    # right_vel = (v + w * width / 2) / r
    target_wheel_vel_L = (v_x_cmd - w_z_cmd * (track_width / 2)) / wheel_radius
    target_wheel_vel_R = (v_x_cmd + w_z_cmd * (track_width / 2)) / wheel_radius

    # 3. Get Actual Wheel Velocities
    # Assuming the asset_cfg.joint_names are passed as ["LH_WHEEL", "RH_WHEEL"]
    asset = env.scene[asset_cfg.name]
    current_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    
    # 4. Calculate Error
    # current_wheel_vel[:, 0] is Left, [:, 1] is Right (Ensure order in config matches)
    error_L = torch.square(target_wheel_vel_L - current_wheel_vel[:, 0])
    error_R = torch.square(target_wheel_vel_R - current_wheel_vel[:, 1])
    
    total_error = error_L + error_R

    # 5. Apply Gait Mask
    return torch.exp(-total_error / (std**2)) * _get_gait_mask(env, gait_mode)
