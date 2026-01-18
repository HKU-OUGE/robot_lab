# Copyright (c) 2024-2025 Tianyang TANG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# Helper: 获取 Gait Mode
# ==============================================================================

def _get_gait(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "gait_mode"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.gait_mode


def _get_gait_mask(env: ManagerBasedRLEnv, gait_mode: int) -> torch.Tensor:
    g = _get_gait(env).squeeze(-1)
    # gait_mode 1: Biped/Stand, gait_mode 0: Quadruped
    return g if gait_mode == 1 else (1.0 - g)
def _get_gait_mask_bool(env: ManagerBasedRLEnv, gait_mode: int) -> torch.Tensor:
    """Returns a boolean mask (True/False) for the specified gait mode."""
    # Reuse the float mask logic: > 0.5 ensures 1.0 becomes True, 0.0 becomes False
    return _get_gait_mask(env, gait_mode) > 0.5

# ==============================================================================
# 1. Quadruped Termination (Gait 0)
# ==============================================================================
def bad_orientation_quadruped(
    env: ManagerBasedRLEnv, limit_roll: float = 1.0, limit_pitch: float = 1.6
) -> torch.Tensor:
    """
    Terminates if roll > 1.0 or pitch > 1.6 (approx 90 deg).
    Only active when gait_mode == 0 (Quadruped).
    """
    # 获取 Base Quaternion
    quat = env.scene["robot"].data.root_quat_w

    # 计算 Roll, Pitch (absolute value)
    roll, pitch, _ = euler_xyz_from_quat(quat)

    # Check limits
    is_bad = (torch.abs(roll) > limit_roll) | (torch.abs(pitch) > limit_pitch)

    # 只在 Quad 模式下生效 (g < 0.5)
    g = _get_gait(env).squeeze(-1)
    is_quad = g < 0.5

    return is_bad & is_quad


# ==============================================================================
# 2. Biped Termination (Gait 1)
# ==============================================================================
def illegal_contact_biped(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    grace_period_s: float = 1.0,
) -> torch.Tensor:
    """
    Terminates if unauthorized links contact the ground.
    Active: gait_mode == 1 (Biped) AND time > 1.0s
    """
    # 1. Check Gait
    g = _get_gait(env).squeeze(-1)
    is_biped = g > 0.5

    # 2. Check Time (Grace Period)
    # episode_length_buf 是步数，乘以 dt 得到秒数
    time_elapsed = env.episode_length_buf * env.step_dt
    is_after_grace = time_elapsed > grace_period_s

    # 3. Check Contacts
    # 获取 Contact Sensor 数据
    # 注意：sensor_cfg 必须正确指向 contact_forces 传感器，并且 body_names 包含了所有非法部位
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # net_forces_w_history: [num_envs, history_len, num_bodies, 3]
    # 我们只看最近一帧 (index 0 if history is latest-first? IsaacLab history usually stores latest at 0 or -1 depending on impl)
    # 通常 sensor.data.net_forces_w 是当前帧 [num_envs, num_bodies, 3]

    # 使用 net_forces_w (当前帧)
    forces = contact_sensor.data.net_forces_w

    # 我们只关心 sensor_cfg.body_ids 指定的那些 body
    # SceneEntityCfg 会解析 body_names 并提供 body_ids (在整个 robot body list 中的索引)
    # 但 ContactSensor 可能会过滤 body。如果 ContactSensor 也是基于同样的 body list，那索引是一致的。
    # 这里我们假设 sensor_cfg 直接用于筛选 ContactSensor 的输出是不够的，我们需要手动筛选。

    # 更安全的方法：ContactSensor 应该在 EnvCfg 里被配置为监测 Robot 的所有 Bodies。
    # 然后这里的 sensor_cfg.body_ids 告诉我们哪些 body 是非法的。

    # 提取非法 body 的受力
    # sensor_cfg.body_ids 是相对于 robot 的索引。
    # 如果 contact sensor 也是监测整个 robot，那么索引是对齐的。
    illegal_forces = forces[:, sensor_cfg.body_ids, :]

    # 计算力的大小
    force_magnitudes = torch.norm(
        illegal_forces, dim=-1
    )  # [num_envs, num_illegal_bodies]
    max_force = torch.max(force_magnitudes, dim=1)[0]  # [num_envs]

    has_contact = max_force > threshold

    return is_biped & is_after_grace & has_contact


def terminate_gate_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    gait_mode: int,
) -> torch.Tensor:
    """Terminate the episode when the robot experiences a collision force, optionally filtered by gait mode.

    Args:
        env: The RL environment manager.
        threshold: The force threshold to trigger termination.
        sensor_cfg: Configuration for the contact sensor (specifies which body parts to monitor).
        gait_command_name: (Optional) The name of the command term storing the gait mode (e.g., "gait").
                           If None, gait checking is disabled.
        trigger_gait_ids: (Optional) A list of gait IDs (integers) or a single ID.
                          Termination will ONLY occur if the current gait command matches one of these IDs.
                          If None, termination triggers regardless of gait mode.
    """
    # 1. Extract the contact sensor defined in the configuration
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 2. Get history of net contact forces
    # Shape: (num_envs, history_len, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # 3. Extract specific bodies if specified in sensor_cfg.body_ids
    forces_on_bodies = net_contact_forces[:, :, sensor_cfg.body_ids]

    # 4. Calculate Force Magnitude (Norm)
    # Shape: (num_envs, history_len, num_selected_bodies)
    contact_magnitudes = torch.norm(forces_on_bodies, dim=-1)

    # 5. Check Maximum force over the history window (dim=1)
    max_force_in_history = torch.max(contact_magnitudes, dim=1)[0]

    # 6. Basic Contact Check
    # Result Shape: (num_envs,)
    has_contact = torch.any(max_force_in_history > threshold, dim=1)
        # Only terminate if contact happened AND we are in one of the trigger gait modes
    return has_contact & _get_gait_mask_bool(env, gait_mode)

def bad_orientation_gated(
    env: ManagerBasedRLEnv, 
    limit_roll: float, 
    limit_pitch: float, 
    gait_mode: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terminates if roll or pitch exceeds limits, ONLY when in the specified gait_mode.
    
    Args:
        env: The environment manager.
        limit_roll: Absolute roll threshold (in radians).
        limit_pitch: Absolute pitch threshold (in radians).
        gait_mode: The gait mode ID to activate this check (0 for Quad, 1 for Biped).
        asset_cfg: The robot asset configuration (default: "robot").
    """
    # 1. 获取机器人的姿态四元数
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w

    # 2. 计算欧拉角 (Roll, Pitch, Yaw)
    roll, pitch, _ = euler_xyz_from_quat(quat)

    # 3. 检查是否超过阈值 (取绝对值)
    is_bad_orientation = (torch.abs(roll) > limit_roll) | (torch.abs(pitch) > limit_pitch)

    # 4. 获取 Gait Mask (检查当前是否处于指定的 gait_mode)
    is_in_mode = _get_gait_mask_bool(env, gait_mode)

    # 5. 只有在指定模式下 且 姿态错误 时才终止
    return is_bad_orientation & is_in_mode
