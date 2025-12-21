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

# ==============================================================================
# 1. Quadruped Termination (Gait 0)
# ==============================================================================
def bad_orientation_quadruped(env: ManagerBasedRLEnv, limit_roll: float = 1.0, limit_pitch: float = 1.6) -> torch.Tensor:
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
    is_quad = (g < 0.5)
    
    return is_bad & is_quad

# ==============================================================================
# 2. Biped Termination (Gait 1)
# ==============================================================================
def illegal_contact_biped(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float = 1.0, 
    grace_period_s: float = 1.0
) -> torch.Tensor:
    """
    Terminates if unauthorized links contact the ground.
    Active: gait_mode == 1 (Biped) AND time > 1.0s
    """
    # 1. Check Gait
    g = _get_gait(env).squeeze(-1)
    is_biped = (g > 0.5)
    
    # 2. Check Time (Grace Period)
    # episode_length_buf 是步数，乘以 dt 得到秒数
    time_elapsed = env.episode_length_buf * env.step_dt
    is_after_grace = (time_elapsed > grace_period_s)
    
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
    force_magnitudes = torch.norm(illegal_forces, dim=-1) # [num_envs, num_illegal_bodies]
    max_force = torch.max(force_magnitudes, dim=1)[0] # [num_envs]
    
    has_contact = (max_force > threshold)
    
    return is_biped & is_after_grace & has_contact