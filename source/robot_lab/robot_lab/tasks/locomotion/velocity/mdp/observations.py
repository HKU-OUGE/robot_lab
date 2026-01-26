# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def height_scan_disc(env, obs_cache=None, sensor_cfg=None, offset=0.5) -> torch.Tensor:
    heights = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    depth = (-heights).clamp(-2.0, 2.0)                 # 先夹到 [-2, 2]
    bins  = torch.round(depth * 10.0) / 10.0            # 对称量化到 0.1 网格
    return bins                                         # 已保证在 [-1, 1]

def obstacle_distance(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """计算 RayCaster 射线的击中距离 (Euclidean Distance)。"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 1. 获取射线源头位置 (World Frame)
    # sensor.data.pos_w 形状通常是 (num_envs, 1, 3) 或 (num_envs, 3)
    source_pos = sensor.data.pos_w
    
    # 2. 获取击中点位置 (World Frame)
    ray_hits = sensor.data.ray_hits_w # (num_envs, num_rays, 3)
    
    # 3. 计算距离 norm(hit - source)
    # 注意维度广播: (num_envs, num_rays, 3) - (num_envs, 1, 3)
    if source_pos.dim() == 2:
        source_pos = source_pos.unsqueeze(1)
        
    dist = torch.norm(ray_hits - source_pos, dim=-1)
    
    # 4. 处理未击中 (Inf/NaN/MaxRange)
    # Isaac Lab 通常会将未击中的点设为极远值，clip 会在 Observation 中处理
    # 如果需要手动处理：
    # dist = torch.where(dist > 5.0, torch.tensor(5.0, device=dist.device), dist)
    
    return dist