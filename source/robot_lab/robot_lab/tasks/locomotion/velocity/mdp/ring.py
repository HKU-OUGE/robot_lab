# Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

# 兼容你项目里常用的 mdp 导入方式
try:
    import robot_lab.tasks.locomotion.velocity.mdp as mdp
except Exception:
    from isaaclab.envs import mdp  # 提供 height_scan 等mdp函数

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
# ---------------------------
# 高度门控：把 height_scanner 的相对高度 -> [0,1] gate 系数
# ---------------------------
def _height_obstacle_gate(
    env: ManagerBasedRLEnv,
    height_sensor_cfg: SceneEntityCfg,
    *,
    t_low: float = 0.10,     # 低障阈值（≤10cm 强烈抑制）
    t_start: float = 0.25,   # 开始使用奖励的阈值（≥25cm）
    t_full: float = 0.50,    # 完全打开奖励（≥50cm）
    low_scale: float = 0.05, # “强烈抑制”时的残余比例（>0 保持可导）
    offset: float = 0.5,
    aggregate: Literal["max", "mean", "p95"] = "max",
) -> torch.Tensor:
    """
    返回形状 [num_envs] 的 gate 系数 ∈ [low_scale, 1]。
    使用 mdp.height_scan 读取相对高度矩阵 [N, R]，对射线聚合得到“前方最高/平均/分位高度”再分段平滑映射。
    """
    heights = mdp.height_scan(env, sensor_cfg=height_sensor_cfg, offset=offset)  # [N, R]
    heights = torch.nan_to_num(heights, nan=-1e6)  # 避免未命中导致 NaN

    if aggregate == "max":
        h = torch.max(heights, dim=1).values
    elif aggregate == "mean":
        h = torch.mean(heights, dim=1)
    else:  # p95
        h, _ = torch.sort(heights, dim=1)
        idx = torch.clamp((0.95 * (h.shape[1] - 1)).long(), min=0)
        h = h.gather(1, idx.unsqueeze(1)).squeeze(1)

    # 分段（t_low -> t_start -> t_full）两段 smoothstep（C^1 连续）
    def smoothstep(x, a, b):
        x = torch.clamp((x - a) / (b - a + 1e-8), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    # 第一段：从 low_scale 过渡到 0.5
    s1 = smoothstep(h, t_low, t_start)
    gate1 = low_scale + (0.5 - low_scale) * s1
    # 第二段：从 0.5 过渡到 1.0
    s2 = smoothstep(h, t_start, t_full)
    gate2 = 0.5 + 0.5 * s2

    # 组合（当 h < t_start 用 gate1，>= t_start 用 gate2）
    gate = torch.where(h < t_start, gate1, gate2)
    # h < t_low 强制为 low_scale；h >= t_full 强制为 1
    gate = torch.where(h <= t_low, torch.full_like(gate, low_scale), gate)
    gate = torch.where(h >= t_full, torch.ones_like(gate), gate)
    return gate


# ---------------------------
# 门控版奖励：原奖励 * gate
# ---------------------------
def gated_handstand_orientation_l2(
    env: ManagerBasedRLEnv,
    target_gravity: list[float],
    height_sensor_cfg: SceneEntityCfg,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    t_low: float = 0.10, t_start: float = 0.25, t_full: float = 0.50, low_scale: float = 0.05,
    offset: float = 0.5, aggregate: str = "max",
) -> torch.Tensor:
    base = mdp.handstand_orientation_l2(env, target_gravity=target_gravity, asset_cfg=asset_cfg)
    gate = _height_obstacle_gate(
        env, height_sensor_cfg, t_low=t_low, t_start=t_start, t_full=t_full,
        low_scale=low_scale, offset=offset, aggregate=aggregate,
    )
    return base * gate


def gated_handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    height_sensor_cfg: SceneEntityCfg,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    t_low: float = 0.10, t_start: float = 0.25, t_full: float = 0.50, low_scale: float = 0.05,
    offset: float = 0.5, aggregate: str = "max",
) -> torch.Tensor:
    base = mdp.handstand_feet_height_exp(env, std=std, target_height=target_height, asset_cfg=asset_cfg)
    gate = _height_obstacle_gate(
        env, height_sensor_cfg, t_low=t_low, t_start=t_start, t_full=t_full,
        low_scale=low_scale, offset=offset, aggregate=aggregate,
    )
    return base * gate


def gated_handstand_feet_on_air(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    *,
    t_low: float = 0.10, t_start: float = 0.25, t_full: float = 0.50, low_scale: float = 0.05,
    offset: float = 0.5, aggregate: str = "max",
) -> torch.Tensor:
    base = mdp.handstand_feet_on_air(env, sensor_cfg=sensor_cfg)
    gate = _height_obstacle_gate(
        env, height_sensor_cfg, t_low=t_low, t_start=t_start, t_full=t_full,
        low_scale=low_scale, offset=offset, aggregate=aggregate,
    )
    return base * gate


def gated_handstand_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    *,
    threshold: float = 0.15,
    t_low: float = 0.10, t_start: float = 0.25, t_full: float = 0.50, low_scale: float = 0.05,
    offset: float = 0.5, aggregate: str = "max",
) -> torch.Tensor:
    base = mdp.handstand_feet_air_time(env, sensor_cfg=sensor_cfg, threshold=threshold)
    gate = _height_obstacle_gate(
        env, height_sensor_cfg, t_low=t_low, t_start=t_start, t_full=t_full,
        low_scale=low_scale, offset=offset, aggregate=aggregate,
    )
    return base * gate