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
import torch.nn.functional as F

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

def height_scan_disc(
    env: ManagerBasedEnv,
    obs_cache: dict | None = None,          # 设默认值让其成为可选参数
    sensor_cfg: SceneEntityCfg | None = None,
    offset: float = 0.5,
) -> torch.Tensor:
    # 调用原始高度扫描:contentReference[oaicite:0]{index=0}
    heights = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    depth = (-heights).clamp(min=-1.0)
    bins = torch.floor(depth * 10.0) / 10.0
    return bins.clamp(max=1.0)

def obstacle_scan_disc(
    env: ManagerBasedEnv,
    obs_cache: dict | None = None,          # 可选：缓存同一步的height_scan结果
    sensor_cfg: SceneEntityCfg | None = None,
    offset: float = 0.5,
) -> torch.Tensor:
    """
    基于 mdp.height_scan 生成离散化障碍占据图：
    1) depth = (-heights).clamp(min=0)   # 正数表示台阶/障碍相对基准面向上凸起的“高度差”
    2) bins  = floor(depth*10)/10        # 以 0.1 m 为步长离散
    3) occ   = 0/1 二值图（bins<0.1→0，bins>=0.1→1）
    返回形状与 height_scan 相同，dtype 与其一致。
    """
    # 1) 取得 height_scan，可用缓存复用
    if obs_cache is not None and "height_scan" in obs_cache:
        heights = obs_cache["height_scan"]
    else:
        heights = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
        if obs_cache is not None:
            obs_cache["height_scan"] = heights

    # 2) 计算离散深度（0.1m 分箱）
    depth = (-heights).clamp(min=0.0)
    bins = torch.floor(depth * 10.0) / 10.0

    # 3) 二值化：bins < 0.1 -> 0,  bins >= 0.1 -> 1
    occ = torch.where(bins >= 0.1,
                      torch.ones_like(bins),
                      torch.zeros_like(bins))
    return occ

def terrain_level_obs(
    env: ManagerBasedRLEnv,
    normalize: bool = True,
    one_hot: bool = False,
    num_levels: int | None = None,
) -> torch.Tensor:
    """
    返回每个子环境当前所处的地形等级（terrain level），用于作为观测输入。

    Args:
        env: Isaac Lab 的 ManagerBasedRLEnv。
        normalize: 若为 True，则把 level 归一化到 [0, 1]（按行数 num_rows 线性缩放）。
        one_hot: 若为 True，则返回 one-hot 向量；否则返回标量（列向量）。
        num_levels: 指定 one-hot 的维度。若不指定，将优先用 terrain_generator.num_rows 推断，
                    否则用 max(level)+1 推断。

    Returns:
        torch.Tensor:
            - one_hot=False: 形状 [N, 1] 的 float 张量。
            - one_hot=True : 形状 [N, num_levels] 的 float 张量。
    """
    # 1) 读取 terrain 对象与当前各 env 的 level
    terrain = env.scene.terrain
    if not hasattr(terrain, "terrain_levels") or terrain.terrain_levels is None:
        # 兼容非 generator 地形或未启用课程学习时的情况
        levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    else:
        levels = terrain.terrain_levels.to(device=env.device)  # [N]，整型等级索引

    # 2) one-hot 输出
    if one_hot:
        if num_levels is None:
            # 优先从 terrain_generator.rows 推断（行数即等级数）
            num_rows = getattr(getattr(terrain.cfg, "terrain_generator", object()), "num_rows", None)
            if num_rows is not None:
                num_levels = int(num_rows)
            else:
                # 回退：按当前 batch 的最大 level 推断
                num_levels = int(levels.max().item()) + 1 if levels.numel() > 0 else 1
        levels_clamped = levels.clamp(min=0, max=max(num_levels - 1, 0))
        return F.one_hot(levels_clamped, num_classes=num_levels).float()

    # 3) 标量输出（可归一化）
    levels_f = levels.float()
    if normalize:
        # 归一化到 [0,1]：按行数 num_rows − 1 缩放（level 从 0..num_rows-1 线性增长）
        num_rows = getattr(getattr(terrain.cfg, "terrain_generator", object()), "num_rows", None)
        denom = float(max((num_rows - 1) if num_rows else (levels.max().item()), 1.0))
        out = (levels_f / denom).clamp(0.0, 1.0)
    else:
        out = levels_f
    return out.unsqueeze(-1)  # [N, 1]

def his_height_scan_disc(
    env: ManagerBasedEnv,
    obs_cache: dict | None = None,          # 可选，不依赖
    sensor_cfg: SceneEntityCfg | None = None,
    offset: float = 0.5,
    bin_size: float = 0.1,                  # 离散步长（0.1 → 10 档）
    max_val: float = 1.0,                   # 上限夹紧到 1.0
    use_depth: bool = True,                 # 与你原实现一致：默认把“负高度”当坑深
) -> torch.Tensor:
    """
    返回每个 env 在“本回合截至目前扫描到过的最高离散高度/深度”的标量（[N,1]，范围 0~1）。
    - 累计方式：running max（按回合），非逐帧覆盖
    - 离散方式：floor 到 bin_size 的整数档，再 clamp 到 1.0
    """
    device = env.device

    # 1) 本帧扫描（形状通常是 [N, K]）
    heights = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)  # 与你原调用一致
    vals = (-heights).clamp(min=0.0) if use_depth else heights.clamp(min=0.0)  # 深度/高度二选一

    # 2) 离散到 [0,1] 档位（0.0, 0.1, ..., 1.0）
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    bins = torch.floor(vals / bin_size) * bin_size
    bins = bins.clamp(max=max_val)

    # 3) 取本帧“网格最大值”（每个 env 一个数）
    #   兼容任意后续维度：统一展平成 [N, -1] 再 amax
    bins_max_now = bins.reshape(env.num_envs, -1).amax(dim=1)  # [N]

    # 4) 按回合累计历史最大（状态保存在 env 上）
    if (not hasattr(env, "_height_scan_disc_max")) or (env._height_scan_disc_max is None) \
       or (env._height_scan_disc_max.shape[0] != env.num_envs):
        env._height_scan_disc_max = torch.zeros(env.num_envs, dtype=torch.float32, device=device)

    env._height_scan_disc_max = torch.maximum(env._height_scan_disc_max, bins_max_now)

    # 5) 返回 [N,1]（给策略/critic当标量观测）
    out = env._height_scan_disc_max.clamp(0.0, max_val).unsqueeze(-1)  # [N,1]
    return out

def clear_height_scan_disc(env: ManagerBasedEnv, env_ids):
    dev = env.device
    ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
    if (not hasattr(env, "_height_scan_disc_max")) or (env._height_scan_disc_max is None) \
        or (env._height_scan_disc_max.shape[0] != env.num_envs):
        env._height_scan_disc_max = torch.zeros(env.num_envs, dtype=torch.float32, device=dev)
    else:
        env._height_scan_disc_max[ids] = 0.0
