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
    env, std: float, target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "posture",
) -> torch.Tensor:
    # 原逻辑保留：指数核；仅按姿态自动选脚，flat 时不做高度奖励(返回0)
    asset: RigidObject = env.scene[asset_cfg.name]
    ids = _foot_ids(env, asset)
    state = _posture_from_command(env, command_name)  # [N]
    pos_w = asset.data.body_pos_w  # [N,B,3]，RigidObject 数据在世界系下组织:contentReference[oaicite:3]{index=3}

    def _mean_h(id_list):
        return (pos_w[:, id_list, 2].mean(dim=1) if id_list else torch.zeros(env.num_envs, device=env.device))

    hF, hB = _mean_h(ids["front"]), _mean_h(ids["back"])
    hL, hR = _mean_h(ids["left"]),  _mean_h(ids["right"])
    sel_h = torch.zeros(env.num_envs, device=env.device)
    sel_h = torch.where(state == 1, hF, sel_h)  # front
    sel_h = torch.where(state == 2, hB, sel_h)  # back
    sel_h = torch.where(state == 3, hL, sel_h)  # left
    sel_h = torch.where(state == 4, hR, sel_h)  # right
    # flat(0) → 不奖励抬脚：直接返回全0
    if (state == 0).any():
        # 对 flat 的那些 env，sel_h 仍为 0，下面算出来的误差是 (0 - target_height)^2；
        # 为保持“最小改动”且不影响其它状态，这里直接对 flat 掩码置零奖励。
        mask_flat = (state == 0).float()
    else:
        mask_flat = torch.zeros_like(sel_h)

    feet_height_error = torch.square(sel_h - target_height)
    rew = torch.exp(-feet_height_error / (std**2))
    return rew * (1.0 - mask_flat)

def handstand_feet_on_air(
    env, sensor_cfg: SceneEntityCfg, command_name: str = "posture"
) -> torch.Tensor:
    # 原逻辑保留：all(first_air)；仅按姿态自动选脚，flat 时返回 0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene["robot"]
    ids = _foot_ids(env, asset)
    first_air = contact_sensor.compute_first_air(env.step_dt)  # [N,B]:contentReference[oaicite:4]{index=4}
    state = _posture_from_command(env, command_name)

    def _all_air(id_list):
        return (first_air[:, id_list].all(dim=1).float() if id_list else torch.zeros(env.num_envs, device=env.device))

    aF, aB = _all_air(ids["front"]), _all_air(ids["back"])
    aL, aR = _all_air(ids["left"]),  _all_air(ids["right"])
    rew = torch.zeros(env.num_envs, device=env.device)
    rew = torch.where(state == 1, aF, rew)  # front
    rew = torch.where(state == 2, aB, rew)  # back
    rew = torch.where(state == 3, aL, rew)  # left
    rew = torch.where(state == 4, aR, rew)  # right
    # flat(0) → 0
    return rew

def handstand_feet_air_time(
    env, sensor_cfg: SceneEntityCfg, threshold: float, command_name: str = "posture"
) -> torch.Tensor:
    # 原逻辑保留：sum((last_air_time - threshold) * first_contact)；仅按姿态自动选脚，flat 时返回 0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene["robot"]
    ids = _foot_ids(env, asset)
    state = _posture_from_command(env, command_name)

    first_contact = contact_sensor.compute_first_contact(env.step_dt)  # [N,B]:contentReference[oaicite:5]{index=5}
    last_air_time = contact_sensor.data.last_air_time                  # [N,B]

    def _air_time(id_list):
        if not id_list:
            return torch.zeros(env.num_envs, device=env.device)
        return ((last_air_time[:, id_list] - threshold) * first_contact[:, id_list]).sum(dim=1)

    tF, tB = _air_time(ids["front"]), _air_time(ids["back"])
    tL, tR = _air_time(ids["left"]),  _air_time(ids["right"])
    rew = torch.zeros(env.num_envs, device=env.device)
    rew = torch.where(state == 1, tF, rew)  # front
    rew = torch.where(state == 2, tB, rew)  # back
    rew = torch.where(state == 3, tL, rew)  # left
    rew = torch.where(state == 4, tR, rew)  # right
    # flat(0) → 0
    return rew

def handstand_orientation_l2(
    env, target_gravity: list[float] | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "posture",
) -> torch.Tensor:
    # 原行为：若给了 target_gravity 就按原 L2 惩罚；否则根据姿态命令自动给目标向量
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # [N,3]：重力方向在 base 帧的投影（官方 API）:contentReference[oaicite:6]{index=6}
    if target_gravity is None:
        state = _posture_from_command(env, command_name)
        # five-state 的离散目标重力向量（base 帧）：
        # flat→[0,0,-1]；front→[-1,0,0]；back→[1,0,0]；left→[0,-1,0]；right→[0,1,0]
        T = torch.tensor([[0., 0., -1.],
                          [-1., 0., 0.],
                          [ 1., 0., 0.],
                          [ 0., -1.,0.],
                          [ 0., 1., 0.]], device=env.device, dtype=g_b.dtype)  # shape [5,3]
        tgt = T[state]  # [N,3]
    else:
        tgt = torch.tensor(target_gravity, device=env.device, dtype=g_b.dtype).expand_as(g_b)
    return torch.sum(torch.square(g_b - tgt), dim=1)
