# Copyright (c) 2025 Tianyang TANG
# SPDX-License-Identifier: Apache-2.0

"""Symmetry augmentation for SiriusW (12 leg joints + 4 wheel joints).

Joint order (Isaac Lab DOF indexing):
[
 'LF_HAA','LH_HAA','RF_HAA','RH_HAA',
 'LF_HFE','LH_HFE','RF_HFE','RH_HFE',
 'LF_KFE','LH_KFE','RF_KFE','RH_KFE',
 'LF_WHEEL','LH_WHEEL','RF_WHEEL','RH_WHEEL'
]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING
import torch
from tensordict import TensorDict, TensorDictBase  # 新增
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states_siriusw", "ObsLayout", "SiriusWSymmetrySigns"]


# -------------------------
# Configurable slices/signs
# -------------------------

@dataclass
class ObsLayout:
    ang_vel: slice      = field(default_factory=lambda: slice(0, 3))
    proj_gravity: slice = field(default_factory=lambda: slice(3, 6))
    vel_cmd: slice      = field(default_factory=lambda: slice(6, 9))
    leg_pos: slice      = field(default_factory=lambda: slice(9, 21))
    joint_vel: slice    = field(default_factory=lambda: slice(21, 37))  
    # last_actions: place at the END of obs with length = actions_dim (robust for variable dims)
    # If your layout differs, override in call: ObsLayout(...)

def _ensure_2d(x: torch.Tensor | None):
    if x is None:
        return None
    return x.unsqueeze(0) if x.dim() == 1 else x
def _infer_layout_for_key(key: str, D: int) -> ObsLayout:
    """根据 obs 键名与维度推断切片布局。"""
    layout = ObsLayout()
    if key == "policy":
        # policy 可能是 42(旧) 或 54(新，含16维 joint_vel)
        layout.ang_vel = slice(0, 3)
        layout.proj_gravity = slice(3, 6)
        layout.vel_cmd = slice(6, 9)
        layout.leg_pos = slice(9, 21)
        # joint_vel：从 21 开始直到 actions 之前
        # 新版 policy 通常是 [3,3,3,12,16,16,1] -> 54
        # 老版可能是 [3,3,3,12,4,16,1] -> 42
        # 这里根据总长 D 反推 joint_vel 长度
        # 倒数第1维是 height_scan(1)，倒数第2块是 actions(16)
        jv_len = (D - 1 - 16) - 21
        layout.joint_vel = slice(21, 21 + jv_len)  # 4 或 16
    else:  # "critic"
        # critic 一般固定 57 = [3 lin]+[3 ang]+[3 g]+[3 cmd]+12+16+16+1
        layout.ang_vel = slice(3, 6)
        layout.proj_gravity = slice(6, 9)
        layout.vel_cmd = slice(9, 12)
        layout.leg_pos = slice(12, 24)
        layout.joint_vel = slice(24, 40)
    return layout
def _tile_along_batch(x: torch.Tensor, times: int) -> torch.Tensor:
    """把任意形状张量在 batch 维(0维)重复 times 次。"""
    reps = [times] + [1] * (x.ndim - 1)
    return x.repeat(*reps)
@dataclass
class SiriusWSymmetrySigns:
    # Whether to flip wheel sign under mirrors (depends on your USD joint axis definitions)
    wheel_lr_sign: float = +1.0   # left-right mirror: usually +1
    wheel_fb_sign: float = -1.0   # front-back mirror: often -1 (forward↔backward)


# -------------------------
# Public API
# -------------------------

# -------------------------
# 全对称增强 左右/前后/对角
# -------------------------
@torch.no_grad()
def compute_symmetric_states_siriusw(
    env,
    obs=None,                # 现在允许 TensorDict / Tensor / None
    actions: torch.Tensor | None = None,
    obs_type: str | None = None,     # 兼容你之前的签名，可不使用
    signs: SiriusWSymmetrySigns = SiriusWSymmetrySigns(),
):
    # ----------------------------
    # 1) 处理 Observation (可为 TensorDict)
    # ----------------------------
    if isinstance(obs, TensorDictBase):
        # 逐键增强：通常包含 'policy' 与 'critic'
        keys = list(obs.keys())
        # 取 batch 大小
        n = obs.batch_size[0]
        out = {}

        for k in keys:
            val = obs[k]        # shape: [n, D]（观测向量）
            if val.ndim == 1:
                val = val.unsqueeze(0)
            if val.ndim != 2:
                # 非扁平观测（例如网格/图像），这里简单在 batch 维复制4次
                out[k] = _tile_along_batch(val, 4)
                continue

            D = val.shape[1]
            # 确定布局
            layout = _infer_layout_for_key(k, D)
            # 生成四种对称
            lr   = _obs_left_right_siriusw(env.unwrapped, val, k, layout, signs)
            fb   = _obs_front_back_siriusw(env.unwrapped, val, k, layout, signs)
            diag = _obs_front_back_siriusw(env.unwrapped, lr,  k, layout, signs)
            out[k] = torch.cat([val, lr, fb, diag], dim=0)

        obs_aug = TensorDict(out, batch_size=[n * 4])

    else:
        # 兼容：obs 是 Tensor（老管线/测试用）
        if obs is not None:
            x = obs
            if x.dim() == 1:
                x = x.unsqueeze(0)
            n, D = x.shape
            # 若未明确 obs_type，则按长度猜
            if obs_type is None:
                obs_type = "critic" if D >= 57 else "policy"
            layout = _infer_layout_for_key(obs_type, D)
            lr   = _obs_left_right_siriusw(env.unwrapped, x,  obs_type, layout, signs)
            fb   = _obs_front_back_siriusw(env.unwrapped, x,  obs_type, layout, signs)
            diag = _obs_front_back_siriusw(env.unwrapped, lr, obs_type, layout, signs)
            obs_aug = torch.cat([x, lr, fb, diag], dim=0)
        else:
            obs_aug = None

    # ----------------------------
    # 2) 处理 Action（仍为 Tensor）
    # ----------------------------
    if actions is not None:
        a = actions
        if a.dim() == 1:
            a = a.unsqueeze(0)
        n, A = a.shape
        if A != 16:
            raise AssertionError(f"SiriusW expects action_dim=16 (12腿+4轮)，当前 {A}")
        a_lr   = _actions_left_right_siriusw(a, signs)
        a_fb   = _actions_front_back_siriusw(a, signs)
        a_diag = _actions_front_back_siriusw(a_lr, signs)
        act_aug = torch.cat([a, a_lr, a_fb, a_diag], dim=0)
    else:
        act_aug = None

    return obs_aug, act_aug

# 注意：_obs_left_right_auto / _obs_front_back_auto 内部使用
# jv_len = layout.joint_vel.stop - layout.joint_vel.start
# 当 jv_len==4 时只交换四个轮速；否则按 16 维（12腿+4轮）处理
# -------------------------
# 左右对称增强
# -------------------------
# @torch.no_grad()
# def compute_symmetric_states_siriusw(
#     env,
#     obs=None,                      # TensorDict / Tensor / None
#     actions: torch.Tensor | None = None,
#     obs_type: str | None = None,
#     signs: SiriusWSymmetrySigns = SiriusWSymmetrySigns(),
# ):
#     # ---------- OBS ----------
#     if isinstance(obs, TensorDictBase):
#         keys = list(obs.keys())
#         n = obs.batch_size[0]
#         out = {}
#         for k in keys:
#             val = obs[k]
#             if val.ndim == 1:
#                 val = val.unsqueeze(0)
#             if val.ndim == 2:
#                 D = val.shape[1]
#                 layout = _infer_layout_for_key(k, D)
#                 lr = _obs_left_right_siriusw(env.unwrapped, val, k, layout, signs)
#                 out[k] = torch.cat([val, lr], dim=0)                 # ← 只拼接 原/LR
#             else:
#                 # 非扁平观测（如高度网格）暂时直接复制，后续可在此处做“列翻转”
#                 out[k] = torch.cat([val, val.clone()], dim=0)
#         obs_aug = TensorDict(out, batch_size=[n * 2])                 # ← n*2
#     else:
#         if obs is not None:
#             x = obs.unsqueeze(0) if obs.dim() == 1 else obs
#             n, D = x.shape
#             if obs_type is None:
#                 obs_type = "critic" if D >= 57 else "policy"
#             layout = _infer_layout_for_key(obs_type, D)
#             lr = _obs_left_right_siriusw(env.unwrapped, x, obs_type, layout, signs)
#             obs_aug = torch.cat([x, lr], dim=0)                       # ← 只 原/LR
#         else:
#             obs_aug = None

#     # ---------- ACTIONS ----------
#     if actions is not None:
#         a = actions.unsqueeze(0) if actions.dim() == 1 else actions
#         n, A = a.shape
#         assert A == 16, f"SiriusW expects action_dim=16, got {A}"
#         a_lr = _actions_left_right_siriusw(a, signs)
#         act_aug = torch.cat([a, a_lr], dim=0)                         # ← 只 原/LR
#     else:
#         act_aug = None

#     return obs_aug, act_aug
# -------------------------
# Observation transforms
# -------------------------

def _obs_left_right_siriusw(
    env: ManagerBasedRLEnv,
    obs: torch.Tensor,
    obs_type: str,
    layout: ObsLayout,
    signs: SiriusWSymmetrySigns,
) -> torch.Tensor:
    x = obs.clone()
    dev = x.device

    # ang vel [wx, wy, wz] → LR: [-wx, +wy, -wz]
    x[:, layout.ang_vel] *= torch.tensor([-1.0, 1.0, -1.0], device=dev)

    # projected gravity [gx, gy, gz] → LR: [+gx, -gy, +gz]
    x[:, layout.proj_gravity] *= torch.tensor([+1.0, -1.0, +1.0], device=dev)

    # velocity command [vx, vy, wz] → LR: [+vx, -vy, -wz]
    x[:, layout.vel_cmd] *= torch.tensor([+1.0, -1.0, -1.0], device=dev)

    # leg joint positions (12) : swap L↔R, flip HAA sign
    x[:, layout.leg_pos] = _switch_leg12_left_right(x[:, layout.leg_pos], flip_haa=True)

    # joint velocities (16 = 12 legs + 4 wheels): swap L↔R, flip leg HAA; wheel sign per config
    x[:, layout.joint_vel] = _switch_all16_left_right(
        x[:, layout.joint_vel],
        wheel_sign=signs.wheel_lr_sign,
        flip_leg_haa=True,
    )

    # last actions: assume at the end and length==16
    last_actions = x.shape[1] - 16
    x[:, last_actions:] = _switch_all16_left_right(
        x[:, last_actions:],
        wheel_sign=signs.wheel_lr_sign,
        flip_leg_haa=True,
    )

    # Height-scan等网格型观测若存在：LR 可在其内部做列翻转（此处留给你的 obs 管线）
    return x


def _obs_front_back_siriusw(
    env: ManagerBasedRLEnv,
    obs: torch.Tensor,
    obs_type: str,
    layout: ObsLayout,
    signs: SiriusWSymmetrySigns,
) -> torch.Tensor:
    x = obs.clone()
    dev = x.device

    # ang vel [wx, wy, wz] → FB: [+wx, -wy, -wz]
    x[:, layout.ang_vel] *= torch.tensor([+1.0, -1.0, -1.0], device=dev)

    # projected gravity [gx, gy, gz] → FB: [-gx, +gy, +gz]
    x[:, layout.proj_gravity] *= torch.tensor([-1.0, +1.0, +1.0], device=dev)

    # velocity command [vx, vy, wz] → FB: [-vx, +vy, -wz]
    x[:, layout.vel_cmd] *= torch.tensor([-1.0, +1.0, -1.0], device=dev)

    # leg joint positions (12) : swap F↔H, flip HFE/KFE sign
    x[:, layout.leg_pos] = _switch_leg12_front_back(x[:, layout.leg_pos], flip_hfe_kfe=True)

    # joint velocities (16 = 12 legs + 4 wheels): swap F↔H, flip leg HFE/KFE; wheel sign per config
    x[:, layout.joint_vel] = _switch_all16_front_back(
        x[:, layout.joint_vel],
        wheel_sign=signs.wheel_fb_sign,
        flip_leg_hfe_kfe=True,
    )

    # last actions: assume at the end and length==16
    last_actions = x.shape[1] - 16
    x[:, last_actions:] = _switch_all16_front_back(
        x[:, last_actions:],
        wheel_sign=signs.wheel_fb_sign,
        flip_leg_hfe_kfe=True,
    )

    # Height-scan等网格型观测若存在：FB 可在其内部做行翻转
    return x


# -------------------------
# Action transforms
# -------------------------

def _actions_left_right_siriusw(actions: torch.Tensor, signs: SiriusWSymmetrySigns) -> torch.Tensor:
    a = actions.clone()
    a[:] = _switch_all16_left_right(a[:], wheel_sign=signs.wheel_lr_sign, flip_leg_haa=True)
    return a


def _actions_front_back_siriusw(actions: torch.Tensor, signs: SiriusWSymmetrySigns) -> torch.Tensor:
    a = actions.clone()
    a[:] = _switch_all16_front_back(a[:], wheel_sign=signs.wheel_fb_sign, flip_leg_hfe_kfe=True)
    return a


# -------------------------
# Helpers: index maps & swaps
# -------------------------

# Indices in 16-dim (12 legs + 4 wheels)
IDX_HAA = [0, 1, 2, 3]
IDX_HFE = [4, 5, 6, 7]
IDX_KFE = [8, 9, 10, 11]
IDX_WHL = [12, 13, 14, 15]

def _switch_leg12_left_right(j: torch.Tensor, flip_haa: bool = True) -> torch.Tensor:
    """LR swap on 12-dim leg-only vectors (pos/vel)."""
    out = torch.zeros_like(j)
    # HAA
    out[..., [0, 1]]  = j[..., [2, 3]]   # LF_HAA <- RF_HAA, LH_HAA <- RH_HAA
    out[..., [2, 3]]  = j[..., [0, 1]]   # RF_HAA <- LF_HAA, RH_HAA <- LH_HAA
    # HFE
    out[..., [4, 5]]  = j[..., [6, 7]]
    out[..., [6, 7]]  = j[..., [4, 5]]
    # KFE
    out[..., [8, 9]]  = j[..., [10, 11]]
    out[..., [10,11]] = j[..., [8, 9]]
    if flip_haa:
        out[..., IDX_HAA] *= -1.0
    return out

def _switch_leg12_front_back(j: torch.Tensor, flip_hfe_kfe: bool = True) -> torch.Tensor:
    """FB swap on 12-dim leg-only vectors (pos/vel)."""
    out = torch.zeros_like(j)
    # HAA: (LF<->LH), (RF<->RH)
    out[..., [0, 2]]  = j[..., [1, 3]]   # LF<-LH, RF<-RH
    out[..., [1, 3]]  = j[..., [0, 2]]   # LH<-LF, RH<-RF
    # HFE
    out[..., [4, 6]]  = j[..., [5, 7]]
    out[..., [5, 7]]  = j[..., [4, 6]]
    # KFE
    out[..., [8,10]]  = j[..., [9,11]]
    out[..., [9,11]]  = j[..., [8,10]]
    if flip_hfe_kfe:
        out[..., IDX_HFE + IDX_KFE] *= -1.0
    return out

def _switch_all16_left_right(j: torch.Tensor, wheel_sign: float = +1.0, flip_leg_haa: bool = True) -> torch.Tensor:
    """LR swap on 16-dim vectors (12 legs + 4 wheels)."""
    out = torch.zeros_like(j)
    # legs
    out[..., :12] = _switch_leg12_left_right(j[..., :12], flip_haa=flip_leg_haa)
    # wheels: (LF_WHEEL <-> RF_WHEEL), (LH_WHEEL <-> RH_WHEEL)
    out[..., [12, 13]] = j[..., [14, 15]]
    out[..., [14, 15]] = j[..., [12, 13]]
    # wheel sign (model-dependent)
    if wheel_sign != 1.0:
        out[..., IDX_WHL] *= wheel_sign
    return out

def _switch_all16_front_back(j: torch.Tensor, wheel_sign: float = -1.0, flip_leg_hfe_kfe: bool = True) -> torch.Tensor:
    """FB swap on 16-dim vectors (12 legs + 4 wheels)."""
    out = torch.zeros_like(j)
    # legs
    out[..., :12] = _switch_leg12_front_back(j[..., :12], flip_hfe_kfe=flip_leg_hfe_kfe)
    # wheels: (LF_WHEEL <-> LH_WHEEL), (RF_WHEEL <-> RH_WHEEL)
    out[..., [12, 14]] = j[..., [13, 15]]   # front<-hind per side
    out[..., [13, 15]] = j[..., [12, 14]]   # hind<-front per side
    # wheel sign (model-dependent; often -1 for FB)
    if wheel_sign != 1.0:
        out[..., IDX_WHL] *= wheel_sign
    return out
