"""Backflip-specific reward functions (NEW FILE)."""
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
# acrobatics.py
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import math


def track_pitch_velocity_exp(env, std: float, target: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Exponential reward for tracking target pitch angular velocity (NEW FUNCTION)."""
    asset = env.scene[asset_cfg.name]
    ang_vel_y = asset.data.root_ang_vel_b[:, 1]  # Pitch axis in base frame
    error = torch.abs(ang_vel_y - target)
    return torch.exp(-error / (std ** 2))


def flip_completion(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    current_quat = env.scene["robot"].data.root_link_state_w[:, 3:7]
    init_quat = env.scene["robot"].data.default_root_state[:, 3:7]
    quat_diff = quat_mul(current_quat, quat_conjugate(init_quat))
    angle = 2 * torch.atan2(torch.norm(quat_diff[:, 1:], dim=1), quat_diff[:, 0])
    return angle / (2 * math.pi)  # 0-1 normalized progress

def base_lin_vel_z(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Reward upward velocity exceeding threshold."""
    asset = env.scene["robot"]
    vel_z = asset.data.root_lin_vel_w[:, 2]  # World Z-axis
    return torch.clamp(vel_z - threshold, min=0.0)  # Now uses threshold correctly

def air_time(env):
    """
    奖励：若当前 Z 速度 > 2.0，则奖励 1 * dt；否则 0。
    """
    return (env.scene["robot"].data.root_lin_vel_w[:, 2] > 2.0).float() * env.step_dt

def clamped_flip_completion(env):
    """
    Clamp the result of flip_completion into [0.0, 1.0] as a bonus reward.
    """
    return torch.clamp(flip_completion(env, threshold=0.0), 0.0, 1.0)

def reverse_rotation(env):
    """
    Penalize negative pitch angular velocity.
    """
    return torch.clamp(env.scene["robot"].data.root_ang_vel_b[:, 1], max=0.0)

def low_jump_penalty(env, threshold: float = 2.0, decay_rate: float = 1.0, fall_penalty_scale: float = 2.0):
    """
    Penalize weak or insufficient jumps using exponential decay,
    and apply a stronger penalty for downward motion.
    
    Parameters:
    - threshold: minimum z-velocity to be considered a 'good' jump
    - decay_rate: controls how fast the penalty increases as velocity drops
    - fall_penalty_scale: additional penalty multiplier for negative z-velocity (falling)
    """
    v_z = env.scene["robot"].data.root_lin_vel_w[:, 2]
    
    # Base penalty for insufficient jump velocity
    jump_penalty = torch.where(
        v_z >= threshold,
        torch.zeros_like(v_z),                          # No penalty if jump is strong enough
        torch.exp(-decay_rate * v_z)                    # Penalize low jump velocity
    )

    # Additional penalty for falling (negative v_z)
    fall_penalty = torch.where(
        v_z < 0.0,
        fall_penalty_scale * torch.abs(v_z),            # Linearly punish falling fast
        torch.zeros_like(v_z)
    )

    return jump_penalty + fall_penalty


def early_landing(env):
    """
    Penalize low altitude (<1.5 m).
    """
    return (env.scene["robot"].data.root_pos_w[:, 2] < 1.5).float()

def excessive_roll_penalty(env, threshold=0.5):
    """
    Penalize large roll angles computed from quaternion.
    Threshold in radians.
    """
    q = env.scene["robot"].data.root_quat_w  # shape: (N, 4), [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Compute roll angle from quaternion
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Penalize large absolute roll
    penalty = torch.clamp(torch.abs(roll) - threshold, min=0.0)
    return penalty

def pitch_angle_reward(env, target_angle=math.pi*2, tolerance=0.2):
    """鼓励 pitch 完成接近 360° 的旋转"""
    q = env.scene["robot"].data.root_quat_w  # (N, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # 计算 pitch 角度
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))  # [-pi/2, pi/2]，这里可能需改用其他方式
    
    error = torch.abs(pitch - target_angle)
    return torch.exp(-error / tolerance)

