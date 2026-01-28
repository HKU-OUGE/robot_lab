# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations
import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.6 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)

POSE_EPS = 1e-3
def _mk_ranges(roll, pitch):
    return UniformPoseCommandCfg.Ranges(
        pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0),
        roll=(roll-POSE_EPS, roll+POSE_EPS),
        pitch=(pitch-POSE_EPS, pitch+POSE_EPS),
        yaw=(-math.pi, math.pi),
    )
    

def switch_posture(env, env_ids, old_ranges, *, switch_time_s=10.0, next_state="front"):
    t = env.common_step_counter * env.step_dt  # 全局步计数 * dt
    if t < switch_time_s:
        # 初始保持“平”
        return mdp.modify_term_cfg.NO_CHANGE
    # 10s 之后切到目标状态（返回单一 Ranges；不要用 batch ）
    presets = {
        "flat":  (0.0, 0.0),
        "front": (0.0, +0.5*math.pi),
        "back":  (0.0, -0.5*math.pi),
        "left":  (-0.5*math.pi, 0.0),
        "right": (+0.5*math.pi, 0.0),
    }
    roll, pitch = presets[next_state]
    return _mk_ranges(roll, pitch)
