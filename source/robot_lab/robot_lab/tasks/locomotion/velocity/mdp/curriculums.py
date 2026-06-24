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
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return env ids as a tensor on the environment device."""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _episode_reward_score(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    reward_term_name: str,
    default_score: float = 0.0,
    default_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Return the per-env episode-averaged weighted reward and its configured weight."""
    episode_sums = getattr(env.reward_manager, "_episode_sums", {})
    if reward_term_name not in episode_sums:
        return (
            torch.full((len(env_ids),), default_score, device=env.device),
            default_weight,
        )
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    score = episode_sums[reward_term_name][env_ids] / env.max_episode_length_s
    return score, reward_term_cfg.weight


def _velocity_tracking_error(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-env absolute x, y and yaw command tracking errors."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_error = torch.abs(command[env_ids, :2] - asset.data.root_lin_vel_b[env_ids, :2])
    yaw_error = torch.abs(command[env_ids, 2] - asset.data.root_ang_vel_b[env_ids, 2])
    return lin_error[:, 0], lin_error[:, 1], yaw_error


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


def terrain_levels_student_no_prior(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy_exp",
    orientation_term_name: str = "flat_orientation_l2",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_level: int = 0,
    initial_max_level: int = 4,
    max_level: int = 8,
    warmup_episodes: int = 2,
    unlock_every_episodes: int = 5,
    promote_after_successes: int = 2,
    demote_after_failures: int = 1,
    promote_reward_threshold: float = 0.55,
    demote_reward_threshold: float = 0.25,
    orientation_min_score: float = -1.2,
    max_x_vel_error: float = 0.55,
    max_y_vel_error: float = 0.35,
    max_yaw_vel_error: float = 0.40,
) -> torch.Tensor:
    """Terrain curriculum for no-prior student distillation.

    The stock velocity terrain curriculum promotes/demotes from travel distance.
    That is a poor fit here because the no-prior student must first learn the
    teacher's side-step posture, hip abduction and box-joint difference on
    static or slow lateral-step scenes. This curriculum instead advances when
    episodes survive to timeout with acceptable tracking/orientation, and
    demotes only on instability or clearly poor episode quality.
    """
    ids = _resolve_env_ids(env, env_ids)
    terrain = env.scene.terrain
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is None or len(ids) == 0:
        return torch.tensor(0.0, device=env.device)

    if not hasattr(env, "_student_np_success_streak"):
        env._student_np_success_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        env._student_np_failure_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    elapsed_episodes = env.common_step_counter // env.max_episode_length
    unlocked = max(0, elapsed_episodes - warmup_episodes) // max(1, unlock_every_episodes)
    allowed_max_level = min(max_level, initial_max_level + unlocked)

    # The first reset happens before any policy rollout. Keep the low/mid
    # initialization and only report state.
    if env.common_step_counter == 0:
        terrain_levels[ids] = torch.clamp(terrain_levels[ids], min=min_level, max=initial_max_level)
        terrain.env_origins[ids] = terrain.terrain_origins[terrain_levels[ids], terrain.terrain_types[ids]]
        return torch.mean(terrain_levels.float())

    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return torch.mean(terrain_levels.float())

    tracking_score, tracking_weight = _episode_reward_score(env, ids, reward_term_name, default_score=0.0)
    tracking_good = tracking_score > promote_reward_threshold * tracking_weight
    tracking_bad = tracking_score < demote_reward_threshold * tracking_weight

    orientation_score, _ = _episode_reward_score(env, ids, orientation_term_name, default_score=0.0, default_weight=-1.0)
    orientation_good = orientation_score > orientation_min_score
    x_error, y_error, yaw_error = _velocity_tracking_error(env, ids, command_name, asset_cfg)
    velocity_good = (x_error < max_x_vel_error) & (y_error < max_y_vel_error) & (yaw_error < max_yaw_vel_error)

    timed_out = env.reset_time_outs[ids]
    terminated = env.reset_terminated[ids]
    success = timed_out & (~terminated) & tracking_good & orientation_good & velocity_good
    failure = terminated | (timed_out & (~orientation_good | tracking_bad))

    env._student_np_success_streak[ids] = torch.where(
        success,
        env._student_np_success_streak[ids] + 1,
        torch.zeros_like(env._student_np_success_streak[ids]),
    )
    env._student_np_failure_streak[ids] = torch.where(
        failure,
        env._student_np_failure_streak[ids] + 1,
        torch.zeros_like(env._student_np_failure_streak[ids]),
    )

    levels = terrain_levels[ids]
    move_up = (
        (env._student_np_success_streak[ids] >= promote_after_successes)
        & (levels < allowed_max_level)
    )
    if env.common_step_counter < warmup_episodes * env.max_episode_length:
        move_up = torch.zeros_like(move_up)
    move_down = (env._student_np_failure_streak[ids] >= demote_after_failures) & (levels > min_level)
    move_down &= ~move_up

    terrain.update_env_origins(ids, move_up, move_down)
    moved = move_up | move_down
    env._student_np_success_streak[ids[moved]] = 0
    env._student_np_failure_streak[ids[moved]] = 0

    return torch.mean(terrain.terrain_levels.float())


def command_levels_vel_student_no_prior(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy_exp",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    range_multiplier: Sequence[float] = (0.5, 1.0),
    terrain_gate_levels: Sequence[float] = (4.5, 6.0, 7.0),
    command_multipliers: Sequence[float] = (0.5, 0.65, 0.8, 1.0),
    delta: float = 0.05,
    reward_threshold: float = 0.55,
    success_rate_threshold: float = 0.70,
    max_x_vel_error: float = 0.45,
    max_y_vel_error: float = 0.25,
    max_yaw_vel_error: float = 0.35,
    allow_decrease: bool = True,
) -> torch.Tensor:
    """Command curriculum coupled to the no-prior student terrain curriculum."""
    ids = _resolve_env_ids(env, env_ids)
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    if env.common_step_counter == 0 or not hasattr(env, "_student_np_original_vel_x"):
        env._student_np_original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._student_np_original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._student_np_final_vel_x = env._student_np_original_vel_x * range_multiplier[1]
        env._student_np_final_vel_y = env._student_np_original_vel_y * range_multiplier[1]
        env._student_np_command_multiplier = torch.tensor(float(range_multiplier[0]), device=env.device)
        env._student_np_last_command_update_step = -env.max_episode_length

    current_multiplier = env._student_np_command_multiplier
    base_velocity_ranges.lin_vel_x = (env._student_np_original_vel_x * current_multiplier).tolist()
    base_velocity_ranges.lin_vel_y = (env._student_np_original_vel_y * current_multiplier).tolist()

    if env.common_step_counter == 0 or not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)

    if env.common_step_counter - env._student_np_last_command_update_step >= env.max_episode_length:
        env._student_np_last_command_update_step = env.common_step_counter

        terrain_levels = getattr(env.scene.terrain, "terrain_levels", None)
        mean_terrain_level = (
            torch.mean(terrain_levels.float()) if terrain_levels is not None else torch.tensor(0.0, device=env.device)
        )

        target_multiplier = torch.tensor(float(command_multipliers[0]), device=env.device)
        for gate, multiplier in zip(terrain_gate_levels, command_multipliers[1:]):
            if mean_terrain_level.item() >= gate:
                target_multiplier = torch.tensor(float(multiplier), device=env.device)

        tracking_score, tracking_weight = _episode_reward_score(env, ids, reward_term_name, default_score=0.0)
        tracking_good = torch.mean(tracking_score).item() > reward_threshold * tracking_weight
        x_error, y_error, yaw_error = _velocity_tracking_error(env, ids, command_name, asset_cfg)
        velocity_good = (
            torch.mean(x_error).item() < max_x_vel_error
            and torch.mean(y_error).item() < max_y_vel_error
            and torch.mean(yaw_error).item() < max_yaw_vel_error
        )
        success_rate = (
            torch.mean((env.reset_time_outs[ids] & (~env.reset_terminated[ids])).float())
            if len(ids) > 0
            else torch.tensor(0.0, device=env.device)
        )
        stable = tracking_good and velocity_good and success_rate.item() >= success_rate_threshold

        if stable and current_multiplier.item() < target_multiplier.item():
            current_multiplier = torch.minimum(current_multiplier + delta, target_multiplier)
        elif allow_decrease and (not stable) and current_multiplier.item() > command_multipliers[0]:
            current_multiplier = torch.maximum(
                current_multiplier - delta,
                torch.tensor(float(command_multipliers[0]), device=env.device),
            )

        current_multiplier = torch.clamp(current_multiplier, min=float(command_multipliers[0]), max=float(range_multiplier[1]))
        env._student_np_command_multiplier = current_multiplier
        base_velocity_ranges.lin_vel_x = (env._student_np_original_vel_x * current_multiplier).tolist()
        base_velocity_ranges.lin_vel_y = (env._student_np_original_vel_y * current_multiplier).tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def highstep_rear_branch_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    min_commit_steps: int = 3,
) -> dict[str, torch.Tensor]:
    """Log which rear foot leads during high-step climbing without changing rewards."""
    ids = _resolve_env_ids(env, env_ids)
    zero = torch.tensor(0.0, device=env.device)
    if len(ids) == 0 or not hasattr(env, "_highstep_rear_branch_lead"):
        return {
            "valid_rate": zero,
            "rl_first_rate": zero,
            "rr_first_rate": zero,
            "both_first_rate": zero,
            "rl_minus_rr_first_rate": zero,
            "one_sided_stall_ratio": zero,
            "second_clear_rate": zero,
        }

    lead = env._highstep_rear_branch_lead[ids]
    commit_steps = env._highstep_rear_branch_commit_steps[ids]
    one_sided_steps = env._highstep_rear_branch_one_sided_steps[ids]
    second_clear_steps = env._highstep_rear_branch_second_clear_steps[ids]

    valid = (commit_steps >= float(min_commit_steps)) & (lead >= 0)
    valid_count = torch.clamp(valid.float().sum(), min=1.0)
    rl_first_rate = ((lead == 0) & valid).float().sum() / valid_count
    rr_first_rate = ((lead == 1) & valid).float().sum() / valid_count
    both_first_rate = ((lead == 2) & valid).float().sum() / valid_count
    valid_rate = valid.float().mean()
    one_sided_stall_ratio = one_sided_steps.sum() / torch.clamp(commit_steps.sum(), min=1.0)
    second_clear_rate = ((second_clear_steps > 0.0) & valid).float().sum() / valid_count

    # Reset only the envs that are ending now; the next episode should start fresh.
    env._highstep_rear_branch_lead[ids] = -1
    env._highstep_rear_branch_commit_steps[ids] = 0.0
    env._highstep_rear_branch_one_sided_steps[ids] = 0.0
    env._highstep_rear_branch_second_clear_steps[ids] = 0.0

    return {
        "valid_rate": valid_rate,
        "rl_first_rate": rl_first_rate,
        "rr_first_rate": rr_first_rate,
        "both_first_rate": both_first_rate,
        "rl_minus_rr_first_rate": rl_first_rate - rr_first_rate,
        "one_sided_stall_ratio": one_sided_stall_ratio,
        "second_clear_rate": second_clear_rate,
    }


def terrain_levels_vel_highstep(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_distance: float = 3.2,
    move_down_command_factor: float = 0.35,
    move_down_min_distance: float = 0.4,
    move_down_max_distance: float = 1.8,
    min_height_gain: float = 0.0,
    height_gain_required_level: int = 0,
    nominal_base_height: float = 0.44,
    climb_up_distance: float = 0.85,
    climb_height_gain: float = 0.08,
    climb_height_required_level: int = 3,
    climb_hold_distance: float = 0.55,
    climb_hold_height_gain: float = 0.04,
    stage_update_thresholds: Sequence[int] | None = None,
    stage_max_levels: Sequence[int] | None = None,
    num_steps_per_update: int = 24,
) -> torch.Tensor:
    """Terrain curriculum for high-step climbing.

    The stock terrain curriculum requires crossing half of the terrain tile.
    For the high-step task, the useful behavior is climbing the obstacle and
    making controlled forward progress, so the distance gate is slightly shorter
    and the demotion gate is less aggressive.
    """
    ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1)
    height_gain = asset.data.root_pos_w[ids, 2] - env.scene.env_origins[ids, 2] - nominal_base_height
    terrain_levels = getattr(terrain, "terrain_levels", None)
    allowed_max_level = None
    if stage_update_thresholds is not None and stage_max_levels is not None:
        if len(stage_max_levels) != len(stage_update_thresholds) + 1:
            raise ValueError(
                "stage_max_levels must contain one more value than stage_update_thresholds "
                "for terrain_levels_vel_highstep."
            )
        update_count = env.common_step_counter // max(int(num_steps_per_update), 1)
        allowed_max_level = int(stage_max_levels[0])
        for threshold, max_level in zip(stage_update_thresholds, stage_max_levels[1:]):
            if update_count >= int(threshold):
                allowed_max_level = int(max_level)

    if terrain_levels is not None:
        if allowed_max_level is not None:
            over_allowed = terrain_levels[ids] > allowed_max_level
            if bool(torch.any(over_allowed).item()):
                clipped_ids = ids[over_allowed]
                terrain_levels[clipped_ids] = allowed_max_level
                terrain.env_origins[clipped_ids] = terrain.terrain_origins[
                    terrain_levels[clipped_ids], terrain.terrain_types[clipped_ids]
                ]
        levels = terrain_levels[ids]
        height_gate = (levels < height_gain_required_level) | (height_gain > min_height_gain)
        climb_level_gate = levels >= climb_height_required_level
    else:
        height_gate = height_gain > min_height_gain
        climb_level_gate = torch.ones_like(height_gate, dtype=torch.bool)

    distance_move_up = (distance > move_up_distance) & height_gate
    climb_move_up = (distance > climb_up_distance) & (height_gain > climb_height_gain) & climb_level_gate
    move_up = distance_move_up | climb_move_up
    if allowed_max_level is not None and terrain_levels is not None:
        move_up &= terrain_levels[ids] < allowed_max_level

    commanded_distance = torch.norm(command[ids, :2], dim=1) * env.max_episode_length_s * move_down_command_factor
    move_down_distance = torch.clamp(
        commanded_distance,
        min=move_down_min_distance,
        max=move_down_max_distance,
    )
    climb_hold = (distance > climb_hold_distance) & (height_gain > climb_hold_height_gain) & climb_level_gate
    move_down = distance < move_down_distance
    move_down *= ~(move_up | climb_hold)

    terrain.update_env_origins(ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def command_levels_vel_highstep(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 0.85),
    gated_multiplier: float = 0.6,
    terrain_gate_level: float = 2.5,
    delta: float = 0.05,
    reward_threshold: float = 0.75,
) -> torch.Tensor:
    """Velocity-command curriculum with a terrain gate for high-step training."""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    if env.common_step_counter == 0 or not hasattr(env, "_highstep_original_vel_x"):
        env._highstep_original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._highstep_original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._highstep_original_yaw = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._highstep_initial_vel_x = env._highstep_original_vel_x * range_multiplier[0]
        env._highstep_initial_vel_y = env._highstep_original_vel_y * range_multiplier[0]
        env._highstep_initial_yaw = env._highstep_original_yaw * range_multiplier[0]
        env._highstep_final_vel_x = env._highstep_original_vel_x * range_multiplier[1]
        env._highstep_final_vel_y = env._highstep_original_vel_y * range_multiplier[1]
        env._highstep_final_yaw = env._highstep_original_yaw * range_multiplier[1]
        env._highstep_gated_vel_x = env._highstep_original_vel_x * gated_multiplier
        env._highstep_gated_vel_y = env._highstep_original_vel_y * gated_multiplier
        env._highstep_gated_yaw = env._highstep_original_yaw * gated_multiplier

        base_velocity_ranges.lin_vel_x = env._highstep_initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._highstep_initial_vel_y.tolist()
        base_velocity_ranges.ang_vel_z = env._highstep_initial_yaw.tolist()

    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        tracking_score = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s

        if tracking_score > reward_threshold * reward_term_cfg.weight:
            terrain_levels = getattr(env.scene.terrain, "terrain_levels", None)
            mean_terrain_level = (
                torch.mean(terrain_levels.float()) if terrain_levels is not None else torch.tensor(0.0, device=env.device)
            )
            target_vel_x = torch.where(
                mean_terrain_level >= terrain_gate_level,
                env._highstep_final_vel_x,
                env._highstep_gated_vel_x,
            )
            target_vel_y = torch.where(
                mean_terrain_level >= terrain_gate_level,
                env._highstep_final_vel_y,
                env._highstep_gated_vel_y,
            )
            target_yaw = torch.where(
                mean_terrain_level >= terrain_gate_level,
                env._highstep_final_yaw,
                env._highstep_gated_yaw,
            )

            current_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
            current_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
            current_yaw = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
            step = torch.tensor(delta, device=env.device)
            new_vel_x = current_vel_x + torch.clamp(target_vel_x - current_vel_x, min=-step, max=step)
            new_vel_y = current_vel_y + torch.clamp(target_vel_y - current_vel_y, min=-step, max=step)
            new_yaw = current_yaw + torch.clamp(target_yaw - current_yaw, min=-step, max=step)

            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()
            base_velocity_ranges.ang_vel_z = new_yaw.tolist()

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
