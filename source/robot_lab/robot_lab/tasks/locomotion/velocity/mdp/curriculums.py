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

from .highstep_schedule import global_update as _global_update

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
            "active_rate": zero,
            "active_step_rate": zero,
            "soft_active_step_rate": zero,
            "valid_rate": zero,
            "lead_capture_rate": zero,
            "lead_candidate_step_rate": zero,
            "rl_first_rate": zero,
            "rr_first_rate": zero,
            "both_first_rate": zero,
            "rl_minus_rr_first_rate": zero,
            "one_sided_stall_ratio": zero,
            "second_clear_rate": zero,
            "active_gate_mean": zero,
            "active_gate_max": zero,
            "task_gate_mean": zero,
            "task_gate_max": zero,
            "terrain_gate_mean": zero,
            "terrain_gate_max": zero,
            "commit_gate_mean": zero,
            "commit_gate_max": zero,
            "cmd_gate_mean": zero,
            "cmd_gate_max": zero,
            "first_score_mean": zero,
            "first_score_max": zero,
            "second_score_mean": zero,
            "second_score_max": zero,
            "post_lead_stall_base_mean": zero,
            "post_lead_stall_base_max": zero,
            "post_lead_stall_second_low_mean": zero,
            "post_lead_stall_progress_low_mean": zero,
            "post_lead_stall_modifier_mean": zero,
            "post_lead_stall_signal_mean": zero,
            "post_lead_stall_signal_max": zero,
            "lead_support_drive_gate_mean": zero,
            "lead_support_drive_lead_mean": zero,
            "lead_support_drive_height_mean": zero,
            "lead_support_drive_height_active_mean": zero,
            "lead_support_drive_progress_mean": zero,
            "lead_support_drive_lift_velocity_mean": zero,
            "lead_support_drive_lift_velocity_active_mean": zero,
            "lead_support_drive_body_mean": zero,
            "lead_support_drive_body_active_mean": zero,
            "lead_support_drive_signal_mean": zero,
            "lead_support_drive_signal_max": zero,
            "lead_support_drive_height_max": zero,
            "lead_support_drive_lift_velocity_max": zero,
            "lead_support_drive_body_max": zero,
            "post_lead_drive_gate_mean": zero,
            "post_lead_drive_lead_mean": zero,
            "post_lead_drive_height_mean": zero,
            "post_lead_drive_lift_velocity_mean": zero,
            "post_lead_drive_progress_mean": zero,
            "post_lead_drive_second_mean": zero,
            "post_lead_drive_stage_gate_mean": zero,
            "post_lead_drive_signal_mean": zero,
            "post_lead_drive_signal_max": zero,
            "post_clear_recovery_gate_mean": zero,
            "post_clear_recovery_posture_mean": zero,
            "post_clear_recovery_base_clearance_mean": zero,
            "post_clear_recovery_forward_mean": zero,
            "post_clear_recovery_rear_advance_mean": zero,
            "post_clear_recovery_signal_mean": zero,
            "post_clear_recovery_signal_max": zero,
            "post_clear_stall_margin_mean": zero,
            "post_clear_stall_margin_min_mean": zero,
            "post_clear_stall_counter_max_mean": zero,
            "post_clear_stall_signal_mean": zero,
            "scanner_pretrigger_gate_mean": zero,
            "scanner_pretrigger_terrain_gate_mean": zero,
            "scanner_pretrigger_commit_gate_mean": zero,
            "scanner_pretrigger_low_cmd_gate_mean": zero,
            "scanner_pretrigger_front_lift_mean": zero,
            "scanner_pretrigger_uncommanded_vel_mean": zero,
            "scanner_pretrigger_signal_mean": zero,
            "scanner_pretrigger_signal_max": zero,
            "rear_approach_width_active_rate": zero,
            "rear_approach_width_gate_mean": zero,
            "rear_approach_width_terrain_gate_mean": zero,
            "rear_approach_width_commit_gate_mean": zero,
            "rear_approach_width_cmd_gate_mean": zero,
            "rear_approach_width_mean": zero,
            "rear_approach_width_active_mean": zero,
            "rear_approach_min_abs_y_mean": zero,
            "rear_approach_min_abs_y_active_mean": zero,
            "rear_approach_min_abs_y_active_min": zero,
            "rear_approach_width_violation_rate": zero,
            "rear_approach_center_violation_rate": zero,
            "rear_approach_center_deficit_max": zero,
            "rear_approach_width_signal_mean": zero,
            "rear_approach_width_signal_max": zero,
            "rear_motion_width_active_rate": zero,
            "rear_motion_width_gate_mean": zero,
            "rear_motion_width_terrain_gate_mean": zero,
            "rear_motion_width_commit_gate_mean": zero,
            "rear_motion_width_cmd_gate_mean": zero,
            "rear_motion_width_second_mean": zero,
            "rear_motion_width_mean": zero,
            "rear_motion_width_active_mean": zero,
            "rear_motion_min_abs_y_mean": zero,
            "rear_motion_min_abs_y_active_mean": zero,
            "rear_motion_min_abs_y_active_min": zero,
            "rear_motion_width_violation_rate": zero,
            "rear_motion_center_violation_rate": zero,
            "rear_motion_center_deficit_max": zero,
            "rear_motion_width_signal_mean": zero,
            "rear_motion_width_signal_max": zero,
            "support_stability_active_rate": zero,
            "support_stability_gate_mean": zero,
            "support_stability_front_contact_active_mean": zero,
            "support_stability_front_slip_active_mean": zero,
            "support_stability_posture_rate_active_mean": zero,
            "support_stability_rear_lag_active_mean": zero,
            "support_stability_support_width_active_mean": zero,
            "support_stability_signal_mean": zero,
            "support_stability_signal_max": zero,
            "front_lift_guard_gate_mean": zero,
            "front_lift_guard_signal_mean": zero,
            "front_lift_guard_signal_max": zero,
            "front_lift_guard_max_lift_mean": zero,
            "front_lift_guard_asymmetry_mean": zero,
            "front_lift_guard_height_excess_mean": zero,
            "front_lift_guard_asymmetry_excess_mean": zero,
            "front_lift_guard_terrain_gate_mean": zero,
            "front_lift_guard_commit_gate_mean": zero,
            "fl_forward_flat_active_gate_mean": zero,
            "fl_forward_flat_forward_gate_mean": zero,
            "fl_forward_flat_backward_gate_mean": zero,
            "fl_forward_flat_lateral_gate_mean": zero,
            "fl_forward_flat_yaw_gate_mean": zero,
            "fl_forward_flat_flat_gate_mean": zero,
            "fl_forward_flat_highstep_relief_mean": zero,
            "fl_forward_flat_lift_mean": zero,
            "fr_forward_flat_lift_mean": zero,
            "fl_minus_fr_forward_flat_mean": zero,
            "fl_forward_flat_height_excess_mean": zero,
            "fl_forward_flat_asymmetry_excess_mean": zero,
            "fl_forward_flat_overlift_rate": zero,
            "fl_forward_flat_penalty_signal_mean": zero,
            "fl_forward_flat_penalty_signal_max": zero,
            "target_limit_violation_rate_mean": zero,
            "target_limit_step_violation_rate_mean": zero,
            "target_limit_max_delta_mean": zero,
            "target_limit_violation_max_consecutive_mean": zero,
            "target_margin_violation_rate_mean": zero,
            "target_margin_max_delta_mean": zero,
            "target_limit_penalty_mean": zero,
        }

    lead = env._highstep_rear_branch_lead[ids]
    commit_steps = env._highstep_rear_branch_commit_steps[ids]
    one_sided_steps = env._highstep_rear_branch_one_sided_steps[ids]
    second_clear_steps = env._highstep_rear_branch_second_clear_steps[ids]

    def _buffer(buffer_name: str) -> torch.Tensor:
        if hasattr(env, buffer_name):
            return getattr(env, buffer_name)[ids]
        return torch.zeros_like(commit_steps)

    sample_steps = _buffer("_highstep_rear_branch_sample_steps")
    sample_steps = torch.clamp(sample_steps, min=1.0)

    def _episode_mean(buffer_name: str) -> torch.Tensor:
        values = _buffer(buffer_name)
        return (values / sample_steps).mean()

    def _episode_max_mean(buffer_name: str) -> torch.Tensor:
        return _buffer(buffer_name).mean()

    def _episode_active_min_mean(buffer_name: str, active_steps_name: str) -> torch.Tensor:
        active = _buffer(active_steps_name) > 0.0
        values = _buffer(buffer_name)
        return torch.where(active, values, torch.zeros_like(values)).mean()

    active = commit_steps >= float(min_commit_steps)
    valid = active & (lead >= 0)
    active_count = torch.clamp(active.float().sum(), min=1.0)
    valid_count = torch.clamp(valid.float().sum(), min=1.0)
    rl_first_rate = ((lead == 0) & valid).float().sum() / valid_count
    rr_first_rate = ((lead == 1) & valid).float().sum() / valid_count
    both_first_rate = ((lead == 2) & valid).float().sum() / valid_count
    active_rate = active.float().mean()
    active_step_rate = _episode_mean("_highstep_rear_branch_active_steps")
    soft_active_step_rate = _episode_mean("_highstep_rear_branch_soft_active_steps")
    valid_rate = valid.float().mean()
    lead_capture_rate = valid.float().sum() / active_count
    lead_candidate_step_rate = _episode_mean("_highstep_rear_branch_lead_candidate_steps")
    one_sided_stall_ratio = one_sided_steps.sum() / torch.clamp(commit_steps.sum(), min=1.0)
    second_clear_rate = ((second_clear_steps > 0.0) & valid).float().sum() / valid_count
    active_gate_mean = _episode_mean("_highstep_rear_branch_active_gate_sum")
    active_gate_max = _episode_max_mean("_highstep_rear_branch_active_gate_max")
    task_gate_mean = _episode_mean("_highstep_rear_branch_task_gate_sum")
    task_gate_max = _episode_max_mean("_highstep_rear_branch_task_gate_max")
    terrain_gate_mean = _episode_mean("_highstep_rear_branch_terrain_gate_sum")
    terrain_gate_max = _episode_max_mean("_highstep_rear_branch_terrain_gate_max")
    commit_gate_mean = _episode_mean("_highstep_rear_branch_commit_gate_sum")
    commit_gate_max = _episode_max_mean("_highstep_rear_branch_commit_gate_max")
    cmd_gate_mean = _episode_mean("_highstep_rear_branch_cmd_gate_sum")
    cmd_gate_max = _episode_max_mean("_highstep_rear_branch_cmd_gate_max")
    first_score_mean = _episode_mean("_highstep_rear_branch_first_score_sum")
    first_score_max = _episode_max_mean("_highstep_rear_branch_first_score_max")
    second_score_mean = _episode_mean("_highstep_rear_branch_second_score_sum")
    second_score_max = _episode_max_mean("_highstep_rear_branch_second_score_max")
    post_lead_sample_steps = torch.clamp(_buffer("_highstep_post_lead_stall_sample_steps"), min=1.0)

    def _post_lead_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / post_lead_sample_steps).mean()

    post_lead_stall_base_mean = _post_lead_mean("_highstep_post_lead_stall_base_sum")
    post_lead_stall_base_max = _episode_max_mean("_highstep_post_lead_stall_base_max")
    post_lead_stall_second_low_mean = _post_lead_mean("_highstep_post_lead_stall_second_low_sum")
    post_lead_stall_progress_low_mean = _post_lead_mean("_highstep_post_lead_stall_progress_low_sum")
    post_lead_stall_modifier_mean = _post_lead_mean("_highstep_post_lead_stall_modifier_sum")
    post_lead_stall_signal_mean = _post_lead_mean("_highstep_post_lead_stall_signal_sum")
    post_lead_stall_signal_max = _episode_max_mean("_highstep_post_lead_stall_signal_max")
    lead_support_sample_steps = torch.clamp(_buffer("_highstep_lead_support_drive_sample_steps"), min=1.0)

    def _lead_support_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / lead_support_sample_steps).mean()

    def _lead_support_active_mean(buffer_name: str) -> torch.Tensor:
        gate_weight = _buffer("_highstep_lead_support_drive_gate_sum")
        active = gate_weight > 1.0e-6
        value = _buffer(buffer_name) / torch.clamp(gate_weight, min=1.0e-6)
        return torch.where(active, value, torch.zeros_like(value)).mean()

    lead_support_drive_gate_mean = _lead_support_mean("_highstep_lead_support_drive_gate_sum")
    lead_support_drive_lead_mean = _lead_support_mean("_highstep_lead_support_drive_lead_sum")
    lead_support_drive_height_mean = _lead_support_mean("_highstep_lead_support_drive_height_sum")
    lead_support_drive_height_active_mean = _lead_support_active_mean(
        "_highstep_lead_support_drive_height_active_sum"
    )
    lead_support_drive_progress_mean = _lead_support_mean("_highstep_lead_support_drive_progress_sum")
    lead_support_drive_lift_velocity_mean = _lead_support_mean("_highstep_lead_support_drive_lift_velocity_sum")
    lead_support_drive_lift_velocity_active_mean = _lead_support_active_mean(
        "_highstep_lead_support_drive_lift_velocity_active_sum"
    )
    lead_support_drive_body_mean = _lead_support_mean("_highstep_lead_support_drive_body_sum")
    lead_support_drive_body_active_mean = _lead_support_active_mean(
        "_highstep_lead_support_drive_body_active_sum"
    )
    lead_support_drive_signal_mean = _lead_support_mean("_highstep_lead_support_drive_signal_sum")
    lead_support_drive_signal_max = _episode_max_mean("_highstep_lead_support_drive_signal_max")
    lead_support_drive_height_max = _episode_max_mean("_highstep_lead_support_drive_height_max")
    lead_support_drive_lift_velocity_max = _episode_max_mean("_highstep_lead_support_drive_lift_velocity_max")
    lead_support_drive_body_max = _episode_max_mean("_highstep_lead_support_drive_body_max")
    post_lead_drive_sample_steps = torch.clamp(_buffer("_highstep_post_lead_drive_sample_steps"), min=1.0)

    def _post_lead_drive_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / post_lead_drive_sample_steps).mean()

    post_lead_drive_gate_mean = _post_lead_drive_mean("_highstep_post_lead_drive_gate_sum")
    post_lead_drive_lead_mean = _post_lead_drive_mean("_highstep_post_lead_drive_lead_sum")
    post_lead_drive_height_mean = _post_lead_drive_mean("_highstep_post_lead_drive_height_sum")
    post_lead_drive_lift_velocity_mean = _post_lead_drive_mean("_highstep_post_lead_drive_lift_velocity_sum")
    post_lead_drive_progress_mean = _post_lead_drive_mean("_highstep_post_lead_drive_progress_sum")
    post_lead_drive_second_mean = _post_lead_drive_mean("_highstep_post_lead_drive_second_sum")
    post_lead_drive_stage_gate_mean = _post_lead_drive_mean(
        "_highstep_post_lead_drive_stage_gate_sum"
    )
    post_lead_drive_signal_mean = _post_lead_drive_mean("_highstep_post_lead_drive_signal_sum")
    post_lead_drive_signal_max = _episode_max_mean("_highstep_post_lead_drive_signal_max")
    post_clear_recovery_sample_steps = torch.clamp(
        _buffer("_highstep_post_clear_recovery_sample_steps"), min=1.0
    )

    def _post_clear_recovery_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / post_clear_recovery_sample_steps).mean()

    post_clear_recovery_gate_mean = _post_clear_recovery_mean("_highstep_post_clear_recovery_gate_sum")
    post_clear_recovery_posture_mean = _post_clear_recovery_mean(
        "_highstep_post_clear_recovery_posture_sum"
    )
    post_clear_recovery_base_clearance_mean = _post_clear_recovery_mean(
        "_highstep_post_clear_recovery_base_clearance_sum"
    )
    post_clear_recovery_forward_mean = _post_clear_recovery_mean("_highstep_post_clear_recovery_forward_sum")
    post_clear_recovery_rear_advance_mean = _post_clear_recovery_mean(
        "_highstep_post_clear_recovery_rear_advance_sum"
    )
    post_clear_recovery_signal_mean = _post_clear_recovery_mean("_highstep_post_clear_recovery_signal_sum")
    post_clear_recovery_signal_max = _episode_max_mean("_highstep_post_clear_recovery_signal_max")
    post_clear_stall_sample_steps_raw = _buffer("_highstep_post_clear_stall_sample_steps")
    post_clear_stall_sample_steps = torch.clamp(post_clear_stall_sample_steps_raw, min=1.0)
    post_clear_stall_margin_mean = (
        _buffer("_highstep_post_clear_stall_margin_sum") / post_clear_stall_sample_steps
    ).mean()
    post_clear_stall_margin_min_mean = _episode_active_min_mean(
        "_highstep_post_clear_stall_margin_min",
        "_highstep_post_clear_stall_sample_steps",
    )
    post_clear_stall_counter_max_mean = _episode_max_mean(
        "_highstep_post_clear_stall_counter_max"
    )
    post_clear_stall_signal_mean = (
        _buffer("_highstep_post_clear_stall_signal_sum") / post_clear_stall_sample_steps
    ).mean()
    scanner_pretrigger_sample_steps = torch.clamp(
        _buffer("_highstep_scanner_pretrigger_sample_steps"), min=1.0
    )

    def _scanner_pretrigger_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / scanner_pretrigger_sample_steps).mean()

    scanner_pretrigger_gate_mean = _scanner_pretrigger_mean("_highstep_scanner_pretrigger_gate_sum")
    scanner_pretrigger_terrain_gate_mean = _scanner_pretrigger_mean(
        "_highstep_scanner_pretrigger_terrain_gate_sum"
    )
    scanner_pretrigger_commit_gate_mean = _scanner_pretrigger_mean(
        "_highstep_scanner_pretrigger_commit_gate_sum"
    )
    scanner_pretrigger_low_cmd_gate_mean = _scanner_pretrigger_mean(
        "_highstep_scanner_pretrigger_low_cmd_gate_sum"
    )
    scanner_pretrigger_front_lift_mean = _scanner_pretrigger_mean(
        "_highstep_scanner_pretrigger_front_lift_sum"
    )
    scanner_pretrigger_uncommanded_vel_mean = _scanner_pretrigger_mean(
        "_highstep_scanner_pretrigger_uncommanded_vel_sum"
    )
    scanner_pretrigger_signal_mean = _scanner_pretrigger_mean("_highstep_scanner_pretrigger_signal_sum")
    scanner_pretrigger_signal_max = _episode_max_mean("_highstep_scanner_pretrigger_signal_max")
    rear_approach_width_sample_steps = torch.clamp(
        _buffer("_highstep_rear_approach_width_sample_steps"), min=1.0
    )
    rear_approach_width_active_steps = torch.clamp(
        _buffer("_highstep_rear_approach_width_active_steps"), min=1.0
    )
    rear_approach_width_gate_sum = _buffer("_highstep_rear_approach_width_gate_sum")

    def _rear_approach_width_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / rear_approach_width_sample_steps).mean()

    def _rear_approach_width_active_mean(buffer_name: str) -> torch.Tensor:
        active = rear_approach_width_gate_sum > 1.0e-6
        value = _buffer(buffer_name) / torch.clamp(rear_approach_width_gate_sum, min=1.0e-6)
        return torch.where(active, value, torch.zeros_like(value)).mean()

    rear_approach_width_active_rate = (
        _buffer("_highstep_rear_approach_width_active_steps") / rear_approach_width_sample_steps
    ).mean()
    rear_approach_width_gate_mean = _rear_approach_width_mean("_highstep_rear_approach_width_gate_sum")
    rear_approach_width_terrain_gate_mean = _rear_approach_width_mean(
        "_highstep_rear_approach_width_terrain_gate_sum"
    )
    rear_approach_width_commit_gate_mean = _rear_approach_width_mean(
        "_highstep_rear_approach_width_commit_gate_sum"
    )
    rear_approach_width_cmd_gate_mean = _rear_approach_width_mean("_highstep_rear_approach_width_cmd_gate_sum")
    rear_approach_width_mean = _rear_approach_width_mean("_highstep_rear_approach_width_sum")
    rear_approach_width_active_mean = _rear_approach_width_active_mean(
        "_highstep_rear_approach_width_active_sum"
    )
    rear_approach_min_abs_y_mean = _rear_approach_width_mean("_highstep_rear_approach_min_abs_y_sum")
    rear_approach_min_abs_y_active_mean = _rear_approach_width_active_mean(
        "_highstep_rear_approach_min_abs_y_active_sum"
    )
    rear_approach_min_abs_y_active_min = _episode_active_min_mean(
        "_highstep_rear_approach_min_abs_y_active_min",
        "_highstep_rear_approach_width_active_steps",
    )
    rear_approach_width_violation_rate = (
        _buffer("_highstep_rear_approach_width_violation_steps") / rear_approach_width_active_steps
    ).mean()
    rear_approach_center_violation_rate = (
        _buffer("_highstep_rear_approach_center_violation_steps") / rear_approach_width_active_steps
    ).mean()
    rear_approach_center_deficit_max = _episode_max_mean("_highstep_rear_approach_center_deficit_max")
    rear_approach_width_signal_mean = _rear_approach_width_mean("_highstep_rear_approach_signal_sum")
    rear_approach_width_signal_max = _episode_max_mean("_highstep_rear_approach_signal_max")
    rear_motion_width_sample_steps = torch.clamp(_buffer("_highstep_rear_motion_width_sample_steps"), min=1.0)
    rear_motion_width_active_steps = torch.clamp(_buffer("_highstep_rear_motion_width_active_steps"), min=1.0)
    rear_motion_width_gate_sum = _buffer("_highstep_rear_motion_width_gate_sum")

    def _rear_motion_width_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / rear_motion_width_sample_steps).mean()

    def _rear_motion_width_active_mean(buffer_name: str) -> torch.Tensor:
        active = rear_motion_width_gate_sum > 1.0e-6
        value = _buffer(buffer_name) / torch.clamp(rear_motion_width_gate_sum, min=1.0e-6)
        return torch.where(active, value, torch.zeros_like(value)).mean()

    rear_motion_width_active_rate = (
        _buffer("_highstep_rear_motion_width_active_steps") / rear_motion_width_sample_steps
    ).mean()
    rear_motion_width_gate_mean = _rear_motion_width_mean("_highstep_rear_motion_width_gate_sum")
    rear_motion_width_terrain_gate_mean = _rear_motion_width_mean("_highstep_rear_motion_width_terrain_gate_sum")
    rear_motion_width_commit_gate_mean = _rear_motion_width_mean("_highstep_rear_motion_width_commit_gate_sum")
    rear_motion_width_cmd_gate_mean = _rear_motion_width_mean("_highstep_rear_motion_width_cmd_gate_sum")
    rear_motion_width_second_mean = _rear_motion_width_mean("_highstep_rear_motion_width_second_sum")
    rear_motion_width_mean = _rear_motion_width_mean("_highstep_rear_motion_width_sum")
    rear_motion_width_active_mean = _rear_motion_width_active_mean("_highstep_rear_motion_width_active_sum")
    rear_motion_min_abs_y_mean = _rear_motion_width_mean("_highstep_rear_motion_min_abs_y_sum")
    rear_motion_min_abs_y_active_mean = _rear_motion_width_active_mean(
        "_highstep_rear_motion_min_abs_y_active_sum"
    )
    rear_motion_min_abs_y_active_min = _episode_active_min_mean(
        "_highstep_rear_motion_min_abs_y_active_min",
        "_highstep_rear_motion_width_active_steps",
    )
    rear_motion_width_violation_rate = (
        _buffer("_highstep_rear_motion_width_violation_steps") / rear_motion_width_active_steps
    ).mean()
    rear_motion_center_violation_rate = (
        _buffer("_highstep_rear_motion_center_violation_steps") / rear_motion_width_active_steps
    ).mean()
    rear_motion_center_deficit_max = _episode_max_mean("_highstep_rear_motion_center_deficit_max")
    rear_motion_width_signal_mean = _rear_motion_width_mean("_highstep_rear_motion_signal_sum")
    rear_motion_width_signal_max = _episode_max_mean("_highstep_rear_motion_signal_max")
    support_stability_sample_steps = torch.clamp(
        _buffer("_highstep_support_stability_sample_steps"),
        min=1.0,
    )
    support_stability_active_steps = torch.clamp(
        _buffer("_highstep_support_stability_active_steps"),
        min=1.0,
    )
    support_stability_gate_sum = _buffer("_highstep_support_stability_gate_sum")

    def _support_stability_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / support_stability_sample_steps).mean()

    def _support_stability_active_mean(buffer_name: str) -> torch.Tensor:
        active = support_stability_gate_sum > 1.0e-6
        value = _buffer(buffer_name) / torch.clamp(support_stability_gate_sum, min=1.0e-6)
        return torch.where(active, value, torch.zeros_like(value)).mean()

    support_stability_active_rate = (
        _buffer("_highstep_support_stability_active_steps") / support_stability_sample_steps
    ).mean()
    support_stability_gate_mean = _support_stability_mean("_highstep_support_stability_gate_sum")
    support_stability_front_contact_active_mean = _support_stability_active_mean(
        "_highstep_support_stability_front_contact_sum"
    )
    support_stability_front_slip_active_mean = _support_stability_active_mean(
        "_highstep_support_stability_front_slip_sum"
    )
    support_stability_posture_rate_active_mean = _support_stability_active_mean(
        "_highstep_support_stability_posture_rate_sum"
    )
    support_stability_rear_lag_active_mean = _support_stability_active_mean(
        "_highstep_support_stability_rear_lag_sum"
    )
    support_stability_support_width_active_mean = _support_stability_active_mean(
        "_highstep_support_stability_support_width_sum"
    )
    support_stability_signal_mean = _support_stability_mean("_highstep_support_stability_signal_sum")
    support_stability_signal_max = _episode_max_mean("_highstep_support_stability_signal_max")
    front_lift_guard_sample_steps = torch.clamp(_buffer("_highstep_front_lift_guard_sample_steps"), min=1.0)

    def _front_lift_guard_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / front_lift_guard_sample_steps).mean()

    front_lift_guard_gate_mean = _front_lift_guard_mean("_highstep_front_lift_guard_gate_sum")
    front_lift_guard_signal_mean = _front_lift_guard_mean("_highstep_front_lift_guard_signal_sum")
    front_lift_guard_signal_max = _episode_max_mean("_highstep_front_lift_guard_signal_max")
    front_lift_guard_max_lift_mean = _front_lift_guard_mean("_highstep_front_lift_guard_max_lift_sum")
    front_lift_guard_asymmetry_mean = _front_lift_guard_mean("_highstep_front_lift_guard_asym_sum")
    front_lift_guard_height_excess_mean = _front_lift_guard_mean(
        "_highstep_front_lift_guard_height_excess_sum"
    )
    front_lift_guard_asymmetry_excess_mean = _front_lift_guard_mean(
        "_highstep_front_lift_guard_asym_excess_sum"
    )
    front_lift_guard_terrain_gate_mean = _front_lift_guard_mean(
        "_highstep_front_lift_guard_terrain_gate_sum"
    )
    front_lift_guard_commit_gate_mean = _front_lift_guard_mean(
        "_highstep_front_lift_guard_commit_gate_sum"
    )
    fl_forward_flat_sample_steps = torch.clamp(_buffer("_highstep_fl_forward_flat_sample_steps"), min=1.0)

    def _fl_forward_flat_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / fl_forward_flat_sample_steps).mean()

    fl_forward_flat_active_gate_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_active_gate_sum")
    fl_forward_flat_forward_gate_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_forward_gate_sum")
    fl_forward_flat_backward_gate_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_backward_gate_sum")
    fl_forward_flat_lateral_gate_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_lateral_gate_sum")
    fl_forward_flat_yaw_gate_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_yaw_gate_sum")
    fl_forward_flat_flat_gate_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_flat_gate_sum")
    fl_forward_flat_highstep_relief_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_highstep_relief_sum")
    fl_forward_flat_lift_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_fl_lift_sum")
    fr_forward_flat_lift_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_fr_lift_sum")
    fl_minus_fr_forward_flat_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_fl_minus_fr_sum")
    fl_forward_flat_height_excess_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_height_excess_sum")
    fl_forward_flat_asymmetry_excess_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_asym_excess_sum")
    fl_forward_flat_overlift_rate = _fl_forward_flat_mean("_highstep_fl_forward_flat_overlift_sum")
    fl_forward_flat_penalty_signal_mean = _fl_forward_flat_mean("_highstep_fl_forward_flat_penalty_sum")
    fl_forward_flat_penalty_signal_max = _episode_max_mean("_highstep_fl_forward_flat_penalty_max")
    target_limit_sample_steps = torch.clamp(_buffer("_highstep_target_limit_sample_steps"), min=1.0)

    def _target_limit_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / target_limit_sample_steps).mean()

    target_limit_violation_rate_mean = _target_limit_mean("_highstep_target_limit_violation_sum")
    target_limit_step_violation_rate_mean = _target_limit_mean(
        "_highstep_target_limit_step_violation_sum"
    )
    target_limit_max_delta_mean = _episode_max_mean("_highstep_target_limit_max_delta_episode")
    target_limit_violation_max_consecutive_mean = _episode_max_mean(
        "_highstep_target_limit_violation_max_consecutive"
    )
    target_margin_violation_rate_mean = _target_limit_mean("_highstep_target_margin_violation_sum")
    target_margin_max_delta_mean = _episode_max_mean("_highstep_target_margin_max_delta_episode")
    target_limit_penalty_mean = _target_limit_mean("_highstep_target_limit_penalty_sum")

    # Reset only the envs that are ending now; the next episode should start fresh.
    env._highstep_rear_branch_lead[ids] = -1
    env._highstep_rear_branch_commit_steps[ids] = 0.0
    if hasattr(env, "_highstep_rear_branch_lead_steps"):
        env._highstep_rear_branch_lead_steps[ids] = 0.0
    env._highstep_rear_branch_one_sided_steps[ids] = 0.0
    env._highstep_rear_branch_second_clear_steps[ids] = 0.0
    if hasattr(env, "_highstep_post_clear_origin_valid"):
        env._highstep_post_clear_origin_valid[ids] = False
        env._highstep_post_clear_origin_rear_pos_w[ids] = 0.0
        env._highstep_post_clear_heading_w[ids] = 0.0
        env._highstep_post_clear_origin_step[ids] = 0.0
    for buffer_name in (
        "_highstep_rear_branch_sample_steps",
        "_highstep_rear_branch_active_steps",
        "_highstep_rear_branch_soft_active_steps",
        "_highstep_rear_branch_lead_candidate_steps",
        "_highstep_rear_branch_active_gate_sum",
        "_highstep_rear_branch_active_gate_max",
        "_highstep_rear_branch_task_gate_sum",
        "_highstep_rear_branch_task_gate_max",
        "_highstep_rear_branch_terrain_gate_sum",
        "_highstep_rear_branch_terrain_gate_max",
        "_highstep_rear_branch_commit_gate_sum",
        "_highstep_rear_branch_commit_gate_max",
        "_highstep_rear_branch_cmd_gate_sum",
        "_highstep_rear_branch_cmd_gate_max",
        "_highstep_rear_branch_first_score_sum",
        "_highstep_rear_branch_first_score_max",
        "_highstep_rear_branch_second_score_sum",
        "_highstep_rear_branch_second_score_max",
        "_highstep_post_lead_stall_sample_steps",
        "_highstep_post_lead_stall_base_sum",
        "_highstep_post_lead_stall_base_max",
        "_highstep_post_lead_stall_second_low_sum",
        "_highstep_post_lead_stall_progress_low_sum",
        "_highstep_post_lead_stall_modifier_sum",
        "_highstep_post_lead_stall_signal_sum",
        "_highstep_post_lead_stall_signal_max",
        "_highstep_lead_support_drive_sample_steps",
        "_highstep_lead_support_drive_gate_sum",
        "_highstep_lead_support_drive_lead_sum",
        "_highstep_lead_support_drive_height_sum",
        "_highstep_lead_support_drive_height_active_sum",
        "_highstep_lead_support_drive_progress_sum",
        "_highstep_lead_support_drive_lift_velocity_sum",
        "_highstep_lead_support_drive_lift_velocity_active_sum",
        "_highstep_lead_support_drive_body_sum",
        "_highstep_lead_support_drive_body_active_sum",
        "_highstep_lead_support_drive_signal_sum",
        "_highstep_lead_support_drive_signal_max",
        "_highstep_lead_support_drive_height_max",
        "_highstep_lead_support_drive_lift_velocity_max",
        "_highstep_lead_support_drive_body_max",
        "_highstep_post_lead_drive_sample_steps",
        "_highstep_post_lead_drive_gate_sum",
        "_highstep_post_lead_drive_lead_sum",
        "_highstep_post_lead_drive_height_sum",
        "_highstep_post_lead_drive_lift_velocity_sum",
        "_highstep_post_lead_drive_progress_sum",
        "_highstep_post_lead_drive_second_sum",
        "_highstep_post_lead_drive_stage_gate_sum",
        "_highstep_post_lead_drive_signal_sum",
        "_highstep_post_lead_drive_signal_max",
        "_highstep_post_clear_recovery_sample_steps",
        "_highstep_post_clear_recovery_gate_sum",
        "_highstep_post_clear_recovery_posture_sum",
        "_highstep_post_clear_recovery_base_clearance_sum",
        "_highstep_post_clear_recovery_forward_sum",
        "_highstep_post_clear_recovery_rear_advance_sum",
        "_highstep_post_clear_recovery_signal_sum",
        "_highstep_post_clear_recovery_signal_max",
        "_highstep_post_clear_stall_near_edge_counter",
        "_highstep_post_clear_stall_sample_steps",
        "_highstep_post_clear_stall_margin_sum",
        "_highstep_post_clear_stall_counter_max",
        "_highstep_post_clear_stall_signal_sum",
        "_highstep_scanner_pretrigger_sample_steps",
        "_highstep_scanner_pretrigger_gate_sum",
        "_highstep_scanner_pretrigger_terrain_gate_sum",
        "_highstep_scanner_pretrigger_commit_gate_sum",
        "_highstep_scanner_pretrigger_low_cmd_gate_sum",
        "_highstep_scanner_pretrigger_front_lift_sum",
        "_highstep_scanner_pretrigger_uncommanded_vel_sum",
        "_highstep_scanner_pretrigger_signal_sum",
        "_highstep_scanner_pretrigger_signal_max",
        "_highstep_rear_approach_width_sample_steps",
        "_highstep_rear_approach_width_active_steps",
        "_highstep_rear_approach_width_gate_sum",
        "_highstep_rear_approach_width_terrain_gate_sum",
        "_highstep_rear_approach_width_commit_gate_sum",
        "_highstep_rear_approach_width_cmd_gate_sum",
        "_highstep_rear_approach_width_sum",
        "_highstep_rear_approach_width_active_sum",
        "_highstep_rear_approach_min_abs_y_sum",
        "_highstep_rear_approach_min_abs_y_active_sum",
        "_highstep_rear_approach_width_violation_steps",
        "_highstep_rear_approach_center_violation_steps",
        "_highstep_rear_approach_center_deficit_max",
        "_highstep_rear_approach_signal_sum",
        "_highstep_rear_approach_signal_max",
        "_highstep_rear_motion_width_sample_steps",
        "_highstep_rear_motion_width_active_steps",
        "_highstep_rear_motion_width_gate_sum",
        "_highstep_rear_motion_width_terrain_gate_sum",
        "_highstep_rear_motion_width_commit_gate_sum",
        "_highstep_rear_motion_width_cmd_gate_sum",
        "_highstep_rear_motion_width_second_sum",
        "_highstep_rear_motion_width_sum",
        "_highstep_rear_motion_width_active_sum",
        "_highstep_rear_motion_min_abs_y_sum",
        "_highstep_rear_motion_min_abs_y_active_sum",
        "_highstep_rear_motion_width_violation_steps",
        "_highstep_rear_motion_center_violation_steps",
        "_highstep_rear_motion_center_deficit_max",
        "_highstep_rear_motion_signal_sum",
        "_highstep_rear_motion_signal_max",
        "_highstep_support_stability_sample_steps",
        "_highstep_support_stability_active_steps",
        "_highstep_support_stability_gate_sum",
        "_highstep_support_stability_front_contact_sum",
        "_highstep_support_stability_front_slip_sum",
        "_highstep_support_stability_posture_rate_sum",
        "_highstep_support_stability_rear_lag_sum",
        "_highstep_support_stability_support_width_sum",
        "_highstep_support_stability_signal_sum",
        "_highstep_support_stability_signal_max",
        "_highstep_front_lift_guard_sample_steps",
        "_highstep_front_lift_guard_gate_sum",
        "_highstep_front_lift_guard_signal_sum",
        "_highstep_front_lift_guard_signal_max",
        "_highstep_front_lift_guard_max_lift_sum",
        "_highstep_front_lift_guard_asym_sum",
        "_highstep_front_lift_guard_height_excess_sum",
        "_highstep_front_lift_guard_asym_excess_sum",
        "_highstep_front_lift_guard_terrain_gate_sum",
        "_highstep_front_lift_guard_commit_gate_sum",
        "_highstep_fl_forward_flat_sample_steps",
        "_highstep_fl_forward_flat_active_gate_sum",
        "_highstep_fl_forward_flat_forward_gate_sum",
        "_highstep_fl_forward_flat_backward_gate_sum",
        "_highstep_fl_forward_flat_lateral_gate_sum",
        "_highstep_fl_forward_flat_yaw_gate_sum",
        "_highstep_fl_forward_flat_flat_gate_sum",
        "_highstep_fl_forward_flat_highstep_relief_sum",
        "_highstep_fl_forward_flat_fl_lift_sum",
        "_highstep_fl_forward_flat_fr_lift_sum",
        "_highstep_fl_forward_flat_fl_minus_fr_sum",
        "_highstep_fl_forward_flat_height_excess_sum",
        "_highstep_fl_forward_flat_asym_excess_sum",
        "_highstep_fl_forward_flat_overlift_sum",
        "_highstep_fl_forward_flat_penalty_sum",
        "_highstep_fl_forward_flat_penalty_max",
        "_highstep_target_limit_sample_steps",
        "_highstep_target_limit_violation_sum",
        "_highstep_target_limit_step_violation_sum",
        "_highstep_target_limit_max_delta_episode",
        "_highstep_target_limit_violation_streak",
        "_highstep_target_limit_violation_max_consecutive",
        "_highstep_target_margin_violation_sum",
        "_highstep_target_margin_max_delta_episode",
        "_highstep_target_limit_penalty_sum",
    ):
        if hasattr(env, buffer_name):
            getattr(env, buffer_name)[ids] = 0.0
    for buffer_name in (
        "_highstep_rear_approach_min_abs_y_active_min",
        "_highstep_rear_motion_min_abs_y_active_min",
        "_highstep_post_clear_stall_margin_min",
    ):
        if hasattr(env, buffer_name):
            getattr(env, buffer_name)[ids] = 10.0

    return {
        "active_rate": active_rate,
        "active_step_rate": active_step_rate,
        "soft_active_step_rate": soft_active_step_rate,
        "valid_rate": valid_rate,
        "lead_capture_rate": lead_capture_rate,
        "lead_candidate_step_rate": lead_candidate_step_rate,
        "rl_first_rate": rl_first_rate,
        "rr_first_rate": rr_first_rate,
        "both_first_rate": both_first_rate,
        "rl_minus_rr_first_rate": rl_first_rate - rr_first_rate,
        "one_sided_stall_ratio": one_sided_stall_ratio,
        "second_clear_rate": second_clear_rate,
        "active_gate_mean": active_gate_mean,
        "active_gate_max": active_gate_max,
        "task_gate_mean": task_gate_mean,
        "task_gate_max": task_gate_max,
        "terrain_gate_mean": terrain_gate_mean,
        "terrain_gate_max": terrain_gate_max,
        "commit_gate_mean": commit_gate_mean,
        "commit_gate_max": commit_gate_max,
        "cmd_gate_mean": cmd_gate_mean,
        "cmd_gate_max": cmd_gate_max,
        "first_score_mean": first_score_mean,
        "first_score_max": first_score_max,
        "second_score_mean": second_score_mean,
        "second_score_max": second_score_max,
        "post_lead_stall_base_mean": post_lead_stall_base_mean,
        "post_lead_stall_base_max": post_lead_stall_base_max,
        "post_lead_stall_second_low_mean": post_lead_stall_second_low_mean,
        "post_lead_stall_progress_low_mean": post_lead_stall_progress_low_mean,
        "post_lead_stall_modifier_mean": post_lead_stall_modifier_mean,
        "post_lead_stall_signal_mean": post_lead_stall_signal_mean,
        "post_lead_stall_signal_max": post_lead_stall_signal_max,
        "lead_support_drive_gate_mean": lead_support_drive_gate_mean,
        "lead_support_drive_lead_mean": lead_support_drive_lead_mean,
        "lead_support_drive_height_mean": lead_support_drive_height_mean,
        "lead_support_drive_height_active_mean": lead_support_drive_height_active_mean,
        "lead_support_drive_progress_mean": lead_support_drive_progress_mean,
        "lead_support_drive_lift_velocity_mean": lead_support_drive_lift_velocity_mean,
        "lead_support_drive_lift_velocity_active_mean": lead_support_drive_lift_velocity_active_mean,
        "lead_support_drive_body_mean": lead_support_drive_body_mean,
        "lead_support_drive_body_active_mean": lead_support_drive_body_active_mean,
        "lead_support_drive_signal_mean": lead_support_drive_signal_mean,
        "lead_support_drive_signal_max": lead_support_drive_signal_max,
        "lead_support_drive_height_max": lead_support_drive_height_max,
        "lead_support_drive_lift_velocity_max": lead_support_drive_lift_velocity_max,
        "lead_support_drive_body_max": lead_support_drive_body_max,
        "post_lead_drive_gate_mean": post_lead_drive_gate_mean,
        "post_lead_drive_lead_mean": post_lead_drive_lead_mean,
        "post_lead_drive_height_mean": post_lead_drive_height_mean,
        "post_lead_drive_lift_velocity_mean": post_lead_drive_lift_velocity_mean,
        "post_lead_drive_progress_mean": post_lead_drive_progress_mean,
        "post_lead_drive_second_mean": post_lead_drive_second_mean,
        "post_lead_drive_stage_gate_mean": post_lead_drive_stage_gate_mean,
        "post_lead_drive_signal_mean": post_lead_drive_signal_mean,
        "post_lead_drive_signal_max": post_lead_drive_signal_max,
        "post_clear_recovery_gate_mean": post_clear_recovery_gate_mean,
        "post_clear_recovery_posture_mean": post_clear_recovery_posture_mean,
        "post_clear_recovery_base_clearance_mean": post_clear_recovery_base_clearance_mean,
        "post_clear_recovery_forward_mean": post_clear_recovery_forward_mean,
        "post_clear_recovery_rear_advance_mean": post_clear_recovery_rear_advance_mean,
        "post_clear_recovery_signal_mean": post_clear_recovery_signal_mean,
        "post_clear_recovery_signal_max": post_clear_recovery_signal_max,
        "post_clear_stall_margin_mean": post_clear_stall_margin_mean,
        "post_clear_stall_margin_min_mean": post_clear_stall_margin_min_mean,
        "post_clear_stall_counter_max_mean": post_clear_stall_counter_max_mean,
        "post_clear_stall_signal_mean": post_clear_stall_signal_mean,
        "scanner_pretrigger_gate_mean": scanner_pretrigger_gate_mean,
        "scanner_pretrigger_terrain_gate_mean": scanner_pretrigger_terrain_gate_mean,
        "scanner_pretrigger_commit_gate_mean": scanner_pretrigger_commit_gate_mean,
        "scanner_pretrigger_low_cmd_gate_mean": scanner_pretrigger_low_cmd_gate_mean,
        "scanner_pretrigger_front_lift_mean": scanner_pretrigger_front_lift_mean,
        "scanner_pretrigger_uncommanded_vel_mean": scanner_pretrigger_uncommanded_vel_mean,
        "scanner_pretrigger_signal_mean": scanner_pretrigger_signal_mean,
        "scanner_pretrigger_signal_max": scanner_pretrigger_signal_max,
        "rear_approach_width_active_rate": rear_approach_width_active_rate,
        "rear_approach_width_gate_mean": rear_approach_width_gate_mean,
        "rear_approach_width_terrain_gate_mean": rear_approach_width_terrain_gate_mean,
        "rear_approach_width_commit_gate_mean": rear_approach_width_commit_gate_mean,
        "rear_approach_width_cmd_gate_mean": rear_approach_width_cmd_gate_mean,
        "rear_approach_width_mean": rear_approach_width_mean,
        "rear_approach_width_active_mean": rear_approach_width_active_mean,
        "rear_approach_min_abs_y_mean": rear_approach_min_abs_y_mean,
        "rear_approach_min_abs_y_active_mean": rear_approach_min_abs_y_active_mean,
        "rear_approach_min_abs_y_active_min": rear_approach_min_abs_y_active_min,
        "rear_approach_width_violation_rate": rear_approach_width_violation_rate,
        "rear_approach_center_violation_rate": rear_approach_center_violation_rate,
        "rear_approach_center_deficit_max": rear_approach_center_deficit_max,
        "rear_approach_width_signal_mean": rear_approach_width_signal_mean,
        "rear_approach_width_signal_max": rear_approach_width_signal_max,
        "rear_motion_width_active_rate": rear_motion_width_active_rate,
        "rear_motion_width_gate_mean": rear_motion_width_gate_mean,
        "rear_motion_width_terrain_gate_mean": rear_motion_width_terrain_gate_mean,
        "rear_motion_width_commit_gate_mean": rear_motion_width_commit_gate_mean,
        "rear_motion_width_cmd_gate_mean": rear_motion_width_cmd_gate_mean,
        "rear_motion_width_second_mean": rear_motion_width_second_mean,
        "rear_motion_width_mean": rear_motion_width_mean,
        "rear_motion_width_active_mean": rear_motion_width_active_mean,
        "rear_motion_min_abs_y_mean": rear_motion_min_abs_y_mean,
        "rear_motion_min_abs_y_active_mean": rear_motion_min_abs_y_active_mean,
        "rear_motion_min_abs_y_active_min": rear_motion_min_abs_y_active_min,
        "rear_motion_width_violation_rate": rear_motion_width_violation_rate,
        "rear_motion_center_violation_rate": rear_motion_center_violation_rate,
        "rear_motion_center_deficit_max": rear_motion_center_deficit_max,
        "rear_motion_width_signal_mean": rear_motion_width_signal_mean,
        "rear_motion_width_signal_max": rear_motion_width_signal_max,
        "support_stability_active_rate": support_stability_active_rate,
        "support_stability_gate_mean": support_stability_gate_mean,
        "support_stability_front_contact_active_mean": support_stability_front_contact_active_mean,
        "support_stability_front_slip_active_mean": support_stability_front_slip_active_mean,
        "support_stability_posture_rate_active_mean": support_stability_posture_rate_active_mean,
        "support_stability_rear_lag_active_mean": support_stability_rear_lag_active_mean,
        "support_stability_support_width_active_mean": support_stability_support_width_active_mean,
        "support_stability_signal_mean": support_stability_signal_mean,
        "support_stability_signal_max": support_stability_signal_max,
        "front_lift_guard_gate_mean": front_lift_guard_gate_mean,
        "front_lift_guard_signal_mean": front_lift_guard_signal_mean,
        "front_lift_guard_signal_max": front_lift_guard_signal_max,
        "front_lift_guard_max_lift_mean": front_lift_guard_max_lift_mean,
        "front_lift_guard_asymmetry_mean": front_lift_guard_asymmetry_mean,
        "front_lift_guard_height_excess_mean": front_lift_guard_height_excess_mean,
        "front_lift_guard_asymmetry_excess_mean": front_lift_guard_asymmetry_excess_mean,
        "front_lift_guard_terrain_gate_mean": front_lift_guard_terrain_gate_mean,
        "front_lift_guard_commit_gate_mean": front_lift_guard_commit_gate_mean,
        "fl_forward_flat_active_gate_mean": fl_forward_flat_active_gate_mean,
        "fl_forward_flat_forward_gate_mean": fl_forward_flat_forward_gate_mean,
        "fl_forward_flat_backward_gate_mean": fl_forward_flat_backward_gate_mean,
        "fl_forward_flat_lateral_gate_mean": fl_forward_flat_lateral_gate_mean,
        "fl_forward_flat_yaw_gate_mean": fl_forward_flat_yaw_gate_mean,
        "fl_forward_flat_flat_gate_mean": fl_forward_flat_flat_gate_mean,
        "fl_forward_flat_highstep_relief_mean": fl_forward_flat_highstep_relief_mean,
        "fl_forward_flat_lift_mean": fl_forward_flat_lift_mean,
        "fr_forward_flat_lift_mean": fr_forward_flat_lift_mean,
        "fl_minus_fr_forward_flat_mean": fl_minus_fr_forward_flat_mean,
        "fl_forward_flat_height_excess_mean": fl_forward_flat_height_excess_mean,
        "fl_forward_flat_asymmetry_excess_mean": fl_forward_flat_asymmetry_excess_mean,
        "fl_forward_flat_overlift_rate": fl_forward_flat_overlift_rate,
        "fl_forward_flat_penalty_signal_mean": fl_forward_flat_penalty_signal_mean,
        "fl_forward_flat_penalty_signal_max": fl_forward_flat_penalty_signal_max,
        "target_limit_violation_rate_mean": target_limit_violation_rate_mean,
        "target_limit_step_violation_rate_mean": target_limit_step_violation_rate_mean,
        "target_limit_max_delta_mean": target_limit_max_delta_mean,
        "target_limit_violation_max_consecutive_mean": target_limit_violation_max_consecutive_mean,
        "target_margin_violation_rate_mean": target_margin_violation_rate_mean,
        "target_margin_max_delta_mean": target_margin_max_delta_mean,
        "target_limit_penalty_mean": target_limit_penalty_mean,
    }


def _highstep_action_score_from_metric_values(
    metrics: dict[str, torch.Tensor],
    support_floor: float = 0.35,
    support_gate_floor: float = 0.22,
    entry_floor: float = 0.55,
    second_clear_floor: float = 0.75,
    lead_valid_floor: float = 0.35,
    support_activation_gate_target: float = 0.006,
    support_signal_target: float = 0.004,
    pre_support_cap: float = 0.42,
    support_cap_gain: float = 0.40,
    second_cap_gain: float = 0.18,
    support_quality_scale: float = 1.0,
    support_bottleneck_blend: float | torch.Tensor = 1.0,
    support_bottleneck_warmup_min_gate: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute the rebuilt high-step behavior score from branch/rear-stage metrics."""
    device = next(iter(metrics.values())).device if metrics else torch.device("cpu")
    zero = torch.tensor(0.0, device=device)
    first_score = torch.clamp(metrics.get("first_score_max", zero), 0.0, 1.0)
    second_score = torch.clamp(metrics.get("second_score_max", zero), 0.0, 1.0)
    second_clear_rate = torch.clamp(metrics.get("second_clear_rate", zero), 0.0, 1.0)
    entry_score = torch.clamp(0.45 * first_score + 0.30 * second_score + 0.25 * second_clear_rate, 0.0, 1.0)

    valid_rate = torch.clamp(metrics.get("valid_rate", zero), 0.0, 1.0)
    post_lead_gate = torch.clamp(metrics.get("post_lead_drive_gate_mean", zero), 0.0, 1.0)
    lead_support_gate = torch.clamp(metrics.get("lead_support_drive_gate_mean", zero), 0.0, 1.0)
    support_height = torch.clamp(
        torch.maximum(
            metrics.get("post_lead_drive_height_mean", zero),
            metrics.get("lead_support_drive_height_active_mean", zero),
        ),
        0.0,
        1.0,
    )
    support_second = torch.clamp(metrics.get("post_lead_drive_second_mean", zero), 0.0, 1.0)
    support_body = torch.clamp(metrics.get("lead_support_drive_body_active_mean", zero), 0.0, 1.0)
    support_signal = torch.clamp(
        torch.maximum(
            metrics.get("post_lead_drive_signal_mean", zero),
            metrics.get("lead_support_drive_signal_mean", zero),
        ),
        0.0,
        1.0,
    )
    lead_activation = torch.clamp(valid_rate / max(lead_valid_floor, 1.0e-6), 0.0, 1.0)
    drive_gate_activation = torch.clamp(
        torch.maximum(post_lead_gate, lead_support_gate) / max(support_activation_gate_target, 1.0e-6),
        0.0,
        1.0,
    )
    signal_activation = torch.clamp(support_signal / max(support_signal_target, 1.0e-6), 0.0, 1.0)
    drive_activation = torch.maximum(drive_gate_activation, signal_activation)
    support_activation = lead_activation * drive_activation
    # Progress stays diagnostic; promotion requires actual post-entry body lift/support.
    support_height_gate = torch.clamp(support_height / 0.35, 0.0, 1.0)
    support_quality = torch.clamp(
        0.68 * support_height + 0.20 * support_body + 0.12 * support_second,
        0.0,
        1.0,
    )
    support_quality_scaled = torch.clamp(support_quality / max(float(support_quality_scale), 1.0e-6), 0.0, 1.0)
    support_score = torch.clamp(
        support_quality_scaled * support_activation * support_height_gate,
        0.0,
        1.0,
    )

    one_sided = torch.clamp(metrics.get("one_sided_stall_ratio", zero), 0.0, 1.0)
    stall_signal = torch.clamp(metrics.get("post_lead_stall_signal_mean", zero), 0.0, 1.0)
    support_stability_signal = torch.clamp(metrics.get("support_stability_signal_mean", zero), 0.0, 1.0)
    support_stability_rear_lag = torch.clamp(
        metrics.get("support_stability_rear_lag_active_mean", zero),
        0.0,
        1.0,
    )
    support_stability_posture_rate = torch.clamp(
        metrics.get("support_stability_posture_rate_active_mean", zero),
        0.0,
        1.0,
    )
    support_stability_front_slip = torch.clamp(
        metrics.get("support_stability_front_slip_active_mean", zero),
        0.0,
        1.0,
    )
    support_stability_risk = torch.clamp(
        torch.maximum(
            support_stability_signal,
            0.45 * support_stability_rear_lag
            + 0.35 * support_stability_posture_rate
            + 0.20 * support_stability_front_slip,
        ),
        0.0,
        1.0,
    )
    rear_centerline_instant_risk = torch.clamp(
        torch.maximum(
            metrics.get("rear_approach_center_deficit_max", zero),
            metrics.get("rear_motion_center_deficit_max", zero),
        ),
        0.0,
        1.0,
    )
    safety_score = torch.clamp(
        1.0
        - 0.70 * one_sided
        - 0.45 * stall_signal
        - 0.30 * support_stability_risk
        - 0.70 * rear_centerline_instant_risk,
        0.0,
        1.0,
    )

    raw_total = torch.clamp(0.35 * entry_score + 0.50 * support_score + 0.15 * safety_score, 0.0, 1.0)
    entry_gate = torch.clamp(entry_score / max(entry_floor, 1.0e-6), 0.0, 1.0)
    support_gate = torch.clamp(support_score / max(support_floor, 1.0e-6), 0.0, 1.0)
    second_gate = torch.clamp(second_clear_rate / max(second_clear_floor, 1.0e-6), 0.0, 1.0)
    support_bottleneck_hard_gate = torch.clamp(
        (support_score - support_gate_floor) / max(support_floor - support_gate_floor, 1.0e-6),
        0.0,
        1.0,
    )
    bottleneck_blend = torch.as_tensor(support_bottleneck_blend, device=device).clamp(0.0, 1.0)
    relaxed_bottleneck_gate = torch.maximum(
        support_bottleneck_hard_gate,
        torch.full_like(support_bottleneck_hard_gate, float(support_bottleneck_warmup_min_gate)),
    )
    support_bottleneck_gate = relaxed_bottleneck_gate + bottleneck_blend * (
        support_bottleneck_hard_gate - relaxed_bottleneck_gate
    )
    stage_cap = torch.clamp(
        pre_support_cap + support_cap_gain * support_gate + second_cap_gain * second_gate,
        0.0,
        1.0,
    )
    centerline_gate = torch.clamp((1.0 - rear_centerline_instant_risk) / 0.75, 0.0, 1.0)
    total = torch.minimum(raw_total, stage_cap) * entry_gate * support_bottleneck_gate * centerline_gate
    hard_total = torch.minimum(raw_total, stage_cap) * entry_gate * support_bottleneck_hard_gate * centerline_gate
    support_floor_violation = (support_score < support_floor).to(dtype=total.dtype)
    return {
        "total": total,
        "hard_total": hard_total,
        "raw_total": raw_total,
        "entry_score": entry_score,
        "support_score": support_score,
        "safety_score": safety_score,
        "entry_gate": entry_gate,
        "support_gate": support_gate,
        "support_bottleneck_gate": support_bottleneck_gate,
        "support_bottleneck_hard_gate": support_bottleneck_hard_gate,
        "support_bottleneck_warmup_blend": bottleneck_blend,
        "lead_activation": lead_activation,
        "drive_activation": drive_activation,
        "support_activation": support_activation,
        "support_quality_raw": support_quality,
        "support_quality_scaled": support_quality_scaled,
        "support_quality_scale": torch.as_tensor(float(support_quality_scale), device=device),
        "second_gate": second_gate,
        "stage_cap": stage_cap,
        "bottleneck_gap": torch.clamp(raw_total - total, min=0.0),
        "hard_bottleneck_gap": torch.clamp(raw_total - hard_total, min=0.0),
        "support_floor_violation_rate": support_floor_violation,
        "support_stability_risk": support_stability_risk,
        "rear_centerline_instant_risk": rear_centerline_instant_risk,
        "rear_centerline_gate": centerline_gate,
    }


def _highstep_action_score_from_buffers(
    env: ManagerBasedRLEnv,
    ids: torch.Tensor,
    min_commit_steps: int = 1,
    support_floor: float = 0.35,
    support_gate_floor: float = 0.22,
    entry_floor: float = 0.55,
    second_clear_floor: float = 0.75,
    lead_valid_floor: float = 0.35,
    support_activation_gate_target: float = 0.006,
    support_signal_target: float = 0.004,
    support_quality_scale: float = 1.0,
    support_bottleneck_blend: float | torch.Tensor = 1.0,
    support_bottleneck_warmup_min_gate: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Read current episode buffers without resetting them and compute the action score."""
    zero = torch.tensor(0.0, device=env.device)
    if len(ids) == 0 or not hasattr(env, "_highstep_rear_branch_lead"):
        return {
            "total": zero,
            "hard_total": zero,
            "raw_total": zero,
            "entry_score": zero,
            "support_score": zero,
            "safety_score": zero,
            "entry_gate": zero,
            "support_gate": zero,
            "support_bottleneck_gate": zero,
            "support_bottleneck_hard_gate": zero,
            "support_bottleneck_warmup_blend": zero,
            "lead_activation": zero,
            "drive_activation": zero,
            "support_activation": zero,
            "second_gate": zero,
            "stage_cap": zero,
            "bottleneck_gap": zero,
            "hard_bottleneck_gap": zero,
            "support_stability_risk": zero,
            "support_floor_violation_rate": torch.ones((), device=env.device),
            "rear_centerline_gate": zero,
        }

    def _buffer(buffer_name: str) -> torch.Tensor:
        if hasattr(env, buffer_name):
            return getattr(env, buffer_name)[ids]
        return torch.zeros((len(ids),), device=env.device)

    sample_steps = torch.clamp(_buffer("_highstep_rear_branch_sample_steps"), min=1.0)
    commit_steps = _buffer("_highstep_rear_branch_commit_steps")
    active = commit_steps >= float(min_commit_steps)
    active_count = torch.clamp(active.float().sum(), min=1.0)
    valid = active & (_buffer("_highstep_rear_branch_lead").to(dtype=torch.long) >= 0)
    valid_count = torch.clamp(valid.float().sum(), min=1.0)
    valid_rate = valid.float().mean()
    second_clear_steps = _buffer("_highstep_rear_branch_second_clear_steps")
    second_clear_rate = ((second_clear_steps > 0.0) & valid).float().sum() / valid_count
    one_sided_steps = _buffer("_highstep_rear_branch_one_sided_steps")
    one_sided_stall_ratio = one_sided_steps.sum() / torch.clamp(commit_steps.sum(), min=1.0)

    def _episode_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / sample_steps).mean()

    def _episode_max_mean(buffer_name: str) -> torch.Tensor:
        return _buffer(buffer_name).mean()

    post_lead_sample_steps = torch.clamp(_buffer("_highstep_post_lead_drive_sample_steps"), min=1.0)

    def _post_lead_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / post_lead_sample_steps).mean()

    lead_support_sample_steps = torch.clamp(_buffer("_highstep_lead_support_drive_sample_steps"), min=1.0)

    def _lead_support_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / lead_support_sample_steps).mean()

    def _lead_support_active_mean(buffer_name: str) -> torch.Tensor:
        gate_weight = _buffer("_highstep_lead_support_drive_gate_sum")
        active = gate_weight > 1.0e-6
        value = _buffer(buffer_name) / torch.clamp(gate_weight, min=1.0e-6)
        return torch.where(active, value, torch.zeros_like(value)).mean()

    stall_sample_steps = torch.clamp(_buffer("_highstep_post_lead_stall_sample_steps"), min=1.0)

    def _post_lead_stall_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / stall_sample_steps).mean()

    support_stability_sample_steps = torch.clamp(_buffer("_highstep_support_stability_sample_steps"), min=1.0)
    support_stability_gate_sum = _buffer("_highstep_support_stability_gate_sum")

    def _support_stability_mean(buffer_name: str) -> torch.Tensor:
        return (_buffer(buffer_name) / support_stability_sample_steps).mean()

    def _support_stability_active_mean(buffer_name: str) -> torch.Tensor:
        active = support_stability_gate_sum > 1.0e-6
        value = _buffer(buffer_name) / torch.clamp(support_stability_gate_sum, min=1.0e-6)
        return torch.where(active, value, torch.zeros_like(value)).mean()

    metrics = {
        "first_score_max": _episode_max_mean("_highstep_rear_branch_first_score_max"),
        "second_score_max": _episode_max_mean("_highstep_rear_branch_second_score_max"),
        "valid_rate": valid_rate,
        "second_clear_rate": second_clear_rate,
        "one_sided_stall_ratio": one_sided_stall_ratio,
        "lead_support_drive_gate_mean": _lead_support_mean("_highstep_lead_support_drive_gate_sum"),
        "lead_support_drive_height_active_mean": _lead_support_active_mean(
            "_highstep_lead_support_drive_height_active_sum"
        ),
        "lead_support_drive_body_active_mean": _lead_support_active_mean(
            "_highstep_lead_support_drive_body_active_sum"
        ),
        "lead_support_drive_signal_mean": _lead_support_mean("_highstep_lead_support_drive_signal_sum"),
        "post_lead_drive_gate_mean": _post_lead_mean("_highstep_post_lead_drive_gate_sum"),
        "post_lead_drive_height_mean": _post_lead_mean("_highstep_post_lead_drive_height_sum"),
        "post_lead_drive_progress_mean": _post_lead_mean("_highstep_post_lead_drive_progress_sum"),
        "post_lead_drive_second_mean": _post_lead_mean("_highstep_post_lead_drive_second_sum"),
        "post_lead_drive_signal_mean": _post_lead_mean("_highstep_post_lead_drive_signal_sum"),
        "post_lead_drive_signal_max": _episode_max_mean("_highstep_post_lead_drive_signal_max"),
        "post_lead_stall_signal_mean": _post_lead_stall_mean("_highstep_post_lead_stall_signal_sum"),
        "support_stability_signal_mean": _support_stability_mean("_highstep_support_stability_signal_sum"),
        "support_stability_rear_lag_active_mean": _support_stability_active_mean(
            "_highstep_support_stability_rear_lag_sum"
        ),
        "support_stability_posture_rate_active_mean": _support_stability_active_mean(
            "_highstep_support_stability_posture_rate_sum"
        ),
        "support_stability_front_slip_active_mean": _support_stability_active_mean(
            "_highstep_support_stability_front_slip_sum"
        ),
        "rear_approach_center_deficit_max": _episode_max_mean(
            "_highstep_rear_approach_center_deficit_max"
        ),
        "rear_motion_center_deficit_max": _episode_max_mean(
            "_highstep_rear_motion_center_deficit_max"
        ),
    }
    del active_count  # kept in the formula while debugging, but no longer needed directly.
    return _highstep_action_score_from_metric_values(
        metrics,
        support_floor=support_floor,
        support_gate_floor=support_gate_floor,
        entry_floor=entry_floor,
        second_clear_floor=second_clear_floor,
        lead_valid_floor=lead_valid_floor,
        support_activation_gate_target=support_activation_gate_target,
        support_signal_target=support_signal_target,
        support_quality_scale=support_quality_scale,
        support_bottleneck_blend=support_bottleneck_blend,
        support_bottleneck_warmup_min_gate=support_bottleneck_warmup_min_gate,
    )


def highstep_action_score_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    min_commit_steps: int = 1,
    support_floor: float = 0.35,
    support_gate_floor: float = 0.22,
    entry_floor: float = 0.55,
    second_clear_floor: float = 0.75,
    lead_valid_floor: float = 0.35,
    support_activation_gate_target: float = 0.006,
    support_signal_target: float = 0.004,
    support_quality_scale: float = 1.0,
    support_bottleneck_start_update: int = 0,
    support_bottleneck_ramp_updates: int = 1,
    support_bottleneck_warmup_min_gate: float = 0.0,
    num_steps_per_update: int = 24,
) -> dict[str, torch.Tensor]:
    """Log branch metrics plus the behavior score used by the rebuilt task."""
    metrics = highstep_rear_branch_metrics(env, env_ids, min_commit_steps=min_commit_steps)
    update_count = _global_update(env, num_steps_per_update)
    bottleneck_blend = max(
        0.0,
        min(
            1.0,
            (update_count - float(support_bottleneck_start_update))
            / max(float(support_bottleneck_ramp_updates), 1.0),
        ),
    )
    env._highstep_schedule_update = update_count
    env._highstep_support_bottleneck_blend = bottleneck_blend
    score = _highstep_action_score_from_metric_values(
        metrics,
        support_floor=support_floor,
        support_gate_floor=support_gate_floor,
        entry_floor=entry_floor,
        second_clear_floor=second_clear_floor,
        lead_valid_floor=lead_valid_floor,
        support_activation_gate_target=support_activation_gate_target,
        support_signal_target=support_signal_target,
        support_quality_scale=support_quality_scale,
        support_bottleneck_blend=bottleneck_blend,
        support_bottleneck_warmup_min_gate=support_bottleneck_warmup_min_gate,
    )
    total = score["total"]
    hard_total = score.get("hard_total", total)
    if not hasattr(env, "_highstep_action_score_best"):
        env._highstep_action_score_best = hard_total.detach().clone()
    env._highstep_action_score_best = torch.maximum(env._highstep_action_score_best, hard_total.detach())
    score["score_drop_from_best"] = torch.clamp(env._highstep_action_score_best - hard_total, min=0.0)
    return {**metrics, **score}


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
        update_count = int(_global_update(env, num_steps_per_update))
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


def terrain_levels_vel_highstep_action_score(
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
    score_gate_required_level: int = 1,
    score_warmup_updates: int = 300,
    score_stage_update_thresholds: Sequence[int] = (450, 1000, 1800),
    action_score_thresholds: Sequence[float] = (0.16, 0.28, 0.40, 0.50),
    support_score_thresholds: Sequence[float] = (0.04, 0.12, 0.22, 0.32),
    support_floor: float = 0.35,
    support_gate_floor: float = 0.22,
    support_bottleneck_start_update: int = 0,
    support_bottleneck_ramp_updates: int = 1,
    support_bottleneck_warmup_min_gate: float = 0.0,
    regression_tolerance: float = 0.18,
    support_regression_tolerance: float = 0.12,
    entry_floor: float = 0.55,
    second_clear_floor: float = 0.75,
    lead_valid_floor: float = 0.35,
    support_activation_gate_target: float = 0.006,
    support_signal_target: float = 0.004,
    support_quality_scale: float = 1.0,
    success_hold_height_gain: float = 0.015,
    success_hold_action_score: float = 0.60,
    success_hold_support_score: float = 0.45,
) -> torch.Tensor:
    """High-step terrain curriculum gated by the rebuilt behavior score.

    Unlike terrain_levels_vel_highstep(), terrain promotion is blocked when the
    current high-step action score or post-lead support score has already fallen
    below the moving best by too much.  This prevents terrain difficulty from
    rising while the actual high-step motion regresses.
    """
    ids = _resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1)
    height_gain = asset.data.root_pos_w[ids, 2] - env.scene.env_origins[ids, 2] - nominal_base_height
    terrain_levels = getattr(terrain, "terrain_levels", None)
    update_count = int(_global_update(env, num_steps_per_update))

    allowed_max_level = None
    if stage_update_thresholds is not None and stage_max_levels is not None:
        if len(stage_max_levels) != len(stage_update_thresholds) + 1:
            raise ValueError(
                "stage_max_levels must contain one more value than stage_update_thresholds "
                "for terrain_levels_vel_highstep_action_score."
            )
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
        levels = torch.zeros_like(distance, dtype=torch.long)
        height_gate = height_gain > min_height_gain
        climb_level_gate = torch.ones_like(height_gate, dtype=torch.bool)

    distance_move_up = (distance > move_up_distance) & height_gate
    climb_move_up = (distance > climb_up_distance) & (height_gain > climb_height_gain) & climb_level_gate
    move_up = distance_move_up | climb_move_up
    if allowed_max_level is not None and terrain_levels is not None:
        move_up &= terrain_levels[ids] < allowed_max_level

    bottleneck_blend = max(
        0.0,
        min(
            1.0,
            (float(update_count) - float(support_bottleneck_start_update))
            / max(float(support_bottleneck_ramp_updates), 1.0),
        ),
    )
    score = _highstep_action_score_from_buffers(
        env,
        ids,
        min_commit_steps=1,
        support_floor=support_floor,
        support_gate_floor=support_gate_floor,
        entry_floor=entry_floor,
        second_clear_floor=second_clear_floor,
        lead_valid_floor=lead_valid_floor,
        support_activation_gate_target=support_activation_gate_target,
        support_signal_target=support_signal_target,
        support_quality_scale=support_quality_scale,
        support_bottleneck_blend=bottleneck_blend,
        support_bottleneck_warmup_min_gate=support_bottleneck_warmup_min_gate,
    )
    total_score = score["total"]
    hard_total_score = score.get("hard_total", total_score)
    support_score = score["support_score"]
    if not hasattr(env, "_highstep_action_score_curriculum_best"):
        env._highstep_action_score_curriculum_best = hard_total_score.detach().clone()
    if not hasattr(env, "_highstep_support_score_curriculum_best"):
        env._highstep_support_score_curriculum_best = support_score.detach().clone()
    env._highstep_action_score_curriculum_best = torch.maximum(
        env._highstep_action_score_curriculum_best,
        hard_total_score.detach(),
    )
    env._highstep_support_score_curriculum_best = torch.maximum(
        env._highstep_support_score_curriculum_best,
        support_score.detach(),
    )

    score_stage = 0
    for threshold in score_stage_update_thresholds:
        if update_count >= int(threshold):
            score_stage += 1
    score_stage = min(score_stage, len(action_score_thresholds) - 1, len(support_score_thresholds) - 1)
    action_threshold = torch.tensor(action_score_thresholds[score_stage], device=env.device)
    support_threshold = torch.tensor(support_score_thresholds[score_stage], device=env.device)
    regression_floor = env._highstep_action_score_curriculum_best - regression_tolerance
    support_regression_floor = env._highstep_support_score_curriculum_best - support_regression_tolerance
    score_ok = (hard_total_score >= action_threshold) & (support_score >= support_threshold)
    score_ok &= (hard_total_score >= regression_floor) & (support_score >= support_regression_floor)
    height_success_hold = (height_gain > success_hold_height_gain) & climb_level_gate
    score_success_hold = (hard_total_score >= success_hold_action_score) & (support_score >= success_hold_support_score)
    highstep_success_hold = height_success_hold | (score_success_hold & climb_level_gate)
    score_gate = (
        (update_count < int(score_warmup_updates))
        or bool(torch.all(levels < int(score_gate_required_level)).item())
        or bool(score_ok.item())
    )
    if not score_gate:
        move_up &= torch.zeros_like(move_up)

    commanded_distance = torch.norm(command[ids, :2], dim=1) * env.max_episode_length_s * move_down_command_factor
    move_down_distance = torch.clamp(
        commanded_distance,
        min=move_down_min_distance,
        max=move_down_max_distance,
    )
    climb_hold = (distance > climb_hold_distance) & (height_gain > climb_hold_height_gain) & climb_level_gate
    move_down = distance < move_down_distance
    move_down *= ~(move_up | climb_hold | highstep_success_hold)

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

    # A preserve resume restores these tensors and the live command ranges
    # before RslRlVecEnvWrapper performs its first reset.  Only initialize when
    # no checkpoint-paired command state exists; process-local step zero must
    # never overwrite a restored adaptive curriculum.
    if not hasattr(env, "_highstep_original_vel_x"):
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
