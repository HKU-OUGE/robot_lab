# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg 
import robot_lab.tasks.locomotion.velocity.mdp as mdp
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, ObservationsCfg, RewardsCfg,
)

##
# Pre-defined configs
##
# use cloud assets
# from isaaclab_assets.robots.unitree import ARCLAB_ARCDOG_CFG  # isort: skip
# use local assets
from robot_lab.assets.arclab import ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG  # isort: skip
from robot_lab.terrains.config.rough import HIGHSTEP_TERRAINS_CFG  # isort:skip


_V18_0707_STUDENT_ENV = Path(
    "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-05_00-13-46/params/env.yaml"
)
_V18_0707_STUDENT_ENV_SHA256 = (
    "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636"
)
_V18_CURRENT_ROBUST_STUDENT_ENV = Path(
    "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "historical_0707_exact_new_teacher_Student/"
    "2026-07-14_05-34-54_historical_0707_exact_E700_20260714_053449/params/env.yaml"
)
_V18_CURRENT_ROBUST_STUDENT_ENV_SHA256 = (
    "61d70655405a49ad8fd72377aed3e193b0d8435898ca1fd39316d08f1989a729"
)
_V18_0707_TEACHER_ENV = Path(
    "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-04_01-45-21/params/env.yaml"
)
_V18_0707_TEACHER_ENV_SHA256 = (
    "25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7"
)


def _v18_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v18_child(target: Any, key: str) -> Any:
    return target.get(key) if isinstance(target, dict) else getattr(target, key, None)


def _v18_disable_absent_manager_terms(cfg: Any, snapshot: Mapping[str, Any]) -> None:
    """Disable current manager terms absent from the hash-frozen snapshot."""
    current = cfg.to_dict()
    for root in ("actions", "events", "rewards", "terminations", "curriculum", "commands", "recorders"):
        target = getattr(cfg, root, None)
        saved = snapshot.get(root, {})
        if target is None or not isinstance(saved, Mapping):
            continue
        for name in current.get(root, {}):
            if name not in saved and hasattr(target, name):
                setattr(target, name, None)
    saved_groups = snapshot.get("observations", {})
    current_groups = current.get("observations", {})
    for group_name, terms in current_groups.items():
        group = getattr(cfg.observations, group_name, None)
        if group_name not in saved_groups:
            if hasattr(cfg.observations, group_name):
                setattr(cfg.observations, group_name, None)
            continue
        if group is None or not isinstance(terms, Mapping):
            continue
        for term_name in terms:
            if term_name not in saved_groups[group_name] and hasattr(group, term_name):
                setattr(group, term_name, None)


def _v18_prune_saved_mapping_extras(target: Any, source: Any) -> None:
    """Remove post-0707 keys from mutable parameter dictionaries only."""
    if not isinstance(source, Mapping) or target is None:
        return
    if isinstance(target, dict):
        for key in list(target):
            if key not in source:
                del target[key]
        for key, value in source.items():
            if key in target:
                _v18_prune_saved_mapping_extras(target[key], value)
        return
    for key, value in source.items():
        _v18_prune_saved_mapping_extras(_v18_child(target, key), value)


def _v18_materialize_saved_optional_values(target: Any, source: Any) -> None:
    """Materialize saved values where configclass currently stores ``None``.

    Isaac Lab's ``from_dict`` validates against the current runtime value type,
    so a saved integer such as ``seed`` cannot replace a default ``None`` until
    the optional slot has been materialized explicitly.
    """
    if not isinstance(source, Mapping) or target is None:
        return
    if isinstance(target, dict):
        for key, value in source.items():
            if key in target and target[key] is None and value is not None:
                target[key] = value
            elif key in target:
                _v18_materialize_saved_optional_values(target[key], value)
        return
    for key, value in source.items():
        child = _v18_child(target, key)
        if child is None and value is not None and hasattr(target, key):
            setattr(target, key, value)
        else:
            _v18_materialize_saved_optional_values(child, value)


def _v18_projection_mismatches(runtime: Any, snapshot: Any, path: str = "") -> list[str]:
    mismatches: list[str] = []
    if isinstance(snapshot, Mapping):
        if not isinstance(runtime, Mapping):
            return [path or "<root>"]
        for key, value in snapshot.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in runtime:
                mismatches.append(child_path)
            else:
                mismatches.extend(_v18_projection_mismatches(runtime[key], value, child_path))
    elif runtime != snapshot:
        mismatches.append(path)
    return mismatches


def _apply_v18_saved_environment_profile(cfg: Any, path: Path, expected_sha256: str) -> None:
    """Mechanically restore a trusted saved env snapshot and fail closed on drift."""
    path = path.resolve(strict=True)
    actual_sha = _v18_sha256(path)
    if actual_sha != expected_sha256:
        raise RuntimeError(
            f"v1.8 environment snapshot SHA mismatch: {path}: {actual_sha} != {expected_sha256}"
        )
    snapshot = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("v1.8 environment snapshot is not a mapping")
    _v18_disable_absent_manager_terms(cfg, snapshot)
    _v18_prune_saved_mapping_extras(cfg, snapshot)
    _v18_materialize_saved_optional_values(cfg, snapshot)
    cfg.from_dict(dict(snapshot))
    runtime = cfg.to_dict()
    mismatches = _v18_projection_mismatches(runtime, snapshot)
    if mismatches:
        raise RuntimeError(
            "v1.8 saved environment projection mismatch at: " + ", ".join(mismatches[:12])
        )


def v18_environment_profile_contract(task_name: str) -> dict[str, str]:
    """Return the exact source profile binding for manifests and launch audits."""
    profiles = {
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-ArcdogAdjustableLeg-v0": (
            "stage_a_0707_student_bootstrap", _V18_0707_STUDENT_ENV, _V18_0707_STUDENT_ENV_SHA256
        ),
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Robust-ArcdogAdjustableLeg-v0": (
            "stage_b_current_robust", _V18_CURRENT_ROBUST_STUDENT_ENV,
            _V18_CURRENT_ROBUST_STUDENT_ENV_SHA256
        ),
        "RobotLab-Isaac-Velocity-HighstepActionScoreTeacherV18Bootstrap-ArcdogAdjustableLeg-v0": (
            "teacher_0707_bootstrap", _V18_0707_TEACHER_ENV, _V18_0707_TEACHER_ENV_SHA256
        ),
    }
    if task_name not in profiles:
        raise KeyError(f"unknown v1.8 environment profile task: {task_name}")
    name, path, digest = profiles[task_name]
    return {"profile": name, "source_env_yaml": str(path), "source_env_yaml_sha256": digest}


@configclass
class ArcdogAdjustableLegHighstepRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    rotate_joint_pos_penalty = RewTerm(
        func=mdp.joint_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh_joint|calf_joint)$"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.3,
        },
    )

    prismatic_joint_pos_penalty  = RewTerm(
        func=mdp.highstep_box_default_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "min_cmd_x": 0.10,
            "command_gate_width": 0.25,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "hold_scale": 16.0,
            "highstep_scale": 0.08,
        },
    )

    stand_still_flat = RewTerm(
        func=mdp.stand_still_flat_orientation_bonus,
        weight=0.0,  # 默认权重设为0，在 EnvCfg 中具体配置
        params={
            "command_name": "base_velocity",
            "std": 0.1,               # 控制对倾斜的敏感度，越小越严格
            "command_threshold": 0.1,  # 速度指令小于此值视为静止
            "ignore_yaw_command": False,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    moving_flat = RewTerm(
        func=mdp.moving_flat_orientation_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.16,
            "command_threshold": 0.12,
            "command_max": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    blind_climbing_bonus = RewTerm(
        func=mdp.blind_climbing_vel_z_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "pitch_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    pitch_up_on_obstacle = RewTerm(
        func=mdp.climbing_pitch_up_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_cmd_x": 0.25,
            "min_actual_vel_x": 0.02,
            "max_actual_vel_x": 0.35,
            "min_pitch_metric": 0.03,
        },
    )

    front_legs_reach = RewTerm(
        func=mdp.front_legs_highstep_reach_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "min_pitch_metric": 0.02,
            "min_height_diff": 0.05,
            "target_height_diff": 0.30,
            "relative_lift_min": -0.34,
            "relative_lift_target": -0.08,
            "min_front_x": 0.20,
            "target_front_x": 0.45,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "terrain_gate_floor": 0.25,
        },
    )

    front_feet_highstep_clearance = RewTerm(
        func=mdp.front_feet_highstep_clearance_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.03,
            "clearance_window": 0.12,
        },
    )

    rear_feet_highstep_clearance = RewTerm(
        func=mdp.rear_feet_highstep_clearance_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.045,
            "clearance_window": 0.18,
            "commit_gate_scale": 0.80,
            "single_rear_weight": 0.20,
            "min_rear_weight": 0.50,
            "single_rear_temperature": 0.12,
            "rear_x_min": -0.48,
            "rear_x_target": -0.08,
            "rear_x_weight": 0.05,
            "rear_x_single_weight": 1.0,
        },
    )

    rear_first_foot_highstep_preclearance = RewTerm(
        func=mdp.rear_first_foot_highstep_preclearance_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.06,
            "clearance_window": 0.16,
            "first_clear_min": 0.55,
            "second_clear_suppress_min": 0.72,
            "commit_gate_min": 0.25,
            "commit_gate_scale": 0.90,
            "max_roll_metric": 0.28,
        },
    )

    rear_feet_under_step_after_commit = RewTerm(
        func=mdp.rear_feet_under_step_after_commit_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.02,
            "under_window": 0.16,
            "commit_gate_floor": 0.25,
            "nominal_base_height": 0.44,
            "min_distance": 0.16,
            "target_distance": 0.56,
            "min_height_gain": -0.02,
            "target_height_gain": 0.08,
            "min_forward_vel": 0.06,
            "relief_forward_vel": 0.22,
            "stall_floor": 0.05,
            "worst_rear_weight": 0.82,
        },
    )

    rear_second_foot_highstep_clearance = RewTerm(
        func=mdp.rear_second_foot_highstep_clearance_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "first_rear_gate_min": 0.55,
            "commit_gate_scale": 0.90,
            "nominal_base_height": 0.44,
            "min_distance": 0.18,
            "target_distance": 0.60,
            "min_height_gain": -0.02,
            "target_height_gain": 0.08,
            "progress_floor": 0.20,
            "max_roll_metric": 0.24,
            "branch_lead_threshold": 0.62,
            "branch_second_clear_threshold": 0.72,
            "branch_one_sided_gap": 0.22,
        },
    )

    second_rear_clear_deadline = RewTerm(
        func=mdp.second_rear_clear_deadline_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "second_clear_min": 0.62,
            "deadline_steps": 32.0,
            "deadline_grace_steps": 6.0,
            "commit_gate_scale": 0.90,
            "max_roll_metric": 0.28,
        },
    )

    one_sided_rear_stall_time = RewTerm(
        func=mdp.one_sided_rear_stall_time_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "lead_threshold": 0.45,
            "one_sided_gap": 0.16,
            "grace_steps": 12.0,
            "ramp_steps": 24.0,
            "commit_gate_scale": 0.90,
        },
    )

    lead_rear_support_drive = RewTerm(
        func=mdp.lead_rear_support_drive_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "box_joint_names": {
                "FL": "FL_box_joint",
                "FR": "FR_box_joint",
                "RL": "RL_box_joint",
                "RR": "RR_box_joint",
            },
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "rear_push_target": 0.003,
            "target_std": 0.016,
            "lead_box_floor": 0.25,
            "nominal_base_height": 0.44,
            "min_distance": 0.20,
            "target_distance": 0.62,
            "min_height_gain": -0.02,
            "target_height_gain": 0.09,
            "target_forward_vel": 0.28,
            "body_drive_floor": 0.10,
            "commit_gate_scale": 0.90,
        },
    )

    post_lead_body_drive = RewTerm(
        func=mdp.post_lead_body_drive_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "lead_score_min": 0.58,
            "second_clear_min": 0.58,
            "min_elapsed_steps": 1.0,
            "elapsed_ramp_steps": 6.0,
            "deadline_steps": 38.0,
            "deadline_grace_steps": 8.0,
            "nominal_base_height": 0.44,
            "min_distance": 0.20,
            "target_distance": 0.64,
            "min_height_gain": -0.01,
            "target_height_gain": 0.10,
            "target_forward_vel": 0.28,
            "progress_floor": 0.25,
            "second_floor": 0.35,
            "body_height_weight": 0.50,
            "body_progress_weight": 0.35,
            "second_clear_weight": 0.15,
            "max_roll_metric": 0.30,
            "commit_gate_scale": 0.85,
        },
    )

    post_lead_stall_penalty = RewTerm(
        func=mdp.post_lead_stall_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "lead_score_min": 0.55,
            "second_low_threshold": 0.58,
            "one_sided_gap": 0.14,
            "grace_steps": 10.0,
            "ramp_steps": 26.0,
            "target_progress": 0.45,
            "nominal_base_height": 0.44,
            "min_distance": 0.20,
            "target_distance": 0.62,
            "min_height_gain": -0.01,
            "target_height_gain": 0.09,
            "target_forward_vel": 0.24,
            "max_roll_metric": 0.35,
            "commit_gate_scale": 0.85,
        },
    )

    post_clear_recovery = RewTerm(
        func=mdp.post_clear_recovery_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "second_clear_min": 0.68,
            "min_elapsed_steps": 8.0,
            "elapsed_ramp_steps": 10.0,
            "min_base_clearance": 0.34,
            "target_base_clearance": 0.43,
            "target_forward_vel": 0.22,
            "platform_half_width": 1.5,
            "min_rear_margin": 0.04,
            "target_rear_margin": 0.18,
            "max_abs_pitch_metric": 0.24,
            "max_abs_roll_metric": 0.24,
            "posture_weight": 0.45,
            "base_clearance_weight": 0.35,
            "forward_weight": 0.20,
            "rear_advance_weight": 0.35,
            "post_clear_latch_scale": 1.0,
            "commit_gate_scale": 0.85,
        },
    )

    post_clear_rear_advance_stall = RewTerm(
        func=mdp.post_clear_rear_advance_stall_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "platform_half_width": 1.5,
            "min_rear_margin": 0.04,
            "near_edge_margin": 0.12,
            "target_rear_margin": 0.18,
            "grace_steps": 8.0,
            "ramp_steps": 32.0,
        },
    )

    scanner_pretrigger_penalty = RewTerm(
        func=mdp.scanner_pretrigger_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "terrain_gate_min": 0.20,
            "commit_gate_cutoff": 0.22,
            "low_cmd_threshold": 0.12,
            "front_lift_limit": -0.14,
            "front_lift_window": 0.14,
            "forward_vel_margin": 0.10,
            "forward_vel_window": 0.25,
            "front_lift_weight": 0.55,
            "uncommanded_vel_weight": 0.45,
        },
    )

    rear_approach_width_penalty = RewTerm(
        func=mdp.rear_approach_width_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.06,
            "cmd_gate_width": 0.18,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "terrain_gate_min": 0.08,
            "commit_gate_cutoff": 0.28,
            "min_rear_width": 0.26,
            "width_window": 0.08,
            "min_rear_abs_y": 0.11,
            "center_window": 0.05,
            "width_weight": 0.65,
            "center_weight": 0.35,
            "critical_center_boost": 0.0,
            "critical_min_rear_abs_y": 0.075,
            "critical_center_window": 0.04,
        },
    )

    rear_highstep_motion_width_penalty = RewTerm(
        func=mdp.rear_highstep_motion_width_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.04,
            "cmd_gate_width": 0.18,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "terrain_gate_min": 0.05,
            "commit_gate_floor": 0.25,
            "min_rear_width": 0.255,
            "width_window": 0.08,
            "min_rear_abs_y": 0.105,
            "center_window": 0.05,
            "width_weight": 0.70,
            "center_weight": 0.30,
            "critical_center_boost": 0.0,
            "critical_min_rear_abs_y": 0.075,
            "critical_center_window": 0.04,
            "post_clear_relief": 0.45,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
        },
    )

    highstep_support_stability_penalty = RewTerm(
        func=mdp.highstep_support_stability_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            ),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.06,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "commit_gate_floor": 0.18,
            "contact_threshold": 4.0,
            "contact_force_window": 32.0,
            "front_slip_deadband": 0.08,
            "front_slip_window": 0.24,
            "roll_deadband": 0.16,
            "roll_window": 0.20,
            "roll_rate_deadband": 0.45,
            "roll_rate_window": 1.10,
            "yaw_rate_deadband": 0.18,
            "yaw_rate_window": 0.65,
            "lead_score_min": 0.44,
            "second_clear_min": 0.62,
            "rear_lag_grace_steps": 10.0,
            "rear_lag_ramp_steps": 30.0,
            "one_sided_gap": 0.18,
            "min_support_width": 0.25,
            "support_width_window": 0.10,
            "min_support_abs_y": 0.09,
            "support_center_window": 0.06,
            "front_support_grace_steps": 8.0,
            "front_support_ramp_steps": 16.0,
            "front_contact_weight": 0.25,
            "front_slip_weight": 0.18,
            "posture_rate_weight": 0.32,
            "rear_lag_weight": 0.18,
            "support_width_weight": 0.07,
        },
    )

    highstep_forward_progress = RewTerm(
        func=mdp.highstep_forward_progress_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "min_cmd_x": 0.08,
            "target_forward_vel": 0.35,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "commit_gate_scale": 0.70,
        },
    )

    highstep_body_lift = RewTerm(
        func=mdp.highstep_body_lift_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "nominal_base_height": 0.44,
            "min_lift": -0.04,
            "target_lift": 0.08,
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "commit_gate_scale": 0.70,
        },
    )

    highstep_base_advance_lift = RewTerm(
        func=mdp.highstep_base_advance_lift_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "nominal_base_height": 0.44,
            "min_distance": 0.18,
            "target_distance": 0.60,
            "min_height_gain": -0.02,
            "target_height_gain": 0.08,
            "distance_floor": 0.35,
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "commit_gate_scale": 0.90,
        },
    )

    base_height_floor_penalty = RewTerm(
        func=mdp.base_height_floor_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "min_height": 0.30,
        },
    )

    highstep_leg_support_contact = RewTerm(
        func=mdp.highstep_leg_support_contact_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", ".*_box"]),
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "contact_threshold": 2.0,
            "contact_force_window": 30.0,
            "max_contact_score": 3.0,
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "commit_gate_scale": 0.75,
            "support_body_names": [
                "FL_.*thigh.*",
                "FR_.*thigh.*",
                "FL_.*calf.*",
                "FR_.*calf.*",
                "FL_.*box.*",
                "FR_.*box.*",
            ],
            "support_pose_scale": 0.55,
            "support_x_min": 0.02,
            "support_x_target": 0.24,
            "support_height_margin": 0.10,
            "support_height_window": 0.18,
            "support_phase_floor": 0.85,
        },
    )

    non_forward_highstep_pitch = RewTerm(
        func=mdp.non_forward_highstep_pitch_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "forward_cmd_threshold": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "pitch_deadband": 0.12,
        },
    )

    backward_pitch_stability = RewTerm(
        func=mdp.backward_motion_pitch_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_backward_cmd": 0.05,
            "full_backward_cmd": 0.22,
            "pitch_deadband": 0.08,
            "pitch_limit": 0.32,
            "height_soft_limit": 0.56,
            "height_limit": 0.72,
            "height_weight": 0.40,
        },
    )

    horse_rearing_bonus = RewTerm(
        func=mdp.horse_rearing_posture_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "target_pitch_deg": 35.0,
            "target_height_diff": 0.35,
            "min_front_x": 0.18,
            "target_front_x": 0.42,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "terrain_gate_floor": 0.25,
        },
    )

    front_legs_quiet_penalty = RewTerm(
        func=mdp.front_legs_quiet_on_step_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "front_knee_names": ["FL_calf_joint", "FR_calf_joint"],
            "step_height_threshold": 0.30,
        },
    )

    front_legs_lift_guard_penalty = RewTerm(
        func=mdp.front_legs_lift_guard_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "relative_lift_limit": -0.14,
            "relative_lift_window": 0.14,
            "asymmetry_limit": 0.16,
            "asymmetry_window": 0.16,
            "terrain_relief_scale": 1.0,
            "commit_relief_scale": 0.90,
            "height_weight": 0.60,
            "asymmetry_weight": 0.40,
            "min_command_norm": 0.02,
            "command_gate_width": 0.18,
            "stage_start_update": 0,
            "stage_ramp_updates": 1,
            "num_steps_per_update": 24,
        },
    )

    forward_flat_left_front_lift_penalty = RewTerm(
        func=mdp.forward_flat_left_front_lift_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "left_front_foot_name": "FL_foot",
            "right_front_foot_name": "FR_foot",
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "min_forward_cmd": 0.08,
            "forward_gate_width": 0.18,
            "max_side_cmd": 0.12,
            "side_gate_width": 0.18,
            "max_yaw_cmd": 0.16,
            "yaw_gate_width": 0.22,
            "terrain_gate_cutoff": 0.04,
            "commit_gate_cutoff": 0.035,
            "fl_lift_limit": -0.24,
            "fl_lift_window": 0.08,
            "fl_over_fr_limit": 0.04,
            "fl_over_fr_window": 0.08,
            "height_weight": 0.40,
            "asymmetry_weight": 0.60,
            "stage_start_update": 0,
            "stage_ramp_updates": 1,
            "num_steps_per_update": 24,
        },
    )

    rear_legs_drive_bonus = RewTerm(
        func=mdp.rear_legs_power_drive_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "rear_drive_joint_names": ["RL_thigh_joint", "RR_thigh_joint", "RL_calf_joint", "RR_calf_joint"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "nominal_base_height": 0.44,
            "min_body_lift": -0.10,
            "target_positive_power": 50.0,
            "max_power_score": 1.0,
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "commit_gate_scale": 0.75,
            "lift_gate_floor": 0.30,
            "use_abs_power": True,
        },
    )

    highstep_rear_push_posture = RewTerm(
        func=mdp.highstep_rear_push_posture_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "nominal_base_height": 0.44,
            "rear_back_min": 0.18,
            "rear_back_target": 0.46,
            "min_front_rear_height_diff": 0.08,
            "target_front_rear_height_diff": 0.28,
            "min_distance": 0.22,
            "target_distance": 0.65,
            "min_height_gain": -0.02,
            "target_height_gain": 0.08,
            "commit_gate_floor": 0.30,
            "progress_floor": 0.20,
        },
    )

    highstep_rear_box_push = RewTerm(
        func=mdp.highstep_rear_box_push_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "box_joint_names": {
                "FL": "FL_box_joint",
                "FR": "FR_box_joint",
                "RL": "RL_box_joint",
                "RR": "RR_box_joint",
            },
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.05,
            "front_x_min": 0.15,
            "rear_x_max": -0.20,
            "max_abs_y": 0.35,
            "height_threshold": 0.035,
            "height_gate_width": 0.12,
            "nominal_base_height": 0.44,
            "rear_push_target": 0.004,
            "target_std": 0.012,
            "min_distance": 0.18,
            "target_distance": 0.60,
            "min_height_gain": -0.02,
            "target_height_gain": 0.08,
            "commit_gate_floor": 0.35,
            "progress_floor": 0.30,
        },
    )

    highstep_bridge_stall_penalty = RewTerm(
        func=mdp.highstep_bridge_stall_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "nominal_base_height": 0.44,
            "rear_back_min": 0.18,
            "rear_back_target": 0.46,
            "min_distance": 0.20,
            "target_distance": 0.58,
            "min_height_gain": -0.02,
            "target_height_gain": 0.07,
            "commit_gate_floor": 0.30,
        },
    )

    highstep_box_phase_prior = RewTerm(
        func=mdp.highstep_box_phase_prior_alignment_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "box_joint_names": {
                "FL": "FL_box_joint",
                "FR": "FR_box_joint",
                "RL": "RL_box_joint",
                "RR": "RR_box_joint",
            },
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.05,
            "front_x_min": 0.15,
            "rear_x_max": -0.20,
            "max_abs_y": 0.35,
            "height_threshold": 0.035,
            "height_gate_width": 0.12,
            "commit_gate_floor": 0.25,
            "front_reach_target": 0.014,
            "rear_approach_target": 0.034,
            "front_support_target": 0.034,
            "rear_push_target": 0.012,
            "target_std": 0.015,
        },
    )

    roll_yaw_orientation_penalty = RewTerm(
        func=mdp.roll_yaw_orientation_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 惩罚伸缩腿的剧烈加速度 (震荡的主要特征)
    box_joint_acc_penalty = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0, # 在 EnvCfg 中激活
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )
    
    # 针对伸缩腿的关节速度惩罚
    box_joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,  # 使用关节速度，它支持 asset_cfg
        weight=-0.01,           # 权重建议：从 -0.01 到 -0.05 开始尝试，太大会导致腿动不了
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )

    box_joint_action_rate = RewTerm(
        func=mdp.action_rate_l2_by_name_command_scale,
        weight=-0.0, 
        params={
            # 使用正则表达式匹配你的伸缩关节，比如包含 "box_joint" 的所有关节
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*box_joint.*"),
            "command_name": "base_velocity",
            "stand_still_scale": 1.0,
            "moving_scale": 0.5,
            "command_threshold": 0.12,
            "ignore_yaw_command": False,
        }
    )

    # 针对伸缩腿的专属限位惩罚
    box_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,  # 复用同一个底层函数
        weight=0.0,                 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )


    # stand_still_joint_vel = RewTerm(
    #     func=mdp.stand_still_joint_vel_penalty, # 调用我们刚才写的自定义函数
    #     weight=0.0,  # ！！！权重建议从 -0.5 开始尝试。如果还晃，可以加大到 -1.0 甚至 -2.0
    #     params={
    #         "command_name": "base_velocity", # 确保这里的名字和你的指令管理器中一致
    #         "command_threshold": 0.1,        # 只有指令速度 < 0.1 时才惩罚
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    stand_still_revolute_joint_vel = RewTerm(
        func=mdp.stand_still_joint_vel_penalty, # 替换为你的实际路径
        weight=0.0,  # 针对 rad/s 的权重，数值通常较大，权重可以适中
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            # 使用正则表达式匹配所有的 hip, thigh, calf 关节 (例如 FL_hip_joint, FR_thigh_joint 等)
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*hip_joint.*", ".*thigh_joint.*", ".*calf_joint.*"]
            ),
        },
    )

    stand_still_prismatic_joint_vel = RewTerm(
        func=mdp.stand_still_joint_vel_penalty, # 替换为你的实际路径
        weight=0.0,  # ！！！注意：直线速度(m/s)的数值通常比角速度(rad/s)小得多，因此可能需要更大的负权重
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            # 匹配 box_joint
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*box_joint.*"]
            ),
        },
    )

    stand_still_base_ang_vel = RewTerm(
        func=mdp.stand_still_base_ang_vel_penalty,
        weight=0.0,  # 权重可以从 -0.5 到 -2.0 尝试
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    stand_still_base_lin_vel = RewTerm(
        func=mdp.stand_still_base_lin_vel_penalty,
        weight=0.0,  
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    # 新增：倾斜自适应的高度惩罚
    base_height_relaxed = RewTerm(
        func=mdp.base_height_l2_relaxed_on_tilt, # 指向刚才写的新函数
        weight=0.0, # 默认 0，在 EnvCfg 中激活
        params={
            "target_height": 0.44,
            "tilt_sensitivity": 2.0, # 建议设为 1.0 到 3.0 之间
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    # =====================================================================
    # 新增：足端间距惩罚 (解决内八字/走钢丝步态)
    # =====================================================================
    feet_stance_width = RewTerm(
        func=mdp.feet_stance_width_penalty,
        weight=0.0, # 在 EnvCfg 中激活
        params={
            "min_width": 0.31, # 期望的最小足端横向距离（米）。请根据你机器人的实际肩宽进行调整！
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # # =====================================================================
    # # 进阶自适应足端间距惩罚 (全参数化)
    # # =====================================================================
    # feet_stance_width_advanced_adaptive = RewTerm(
    #     func=mdp.feet_stance_width_advanced_adaptive_penalty, 
    #     weight=0.0, 
    #     params={
    #         # 距离阈值参数
    #         "flat_min_width": 0.31,               # 【平地】期望的最小足端横向距离
    #         "rough_stationary_min_width": 0.25,   # 【上地形+静止】保证站立不倒的最小距离
    #         "rough_moving_min_width": 0.12,       # 【上地形+运动】极度放宽，允许走钢丝/避障步态
            
    #         # 速度判定参数
    #         "stationary_speed_threshold": 0.10,   # 速度低于 0.10m/s 视为完全静止
    #         "moving_speed_threshold": 0.30,       # 速度高于 0.30m/s 视为完全运动
            
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #     },
    # )

    feet_stance_width = RewTerm(
        func=mdp.feet_stance_width_adaptive_penalty, 
        weight= 0.0,  # 建议保持在 -1.0 到 -2.0 之间
        params={
            "min_width": 0.32,               
            "command_speed_threshold": 0.25,  # 【关键】阈值调小！指令速度低于 0.15m/s 就视为“准备静止”，开始张开腿
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    anti_pronk_contact = RewTerm(
        func=mdp.anti_pronk_contact_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "front_foot_names": ("FL_foot", "FR_foot"),
            "rear_foot_names": ("RL_foot", "RR_foot"),
            "command_threshold": 0.08,
            "velocity_threshold": 0.08,
            "contact_threshold": 1.0,
            "all_air_weight": 1.0,
            "all_contact_weight": 0.4,
            "same_side_pair_weight": 0.6,
            "include_yaw_command": True,
        },
    )


@configclass
class ArcdogAdjustableLegHighstepActionScoreRewardsCfg(ArcdogAdjustableLegHighstepRewardsCfg):
    """Extra terms for the rebuilt high-step task.

    The old highstep task keeps its original reward surface.  This derived
    reward cfg adds a single composite behavior score so the new task optimizes
    the same stages that are inspected during play.
    """

    highstep_action_score = RewTerm(
        func=mdp.highstep_action_score_bonus,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "front_foot_names": ["FL_foot", "FR_foot"],
            "rear_foot_names": ["RL_foot", "RR_foot"],
            "min_cmd_x": 0.08,
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "clearance_margin": 0.04,
            "clearance_window": 0.18,
            "first_clear_min": 0.54,
            "second_clear_min": 0.58,
            "lead_score_min": 0.52,
            "one_sided_gap": 0.16,
            "nominal_base_height": 0.44,
            "min_distance": 0.20,
            "target_distance": 0.64,
            "min_height_gain": -0.01,
            "target_height_gain": 0.11,
            "target_forward_vel": 0.24,
            "entry_weight": 0.35,
            "support_weight": 0.50,
            "safety_weight": 0.15,
            "entry_floor": 0.55,
            "support_floor": 0.45,
            "second_floor": 0.45,
            "pre_support_cap": 0.42,
            "support_cap_gain": 0.40,
            "second_cap_gain": 0.18,
            "max_roll_metric": 0.34,
            "commit_gate_scale": 0.85,
            "stage_start_update": 0,
            "stage_ramp_updates": 1,
            "num_steps_per_update": 24,
        },
    )


@configclass
class ArclabArcdogAdjustableLegHighstepEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ArcdogAdjustableLegHighstepRewardsCfg = ArcdogAdjustableLegHighstepRewardsCfg()


    base_link_name = "base"
    trunk_link_name = "trunk"
    hip_link_name = ".*_thigh"
    knee_link_name = ".*_calf"
    abad_link_name = ".*_hip"
    foot_link_name = ".*_foot"
    extension_link_name = ".*_box"

    # fmt: off
    joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint",
        "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint",
        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree a1
        self.scene.robot = ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator = HIGHSTEP_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 1
        self.events.randomize_reset_base.func = mdp.reset_root_state_highstep_approach
        self.events.randomize_reset_base.params.update(
            {
                "flat_patch_key": "target",
                "high_origin_threshold": 0.035,
                "low_patch_z_margin": 0.04,
                # Delivery route: keep dense near-step starts to prioritize
                # high-step climbing over flat-gait cleanup.
                "approach_ratio": 0.88,
                "approach_distance_range": (1.75, 2.45),
                "approach_yaw_noise_range": (-0.16, 0.16),
            }
        )

         # [新增] 解决打滑问题：修改地形的物理材质属性
        self.scene.terrain.physics_material.friction_combine_mode = "average"    # 避免 multiply 导致极小值
        self.scene.terrain.physics_material.restitution_combine_mode = "average"
        self.scene.terrain.physics_material.static_friction = 1.2               # 提高基础静摩擦力
        self.scene.terrain.physics_material.dynamic_friction = 1.2               # 提高基础动摩擦力

        # ------------------------------Observations------------------------------
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 因为在你的 Base Cfg 中，policy 组里已经没有这两个属性了，
        # 所以不需要（也不能）再在这里把它们设为 None 来禁用。
        # self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None

        # ==========================================
        # 1. 更新 Policy 组的关节名称 (直接覆写 params 字典，彻底避免 KeyError)
        # ==========================================
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
        }

        # ==========================================
        # 2. 更新 Estimator 组的关节名称 (VAE 专用的历史组，加入 hasattr 保护)
        # ==========================================
        if hasattr(self.observations, "estimator") and self.observations.estimator is not None:
            self.observations.estimator.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }
            self.observations.estimator.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }
            # ==========================================
            # 【核心修改】严格对齐 Estimator 和 Policy 的 Scale！
            # 解决 Sim-to-Sim 中 VAE 接收到缩小 20 倍的速度导致动作发疯的 BUG
            # ==========================================
            self.observations.estimator.base_ang_vel.scale = 0.25
            self.observations.estimator.joint_vel.scale = 0.05

        # ==========================================
        # 3. 更新 Critic 组的关节名称 (加入 hasattr 保护)
        # ==========================================
        if hasattr(self.observations, "critic") and self.observations.critic is not None:
            self.observations.critic.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }
            self.observations.critic.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }

        # self.observations.policy.base_lin_vel.scale = 2.0
        # self.observations.policy.base_ang_vel.scale = 0.25
        # self.observations.policy.joint_pos.scale = 1.0
        # self.observations.policy.joint_vel.scale = 0.05
        # self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
        #     self.joint_names
        # )
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
        #     self.joint_names
        # )

        # ------------------------------Actions------------------------------
        # # 强制将 Action 的零位对齐到 0.075，这样网络输出 0 时，腿保持在 0.075
        # self.scene.robot.default_joint_angles = {
        #     "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, 
        #     "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
        #     "FL_thigh_joint": 0.6, "FR_thigh_joint": 0.6, 
        #     "RL_thigh_joint": 0.6, "RR_thigh_joint": 0.6,
        #     "FL_calf_joint": -0.95, "FR_calf_joint": -0.95, 
        #     "RL_calf_joint": -0.95, "RR_calf_joint": -0.95,
        #     # 关键：这里必须与 init_state 一致
        #     "FL_box_joint": 0.1, "FR_box_joint": 0.1, 
        #     "RL_box_joint": 0.1, "RR_box_joint": 0.1,
        # }
        
        self.actions.joint_pos = mdp.PhasedHighstepBoxBiasJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.joint_names,
            scale={
                ".*_box_joint": 0.02,
                ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
            },
            use_default_offset=True,
            clip={".*": (-60.0, 60.0)},
            preserve_order=True,
            front_x_min=0.25,
            rear_x_max=-0.20,
            max_abs_y=0.30,
            height_threshold=0.060,
            height_gate_width=0.14,
            min_forward_command=0.10,
            command_gate_width=0.25,
            commit_height_delta_min=0.04,
            commit_height_delta_target=0.18,
            commit_gate_floor=0.0,
            front_reach_box_bias=-0.020,
            rear_approach_box_bias=0.002,
            front_support_box_bias=0.002,
            rear_push_box_bias=-0.022,
            min_box_target=0.000,
            max_box_target=0.060,
            prior_start_update=300,
            prior_full_update=900,
            num_steps_per_update=24,
        )

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_apply_external_force_torque.params[
            "asset_cfg"
        ].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = [".*"]
        self.events.randomize_joint_friction.params["asset_cfg"].joint_names = [".*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.8, 1.5)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.8, 1.5)
        
        self.events.randomize_screw_joints.params["asset_cfg"].joint_names = [".*_box_joint"]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -20

        # High-step climbing needs the old 2026-03-17 style pitch-and-drive motion.
        # Keep roll/yaw controlled, but do not heavily punish vertical lift or body pitch.
        self.rewards.lin_vel_z_l2.weight = -0.05
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.roll_yaw_orientation_penalty.weight = -0.5
        self.rewards.base_height_l2.weight = -0.8
        self.rewards.base_height_l2.params["target_height"] = 0.44
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.rewards.stand_still_flat.weight = 0.0
        self.rewards.moving_flat.weight = 0.0
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.rewards.blind_climbing_bonus.weight = 0.0
        self.rewards.pitch_up_on_obstacle.weight = 0.05
        self.rewards.horse_rearing_bonus.weight = 0.05
        self.rewards.front_legs_reach.weight = 0.55
        self.rewards.front_feet_highstep_clearance.weight = 0.25
        self.rewards.rear_feet_highstep_clearance.weight = 1.05
        self.rewards.rear_feet_highstep_clearance.params["clearance_margin"] = 0.04
        self.rewards.rear_feet_highstep_clearance.params["single_rear_weight"] = 0.40
        self.rewards.rear_feet_highstep_clearance.params["min_rear_weight"] = 0.35
        self.rewards.rear_feet_highstep_clearance.params["rear_x_single_weight"] = 0.45
        self.rewards.rear_feet_highstep_clearance.params["rear_x_weight"] = 0.12
        self.rewards.rear_first_foot_highstep_preclearance.weight = 1.05
        self.rewards.rear_first_foot_highstep_preclearance.params.update(
            {
                "clearance_margin": 0.07,
                "first_clear_min": 0.56,
                "second_clear_suppress_min": 0.72,
                "commit_gate_min": 0.25,
                "commit_gate_scale": 0.75,
                "max_roll_metric": 0.28,
            }
        )
        self.rewards.rear_feet_under_step_after_commit.weight = -0.65
        self.rewards.rear_feet_under_step_after_commit.params["clearance_margin"] = 0.04
        self.rewards.rear_feet_under_step_after_commit.params["stall_floor"] = 0.14
        self.rewards.rear_feet_under_step_after_commit.params["worst_rear_weight"] = 0.82
        self.rewards.rear_second_foot_highstep_clearance.weight = 1.05
        self.rewards.rear_second_foot_highstep_clearance.params.update(
            {
                "progress_floor": 0.20,
                "branch_lead_threshold": 0.45,
                "branch_second_clear_threshold": 0.60,
                "branch_one_sided_gap": 0.12,
            }
        )
        self.rewards.second_rear_clear_deadline.weight = 0.85
        self.rewards.second_rear_clear_deadline.params.update(
            {
                "second_clear_min": 0.56,
                "deadline_steps": 30.0,
                "deadline_grace_steps": 4.0,
            }
        )
        self.rewards.one_sided_rear_stall_time.weight = -0.75
        self.rewards.one_sided_rear_stall_time.params.update(
            {
                "lead_threshold": 0.45,
                "one_sided_gap": 0.14,
                "grace_steps": 8.0,
                "ramp_steps": 20.0,
            }
        )
        self.rewards.lead_rear_support_drive.weight = 1.25
        self.rewards.lead_rear_support_drive.params.update(
            {
                "rear_push_target": 0.003,
                "target_std": 0.016,
                "lead_box_floor": 0.30,
                "target_height_gain": 0.10,
                "target_forward_vel": 0.22,
                "body_drive_floor": 0.05,
                "second_clear_relief_min": 0.70,
                "second_pending_floor": 0.35,
            }
        )
        self.rewards.post_lead_body_drive.weight = 1.70
        self.rewards.post_lead_body_drive.params.update(
            {
                "lead_score_min": 0.54,
                "second_clear_min": 0.55,
                "deadline_steps": 34.0,
                "deadline_grace_steps": 5.0,
                "target_height_gain": 0.11,
                "target_forward_vel": 0.22,
                "progress_floor": 0.18,
                "second_floor": 0.10,
                "body_height_weight": 0.62,
                "body_progress_weight": 0.33,
                "second_clear_weight": 0.05,
            }
        )
        self.rewards.post_lead_stall_penalty.weight = -1.20
        self.rewards.post_lead_stall_penalty.params.update(
            {
                "lead_score_min": 0.52,
                "second_low_threshold": 0.60,
                "one_sided_gap": 0.14,
                "one_sided_floor": 0.35,
                "grace_steps": 4.0,
                "ramp_steps": 16.0,
                "target_progress": 0.50,
                "base_floor": 0.35,
                "second_low_weight": 0.40,
                "progress_low_weight": 0.35,
            }
        )
        # Highstep-only policy path: deployment will switch back to a flat policy
        # after climbing, so this is only a light pose-compatibility term.
        self.rewards.post_clear_recovery.weight = 0.12
        self.rewards.post_clear_recovery.params.update(
            {
                "second_clear_min": 0.68,
                "min_elapsed_steps": 8.0,
                "elapsed_ramp_steps": 10.0,
                "min_base_clearance": 0.35,
                "target_base_clearance": 0.43,
                "target_forward_vel": 0.22,
                "max_abs_pitch_metric": 0.24,
                "max_abs_roll_metric": 0.24,
                "posture_weight": 0.45,
                "base_clearance_weight": 0.35,
                "forward_weight": 0.20,
            }
        )
        # Keep scanner-triggered pre-lift from becoming absurd, but do not let
        # this auxiliary constraint dominate the dedicated climbing policy.
        self.rewards.scanner_pretrigger_penalty.weight = -0.02
        self.rewards.scanner_pretrigger_penalty.params.update(
            {
                "terrain_gate_min": 0.20,
                "commit_gate_cutoff": 0.22,
                "low_cmd_threshold": 0.12,
                "front_lift_limit": -0.14,
                "front_lift_window": 0.14,
                "forward_vel_margin": 0.10,
                "forward_vel_window": 0.25,
            }
        )
        self.rewards.rear_approach_width_penalty.weight = -0.10
        self.rewards.rear_approach_width_penalty.params.update(
            {
                "terrain_gate_min": 0.08,
                "commit_gate_cutoff": 0.30,
                "min_rear_width": 0.26,
                "width_window": 0.08,
                "min_rear_abs_y": 0.11,
                "center_window": 0.05,
                "stage_start_update": 0,
                "stage_ramp_updates": 120,
            }
        )
        self.rewards.highstep_forward_progress.weight = 1.20
        self.rewards.highstep_body_lift.weight = 0.90
        self.rewards.highstep_base_advance_lift.weight = 1.25
        self.rewards.base_height_floor_penalty.weight = -0.25
        self.rewards.base_height_floor_penalty.params["min_height"] = 0.30
        self.rewards.highstep_leg_support_contact.weight = 0.50
        # Do not pay for a front-leg support-looking pose without actual support contact.
        # 2026-06-28_08-23-04 showed rear climbing promise, but this pose shortcut
        # can let the policy satisfy body-drive rewards through abnormal front lifting.
        self.rewards.highstep_leg_support_contact.params["support_pose_scale"] = 0.55
        self.rewards.non_forward_highstep_pitch.weight = -0.35
        self.rewards.backward_pitch_stability.weight = -1.20
        self.rewards.front_legs_quiet_penalty.weight = -0.08
        self.rewards.front_legs_quiet_penalty.params["step_height_threshold"] = 0.12
        # Disabled after 2026-06-28_09-53-57: local events showed this guard
        # recovered neither the rear-step entry nor post-lead body drive metrics.
        self.rewards.front_legs_lift_guard_penalty.weight = 0.0
        self.rewards.front_legs_lift_guard_penalty.params.update(
            {
                "relative_lift_limit": -0.14,
                "relative_lift_window": 0.14,
                "asymmetry_limit": 0.16,
                "asymmetry_window": 0.16,
                "terrain_relief_scale": 1.0,
                "commit_relief_scale": 0.90,
                "height_weight": 0.60,
                "asymmetry_weight": 0.40,
                "stage_start_update": 0,
                "stage_ramp_updates": 1,
            }
        )
        # Disabled after 2026-06-28_17-08-45: local events showed this FL
        # reward controlled flat FL lift but collapsed rear-step entry/support.
        # The flat-walk issue is handled by reset distribution instead.
        self.rewards.forward_flat_left_front_lift_penalty.weight = 0.0
        self.rewards.forward_flat_left_front_lift_penalty.params.update(
            {
                "fl_lift_limit": -0.24,
                "fl_lift_window": 0.08,
                "fl_over_fr_limit": 0.04,
                "fl_over_fr_window": 0.08,
                "terrain_gate_cutoff": 0.04,
                "commit_gate_cutoff": 0.035,
                "height_weight": 0.40,
                "asymmetry_weight": 0.60,
                "stage_start_update": 0,
                "stage_ramp_updates": 1,
            }
        )
        self.rewards.rear_legs_drive_bonus.weight = 1.10
        self.rewards.highstep_rear_push_posture.weight = 1.20
        self.rewards.highstep_rear_box_push.weight = 1.25
        self.rewards.highstep_bridge_stall_penalty.weight = -1.10
        self.rewards.highstep_box_phase_prior.weight = 1.00
        self.rewards.highstep_box_phase_prior.params["rear_push_target"] = 0.003
        self.rewards.highstep_box_phase_prior.params["target_std"] = 0.016
        self.rewards.front_legs_reach.params["terrain_gate_floor"] = 0.0
        self.rewards.horse_rearing_bonus.params["terrain_gate_floor"] = 0.0

        highstep_terms = (
            self.rewards.front_legs_reach,
            self.rewards.front_feet_highstep_clearance,
            self.rewards.front_legs_lift_guard_penalty,
            self.rewards.forward_flat_left_front_lift_penalty,
            self.rewards.rear_feet_highstep_clearance,
            self.rewards.rear_first_foot_highstep_preclearance,
            self.rewards.rear_feet_under_step_after_commit,
            self.rewards.rear_second_foot_highstep_clearance,
            self.rewards.second_rear_clear_deadline,
            self.rewards.one_sided_rear_stall_time,
            self.rewards.lead_rear_support_drive,
            self.rewards.post_lead_body_drive,
            self.rewards.post_lead_stall_penalty,
            self.rewards.post_clear_recovery,
            self.rewards.scanner_pretrigger_penalty,
            self.rewards.rear_approach_width_penalty,
            self.rewards.highstep_forward_progress,
            self.rewards.highstep_body_lift,
            self.rewards.highstep_base_advance_lift,
            self.rewards.highstep_leg_support_contact,
            self.rewards.non_forward_highstep_pitch,
            self.rewards.horse_rearing_bonus,
            self.rewards.rear_legs_drive_bonus,
            self.rewards.highstep_rear_push_posture,
            self.rewards.highstep_bridge_stall_penalty,
        )
        for term in highstep_terms:
            term.params["front_x_min"] = 0.25
            term.params["max_abs_y"] = 0.30
            term.params["height_threshold"] = 0.06
            term.params["height_gate_width"] = 0.14

        early_highstep_terms = (
            self.rewards.blind_climbing_bonus,
            self.rewards.pitch_up_on_obstacle,
            self.rewards.front_legs_reach,
            self.rewards.front_feet_highstep_clearance,
            self.rewards.horse_rearing_bonus,
            self.rewards.highstep_box_phase_prior,
            self.rewards.scanner_pretrigger_penalty,
            self.rewards.rear_approach_width_penalty,
        )
        late_highstep_terms = (
            self.rewards.rear_feet_highstep_clearance,
            self.rewards.rear_first_foot_highstep_preclearance,
            self.rewards.rear_feet_under_step_after_commit,
            self.rewards.rear_second_foot_highstep_clearance,
            self.rewards.second_rear_clear_deadline,
            self.rewards.one_sided_rear_stall_time,
            self.rewards.lead_rear_support_drive,
            self.rewards.post_lead_body_drive,
            self.rewards.post_lead_stall_penalty,
            self.rewards.post_clear_recovery,
            self.rewards.highstep_forward_progress,
            self.rewards.highstep_body_lift,
            self.rewards.highstep_base_advance_lift,
            self.rewards.highstep_leg_support_contact,
            self.rewards.rear_legs_drive_bonus,
            self.rewards.highstep_rear_push_posture,
            self.rewards.highstep_rear_box_push,
            self.rewards.highstep_bridge_stall_penalty,
        )
        for term in early_highstep_terms:
            term.params["stage_start_update"] = 150
            term.params["stage_ramp_updates"] = 300
            term.params["num_steps_per_update"] = 24
        for term in late_highstep_terms:
            term.params["stage_start_update"] = 450
            term.params["stage_ramp_updates"] = 500
            term.params["num_steps_per_update"] = 24
        self.rewards.rear_approach_width_penalty.params["stage_start_update"] = 0
        self.rewards.rear_approach_width_penalty.params["stage_ramp_updates"] = 120
        self.rewards.rear_approach_width_penalty.params["num_steps_per_update"] = 24
        entry_first_rear_terms = (
            self.rewards.rear_feet_highstep_clearance,
            self.rewards.rear_first_foot_highstep_preclearance,
            self.rewards.rear_feet_under_step_after_commit,
        )
        delayed_rear_finish_terms = (
            self.rewards.second_rear_clear_deadline,
            self.rewards.one_sided_rear_stall_time,
            self.rewards.lead_rear_support_drive,
            self.rewards.post_lead_body_drive,
            self.rewards.post_lead_stall_penalty,
            self.rewards.post_clear_recovery,
        )
        for term in entry_first_rear_terms:
            term.params["stage_start_update"] = 300
            term.params["stage_ramp_updates"] = 450
        for term in delayed_rear_finish_terms:
            term.params["stage_start_update"] = 900
            term.params["stage_ramp_updates"] = 700

        self.rewards.rear_legs_drive_bonus.params["target_positive_power"] = 65.0
        self.rewards.rear_legs_drive_bonus.params["lift_gate_floor"] = 0.45

        self.rewards.highstep_rear_push_posture.params.update(
            {
                "rear_back_min": 0.15,
                "rear_back_target": 0.42,
                "min_front_rear_height_diff": 0.07,
                "target_front_rear_height_diff": 0.32,
                "commit_gate_floor": 0.35,
                "progress_floor": 0.20,
            }
        )
        self.rewards.highstep_rear_box_push.params.update(
            {
                "min_cmd_x": 0.10,
                "front_x_min": 0.25,
                "max_abs_y": 0.30,
                "height_threshold": 0.06,
                "height_gate_width": 0.14,
                "rear_push_target": 0.003,
                "target_std": 0.016,
                "commit_gate_floor": 0.35,
                "progress_floor": 0.25,
            }
        )
        self.rewards.highstep_box_phase_prior.params.update(
            {
                "min_cmd_x": 0.10,
                "front_x_min": 0.25,
                "max_abs_y": 0.30,
                "height_threshold": 0.06,
                "height_gate_width": 0.14,
                "commit_gate_floor": 0.25,
            }
        )
        self.rewards.highstep_bridge_stall_penalty.params.update(
            {
                "rear_back_min": 0.16,
                "rear_back_target": 0.42,
                "commit_gate_floor": 0.35,
            }
        )
        self.rewards.rear_feet_highstep_clearance.params["commit_gate_scale"] = 0.75
        self.rewards.highstep_forward_progress.params["commit_gate_scale"] = 0.65
        self.rewards.highstep_body_lift.params["commit_gate_scale"] = 0.70
        self.rewards.highstep_base_advance_lift.params["commit_gate_scale"] = 0.80
        self.rewards.highstep_leg_support_contact.params["commit_gate_scale"] = 0.65
        self.rewards.rear_legs_drive_bonus.params["commit_gate_scale"] = 0.75

        # Joint penaltie
        # self.rewards.joint_torques_l2.weight = -2.5e-6
        # 测试 暂时取消此惩罚
        self.rewards.joint_vel_l2.weight = -0.0025
        self.rewards.box_joint_vel_penalty.weight = -0.015
        self.rewards.joint_acc_l2.weight = -3.0e-8
        self.rewards.box_joint_acc_penalty.weight = -1.0e-5
        self.rewards.joint_pos_limits.weight = -0.05
        self.rewards.joint_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=".*_(hip|thigh|calf)_joint"
        )
        self.rewards.box_joint_pos_limits.weight = -3.0
        # 禁止超速
        self.rewards.joint_vel_limits.weight = -0.3

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.18
        # UNUESD self.rewards.action_l2.weight = 0.0
        self.rewards.box_joint_action_rate.weight = -0.08

        # Contact sensor
        self.rewards.undesired_contacts.weight = -0.2
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", "trunk"
        ]

        # self.rewards.contact_forces.weight = -0.005
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 6
        self.rewards.track_ang_vel_z_exp.weight = 3.5

        # Others
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = -0.01
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_contact.params["expect_contact_num"] = 2
        self.rewards.anti_pronk_contact.weight = 0.0
        self.rewards.anti_pronk_contact.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = 0.0
        self.rewards.stand_still_without_cmd.weight = -3.5
        # self.rewards.stand_still_joint_vel.weight = -0.3
        self.rewards.stand_still_revolute_joint_vel.weight = -0.10
        self.rewards.stand_still_prismatic_joint_vel.weight = -0.40
        self.rewards.stand_still_base_ang_vel.weight = -0.80
        self.rewards.stand_still_base_lin_vel.weight = -2.00
        self.rewards.stand_still_revolute_joint_vel.params["command_threshold"] = 0.06
        self.rewards.stand_still_prismatic_joint_vel.params["command_threshold"] = 0.06
        self.rewards.stand_still_base_ang_vel.params["command_threshold"] = 0.06
        self.rewards.stand_still_base_lin_vel.params["command_threshold"] = 0.08
        # self.rewards.joint_position_penalty.weight = -0.9
        # self.rewards.joint_position_penalty.params["stand_still_scale"] = 1.5
        # self.rewards.joint_position_penalty.params["velocity_threshold"] = 0.3
        self.rewards.rotate_joint_pos_penalty.weight = -0.03
        self.rewards.prismatic_joint_pos_penalty.weight = -2.5
        self.rewards.prismatic_joint_pos_penalty.params["hold_scale"] = 16.0
        self.rewards.prismatic_joint_pos_penalty.params["highstep_scale"] = 0.08
        self.rewards.feet_height_exp.weight = 0.15
        self.rewards.feet_height_exp.params["target_height"] = 0.12
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        # self.rewards.feet_height_body_exp.weight = -4.9
        # self.rewards.feet_height_body_exp.params["target_height"] = -0.32
        # self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        self.rewards.feet_gait.weight = 0.50
        self.rewards.feet_gait.params["velocity_threshold"] = 0.50
        # trotting
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )
        # pronking
        # self.rewards.feet_gait.params["synced_feet_pair_names"] = (
        #     ("FL_foot", "FR_foot"),
        #     ("RR_foot", "RL_foot"),
        # ) 

        # # =====================================================================
        # # 激活足端间距惩罚
        # # =====================================================================
        # # 权重设为负数。-5.0 属于中等偏上的惩罚力度，足以引起策略的重视。
        # # 如果发现机器人腿张得太开，可以把权重改小（如 -2.0）或者减小 min_width。
        # self.rewards.feet_stance_width.weight = -5.5
        # self.rewards.feet_stance_width_advanced_adaptive.weight = -5.5
        # Highstep climbing needs hip/leg lateral freedom. Keep this at zero;
        # disable_zero_weight_rewards() will remove the term safely below.
        self.rewards.feet_stance_width.weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ArclabArcdogAdjustableLegHighstepEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     self.base_link_name,
        #     self.trunk_link_name,
        #     # self.abad_link_name,
        #     # self.knee_link_name,
        #     # self.hip_link_name,
        # ]
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, # 具体的函数名取决于你使用的 Isaac Lab 版本
            params={
                "limit_angle": 1.2, # 允许的最大倾斜角，1.2 弧度大约是 68 度。
                # 爬高台时 pitch (俯仰角) 会很大，所以这个角度要放宽，不能设成 0.5 这种小角度
                "asset_cfg": SceneEntityCfg("robot")
            }
        )
        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel_highstep
        self.curriculum.terrain_levels.params = {
            "move_up_distance": 3.2,
            "move_down_command_factor": 0.28,
            "move_down_min_distance": 0.35,
            "move_down_max_distance": 1.2,
            "min_height_gain": 0.015,
            "height_gain_required_level": 2,
            "nominal_base_height": 0.44,
            "climb_up_distance": 0.50,
            "climb_height_gain": 0.025,
            "climb_height_required_level": 0,
            "climb_hold_distance": 0.35,
            "climb_hold_height_gain": 0.015,
            "stage_update_thresholds": (300, 900, 1800),
            "stage_max_levels": (2, 3, 5, 8),
            "num_steps_per_update": 24,
        }
        self.curriculum.command_levels.func = mdp.command_levels_vel_highstep
        self.curriculum.command_levels.params = {
            "reward_term_name": "track_lin_vel_xy_exp",
            "range_multiplier": (0.35, 0.75),
            "gated_multiplier": 0.55,
            # Fresh highstep training keeps the final command release gated by terrain progress.
            # Resume/refine runs relax this gate in scripts/rsl_rl/base/train.py.
            "terrain_gate_level": 2.6,
            "delta": 0.03,
            "reward_threshold": 0.68,
        }
        self.curriculum.highstep_rear_branch_metrics = CurrTerm(
            func=mdp.highstep_rear_branch_metrics,
            params={"min_commit_steps": 1},
        )


        # ------------------------------Commands------------------------------
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.08, 0.08)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.22, 0.22)
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.lin_vel_threshold = 0.05


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreEnvCfg(ArclabArcdogAdjustableLegHighstepEnvCfg):
    """Rebuilt high-step task whose curriculum is gated by behavior score."""

    rewards: ArcdogAdjustableLegHighstepActionScoreRewardsCfg = (
        ArcdogAdjustableLegHighstepActionScoreRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # Preserve the validated model_172300 Teacher action prior exactly.
        # Delay randomization remains opt-in in the Robust subclass below.
        self.actions.joint_pos.min_action_delay_steps = 0
        self.actions.joint_pos.max_action_delay_steps = 0

        # Start a new high-step-only task from focused front-approach samples.
        self.scene.terrain.max_init_terrain_level = 1
        self.events.randomize_reset_base.params.update(
            {
                "approach_ratio": 0.94,
                "approach_distance_range": (1.55, 2.15),
                "approach_yaw_noise_range": (-0.12, 0.12),
            }
        )
        self.events.randomize_highstep_foot_under_hip_reset = EventTerm(
            func=mdp.randomize_highstep_foot_under_hip_reset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "hip_joint_names": ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                "thigh_joint_names": ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                "calf_joint_names": ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"],
                "box_joint_names": ["FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint"],
                "hip_outward_range": (0.0, 0.060),
                "hip_noise_range": (-0.018, 0.018),
                "thigh_position_range": (-0.030, 0.030),
                "calf_position_range": (-0.030, 0.030),
                "box_position_range": (-0.004, 0.004),
                "velocity_range": (-0.12, 0.12),
                "rear_inward_prob": 0.36,
                "rear_inward_range": (0.018, 0.060),
            },
        )
        self.events.randomize_limb_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="interval",
            interval_range_s=(0.35, 0.85),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        self.abad_link_name,
                        self.hip_link_name,
                        self.knee_link_name,
                        self.extension_link_name,
                        self.foot_link_name,
                    ],
                ),
                "force_range": (-3.5, 3.5),
                "torque_range": (-1.0, 1.0),
            },
        )
        if hasattr(self.actions.joint_pos, "prior_start_update"):
            self.actions.joint_pos.prior_start_update = 80
            self.actions.joint_pos.prior_full_update = 520
            self.actions.joint_pos.min_forward_command = 0.06
            self.actions.joint_pos.command_gate_width = 0.22

        # Completion-style highstep objective:
        # keep command response, but demote ordinary locomotion tracking below the
        # staged climb score so later training cannot optimize "walking" over
        # first-rear clear, support drive, and second-rear clear.
        self.rewards.track_lin_vel_xy_exp.weight = 0.65
        self.rewards.track_ang_vel_z_exp.weight = 1.55
        self.rewards.highstep_forward_progress.weight = 0.65
        self.rewards.highstep_body_lift.weight = 1.15
        self.rewards.highstep_base_advance_lift.weight = 1.05
        self.rewards.base_height_l2.weight = -0.25
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_gait.weight = 0.15

        # Main behavior objective: entry + post-lead support + second rear clear.
        self.rewards.highstep_action_score.weight = 3.60
        self.rewards.highstep_action_score.params.update(
            {
                "first_clear_min": 0.50,
                "second_clear_min": 0.54,
                "lead_score_min": 0.48,
                "support_floor": 0.45,
                "support_gate_floor": 0.08,
                "support_bottleneck_start_update": 650,
                "support_bottleneck_ramp_updates": 700,
                "support_bottleneck_warmup_min_gate": 0.0,
                "target_height_gain": 0.09,
                "target_forward_vel": 0.18,
                "centerline_min_rear_width": 0.32,
                "centerline_width_window": 0.14,
                "centerline_min_rear_abs_y": 0.165,
                "centerline_center_window": 0.105,
                "stage_start_update": 80,
                "stage_ramp_updates": 320,
            }
        )
        self.rewards.front_legs_reach.weight = 0.45
        self.rewards.front_feet_highstep_clearance.weight = 0.20
        self.rewards.rear_feet_highstep_clearance.weight = 1.30
        self.rewards.rear_feet_highstep_clearance.params["clearance_margin"] = 0.03
        self.rewards.rear_first_foot_highstep_preclearance.weight = 1.65
        self.rewards.rear_first_foot_highstep_preclearance.params.update(
            {
                "clearance_margin": 0.05,
                "first_clear_min": 0.50,
                "commit_gate_min": 0.12,
                "terrain_commit_min": 0.20,
                "terrain_commit_floor": 0.35,
                "max_roll_metric": 0.34,
            }
        )
        self.rewards.rear_second_foot_highstep_clearance.weight = 1.35
        self.rewards.lead_rear_support_drive.weight = 2.40
        self.rewards.lead_rear_support_drive.params.update(
            {
                "lead_box_floor": 0.40,
                "target_height_gain": 0.08,
                "target_forward_vel": 0.16,
                "body_drive_floor": 0.0,
                "body_height_weight": 0.85,
                "body_progress_weight": 0.15,
                "body_lift_velocity_weight": 0.25,
                "lift_velocity_target": 0.16,
                "progress_lift_gate_start": 0.20,
                "progress_lift_gate_end": 0.55,
                "second_pending_floor": 0.55,
            }
        )
        self.rewards.post_lead_body_drive.weight = 3.40
        self.rewards.post_lead_body_drive.params["target_forward_vel"] = 0.16
        self.rewards.post_lead_stall_penalty.weight = -0.95
        self.rewards.second_rear_clear_deadline.weight = 1.25
        self.rewards.one_sided_rear_stall_time.weight = -0.70
        self.rewards.highstep_rear_box_push.weight = 1.45
        self.rewards.highstep_box_phase_prior.weight = 1.05
        self.rewards.highstep_leg_support_contact.params["support_pose_scale"] = 0.45

        # After both rear feet clear, keep enough command-following pressure to
        # leave the platform entry edge instead of collecting a static completion pose.
        self.rewards.post_clear_recovery.weight = 0.75
        # The only modified Teacher mechanism is the post-clear target mode:
        # absolute entry-edge margin in recovery plus a consecutive near-edge
        # timer in the paired stall penalty.  Both term weights remain fixed.
        self.rewards.post_clear_rear_advance_stall.weight = -0.50
        # 2026-07-03: keep the ActionScore bottleneck honest. First-rear entry
        # gets dense reward from the rear/support terms; the composite score
        # must not fake support progress before the support phase is real.
        for term in (
            self.rewards.second_rear_clear_deadline,
            self.rewards.lead_rear_support_drive,
            self.rewards.post_lead_body_drive,
        ):
            term.params["stage_start_update"] = min(term.params.get("stage_start_update", 0), 260)
            term.params["stage_ramp_updates"] = min(term.params.get("stage_ramp_updates", 1), 300)
        for term in (self.rewards.one_sided_rear_stall_time, self.rewards.post_lead_stall_penalty):
            term.params["stage_start_update"] = min(term.params.get("stage_start_update", 0), 480)
            term.params["stage_ramp_updates"] = min(term.params.get("stage_ramp_updates", 1), 360)
        self.rewards.post_clear_recovery.params["stage_start_update"] = min(
            self.rewards.post_clear_recovery.params.get("stage_start_update", 0), 650
        )
        self.rewards.post_clear_recovery.params["stage_ramp_updates"] = min(
            self.rewards.post_clear_recovery.params.get("stage_ramp_updates", 1), 400
        )
        for term in (
            self.rewards.rear_feet_highstep_clearance,
            self.rewards.rear_first_foot_highstep_preclearance,
            self.rewards.rear_feet_under_step_after_commit,
        ):
            term.params["stage_start_update"] = min(term.params.get("stage_start_update", 0), 120)
            term.params["stage_ramp_updates"] = min(term.params.get("stage_ramp_updates", 1), 220)
        self.rewards.rear_second_foot_highstep_clearance.params["stage_start_update"] = min(
            self.rewards.rear_second_foot_highstep_clearance.params.get("stage_start_update", 0), 220
        )
        self.rewards.rear_second_foot_highstep_clearance.params["stage_ramp_updates"] = min(
            self.rewards.rear_second_foot_highstep_clearance.params.get("stage_ramp_updates", 1), 280
        )
        self.rewards.post_lead_body_drive.params.update(
            {
                "lead_score_min": 0.40,
                "lead_ready_floor": 0.35,
                "min_elapsed_steps": 0.0,
                "elapsed_ramp_steps": 4.0,
                "elapsed_gate_floor": 0.55,
                "deadline_steps": 54.0,
                "deadline_grace_steps": 12.0,
                "roll_gate_floor": 0.35,
                "target_height_gain": 0.08,
                "progress_floor": 0.0,
                "body_height_weight": 0.84,
                "body_progress_weight": 0.06,
                "second_clear_weight": 0.10,
                "body_lift_velocity_weight": 0.20,
                "lift_velocity_target": 0.16,
                "progress_lift_gate_start": 0.20,
                "progress_lift_gate_end": 0.55,
                "second_lift_gate_start": 0.15,
                "second_lift_gate_end": 0.45,
            }
        )
        self.rewards.post_clear_recovery.params.update(
            {
                "second_clear_min": 0.60,
                "min_elapsed_steps": 4.0,
                "elapsed_ramp_steps": 8.0,
                "target_forward_vel": 0.30,
                # Absolute safety target for the slower rear foot.  The 0.04 m
                # floor matches the existing clearance/q05 criterion; 0.18 m
                # is the promised inward target after both rear feet clear.
                "platform_half_width": 1.5,
                "min_rear_margin": 0.04,
                "target_rear_margin": 0.18,
                "posture_weight": 0.15,
                "base_clearance_weight": 0.10,
                "forward_weight": 0.25,
                "rear_advance_weight": 0.50,
                "post_clear_latch_scale": 1.0,
            }
        )
        self.rewards.scanner_pretrigger_penalty.weight = -0.015
        self.rewards.rear_approach_width_penalty.weight = -0.62
        self.rewards.rear_approach_width_penalty.params.update(
            {
                "terrain_gate_min": 0.03,
                "terrain_gate_floor": 0.18,
                "commit_gate_cutoff": 0.72,
                "precommit_gate_floor": 0.08,
                "min_rear_width": 0.34,
                "width_window": 0.11,
                "min_rear_abs_y": 0.175,
                "center_window": 0.095,
                "width_weight": 0.16,
                "center_weight": 0.50,
                "hard_center_weight": 0.34,
                "critical_center_boost": 0.80,
                "critical_min_rear_abs_y": 0.120,
                "critical_center_window": 0.065,
                "stage_start_update": 0,
                "stage_ramp_updates": 20,
                "num_steps_per_update": 24,
            }
        )
        self.rewards.rear_highstep_motion_width_penalty.weight = -0.52
        self.rewards.rear_highstep_motion_width_penalty.params.update(
            {
                "min_rear_width": 0.34,
                "width_window": 0.11,
                "min_rear_abs_y": 0.170,
                "center_window": 0.095,
                "width_weight": 0.17,
                "center_weight": 0.49,
                "hard_center_weight": 0.34,
                "critical_center_boost": 0.85,
                "critical_min_rear_abs_y": 0.115,
                "critical_center_window": 0.065,
                "post_clear_relief": 0.80,
                "stage_start_update": 0,
                "stage_ramp_updates": 60,
                "num_steps_per_update": 24,
            }
        )
        self.rewards.highstep_support_stability_penalty.weight = -0.18
        self.rewards.highstep_support_stability_penalty.params.update(
            {
                "stage_start_update": 0,
                "stage_ramp_updates": 220,
                "num_steps_per_update": 24,
            }
        )
        self.rewards.front_legs_lift_guard_penalty.weight = 0.0
        self.rewards.forward_flat_left_front_lift_penalty.weight = 0.0
        # Hardware-oriented stability refine. Keep climb rewards intact, but
        # make noisy/high-frequency actuation more expensive before distillation.
        self.rewards.action_rate_l2.weight = -0.26
        self.rewards.box_joint_action_rate.weight = -0.12
        self.rewards.joint_vel_l2.weight = -0.0032
        self.rewards.box_joint_vel_penalty.weight = -0.020
        self.rewards.joint_acc_l2.weight = -4.0e-8
        self.rewards.box_joint_acc_penalty.weight = -1.3e-5

        # Terrain promotion is now gated by behavior score, not just distance/height.
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel_highstep_action_score
        self.curriculum.terrain_levels.params = {
            "move_up_distance": 3.2,
            "move_down_command_factor": 0.28,
            "move_down_min_distance": 0.35,
            "move_down_max_distance": 1.2,
            "min_height_gain": 0.015,
            "height_gain_required_level": 2,
            "nominal_base_height": 0.44,
            "climb_up_distance": 0.50,
            "climb_height_gain": 0.025,
            "climb_height_required_level": 0,
            "climb_hold_distance": 0.35,
            "climb_hold_height_gain": 0.015,
            "success_hold_height_gain": 0.015,
            "success_hold_action_score": 0.60,
            "success_hold_support_score": 0.45,
            "stage_update_thresholds": (650, 1300, 2200, 3200),
            "stage_max_levels": (1, 2, 3, 5, 8),
            "num_steps_per_update": 24,
            "score_gate_required_level": 1,
            "score_warmup_updates": 450,
            "score_stage_update_thresholds": (800, 1400, 2200),
            "action_score_thresholds": (0.14, 0.28, 0.44, 0.60),
            "support_score_thresholds": (0.005, 0.08, 0.24, 0.45),
            "support_floor": 0.45,
            "support_gate_floor": 0.08,
            "support_bottleneck_start_update": 650,
            "support_bottleneck_ramp_updates": 700,
            "support_bottleneck_warmup_min_gate": 0.0,
            "regression_tolerance": 0.10,
            "support_regression_tolerance": 0.12,
            "entry_floor": 0.55,
            "second_clear_floor": 0.75,
            "lead_valid_floor": 0.35,
            "support_activation_gate_target": 0.0010,
            "support_signal_target": 0.0010,
            "support_quality_scale": 0.80,
        }
        self.curriculum.command_levels.params.update(
            {
                "range_multiplier": (0.45, 0.90),
                "gated_multiplier": 0.75,
                "terrain_gate_level": 1.5,
                "delta": 0.035,
                "reward_threshold": 0.55,
            }
        )
        self.curriculum.highstep_rear_branch_metrics = None
        self.curriculum.highstep_action_score = CurrTerm(
            func=mdp.highstep_action_score_metrics,
            params={
                "min_commit_steps": 1,
                "support_floor": 0.45,
                "support_gate_floor": 0.08,
                "support_bottleneck_start_update": 650,
                "support_bottleneck_ramp_updates": 700,
                "support_bottleneck_warmup_min_gate": 0.0,
                "num_steps_per_update": 24,
                "entry_floor": 0.55,
                "second_clear_floor": 0.75,
                "lead_valid_floor": 0.35,
                "support_activation_gate_target": 0.0010,
                "support_signal_target": 0.0010,
                "support_quality_scale": 0.80,
            },
        )

        # Manual-control envelope for the completion task.  x command still
        # matters, backward command is represented explicitly, and yaw is given a
        # much wider range because the operator needs live heading correction
        # while approaching and committing to the step.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.18, 0.72)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.06, 0.06)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.55, 0.55)
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)
        self.commands.base_velocity.lin_vel_threshold = 0.04

        self.disable_zero_weight_rewards()


@configclass
class ArclabArcdogAdjustableLegHighstepStudentNoPriorEnvCfg(ArclabArcdogAdjustableLegHighstepEnvCfg):
    """Highstep student env without the highstep action prior."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=self.joint_names,
            scale={
                ".*_box_joint": 0.02,
                ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
            },
            use_default_offset=True,
            clip={".*": (-60.0, 60.0)},
            preserve_order=True,
        )

        self.disable_zero_weight_rewards()


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorEnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreEnvCfg
):
    """Action-score highstep student env without the highstep action prior."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=self.joint_names,
            scale={
                ".*_box_joint": 0.02,
                ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
            },
            use_default_offset=True,
            clip={".*": (-60.0, 60.0)},
            preserve_order=True,
        )

        self.disable_zero_weight_rewards()


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV15EnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorEnvCfg
):
    """v1.5 headless training env without non-physical remote marker assets."""

    def __post_init__(self):
        super().__post_init__()
        # Command debug arrows are viewport-only and fetch a remote USD.  They
        # are not policy observations and have no physics/reward/action effect.
        self.commands.base_velocity.debug_vis = False


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18BootstrapEnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorEnvCfg
):
    """Stage-A Student environment mechanically restored from the 0707 snapshot."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v18_saved_environment_profile(
            self, _V18_0707_STUDENT_ENV, _V18_0707_STUDENT_ENV_SHA256
        )


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18RobustEnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV15EnvCfg
):
    """Stage-B Student environment restored from the v1.7.1 E700 snapshot."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v18_saved_environment_profile(
            self, _V18_CURRENT_ROBUST_STUDENT_ENV,
            _V18_CURRENT_ROBUST_STUDENT_ENV_SHA256,
        )


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreTeacherV18BootstrapEnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreEnvCfg
):
    """Teacher bootstrap environment retaining the hash-frozen 0707 action prior."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v18_saved_environment_profile(
            self, _V18_0707_TEACHER_ENV, _V18_0707_TEACHER_ENV_SHA256
        )


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreRobustEnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreEnvCfg
):
    """Deployment-robust teacher used only after the standard teacher passes."""

    def __post_init__(self):
        super().__post_init__()

        # Wire-level /joint_commands from both 0707 bags agree on these RL2
        # revolute gains.  Center the Robust distribution on measured hardware;
        # the standard Teacher remains unchanged as a scientific baseline.
        self.scene.robot.actuators["legs_hip"].stiffness = 55.0
        self.scene.robot.actuators["legs_hip"].damping = 1.5
        self.scene.robot.actuators["legs_thigh"].stiffness = 65.0
        self.scene.robot.actuators["legs_thigh"].damping = 1.5
        self.scene.robot.actuators["legs_calf"].stiffness = 80.0
        self.scene.robot.actuators["legs_calf"].damping = 2.5

        self.actions.joint_pos.min_action_delay_steps = 0
        self.actions.joint_pos.max_action_delay_steps = 1
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_with_persistent_bias
        self.observations.estimator.joint_pos.func = mdp.joint_pos_rel_with_persistent_bias
        self.events.randomize_highstep_joint_observation_bias = EventTerm(
            func=mdp.randomize_highstep_joint_observation_bias,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "hip_bias_range": (-0.020, 0.020),
                "leg_bias_range": (-0.012, 0.012),
                "box_bias_range": (-0.0015, 0.0015),
            },
        )
        self.rewards.post_clear_rear_advance_stall = RewTerm(
            func=mdp.post_clear_rear_advance_stall_penalty,
            weight=-0.50,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "rear_foot_names": ["RL_foot", "RR_foot"],
                "min_cmd_x": 0.08,
                "platform_half_width": 1.5,
                "min_rear_margin": 0.04,
                "near_edge_margin": 0.12,
                "target_rear_margin": 0.18,
                "grace_steps": 8.0,
                "ramp_steps": 32.0,
            },
        )


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorEnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreRobustEnvCfg
):
    """Student task matching the robust teacher dynamics without an action prior."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = mdp.DelayedJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.joint_names,
            scale={
                ".*_box_joint": 0.02,
                ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
            },
            use_default_offset=True,
            clip={".*": (-60.0, 60.0)},
            preserve_order=True,
            min_action_delay_steps=0,
            max_action_delay_steps=1,
        )


@configclass
class ArcdogAdjustableLegHighstepR2TeacherContextCfg(ObsGroup):
    """Training-only exact context for the frozen Teacher post-prior."""

    prior_context = ObsTerm(
        func=mdp.HighstepTeacherPriorContext,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "foot_asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "front_x_min": 0.25,
            "rear_x_max": -0.20,
            "max_abs_y": 0.30,
        },
        history_length=0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0
        self.flatten_history_dim = True


@configclass
class ArcdogAdjustableLegHighstepR2ObservationsCfg(ObservationsCfg):
    """Existing deployment observations plus one isolated rollout-label group."""

    teacher_context: ArcdogAdjustableLegHighstepR2TeacherContextCfg = (
        ArcdogAdjustableLegHighstepR2TeacherContextCfg()
    )


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorR2EnvCfg(
    ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorEnvCfg
):
    """R2-only Student task; policy, estimator and critic observations stay unchanged."""

    observations: ArcdogAdjustableLegHighstepR2ObservationsCfg = (
        ArcdogAdjustableLegHighstepR2ObservationsCfg()
    )
