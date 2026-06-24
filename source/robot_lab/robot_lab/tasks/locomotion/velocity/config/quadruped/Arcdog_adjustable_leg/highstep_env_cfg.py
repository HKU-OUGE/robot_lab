# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg 
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, RewardsCfg,
)

##
# Pre-defined configs
##
# use cloud assets
# from isaaclab_assets.robots.unitree import ARCLAB_ARCDOG_CFG  # isort: skip
# use local assets
from robot_lab.assets.arclab import ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG  # isort: skip
from robot_lab.terrains.config.rough import HIGHSTEP_TERRAINS_CFG  # isort:skip


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
            "rear_x_weight": 0.10,
            "rear_x_single_weight": 1.0,
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
                "approach_ratio": 0.75,
                "approach_distance_range": (1.65, 2.65),
                "approach_yaw_noise_range": (-0.25, 0.25),
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
        self.rewards.front_legs_reach.weight = 0.75
        self.rewards.front_feet_highstep_clearance.weight = 0.40
        self.rewards.rear_feet_highstep_clearance.weight = 1.15
        self.rewards.rear_feet_highstep_clearance.params["single_rear_weight"] = 0.20
        self.rewards.rear_feet_highstep_clearance.params["min_rear_weight"] = 0.50
        self.rewards.rear_feet_highstep_clearance.params["rear_x_single_weight"] = 1.0
        self.rewards.rear_feet_highstep_clearance.params["rear_x_weight"] = 0.10
        self.rewards.rear_feet_under_step_after_commit.weight = -0.50
        self.rewards.rear_feet_under_step_after_commit.params["worst_rear_weight"] = 0.82
        self.rewards.rear_second_foot_highstep_clearance.weight = 0.95
        self.rewards.rear_second_foot_highstep_clearance.params.update(
            {
                "progress_floor": 0.20,
                "branch_lead_threshold": 0.45,
                "branch_second_clear_threshold": 0.60,
                "branch_one_sided_gap": 0.12,
            }
        )
        self.rewards.highstep_forward_progress.weight = 1.60
        self.rewards.highstep_body_lift.weight = 0.85
        self.rewards.highstep_base_advance_lift.weight = 1.75
        self.rewards.base_height_floor_penalty.weight = -0.25
        self.rewards.base_height_floor_penalty.params["min_height"] = 0.30
        self.rewards.highstep_leg_support_contact.weight = 0.50
        self.rewards.non_forward_highstep_pitch.weight = -0.35
        self.rewards.backward_pitch_stability.weight = -1.20
        self.rewards.front_legs_quiet_penalty.weight = -0.08
        self.rewards.front_legs_quiet_penalty.params["step_height_threshold"] = 0.12
        self.rewards.rear_legs_drive_bonus.weight = 0.90
        self.rewards.highstep_rear_push_posture.weight = 1.30
        self.rewards.highstep_rear_box_push.weight = 1.10
        self.rewards.highstep_bridge_stall_penalty.weight = -1.10
        self.rewards.highstep_box_phase_prior.weight = 0.85
        self.rewards.highstep_box_phase_prior.params["rear_push_target"] = 0.003
        self.rewards.highstep_box_phase_prior.params["target_std"] = 0.016
        self.rewards.front_legs_reach.params["terrain_gate_floor"] = 0.0
        self.rewards.horse_rearing_bonus.params["terrain_gate_floor"] = 0.0

        highstep_terms = (
            self.rewards.front_legs_reach,
            self.rewards.front_feet_highstep_clearance,
            self.rewards.rear_feet_highstep_clearance,
            self.rewards.rear_feet_under_step_after_commit,
            self.rewards.rear_second_foot_highstep_clearance,
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
        )
        late_highstep_terms = (
            self.rewards.rear_feet_highstep_clearance,
            self.rewards.rear_feet_under_step_after_commit,
            self.rewards.rear_second_foot_highstep_clearance,
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
        self.rewards.rear_feet_highstep_clearance.params["commit_gate_scale"] = 0.90
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
            "climb_height_required_level": 2,
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
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (-0.30, 0.85)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.60, 0.60)
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)
        self.commands.base_velocity.lin_vel_threshold = 0.05


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
