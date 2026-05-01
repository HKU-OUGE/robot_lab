# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
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
from robot_lab.assets.arclab import ARCLAB_ARCDOG_NEW_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip


@configclass
class ArcdogRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # =====================================================================
    # 新增：对角腿对称性惩罚 (解决单腿异常抬高 / 强制 Trot 步态)
    # =====================================================================
    gait_symmetry_penalty = RewTerm(
        func=mdp.diagonal_gait_symmetry_penalty, # 指向我们在第一步写的底层函数
        weight=0.0, # 默认设为 0，在主 EnvCfg 中激活
        params={
            # 使用正则表达式匹配 4 个足端刚体
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # =====================================================================
    # 新增：动作二阶导数 (动作加速度) 惩罚，用于提高动作平滑度
    # =====================================================================
    action_acceleration_penalty = RewTerm(
        func=mdp.action_acceleration_l2,
        weight=0.0, # 默认设为 0，在主 EnvCfg 中激活
    )

    feet_stance_width = RewTerm(
        func=mdp.feet_stance_width_adaptive_penalty, 
        weight= 0.0,  # 建议保持在 -1.0 到 -2.0 之间
        params={
            "min_width": 0.32,               
            "command_speed_threshold": 0.25,  # 【关键】阈值调小！指令速度低于 0.15m/s 就视为“准备静止”，开始张开腿
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

@configclass
class ArclabArcdogRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ArcdogRewardsCfg = ArcdogRewardsCfg()
    base_link_name = "base"
    trunk_link_name = "trunk"
    hip_link_name = ".*_hip"
    knee_link_name = ".*_calf"
    abad_link_name = ".*_thigh"
    foot_link_name = ".*_foot"

    # fmt: off
    joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint",
        "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint",
        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree a1
        self.scene.robot = ARCLAB_ARCDOG_NEW_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator=ROUGH_TERRAINS_CFG
        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
            self.joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
            self.joint_names
        )

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-60.0, 60.0)}
        self.actions.joint_pos.joint_names = self.joint_names

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

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -20

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.5
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.base_height_l2.params["target_height"] = 0.365
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        # Joint penaltie
        # self.rewards.joint_torques_l2.weight = -2.5e-6
        # 测试 暂时取消此惩罚
        self.rewards.joint_vel_l2.weight = -0.005
        # self.rewards.joint_acc_l2.weight = -1.0e-6
        self.rewards.joint_pos_limits.weight = -0.05
        # 禁止超速
        self.rewards.joint_vel_limits.weight = -0.05

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.5
        # UNUESD self.rewards.action_l2.weight = 0.0
        self.rewards.action_acceleration_penalty.weight = -0.06

        # Contact sensor
        self.rewards.undesired_contacts.weight = -0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", "trunk", ".*_hip", ".*_thigh"
        ]

        # self.rewards.contact_forces.weight = -0.005
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 10.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0

        # Others
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = -0.05
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_stumble.weight = -0.05
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.joint_power.weight = -2e-5
        # 测试 暂时取消
        self.rewards.joint_power.weight = -2e-7
        self.rewards.stand_still_without_cmd.weight = -4.0
        self.rewards.joint_position_penalty.weight = -0.5
        self.rewards.joint_position_penalty.params["stand_still_scale"] = 1.8
        self.rewards.joint_position_penalty.params["velocity_threshold"] = 0.3
        self.rewards.feet_height_exp.weight = 1.0
        self.rewards.feet_height_exp.params["target_height"] = 0.12
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]  
        self.rewards.feet_height_body_exp.weight = -0.5
        self.rewards.feet_height_body_exp.params["target_height"] = -0.23
        self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_gait.weight = 5.0
        self.rewards.feet_gait.params["velocity_threshold"] = 0.5
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
        # self.rewards.gait_symmetry_penalty.weight = -3.0
        self.rewards.feet_stance_width.weight = -5.0
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ArclabArcdogRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            self.trunk_link_name,
            # self.abad_link_name,
            # self.knee_link_name,
            # self.hip_link_name,
        ]
        # self.terminations.illegal_contact = None
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels.params["range_multiplier"] = (0.1,1.0)


        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)
