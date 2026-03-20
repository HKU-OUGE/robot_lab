# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip


@configclass
class ArcdogAdjustableLegRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    rotate_joint_pos_penalty = RewTerm(
        func=mdp.joint_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh_joint|calf_joint)$"),
            "stand_still_scale": 1.8,
            "velocity_threshold": 0.3,
        },
    )

    prismatic_joint_pos_penalty  = RewTerm(
        func=mdp.joint_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )

    stand_still_flat = RewTerm(
        func=mdp.stand_still_flat_orientation_bonus,
        weight=0.0,  # 默认权重设为0，在 EnvCfg 中具体配置
        params={
            "command_name": "base_velocity",
            "std": 0.1,               # 控制对倾斜的敏感度，越小越严格
            "command_threshold": 0.1,  # 速度指令小于此值视为静止
            "asset_cfg": SceneEntityCfg("robot"),
        },
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


@configclass
class ArclabArcdogAdjustableLegBodyflatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ArcdogAdjustableLegRewardsCfg = ArcdogAdjustableLegRewardsCfg()


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
        
        # reduce action scale
        # self.actions.joint_pos.scale = 0.1
        self.actions.joint_pos.scale = {
            ".*_box_joint": 0.02, 
            ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
        }
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
        # self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        self.events.randomize_screw_joints.params["asset_cfg"].joint_names = [".*_box_joint"]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -20

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height_l2.weight = -3
        self.rewards.base_height_l2.params["target_height"] = 0.40
        # 设置静止水平奖励的权重
        # 这是一个正向奖励(Bonus)，所以权重为正。
        # 建议值: 0.5 ~ 2.0。如果机器人在坡上静止时还是歪的，可以调大这个值。
        self.rewards.stand_still_flat.weight = 3.0 
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
        self.rewards.box_joint_vel_penalty.weight = -0.01 
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.box_joint_acc_penalty.weight = -1.0e-5 # 伸缩关节的加速度惩罚，建议比全局高 1-2 个数量级 
        self.rewards.joint_pos_limits.weight = -0.05
        # 禁止超速
        self.rewards.joint_vel_limits.weight = -0.3

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.08
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", "trunk", ".*_hip", ".*_thigh", ".*_calf"
        ]

        # self.rewards.contact_forces.weight = -0.005
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 6
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # Others
        self.rewards.feet_air_time.weight = 3.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = -0.01
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
        self.rewards.joint_power.weight = -2e-6
        self.rewards.stand_still_without_cmd.weight = -3.5
        # self.rewards.joint_position_penalty.weight = -0.9
        # self.rewards.joint_position_penalty.params["stand_still_scale"] = 1.5
        # self.rewards.joint_position_penalty.params["velocity_threshold"] = 0.3
        self.rewards.rotate_joint_pos_penalty.weight = -0.03
        self.rewards.prismatic_joint_pos_penalty.weight = -3
        self.rewards.feet_height_exp.weight = 2.0
        self.rewards.feet_height_exp.params["target_height"] = 0.18
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        # self.rewards.feet_height_body_exp.weight = -4.9
        # self.rewards.feet_height_body_exp.params["target_height"] = -0.32
        # self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        self.rewards.feet_gait.weight = 2.0
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
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ArclabArcdogAdjustableLegBodyflatEnvCfg":
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
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)
