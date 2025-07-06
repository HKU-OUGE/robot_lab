# Copyright (c) 2024-2025 Tang Tianyang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
# use cloud assets
# from isaaclab_assets.robots.unitree import CUHKLRLSirius_CFG  # isort: skip
# use local assets
from robot_lab.assets.cuhklrl import CUHKLRL_SIRIUS_CFG  # isort: skip


@configclass
class CUHKLRLSiriusRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    trunk_link_name = "trunk"
    hip_link_name = ".*_hip"
    knee_link_name = ".*_calf"
    abad_link_name = ".*_thigh"
    foot_link_name = ".*_foot"

    # fmt: off
    # joint_names = [
    #     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    # ]
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
        self.scene.robot = CUHKLRL_SIRIUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )
        self.scene.height_scanner_base.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )
        # # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (
        #     0.025,
        #     0.1,
        # )
        # self.scene.terrain.terrain_generator.sub_terrains[
        #     "random_rough"
        # ].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = (
        #     0.01
        # )

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
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_apply_external_force_torque.params[
            "asset_cfg"
        ].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -3.5
        self.rewards.base_height_l2.weight = -3.5
        self.rewards.base_height_l2.params["target_height"] = 0.52
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        # # 添加对 a_1, a_3, a_5, a_7 的目标关节角奖励，90度
        # self.rewards.create_joint_deviation_l1_rewterm(
        #     attr_name="knee_target_pose_l1",
        #     weight=-1.0,
        #     joint_names_pattern=["a_1", "a_3", "a_5", "a_7"]
        # )

        # Joint penaltie
        # self.rewards.joint_torques_l2.weight = -2.5e-5
        # 测试 暂时取消此惩罚
        self.rewards.joint_torques_l2.weight = 0
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = -0.005
        self.rewards.joint_acc_l2.weight = -3.0e-8
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_pos_limits.weight = -5.0
        # 禁止超速
        self.rewards.joint_vel_limits.weight = -5.0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*{self.foot_link_name}).*"
        ]
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        #     "body", "trunk", "hip.*", "upper.*", "lower.*"
        # ]

        self.rewards.contact_forces.weight = 0.0005
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.weight = 6.0

        # Others
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_air_time.params["mode_time"] = 0.32
        self.rewards.feet_contact.weight = 0.5
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_stumble.weight = -10.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.joint_power.weight = -2e-5
        # 测试 暂时取消
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still_without_cmd.weight = 0.1
        self.rewards.joint_position_penalty.weight = -0.1
        # self.rewards.joint_position_penalty.weight = 0.0
        self.rewards.feet_height_exp.weight = 4.0
        self.rewards.feet_height_exp.params["target_height"] = 0.20
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]  
        self.rewards.feet_height_body_exp.weight = 2.5
        self.rewards.feet_height_body_exp.params["target_height"] = -0.38
        self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_gait.weight = 5.0
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
        if self.__class__.__name__ == "CUHKLRLSiriusRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            self.trunk_link_name,
            self.abad_link_name,
            self.knee_link_name,
            self.hip_link_name,
        ]
        # ------------------------------Commands------------------------------
