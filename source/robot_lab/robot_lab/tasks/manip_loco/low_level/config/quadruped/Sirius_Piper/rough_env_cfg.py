# Copyright (c) 2024-2025 Tang Tianyang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_lab.tasks.manip_loco.low_level.sirius_piper_lab_env_cfg import(
    ManipulationLocomotionEnvCfg
)

##
# Pre-defined configs
##
# use cloud assets
# from isaaclab_assets.robots.unitree import CUHKLRLSirius_CFG  # isort: skip
# use local assets
# from robot_lab.assets.cuhklrl import CUHKLRL_SIRIUS_CFG  # isort: skip
from robot_lab.assets.arclab_atec import ARCLAB_ATEC_CFG


@configclass
class CUHKLRLSiriusPiperRoughEnvCfg(ManipulationLocomotionEnvCfg):
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
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree a1
        self.scene.robot = ARCLAB_ATEC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )
        self.scene.height_scanner_base.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )


        self.events.push_robot = None

        # command
        self.commands.ee_pose.is_Go2ARM_Flat = False #TODO
        # velocity command
        # init
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (0.0, 0.0)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (0.1, 0.5)
        self.commands.base_velocity.ranges_final.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges_final.ang_vel_z = (0.0, 0.0)
  
        # position command 
        # init
        self.commands.ee_pose.ranges_init.pos_x = (0.45, 0.5)
        self.commands.ee_pose.ranges_init.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_init.pos_z = (0.35, 0.4)
        # final
        self.commands.ee_pose.ranges_final.pos_x = (0.45, 0.5)
        self.commands.ee_pose.ranges_final.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_final.pos_z = (0.35, 0.4)


        # reward weight
        # arm
        self.rewards.end_effector_position_tracking.weight = 2.5
        self.rewards.end_effector_orientation_tracking.weight = -1.5
        self.rewards.end_effector_action_rate.weight = -0.005
        self.rewards.end_effector_action_smoothness.weight = -0.02
        # leg
        self.rewards.tracking_lin_vel_x_l1.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.foot_contact.weight = 0.003
        self.rewards.hip_deviation.weight = -0.4
        self.rewards.joint_deviation.weight = -0.04
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.height_reward.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -1.0

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

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        # self.rewards.is_terminated.weight = 0

        # # Root penalties
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.05
        # self.rewards.flat_orientation_l2.weight = -3.5
        # self.rewards.base_height_l2.weight = -3.5
        # self.rewards.base_height_l2.params["target_height"] = 0.50
        # self.rewards.base_height_l2.params["asset_cfg"].body_names = [
        #     self.base_link_name
        # ]
        # self.rewards.body_lin_acc_l2.weight = 0
        # self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [
        #     self.base_link_name
        # ]
        # # # 添加对 a_1, a_3, a_5, a_7 的目标关节角奖励，90度
        # # self.rewards.create_joint_deviation_l1_rewterm(
        # #     attr_name="knee_target_pose_l1",
        # #     weight=-1.0,
        # #     joint_names_pattern=["a_1", "a_3", "a_5", "a_7"]
        # # )

        # # Joint penaltie
        # # self.rewards.joint_torques_l2.weight = -2.5e-5
        # # 测试 暂时取消此惩罚
        # self.rewards.joint_torques_l2.weight = 0
        # # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        # self.rewards.joint_vel_l2.weight = -0.005
        # self.rewards.joint_acc_l2.weight = -3.0e-8
        # # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        # self.rewards.joint_pos_limits.weight = -5.0
        # # 禁止超速
        # self.rewards.joint_vel_limits.weight = -5.0

        # # Action penalties
        # self.rewards.action_rate_l2.weight = -0.01
        # # UNUESD self.rewards.action_l2.weight = 0.0

        # # Contact sensor
        # self.rewards.undesired_contacts.weight = -1.5
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        #     f"^(?!.*{self.foot_link_name}).*"
        # ]
        # # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        # #     "body", "trunk", "hip.*", "upper.*", "lower.*"
        # # ]

        # self.rewards.contact_forces.weight = 0.0005
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]

        # # Velocity-tracking rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 6.0
        # self.rewards.track_ang_vel_z_exp.weight = 6.0

        # # Others
        # self.rewards.feet_air_time.weight = 4.0
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        # self.rewards.feet_air_time.params["mode_time"] = 0.32
        # self.rewards.feet_contact.weight = 0.5
        # self.rewards.feet_contact.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        # self.rewards.feet_stumble.weight = -10.0
        # self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        # self.rewards.feet_slide.weight = -0.5
        # self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # # self.rewards.joint_power.weight = -2e-5
        # # 测试 暂时取消
        # self.rewards.joint_power.weight = -2e-5
        # self.rewards.stand_still_without_cmd.weight = 0.1
        # self.rewards.joint_position_penalty.weight = -0.1
        # # self.rewards.joint_position_penalty.weight = 0.0
        # self.rewards.feet_height_exp.weight = 4.0
        # self.rewards.feet_height_exp.params["target_height"] = 0.20
        # self.rewards.feet_height_exp.params["asset_cfg"].body_names = [
        #     self.foot_link_name
        # ]  
        # self.rewards.feet_height_body_exp.weight = 2.5
        # self.rewards.feet_height_body_exp.params["target_height"] = -0.38
        # self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        # self.rewards.feet_gait.weight = 5.0
        # # trotting
        # self.rewards.feet_gait.params["synced_feet_pair_names"] = (
        #     ("FL_foot", "RR_foot"),
        #     ("FR_foot", "RL_foot"),
        # )
        # # pronking
        # # self.rewards.feet_gait.params["synced_feet_pair_names"] = (
        # #     ("FL_foot", "FR_foot"),
        # #     ("RR_foot", "RL_foot"),
        # # ) 
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CUHKLRLSiriusPiperRoughEnvCfg":
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
