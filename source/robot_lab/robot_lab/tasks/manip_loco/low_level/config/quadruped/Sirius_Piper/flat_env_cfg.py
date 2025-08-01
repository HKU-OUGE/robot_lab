# Copyright (c) 2024-2025 Tang Tianyang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.tasks.manip_loco.low_level.wbc_env_cfg import(
    LowLevelWBCEnvCfg
)

from robot_lab.assets.arclab_atec import ARCLAB_ATEC_CFG


@configclass
class CUHKLRLSiriusPiperFlatEnvCfg(LowLevelWBCEnvCfg):

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
        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = ARCLAB_ATEC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
            self.joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
            self.joint_names
        )
        self.actions.joint_pos.joint_names = (
            self.joint_names
        )
class CUHKLRLSiriusPiperFlatEnvCfg_PLAY(CUHKLRLSiriusPiperFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.events.reset_base = None
        self.commands.ee_pose.is_QuadrupedARM = False
        self.commands.base_velocity.is_QuadrupedARM = False
        self.commands.ee_pose.is_QuadrupedARM_Play = True
        
        # self.commands.base_velocity.resampling_time_range = (5.0,10.5)
        # self.commands.base_velocity.rel_standing_envs = 0.2
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
       
        # self.commands.ee_pose.resampling_time_range = (2.0,5.0)
        # self.commands.ee_pose.ranges.pos_x = (0.45, 0.45)
        # self.commands.ee_pose.ranges.pos_y = (0.0, 0.0)
        # self.commands.ee_pose.ranges.pos_z = (0.3, 0.3)
