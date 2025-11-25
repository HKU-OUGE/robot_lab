# Copyright (c) 2024-2025 Tang Tianyang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import Go2PIPERRoughEnvCfg


@configclass
class Go2PIPERFlatEnvCfg(Go2PIPERRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        # flat terrain 
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # velocity command
        self.commands.base_velocity.curriculum_coeff = 5000 # 4096
        # init
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (-0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (-0.0, 0.0)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-0.5, 0.5)
  
        # position command 
        self.commands.ee_pose.curriculum_coeff = 4000 # 3000
        # init
        self.commands.ee_pose.ranges_init.pos_x = (0.45, 0.45)
        self.commands.ee_pose.ranges_init.pos_y = (-0.0, 0.0)
        self.commands.ee_pose.ranges_init.pos_z = (0.50, 0.50)
        # final
        self.commands.ee_pose.ranges_final.pos_x = (0.2, 0.5)
        self.commands.ee_pose.ranges_final.pos_y = (-0.35, 0.35)
        self.commands.ee_pose.ranges_final.pos_z = (0.30, 0.7)


        # reward weight
        # arm
        self.rewards.end_effector_position_tracking.weight = 2.5 #2.5
        self.rewards.end_effector_orientation_tracking.weight = -1.0 #-1.5
        self.rewards.end_effector_action_rate.weight = -0.05 #-0.005 
        self.rewards.end_effector_action_smoothness.weight = -0.05 #-0.02
        
        # leg

        self.rewards.feet_gait.weight = 1.0
        # trotting
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.F_feet_air_time.weight = 0.0 #0.5
        self.rewards.R_feet_air_time.weight = 0.0 #0.5
        self.rewards.feet_height.weight = 0.0 #TODO
        self.rewards.feet_height_body.weight = 0.0 #TODO
        self.rewards.foot_contact.weight = 0.03 #0.003
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0


        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.height_reward.weight = -2.5
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.thigh_contact.weight = -0.5
        self.rewards.calf_contact.weight = -0.5
        self.rewards.hip_deviation.weight = -0.3
        self.rewards.joint_deviation.weight = -0.3
        self.rewards.arm_joint_deviation.weight = -0.1
        self.rewards.is_terminated.weight = -200


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2PIPERFlatEnvCfg":
            self.disable_zero_weight_rewards()
