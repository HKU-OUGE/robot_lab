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
        self.commands.base_velocity.curriculum_coeff = 4096 # 4096
        # init
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.1)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (-0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (-0.1, 0.1)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-0.5, 0.5)
  
        # position command 
        self.commands.ee_pose.curriculum_coeff = 4000 # 3000
        # init
        self.commands.ee_pose.ranges_init.pos_x = (0.2, 0.3)
        self.commands.ee_pose.ranges_init.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_init.pos_z = (0.55, 0.60)
        # final
        self.commands.ee_pose.ranges_final.pos_x = (0.15, 0.5)
        self.commands.ee_pose.ranges_final.pos_y = (-0.35, 0.35)
        self.commands.ee_pose.ranges_final.pos_z = (0.35, 0.7)


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2PIPERFlatEnvCfg":
            self.disable_zero_weight_rewards()
