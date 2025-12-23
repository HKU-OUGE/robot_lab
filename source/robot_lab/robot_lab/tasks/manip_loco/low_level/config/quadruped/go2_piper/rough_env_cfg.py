from isaaclab.utils import configclass

from robot_lab.tasks.manip_loco.low_level.go2_piper_lab_env_cfg import ManipulationLocomotionEnvCfg
from robot_lab.assets.go2_piper_cfg import GO2PIPER_CFG

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip

@configclass
class Go2PIPERRoughEnvCfg(ManipulationLocomotionEnvCfg):
    base_link_name = "base"
    trunk_link_name = "trunk"
    hip_link_name = ".*_hip"
    knee_link_name = ".*_calf"
    abad_link_name = ".*_thigh"
    foot_link_name = ".*_foot"
    arm_link_name = "link.*"

    # fmt: off
    # joint_names = [
    #     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    # ]
    joint_names_full_body = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint",
        "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint",
        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]

    joint_names_quadruped = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint",
        "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint",
        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]

    joint_names_arm = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    # fmt: on
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()

        self.scene.robot = GO2PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # event
        self.scene.height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        )
        # self.scene.height_scanner_base.prim_path = (
        #     "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # )
        # self.scene.terrain.terrain_generator=ROUGH_TERRAINS_CFG
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator=ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.events.push_robot = None

        # velocity command
        self.commands.base_velocity.curriculum_coeff = 20 # 4096
        # init
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (0.0, 0.0)
        # final
        self.commands.base_velocity.ranges_final.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
  
        # position command 
        self.commands.ee_pose.curriculum_coeff = 20 # 3000
        # init
        self.commands.ee_pose.ranges_init.pos_x = (0.1, 0.1)
        self.commands.ee_pose.ranges_init.pos_y = (-0.0, 0.0)
        self.commands.ee_pose.ranges_init.pos_z = (0.4, 0.4)
        # final
        self.commands.ee_pose.ranges_final.pos_x = (0.1, 0.5)
        self.commands.ee_pose.ranges_final.pos_y = (-0.35, 0.35)
        self.commands.ee_pose.ranges_final.pos_z = (-0.2, 0.45)
        # roll=(-0.0, 0.0),
        # pitch=(3.3 - 3.14 / 9,3.3 + 3.14 / 9),  # depends on end-effector axis
        # # pitch=(3.14 - 3.14 / 9,3.14 + 3.14 / 9),  # depends on end-effector axis
        # # pitch=(1.57 - 3.14 / 9,1.57 + 3.14 / 9),  # depends on end-effector axis
        # # pitch=(- 3.14 / 9, 3.14 / 9),  # depends on end-effector axis
        # yaw=(-3.14 / 9, 3.14 / 9),
        # self.commands.ee_pose.ranges_final.roll = (-0.7,0.7)
        # self.commands.ee_pose.ranges_final.pitch=(3.3 - 3.14 / 4,3.3 + 3.14 / 4)
        # self.commands.ee_pose.ranges_final.yaw=(-3.14 / 4, 3.14 / 4)


        # reward weight
        # arm
        self.rewards.end_effector_position_tracking.weight = 2.5 #2.5
        self.rewards.end_effector_orientation_tracking.weight = -1.0 #-1.5
        self.rewards.end_effector_action_rate.weight = -0.00 #-0.005 
        self.rewards.end_effector_action_smoothness.weight = -0.00 #-0.02

        # self.rewards.end_effector_position_tracking.weight = 2.5 #2.5
        # self.rewards.end_effector_orientation_tracking.weight = -0.5 #-1.5
        # self.rewards.end_effector_action_rate.weight = -0.00 #-0.005 
        # self.rewards.end_effector_action_smoothness.weight = -0.00 #-0.02
        
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
        self.rewards.action_rate_l2.weight = -0.00
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.height_reward.weight = -2.5
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.thigh_contact.weight = -1.5
        self.rewards.calf_contact.weight = -1.5
        self.rewards.hip_deviation.weight = -0.3
        self.rewards.joint_deviation.weight = -0.3
        self.rewards.arm_joint_deviation.weight = -0.01
        self.rewards.is_terminated.weight = -200.0

        # self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
            self.joint_names_full_body
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
            self.joint_names_full_body
        )

        # self.observations.policy.joint_pos_leg_rel.scale = 1.0
        # self.observations.policy.joint_vel.scale = 0.05
        # self.observations.policy.joint_pos_arm_abs.scale = 1.0
        # self.observations.policy.joint_pos_leg_rel.params["asset_cfg"].joint_names = (
        #     self.joint_names_quadruped
        # )
        # self.observations.policy.joint_pos_arm_abs.params["asset_cfg"].joint_names = (
        #     self.joint_names_arm
        # )

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-60.0, 60.0)}
        self.actions.joint_pos.joint_names = self.joint_names_quadruped
        self.actions.arm_pose.joint_names = self.joint_names_arm

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2PIPERRoughEnvCfg":
            self.disable_zero_weight_rewards()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            self.trunk_link_name,
            # self.abad_link_name,
            # self.knee_link_name,
            # self.hip_link_name,
            # self.arm_link_name
        ]
        # self.terminations.illegal_contact = None
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels.params["range_multiplier"] = (0.1,1.0)