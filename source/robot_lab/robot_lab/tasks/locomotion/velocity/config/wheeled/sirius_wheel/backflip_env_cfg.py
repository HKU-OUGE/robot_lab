# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import isaaclab.sim as sim_utils
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from ....mdp.backflip import track_pitch_velocity_exp, flip_completion, base_lin_vel_z, air_time, clamped_flip_completion, reverse_rotation, low_jump_penalty, early_landing, excessive_roll_penalty, pitch_angle_reward
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from robot_lab.assets.cuhklrl import CUHKLRL_SIRIUS_WHEEL_CFG  # isort: skip


@configclass
class CUHKLRLSiriusWActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class CUHKLRLSiriusWRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    upward_thrust = RewTerm(
        func=mdp.base_lin_vel_z,
        weight=12,
        params={"threshold": 1.5}  # Minimum upward velocity to reward (m/s)
    )

    air_time = RewTerm(
        func=mdp.air_time,
        weight=2.0
    )
    # ==== BACKFLIP ENFORCEMENT ====
    backflip_ang_vel = RewTerm(
        func=track_pitch_velocity_exp,
        weight=5.0,  # Brutal angular velocity tracking (was 2.0)
        params={"std": 0.3, "target": 4}  # Tighter tolerance
    )
    
    full_flip_bonus = RewTerm(
        func=mdp.clamped_flip_completion,
        weight=50.0  # Reward increases linearly with rotation
    )


    # ==== Progressive Rotation Reward ====
    flip_progress = RewTerm(
        func=mdp.flip_completion,
        weight=50.0,
        params={"threshold": 0.0}  # Track 0-100% progress
    )
    
    # ==== Penalize Reverse Rotation ====
    reverse_rotation = RewTerm(
        func=mdp.reverse_rotation,
        weight=-5.0  # Punish negative pitch angular velocity
    )


    # ==== ANTI-LAZINESS PENALTIES ====
    low_jump_penalty = RewTerm(  # NEW: Punish weak jumps
        func=mdp.low_jump_penalty,
        weight=-5.0
    )

    early_landing = RewTerm(  # NEW: Penalize ground contact <1m height
        func=mdp.early_landing,
        weight=-5.0
    )
    excessive_roll = RewTerm(
        func=mdp.excessive_roll_penalty,
        weight=-150.0,  # 可调，惩罚过大 roll
        params={"threshold": 0.7}  # 超过 0.5 rad (~28.6°) 就惩罚
    )
    pitch_angle_reward = RewTerm(
        func=mdp.pitch_angle_reward,
        weight=30.0,
        params={
            "target_angle": math.pi * 2,  # 完整后空翻
            "tolerance": 0.3
        }
    )



@configclass
class CUHKLRLSiriusWBackFlipEnvCfg(LocomotionVelocityRoughEnvCfg):
    episode_length_s = 7.0
    actions: CUHKLRLSiriusWActionsCfg = CUHKLRLSiriusWActionsCfg()
    rewards: CUHKLRLSiriusWRewardsCfg = CUHKLRLSiriusWRewardsCfg()
    base_link_name = "trunk"
    foot_link_name = ".*_FOOT"
    wheel_joint_name = ".*_WHEEL"
    # fmt: off
    joint_names = [
        "LF_HAA", "LF_HFE", "LF_KFE",
        "LH_HAA", "LH_HFE", "LH_KFE",
        "RF_HAA", "RF_HFE", "RF_KFE",
        "RH_HAA", "RH_HFE", "RH_KFE",
        "LF_WHEEL", "LH_WHEEL", "RF_WHEEL", "RH_WHEEL",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.only_positive_rewards = True

        # ------------------------------Sence------------------------------
        # switch robot to unitree b2w
        CUHKLRL_SIRIUS_WHEEL_CFG.init_state.pos=(0.0, 0.0, 0.35)
        CUHKLRL_SIRIUS_WHEEL_CFG.init_state.joint_pos={
            "LF_HAA": 0.00,
            "LH_HAA": 0.00,
            "RF_HAA": -0.00,
            "RH_HAA": -0.00,
            "LF_HFE": 0.67,
            "LH_HFE": -0.67,
            "RF_HFE": 0.67,
            "RH_HFE": -0.67,
            "LF_KFE": -2.1,
            "LH_KFE": 2.1,
            "RF_KFE": -2.1,
            "RH_KFE": 2.1,
            "LF_WHEEL": 0.00,
            "LH_WHEEL": 0.00,
            "RF_WHEEL": 0.00,
            "RH_WHEEL": 0.00,
        }
        self.scene.robot = CUHKLRL_SIRIUS_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.terrain.usd_path ="/home/ouge/Software/robot_lab/source/robot_lab/data/Terrains/Flat_Mountain_B/Flat_Mountain_B.usd"
        # self.scene.terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="usd",  # 使用 .usd 文件作为地形
        #     usd_path="/home/ouge/Software/robot_lab/source/robot_lab/data/Terrains/Flat_Mountain_A/Flat_Mountain_A.usd",  # 指定 .usd 文件路径
        #     collision_group=-1,
        #     visual_material=None,  # 可选：设置视觉材质
        #     max_init_terrain_level=5,
        #     debug_vis=False,
        # )
        # self.scene.terrain.terrain_generator = None
        # # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[self.wheel_joint_name]
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[self.wheel_joint_name]
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.5
        self.observations.policy.joint_vel.scale = 0.5
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 10.0
        self.actions.joint_vel.scale = 10.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names[:-4]
        self.actions.joint_vel.joint_names = self.joint_names[-4:]

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.60
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        # self.rewards.joint_torques_l2.weight = -2.5e-6
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        # self.rewards.joint_torques_wheel_l2.weight = -2.5e-6
        self.rewards.joint_torques_wheel_l2.weight = 0.0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_acc_wheel_l2.weight = 0.0
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = [self.wheel_joint_name]

        # Action penalties
        # self.rewards.action_rate_l2.weight = -0.00005
        self.rewards.action_rate_l2.weight = 0.0
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0.0005
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = -0.5
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -10.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.weight = 0.0
        self.rewards.joint_power.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.stand_still_without_cmd.weight = 0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        # self.rewards.joint_position_penalty.weight = -0.5
        self.rewards.joint_position_penalty.weight = 0.0
        self.rewards.joint_position_penalty.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_position_penalty.params["velocity_threshold"] = 100
        self.rewards.feet_height_exp.weight = 0
        self.rewards.feet_height_exp.params["target_height"] = 0.0
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body_exp.weight = 0
        self.rewards.feet_height_body_exp.params["target_height"] = -0.4
        self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("LF_FOOT", "RH_FOOT"), ("RF_FOOT", "LH_FOOT"))
        self.rewards.wheel_spin_in_air_penalty.weight = 0
        self.rewards.wheel_spin_in_air_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_spin_in_air_penalty.params["asset_cfg"].joint_names = [self.wheel_joint_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CUHKLRLSiriusWBackFlipEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # ------------------------------Terrains------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
