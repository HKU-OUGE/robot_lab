# Copyright (c) 2024-2025 Tianyang TANG
# SPDX-License-Identifier: Apache-2.0
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg, CommandsCfg, ObservationsCfg, TerminationsCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.envs.mdp.commands.commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
##
# Pre-defined configs
##
from robot_lab.assets.cuhklrl import CUHKLRL_SIRIUS_WHEEL_CFG  # isort: skip
from robot_lab.terrains.config.rough import *
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip

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
class CUHKLRLSiriusWCommandsCfg(CommandsCfg):
    """Action specifications for the MDP."""
    # # 新增基于地形的目标位姿命令
    # goal_pose = TerrainBasedPose2dCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(20.0, 20.0),  # 关键动作阶段不换目标；也可在事件里显式重采样
    #     debug_vis=False,                      # 可视化目标箭头
    #     simple_heading=True,                 # 默认正对目标；需要“贴边”时可改 False 并自行给 heading
    #     ranges=TerrainBasedPose2dCommandCfg.Ranges(
    #         heading=(0.0, 3.14), # useless if simple_heading=True
    #     ),
    # )
@configclass
class CUHKLRLSiriusWTerminationsCfg(TerminationsCfg):
    terminate = None


@configclass
class CUHKLRLSiriusWRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
            "command_threshold": 0.3,
        },
    )
    wheels_stop_without_cmd = RewTerm(
        func=mdp.wheels_stop_without_cmd,
        weight=0.0,   # 惩罚系数，可调
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_WHEEL"),
        },
    )
    wheel_slip = RewTerm(
        func=mdp.wheel_slip_l1,
        # weight=-0.1,   # 前2500
        weight=0.0, 
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_WHEEL"),
        },
    )
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    wheel_action_l2 = RewTerm(
        func=mdp.wheel_action_l2,
        weight=0.0,   #前2500
        # weight=0.0,
        params={"wheel_ids": [12, 13, 14, 15]},
    )
    abad_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 1.1,
            "command_threshold": 1.1,
        },
    )
    knee_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    feet_air_time_wheel = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_link"),
            "command_name": "base_velocity",
            "threshold": 0.75,
        },
    )
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    is_alive = RewTerm(func=mdp.is_alive_gated, weight=1.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-1.0)

@configclass
class CUHKLRLSiriusWObservationsCfg(ObservationsCfg):
    """Reward terms for the MDP."""
    @configclass
    class CUHKLRLSiriusWPolicyCfg(ObservationsCfg.PolicyCfg):
        obs_scan = None
        gait_command = ObsTerm(func=mdp.gait_mode_obs)
        # [Analysis Fix] Remove mixed joint_vel, split into legs and wheels
        joint_vel = None
        joint_vel_legs = ObsTerm(func=mdp.joint_vel_rel)
        joint_vel_wheels = ObsTerm(func=mdp.joint_vel_rel)
    @configclass
    class CUHKLRLSiriusWCriticCfg(ObservationsCfg.CriticCfg):
        obs_scan = None
        gait_command = ObsTerm(func=mdp.gait_mode_obs)
        # [Analysis Fix] Remove mixed joint_vel, split into legs and wheels
        joint_vel = None
        joint_vel_legs = ObsTerm(func=mdp.joint_vel_rel)
        joint_vel_wheels = ObsTerm(func=mdp.joint_vel_rel)

    policy: CUHKLRLSiriusWPolicyCfg = CUHKLRLSiriusWPolicyCfg()
    critic: CUHKLRLSiriusWCriticCfg = CUHKLRLSiriusWCriticCfg()



@configclass
class CUHKLRLSiriusWMoEEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: CUHKLRLSiriusWActionsCfg = CUHKLRLSiriusWActionsCfg()
    rewards: CUHKLRLSiriusWRewardsCfg = CUHKLRLSiriusWRewardsCfg()
    commands: CUHKLRLSiriusWCommandsCfg = CUHKLRLSiriusWCommandsCfg()
    observations: CUHKLRLSiriusWObservationsCfg = CUHKLRLSiriusWObservationsCfg()
    terminations: CUHKLRLSiriusWTerminationsCfg = CUHKLRLSiriusWTerminationsCfg()
    base_link_name = "trunk"
    foot_link_name = ".*_FOOT_link"
    calf_link_name = ".*_shank_link"
    wheel_joint_name = ".*_WHEEL"
    # fmt: off
    leg_joint_names = [
        "LF_HAA", "LF_HFE", "LF_KNEE",
        "LH_HAA", "LH_HFE", "LH_KNEE",
        "RF_HAA", "RF_HFE", "RF_KNEE",
        "RH_HAA", "RH_HFE", "RH_KNEE",
    ]
    wheel_joint_names = [
        "LF_WHEEL", "LH_WHEEL", "RF_WHEEL", "RH_WHEEL",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.only_positive_rewards = True
        CUHKLRL_SIRIUS_WHEEL_CFG.init_state.pos=(0.0, 0.0, 0.55)
        # CUHKLRL_SIRIUS_WHEEL_CFG.init_state.joint_pos={
        #     "LF_HAA": 0.00,
        #     "LH_HAA": 0.00,
        #     "RF_HAA": -0.00,
        #     "RH_HAA": -0.00,
        #     "LF_HFE": 0.52,
        #     "LH_HFE": -0.52,
        #     "RF_HFE": 0.52,
        #     "RH_HFE": -0.52,
        #     "LF_KNEE": -1.6,
        #     "LH_KNEE": 1.6,
        #     "RF_KNEE": -1.6,
        #     "RH_KNEE": 1.6,
        #     "LF_WHEEL": 0.00,
        #     "LH_WHEEL": 0.00,
        #     "RF_WHEEL": 0.00,
        #     "RH_WHEEL": 0.00,
        # }
        # ------------------------------Sence------------------------------
        # switch robot to unitree b2w
        self.scene.robot = CUHKLRL_SIRIUS_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ray_caster = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator=ROUGH_TERRAINS_CFG
        # self.observations.policy.height_scan = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names, preserve_order=True
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel
        self.observations.critic.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names, preserve_order=True
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0

        # self.observations.policy.base_lin_vel = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.6
        self.actions.joint_vel.scale = 1.5
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names[:-4]
        self.actions.joint_vel.joint_names = self.joint_names[-4:]

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (1.0, 1.0)
        self.events.randomize_actuator_gains = None

        # ------------------------------Rewards------------------------------
        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0 # 前1500轮
        # self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0.0
        # Joint penalties
        self.rewards.joint_deviation_hip_roll.weight = -1.0
        self.rewards.joint_deviation_hip_roll.params["asset_cfg"].joint_names = self.leg_joint_names
        # Joint penalties
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0.0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = 0.0
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.joint_vel_wheel_l2.weight = -2.5e-4
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.stand_still_without_cmd.weight = -1.0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        # Velocity-tracking rewards
        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*({self.foot_link_name})).*"
        ]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.5}
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.25}
        )
        # 1. Legs (Standard scale)
        self.observations.policy.joint_vel_legs.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel_legs.scale = 0.25 # Typically ~4 rad/s -> 1.0
        self.observations.critic.joint_vel_legs.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names, preserve_order=True
        )
        self.observations.critic.joint_vel_legs.scale = 0.25

        # 2. Wheels (Reduced scale for high speed)
        self.observations.policy.joint_vel_wheels.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel_wheels.scale = 0.05 # Typically ~20 rad/s -> 1.0
        self.observations.critic.joint_vel_wheels.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names, preserve_order=True
        )
        self.observations.critic.joint_vel_wheels.scale = 0.05
        # self.rewards.upward.weight = 1.0
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CUHKLRLSiriusWMoEEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_abad_link"]
        # self.terminations.illegal_contact = None
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None
        # ------------------------------Commands------------------------------
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (3.14, 3.14)
        self.curriculum.command_levels.params["range_multiplier"] = (1.0, 1.0)