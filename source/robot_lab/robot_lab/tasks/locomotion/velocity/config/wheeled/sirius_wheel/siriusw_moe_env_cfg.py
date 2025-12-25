# Copyright (c) 2024-2025 Tianyang TANG
# SPDX-License-Identifier: Apache-2.0
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
# 注意：确保 mdp 指向修改后的 rewards.py
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg, CommandsCfg, ObservationsCfg, TerminationsCfg, EventCfg
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
from robot_lab.assets.cuhklrl import CUHKLRL_SIRIUS_WHEEL_DELAY_CFG  # isort: skip
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
    pass

@configclass
class CUHKLRLSiriusWTerminationsCfg(TerminationsCfg):
    quad_orientation = DoneTerm(
        func=mdp.bad_orientation_quadruped,
        params={"limit_roll": 1.0, "limit_pitch": 1.0},
    )
    
    biped_illegal_contact = DoneTerm(
        func=mdp.illegal_contact_biped,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["trunk", "LF_.*", "RF_.*", "LH_abad_link", "LH_thigh_link", "RH_abad_link", "RH_thigh_link"]
            ),
            "threshold": 1.0,
            "grace_period_s": 3.0,
        },
    )

@configclass
class CUHKLRLSiriusWRewardsCfg:
    """Reward terms for the MDP."""
    # === 1. Global Rewards ===
    is_alive = RewTerm(func=mdp.is_alive_gated, weight=1.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # torque_exceed = RewTerm(func=mdp.torque_exceed_limit, weight=-2.0)
    
    # [Check] Aligned with formula ||a_lat - a||_2 (Norm)
    action_rate_l2 = RewTerm(func=mdp.action_rate_norm, weight=-0.03)

    # ==============================================================================
    # 2. Quadrupedal Rewards (Gait 0)
    # ==============================================================================
    track_lin_vel_xy_quad = RewTerm(
        func=mdp.track_lin_vel_xy_gated, weight=7.0, 
        params={"std": 0.5, "command_name": "base_velocity", "gait_mode": 0}
    )
    track_ang_vel_z_quad = RewTerm(
        func=mdp.track_ang_vel_z_gated, weight=2.5,
        params={"std": 0.5, "command_name": "base_velocity", "gait_mode": 0}
    )
    
    # Regularization
    joint_pos_quad = RewTerm(
        func=mdp.joint_pos_penalty_gated, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 0}
    )
    joint_vel_quad = RewTerm(
        func=mdp.joint_vel_penalty_gated, weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 0}
    )
    
    # [Check] Aligned with formula || q_ddot ||_2 (Norm)
    joint_acc_legs_quad = RewTerm(
        func=mdp.joint_acc_norm_gated, weight=-2.0e-6, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 0}
    )

    joint_acc_wheels_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.0e-8, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_WHEEL"])}
    )
    
    joint_power_wheels = RewTerm(
        func=mdp.joint_power, weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_WHEEL"])}
    )
    joint_power_legs = RewTerm(
        func=mdp.joint_power,
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"])} 
    )
    # wheel_spin_in_air = RewTerm(
    #     func=mdp.wheel_spin_in_air_penalty, weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_WHEEL"]),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT_link"])
    #     }
    # )

    ang_vel_xy_stability = RewTerm(func=mdp.ang_vel_xy_stability, weight=-0.2)
    
    # feet_in_air_quad = RewTerm(
    #     func=mdp.feet_in_air_quad, weight=-0.05,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", 
    #             body_names=[
    #                 "LF_FOOT_link", "LH_FOOT_link", "RF_FOOT_link", "RH_FOOT_link", 
    #                 "LF_shank_link", "LH_shank_link", "RF_shank_link", "RH_shank_link"
    #             ]
    #         )
    #     }
    # )

    # 这让机器人在四足模式下没有速度指令时，强制轮子静止
    wheels_stand_still_quad = RewTerm(
        func=mdp.wheels_stop_without_cmd_gated,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_WHEEL"]),
            "gait_mode": 0,
            "command_threshold": 0.1
        }
    )
    
    front_hip_pos_quad = RewTerm(
        func=mdp.joint_deviation_gated, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LF_H.*", "RF_H.*"]), "gait_mode": 0}
    )
    rear_hip_pos_quad = RewTerm(
        func=mdp.joint_deviation_gated, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LH_H.*", "RH_H.*"]), "gait_mode": 0}
    )
    
    base_height_quad = RewTerm(
        func=mdp.base_height_quad, weight=-0.1,
        params={"target_height": 0.55, "asset_cfg": SceneEntityCfg("robot")} 
    )
    
    balance_quad = RewTerm(
        func=mdp.balance_quad, weight=-2.0e-5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LF_FOOT_link", "LH_FOOT_link", "RF_FOOT_link", "RH_FOOT_link"])}
    )
    
    # [Correction] Changed to joint_limit_gated (Count) to match formula || boolean_vec ||_1
    # Previously used joint_limit_l1_gated (Magnitude) which was misaligned with the boolean logic in MD.
    joint_limit_quad = RewTerm(
        func=mdp.joint_limit_gated, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 0}
    )

    # ==============================================================================
    # 3. Bipedal Rewards (Gait 1)
    # ==============================================================================
    track_lin_vel_xy_biped = RewTerm(
        func=mdp.track_lin_vel_xy_biped_gated, weight=3.0, 
        params={"std": 0.5, "command_name": "base_velocity", "target_height": 1.0}
    )
    # [Check] Use specialized biped tracking function (handles posture/height gate) - Aligned
    track_ang_vel_z_biped = RewTerm(
        func=mdp.track_ang_vel_z_biped_gated, weight=2.5,
        params={"std": 0.5, "command_name": "base_velocity", "target_height": 1.0}
    )
    
    biped_orientation = RewTerm(func=mdp.biped_stand_orientation, weight=1.0)
    
    biped_height = RewTerm(
        func=mdp.biped_stand_height_linear, weight=0.8, 
        params={"target_height": 1.0}
    ) 
    
    rear_air_biped = RewTerm(
        func=mdp.rear_air_biped, weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=["LH_FOOT_link", "RH_FOOT_link", "LH_shank_link", "RH_shank_link"]
            )
        }
    )
    
    front_hip_pos_biped = RewTerm(
        func=mdp.joint_deviation_gated, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LF_H.*", "RF_H.*"]), "gait_mode": 1}
    )
    rear_hip_pos_biped = RewTerm(
        func=mdp.joint_deviation_gated, weight=-0.18,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LH_H.*", "RH_H.*"]), "gait_mode": 1}
    )
    
    rear_pos_balance_biped = RewTerm(
        func=mdp.rear_pos_balance_biped, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LH_HAA", "LH_HFE", "LH_KNEE", "RH_HAA", "RH_HFE", "RH_KNEE"])}
    )
    
    biped_arm_pos = RewTerm(
        func=mdp.biped_front_legs_lift, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LF_.*", "RF_.*"]), "time_threshold_s": 1.0}
    )
    
    # [Check] Biped front vel (Squared + Time Gating) - Aligned
    biped_arm_vel = RewTerm(
        func=mdp.biped_front_joint_vel, weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LF_.*", "RF_.*"]), "gait_mode": 1, "time_threshold_s": 1.0}
    )
    
    # [Check] Front Joint Acc (Squared + Time Gating) - Aligned
    biped_arm_acc = RewTerm(
        func=mdp.biped_front_joint_acc, weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LF_.*", "RF_.*"]), "gait_mode": 1, "time_threshold_s": 1.0}
    )

    legs_energy_biped = RewTerm(
        func=mdp.joint_power_gated, weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 1}
    )
    
    # [Check] Biped joint limits uses count - Aligned
    joint_limit_biped = RewTerm(
        func=mdp.joint_limit_gated, weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 1}
    )
    
    joint_vel_biped = RewTerm(
        func=mdp.joint_vel_penalty_gated, weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 1}
    )
    
    # [Check] Aligned with formula || q_ddot ||_2 (Norm)
    joint_acc_legs_biped = RewTerm(
        func=mdp.joint_acc_norm_gated, weight=-3.0e-6, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_(HAA|HFE|KNEE)"]), "gait_mode": 1}
    )

    # [Check] Aligned with formula Sum( Indicator( ||f|| > 1 ) )
    biped_collision = RewTerm(
        func=mdp.undesired_contacts_count, weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0}
    )

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
class CUHKLRLSiriusWEventsCfg(EventCfg):
    """Events terms for the MDP."""
    reset_gait_quad = EventTerm(
        func=mdp.set_gait_mode_fixed,
        mode="reset",
        params={"mode_val": 0.0},
    )

    interval_gait_flip = EventTerm(
        func=mdp.set_gait_mode_flip,
        mode="interval",
        interval_range_s=(15.0, 20.0), 
    )


@configclass
class CUHKLRLSiriusWMoEEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: CUHKLRLSiriusWActionsCfg = CUHKLRLSiriusWActionsCfg()
    rewards: CUHKLRLSiriusWRewardsCfg = CUHKLRLSiriusWRewardsCfg()
    commands: CUHKLRLSiriusWCommandsCfg = CUHKLRLSiriusWCommandsCfg()
    observations: CUHKLRLSiriusWObservationsCfg = CUHKLRLSiriusWObservationsCfg()
    terminations: CUHKLRLSiriusWTerminationsCfg = CUHKLRLSiriusWTerminationsCfg()
    events: CUHKLRLSiriusWEventsCfg = CUHKLRLSiriusWEventsCfg()
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
        
        # Init state configuration
        CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.init_state.pos=(0.0, 0.0, 0.55)
        
        # ------------------------------Scene------------------------------
        # switch robot
        self.scene.robot = CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ray_caster = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, scale=1.0, clip=(-2.0, 2.0)
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, scale=1.0, clip=(-2.0, 2.0)
        )
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

        # ------------------------------Actions------------------------------
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
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (1.0, 1.0)

        # ------------------------------Rewards------------------------------


        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        if self.__class__.__name__ == "CUHKLRLSiriusWMoEEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact = None
        # ------------------------------Commands & Terrain------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.curriculum.command_levels = None