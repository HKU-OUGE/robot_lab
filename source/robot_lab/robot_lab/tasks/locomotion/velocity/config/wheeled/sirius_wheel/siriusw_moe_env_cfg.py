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
from robot_lab.assets.cuhklrl import CUHKLRL_SIRIUS_WHEEL_DELAY_CFG  # isort: skip
# [修改 1] 确保引入了 Isaac Lab 默认的崎岖地形配置
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
    # [修改 A] 恢复 Termination，但配置得更智能
    # 我们不希望机器人“落地成盒”，但也不希望它“倒扣着混日子”
    
    # 1. 非法接触：只针对机身 (Trunk/Base)。允许膝盖偶尔跪地。
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )

    # 2. 姿态失控：如果倾斜超过约 60度 (1.0 rad)，认为已翻车，直接重置
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )

@configclass
class CUHKLRLSiriusWRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""
    
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    
    wheels_stop_without_cmd = RewTerm(
        func=mdp.wheels_stop_without_cmd,
        weight=0.0,   
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )
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
    abad_pos_penalty = RewTerm(
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
    wheel_mirror = RewTerm(
        func=mdp.wheel_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["LF_WHEEL", "RF_WHEEL"], ["LH_WHEEL", "RH_WHEEL"]],
        },
    )
    
    # [修改 D] 启用膝盖/小腿接触惩罚
    # 之前是 0.0，导致机器人学会了用膝盖跪着走（因为 undesired_contacts 排除了 calf）
    # 现在给予惩罚，逼迫它伸直腿站起来
    knee_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,  # 惩罚权重
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_shank_link"), "threshold": 1.0},
    )

@configclass
class CUHKLRLSiriusWObservationsCfg(ObservationsCfg):
    """Reward terms for the MDP."""
    @configclass
    class CUHKLRLSiriusWPolicyCfg(ObservationsCfg.PolicyCfg):
        obs_scan = None
        
    @configclass
    class CUHKLRLSiriusWCriticCfg(ObservationsCfg.CriticCfg):
        obs_scan = None

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
        
        # Init state configuration
        # [修改 A] 提高出生高度，防止开局撞地 (0.55 -> 0.60)
        CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.init_state.pos=(0.0, 0.0, 0.60)
        
        # ------------------------------Scene------------------------------
        # switch robot
        self.scene.robot = CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ray_caster = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        
        # [修改 1] 启用复杂地形
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        # [修改 C] 强制从难度 0 开始，防止开局太难
        self.scene.terrain.max_init_terrain_level = 0
        
        # [修改 2] 启用 Height Scan (高程图)
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, scale=1.0, clip=(-2.0, 2.0)
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, scale=1.0, clip=(-2.0, 2.0)
        )
        
        # 其他观测配置
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
        self.observations.policy.joint_vel.func = mdp.joint_vel_rel
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel.scale = 0.5
        self.observations.critic.joint_vel.scale = 0.5

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = 0.25
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
                "z": (0.0, 0.2),
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
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (1.0, 1.0)

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0 
        self.rewards.ang_vel_xy_l2.weight = -0.1 # [修改] 稍微加强平稳性惩罚，减少抖动
        self.rewards.flat_orientation_l2.weight = -2.5 
        
        # [修改 B] 核心修改：增加基座高度奖励，强迫站立
        self.rewards.base_height_l2.weight = -2.0 
        self.rewards.base_height_l2.params["target_height"] = 0.55 
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        
        # [修改 3] 在 post_init 中应用轮子扭矩惩罚的参数
        self.rewards.joint_torques_wheel_l2.weight = -1.0e-5  
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_vel_l2.weight = -2.5e-6
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        
        # 适当增加轮子速度惩罚，防止空转
        self.rewards.joint_vel_wheel_l2.weight = -1.0e-7 
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = -5.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -1e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.joint_pos_penalty.weight = -0.1
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.abad_pos_penalty.weight = 0.0
        self.rewards.abad_pos_penalty.params["asset_cfg"].joint_names = [
            "LF_HAA", "RF_HAA", "LH_HAA", "RH_HAA", 
            "LF_HFE", "RF_HFE", "LH_HFE", "RH_HFE", 
        ]
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.feet_continue_contact.weight = 0.0
        self.rewards.feet_continue_contact.params["expect_contact_num"] = 4
        self.rewards.feet_continue_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        # Action penalties
        # [修改] 稍微降低动作平滑性惩罚 (-0.01 -> -0.005)
        # 机器人现在抖动很厉害 (action_rate_l2 高)，降低这个权重可以让它更自由地调整姿态，
        # 而不是因为怕惩罚而“僵住”或剧烈震荡
        self.rewards.action_rate_l2.weight = -0.005

        # Contact sensor
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*({self.foot_link_name})).*"
        ]

        self.rewards.knee_undesired_contacts.weight = 0.0
        self.rewards.knee_undesired_contacts.params["sensor_cfg"].body_names = self.calf_link_name
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=5.0, params={"command_name": "base_velocity", "std": 0.5}
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.5}
        )
        
        # Others (保持原样)
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("LF_FOOT_link", "RF_FOOT_link"), ("LH_FOOT_link", "RH_FOOT_link"))
        self.rewards.upward.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["RF_(HAA|HFE|KNEE).*", "LH_(HAA|HFE|KNEE).*"],
            ["LF_(HAA|HFE|KNEE).*", "RH_(HAA|HFE|KNEE).*"],
        ]
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        if self.__class__.__name__ == "CUHKLRLSiriusWMoEEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation.params["limit_angle"] = 1.2

        # ------------------------------Commands & Terrain------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (3.14, 3.14)
        self.curriculum.command_levels.params["range_multiplier"] = (1.0, 1.0)