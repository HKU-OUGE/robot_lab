# Copyright (c) 2024-2025 Tianyang TANG
# SPDX-License-Identifier: Apache-2.0
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
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
from robot_lab.terrains.config.rough import *
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
import math

# [Updated] Bipedal Stand Pose (Converted from Degrees to Radians)
# LF/RF: HFE 100 deg -> 1.75 rad, KNEE -130 deg -> -2.27 rad (Folded)
# LH/RH: HFE 55 deg -> 0.96 rad,  KNEE 45 deg -> 0.79 rad   (Squat Support)
_BIPED_TARGET_JOINTS = [
    0.00,  1.75, -2.27,  # LF
    0.00,  0.96,  0.79,  # LH
    -0.00, 1.75, -2.27,  # RF
    -0.00, 0.96,  0.79   # RH
]


@configclass
class CUHKLRLSiriusWActionsCfg(ActionsCfg):
    """Action specifications for t    # feet_contact_quad = RewTerm(
    #     func=mdp.feet_contact_gated,
    #     weight=2.0,  # 给予正向奖励，数值可根据训练情况调整
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_link"),
    #         "gait_mode": 0,
    #         "threshold": 1.0,
    #     },
    # )he MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )

@configclass
class CUHKLRLSiriusWCommandsCfg(CommandsCfg):
    """Action specifications for the MDP."""
    # goal_pose = ... (commented out as in original)

@configclass
class CUHKLRLSiriusWTerminationsCfg(TerminationsCfg):
    # gate_collision_trot_only = DoneTerm(
    #     func=mdp.terminate_gate_contact,
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk"]),
    #         "gait_mode": 0,
    #     },
    # )
    quad_bad_orientation = DoneTerm(
        func=mdp.bad_orientation_gated, # 指向刚才添加的函数
        params={
            "limit_roll": 1.2,   # 约 68度
            "limit_pitch": 1.2,  # 约 68度
            "gait_mode": 0,      # 0 代表 Quadruped 模式
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    illegal_contact = None

@configclass
class CUHKLRLSiriusWRewardsCfg:
    """Reward terms for the MDP."""
    
    # ==============================================================================================
    # 1. Global Penalties (Apply to both modes)
    # ==============================================================================================
    
    # Alive / Terminated
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-1.0)
    # leg_action_l2 = RewTerm(
    #     func=mdp.wheel_action_l2,
    #     weight=-0.005,
    #     # weight=0.0,
    #     params={"wheel_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    # )
    wheel_action_l2 = RewTerm(
        func=mdp.wheel_action_l2,
        weight=-0.001,
        # weight=0.0,
        params={"wheel_ids": [12, 13, 14, 15]},
    )
    wheel_sync_penalty = RewTerm(
        func=mdp.wheel_sync_penalty,
        weight=-0.005,
        # weight=0.0,
        params={"wheel_ids": [12, 13, 14, 15]},
    )
    # Torque / Acc Limits
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=0.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    leg_joint_power = RewTerm(
        func=mdp.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )
    wheel_joint_power = RewTerm(
        func=mdp.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    # Knee collision is always bad
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=0.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    # )

    # ==============================================================================================
    # 2. Quadruped Mode Rewards (Gait Mode = 0)
    # ==============================================================================================
    
    # Stability in Quad mode
    lin_vel_z_l2_quad = RewTerm(
        func=mdp.lin_vel_z_l2_gated, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot"), "gait_mode": 0}
    )
    ang_vel_xy_l2_quad = RewTerm(
        func=mdp.ang_vel_xy_l2_gated, 
        weight=-0.05, 
        params={"asset_cfg": SceneEntityCfg("robot"), "gait_mode": 0}
    )
    flat_orientation_l2_quad = RewTerm(
        func=mdp.flat_orientation_l2_gated, 
        weight=0.0, 
        params={"asset_cfg": SceneEntityCfg("robot"), "gait_mode": 0}
    )
    
    # Joint Regularization (Quad only)
    # Strict adherence to default pose when in Quad mode
    leg_deviation_l1_quad = RewTerm(
        func=mdp.joint_deviation_l1_gated,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""), 
            "gait_mode": 0  # 0 for Quadruped mode, 1 for Biped
        },
    )
    leg_pos_penalty_adaptive_quad = RewTerm(
        func=mdp.joint_pos_penalty_adaptive_gated,
        weight=-1.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("height_scanner_base"), 
            "gait_mode": 0,
            
            # 关键参数调整
            "h_free_min": 0.15,   # 根据你的 Floating Ring 高度差调整
            "h_free_max": 0.45,   
            "alpha": 0.05,        # 降低平滑系数，防止闪烁
            # "pitch_threshold": ... 已移除
        }
    )
    # Tracking (Quad)
    track_lin_vel_xy_quad = RewTerm(
        func=mdp.track_lin_vel_xy_exp_gated, 
        weight=3.5, 
        params={"command_name": "base_velocity", "std": 0.4, "gait_mode": 0}
    )
    track_ang_vel_z_quad = RewTerm(
        func=mdp.track_ang_vel_z_exp_gated, 
        weight=3.5, 
        params={"command_name": "base_velocity", "std": 0.4, "gait_mode": 0}
    )
    action_rate_l2_quad = RewTerm(
        func=mdp.action_rate_l2_gated,
        weight=-0.01,
        params={"gait_mode": 0}
    )
    undesired_contacts_quad = RewTerm(
        func=mdp.undesired_contacts_gated,
        weight=0.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[""]), 
            "threshold": 1.0,
            "gait_mode": 0
        },
    )
    leg_joint_vel_quad = RewTerm(
        func=mdp.joint_vel_penalty_gated,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[""]), # 腿关节
            "gait_mode": 0
        }
    )
    abad_strict_straight_quad = RewTerm(
        func=mdp.joint_deviation_abad_straight_gated,
        weight=-5.0,  # 给予较强的负向惩罚 (建议从 -0.5 开始调试，不够就加到 -2.0)
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_HAA"), # 正则匹配所有 HAA 关节
            "command_name": "base_velocity",
            "gait_mode": 0,       # 仅在四足模式生效
            "rot_threshold": 0.1, # 当转向命令小于 0.1 rad/s 时生效
        },
    )
    # wheel_sync_straight_quad = RewTerm(
    #     func=mdp.wheel_sync_when_straight_gated,
    #     weight=-0.05,  # 负值表示惩罚，数值大小需根据训练曲线调整
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_WHEEL"), # 选中所有车轮关节
    #         "command_name": "base_velocity",
    #         "gait_mode": 0,
    #         "rot_threshold": 0.1, # 当旋转指令小于 0.1 rad/s 时生效
    #     },
    # )
    # feet_contact_quad = RewTerm(
    #     func=mdp.feet_contact_gated,
    #     weight=2.0,  # 给予正向奖励，数值可根据训练情况调整
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_link"),
    #         "gait_mode": 0,
    #         "threshold": 1.0,
    #     },
    # )
    # wheels_stop_quad = RewTerm(
    #     func=mdp.wheels_stop_without_cmd_gated, # 需确认 rewards.py 中有此 gated 版本，或使用通用 gated 包装
    #     weight=-0.01,  # 给予较强惩罚
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_WHEEL"]), 
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1,
    #         "gait_mode": 0
    #     },
    # )
    # ==============================================================================
    # 3. Bipedal Rewards (Gait 1) - Standing & Driving
    # ==============================================================================
    
    # [平衡与姿态]
    vertical_orientation_l2_biped = RewTerm(
        func=mdp.orientation_align_gravity_gated,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "target_gravity": [-1.0, 0.0, 0.0], # 假设机器人Z轴朝前，-X轴朝重力方向（需要确认你的机器人坐标系）
            # 注意：通常直立是 Z 轴反向重力。如果你的 orientation_align_gravity_gated 内部是 projected gravity
            # 且你的机器人站立时基座 X 轴朝上？或者 Z 轴朝上？
            # 假设标准四足，站立（双足）时是屁股着地头朝上，此时基座通常 Pitch=90度。
            # 如果是 Pitch=90度，那么 projected gravity on base frame 应该是 X= -1 (重力指向 -X)
            "gait_mode": 1
        },
    )
    
    base_height_l2_biped = RewTerm(
        func=mdp.base_height_l2_gated,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "target_height": 1.0, 
            "gait_mode": 1
        },
    )

    # [Task Space Hand Position] - 解决手的位置不理想
    hand_height_biped = RewTerm(
        func=mdp.feet_height_exp_gated,
        weight=5.0, # 正向奖励，鼓励到达
        params={
            "target_height": 0.5, # 手的高度目标，比基座稍高
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=["(LF|RF)_FOOT_link"]), # 前脚/手
            "gait_mode": 1
        }
    )

    abad_deviation_l1_biped = RewTerm(
        func=mdp.joint_deviation_l1_gated,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""), 
            "gait_mode": 1  # 0 for Quadruped mode, 1 for Biped
        },
    )
    # 1. 加大动作平滑性惩罚
    action_rate_l2_biped = RewTerm(
        func=mdp.action_rate_l2_gated,
        weight=-0.01, # 原 -0.01，增加5倍
        params={"gait_mode": 1}
    )
    
    # 2. 惩罚前腿（手）的关节速度，让其保持静止
    arm_joint_vel_biped = RewTerm(
        func=mdp.joint_vel_penalty_gated,
        weight=-0.01, # 较强惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(LF|RF)_(HAA|HFE|KNEE)"]), # 前腿关节
            "gait_mode": 1
        }
    )
    
    # 3. 惩罚后腿的关节速度（减少抽搐），但要比前腿轻，因为需要平衡
    legs_joint_vel_biped = RewTerm(
        func=mdp.joint_vel_penalty_gated,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(LH|RH)_(HAA|HFE|KNEE)"]), # 后腿关节
            "gait_mode": 1
        }
    )

    # [Tracking]
    # track_lin_vel_xy_biped = RewTerm(
    #     func=mdp.track_lin_vel_xy_heading_posture_gated,
    #     weight=5.0,
    #     params={"command_name": "base_velocity", "std": 0.4, "gait_mode": 1},
    # )
    
    # track_ang_vel_z_biped = RewTerm(
    #     func=mdp.track_ang_vel_z_world_posture_gated,
    #     weight=3.0,
    #     params={"command_name": "base_velocity", "std": 0.4, "gait_mode": 1},
    # )
    # joint_deviation_l1_biped = RewTerm(
    #     func=mdp.joint_pos_target_l1_posture_robust,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[
    #             "LF_HAA", "LF_HFE", "LF_KNEE",
    #             "LH_HAA", "LH_HFE", "LH_KNEE",
    #             "RF_HAA", "RF_HFE", "RF_KNEE",
    #             "RH_HAA", "RH_HFE", "RH_KNEE",
    #         ]),
    #         "gait_mode": 1,
    #         "target_pos_list": _BIPED_TARGET_JOINTS, # 引用上方更新后的收臂姿态
    #         "target_gravity": [-1.0, 0.0, 0.0],
    #         "gravity_threshold": 0.5,
    #     },
    # )
    track_wheel_drive_biped = RewTerm(
        func=mdp.track_rear_wheel_velocity_exp_gated, # Ensure this is imported/available from handstand.py
        weight=2.0, # Strong weight to force movement
        params={
            "command_name": "base_velocity", 
            "wheel_radius": 0.01,    # CHECK YOUR ROBOT'S WHEEL RADIUS
            "track_width": 0.4,     # CHECK YOUR ROBOT'S REAR TRACK WIDTH
            "std": 5.0,             # Wheel velocities are high (rad/s), use larger std
            "gait_mode": 1,
            # IMPORTANT: Select ONLY the rear driving wheels here
            "asset_cfg": SceneEntityCfg("robot", joint_names=["LH_WHEEL", "RH_WHEEL"]),
        },
    )
    feet_contact_biped = RewTerm(
        func=mdp.feet_contact_gated,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(LH|RH)_FOOT_link"),
            "gait_mode": 1,
            "threshold": 50.0,
        },
    )
    undesired_contacts_biped = RewTerm(
        func=mdp.undesired_contacts_gated,
        weight=-8.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(LF|RF)_FOOT_link", "trunk"]), 
            "threshold": 1.0,
            "gait_mode": 1
        },
    )

@configclass
class CUHKLRLSiriusWObservationsCfg(ObservationsCfg):
    """Reward terms for the MDP."""
    @configclass
    class CUHKLRLSiriusWPolicyCfg(ObservationsCfg.PolicyCfg):
        obs_scan = None
        gait_command = ObsTerm(func=mdp.gait_mode_obs)
        joint_vel = None
        joint_vel_legs = ObsTerm(func=mdp.joint_vel_rel)
        joint_vel_wheels = ObsTerm(func=mdp.joint_vel_rel)
        terrain_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_base")},
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            scale=1.0,
        )
    @configclass
    class CUHKLRLSiriusWCriticCfg(ObservationsCfg.CriticCfg):
        obs_scan = None
        gait_command = ObsTerm(func=mdp.gait_mode_obs)
        joint_vel = None
        joint_vel_legs = ObsTerm(func=mdp.joint_vel_rel)
        joint_vel_wheels = ObsTerm(func=mdp.joint_vel_rel)
        terrain_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_base")},
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

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
    # interval_gait_flip = EventTerm(
    #     func=mdp.set_gait_mode_flip,
    #     mode="interval",
    #     interval_range_s=(8.0, 10.0), 
    # )

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
    abad_joint_names = [
        "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
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
        CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.init_state.pos=(0.0, 0.0, 0.55)
        # ------------------------------Sence------------------------------
        # switch robot to unitree b2w
        self.scene.robot = CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ray_caster = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner = None
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.terrain.terrain_generator=ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator=FLOATING_RING2_TERRAINS_CFG
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
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
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.rewards.leg_deviation_l1_quad.weight = -0.1
        self.rewards.leg_deviation_l1_quad.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.leg_joint_vel_quad.weight = -0.002
        self.rewards.leg_joint_vel_quad.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.undesired_contacts_quad.weight = -0.2
        self.rewards.undesired_contacts_quad.params["sensor_cfg"].body_names = [f"^(?!.*({self.foot_link_name})).*"]
        self.rewards.leg_joint_power.weight = -2e-5
        self.rewards.leg_joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_joint_power.weight = -2e-5
        self.rewards.wheel_joint_power.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.abad_deviation_l1_biped.params["asset_cfg"].joint_names = self.abad_joint_names
        self.rewards.leg_pos_penalty_adaptive_quad.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still_without_cmd.weight = -2.0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        # 1. Legs (Standard scale)
        self.observations.policy.joint_vel_legs.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel_legs.scale = 0.25 
        self.observations.critic.joint_vel_legs.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names, preserve_order=True
        )
        self.observations.critic.joint_vel_legs.scale = 0.25

        # 2. Wheels (Reduced scale for high speed)
        self.observations.policy.joint_vel_wheels.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel_wheels.scale = 0.05 
        self.observations.critic.joint_vel_wheels.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names, preserve_order=True
        )
        self.observations.critic.joint_vel_wheels.scale = 0.05

        if self.__class__.__name__ == "CUHKLRLSiriusWMoEEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.curriculum.command_levels = None