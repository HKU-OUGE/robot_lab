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
from robot_lab.terrains.config.rough import *
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip

@configclass
class CUHKLRLSiriusWActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    # )

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
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_FOOT_link"),
            "command_name": "base_velocity",
            "threshold": 0.75,
        },
    )
@configclass
class CUHKLRLSiriusWObservationsCfg(ObservationsCfg):
    """Reward terms for the MDP."""
    @configclass
    class CUHKLRLSiriusWPolicyCfg(ObservationsCfg.PolicyCfg):
        # # ... 你已有的观测项
        # front_scan = ObsTerm(
        #             func=mdp.height_scan,                      # 调用离散化后的扫描函数
        #             params={"sensor_cfg": SceneEntityCfg("front_height")},
        #             # 根据需要保留噪声，或设为 0
        #             noise=Unoise(n_min=0.0, n_max=0.0),
        #             clip=(-2.0, 2.0),
        #             scale=1.0,
        #         )
        # back_scan = ObsTerm(
        #             func=mdp.height_scan,                      # 调用离散化后的扫描函数
        #             params={"sensor_cfg": SceneEntityCfg("back_height")},
        #             # 根据需要保留噪声，或设为 0
        #             noise=Unoise(n_min=0.0, n_max=0.0),
        #             clip=(-2.0, 2.0),
        #             scale=1.0,
        #         )
        base_lin_vel = None
        obs_scan = None
    @configclass
    class CUHKLRLSiriusWStudentPolicyCfg(ObservationsCfg.PolicyCfg):
        # # ... 你已有的观测项
        height_scan = None
        base_lin_vel = None
    @configclass
    class CUHKLRLSiriusWCriticCfg(ObservationsCfg.CriticCfg):
        # img_feat = ObsTerm(
        #             func=mdp.observations.image_features,                      # 调用离散化后的扫描函数
        #             params={
        #                 "sensor_cfg": SceneEntityCfg("main_camera"),  # 关键：相机名
        #                 "data_type": "rgb",                           # 也可 "distance_to_camera"
        #                 "model_name": "resnet18",                     # 默认即 resnet18
        #                 # "model_device": "cuda:0",                     # 可把特征提取放到独立设备
        #             },
        #             clip=None,
        #             scale=1.0,
        #         )
        # front_scan = ObsTerm(
        #             func=mdp.height_scan,                      # 调用离散化后的扫描函数
        #             params={"sensor_cfg": SceneEntityCfg("front_height")},
        #             # 根据需要保留噪声，或设为 0
        #             noise=Unoise(n_min=0.0, n_max=0.0),
        #             clip=(-2.0, 2.0),
        #             scale=1.0,
        #         )
        # back_scan = ObsTerm(
        #             func=mdp.height_scan,                      # 调用离散化后的扫描函数
        #             params={"sensor_cfg": SceneEntityCfg("back_height")},
        #             # 根据需要保留噪声，或设为 0
        #             noise=Unoise(n_min=0.0, n_max=0.0),
        #             clip=(-2.0, 2.0),
        #             scale=1.0,
        #         )
        obs_scan = None

    policy: CUHKLRLSiriusWPolicyCfg = CUHKLRLSiriusWPolicyCfg()
    critic: CUHKLRLSiriusWCriticCfg = CUHKLRLSiriusWCriticCfg()
    # student_policy: CUHKLRLSiriusWStudentPolicyCfg = CUHKLRLSiriusWStudentPolicyCfg()



@configclass
class CUHKLRLSiriusWWheelEXPEnvCfg(LocomotionVelocityRoughEnvCfg):
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
        CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.init_state.pos=(0.0, 0.0, 0.55)
        # wheel.velocity_limit_sim = 0.1
        # CUHKLRL_SIRIUS_WHEEL_CFG.init_state.joint_pos={
        #     "LF_HAA": 0.00,
        #     "LH_HAA": 0.00,
        #     "RF_HAA": -0.00,
        #     "RH_HAA": -0.00,
        #     "LF_HFE": 0.2,
        #     "LH_HFE": -0.2,
        #     "RF_HFE": 0.2,
        #     "RH_HFE": -0.2,
        #     "LF_KNEE": -1.2,
        #     "LH_KNEE": 1.2,
        #     "RF_KNEE": -1.2,
        #     "RH_KNEE": 1.2,
        #     "LF_WHEEL": 0.00,
        #     "LH_WHEEL": 0.00,
        #     "RF_WHEEL": 0.00,
        #     "RH_WHEEL": 0.00,
        # }
        # ------------------------------Sence------------------------------
        # switch robot to unitree b2w
        self.scene.robot = CUHKLRL_SIRIUS_WHEEL_DELAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ray_caster = None
        # self.scene.front_height = RayCasterCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/base",
        #     offset=RayCasterCfg.OffsetCfg(pos=(0.75, 0.0, 20.0)),
        #     ray_alignment='yaw',
        #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.5, 0.5]),
        #     debug_vis=True,
        #     mesh_prim_paths=["/World/ground"],
        # )
        # self.scene.back_height = RayCasterCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/base",
        #     offset=RayCasterCfg.OffsetCfg(pos=(-0.65, 0.0, 20.0)),
        #     ray_alignment='yaw',
        #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.5, 0.5]),
        #     debug_vis=True,
        #     mesh_prim_paths=["/World/ground"],
        # )
        # self.scene.front_height.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.back_height.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.front_height.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.back_height.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator=EASY_ROUGH_TERRAINS_CFG
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
        # self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.func = mdp.joint_vel_rel
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel.scale = 0.5
        self.observations.critic.joint_vel.scale = 0.5

        # self.observations.student_policy.joint_pos.func = mdp.joint_pos_rel
        # self.observations.student_policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=self.leg_joint_names, preserve_order=True
        # )
        # self.observations.student_policy.base_ang_vel.scale = 0.25
        # self.observations.student_policy.joint_pos.scale = 1.0
        # self.observations.student_policy.joint_vel.func = mdp.joint_vel_rel
        # self.observations.student_policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=self.leg_joint_names, preserve_order=True
        # )
        # self.observations.student_policy.joint_vel.scale = 0.5
        # self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names[:-4]

        # ------------------------------Actions------------------------------
        # reduce action scale
        # self.actions.joint_pos.scale = 0.25
        # self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # self.actions.joint_pos.joint_names = self.joint_names[:-4]
        self.actions.joint_vel.scale = 1.5
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.joint_names = self.joint_names[-4:]

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (3.14, 3.14),
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
        # self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.55, 0.85)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.45, 0.75)
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None
        # self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        # self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        # Joint penalties
        self.rewards.joint_deviation_hip_roll.weight = -0.1
        self.rewards.joint_deviation_hip_roll.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_deviation_knee.weight = -0.05
        self.rewards.joint_deviation_knee.params["asset_cfg"].joint_names = self.leg_joint_names
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
            func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.25}
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.25}
        )
        # Others
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("LF_FOOT_link", "RF_FOOT_link"), ("LH_FOOT_link", "RH_FOOT_link"))
        self.rewards.upward.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["RF_(HAA|HFE|KNEE).*", "LH_(HAA|HFE|KNEE).*"],
            ["LF_(HAA|HFE|KNEE).*", "RH_(HAA|HFE|KNEE).*"],
        ]
        # self.rewards.upward.weight = 1.0
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CUHKLRLSiriusWWheelEXPEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_abad_link"]
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.illegal_contact = None
        # ------------------------------Commands------------------------------
        # ------------------------------Commands------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",                     # 用平面替代阶梯/噪声等生成器
            terrain_generator=SAND_TERRAINS_CFG,                   # plane 不需要生成器
            max_init_terrain_level=10,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # 沙地 => 低附着、无弹跳
                static_friction=1.0,                 # 静摩擦略高于动摩擦
                dynamic_friction=1.0,                # 低动摩擦，容易打滑
                restitution=0.0,                      # 无弹性
                friction_combine_mode="min",          # 与轮胎等相互作用时取更低一方的摩擦
                restitution_combine_mode="min",
            ),
            # 仅视觉：可保持原材质，或换成沙子质感的 MDL / PreviewSurface
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.65, 0.50),     # 沙色；只是渲染，与物理无关
                roughness=0.9,
                metallic=0.0,
            ),
            debug_vis=False,
        )
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (3.14, 3.14)
        self.curriculum.command_levels.params["range_multiplier"] = (1.0, 1.0)