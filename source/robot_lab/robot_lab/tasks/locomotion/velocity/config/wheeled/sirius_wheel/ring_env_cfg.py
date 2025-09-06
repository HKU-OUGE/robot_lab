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
    #     debug_vis=True,                      # 可视化目标箭头
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

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    wheels_stop_without_cmd = RewTerm(
        func=mdp.wheels_stop_without_cmd,
        weight=0.0,   # 惩罚系数，可调
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
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
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
@configclass
class CUHKLRLSiriusWObservationsCfg(ObservationsCfg):
    """Reward terms for the MDP."""
    @configclass
    class CUHKLRLSiriusWPolicyCfg(ObservationsCfg.PolicyCfg):
        # # ... 你已有的观测项
        # obs_scan = ObsTerm(
        #             func=mdp.obstacle_scan_disc,                      # 调用离散化后的扫描函数
        #             params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #             # 根据需要保留噪声，或设为 0
        #             noise=Unoise(n_min=0.0, n_max=0.0),
        #             clip=(0.0, 1.0),
        #             scale=1.0,
        #         )
        obs_scan = None

    policy: CUHKLRLSiriusWPolicyCfg = CUHKLRLSiriusWPolicyCfg()



@configclass
class CUHKLRLSiriusWRingEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: CUHKLRLSiriusWActionsCfg = CUHKLRLSiriusWActionsCfg()
    rewards: CUHKLRLSiriusWRewardsCfg = CUHKLRLSiriusWRewardsCfg()
    commands: CUHKLRLSiriusWCommandsCfg = CUHKLRLSiriusWCommandsCfg()
    observations: CUHKLRLSiriusWObservationsCfg = CUHKLRLSiriusWObservationsCfg()
    terminations: CUHKLRLSiriusWTerminationsCfg = CUHKLRLSiriusWTerminationsCfg()
    base_link_name = "trunk"
    foot_link_name = ".*_FOOT"
    calf_link_name = ".*_calf"
    wheel_joint_name = ".*_WHEEL"
    # fmt: off
    leg_joint_names = [
        "LF_HAA", "LF_HFE", "LF_KFE",
        "LH_HAA", "LH_HFE", "LH_KFE",
        "RF_HAA", "RF_HFE", "RF_KFE",
        "RH_HAA", "RH_HFE", "RH_KFE",
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
        CUHKLRL_SIRIUS_WHEEL_CFG.init_state.pos=(0.0, 0.0, 0.65)
        # CUHKLRL_SIRIUS_WHEEL_CFG.init_state.joint_pos={
        #     "LF_HAA": 0.00,
        #     "LH_HAA": 0.00,
        #     "RF_HAA": -0.00,
        #     "RH_HAA": -0.00,
        #     "LF_HFE": 0.2,
        #     "LH_HFE": -0.2,
        #     "RF_HFE": 0.2,
        #     "RH_HFE": -0.2,
        #     "LF_KFE": -1.2,
        #     "LH_KFE": 1.2,
        #     "RF_KFE": -1.2,
        #     "RH_KFE": 1.2,
        #     "LF_WHEEL": 0.00,
        #     "LH_WHEEL": 0.00,
        #     "RF_WHEEL": 0.00,
        #     "RH_WHEEL": 0.00,
        # }
        # ------------------------------Sence------------------------------
        # switch robot to unitree b2w
        self.scene.robot = CUHKLRL_SIRIUS_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.9, 0.0, 20.0)),
            ray_alignment='yaw',
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator=FLOATING_RING_TERRAINS_CFG
        # self.scene.main_camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name + "/main_camera",
        #     update_period=1.0 / 30.0,          # 30 Hz
        #     height=120,
        #     width=120,
        #     data_types=["depth"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         horizontal_aperture=20.955,    # mm
        #         focal_length=11.0,             # mm →  FOV ≈ 2 * atan(0.5*A / f) ≈ 87°
        #         clipping_range=(0.1, 10.0),    # m
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.45, 0.0, 0.0),
        #         rot=(0.5, -0.5, 0.5, -0.5),
        #         convention="ros"
        #     ),
        # )
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
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # self.observations.critic.height_scan = None
        # # no terrain curriculum
        # self.curriculum.terrain_levels = None

        # self.scene.height_scanner = None
        # self.scene.ray_caster = RayCasterCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/trunk",
        #     offset=RayCasterCfg.OffsetCfg(pos=(0, 0, -0.2)),
        #     mesh_prim_paths=["/World/ground"],
        #     ray_alignment="yaw",
        #     pattern_cfg=patterns.LidarPatternCfg(
        #         channels=4, vertical_fov_range=[-20, 20], horizontal_fov_range=[-180, 180], horizontal_res=10.0
        #     ),
        #     # debug_vis=not args_cli.headless,
        #     debug_vis=True,
        # )
        # ------------------------------Observations------------------------------
        # self.observations.policy.height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("ray_caster")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )
        # self.observations.policy.height_scan = None
        # self.observations.critic.height_scan = ObsTerm(
        #     func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("ray_caster")}, scale=1.0, clip=(-1.0, 1.0)
        # )
        # self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        # self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[self.wheel_joint_name]
        # )
        # self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        # self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[self.wheel_joint_name]
        # )
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
            "robot", joint_names=self.wheel_joint_names, preserve_order=True
        )
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan_disc,                      # 调用离散化后的扫描函数
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            # 根据需要保留噪声，或设为 0
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(0.0, 1.0),
            scale=1.0,
        )
        # self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names[:-4]

        # ------------------------------Actions------------------------------
        # reduce action scale
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
                "yaw": (3.14, 3.14),
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
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.6, 1.2)
        # self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        # self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = -2.5e-6
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = -2.5e-8
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -2.5
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still_without_cmd.weight = -2.0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.feet_continue_contact.weight = 0.1
        self.rewards.feet_continue_contact.params["expect_contact_num"] = 4
        self.rewards.feet_continue_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # Velocity-tracking rewards
        # Action penalties
        self.rewards.action_rate_l2.weight = -0.005

        # Contact sensor
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*({self.foot_link_name}|{self.calf_link_name})).*"
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            f"^(?!.*({self.foot_link_name})).*"
        ]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_height.weight = 0
        # self.rewards.feet_height.params["target_height"] = 0.1
        # self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_height_body.weight = 0
        # self.rewards.feet_height_body.params["target_height"] = -0.2
        # self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("LF_FOOT", "RF_FOOT"), ("LH_FOOT", "RH_FOOT"))
        self.rewards.upward.weight = 1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["RF_(HAA|HFE|KFE).*", "LH_(HAA|HFE|KFE).*"],
            ["LF_(HAA|HFE|KFE).*", "RH_(HAA|HFE|KFE).*"],
        ]
        # self.rewards.upward.weight = 1.0
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CUHKLRLSiriusWRingEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.illegal_contact = None
        # ------------------------------Commands------------------------------
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = (3.14, 3.14)
        self.curriculum.command_levels.params["range_multiplier"] = (1.0, 1.0)