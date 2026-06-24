# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, RewardsCfg,
)

##
# Pre-defined configs
##
# use cloud assets
# from isaaclab_assets.robots.unitree import ARCLAB_ARCDOG_CFG  # isort: skip
# use local assets
from robot_lab.assets.arclab import ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip


@configclass
class ArcdogAdjustableLegRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    rotate_joint_pos_penalty = RewTerm(
        func=mdp.joint_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh_joint|calf_joint)$"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.3,
        },
    )

    prismatic_joint_pos_penalty  = RewTerm(
        func=mdp.lateral_step_scaled_joint_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.5,
            "foot_body_names": {
                "FL": "FL_foot",
                "FR": "FR_foot",
                "RL": "RL_foot",
                "RR": "RR_foot",
            },
            "height_threshold": 0.03,
            "gate_width": 0.12,
            "relief_scale": 0.25,
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 5.0,
            "min_contacts_per_side": 1,
        },
    )

    stand_still_flat = RewTerm(
        func=mdp.stand_still_flat_orientation_bonus,
        weight=0.0,  # 默认权重设为0，在 EnvCfg 中具体配置
        params={
            "command_name": "base_velocity",
            "std": 0.1,               # 控制对倾斜的敏感度，越小越严格
            "command_threshold": 0.1,  # 速度指令小于此值视为静止
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    # 惩罚伸缩腿的剧烈加速度 (震荡的主要特征)
    box_joint_acc_penalty = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0, # 在 EnvCfg 中激活
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )

    # 针对伸缩腿的关节速度惩罚
    box_joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,  # 使用关节速度，它支持 asset_cfg
        weight=-0.01,           # 权重建议：从 -0.01 到 -0.05 开始尝试，太大会导致腿动不了
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )

    box_joint_action_rate = RewTerm(
        func=mdp.lateral_step_scaled_action_rate_l2_by_name,
        weight=-0.0,
        params={
            # 使用正则表达式匹配你的伸缩关节，比如包含 "box_joint" 的所有关节
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*box_joint.*"),
            "foot_body_names": {
                "FL": "FL_foot",
                "FR": "FR_foot",
                "RL": "RL_foot",
                "RR": "RR_foot",
            },
            "height_threshold": 0.03,
            "gate_width": 0.12,
            "relief_scale": 0.35,
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 5.0,
            "min_contacts_per_side": 1,
        }
    )

    # 针对伸缩腿的专属限位惩罚
    box_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,  # 复用同一个底层函数
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )


    # stand_still_joint_vel = RewTerm(
    #     func=mdp.stand_still_joint_vel_penalty, # 调用我们刚才写的自定义函数
    #     weight=0.0,  # ！！！权重建议从 -0.5 开始尝试。如果还晃，可以加大到 -1.0 甚至 -2.0
    #     params={
    #         "command_name": "base_velocity", # 确保这里的名字和你的指令管理器中一致
    #         "command_threshold": 0.1,        # 只有指令速度 < 0.1 时才惩罚
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    stand_still_revolute_joint_vel = RewTerm(
        func=mdp.stand_still_joint_vel_penalty, # 替换为你的实际路径
        weight=0.0,  # 针对 rad/s 的权重，数值通常较大，权重可以适中
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            # 使用正则表达式匹配所有的 hip, thigh, calf 关节 (例如 FL_hip_joint, FR_thigh_joint 等)
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*hip_joint.*", ".*thigh_joint.*", ".*calf_joint.*"]
            ),
        },
    )

    stand_still_prismatic_joint_vel = RewTerm(
        func=mdp.lateral_step_scaled_stand_still_joint_vel_penalty,
        weight=0.0,  # ！！！注意：直线速度(m/s)的数值通常比角速度(rad/s)小得多，因此可能需要更大的负权重
        params={
            "command_name": "base_velocity",
            "foot_body_names": {
                "FL": "FL_foot",
                "FR": "FR_foot",
                "RL": "RL_foot",
                "RR": "RR_foot",
            },
            "height_threshold": 0.03,
            "gate_width": 0.12,
            "relief_scale": 0.35,
            "command_threshold": 0.1,
            # 匹配 box_joint
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*box_joint.*"]
            ),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 5.0,
            "min_contacts_per_side": 1,
        },
    )

    stand_still_base_ang_vel = RewTerm(
        func=mdp.stand_still_base_ang_vel_penalty,
        weight=0.0,  # 权重可以从 -0.5 到 -2.0 尝试
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    stand_still_base_lin_vel = RewTerm(
        func=mdp.stand_still_base_lin_vel_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    # 新增：倾斜自适应的高度惩罚
    base_height_relaxed = RewTerm(
        func=mdp.base_height_l2_relaxed_on_tilt, # 指向刚才写的新函数
        weight=0.0, # 默认 0，在 EnvCfg 中激活
        params={
            "target_height": 0.44,
            "tilt_sensitivity": 2.0, # 建议设为 1.0 到 3.0 之间
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    # =====================================================================
    # 新增：足端间距惩罚 (解决内八字/走钢丝步态)
    # =====================================================================
    feet_stance_width = RewTerm(
        func=mdp.feet_stance_width_penalty,
        weight=0.0, # 在 EnvCfg 中激活
        params={
            "min_width": 0.31, # 期望的最小足端横向距离（米）。请根据你机器人的实际肩宽进行调整！
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # # =====================================================================
    # # 进阶自适应足端间距惩罚 (全参数化)
    # # =====================================================================
    # feet_stance_width_advanced_adaptive = RewTerm(
    #     func=mdp.feet_stance_width_advanced_adaptive_penalty,
    #     weight=0.0,
    #     params={
    #         # 距离阈值参数
    #         "flat_min_width": 0.31,               # 【平地】期望的最小足端横向距离
    #         "rough_stationary_min_width": 0.25,   # 【上地形+静止】保证站立不倒的最小距离
    #         "rough_moving_min_width": 0.12,       # 【上地形+运动】极度放宽，允许走钢丝/避障步态

    #         # 速度判定参数
    #         "stationary_speed_threshold": 0.10,   # 速度低于 0.10m/s 视为完全静止
    #         "moving_speed_threshold": 0.30,       # 速度高于 0.30m/s 视为完全运动

    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #     },
    # )

    feet_stance_width = RewTerm(
        func=mdp.hip_joint_abduction_min_l2,
        weight=0.0,
        params={
            "min_positions": {
                "FL_hip_joint": 0.18,
                "FR_hip_joint": -0.18,
                "RL_hip_joint": 0.18,
                "RR_hip_joint": -0.18,
            },
            "deadband": 0.01,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                ],
            ),
        },
    )

    lateral_step_flat = RewTerm(
        func=mdp.lateral_step_flat_orientation_l2,
        weight=0.0,
        params={
            "foot_body_names": {
                "FL": "FL_foot",
                "FR": "FR_foot",
                "RL": "RL_foot",
                "RR": "RR_foot",
            },
            "height_threshold": 0.03,
            "gate_width": 0.12,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    lateral_step_hip_abduction = RewTerm(
        func=mdp.lateral_step_hip_abduction_min_l2,
        weight=0.0,
        params={
            "foot_body_names": {
                "FL": "FL_foot",
                "FR": "FR_foot",
                "RL": "RL_foot",
                "RR": "RR_foot",
            },
            "hip_joint_names": {
                "FL": "FL_hip_joint",
                "FR": "FR_hip_joint",
                "RL": "RL_hip_joint",
                "RR": "RR_hip_joint",
            },
            "lower_side_min": 0.30,
            "upper_side_min": 0.18,
            "deadband": 0.01,
            "height_threshold": 0.03,
            "gate_width": 0.12,
            "lower_side_weight": 3.0,
            "upper_side_weight": 0.5,
            "invert_side_height_delta": True,
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 5.0,
            "min_contacts_per_side": 1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    lateral_step_box_length_difference = RewTerm(
        func=mdp.lateral_step_box_length_difference_l2,
        weight=0.0,
        params={
            "foot_body_names": {
                "FL": "FL_foot",
                "FR": "FR_foot",
                "RL": "RL_foot",
                "RR": "RR_foot",
            },
            "box_joint_names": {
                "FL": "FL_box_joint",
                "FR": "FR_box_joint",
                "RL": "RL_box_joint",
                "RR": "RR_box_joint",
            },
            "min_lower_upper_delta": 0.030,
            "deadband": 0.003,
            "height_threshold": 0.03,
            "gate_width": 0.12,
            "pairwise": True,
            "command_name": "base_velocity",
            "command_threshold": 0.12,
            "body_velocity_threshold": 0.18,
            "low_speed_gate_width": 0.20,
            "invert_side_height_delta": True,
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "contact_threshold": 5.0,
            "min_contacts_per_side": 1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class ArclabArcdogAdjustableLegBodyflatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ArcdogAdjustableLegRewardsCfg = ArcdogAdjustableLegRewardsCfg()


    base_link_name = "base"
    trunk_link_name = "trunk"
    hip_link_name = ".*_thigh"
    knee_link_name = ".*_calf"
    abad_link_name = ".*_hip"
    foot_link_name = ".*_foot"
    extension_link_name = ".*_box"

    # fmt: off
    joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint",
        "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint",
        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree a1
        self.scene.robot = ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator=ROUGH_TERRAINS_CFG

         # [新增] 解决打滑问题：修改地形的物理材质属性
        self.scene.terrain.physics_material.friction_combine_mode = "average"    # 避免 multiply 导致极小值
        self.scene.terrain.physics_material.restitution_combine_mode = "average"
        self.scene.terrain.physics_material.static_friction = 1.2               # 提高基础静摩擦力
        self.scene.terrain.physics_material.dynamic_friction = 1.2               # 提高基础动摩擦力

        # ------------------------------Observations------------------------------
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 因为在你的 Base Cfg 中，policy 组里已经没有这两个属性了，
        # 所以不需要（也不能）再在这里把它们设为 None 来禁用。
        # self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None

        # ==========================================
        # 1. 更新 Policy 组的关节名称 (直接覆写 params 字典，彻底避免 KeyError)
        # ==========================================
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
        }

        # ==========================================
        # 2. 更新 Estimator 组的关节名称 (VAE 专用的历史组，加入 hasattr 保护)
        # ==========================================
        if hasattr(self.observations, "estimator") and self.observations.estimator is not None:
            self.observations.estimator.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }
            self.observations.estimator.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }
            # ==========================================
            # 【核心修改】严格对齐 Estimator 和 Policy 的 Scale！
            # 解决 Sim-to-Sim 中 VAE 接收到缩小 20 倍的速度导致动作发疯的 BUG
            # ==========================================
            self.observations.estimator.base_ang_vel.scale = 0.25
            self.observations.estimator.joint_vel.scale = 0.05

        # ==========================================
        # 3. 更新 Critic 组的关节名称 (加入 hasattr 保护)
        # ==========================================
        if hasattr(self.observations, "critic") and self.observations.critic is not None:
            self.observations.critic.joint_pos.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }
            self.observations.critic.joint_vel.params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)
            }

        # self.observations.policy.base_lin_vel.scale = 2.0
        # self.observations.policy.base_ang_vel.scale = 0.25
        # self.observations.policy.joint_pos.scale = 1.0
        # self.observations.policy.joint_vel.scale = 0.05
        # self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
        #     self.joint_names
        # )
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
        #     self.joint_names
        # )

        # ------------------------------Actions------------------------------
        # # 强制将 Action 的零位对齐到 0.075，这样网络输出 0 时，腿保持在 0.075
        # self.scene.robot.default_joint_angles = {
        #     "FL_hip_joint": 0.1, "FR_hip_joint": -0.1,
        #     "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
        #     "FL_thigh_joint": 0.6, "FR_thigh_joint": 0.6,
        #     "RL_thigh_joint": 0.6, "RR_thigh_joint": 0.6,
        #     "FL_calf_joint": -0.95, "FR_calf_joint": -0.95,
        #     "RL_calf_joint": -0.95, "RR_calf_joint": -0.95,
        #     # 关键：这里必须与 init_state 一致
        #     "FL_box_joint": 0.1, "FR_box_joint": 0.1,
        #     "RL_box_joint": 0.1, "RR_box_joint": 0.1,
        # }

        # reduce action scale
        # self.actions.joint_pos.scale = 0.1
        self.actions.joint_pos = mdp.LateralStepBoxBiasJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.joint_names,
            scale={
                ".*_box_joint": 0.02,
                ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
            },
            use_default_offset=True,
            clip={".*": (-60.0, 60.0)},
            preserve_order=True,
            # 侧向台阶姿态先验：低侧伸长、高侧缩短。策略输出仍作为残差叠加。
            height_threshold=0.02,
            gate_width=0.06,
            lower_side_bias=0.018,
            upper_side_bias=-0.014,
            min_box_target=0.000,
            max_box_target=0.060,
            invert_side_height_delta=True,
        )

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_apply_external_force_torque.params[
            "asset_cfg"
        ].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = [".*"]
        self.events.randomize_joint_friction.params["asset_cfg"].joint_names = [".*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.8, 1.5)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.8, 1.5)

        self.events.randomize_screw_joints.params["asset_cfg"].joint_names = [".*_box_joint"]
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.35, 0.35),
            "yaw": (-0.18, 0.18),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
            "roll": (-0.2, 0.2),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.2, 0.2),
        }

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -20

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.3
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.base_height_l2.weight = -1.5
        self.rewards.base_height_l2.params["target_height"] = 0.44
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        # 设置静止水平奖励的权重
        # 这是一个正向奖励(Bonus)，所以权重为正。
        # 建议值: 0.5 ~ 2.0。如果机器人在坡上静止时还是歪的，可以调大这个值。
        self.rewards.stand_still_flat.weight = 1.5
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        # Joint penaltie
        # self.rewards.joint_torques_l2.weight = -2.5e-6
        # 测试 暂时取消此惩罚
        self.rewards.joint_vel_l2.weight = -0.005
        self.rewards.box_joint_vel_penalty.weight = -0.01
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.box_joint_acc_penalty.weight = -1.0e-5 # 伸缩关节的加速度惩罚，建议比全局高 1-2 个数量级
        self.rewards.joint_pos_limits.weight = -0.05
        self.rewards.joint_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=".*_(hip|thigh|calf)_joint"
        )
        self.rewards.box_joint_pos_limits.weight = -40.0
        # 禁止超速
        self.rewards.joint_vel_limits.weight = -0.3

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.40
        # UNUESD self.rewards.action_l2.weight = 0.0
        self.rewards.box_joint_action_rate.weight = -0.55
        self.rewards.box_joint_action_rate.params["relief_scale"] = 0.25

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", "trunk", ".*_hip", ".*_thigh", ".*calf"
        ]

        # self.rewards.contact_forces.weight = -0.005
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [
        #     self.foot_link_name
        # ]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 6
        self.rewards.track_ang_vel_z_exp.weight = 4.0

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = -0.6
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_contact.params["expect_contact_num"] = 2 # 确保期望值为 2
        self.rewards.feet_stumble.weight = -0.01
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.joint_power.weight = -2e-5
        # 测试 暂时取消
        self.rewards.joint_power.weight = -2e-6
        self.rewards.stand_still_without_cmd.weight = -3.5
        self.rewards.stand_still_without_cmd.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=".*_(thigh|calf)_joint"
        )
        # self.rewards.stand_still_joint_vel.weight = -0.3
        self.rewards.stand_still_revolute_joint_vel.weight = -0.3
        self.rewards.stand_still_prismatic_joint_vel.weight = -3.0
        self.rewards.stand_still_base_ang_vel.weight = -3
        self.rewards.stand_still_base_lin_vel.weight = -15
        self.rewards.stand_still_flat.params["command_threshold"] = 0.06
        self.rewards.stand_still_revolute_joint_vel.params["command_threshold"] = 0.06
        self.rewards.stand_still_prismatic_joint_vel.params["command_threshold"] = 0.06
        self.rewards.stand_still_base_ang_vel.params["command_threshold"] = 0.06
        self.rewards.stand_still_base_lin_vel.params["command_threshold"] = 0.08
        # self.rewards.joint_position_penalty.weight = -0.9
        # self.rewards.joint_position_penalty.params["stand_still_scale"] = 1.5
        # self.rewards.joint_position_penalty.params["velocity_threshold"] = 0.3
        self.rewards.rotate_joint_pos_penalty.weight = -0.03
        self.rewards.prismatic_joint_pos_penalty.weight = -40
        self.rewards.feet_height_exp.weight = 1.2
        self.rewards.feet_height_exp.params["target_height"] = 0.095
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        # self.rewards.feet_height_body_exp.weight = -4.9
        # self.rewards.feet_height_body_exp.params["target_height"] = -0.32
        # self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        self.rewards.feet_gait.weight = 2.0
        self.rewards.feet_gait.params["velocity_threshold"] = 0.1
        # trotting
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )
        # pronking
        # self.rewards.feet_gait.params["synced_feet_pair_names"] = (
        #     ("FL_foot", "FR_foot"),
        #     ("RR_foot", "RL_foot"),
        # )

        # # =====================================================================
        # # 激活足端间距惩罚
        # # =====================================================================
        # # 权重设为负数。-5.0 属于中等偏上的惩罚力度，足以引起策略的重视。
        # # 如果发现机器人腿张得太开，可以把权重改小（如 -2.0）或者减小 min_width。
        # self.rewards.feet_stance_width.weight = -5.5
        # self.rewards.feet_stance_width_advanced_adaptive.weight = -5.5
        self.rewards.feet_stance_width.weight = -8.0
        self.rewards.lateral_step_flat.weight = 0.0
        self.rewards.lateral_step_hip_abduction.weight = -8.0
        self.rewards.lateral_step_box_length_difference.weight = -600.0
        self.rewards.lateral_step_box_length_difference.params["command_threshold"] = 0.08
        self.rewards.lateral_step_box_length_difference.params["body_velocity_threshold"] = 0.12
        self.rewards.lateral_step_box_length_difference.params["low_speed_gate_width"] = 0.12

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ArclabArcdogAdjustableLegBodyflatEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     self.base_link_name,
        #     self.trunk_link_name,
        #     # self.abad_link_name,
        #     # self.knee_link_name,
        #     # self.hip_link_name,
        # ]
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, # 具体的函数名取决于你使用的 Isaac Lab 版本
            params={
                "limit_angle": 1.2, # 允许的最大倾斜角，1.2 弧度大约是 68 度。
                # 爬高台时 pitch (俯仰角) 会很大，所以这个角度要放宽，不能设成 0.5 这种小角度
                "asset_cfg": SceneEntityCfg("robot")
            }
        )
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels.params["range_multiplier"] = (0.1,1.0)


        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)


@configclass
class ArclabArcdogAdjustableLegBodyflatStudentNoPriorEnvCfg(ArclabArcdogAdjustableLegBodyflatEnvCfg):
    """Bodyflat student env without the lateral-step action prior."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=self.joint_names,
            scale={
                ".*_box_joint": 0.02,
                ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
            },
            use_default_offset=True,
            clip={".*": (-60.0, 60.0)},
            preserve_order=True,
        )

        # Student no-prior distillation: keep rows ordered by terrain difficulty,
        # but replace distance-based promotion with stability-based promotion.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 4
        self.curriculum.terrain_levels.func = mdp.terrain_levels_student_no_prior
        self.curriculum.terrain_levels.params = {
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            "orientation_term_name": "flat_orientation_l2",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_level": 0,
            "initial_max_level": 4,
            "max_level": 8,
            "warmup_episodes": 2,
            "unlock_every_episodes": 5,
            "promote_after_successes": 2,
            "demote_after_failures": 1,
            "promote_reward_threshold": 0.55,
            "demote_reward_threshold": 0.25,
            "orientation_min_score": -1.2,
            "max_x_vel_error": 0.55,
            "max_y_vel_error": 0.35,
            "max_yaw_vel_error": 0.40,
        }
        self.curriculum.command_levels.func = mdp.command_levels_vel_student_no_prior
        self.curriculum.command_levels.params = {
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            "asset_cfg": SceneEntityCfg("robot"),
            "range_multiplier": (0.5, 0.75),
            "terrain_gate_levels": (5.0, 6.5, 7.5),
            "command_multipliers": (0.5, 0.6, 0.7, 0.75),
            "delta": 0.05,
            "reward_threshold": 0.70,
            "success_rate_threshold": 0.90,
            "max_x_vel_error": 0.45,
            "max_y_vel_error": 0.25,
            "max_yaw_vel_error": 0.35,
            "allow_decrease": True,
        }
        self.disable_zero_weight_rewards()
