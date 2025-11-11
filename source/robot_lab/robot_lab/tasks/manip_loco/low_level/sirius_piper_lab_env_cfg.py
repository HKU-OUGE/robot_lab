import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


import robot_lab.tasks.manip_loco.low_level.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##

SIRIUS_PIPER_ARM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7, noise_range=(-0.05, 0.05), noise_step=0.01, border_width=0.25
        ),
    },
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SIRIUS_PIPER_ARM_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 4.0),
            "dynamic_friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),   # 先收窄，之后再放宽
            "operation": "scale",
            "recompute_inertia": True,                # 关键：改质量后重算惯量
        },
    )

    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    add_ee_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_base"),
            "mass_distribution_params": (-0.1, 0.8),
            "operation": "add",
            "recompute_inertia": True,                # 关键：改质量后重算惯量
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    ## SIRIUS_PIPER
    
    ee_pose = mdp.command_cfg.MLUniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_base",
        resampling_time_range=(6.0,8.0),
        debug_vis=True,
        is_QuadrupedManipulator=True,
        curriculum_coeff = 1000,          
        ranges_final =mdp.command_cfg.MLUniformPoseCommandCfg.Ranges(
            pos_x=(0.58, 0.78),
            pos_y=(-0.35, 0.35),
            pos_z=(0.3, 0.65), # world frame not base frame
            roll=(-0.0, 0.0),
            pitch=(3.14 - 3.14 / 6, 3.14 + 3.14 / 6),  # depends on end-effector axis
            yaw=(-3.14 / 9, 3.14 / 9),
        ),
        ranges = mdp.command_cfg.MLUniformPoseCommandCfg.Ranges(
            pos_x=(0.58, 0.78),
            pos_y=(-0.35, 0.35),
            pos_z=(0.3, 0.65), # world frame not base frame
            roll=(-0.0, 0.0),
            pitch=(3.14 - 3.14 / 9, 3.14 + 3.14 / 9),  # depends on end-effector axis
            yaw=(-3.14 / 9, 3.14 / 9),
        ),
        ranges_init=mdp.command_cfg.MLUniformPoseCommandCfg.Ranges(
            pos_x=(0.60, 0.68), 
            pos_y=(-0.05, 0.05),
            pos_z=(0.45, 0.5), # world frame not base frame
            roll=(-0.0, 0.0),
            pitch=(3.14, 3.14),  # depends on end-effector axis
            yaw=(-0.0, 0.0),
        ),
    )

    base_velocity = mdp.command_cfg.MLUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        is_QuadrupedManipulator=True,
        curriculum_coeff= 1000,         
        ranges=mdp.command_cfg.MLUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5),heading=(-0.0, 0.0)
        ),
        ranges_final=mdp.command_cfg.MLUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.8), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5),heading=(-0.0, 0.0)
        ),
        ranges_init=mdp.command_cfg.MLUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.35), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1),heading=(-0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[
                                                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                                    ],
                                           scale = {"FR_hip_joint": 0.25, "FR_thigh_joint": 0.25, "FR_calf_joint": 0.25,
                                                    "FL_hip_joint": 0.25, "FL_thigh_joint": 0.25, "FL_calf_joint": 0.25,
                                                    "RR_hip_joint": 0.25, "RR_thigh_joint": 0.25, "RR_calf_joint": 0.25,
                                                    "RL_hip_joint": 0.25, "RL_thigh_joint": 0.25, "RL_calf_joint": 0.25,}, 
                                         use_default_offset=True,
                                         preserve_order=True,
    )   
    # arm_pose = mdp.JointPositionActionCfg(asset_name="robot",
    #                                       joint_names=[
    #                                           "joint1", "joint2", "joint3", 
    #                                           "joint4", "joint5", "joint6"],
    #                                        scale = {"joint1":        0.5, # 0.8
    #                                                 "joint2":     0.5, # 0.35
    #                                                 "joint3":        0.5, # 0.35
    #                                                 "joint4": 0.5, # 0.35
    #                                                 "joint5":  0.5, # 0.35
    #                                                 "joint6": 0.5}, # 0.35
    #                                         use_default_offset=True,
    #                                         preserve_order=True,
    # )


    arm_pose = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            body_name="gripper_base",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

    #     self.actions.arm_action = mdp.JointPositionActionCfg(
    #         asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    #     )

    # actions.arm_action = DifferentialInverseKinematicsActionCfg(
    #         asset_name="robot",
    #         joint_names=["panda_joint.*"],
    #         body_name="panda_hand",
    #         controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    #         scale=0.5,
    #         body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    #     )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        Manipulator_pose_command = ObsTerm(func=mdp.generated_commands,
                                   params={"command_name": "ee_pose"}) # dim = 7
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
        
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- ARM 
    # The name must have a prefix of "end_effector_".
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_base"),
                "command_name": "ee_pose",
                "std": 0.2},
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_base"), 
                "command_name": "ee_pose"},
    )

    end_effector_action_rate = RewTerm(func=mdp.action_rate_l2_arm, weight=-0.005)

    end_effector_action_smoothness = RewTerm(func=mdp.arm_action_smoothness_penalty, weight=-0.02)


    # -- LEG

    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight= 0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    F_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight= 0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    R_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight= 1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": 0.12,
            "command_name": "base_velocity",
        },
    )


    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            "command_name": "base_velocity",
        },
    )

    foot_contact = RewTerm(
        func=mdp.standing_feet_contact_force,
        weight= 0.003,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "force_threshold": 7.5,
            "command_threshold": 0.1,
        },
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


    action_smoothness = RewTerm(
        func=mdp.leg_action_smoothness_penalty,
        weight=-0.02,
    )

    height_reward = RewTerm(func=mdp.rewards.base_height_l2, weight=-2.0, params={"target_height": 0.40})

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    thigh_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 0.5},
    )

    calf_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    )

    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 0.5},
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    flat_ori_modify = CurrTerm(func=mdp.modify_reward_weight,
                               params={"term_name": "flat_orientation_l2",
                                       "num_steps": 2000,
                                       "weight": -0.00})

    flat_height_modify = CurrTerm(func=mdp.modify_reward_weight,
                               params={"term_name": "height_reward",
                                       "num_steps": 4000,
                                       "weight": -1.00})
    
##
# Environment configuration
##

@configclass
class ManipulationLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the manipulation locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)