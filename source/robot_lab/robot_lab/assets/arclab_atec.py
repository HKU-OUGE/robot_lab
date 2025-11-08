# Copyright (c) 2025-2026 Tang Tianyang
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Arclab robots.

The following configurations are available:

* :obj:`ARCLAB_ATEC_CFG`: Sirius robot with DC motor model for the legs


Reference: https://github.com/ruihuang1124/quadruped_control_ros2
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


ARCLAB_ATEC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius_Piper/no78/sirius_piper.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.05,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.405),
        joint_pos={
            "FL_hip_joint": 0.00,
            "FR_hip_joint": 0.00,
            "RL_hip_joint": 0.00,
            "RR_hip_joint": 0.00,
            "FL_thigh_joint": 0.83,
            "FR_thigh_joint": 0.83,
            "RL_thigh_joint": -0.83,
            "RR_thigh_joint": -0.83,
            "FL_calf_joint": -1.65,
            "FR_calf_joint": -1.65,
            "RL_calf_joint": 1.65,
            "RR_calf_joint": 1.65,
            # arm
            # "waist": 0.0,
            # "shoulder": 2.0,
            # "elbow": -1.5,
            # "forearm_roll": 0.0,
            # "wrist_angle": -0.5,
            # "wrist_rotate": 0.0,
            "joint1": 0.0,
            "joint2": 2.2,
            "joint3": -1.4,
            "joint4": 0.0,
            "joint5": 0.8,
            "joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip": DCMotorCfg(
            joint_names_expr=[".*hip_joint"],
            effort_limit=35.0,
            saturation_effort=50.0,
            velocity_limit=45.0,
            stiffness=50.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*thigh_joint"],
            effort_limit=35.0,
            saturation_effort=50.0,
            velocity_limit=45.0,
            stiffness=50.0,
            damping=2.0,
            friction=0.0,
        ),
        # "piper_arm_1": DCMotorCfg(
        #     joint_names_expr=["joint1",],
        #     effort_limit=40.5,
        #     saturation_effort=23.5,
        #     velocity_limit=30.0,
        #     stiffness=10.0,
        #     damping=0.5,
        #     friction=0.0,
        # ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*calf_joint"],  
            effort_limit=80.0,
            saturation_effort=100.0,
            velocity_limit=45.0,
            stiffness=50.0,
            damping=2.0,
            friction=0.0,
        ),
        "piper_arm_26": DCMotorCfg(
            # joint_names_expr=["joint1","joint3",
            #                   "joint4","joint5","joint6",
            #                   ],
            # joint_names_expr=["joint2","joint3",
            #                   "joint4","joint5","joint6", "joint1",
            #                   ],
            joint_names_expr=["joint1","joint2","joint3",
                              "joint4","joint5","joint6",
                              ],
            effort_limit=40.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=10.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Sirius Piper using DC motor.
"""
