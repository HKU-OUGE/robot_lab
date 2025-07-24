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
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius_Piper/sirius_piper.usd",
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
        pos=(0.0, 0.0, 0.445),
        joint_pos={
            "FL_hip_joint": 0.00,
            "FR_hip_joint": 0.00,
            "RL_hip_joint": 0.00,
            "RR_hip_joint": 0.00,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": -0.8,
            "RR_thigh_joint": -0.8,
            "FL_calf_joint": -1.6,
            "FR_calf_joint": -1.6,
            "RL_calf_joint": 1.6,
            "RR_calf_joint": 1.6,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            "joint8": 0.0,
            # "joint1": 0.0,
            # "joint2": 0.59,
            # "joint3": -0.23,
            # "joint4": -0.36,
            # "joint5": 0.0,
            # "joint6": 0.0,
            # "joint7": 0.0,
            # "joint8": 0.0,
            # "joint1": 0.0,
            # "joint2": 1.10,
            # "joint3": -1.28,
            # "joint4": 0.0,
            # "joint5": 0.40,
            # "joint6": 0.0,
            # "joint7": 0.0,
            # "joint8": 0.0,
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
            stiffness=100.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*thigh_joint"],
            effort_limit=35.0,
            saturation_effort=50.0,
            velocity_limit=45.0,
            stiffness=100.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*calf_joint"],  
            effort_limit=80.0,
            saturation_effort=100.0,
            velocity_limit=45.0,
            stiffness=100.0,
            damping=2.0,
            friction=0.0,
        ),
        "widow_arm": DCMotorCfg(
            joint_names_expr=["joint1","joint2","joint3",
                              "joint4","joint5","joint6",
                              ],
            effort_limit=45.0,
            saturation_effort=100.0,
            velocity_limit=5.0,
            stiffness=30.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Sirius Piper using DC motor.
"""
