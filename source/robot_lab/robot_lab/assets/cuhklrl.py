# Copyright (c) 2025-2026 Tang Tianyang
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Arclab robots.

The following configurations are available:

* :obj:`CUHKLRL_SIRIUS_CFG`: Sirius robot with DC motor model for the legs


Reference: https://github.com/ruihuang1124/quadruped_control_ros2
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


CUHKLRL_SIRIUS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius/sirius.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "FL_hip_joint": 0.00,
            "FR_hip_joint": -0.00,
            "RL_hip_joint": 0.00,
            "RR_hip_joint": -0.00,
            "FL_thigh_joint": 0.00,
            "FR_thigh_joint": 0.00,
            "RL_thigh_joint": 0.00,
            "RR_thigh_joint": 0.00,
            "FL_calf_joint": -0.00,
            "FR_calf_joint": -0.00,
            "RL_calf_joint": -0.00,
            "RR_calf_joint": -0.00,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip": DCMotorCfg(
            joint_names_expr=[".*hip_joint"],
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*thigh_joint"],
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],  
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Sirius using DC motor.
"""
