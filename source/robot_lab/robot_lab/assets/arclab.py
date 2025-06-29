# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Arclab robots.

The following configurations are available:

* :obj:`ARCLAB_ARCDOG_CFG`: Arcdog robot with DC motor model for the legs


Reference: https://github.com/ruihuang1124/arcdog_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


ARCLAB_ARCDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Arclab/Arcdog/arcdog.usd",
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
        pos=(0.0, 0.0, 0.44),
        joint_pos={
            "FL_HAA": 0.00,
            "FR_HAA": -0.00,
            "RL_HAA": 0.00,
            "RR_HAA": -0.00,
            "FL_HFE": 0.60,
            "FR_HFE": 0.60,
            "RL_HFE": 0.60,
            "RR_HFE": 0.60,
            "FL_KFE": -1.00,
            "FR_KFE": -1.00,
            "RL_KFE": -1.00,
            "RR_KFE": -1.00,
            "FL_box_joint": 0.075,
            "FR_box_joint": 0.075,
            "RL_box_joint": 0.075,
            "RR_box_joint": 0.075,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_fe": DCMotorCfg(
            joint_names_expr=[".*FE"],
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
        "legs_aa": DCMotorCfg(
            joint_names_expr=[".*AA"],
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
        "extensions": DCMotorCfg(
            joint_names_expr=[".*_box_joint"],  
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=0.005,
            stiffness=10000.0,
            damping=200.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree A1 using DC motor.
"""
