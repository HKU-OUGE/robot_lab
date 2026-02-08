# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Arclab robots.

The following configurations are available:

* :obj:`ARCLAB_ARCDOG_CFG`: Arcdog robot with DC motor model for the legs


Reference: https://github.com/ruihuang1124/arcdog_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Arclab/Arcdog_adjustable_leg/arcdog_adjustable_leg.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Arclab/Arcdog_adjustable_leg_fixed_joint/arcdog_adjustable_leg_fixed_joint.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.05,
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
        # pos=(0.0, 0.0, 0.35), #for fixed joint
        joint_pos={
            "FL_hip_joint": 0.0,
            "FR_hip_joint": -0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": -0.0,
            "FL_thigh_joint": 0.7,
            "FR_thigh_joint": 0.7,
            "RL_thigh_joint": 0.7,
            "RR_thigh_joint": 0.7,
            "FL_calf_joint": -1.2,
            "FR_calf_joint": -1.2,
            "RL_calf_joint": -1.2,
            "RR_calf_joint": -1.2,
            "FL_box_joint": 0.04,
            "FR_box_joint": 0.04,
            "RL_box_joint": 0.04,
            "RR_box_joint": 0.04,
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
            stiffness=45.0,
            damping=1.5,
            friction=0.1,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*thigh_joint"],
            effort_limit=35.0,
            saturation_effort=50.0,
            velocity_limit=45.0,
            stiffness=50.0,
            damping=1.5,
            friction=0.1,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*calf_joint"],  
            effort_limit=80.0,
            saturation_effort=100.0,
            velocity_limit=45.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.1,
        ),
        # "extensions": DCMotorCfg(
        #     joint_names_expr=[".*_box_joint"],  
        #     effort_limit=200.0,
        #     saturation_effort=200.0,
        #     velocity_limit=0.2,
        #     stiffness=0.0,
        #     damping=2.0,
        #     friction=0.1,
        # ),
        "extensions": IdealPDActuatorCfg(
            joint_names_expr=[".*_box_joint"],  
            effort_limit=1000.0,
            velocity_limit=0.13,
            stiffness=8000.0,
            damping=100.0,
            armature=0.4,  
            friction=0.1,
        ),
    },
)



ARCLAB_ARCDOG_NEW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Arclab/Arcdog_new/arcdog.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.05,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FR_hip_joint": -0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": -0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.8,
            "FR_calf_joint": -1.8,
            "RL_calf_joint": -1.8,
            "RR_calf_joint": -1.8,
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
            stiffness=45.0,
            damping=1.5,
            friction=0.1,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*thigh_joint"],
            effort_limit=35.0,
            saturation_effort=50.0,
            velocity_limit=45.0,
            stiffness=50.0,
            damping=1.5,
            friction=0.1,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*calf_joint"],  
            effort_limit=80.0,
            saturation_effort=100.0,
            velocity_limit=45.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.1,
        ),
    },
)


UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Arclab/go2/go2_description.usd",
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
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 using DC motor.
"""