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
            "RL_hip_joint": -0.00,
            "RR_hip_joint": -0.00,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": -0.8,
            "RR_thigh_joint": -0.8,
            "FL_calf_joint": -1.6,
            "FR_calf_joint": -1.6,
            "RL_calf_joint": 1.6,
            "RR_calf_joint": 1.6,
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
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*calf_joint"],  
            effort_limit=80.0,
            saturation_effort=100.0,
            velocity_limit=45.0,
            stiffness=50.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)

CUHKLRL_SIRIUS_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius_wheel/sirius_wheel_merge_v3_ori.usd", # 初步
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius_wheel/sirius_wheel_merge_v3_pre.usd", # 调优
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.05,
            angular_damping=0.05,
            max_linear_velocity=200.0,
            max_angular_velocity=200.0,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.60),
        joint_pos={
            "LF_HAA": 0.00,
            "LH_HAA": 0.00,
            "RF_HAA": -0.00,
            "RH_HAA": -0.00,
            "LF_HFE": 0.95,
            "LH_HFE": -0.95,
            "RF_HFE": 0.95,
            "RH_HFE": -0.95,
            "LF_KNEE": -1.6,
            "LH_KNEE": 1.6,
            "RF_KNEE": -1.6,
            "RH_KNEE": 1.6,
            "LF_WHEEL": 0.00,
            "LH_WHEEL": 0.00,
            "RF_WHEEL": 0.00,
            "RH_WHEEL": 0.00,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip": DCMotorCfg(
            joint_names_expr=[".*_HAA"],
            effort_limit=40.0,
            saturation_effort=40.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*_HFE"],
            effort_limit=40.0,
            saturation_effort=40.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*_KNEE"],  
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=8.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_wheel": DCMotorCfg(
            joint_names_expr=[".*_WHEEL"],  
            effort_limit=40.0,
            saturation_effort=40.0,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=3.0,
            friction=0.0,
        ),
    },
)

CUHKLRL_SIRIUS_WHEEL_STAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius_wheel/sirius_wheel_merge_stand.usd",
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
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "LF_HAA": 0.00,
            "LH_HAA": 0.00,
            "RF_HAA": -0.00,
            "RH_HAA": -0.00,
            "LF_HFE": 0.67,
            "LH_HFE": -0.67,
            "RF_HFE": 0.67,
            "RH_HFE": -0.67,
            "LF_KFE": -1.3,
            "LH_KFE": 1.3,
            "RF_KFE": -1.3,
            "RH_KFE": 1.3,
            "LF_WHEEL": 0.00,
            "LH_WHEEL": 0.00,
            "RF_WHEEL": 0.00,
            "RH_WHEEL": 0.00,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_hip": DCMotorCfg(
            joint_names_expr=[".*_HAA"],
            effort_limit=35.0,
            saturation_effort=50.0,
            velocity_limit=45.0,
            stiffness=25.0,
            damping=3.0,
            friction=0.0,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*_HFE"],
            effort_limit=35.0,
            saturation_effort=50.0,
            velocity_limit=45.0,
            stiffness=25.0,
            damping=3.0,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*_KFE"],  
            effort_limit=80.0,
            saturation_effort=100.0,
            velocity_limit=45.0,
            stiffness=25.0,
            damping=3.0,
            friction=0.0,
        ),
        "legs_wheel": DCMotorCfg(
            joint_names_expr=[".*_WHEEL"],  
            effort_limit=80.0,
            saturation_effort=100.0,
            velocity_limit=45.0,
            stiffness=0.0,
            damping=3.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Sirius using DC motor.
"""



CUHKLRL_SIRIUS_WHEEL_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius_Wheel_Piper/sirius_wheel_piper.usd", # 初步
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/CUHKLRL/Sirius_wheel/sirius_wheel_merge_v3_pre.usd", # 调优
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.05,
            angular_damping=0.05,
            max_linear_velocity=200.0,
            max_angular_velocity=200.0,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "LF_HAA": 0.00,
            "LH_HAA": 0.00,
            "RF_HAA": -0.00,
            "RH_HAA": -0.00,
            "LF_HFE": 0.95,
            "LH_HFE": -0.95,
            "RF_HFE": 0.95,
            "RH_HFE": -0.95,
            "LF_KNEE": -1.6,
            "LH_KNEE": 1.6,
            "RF_KNEE": -1.6,
            "RH_KNEE": 1.6,
            "LF_WHEEL": 0.00,
            "LH_WHEEL": 0.00,
            "RF_WHEEL": 0.00,
            "RH_WHEEL": 0.00,
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
            joint_names_expr=[".*_HAA"],
            effort_limit=40.0,
            saturation_effort=40.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_thigh": DCMotorCfg(
            joint_names_expr=[".*_HFE"],
            effort_limit=40.0,
            saturation_effort=40.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[".*_KNEE"],  
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=8.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "legs_wheel": DCMotorCfg(
            joint_names_expr=[".*_WHEEL"],  
            effort_limit=40.0,
            saturation_effort=40.0,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=3.0,
            friction=0.0,
        ),
        "piper_arm": DCMotorCfg(
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