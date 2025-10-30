import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# GO2ARM_USD = os.path.join(current_dir, "go2_arm.usd")

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

GO2ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/go2arm/go2_arm.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,
        #     contact_offset=0.02,
        #     rest_offset=0.005 ,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            # leg
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # arm
            "waist":0.0,
            "shoulder":0.0,
            "elbow":0.1,
            "forearm_roll":-0.0,
            "wrist_angle":-0.54,
            "wrist_rotate":0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=40.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "widow_arm": DCMotorCfg(
            joint_names_expr=["waist","shoulder","forearm_roll",
                              "wrist_angle","wrist_rotate","elbow",
                              ],
            effort_limit=10.0,
            saturation_effort=10.0,# TODO
            velocity_limit=3.14, #TODO
            stiffness=10.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

