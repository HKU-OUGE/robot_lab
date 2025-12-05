import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# GO2ARM_USD = os.path.join(current_dir, "go2_arm.usd")

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab.actuators import ImplicitActuatorCfg

##
# Configuration
##

GO2PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Arclab/go2_piper_center/go2_piper_test.usd",
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
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            # arm
            # "joint1": 0.0,
            # "joint2": 1.7,
            # "joint3": -0.8,
            # "joint4": 0.0,
            # "joint5": -0.8,
            # "joint6": 0.0,
            # "joint1": 0.0,
            # "joint2": 2.2,
            # "joint3": -1.5,
            # "joint4": 0.0,
            # "joint5": -0.6, # -0.6 0.7
            # "joint6": 0.0,
            # "joint1": 0.0,
            # "joint2": 1.2,
            # "joint3": -0.5,
            # "joint4": 0.0,
            # "joint5": -0.55, # -0.6 0.7
            # "joint6": 0.0,
            # #####################
            "joint1": 0.0,
            "joint2": 0.7,
            "joint3": -1.2,
            "joint4": 0.0,
            "joint5": 1.0, # -0.6 0.7
            "joint6": 0.0,
            #####################
            # "joint1": 0.0,
            # "joint2": 0.0,
            # "joint3": 0.0,
            # "joint4": 0.0,
            # "joint5": 0.0, # -0.6 0.7
            # "joint6": 0.0,
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
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
        ),
        # "arm": DCMotorCfg(
        #     joint_names_expr=["joint1","joint2","joint3",
        #                       "joint4","joint5","joint6",
        #                       ],
        #     effort_limit=10.0,
        #     saturation_effort=10.0,# TODO
        #     velocity_limit=3.14, #TODO
        #     stiffness=50.0,
        #     damping=2.0,
        #     friction=0.0,
        # ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], # 替换为你机械臂的关节名称
            stiffness={".*": 400.0}, # 原始代码中的值
            damping={".*": 20.0},    # 原始代码中的值
        ),
    },
)


GO2PIPER_NEW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Arclab/go2_piper_r2/go2_piper.usd",
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
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            # arm
            "joint1": 0.0,
            "joint2": 1.2,
            "joint3": -0.5,
            "joint4": 0.0,
            "joint5": -0.55, # -0.6 0.7
            "joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
            # --- 足部驱动器配置 ---
            # 通常用于 locomotion 的中低刚度 PD 控制
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                stiffness={".*": 35.0},  # 根据你的机器人调整，比如 Go1 大概是 20-80
                damping={".*": 1.0},
            ),
            # --- 机械臂驱动器配置 (关键!) ---
            # 必须有足够高的刚度来精确跟踪 IK 计算出的位置目标
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], # 替换为你机械臂的关节名称
                stiffness={".*": 400.0}, # 原始代码中的值
                damping={".*": 40.0},    # 原始代码中的值
            ),
            # 夹爪通常单独配置
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint7","joint8"],
                stiffness={".*": 800.0},
                damping={".*": 40.0},
            ),
        },
        # 确保包含末端执行器的 Body 名称，IK 需要用到
        # ee_body_name="gripper_base", # 替换为你 URDF 中末端 link 的名称
)

