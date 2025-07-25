# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents
# from .... import low_level_loc_vel_env
# import robot_lab.tasks.manip_loco.low_level.low_level_loc_vel_env as manip_loco_env
# from robot_lab.tasks.manip_loco.low_level.low_level_loc_vel_env import ManagerRLEnv
# from low_level_loc_vel_env import ManagerRLEnv
# from low_level.low_level_loc_vel_env import ManagerRLEnv

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-ManipLocoLowLevel-Flat-Go2-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    # entry_point="robot_lab.tasks.manip_loco.low_level.low_level_loc_vel_env:ManagerRLEnv",
    # entry_point="ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2ARMFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-ManipLocoLowLevel-Rough-Go2-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    # entry_point="low_level_loc_vel_env:ManagerRLEnv",
    # entry_point="ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2ARMRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmRoughPPORunnerCfg",
    },
)
