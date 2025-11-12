# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="RobotLab-Isaac-ManipLocoLowLevel-Rough-SiriusW-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:CUHKLRLSiriusWPiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWPiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-ManipLocoLowLevel-Flat-SiriusW-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CUHKLRLSiriusWPiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWPiperFlatPPORunnerCfg",
    },
)