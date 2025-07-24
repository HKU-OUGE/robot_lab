# Copyright (c) 2025-2026 Junjie Zhu
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import torch
from isaaclab.managers.reward_manager import RewardManager as RewardManagerBase
# from omni.isaac.lab.managers.reward_manager import RewardManager as RewardManagerBase

class RewardManager(RewardManagerBase):

    def __init__(self,cfg, env):
        super().__init__(cfg, env)
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.arm_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) 

    def compute(self, dt: float) -> torch.Tensor:
        # reset computation
        self._reward_buf[:] = 0.0
        self.arm_reward_buf[:] = 0.0 
       
        # iterate over all the reward terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt

            # check if the term is a special term for arm
            if name.startswith("end_effector"):  ## TODO: 
                self.arm_reward_buf += value
            else:
                self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
        return self._reward_buf, self.arm_reward_buf