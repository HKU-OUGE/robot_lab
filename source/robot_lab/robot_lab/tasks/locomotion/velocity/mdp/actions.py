# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class LateralStepBoxBiasJointPositionAction(JointPositionAction):
    """Joint-position action with a bounded lateral-step prior for prismatic box joints.

    The policy still outputs the same raw action vector.  After the standard
    action scaling/default-offset mapping, this term adds a small box-joint
    target bias when the left and right foot pairs are on noticeably different
    heights.  Lower-side box joints are extended and upper-side box joints are
    shortened, then clipped to the configured physical target range.
    """

    cfg: "LateralStepBoxBiasJointPositionActionCfg"

    def __init__(self, cfg: "LateralStepBoxBiasJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)

        foot_body_names = cfg.foot_body_names
        box_joint_names = cfg.box_joint_names
        foot_names = [foot_body_names[k] for k in ("FL", "FR", "RL", "RR")]
        box_names = [box_joint_names[k] for k in ("FL", "FR", "RL", "RR")]

        self._foot_body_ids, _ = self._asset.find_bodies(foot_names, preserve_order=True)
        action_id_by_joint = {joint_name: action_id for action_id, joint_name in enumerate(self._joint_names)}
        missing_box_joints = [joint_name for joint_name in box_names if joint_name not in action_id_by_joint]
        if missing_box_joints:
            raise ValueError(
                f"Box joints {missing_box_joints} are not controlled by the action term. "
                f"Resolved action joints: {self._joint_names}"
            )

        self._box_action_ids = torch.tensor(
            [action_id_by_joint[joint_name] for joint_name in box_names],
            device=self.device,
            dtype=torch.long,
        )
        self._last_lateral_step_box_bias = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_lateral_step_box_gate = torch.zeros(self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        foot_z = self._asset.data.body_pos_w[:, self._foot_body_ids, 2]
        left_z = 0.5 * (foot_z[:, 0] + foot_z[:, 2])
        right_z = 0.5 * (foot_z[:, 1] + foot_z[:, 3])
        side_height_delta = left_z - right_z
        if self.cfg.invert_side_height_delta:
            side_height_delta = -side_height_delta

        gate_width = max(self.cfg.gate_width, 1.0e-6)
        gate = torch.clamp((torch.abs(side_height_delta) - self.cfg.height_threshold) / gate_width, 0.0, 1.0)

        lower_bias = torch.full_like(gate, self.cfg.lower_side_bias)
        upper_bias = torch.full_like(gate, self.cfg.upper_side_bias)
        left_is_lower = side_height_delta < 0.0
        left_bias = torch.where(left_is_lower, lower_bias, upper_bias)
        right_bias = torch.where(left_is_lower, upper_bias, lower_bias)
        box_bias = torch.stack((left_bias, right_bias, left_bias, right_bias), dim=1) * gate.unsqueeze(1)

        box_targets = self._processed_actions[:, self._box_action_ids] + box_bias
        box_targets = torch.clamp(box_targets, min=self.cfg.min_box_target, max=self.cfg.max_box_target)
        self._processed_actions[:, self._box_action_ids] = box_targets

        self._last_lateral_step_box_bias[:] = box_bias
        self._last_lateral_step_box_gate[:] = gate


@configclass
class LateralStepBoxBiasJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`LateralStepBoxBiasJointPositionAction`."""

    class_type: type[ActionTerm] = LateralStepBoxBiasJointPositionAction

    foot_body_names: dict[str, str] = {
        "FL": "FL_foot",
        "FR": "FR_foot",
        "RL": "RL_foot",
        "RR": "RR_foot",
    }
    box_joint_names: dict[str, str] = {
        "FL": "FL_box_joint",
        "FR": "FR_box_joint",
        "RL": "RL_box_joint",
        "RR": "RR_box_joint",
    }
    height_threshold: float = 0.02
    gate_width: float = 0.06
    lower_side_bias: float = 0.018
    upper_side_bias: float = -0.014
    min_box_target: float = 0.000
    max_box_target: float = 0.060
    invert_side_height_delta: bool = False


class ForwardHighstepBoxBiasJointPositionAction(JointPositionAction):
    """Joint-position action with a forward high-step prior for box joints.

    The prior is intentionally limited to the four prismatic box joints.  When
    the yaw-aligned height scanner sees a higher patch in front of the robot
    and the command asks the robot to move forward, front box joints are
    shortened while rear box joints are extended.  The policy action remains a
    residual, so PPO can still learn the gait.
    """

    cfg: "ForwardHighstepBoxBiasJointPositionActionCfg"

    def __init__(self, cfg: "ForwardHighstepBoxBiasJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)

        box_names = [cfg.box_joint_names[k] for k in ("FL", "FR", "RL", "RR")]
        action_id_by_joint = {joint_name: action_id for action_id, joint_name in enumerate(self._joint_names)}
        missing_box_joints = [joint_name for joint_name in box_names if joint_name not in action_id_by_joint]
        if missing_box_joints:
            raise ValueError(
                f"Box joints {missing_box_joints} are not controlled by the action term. "
                f"Resolved action joints: {self._joint_names}"
            )

        self._box_action_ids = torch.tensor(
            [action_id_by_joint[joint_name] for joint_name in box_names],
            device=self.device,
            dtype=torch.long,
        )
        self._last_forward_highstep_box_bias = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_forward_highstep_gate = torch.zeros(self.num_envs, device=self.device)
        self._last_forward_highstep_height_delta = torch.zeros(self.num_envs, device=self.device)

        self._height_sensor = None
        self._front_ray_mask = None
        self._rear_ray_mask = None
        try:
            self._height_sensor = self._env.scene[cfg.height_sensor_name]
            ray_starts = self._height_sensor.ray_starts[0]
            ray_x = ray_starts[:, 0]
            ray_y = ray_starts[:, 1]
            side_mask = torch.abs(ray_y) <= cfg.max_abs_y
            self._front_ray_mask = (ray_x >= cfg.front_x_min) & side_mask
            self._rear_ray_mask = (ray_x <= cfg.rear_x_max) & side_mask
        except (KeyError, AttributeError, IndexError):
            self._height_sensor = None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        if mask is None or not bool(torch.any(mask).item()):
            return fallback
        selected = values[:, mask]
        valid = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
        valid_count = valid.float().sum(dim=1)
        selected_sum = torch.where(valid, selected, torch.zeros_like(selected)).sum(dim=1)
        mean = selected_sum / torch.clamp(valid_count, min=1.0)
        return torch.where(valid_count > 0.0, mean, fallback)

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        gate = torch.zeros(self.num_envs, device=self.device)
        height_delta = torch.zeros_like(gate)
        if self._height_sensor is not None and self._front_ray_mask is not None and self._rear_ray_mask is not None:
            ray_hits_z = self._height_sensor.data.ray_hits_w[..., 2]
            sensor_z = self._height_sensor.data.pos_w[:, 2]
            front_z = self._masked_mean(ray_hits_z, self._front_ray_mask, sensor_z)
            rear_z = self._masked_mean(ray_hits_z, self._rear_ray_mask, front_z)
            height_delta = front_z - rear_z

            height_gate_width = max(self.cfg.height_gate_width, 1.0e-6)
            height_gate = self._smoothstep((height_delta - self.cfg.height_threshold) / height_gate_width)

            command = self._env.command_manager.get_command(self.cfg.command_name)
            cmd_gate_width = max(self.cfg.command_gate_width, 1.0e-6)
            cmd_gate = torch.clamp((command[:, 0] - self.cfg.min_forward_command) / cmd_gate_width, 0.0, 1.0)
            gate = height_gate * cmd_gate

        front_bias = torch.full_like(gate, self.cfg.front_box_bias)
        rear_bias = torch.full_like(gate, self.cfg.rear_box_bias)
        box_bias = torch.stack((front_bias, front_bias, rear_bias, rear_bias), dim=1) * gate.unsqueeze(1)

        box_targets = self._processed_actions[:, self._box_action_ids] + box_bias
        box_targets = torch.clamp(box_targets, min=self.cfg.min_box_target, max=self.cfg.max_box_target)
        self._processed_actions[:, self._box_action_ids] = box_targets

        self._last_forward_highstep_box_bias[:] = box_bias
        self._last_forward_highstep_gate[:] = gate
        self._last_forward_highstep_height_delta[:] = height_delta


@configclass
class ForwardHighstepBoxBiasJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`ForwardHighstepBoxBiasJointPositionAction`."""

    class_type: type[ActionTerm] = ForwardHighstepBoxBiasJointPositionAction

    box_joint_names: dict[str, str] = {
        "FL": "FL_box_joint",
        "FR": "FR_box_joint",
        "RL": "RL_box_joint",
        "RR": "RR_box_joint",
    }
    height_sensor_name: str = "height_scanner"
    command_name: str = "base_velocity"
    front_x_min: float = 0.25
    rear_x_max: float = -0.20
    max_abs_y: float = 0.30
    height_threshold: float = 0.06
    height_gate_width: float = 0.14
    min_forward_command: float = 0.08
    command_gate_width: float = 0.25
    front_box_bias: float = -0.018
    rear_box_bias: float = 0.018
    min_box_target: float = 0.000
    max_box_target: float = 0.060


class PhasedHighstepBoxBiasJointPositionAction(JointPositionAction):
    """Forward high-step box-joint prior split into reach and rear-push phases.

    For the current robot, smaller box-joint position means a longer telescopic
    leg.  A single front/rear bias is too blunt for high steps: the useful
    motion first extends the front legs to catch the step, then extends the
    rear legs to push the body over the edge.
    """

    cfg: "PhasedHighstepBoxBiasJointPositionActionCfg"

    def __init__(self, cfg: "PhasedHighstepBoxBiasJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)

        box_names = [cfg.box_joint_names[k] for k in ("FL", "FR", "RL", "RR")]
        action_id_by_joint = {joint_name: action_id for action_id, joint_name in enumerate(self._joint_names)}
        missing_box_joints = [joint_name for joint_name in box_names if joint_name not in action_id_by_joint]
        if missing_box_joints:
            raise ValueError(
                f"Box joints {missing_box_joints} are not controlled by the action term. "
                f"Resolved action joints: {self._joint_names}"
            )

        self._box_action_ids = torch.tensor(
            [action_id_by_joint[joint_name] for joint_name in box_names],
            device=self.device,
            dtype=torch.long,
        )
        front_foot_names = [cfg.front_foot_body_names[k] for k in ("FL", "FR")]
        rear_foot_names = [cfg.rear_foot_body_names[k] for k in ("RL", "RR")]
        self._front_foot_ids, _ = self._asset.find_bodies(front_foot_names, preserve_order=True)
        self._rear_foot_ids, _ = self._asset.find_bodies(rear_foot_names, preserve_order=True)

        self._last_phased_highstep_box_bias = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_phased_highstep_gate = torch.zeros(self.num_envs, device=self.device)
        self._last_phased_highstep_reach_gate = torch.zeros(self.num_envs, device=self.device)
        self._last_phased_highstep_push_gate = torch.zeros(self.num_envs, device=self.device)
        self._last_phased_highstep_height_delta = torch.zeros(self.num_envs, device=self.device)

        self._height_sensor = None
        self._front_ray_mask = None
        self._rear_ray_mask = None
        try:
            self._height_sensor = self._env.scene[cfg.height_sensor_name]
            ray_starts = self._height_sensor.ray_starts[0]
            ray_x = ray_starts[:, 0]
            ray_y = ray_starts[:, 1]
            side_mask = torch.abs(ray_y) <= cfg.max_abs_y
            self._front_ray_mask = (ray_x >= cfg.front_x_min) & side_mask
            self._rear_ray_mask = (ray_x <= cfg.rear_x_max) & side_mask
        except (KeyError, AttributeError, IndexError):
            self._height_sensor = None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        if mask is None or not bool(torch.any(mask).item()):
            return fallback
        selected = values[:, mask]
        valid = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
        valid_count = valid.float().sum(dim=1)
        selected_sum = torch.where(valid, selected, torch.zeros_like(selected)).sum(dim=1)
        mean = selected_sum / torch.clamp(valid_count, min=1.0)
        return torch.where(valid_count > 0.0, mean, fallback)

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        gate = torch.zeros(self.num_envs, device=self.device)
        height_delta = torch.zeros_like(gate)
        if self._height_sensor is not None and self._front_ray_mask is not None and self._rear_ray_mask is not None:
            ray_hits_z = self._height_sensor.data.ray_hits_w[..., 2]
            sensor_z = self._height_sensor.data.pos_w[:, 2]
            front_z = self._masked_mean(ray_hits_z, self._front_ray_mask, sensor_z)
            rear_z = self._masked_mean(ray_hits_z, self._rear_ray_mask, front_z)
            height_delta = front_z - rear_z

            height_gate_width = max(self.cfg.height_gate_width, 1.0e-6)
            height_gate = self._smoothstep((height_delta - self.cfg.height_threshold) / height_gate_width)

            command = self._env.command_manager.get_command(self.cfg.command_name)
            cmd_gate_width = max(self.cfg.command_gate_width, 1.0e-6)
            cmd_gate = torch.clamp((command[:, 0] - self.cfg.min_forward_command) / cmd_gate_width, 0.0, 1.0)
            gate = height_gate * cmd_gate

        front_z = self._asset.data.body_pos_w[:, self._front_foot_ids, 2].mean(dim=1)
        rear_z = self._asset.data.body_pos_w[:, self._rear_foot_ids, 2].mean(dim=1)
        front_rear_delta = front_z - rear_z
        commit_gate = self._smoothstep(
            (front_rear_delta - self.cfg.commit_height_delta_min)
            / max(self.cfg.commit_height_delta_target - self.cfg.commit_height_delta_min, 1.0e-6)
        )

        reach_gate = gate * (1.0 - commit_gate)
        push_gate = torch.maximum(gate * commit_gate, self.cfg.commit_gate_floor * commit_gate)

        reach_bias = torch.tensor(
            [
                self.cfg.front_reach_box_bias,
                self.cfg.front_reach_box_bias,
                self.cfg.rear_approach_box_bias,
                self.cfg.rear_approach_box_bias,
            ],
            device=self.device,
            dtype=self._processed_actions.dtype,
        )
        push_bias = torch.tensor(
            [
                self.cfg.front_support_box_bias,
                self.cfg.front_support_box_bias,
                self.cfg.rear_push_box_bias,
                self.cfg.rear_push_box_bias,
            ],
            device=self.device,
            dtype=self._processed_actions.dtype,
        )
        box_bias = reach_gate.unsqueeze(1) * reach_bias + push_gate.unsqueeze(1) * push_bias
        update_count = float(self._env.common_step_counter) / max(float(self.cfg.num_steps_per_update), 1.0)
        prior_ramp = max(float(self.cfg.prior_full_update - self.cfg.prior_start_update), 1.0)
        prior_scale = (update_count - float(self.cfg.prior_start_update)) / prior_ramp
        prior_scale = float(max(0.0, min(1.0, prior_scale)))
        box_bias = box_bias * prior_scale

        box_targets = self._processed_actions[:, self._box_action_ids] + box_bias
        box_targets = torch.clamp(box_targets, min=self.cfg.min_box_target, max=self.cfg.max_box_target)
        self._processed_actions[:, self._box_action_ids] = box_targets

        self._last_phased_highstep_box_bias[:] = box_bias
        self._last_phased_highstep_gate[:] = torch.maximum(reach_gate, push_gate) * prior_scale
        self._last_phased_highstep_reach_gate[:] = reach_gate * prior_scale
        self._last_phased_highstep_push_gate[:] = push_gate * prior_scale
        self._last_phased_highstep_height_delta[:] = height_delta


@configclass
class PhasedHighstepBoxBiasJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`PhasedHighstepBoxBiasJointPositionAction`."""

    class_type: type[ActionTerm] = PhasedHighstepBoxBiasJointPositionAction

    box_joint_names: dict[str, str] = {
        "FL": "FL_box_joint",
        "FR": "FR_box_joint",
        "RL": "RL_box_joint",
        "RR": "RR_box_joint",
    }
    front_foot_body_names: dict[str, str] = {"FL": "FL_foot", "FR": "FR_foot"}
    rear_foot_body_names: dict[str, str] = {"RL": "RL_foot", "RR": "RR_foot"}
    height_sensor_name: str = "height_scanner"
    command_name: str = "base_velocity"
    front_x_min: float = 0.15
    rear_x_max: float = -0.20
    max_abs_y: float = 0.35
    height_threshold: float = 0.035
    height_gate_width: float = 0.12
    min_forward_command: float = 0.05
    command_gate_width: float = 0.20
    commit_height_delta_min: float = 0.04
    commit_height_delta_target: float = 0.18
    commit_gate_floor: float = 0.25
    front_reach_box_bias: float = -0.016
    rear_approach_box_bias: float = 0.004
    front_support_box_bias: float = 0.004
    rear_push_box_bias: float = -0.018
    min_box_target: float = 0.000
    max_box_target: float = 0.060
    prior_start_update: int = 300
    prior_full_update: int = 900
    num_steps_per_update: int = 24
