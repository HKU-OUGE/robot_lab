# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def joint_pos_rel_with_persistent_bias(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint position relative to default with a reset-sampled encoder bias."""
    asset: Articulation = env.scene[asset_cfg.name]
    if (
        not hasattr(env, "_highstep_joint_pos_observation_bias")
        or env._highstep_joint_pos_observation_bias.shape != asset.data.joint_pos.shape
    ):
        env._highstep_joint_pos_observation_bias = torch.zeros_like(asset.data.joint_pos)
    joint_ids = asset_cfg.joint_ids
    return (
        asset.data.joint_pos[:, joint_ids]
        - asset.data.default_joint_pos[:, joint_ids]
        + env._highstep_joint_pos_observation_bias[:, joint_ids]
    )


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def height_scan_disc(env, obs_cache=None, sensor_cfg=None, offset=0.5) -> torch.Tensor:
    heights = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    depth = (-heights).clamp(-2.0, 2.0)                 # 先夹到 [-2, 2]
    bins  = torch.round(depth * 10.0) / 10.0            # 对称量化到 0.1 网格
    return bins                                         # 已保证在 [-1, 1]


class HighstepTeacherPriorContext(ManagerTermBase):
    """Exact training-only context used by the frozen high-step Teacher prior.

    The returned columns are permanently ordered as ``height_delta``,
    ``command_x`` and ``front_rear_delta``.  This term is intentionally strict:
    structural loss of the frozen Teacher's scanner, command or foot contract
    raises instead of silently turning the prior label into zeros.
    """

    _EXPECTED_SENSOR_NAME = "height_scanner"
    _EXPECTED_COMMAND_NAME = "base_velocity"
    _EXPECTED_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    _EXPECTED_MASK_CONTRACT = (0.25, -0.20, 0.30)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        sensor_cfg = params["sensor_cfg"]
        foot_asset_cfg = params["foot_asset_cfg"]
        command_name = str(params["command_name"])
        mask_contract = (
            float(params["front_x_min"]),
            float(params["rear_x_max"]),
            float(params["max_abs_y"]),
        )
        if sensor_cfg.name != self._EXPECTED_SENSOR_NAME:
            raise RuntimeError(
                f"Teacher context sensor changed: {sensor_cfg.name!r} != {self._EXPECTED_SENSOR_NAME!r}"
            )
        if command_name != self._EXPECTED_COMMAND_NAME:
            raise RuntimeError(
                f"Teacher context command changed: {command_name!r} != {self._EXPECTED_COMMAND_NAME!r}"
            )
        if mask_contract != self._EXPECTED_MASK_CONTRACT:
            raise RuntimeError(
                "Teacher context ray-mask contract changed: "
                f"{mask_contract!r} != {self._EXPECTED_MASK_CONTRACT!r}"
            )

        try:
            self._height_sensor = env.scene[sensor_cfg.name]
            self._asset = env.scene[foot_asset_cfg.name]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Teacher context scene binding is unavailable") from error

        foot_names = tuple(foot_asset_cfg.body_names or ())
        foot_ids = foot_asset_cfg.body_ids
        if foot_names != self._EXPECTED_FOOT_NAMES or isinstance(foot_ids, slice):
            raise RuntimeError(
                "Teacher context requires resolved FL/FR/RL/RR foot bodies in permanent order"
            )
        try:
            resolved_foot_ids = tuple(int(body_id) for body_id in foot_ids)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Teacher context foot body IDs are invalid") from error
        if len(resolved_foot_ids) != 4 or len(set(resolved_foot_ids)) != 4:
            raise RuntimeError(
                f"Teacher context resolved invalid foot body IDs: {resolved_foot_ids!r}"
            )
        self._front_foot_ids = resolved_foot_ids[:2]
        self._rear_foot_ids = resolved_foot_ids[2:]

        ray_starts = getattr(self._height_sensor, "ray_starts", None)
        if (
            not isinstance(ray_starts, torch.Tensor)
            or ray_starts.ndim != 3
            or ray_starts.shape[0] < 1
            or ray_starts.shape[1] < 1
            or ray_starts.shape[2] < 2
        ):
            raise RuntimeError("Teacher context height scanner has invalid ray_starts")
        ray_xy = ray_starts[0, :, :2]
        if not bool(torch.isfinite(ray_xy).all().item()):
            raise RuntimeError("Teacher context height scanner has non-finite ray starts")
        side_mask = torch.abs(ray_xy[:, 1]) <= mask_contract[2]
        self._front_ray_mask = (ray_xy[:, 0] >= mask_contract[0]) & side_mask
        self._rear_ray_mask = (ray_xy[:, 0] <= mask_contract[1]) & side_mask
        if not bool(torch.any(self._front_ray_mask).item()) or not bool(
            torch.any(self._rear_ray_mask).item()
        ):
            raise RuntimeError("Teacher context ray masks are structurally empty")
        self._num_rays = int(ray_xy.shape[0])
        self._command_name = command_name

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor
    ) -> torch.Tensor:
        selected = values[:, mask]
        valid = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
        valid_count = valid.float().sum(dim=1)
        selected_sum = torch.where(valid, selected, torch.zeros_like(selected)).sum(dim=1)
        mean = selected_sum / torch.clamp(valid_count, min=1.0)
        return torch.where(valid_count > 0.0, mean, fallback)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        foot_asset_cfg: SceneEntityCfg,
        command_name: str,
        front_x_min: float,
        rear_x_max: float,
        max_abs_y: float,
    ) -> torch.Tensor:
        # Parameters remain in the signature because ObservationManager validates
        # them.  Their immutable values and resolved entities were bound above.
        del sensor_cfg, foot_asset_cfg, command_name, front_x_min, rear_x_max, max_abs_y
        if env is not self._env:
            raise RuntimeError("Teacher context term was called with a different environment")

        try:
            ray_hits_w = self._height_sensor.data.ray_hits_w
            sensor_pos_w = self._height_sensor.data.pos_w
            body_pos_w = self._asset.data.body_pos_w
        except AttributeError as error:
            raise RuntimeError("Teacher context live scanner or articulation data is unavailable") from error
        if (
            not isinstance(ray_hits_w, torch.Tensor)
            or ray_hits_w.ndim != 3
            or tuple(ray_hits_w.shape[:2]) != (self.num_envs, self._num_rays)
            or ray_hits_w.shape[2] < 3
        ):
            raise RuntimeError(
                f"Teacher context ray-hit shape is invalid: {getattr(ray_hits_w, 'shape', None)}"
            )
        if (
            not isinstance(sensor_pos_w, torch.Tensor)
            or sensor_pos_w.ndim != 2
            or sensor_pos_w.shape[0] != self.num_envs
            or sensor_pos_w.shape[1] < 3
        ):
            raise RuntimeError(
                f"Teacher context sensor pose shape is invalid: {getattr(sensor_pos_w, 'shape', None)}"
            )
        max_foot_id = max((*self._front_foot_ids, *self._rear_foot_ids))
        if (
            not isinstance(body_pos_w, torch.Tensor)
            or body_pos_w.ndim != 3
            or body_pos_w.shape[0] != self.num_envs
            or body_pos_w.shape[1] <= max_foot_id
            or body_pos_w.shape[2] < 3
        ):
            raise RuntimeError(
                f"Teacher context body-position shape is invalid: {getattr(body_pos_w, 'shape', None)}"
            )

        ray_hits_z = ray_hits_w[..., 2]
        sensor_z = sensor_pos_w[:, 2]
        front_terrain_z = self._masked_mean(ray_hits_z, self._front_ray_mask, sensor_z)
        rear_terrain_z = self._masked_mean(ray_hits_z, self._rear_ray_mask, front_terrain_z)
        height_delta = front_terrain_z - rear_terrain_z

        try:
            command = env.command_manager.get_command(self._command_name)
        except (AttributeError, KeyError, TypeError) as error:
            raise RuntimeError("Teacher context base_velocity command is unavailable") from error
        if (
            not isinstance(command, torch.Tensor)
            or command.ndim != 2
            or command.shape[0] != self.num_envs
            or command.shape[1] < 1
        ):
            raise RuntimeError(
                f"Teacher context command shape is invalid: {getattr(command, 'shape', None)}"
            )
        command_x = command[:, 0]

        front_foot_z = body_pos_w[:, self._front_foot_ids, 2].mean(dim=1)
        rear_foot_z = body_pos_w[:, self._rear_foot_ids, 2].mean(dim=1)
        front_rear_delta = front_foot_z - rear_foot_z
        context = torch.stack((height_delta, command_x, front_rear_delta), dim=-1)
        if tuple(context.shape) != (self.num_envs, 3):
            raise RuntimeError(f"Teacher context output shape changed: {tuple(context.shape)}")
        # Keep the formal fail-closed finite guard without synchronizing the GPU
        # with the CPU on every control step.
        torch._assert_async(
            torch.isfinite(context).all(), "Teacher context output contains non-finite values"
        )
        return context.clone()


class HighstepCriticalTransitionContext(ManagerTermBase):
    """Training-only B300 phase/contact labels for balanced distillation.

    The four output columns are permanently ordered as ``front_lift``,
    ``front_support``, ``first_rear`` and ``second_rear``.  They are stored in
    the rollout buffer only; none is routed into the 570-D Student history,
    critic input, actor input, TorchScript export, or deployment contract.
    """

    _EXPECTED_SENSOR_NAME = "height_scanner"
    _EXPECTED_CONTACT_SENSOR_NAME = "contact_forces"
    _EXPECTED_COMMAND_NAME = "base_velocity"
    _EXPECTED_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        sensor_cfg = params["sensor_cfg"]
        contact_sensor_cfg = params["contact_sensor_cfg"]
        foot_asset_cfg = params["foot_asset_cfg"]
        command_name = str(params["command_name"])
        if sensor_cfg.name != self._EXPECTED_SENSOR_NAME:
            raise RuntimeError("critical-transition height scanner binding changed")
        if contact_sensor_cfg.name != self._EXPECTED_CONTACT_SENSOR_NAME:
            raise RuntimeError("critical-transition contact sensor binding changed")
        if command_name != self._EXPECTED_COMMAND_NAME:
            raise RuntimeError("critical-transition command binding changed")
        if tuple(foot_asset_cfg.body_names or ()) != self._EXPECTED_FOOT_NAMES:
            raise RuntimeError("critical-transition foot order changed")

        self._height_sensor = env.scene[sensor_cfg.name]
        self._contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        self._asset = env.scene[foot_asset_cfg.name]
        self._command_name = command_name
        self._foot_ids = tuple(int(index) for index in foot_asset_cfg.body_ids)
        contact_ids = self._contact_sensor.find_bodies(list(self._EXPECTED_FOOT_NAMES))[0]
        self._contact_ids = tuple(int(index) for index in contact_ids)
        if len(self._foot_ids) != 4 or len(self._contact_ids) != 4:
            raise RuntimeError("critical-transition foot/contact binding is incomplete")

        ray_starts = self._height_sensor.ray_starts[0]
        ray_x = ray_starts[:, 0]
        ray_y = ray_starts[:, 1]
        side = torch.abs(ray_y) <= 0.30
        self._front_ray_mask = (ray_x >= 0.25) & side
        self._rear_ray_mask = (ray_x <= -0.20) & side
        if not bool(torch.any(self._front_ray_mask).item()) or not bool(
            torch.any(self._rear_ray_mask).item()
        ):
            raise RuntimeError("critical-transition terrain ray masks are empty")

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        selected = values[:, mask]
        finite = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
        count = finite.float().sum(dim=1)
        mean = torch.where(finite, selected, torch.zeros_like(selected)).sum(dim=1)
        mean = mean / torch.clamp(count, min=1.0)
        return torch.where(count > 0.0, mean, fallback)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        foot_asset_cfg: SceneEntityCfg,
        command_name: str,
        include_rl_preedge: bool = False,
        rl_preedge_vx_min: float = 0.65,
        rl_preedge_platform_half_width: float = 1.5,
        rl_preedge_outer_margin: float = 0.12,
    ) -> torch.Tensor:
        del sensor_cfg, contact_sensor_cfg, foot_asset_cfg, command_name
        if env is not self._env:
            raise RuntimeError("critical-transition context called with a different environment")

        hits_z = self._height_sensor.data.ray_hits_w[..., 2]
        sensor_z = self._height_sensor.data.pos_w[:, 2]
        front_z = self._masked_mean(hits_z, self._front_ray_mask, sensor_z)
        rear_z = self._masked_mean(hits_z, self._rear_ray_mask, front_z)
        height_delta = front_z - rear_z
        terrain_gate = torch.clamp((height_delta - 0.060) / 0.14, 0.0, 1.0)
        terrain_gate = terrain_gate * terrain_gate * (3.0 - 2.0 * terrain_gate)

        command_x = env.command_manager.get_command(self._command_name)[:, 0]
        command_gate = torch.clamp((command_x - 0.08) / 0.25, 0.0, 1.0)
        active = terrain_gate * command_gate

        foot_z = self._asset.data.body_pos_w[:, self._foot_ids, 2]
        force_z = torch.abs(
            self._contact_sensor.data.net_forces_w[:, self._contact_ids, 2]
        )
        contact_score = torch.clamp((force_z - 5.0) / 40.0, 0.0, 1.0)

        # Front lift is measured from the lower plane and shuts off once both
        # front feet have established support on the upper plane.
        front_lift_height = torch.max(foot_z[:, :2] - rear_z[:, None], dim=1).values
        front_lift_score = torch.clamp((front_lift_height - 0.04) / 0.14, 0.0, 1.0)
        front_top_score = torch.clamp((foot_z[:, :2] - front_z[:, None] + 0.05) / 0.10, 0.0, 1.0)
        front_support_score = torch.min(front_top_score * contact_score[:, :2], dim=1).values
        front_support = active * front_support_score
        front_lift = active * front_lift_score * (1.0 - front_support_score)

        # Rear gates exactly reuse the frozen rear-support clearance semantics.
        rear_clearance_score = torch.clamp(
            (foot_z[:, 2:] - front_z[:, None] + 0.18) / 0.18, 0.0, 1.0
        )
        first_score = torch.max(rear_clearance_score, dim=1).values
        second_score = torch.min(rear_clearance_score, dim=1).values
        first_rear = active * torch.clamp((first_score - 0.20) / 0.32, 0.0, 1.0)
        second_rear = active * torch.clamp((second_score - 0.52) / 0.30, 0.0, 1.0)

        columns = [front_lift, front_support, first_rear, second_rear]
        if include_rl_preedge:
            # Training-only causal label.  The first-rear pre-clearance
            # boundary is spatial: RL has reached the outer 12 cm before the
            # platform entry edge, while both front feet already carry load
            # on the upper plane and RL itself has not established top
            # support.  Do not approximate this with vertical foot clearance:
            # that only becomes active after the pre-edge event.
            rl_top_score = torch.clamp(
                (foot_z[:, 2] - front_z + 0.05) / 0.10, 0.0, 1.0
            )
            front_supported = torch.all(
                (front_top_score >= 0.5) & (force_z[:, :2] > 5.0), dim=1
            )
            unit_forward_b = torch.zeros_like(self._asset.data.root_pos_w)
            unit_forward_b[:, 0] = 1.0
            heading_w = math_utils.quat_apply_yaw(
                self._asset.data.root_quat_w, unit_forward_b
            )[:, :2]
            heading_w = heading_w / torch.clamp(
                torch.linalg.vector_norm(heading_w, dim=1, keepdim=True), min=1.0e-6
            )
            edge_distance = float(rl_preedge_platform_half_width) / torch.clamp(
                torch.amax(torch.abs(heading_w), dim=1), min=1.0e-6
            )
            rl_xy = self._asset.data.body_pos_w[:, self._foot_ids[2], :2]
            outward_distance = torch.sum(
                (rl_xy - env.scene.env_origins[:, :2]) * (-heading_w), dim=1
            )
            rl_edge_margin = edge_distance - outward_distance
            rl_preedge = (
                (command_x >= float(rl_preedge_vx_min))
                & front_supported
                & (rl_top_score < 0.5)
                & (rl_edge_margin >= -float(rl_preedge_outer_margin))
            ).to(dtype=front_lift.dtype)
            columns.append(rl_preedge)
        context = torch.stack(columns, dim=-1)
        torch._assert_async(
            torch.isfinite(context).all(),
            "critical-transition context contains non-finite values",
        )
        return context.clone()
