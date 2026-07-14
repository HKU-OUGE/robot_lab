# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import math as math_utils

from isaaclab.managers import EventTermCfg
from isaaclab.managers import ManagerTermBase
# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv


def randomize_highstep_joint_observation_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hip_bias_range: tuple[float, float] = (-0.020, 0.020),
    leg_bias_range: tuple[float, float] = (-0.012, 0.012),
    box_bias_range: tuple[float, float] = (-0.0015, 0.0015),
) -> None:
    """Sample persistent per-joint encoder offsets for one episode."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    if (
        not hasattr(env, "_highstep_joint_pos_observation_bias")
        or env._highstep_joint_pos_observation_bias.shape != asset.data.joint_pos.shape
    ):
        env._highstep_joint_pos_observation_bias = torch.zeros_like(asset.data.joint_pos)

    bias = env._highstep_joint_pos_observation_bias
    bias[env_ids] = 0.0
    for joint_id, joint_name in enumerate(asset.joint_names):
        if "hip_joint" in joint_name:
            low, high = hip_bias_range
        elif "box_joint" in joint_name:
            low, high = box_bias_range
        else:
            low, high = leg_bias_range
        bias[env_ids, joint_id] = low + (high - low) * torch.rand(len(env_ids), device=asset.device)


def reset_root_state_highstep_approach(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    flat_patch_key: str = "target",
    high_origin_threshold: float = 0.035,
    low_patch_z_margin: float = 0.04,
    approach_ratio: float = 0.75,
    approach_distance_range: tuple[float, float] = (1.65, 2.65),
    approach_yaw_noise_range: tuple[float, float] = (-0.25, 0.25),
):
    """Reset high-step training episodes near climb approaches when possible.

    Standard mesh ``box`` and ``pyramid_stairs`` terrains place their terrain
    origin on the high central platform. A uniform reset around that origin
    mostly creates "walk down/from top" episodes. For high-step training, a
    subset of high-origin terrains is reset on lower flat patches and yawed
    toward the origin, while inverse/pit/flat terrains keep the usual reset.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    root_states = asset.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    num_envs = len(env_ids)

    # Default Isaac Lab style uniform reset around env origin.
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    pose_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=asset.device)
    positions = root_states[:, 0:3] + env_origins + pose_samples[:, 0:3]

    terrain = getattr(env.scene, "terrain", None)
    flat_patches = None if terrain is None else terrain.flat_patches.get(flat_patch_key)
    terrain_levels = None if terrain is None else getattr(terrain, "terrain_levels", None)
    terrain_types = None if terrain is None else getattr(terrain, "terrain_types", None)

    if flat_patches is not None and terrain_levels is not None and terrain_types is not None:
        levels = terrain_levels[env_ids]
        types = terrain_types[env_ids]
        patches = flat_patches[levels, types]  # (N, num_patches, 3), world-frame surface positions

        high_origin = env_origins[:, 2] > high_origin_threshold
        if approach_ratio < 1.0:
            high_origin &= torch.rand(num_envs, device=asset.device) < approach_ratio

        if torch.any(high_origin):
            patch_z = patches[:, :, 2]
            low_patch_mask = patch_z < (env_origins[:, 2:3] - low_patch_z_margin)

            patch_xy = patches[:, :, :2]
            origin_xy = env_origins[:, None, :2]
            dist_to_origin = torch.norm(patch_xy - origin_xy, dim=-1)
            target_dist = math_utils.sample_uniform(
                approach_distance_range[0],
                approach_distance_range[1],
                (num_envs,),
                device=asset.device,
            )
            score = torch.abs(dist_to_origin - target_dist[:, None])
            score = torch.where(low_patch_mask, score, torch.full_like(score, 1.0e6))

            # If the patch sampler did not find a lower patch for a tile, fall
            # back to the lowest available flat patch instead of spawning on top.
            no_low_patch = ~torch.any(low_patch_mask, dim=1)
            score = torch.where(no_low_patch[:, None], patch_z, score)
            selected_patch_ids = torch.argmin(score, dim=1)
            selected_patches = patches[torch.arange(num_envs, device=asset.device), selected_patch_ids]

            positions[high_origin] = selected_patches[high_origin] + root_states[high_origin, 0:3]

            direction = env_origins[:, :2] - selected_patches[:, :2]
            approach_yaw = torch.atan2(direction[:, 1], direction[:, 0])
            yaw_noise = math_utils.sample_uniform(
                approach_yaw_noise_range[0],
                approach_yaw_noise_range[1],
                (num_envs,),
                device=asset.device,
            )
            pose_samples[high_origin, 5] = approach_yaw[high_origin] + yaw_noise[high_origin]

    orientations_delta = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    velocity_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_envs, 6), device=asset.device)
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def set_joint_positions_simple(env: ManagerBasedRLEnv, env_ids, joint_pos: dict):
    """
    在 reset 阶段给指定 env_ids 设置 joint_pos，未指定的关节保持不变，所有关节速度清零。
    仅需传入一个参数 joint_pos={name: value}（在 EventTerm.params 里）
    """
    robot = env.scene["robot"]
    device = robot.data.joint_pos.device

    # env_ids 可能是 None、list、torch.Tensor，统一成 1D LongTensor
    if env_ids is None:
        env_ids = torch.arange(robot.data.joint_pos.shape[0], device=device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    # 复制当前状态
    q  = robot.data.joint_pos.clone()  # [N, DoF]
    dq = robot.data.joint_vel.clone()

    name_to_id = {name: i for i, name in enumerate(robot.joint_names)}
    n_sel = env_ids.shape[0]

    for name, val in (joint_pos or {}).items():
        j = name_to_id.get(name, None)
        if j is None:
            # 找不到关节名就跳过（或改成 raise 更严格）
            continue

        if isinstance(val, (int, float)):
            q[env_ids, j] = float(val)
        else:
            v = torch.as_tensor(val, device=device, dtype=q.dtype)
            if v.ndim == 0:
                v = v.repeat(n_sel)
            elif v.shape[0] != n_sel:
                # 广播/截断到 env_ids 数量
                v = v.reshape(-1).repeat(n_sel)[:n_sel]
            q[env_ids, j] = v

    # 速度清零（只清选中的 env，安全些）
    dq[env_ids, :] = 0.0

    # 写回仿真
    robot.write_joint_state_to_sim(q, dq)


def randomize_highstep_foot_under_hip_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    hip_joint_names: list[str],
    thigh_joint_names: list[str],
    calf_joint_names: list[str],
    box_joint_names: list[str],
    hip_outward_range: tuple[float, float] = (0.0, 0.055),
    hip_noise_range: tuple[float, float] = (-0.018, 0.018),
    thigh_position_range: tuple[float, float] = (-0.035, 0.035),
    calf_position_range: tuple[float, float] = (-0.035, 0.035),
    box_position_range: tuple[float, float] = (-0.006, 0.006),
    velocity_range: tuple[float, float] = (-0.12, 0.12),
    rear_inward_prob: float = 0.0,
    rear_inward_range: tuple[float, float] = (0.015, 0.055),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize reset foot placement around the hip-under-foot neighborhood."""

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    if hasattr(asset.data, "joint_names"):
        joint_names = list(asset.data.joint_names)
    else:
        joint_names = list(asset.joint_names)
    name_to_id = {name: idx for idx, name in enumerate(joint_names)}
    expected_names = hip_joint_names + thigh_joint_names + calf_joint_names + box_joint_names
    missing = [name for name in expected_names if name not in name_to_id]
    if missing:
        raise ValueError(f"Missing highstep reset joints: {missing}")

    hip_ids = [name_to_id[name] for name in hip_joint_names]
    thigh_ids = [name_to_id[name] for name in thigh_joint_names]
    calf_ids = [name_to_id[name] for name in calf_joint_names]
    box_ids = [name_to_id[name] for name in box_joint_names]
    selected_ids = sorted(set(hip_ids + thigh_ids + calf_ids + box_ids))
    selected_ids_t = torch.as_tensor(selected_ids, device=device, dtype=torch.long)

    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()
    num_envs = len(env_ids)

    def _sample(value_range: tuple[float, float], width: int) -> torch.Tensor:
        return math_utils.sample_uniform(value_range[0], value_range[1], (num_envs, width), device=device)

    if hip_ids:
        signs = torch.tensor(
            [1.0 if name.startswith(("FL_", "RL_")) else -1.0 for name in hip_joint_names],
            device=device,
            dtype=joint_pos.dtype,
        ).unsqueeze(0)
        outward = _sample(hip_outward_range, len(hip_ids))
        noise = _sample(hip_noise_range, len(hip_ids))
        hip_delta = torch.clamp(outward + noise, min=0.0)
        if rear_inward_prob > 0.0:
            rear_mask = torch.tensor(
                [name.startswith(("RL_", "RR_")) for name in hip_joint_names],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(0)
            inward_env = (
                torch.rand(num_envs, 1, device=device) < min(max(rear_inward_prob, 0.0), 1.0)
            )
            inward = _sample(rear_inward_range, len(hip_ids))
            hip_delta = torch.where(inward_env & rear_mask, -inward, hip_delta)
        joint_pos[:, hip_ids] += signs * hip_delta

    if thigh_ids:
        joint_pos[:, thigh_ids] += _sample(thigh_position_range, len(thigh_ids))
    if calf_ids:
        joint_pos[:, calf_ids] += _sample(calf_position_range, len(calf_ids))
    if box_ids:
        joint_pos[:, box_ids] += _sample(box_position_range, len(box_ids))

    joint_vel[:, selected_ids] += _sample(velocity_range, len(selected_ids))

    iter_env_ids = env_ids[:, None]
    pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, selected_ids_t]
    joint_pos[:, selected_ids] = joint_pos[:, selected_ids].clamp(pos_limits[..., 0], pos_limits[..., 1])
    vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, selected_ids_t]
    joint_vel[:, selected_ids] = joint_vel[:, selected_ids].clamp(-vel_limits, vel_limits)

    asset.write_joint_state_to_sim(
        joint_pos[:, selected_ids],
        joint_vel[:, selected_ids],
        joint_ids=selected_ids_t,
        env_ids=env_ids,
    )


from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
import math
POSTURE_SET = ("flat", "front", "back", "left", "right")
_PRESETS_RP = {
    "flat":  (0.0,  0.0),
    "front": (0.0,  +0.5 * math.pi),
    "back":  (0.0,  -0.5 * math.pi),
    "left":  (-0.5 * math.pi, 0.0),
    "right": (+0.5 * math.pi, 0.0),
}

def sample_posture_ranges(env, env_ids, old_ranges, *, probs=(0.2,)*5, band=1e-3, yaw_band=0.0, force=None):
    """
    返回一个新的 UniformPoseCommandCfg.Ranges：
    - 若 force 给定（'flat'/'front'/...），就用它；否则按 probs 采样一种。
    - 把 roll/pitch 设为 [target - band, target + band] 的窄区间，yaw 也给一个很小窗口。
    """
    import torch
    device = env.device
    if force is None:
        p = torch.tensor(probs, device=device, dtype=torch.float32)
        if not torch.isclose(p.sum(), torch.tensor(1.0, device=device)):
            p = p / p.sum()
        idx = torch.multinomial(p, 1).item()
        name = POSTURE_SET[idx]
    else:
        name = force
    r_tgt, p_tgt = _PRESETS_RP[name]
    return UniformPoseCommandCfg.Ranges(
        pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0),
        roll=(r_tgt - band, r_tgt + band),
        pitch=(p_tgt - band, p_tgt + band),
        yaw=(-yaw_band, +yaw_band),
    )

@torch.no_grad()
def set_discrete_basevel_ranges(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    *,
    heading_value: float = 0.0,   # 固定朝向（弧度），0 面向 +x，pi 面向 -x
    speed_abs: float = 0.4,       # 线速度幅值
    include_zero: bool = True,    # 是否允许 0 速度桶
    term_name: str = "base_velocity",
) -> None:
    """
    在 reset 时，按离散桶为子环境设置“采样范围”，并调用命令项自身的 reset(env_ids) 完成重采样。
    - 不直接写 command tensor，而是通过修改 cfg.ranges 并立刻按子集 reset 来实现离散取值：
        { +speed_abs }, { 0 }, { -speed_abs }
    - 同时固定 heading，禁用侧移与自转。

    参数
    ----
    env : Isaac Lab 的 ManagerBasedRLEnv
    env_ids : 需要应用本事件的子环境 id 列表
    heading_value : 固定期望朝向（rad），例如 0 或 pi
    speed_abs : 线速度幅值（m/s）
    include_zero : 是否包含 0 速度的桶
    term_name : 命令项名称（默认 "base_velocity"）
    """
    if len(env_ids) == 0:
        return

    # 取得命令项（注意：没有 get_term_cfg）
    cmd_mgr = env.command_manager
    term = cmd_mgr.get_term(term_name)  # CommandTerm
    cfg = term.cfg                      # CommandTermCfg（UniformVelocityCommandCfg）

    # 先设置与“对准箱子/固定朝向且不转弯”相关的范围
    cfg.heading_command = True
    cfg.ranges.heading = (float(heading_value), float(heading_value))
    cfg.ranges.lin_vel_y = (0.0, 0.0)
    cfg.ranges.ang_vel_z = (0.0, 0.0)

    # 把 env_ids 打乱并划分桶：前进 / （可选）零速 / 后退
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = int(env_ids_t.numel())
    perm = env_ids_t[torch.randperm(n, device=env.device)]

    if include_zero:
        n_fwd = int(math.ceil(n / 3.0))
        n_zero = int(math.floor(n / 3.0))
        n_back = n - n_fwd - n_zero
    else:
        n_fwd = n // 2
        n_zero = 0
        n_back = n - n_fwd

    idx_fwd = perm[:n_fwd]
    idx_zero = perm[n_fwd:n_fwd + n_zero] if n_zero > 0 else torch.empty(0, dtype=torch.long, device=env.device)
    idx_back = perm[n_fwd + n_zero:]

    # 关键：对不同桶“临时设置 lin_vel_x 的采样范围”，并仅对该桶重采样
    if idx_fwd.numel() > 0:
        cfg.ranges.lin_vel_x = (float(speed_abs), float(speed_abs))  # 退化为常数 → 离散 +speed_abs
        term.reset(idx_fwd.tolist())

    if idx_zero.numel() > 0:
        cfg.ranges.lin_vel_x = (0.0, 0.0)                            # 离散 0
        term.reset(idx_zero.tolist())

    if idx_back.numel() > 0:
        cfg.ranges.lin_vel_x = (-float(speed_abs), -float(speed_abs))  # 离散 -speed_abs
        term.reset(idx_back.tolist())

class randomize_joint_parameters_with_damping(ManagerTermBase):
    """Randomize the simulated joint parameters including independent damping control.

    This term allows independent randomization of:
    1. Friction (Static & Dynamic)
    2. Damping (Viscous Friction) - NEW!
    3. Armature
    4. Joint Limits
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # extract the used quantities
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        
        # 简单的参数校验
        if cfg.params.get("operation") == "scale":
            # 这里只做简单的存在性检查，具体范围检查略去以保持代码简洁
            pass
        elif cfg.params.get("operation") not in ("abs", "add", "scale", None):
             # 注意：None 是为了容错，默认值通常在 __call__ 处理
             pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,  # <--- 新增参数
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # ==================================================================================
        # 1. 处理 摩擦力 (Static/Dynamic) 和 阻尼 (Viscous/Damping)
        # ==================================================================================
        # 只要有任意一个参数需要随机化，我们就需要调用 write_joint_friction_coefficient_to_sim
        if friction_distribution_params is not None or damping_distribution_params is not None:
            
            # --- A. 准备基础数据 (从默认值克隆) ---
            # 我们先获取所有相关的默认值，确保如果不修改某项，它保持原样
            static_friction = self.asset.data.default_joint_friction_coeff.clone()
            viscous_friction = self.asset.data.default_joint_viscous_friction_coeff.clone()
            
            # 检查 Isaac Sim 版本以决定是否处理 Dynamic Friction
            major_version = int(env.sim.get_version()[0])
            if major_version >= 5:
                dynamic_friction = self.asset.data.default_joint_dynamic_friction_coeff.clone()
            else:
                dynamic_friction = None

            # --- B. 随机化 摩擦力 (Static & Dynamic) ---
            if friction_distribution_params is not None:
                # 1. 随机化 Static Friction
                static_friction = _randomize_prop_by_op(
                    static_friction,
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                static_friction = torch.clamp(static_friction, min=0.0)

                # 2. 随机化 Dynamic Friction (仅限 Isaac Sim 5.0+)
                if dynamic_friction is not None:
                    dynamic_friction = _randomize_prop_by_op(
                        dynamic_friction,
                        friction_distribution_params, # 通常动摩擦和静摩擦使用相同的随机分布参数
                        env_ids,
                        joint_ids,
                        operation=operation,
                        distribution=distribution,
                    )
                    dynamic_friction = torch.clamp(dynamic_friction, min=0.0)
                    # 物理约束：动摩擦 <= 静摩擦
                    dynamic_friction = torch.minimum(dynamic_friction, static_friction)

            # --- C. 随机化 阻尼 (Viscous/Damping) ---
            # 这是解耦的关键：使用独立的 damping_distribution_params
            if damping_distribution_params is not None:
                viscous_friction = _randomize_prop_by_op(
                    viscous_friction,
                    damping_distribution_params, # <--- 使用独立的阻尼参数
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                viscous_friction = torch.clamp(viscous_friction, min=0.0)

            # --- D. 提取切片并写入仿真 ---
            # _randomize_prop_by_op 是原地修改全量 tensor 的，但 write 函数通常需要针对 env_ids 的切片
            # 或者我们可以直接把全量传进去，但为了性能和标准做法，我们提取切片
            
            # 注意：如果 joint_ids 是 slice(None)，我们需要处理维度
            # 为了通用性，我们统一提取 [env_ids, joint_ids] 的数据进行写入
            
            s_f_slice = static_friction[env_ids[:, None], joint_ids]
            v_f_slice = viscous_friction[env_ids[:, None], joint_ids]
            d_f_slice = dynamic_friction[env_ids[:, None], joint_ids] if dynamic_friction is not None else None

            self.asset.write_joint_friction_coefficient_to_sim(
                joint_friction_coeff=s_f_slice,
                joint_dynamic_friction_coeff=d_f_slice,
                joint_viscous_friction_coeff=v_f_slice,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        # ==================================================================================
        # 2. 处理 关节惯量 (Armature) - 保持原有逻辑
        # ==================================================================================
        if armature_distribution_params is not None:
            armature = self.asset.data.default_joint_armature.clone()
            armature = _randomize_prop_by_op(
                armature,
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            # 写入
            self.asset.write_joint_armature_to_sim(
                armature[env_ids[:, None], joint_ids], 
                joint_ids=joint_ids, 
                env_ids=env_ids
            )

        # ==================================================================================
        # 3. 处理 关节限位 (Limits) - 保持原有逻辑
        # ==================================================================================
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
            
            if lower_limit_distribution_params is not None:
                # 针对第0维 (lower limit)
                # 注意：_randomize_prop_by_op 处理的是二维数据 (env, joint)，
                # limits 是 (env, joint, 2)。我们需要先提取出来处理，再放回去。
                lower_limits = joint_pos_limits[..., 0]
                lower_limits = _randomize_prop_by_op(
                    lower_limits,
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                joint_pos_limits[..., 0] = lower_limits

            if upper_limit_distribution_params is not None:
                upper_limits = joint_pos_limits[..., 1]
                upper_limits = _randomize_prop_by_op(
                    upper_limits,
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                joint_pos_limits[..., 1] = upper_limits

            # 检查合法性
            limits_slice = joint_pos_limits[env_ids[:, None], joint_ids]
            if (limits_slice[..., 0] > limits_slice[..., 1]).any():
                raise ValueError(
                    "Randomization resulted in lower limits > upper limits."
                )

            self.asset.write_joint_position_limit_to_sim(
                limits_slice, 
                joint_ids=joint_ids, 
                env_ids=env_ids, 
                warn_limit_violation=False
            )
