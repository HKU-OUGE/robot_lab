# ==============================================================================
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by: Tianyang TANG
# ==============================================================================

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
from isaaclab.utils.dict import print_dict
# from isaaclab.managers import SceneEntityCfg

# import json

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--se2_gamepad", action="store_true", default=False, help="Whether to use se2_gamepad.")
parser.add_argument("--debug", action="store_true", default=False, help="Print debug information (env config, action and observation spaces).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for deterministic play/eval comparisons.")
parser.add_argument(
    "--play_terrain_level",
    type=int,
    default=None,
    help="Force env0 to start on a specific terrain row for play. Higher is harder.",
)
parser.add_argument(
    "--play_terrain_type",
    type=str,
    default=None,
    help="Force env0 to start on a named terrain type for play, e.g. box or pyramid_stairs.",
)
parser.add_argument(
    "--disable_action_prior",
    action="store_true",
    default=False,
    help="Replace custom action-prior action terms with plain joint-position actions during play.",
)
parser.add_argument(
    "--allow_legacy_highstep_schedule_fallback",
    action="store_true",
    default=False,
    help=(
        "Explicitly allow a verified legacy highstep checkpoint with no schedule sidecar to use "
        "its checkpoint iteration. New/patched runs fail closed when the sidecar is absent."
    ),
)
parser.add_argument(
    "--highstep_v18_teacher_bootstrap_profile",
    action="store_true",
    default=False,
    help=(
        "Evaluate model_172300 in the hash-frozen 0707 Teacher environment, "
        "anchoring that profile's schedule at the checkpoint clock without importing an incompatible sidecar."
    ),
)
parser.add_argument(
    "--keep_play_randomization",
    action="store_true",
    default=False,
    help="Keep configured play-time reset/force randomization for robustness eval. Default keeps normal deterministic play.",
)
parser.add_argument(
    "--eval_action_delay_steps",
    type=int,
    default=None,
    help="Evaluation only: force a fixed action delay in control steps when the task supports it.",
)
parser.add_argument(
    "--eval_effort_limits",
    type=float,
    nargs=3,
    default=None,
    metavar=("HIP_NM", "THIGH_NM", "CALF_NM"),
    help=(
        "Evaluation only: override the explicit actuator-model effort_limit for hip/thigh/calf "
        "before environment creation. This does not change training, action scale, action clip, "
        "joint-position clip, or DC-motor saturation_effort."
    ),
)
parser.add_argument(
    "--fixed_velocity_command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Override base velocity observations with a fixed [vx, vy, wz] command for headless play.",
)
parser.add_argument(
    "--highstep_gap_camera",
    type=str,
    choices=("none", "rear_top", "top", "side_top"),
    default="none",
    help="Debug-only camera for highstep sim-to-real gap videos. Default keeps the normal play view.",
)
parser.add_argument(
    "--reset_after_play_terrain_selection",
    action="store_true",
    default=False,
    help="Play/eval only: reset after forcing terrain level/type so env0 starts from the selected cell.",
)
parser.add_argument(
    "--front_step_eval_reset",
    action="store_true",
    default=False,
    help="Play/eval only: place envs on a low patch near the selected high-step terrain origin and yaw toward it.",
)
parser.add_argument(
    "--front_step_eval_side",
    type=str,
    choices=("x-", "x+", "y-", "y+"),
    default="x-",
    help="Side used by --front_step_eval_reset. x- starts at lower x and faces +x toward the terrain origin.",
)
parser.add_argument(
    "--front_step_eval_distance",
    type=float,
    default=None,
    help="Legacy explicit distance from terrain center. It must place the robot outside the platform edge.",
)
parser.add_argument(
    "--front_step_eval_edge_gap",
    type=float,
    default=0.55,
    help="Base clearance outside the near platform edge when an explicit center distance is not provided.",
)
parser.add_argument(
    "--front_step_eval_platform_width",
    type=float,
    default=None,
    help="Override the selected terrain platform width. By default it is read from the terrain config.",
)
parser.add_argument(
    "--front_step_eval_lateral_offset",
    type=float,
    default=0.0,
    help="Lateral offset along the selected terrain side for --front_step_eval_reset.",
)
parser.add_argument(
    "--front_step_eval_yaw_offset_deg",
    type=float,
    default=0.0,
    help="Yaw perturbation in degrees relative to a perpendicular approach to the platform edge.",
)
parser.add_argument(
    "--print_rear_width_metrics",
    action="store_true",
    default=False,
    help="Play/eval only: print rear-foot lateral width stats in the robot heading frame.",
)
parser.add_argument(
    "--rear_width_metric_interval",
    type=int,
    default=60,
    help="Frame interval for --print_rear_width_metrics progress logs.",
)
parser.add_argument(
    "--play_max_steps",
    type=int,
    default=None,
    help="Play/eval only: stop after this many environment steps even when not recording video.",
)
parser.add_argument(
    "--skip_policy_export",
    action="store_true",
    default=False,
    help="Skip JIT/ONNX export during repeated automated evaluations.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--moe", action="store_true", default=False, help="Whether to use MoE.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()
if hydra_args:
    print("[INFO] Ignoring Hydra-style overrides in play.py:", hydra_args)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import numpy as np

import isaaclab.utils.math as math_utils
import rsl_rl_utils
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg
import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.locomotion.velocity.mdp.highstep_schedule import (
    SCHEDULE_MANIFEST_NAME,
    assert_schedule_definition_compatible,
    build_schedule_manifest,
    checkpoint_iteration_from_mapping,
    eval_schedule_fields,
    global_update,
    install_global_update,
    load_source_manifest,
    resolve_loaded_schedule_update,
    runtime_schedule_match,
    schedule_clock_state,
    schedule_definition_from_configs,
    write_manifest,
)

# =========================================================================
# 🌟 注册 VAEActorCritic 和 VAEPPO 到 RSL-RL 命名空间
# =========================================================================
try:
    from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.vae_ppo import VAEActorCritic, VAEPPO
    import rsl_rl.modules.actor_critic as _ac
    import rsl_rl.algorithms.ppo as _ppo
    import rsl_rl.runners.on_policy_runner as _opr

    _ac.VAEActorCritic = VAEActorCritic
    _ppo.VAEPPO = VAEPPO
    _opr.VAEActorCritic = VAEActorCritic
    _opr.VAEPPO = VAEPPO
    print("[INFO] Successfully registered VAEActorCritic and VAEPPO to RSL-RL.")
except ImportError as e:
    print(f"[WARN] Could not import VAE classes: {e}")
# =========================================================================

# --- MoE Actor that can drop-in replace the base policy's actor MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.devices import Se2Gamepad
from isaaclab.devices.gamepad.se2_gamepad import Se2GamepadCfg


def _read_rsl_checkpoint_iteration(checkpoint_path: str) -> int:
    """Read the checkpoint clock before the RSL wrapper performs its reset."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as safe_load_error:
        print(
            "[WARN] weights_only checkpoint metadata read failed; retrying the trusted local checkpoint "
            f"with the legacy loader: {safe_load_error}"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        return checkpoint_iteration_from_mapping(checkpoint)
    finally:
        del checkpoint

def _make_mlp(in_dim: int, hidden: list[int], out_dim: int, act: nn.Module):
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


def _terrain_column_for_name(terrain_generator_cfg, terrain_name: str) -> int:
    """Return a deterministic curriculum column for a named sub-terrain."""
    if terrain_generator_cfg is None or not hasattr(terrain_generator_cfg, "sub_terrains"):
        raise ValueError("--play_terrain_type requires a terrain generator with sub_terrains.")
    terrain_name = terrain_name.strip()
    sub_terrains = terrain_generator_cfg.sub_terrains
    if terrain_name not in sub_terrains:
        available = ", ".join(sub_terrains.keys())
        raise ValueError(f"Unknown --play_terrain_type '{terrain_name}'. Available: {available}")

    names = list(sub_terrains.keys())
    proportions = np.array([float(sub_terrains[name].proportion) for name in names], dtype=np.float64)
    proportions /= np.sum(proportions)
    cumulative = np.cumsum(proportions)
    num_cols = int(terrain_generator_cfg.num_cols)
    for col in range(num_cols):
        sub_index = int(np.min(np.where(col / num_cols + 0.001 < cumulative)[0]))
        if names[sub_index] == terrain_name:
            return col
    raise ValueError(f"No generated column found for terrain type '{terrain_name}'.")


def _apply_play_terrain_selection(env, level: int | None, terrain_type: str | None):
    """Move env0 to a selected generated terrain cell before the wrapper reset."""
    if level is None and terrain_type is None:
        return
    unwrapped = env.unwrapped
    terrain = unwrapped.scene.terrain
    if not all(hasattr(terrain, name) for name in ("terrain_origins", "terrain_levels", "terrain_types", "env_origins")):
        print("[WARN] --play_terrain_* requested, but this terrain does not expose curriculum origins.")
        return

    origins = terrain.terrain_origins
    max_level = int(origins.shape[0]) - 1
    max_type = int(origins.shape[1]) - 1
    selected_level = int(level) if level is not None else int(terrain.terrain_levels[0].item())
    selected_level = max(0, min(selected_level, max_level))

    if terrain_type is None:
        selected_type = int(terrain.terrain_types[0].item())
    else:
        selected_type = _terrain_column_for_name(unwrapped.cfg.scene.terrain.terrain_generator, terrain_type)
    selected_type = max(0, min(selected_type, max_type))

    device = terrain.terrain_levels.device
    env_ids = torch.arange(unwrapped.num_envs, device=device)
    terrain.terrain_levels[env_ids] = selected_level
    terrain.terrain_types[env_ids] = selected_type
    terrain.env_origins[env_ids] = origins[selected_level, selected_type]
    print(
        "[INFO] Play terrain selection: "
        f"level={selected_level}/{max_level}, type_col={selected_type}/{max_type}, requested_type={terrain_type}"
    )


def _front_step_side_vectors(side: str, device, dtype):
    if side == "x-":
        approach = torch.tensor([-1.0, 0.0], device=device, dtype=dtype)
    elif side == "x+":
        approach = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    elif side == "y-":
        approach = torch.tensor([0.0, -1.0], device=device, dtype=dtype)
    else:
        approach = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    lateral = torch.tensor([-approach[1], approach[0]], device=device, dtype=dtype)
    return approach, lateral


def _front_step_platform_width(env, terrain_type: str | None, override: float | None) -> float:
    if override is not None:
        width = float(override)
    else:
        generator_cfg = env.unwrapped.cfg.scene.terrain.terrain_generator
        sub_terrains = getattr(generator_cfg, "sub_terrains", {}) if generator_cfg is not None else {}
        terrain_cfg = sub_terrains.get(terrain_type) if terrain_type is not None else None
        width = float(getattr(terrain_cfg, "platform_width", 0.0))
    if width <= 0.0:
        raise ValueError(
            "Unable to determine the front-step platform width. Pass --front_step_eval_platform_width explicitly."
        )
    return width


def _apply_front_step_eval_reset(
    env,
    side: str,
    distance: float | None,
    edge_gap: float,
    platform_width: float,
    lateral_offset: float,
    yaw_offset_deg: float,
    terrain_type: str | None = None,
    flat_patch_key: str = "target",
    low_patch_z_margin: float = 0.04,
):
    """Place robots on a deterministic low patch and yaw them toward the selected high-step origin."""
    unwrapped = env.unwrapped
    asset = unwrapped.scene["robot"]
    terrain = getattr(unwrapped.scene, "terrain", None)
    device = asset.device
    dtype = asset.data.root_pos_w.dtype
    env_ids = torch.arange(unwrapped.num_envs, device=device)

    root_states = asset.data.default_root_state[env_ids].clone()
    env_origins = unwrapped.scene.env_origins[env_ids].clone()
    positions = root_states[:, 0:3] + env_origins

    approach, lateral = _front_step_side_vectors(side, device, dtype)
    half_width = 0.5 * float(platform_width)
    target_distance = half_width + float(edge_gap) if distance is None else float(distance)
    edge_clearance = target_distance - half_width
    if edge_clearance < 0.15:
        raise ValueError(
            "Invalid front-step reset: base would start inside or too close to the platform. "
            f"distance={target_distance:.3f}, platform_half_width={half_width:.3f}, "
            f"edge_clearance={edge_clearance:.3f}."
        )
    target_xy = (
        env_origins[:, :2]
        + approach.unsqueeze(0) * target_distance
        + lateral.unsqueeze(0) * float(lateral_offset)
    )

    selected_patches = None
    selected_patch_distance = None
    if terrain is not None and hasattr(terrain, "flat_patches"):
        flat_patches = terrain.flat_patches.get(flat_patch_key)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if flat_patches is not None and terrain_levels is not None and terrain_types is not None:
            levels = terrain_levels[env_ids]
            types = terrain_types[env_ids]
            patches = flat_patches[levels, types]
            patch_xy = patches[:, :, :2]
            patch_z = patches[:, :, 2]
            origin_xy = env_origins[:, None, :2]
            side_score = torch.sum((patch_xy - origin_xy) * approach.view(1, 1, 2), dim=-1)
            low_patch_mask = patch_z < (env_origins[:, 2:3] - low_patch_z_margin)
            side_patch_mask = side_score > 0.15
            dist_score = torch.norm(patch_xy - target_xy[:, None, :], dim=-1)

            valid_mask = low_patch_mask & side_patch_mask
            if not torch.all(torch.any(valid_mask, dim=1)):
                missing = torch.nonzero(~torch.any(valid_mask, dim=1), as_tuple=False).flatten().tolist()
                raise RuntimeError(
                    "Front-step reset has no verified low patch on the requested approach side "
                    f"for envs={missing}."
                )

            masked_score = torch.where(valid_mask, dist_score, torch.full_like(dist_score, 1.0e6))
            selected_patch_ids = torch.argmin(masked_score, dim=1)
            selected_patches = patches[torch.arange(unwrapped.num_envs, device=device), selected_patch_ids]
            selected_patch_distance = dist_score[
                torch.arange(unwrapped.num_envs, device=device), selected_patch_ids
            ]
            analytic_box_ground = terrain_type in {"box", "box_hard"}
            if analytic_box_ground:
                # MeshBoxTerrain is an exact flat ground plane plus one centered
                # rectangular platform. Flat-patch samples are sparse random
                # witnesses of that plane, so their nearest XY may be far from a
                # perfectly valid requested start. Verify a single low-plane Z,
                # platform exclusion, and terrain bounds, then construct the
                # analytically valid low patch at the requested XY.
                low_z_min = torch.where(valid_mask, patch_z, torch.full_like(patch_z, float("inf"))).amin(dim=1)
                low_z_max = torch.where(valid_mask, patch_z, torch.full_like(patch_z, -float("inf"))).amax(dim=1)
                if not torch.all(torch.abs(low_z_max - low_z_min) <= 1.0e-4):
                    raise RuntimeError("Box terrain low-patch witnesses do not share one ground height.")
                local_target = target_xy - env_origins[:, :2]
                outside_platform = torch.amax(torch.abs(local_target), dim=1) >= half_width + 0.15
                generator_cfg = unwrapped.cfg.scene.terrain.terrain_generator
                terrain_size = torch.tensor(generator_cfg.size, device=device, dtype=dtype)
                inside_terrain = torch.all(torch.abs(local_target) <= 0.5 * terrain_size - 0.20, dim=1)
                if not torch.all(outside_platform & inside_terrain):
                    raise RuntimeError("Analytic box reset target is not on the verified low ground region.")
                selected_patches = selected_patches.clone()
                selected_patches[:, :2] = target_xy
                selected_patch_distance = torch.zeros_like(selected_patch_distance)
            elif not torch.all(selected_patch_distance <= 0.35):
                raise RuntimeError(
                    "Verified low patch is too far from the requested reset XY; "
                    f"max_distance={float(torch.max(selected_patch_distance).item()):.3f} m."
                )
            positions = selected_patches + root_states[:, 0:3]
            positions[:, :2] = target_xy + root_states[:, :2]
        else:
            raise RuntimeError("Front-step reset requires flat patches for a verified low-ground start.")
    else:
        raise RuntimeError("Front-step reset requires terrain flat patches for a verified low-ground start.")

    direction = env_origins[:, :2] - positions[:, :2]
    yaw = torch.atan2(direction[:, 1], direction[:, 0]) + torch.deg2rad(
        torch.full((unwrapped.num_envs,), float(yaw_offset_deg), device=device, dtype=dtype)
    )
    zeros = torch.zeros_like(yaw)
    orientations_delta = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    velocities = torch.zeros_like(root_states[:, 7:13])

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    root_outward = torch.sum((positions[:, :2] - env_origins[:, :2]) * approach.unsqueeze(0), dim=1)
    actual_clearance = root_outward - half_width
    step_height = env_origins[:, 2] - selected_patches[:, 2]
    root_height_above_low = positions[:, 2] - selected_patches[:, 2]
    reset_valid = bool(
        torch.all(actual_clearance >= 0.15).item()
        # The release matrix targets the user's 30--35 cm high-step objective,
        # while the configured box_hard challenge reaches 38 cm.  Keep about
        # 2 cm tolerance for terrain/low-patch discretization, but reject
        # unrelated low/high obstacles.
        and torch.all((step_height >= 0.28) & (step_height <= 0.40)).item()
        and torch.all((root_height_above_low >= 0.25) & (root_height_above_low <= 0.75)).item()
        and torch.all(selected_patch_distance <= 0.35).item()
    )
    unwrapped._front_step_eval_context = {
        "side": side,
        "approach": approach.clone(),
        "lateral": lateral.clone(),
        "origin_xy": env_origins[:, :2].clone(),
        "top_z": env_origins[:, 2].clone(),
        "low_z": selected_patches[:, 2].clone(),
        "platform_width": float(platform_width),
        "half_width": half_width,
        "target_distance": target_distance,
        "edge_clearance": actual_clearance.clone(),
        "low_patch_verified": True,
        "low_patch_source": "analytic_box_ground" if terrain_type in {"box", "box_hard"} else "sampled_flat_patch",
        "low_patch_xy_error": selected_patch_distance.clone(),
        "step_height": step_height.clone(),
        "root_height_above_low": root_height_above_low.clone(),
        "reset_valid": reset_valid,
        "yaw_offset_deg": float(yaw_offset_deg),
    }
    first_pos = positions[0].detach().cpu().tolist()
    first_origin = env_origins[0].detach().cpu().tolist()
    patch_msg = unwrapped._front_step_eval_context["low_patch_source"]
    print(
        "[INFO] Front-step eval reset: "
        f"side={side}, platform_width={platform_width:.2f}, distance={target_distance:.2f}, "
        f"edge_clearance={float(actual_clearance[0].detach().cpu()):.2f}, "
        f"step_height={float(step_height[0].detach().cpu()):.3f}, "
        f"patch_xy_error={float(selected_patch_distance[0].detach().cpu()):.3f}, "
        f"lateral={lateral_offset:.2f}, yaw_offset_deg={yaw_offset_deg:.1f}, valid={reset_valid}, "
        f"mode={patch_msg}, env0_pos={[round(v, 3) for v in first_pos]}, "
        f"env0_origin={[round(v, 3) for v in first_origin]}, yaw={float(yaw[0].detach().cpu()):.3f}",
        flush=True,
    )


def _safe_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


_EVAL_EFFORT_ACTUATOR_GROUPS = {
    "hip": "legs_hip",
    "thigh": "legs_thigh",
    "calf": "legs_calf",
}
_EVAL_EFFORT_EXPECTED_JOINTS = {
    joint_type: tuple(f"{leg}_{joint_type}_joint" for leg in ("FL", "FR", "RL", "RR"))
    for joint_type in _EVAL_EFFORT_ACTUATOR_GROUPS
}
_EVAL_JOINT_VELOCITY_REFERENCE_LIMITS_RAD_S = {
    "hip": 20.0,
    "thigh": 20.0,
    "calf": 15.89,
}


def _normalize_eval_effort_limits(raw_limits) -> dict[str, float] | None:
    """Validate the atomic hip/thigh/calf evaluation override."""
    if raw_limits is None:
        return None
    try:
        values = [float(value) for value in raw_limits]
    except (TypeError, ValueError) as error:
        raise ValueError("--eval_effort_limits requires three numeric values") from error
    if len(values) != len(_EVAL_EFFORT_ACTUATOR_GROUPS):
        raise ValueError("--eval_effort_limits requires exactly HIP_NM THIGH_NM CALF_NM")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("--eval_effort_limits values must be finite and greater than zero")
    return dict(zip(_EVAL_EFFORT_ACTUATOR_GROUPS, values, strict=True))


def _apply_eval_effort_limit_override(env_cfg, raw_limits) -> dict[str, float] | None:
    """Write only DCMotor ``effort_limit`` fields before ``gym.make``.

    The default ``None`` path is a strict no-op.  In particular this helper
    never changes saturation_effort, effort_limit_sim, an action mapping, or a
    training configuration.
    """
    requested = _normalize_eval_effort_limits(raw_limits)
    if requested is None:
        return None

    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        raise ValueError("--eval_effort_limits requires scene.robot.actuators")

    missing = [group for group in _EVAL_EFFORT_ACTUATOR_GROUPS.values() if group not in actuators]
    if missing:
        raise ValueError(f"--eval_effort_limits missing actuator groups: {missing}")
    invalid = [
        group
        for group in _EVAL_EFFORT_ACTUATOR_GROUPS.values()
        if not hasattr(actuators[group], "effort_limit")
        or not hasattr(actuators[group], "saturation_effort")
    ]
    if invalid:
        raise ValueError(
            "--eval_effort_limits requires explicit DC-motor actuator cfg fields "
            f"effort_limit and saturation_effort; invalid groups: {invalid}"
        )
    for joint_type, group in _EVAL_EFFORT_ACTUATOR_GROUPS.items():
        try:
            saturation = float(actuators[group].saturation_effort)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Actuator group {group!r} must have one numeric saturation_effort"
            ) from error
        if not math.isfinite(saturation) or saturation <= 0.0:
            raise ValueError(f"Actuator group {group!r} has invalid saturation_effort={saturation}")
        if requested[joint_type] > saturation + max(1.0e-6, saturation * 1.0e-6):
            raise ValueError(
                f"Requested {joint_type} effort_limit={requested[joint_type]} exceeds "
                f"the unchanged DC-motor saturation_effort={saturation}"
            )
    for joint_type, group in _EVAL_EFFORT_ACTUATOR_GROUPS.items():
        actuator_cfg = actuators[group]
        actuator_cfg.effort_limit = requested[joint_type]

    print(
        "[INFO] Evaluation-only actuator effort_limit override (N-m): "
        + json.dumps(requested, sort_keys=True),
        flush=True,
    )
    return requested


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _highstep_reward_stage_definition(env_cfg) -> dict[str, dict[str, int]]:
    definition: dict[str, dict[str, int]] = {}
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return definition
    for name in dir(rewards_cfg):
        if name.startswith("_"):
            continue
        term_cfg = getattr(rewards_cfg, name, None)
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict) or "stage_start_update" not in params:
            continue
        definition[name] = {
            "stage_start_update": int(params["stage_start_update"]),
            "stage_ramp_updates": int(params.get("stage_ramp_updates", 1)),
            "num_steps_per_update": int(params.get("num_steps_per_update", 24)),
        }
    return dict(sorted(definition.items()))


def _evaluation_code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = {
        "play.py": Path(__file__).resolve(),
        "highstep_schedule.py": root
        / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        "highstep_env_cfg.py": root
        / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped"
        / "Arcdog_adjustable_leg/highstep_env_cfg.py",
        "actions.py": root / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py",
        "rewards.py": root / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py",
        "curriculums.py": root / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _runtime_eval_action_delay(env, requested_delay: int | None) -> dict[str, object]:
    requested = max(0, int(requested_delay or 0))
    action_terms = getattr(env.unwrapped.action_manager, "_terms", {})
    term = action_terms.get("joint_pos") if isinstance(action_terms, dict) else None
    cfg = getattr(term, "cfg", None)
    supported = bool(cfg is not None and hasattr(cfg, "min_action_delay_steps"))
    runtime_value: int | None = None
    runtime_observed = False

    runtime_delays = getattr(env.unwrapped, "_highstep_action_delay_steps", None)
    if isinstance(runtime_delays, torch.Tensor) and runtime_delays.numel() > 0:
        unique = torch.unique(runtime_delays.detach())
        if unique.numel() == 1:
            runtime_value = int(unique.item())
            runtime_observed = True
    elif supported:
        minimum = int(getattr(cfg, "min_action_delay_steps"))
        maximum = int(getattr(cfg, "max_action_delay_steps"))
        if minimum == maximum == 0:
            runtime_value = 0
            runtime_observed = True
    elif requested == 0:
        runtime_value = 0
        runtime_observed = True

    return {
        "eval_action_delay_requested": requested,
        "eval_action_delay_supported": supported,
        "eval_action_delay_runtime_observed": runtime_observed,
        "eval_action_delay_steps_runtime": runtime_value,
        "eval_action_delay_runtime_match": bool(runtime_observed and runtime_value == requested),
    }


class _EvalEffortFeasibilityTracker:
    """Audit an explicit play-only effort-limit experiment and env0 torque use."""

    NEAR_CAP_RATIO = 0.95

    def __init__(self, env, requested: dict[str, float]):
        self.asset = env.unwrapped.scene["robot"]
        self.requested = dict(requested)
        self.runtime_by_joint: dict[str, float] = {}
        self.runtime_range_by_joint: dict[str, list[float]] = {}
        self.runtime_saturation_by_joint: dict[str, float] = {}
        self.runtime_errors: list[str] = []
        self.joint_type_by_name = {
            name: joint_type
            for joint_type, names in _EVAL_EFFORT_EXPECTED_JOINTS.items()
            for name in names
        }
        self.joint_names = [
            name
            for joint_type in _EVAL_EFFORT_ACTUATOR_GROUPS
            for name in _EVAL_EFFORT_EXPECTED_JOINTS[joint_type]
        ]
        asset_joint_names = list(self.asset.joint_names)
        missing_asset_joints = [name for name in self.joint_names if name not in asset_joint_names]
        if missing_asset_joints:
            raise RuntimeError(
                "Cannot audit --eval_effort_limits; robot is missing joints: "
                f"{missing_asset_joints}"
            )
        self.joint_ids = [asset_joint_names.index(name) for name in self.joint_names]

        runtime_actuators = getattr(self.asset, "actuators", {})
        for joint_type, group in _EVAL_EFFORT_ACTUATOR_GROUPS.items():
            actuator = runtime_actuators.get(group) if isinstance(runtime_actuators, dict) else None
            if actuator is None:
                self.runtime_errors.append(f"missing_runtime_actuator:{group}")
                continue
            actuator_joint_names = list(getattr(actuator, "joint_names", []))
            expected_names = list(_EVAL_EFFORT_EXPECTED_JOINTS[joint_type])
            if set(actuator_joint_names) != set(expected_names):
                self.runtime_errors.append(
                    f"runtime_joint_set:{group}:{actuator_joint_names!r}!={expected_names!r}"
                )
                continue
            try:
                effort = torch.as_tensor(getattr(actuator, "effort_limit")).detach()
                if effort.ndim == 0:
                    effort = effort.reshape(1, 1).expand(1, len(actuator_joint_names))
                elif effort.shape[-1] == len(actuator_joint_names):
                    effort = effort.reshape(-1, len(actuator_joint_names))
                elif effort.numel() == 1:
                    effort = effort.reshape(1, 1).expand(1, len(actuator_joint_names))
                else:
                    raise ValueError(f"unexpected effort_limit shape {tuple(effort.shape)}")
                if effort.shape[0] < 1:
                    raise ValueError("empty effort_limit tensor")
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                self.runtime_errors.append(f"runtime_effort_limit:{group}:{error}")
                continue
            try:
                saturation = torch.as_tensor(getattr(actuator, "_saturation_effort")).detach()
                if saturation.ndim == 0:
                    saturation = saturation.reshape(1, 1).expand(1, len(actuator_joint_names))
                elif saturation.shape[-1] == len(actuator_joint_names):
                    saturation = saturation.reshape(-1, len(actuator_joint_names))
                elif saturation.numel() == 1:
                    saturation = saturation.reshape(1, 1).expand(1, len(actuator_joint_names))
                else:
                    raise ValueError(
                        f"unexpected saturation_effort shape {tuple(saturation.shape)}"
                    )
                if saturation.shape[0] < 1:
                    raise ValueError("empty saturation_effort tensor")
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                self.runtime_errors.append(f"runtime_saturation_effort:{group}:{error}")
                continue

            for local_index, name in enumerate(actuator_joint_names):
                column = effort[:, local_index]
                saturation_column = saturation[:, local_index]
                if not bool(torch.all(torch.isfinite(column)).item()):
                    self.runtime_errors.append(f"non_finite_runtime_effort_limit:{name}")
                    continue
                if not bool(torch.all(torch.isfinite(saturation_column)).item()):
                    self.runtime_errors.append(f"non_finite_runtime_saturation_effort:{name}")
                    continue
                lower = float(torch.min(column).detach().cpu())
                upper = float(torch.max(column).detach().cpu())
                saturation_lower = float(torch.min(saturation_column).detach().cpu())
                self.runtime_by_joint[name] = float(column[0].detach().cpu())
                self.runtime_range_by_joint[name] = [lower, upper]
                self.runtime_saturation_by_joint[name] = float(
                    saturation_column[0].detach().cpu()
                )
                tolerance = max(1.0e-6, abs(self.requested[joint_type]) * 1.0e-6)
                if abs(lower - self.requested[joint_type]) > tolerance or abs(
                    upper - self.requested[joint_type]
                ) > tolerance:
                    self.runtime_errors.append(
                        f"runtime_effort_limit_mismatch:{name}:[{lower},{upper}]"
                    )
                if saturation_lower + tolerance < upper:
                    self.runtime_errors.append(
                        f"runtime_effort_limit_above_saturation:{name}:{upper}>{saturation_lower}"
                    )

        self.runtime_observed = len(self.runtime_by_joint) == len(self.joint_names)
        self.runtime_match = bool(self.runtime_observed and not self.runtime_errors)
        if not self.runtime_match:
            print(
                "[WARN] Evaluation effort-limit runtime verification failed: "
                + json.dumps(self.runtime_errors, sort_keys=True),
                flush=True,
            )

        self.sample_steps = 0
        self.valid_sample_steps = 0
        self.invalid_sample_steps = 0
        self.samples_by_joint: dict[str, list[float]] = {name: [] for name in self.joint_names}
        self.joint_velocity_sample_steps = 0
        self.joint_velocity_valid_sample_steps = 0
        self.joint_velocity_invalid_sample_steps = 0
        self.joint_velocity_samples_by_joint: dict[str, list[float]] = {
            name: [] for name in self.joint_names
        }

    def sample_applied_torque(self) -> None:
        """Sample env0 torque and read-only joint velocity after ``env.step``."""
        self.sample_steps += 1
        applied = getattr(self.asset.data, "applied_torque", None)
        if not isinstance(applied, torch.Tensor) or applied.ndim != 2 or applied.shape[0] < 1:
            self.invalid_sample_steps += 1
        else:
            values = applied[0, self.joint_ids].detach()
            if values.numel() != len(self.joint_names) or not bool(
                torch.all(torch.isfinite(values)).item()
            ):
                self.invalid_sample_steps += 1
            else:
                for name, value in zip(self.joint_names, values.cpu().tolist(), strict=True):
                    self.samples_by_joint[name].append(float(value))
                self.valid_sample_steps += 1

        self.joint_velocity_sample_steps += 1
        joint_velocity = getattr(self.asset.data, "joint_vel", None)
        if (
            not isinstance(joint_velocity, torch.Tensor)
            or joint_velocity.ndim != 2
            or joint_velocity.shape[0] < 1
        ):
            self.joint_velocity_invalid_sample_steps += 1
        else:
            velocity_values = joint_velocity[0, self.joint_ids].detach()
            if velocity_values.numel() != len(self.joint_names) or not bool(
                torch.all(torch.isfinite(velocity_values)).item()
            ):
                self.joint_velocity_invalid_sample_steps += 1
            else:
                for name, value in zip(
                    self.joint_names, velocity_values.cpu().tolist(), strict=True
                ):
                    self.joint_velocity_samples_by_joint[name].append(float(value))
                self.joint_velocity_valid_sample_steps += 1

    def _pooled_abs(self, names: list[str] | tuple[str, ...]) -> list[float]:
        return [abs(value) for name in names for value in self.samples_by_joint[name]]

    def _pooled_joint_velocity_abs(
        self, names: list[str] | tuple[str, ...]
    ) -> list[float]:
        return [
            abs(value)
            for name in names
            for value in self.joint_velocity_samples_by_joint[name]
        ]

    def _near_cap_fraction(self, names: list[str] | tuple[str, ...]) -> float | None:
        if not self.runtime_match:
            return None
        total = 0
        near = 0
        for name in names:
            threshold = self.NEAR_CAP_RATIO * self.runtime_by_joint[name]
            values = self.samples_by_joint[name]
            total += len(values)
            near += sum(abs(value) >= threshold for value in values)
        return near / total if total > 0 else None

    def _joint_velocity_exceedance_fraction(
        self, names: list[str] | tuple[str, ...]
    ) -> float | None:
        total = 0
        exceeded = 0
        for name in names:
            reference = _EVAL_JOINT_VELOCITY_REFERENCE_LIMITS_RAD_S[
                self.joint_type_by_name[name]
            ]
            values = self.joint_velocity_samples_by_joint[name]
            total += len(values)
            exceeded += sum(abs(value) > reference for value in values)
        return exceeded / total if total > 0 else None

    def summary(self) -> dict[str, object]:
        q95_by_joint = {
            name: _safe_quantile([abs(value) for value in self.samples_by_joint[name]], 0.95)
            for name in self.joint_names
        }
        max_by_joint = {
            name: (max((abs(value) for value in self.samples_by_joint[name]), default=None))
            for name in self.joint_names
        }
        near_by_joint = {
            name: self._near_cap_fraction([name])
            for name in self.joint_names
        }
        rear_names = [
            name
            for name in self.joint_names
            if name.startswith("RL_") or name.startswith("RR_")
        ]
        rear_by_type = {
            joint_type: [f"RL_{joint_type}_joint", f"RR_{joint_type}_joint"]
            for joint_type in _EVAL_EFFORT_ACTUATOR_GROUPS
        }
        all_by_type = {
            joint_type: list(_EVAL_EFFORT_EXPECTED_JOINTS[joint_type])
            for joint_type in _EVAL_EFFORT_ACTUATOR_GROUPS
        }
        joint_velocity_q95_by_joint = {
            name: _safe_quantile(
                [abs(value) for value in self.joint_velocity_samples_by_joint[name]],
                0.95,
            )
            for name in self.joint_names
        }
        joint_velocity_max_by_joint = {
            name: max(
                (abs(value) for value in self.joint_velocity_samples_by_joint[name]),
                default=None,
            )
            for name in self.joint_names
        }
        joint_velocity_exceedance_by_joint = {
            name: self._joint_velocity_exceedance_fraction([name])
            for name in self.joint_names
        }
        return {
            "eval_effort_limit_override_requested": True,
            "eval_effort_limit_requested_nm": self.requested,
            "eval_effort_limit_modified_cfg_fields": {
                joint_type: f"scene.robot.actuators.{group}.effort_limit"
                for joint_type, group in _EVAL_EFFORT_ACTUATOR_GROUPS.items()
            },
            "eval_effort_limit_runtime_source": (
                "env.unwrapped.scene['robot'].actuators.<group>.effort_limit"
            ),
            "eval_effort_limit_runtime_saturation_source": (
                "env.unwrapped.scene['robot'].actuators.<group>._saturation_effort"
            ),
            "eval_effort_limit_runtime_observed": self.runtime_observed,
            "eval_effort_limit_runtime_match": self.runtime_match,
            "eval_effort_limit_runtime_errors": self.runtime_errors,
            "eval_effort_limit_runtime_nm_by_joint": self.runtime_by_joint,
            "eval_effort_limit_runtime_range_nm_by_joint": self.runtime_range_by_joint,
            "eval_effort_limit_runtime_saturation_nm_by_joint": (
                self.runtime_saturation_by_joint
            ),
            "applied_torque_sample_env_index": 0,
            "applied_torque_sample_phase": "post_env_step_last_physics_substep",
            "applied_torque_sample_steps": self.sample_steps,
            "applied_torque_valid_sample_steps": self.valid_sample_steps,
            "applied_torque_invalid_sample_steps": self.invalid_sample_steps,
            "applied_torque_near_effort_limit_ratio": self.NEAR_CAP_RATIO,
            "applied_torque_near_effort_limit_semantics": (
                "abs(applied_torque) >= ratio * verified continuous effort_limit; "
                "not the velocity-dependent instantaneous DC-motor cap"
            ),
            "applied_torque_abs_q95_nm_by_joint": q95_by_joint,
            "applied_torque_abs_max_nm_by_joint": max_by_joint,
            "applied_torque_near_effort_limit_fraction_by_joint": near_by_joint,
            "rear_applied_torque_abs_q95_nm": _safe_quantile(self._pooled_abs(rear_names), 0.95),
            "rear_applied_torque_abs_max_nm": max(self._pooled_abs(rear_names), default=None),
            "rear_applied_torque_near_effort_limit_fraction": self._near_cap_fraction(rear_names),
            "rear_applied_torque_abs_q95_nm_by_type": {
                joint_type: _safe_quantile(self._pooled_abs(names), 0.95)
                for joint_type, names in rear_by_type.items()
            },
            "rear_applied_torque_abs_max_nm_by_type": {
                joint_type: max(self._pooled_abs(names), default=None)
                for joint_type, names in rear_by_type.items()
            },
            "rear_applied_torque_near_effort_limit_fraction_by_type": {
                joint_type: self._near_cap_fraction(names)
                for joint_type, names in rear_by_type.items()
            },
            "joint_velocity_cap_override_requested": False,
            "joint_velocity_reference_limits_rad_s": (
                _EVAL_JOINT_VELOCITY_REFERENCE_LIMITS_RAD_S
            ),
            "joint_velocity_reference_limits_are_diagnostic_only": True,
            "joint_velocity_sample_env_index": 0,
            "joint_velocity_sample_phase": "post_env_step",
            "joint_velocity_sample_steps": self.joint_velocity_sample_steps,
            "joint_velocity_valid_sample_steps": self.joint_velocity_valid_sample_steps,
            "joint_velocity_invalid_sample_steps": self.joint_velocity_invalid_sample_steps,
            "joint_velocity_abs_q95_rad_s_by_joint": joint_velocity_q95_by_joint,
            "joint_velocity_abs_max_rad_s_by_joint": joint_velocity_max_by_joint,
            "joint_velocity_reference_exceedance_fraction_by_joint": (
                joint_velocity_exceedance_by_joint
            ),
            "joint_velocity_abs_q95_rad_s_by_type": {
                joint_type: _safe_quantile(
                    self._pooled_joint_velocity_abs(names), 0.95
                )
                for joint_type, names in all_by_type.items()
            },
            "joint_velocity_abs_max_rad_s_by_type": {
                joint_type: max(self._pooled_joint_velocity_abs(names), default=None)
                for joint_type, names in all_by_type.items()
            },
            "joint_velocity_reference_exceedance_fraction_by_type": {
                joint_type: self._joint_velocity_exceedance_fraction(names)
                for joint_type, names in all_by_type.items()
            },
            "rear_joint_velocity_abs_q95_rad_s_by_type": {
                joint_type: _safe_quantile(
                    self._pooled_joint_velocity_abs(names), 0.95
                )
                for joint_type, names in rear_by_type.items()
            },
            "rear_joint_velocity_abs_max_rad_s_by_type": {
                joint_type: max(self._pooled_joint_velocity_abs(names), default=None)
                for joint_type, names in rear_by_type.items()
            },
            "rear_joint_velocity_reference_exceedance_fraction_by_type": {
                joint_type: self._joint_velocity_exceedance_fraction(names)
                for joint_type, names in rear_by_type.items()
            },
        }


class _HighstepEvalTracker:
    """Track the real-test failure sequence as monotonic high-step phases."""

    PHASES = (
        "approach",
        "front_lift",
        "front_edge_contact",
        "front_top_support",
        "rear_clear",
        "top_hold",
    )
    EXPECTED_ACTION_JOINT_ORDER = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
    ]

    def __init__(self, env):
        unwrapped = env.unwrapped
        self.env = unwrapped
        self.asset = unwrapped.scene["robot"]
        self.contact_sensor = unwrapped.scene.sensors["contact_forces"]
        self.context = getattr(unwrapped, "_front_step_eval_context", None)
        self.front_asset_ids = [int(index) for index in self.asset.find_bodies(["FL_foot", "FR_foot"])[0]]
        self.rear_asset_ids = [int(index) for index in self.asset.find_bodies(["RL_foot", "RR_foot"])[0]]
        self.all_asset_ids = self.front_asset_ids + self.rear_asset_ids
        self.front_contact_ids = [
            int(index) for index in self.contact_sensor.find_bodies(["FL_foot", "FR_foot"])[0]
        ]
        self.rear_contact_ids = [
            int(index) for index in self.contact_sensor.find_bodies(["RL_foot", "RR_foot"])[0]
        ]
        self.all_contact_ids = self.front_contact_ids + self.rear_contact_ids
        self.dt = float(unwrapped.step_dt)
        self.phase = 0
        self.phase_samples: dict[str, list[dict[str, float]]] = {name: [] for name in self.PHASES}
        self.critical_samples: list[dict[str, float]] = []
        self.pre_support_samples: list[dict[str, float]] = []
        self.front_lift_streak = 0
        self.front_edge_contact_streak = 0
        self.front_support_streak = 0
        self.rear_clear_streak = 0
        self.top_stable_streak = 0
        self.top_stable_max = 0
        self.front_single_contact_streak = 0
        self.front_single_contact_max = 0
        self.rear_edge_streak = 0
        self.rear_edge_max = 0
        self.contact_confirm_steps = 2
        self.front_support_grace_steps = 8
        self.rear_advance_margin = 0.18
        self.rear_advance_confirm_steps = 3
        self.front_top_contact_streaks = [0, 0]
        self.rear_top_contact_streaks = [0, 0]
        self.front_bilateral_top_streak = 0
        self.rear_bilateral_top_streak = 0
        # Primary real-climb success intentionally depends only on the body and
        # the two rear feet.  A lifted front foot remains useful diagnostics,
        # but does not remove the operator's fixed-stand handoff window.
        self.rear_on_platform_hold_streak = 0
        self.rear_on_platform_hold_max = 0
        self.rear_on_platform_hold_step: int | None = None
        self.rear_on_platform_hold_confirm_steps = 25
        self.rear_on_platform_min_root_edge_margin = 0.20
        self.rear_on_platform_min_rear_edge_margin = 0.04
        self.rear_on_platform_min_root_h_top = 0.28
        self.rear_on_platform_min_rear_width = 0.18
        self.rear_on_platform_min_rear_abs_y = 0.04
        self.rear_on_platform_max_pitch_metric = 0.42
        self.rear_on_platform_max_roll_metric = 0.32
        self.rear_on_platform_hold_samples: list[dict[str, float]] = []
        self.front_first_step_contact_steps: list[int | None] = [None, None]
        self.front_first_contact_steps: list[int | None] = [None, None]
        self.rear_first_contact_steps: list[int | None] = [None, None]
        self.front_bilateral_top_step: int | None = None
        self.first_rear_top_step: int | None = None
        self.second_rear_top_step: int | None = None
        self.rear_advance_step: int | None = None
        self.rear_advance_streak = 0
        self.front_support_after_grace_samples: list[dict[str, float]] = []
        self.front_top_slip_after_grace: list[float] = []
        self.rear_support_after_second_samples: list[dict[str, float]] = []
        self.rear_top_slip_after_second: list[float] = []
        self.front_impact_speed_max = 0.0
        self.front_impact_normal_speed_max = 0.0
        self.terminated_early = False
        self.termination_step: int | None = None
        self.first_root_outward: float | None = None
        self.last_root_outward: float | None = None
        self.last_root_h_top: float | None = None
        self.root_h_top_max = float("-inf")
        self.initial_geometry_checked = False
        self.initial_geometry_valid = False
        self.initial_foot_height_error_max: float | None = None
        self.initial_no_top_contact = False
        self.initial_all_feet_outside_platform = False

        action_terms = getattr(unwrapped.action_manager, "_terms", {})
        self.target_action_term = action_terms.get("joint_pos") if isinstance(action_terms, dict) else None
        target_clip = getattr(self.target_action_term, "_clip", None)
        self.target_joint_names = list(getattr(self.target_action_term, "_joint_names", []))
        target_dim = len(self.target_joint_names)
        raw_joint_ids = getattr(self.target_action_term, "_joint_ids", None)
        if isinstance(raw_joint_ids, slice):
            self.target_joint_ids = list(range(target_dim))[raw_joint_ids]
        elif isinstance(raw_joint_ids, torch.Tensor):
            self.target_joint_ids = [int(value) for value in raw_joint_ids.detach().cpu().flatten().tolist()]
        elif raw_joint_ids is not None:
            self.target_joint_ids = [int(value) for value in raw_joint_ids]
        else:
            self.target_joint_ids = []
        self.target_affine_scale = None
        self.target_affine_offset = None
        affine_valid = False
        if isinstance(target_clip, torch.Tensor) and target_dim > 0:
            try:
                affine_values = []
                for raw_value in (
                    getattr(self.target_action_term, "_scale", None),
                    getattr(self.target_action_term, "_offset", None),
                ):
                    value = torch.as_tensor(raw_value, device=target_clip.device, dtype=target_clip.dtype)
                    if value.ndim > 1:
                        value = value[0]
                    value = value.reshape(-1)
                    if value.numel() == 1:
                        value = value.expand(target_dim)
                    if value.numel() != target_dim or not bool(torch.all(torch.isfinite(value)).item()):
                        raise ValueError("invalid affine action mapping")
                    affine_values.append(value)
                self.target_affine_scale, self.target_affine_offset = affine_values
                affine_valid = True
            except (TypeError, ValueError, RuntimeError):
                affine_valid = False
        self.target_limit_contract_valid = bool(
            isinstance(target_clip, torch.Tensor)
            and target_clip.ndim == 3
            and target_clip.shape[0] >= 1
            and target_clip.shape[-1] == 2
            and target_clip.shape[1] == len(self.target_joint_names)
            and bool(torch.all(torch.isfinite(target_clip[0])).item())
            and bool(torch.all(target_clip[0, :, 1] > target_clip[0, :, 0]).item())
            and affine_valid
        )
        self.target_action_order_valid = bool(
            self.target_joint_names == self.EXPECTED_ACTION_JOINT_ORDER
            and self.target_joint_ids == list(range(len(self.EXPECTED_ACTION_JOINT_ORDER)))
        )
        self.target_limit_sample_steps = 0
        self.target_limit_joint_samples = 0
        self.target_limit_violation_count = 0
        self.target_limit_step_violation_count = 0
        self.target_limit_violation_streak = 0
        self.target_limit_violation_max_consecutive = 0
        self.target_limit_max_delta = 0.0
        self.target_limit_action_input_valid = True
        self.target_limit_invalid_action_steps = 0
        self.target_limit_joint_violation_counts = [0 for _ in range(target_dim)]
        self.target_limit_joint_max_deltas = [0.0 for _ in range(target_dim)]

    def mark_terminated(self, step: int) -> None:
        self.terminated_early = True
        self.termination_step = int(step)

    def _set_phase(self, phase: int, step: int) -> None:
        if phase <= self.phase:
            return
        self.phase = phase
        print(f"[HIGHSTEP_PHASE] step={step} phase={self.PHASES[self.phase]}", flush=True)

    def _contact_force_vectors(self) -> torch.Tensor:
        data = self.contact_sensor.data
        if hasattr(data, "net_forces_w") and data.net_forces_w is not None:
            return data.net_forces_w[0, self.all_contact_ids, :]
        # Older sensor builds may expose only history.  Index zero is the latest
        # sample; taking a history maximum would combine stale force with the
        # current foot position and can fabricate a top-contact event.
        return data.net_forces_w_history[0, 0, self.all_contact_ids, :]

    @staticmethod
    def _env0_value(value):
        if isinstance(value, torch.Tensor) and value.ndim > 1:
            return value[0]
        return value

    def audit_policy_actions(self, actions: torch.Tensor, step: int) -> None:
        """Audit the wrapper-clipped action before the action term clamps its target.

        This is intentionally called before ``env.step`` so terminating actions
        are covered and the metric cannot accidentally inspect the previous
        action stored by Isaac Lab's action manager.
        """
        self.target_limit_sample_steps += 1
        if not self.target_limit_contract_valid or self.target_action_term is None:
            return

        raw = actions[0] if actions.ndim > 1 else actions
        expected_dim = len(self.target_joint_names)
        valid_input = bool(
            raw.ndim == 1
            and raw.numel() == expected_dim
            and bool(torch.all(torch.isfinite(raw)).item())
        )
        if not valid_input:
            self.target_limit_action_input_valid = False
            self.target_limit_invalid_action_steps += 1
            # Fail closed while keeping JSON metrics finite and comparable.
            delta = torch.full(
                (expected_dim,),
                1.0e6,
                device=self.target_action_term._clip.device,
                dtype=self.target_action_term._clip.dtype,
            )
        else:
            raw = raw.to(
                device=self.target_action_term._clip.device,
                dtype=self.target_action_term._clip.dtype,
            )
            target = raw * self.target_affine_scale + self.target_affine_offset
            clip = self.target_action_term._clip[0]
            lower = clip[:, 0]
            upper = clip[:, 1]
            delta = torch.maximum(lower - target, target - upper).clamp(min=0.0)
            if not bool(torch.all(torch.isfinite(target)).item()) or not bool(
                torch.all(torch.isfinite(delta)).item()
            ):
                self.target_limit_action_input_valid = False
                self.target_limit_invalid_action_steps += 1
                delta = torch.full_like(delta, 1.0e6)

        violation = delta > 1.0e-6
        violation_count = int(torch.sum(violation).item())
        step_violated = violation_count > 0
        max_delta = float(torch.max(delta).detach().cpu())

        self.target_limit_joint_samples += int(delta.numel())
        self.target_limit_violation_count += violation_count
        self.target_limit_step_violation_count += int(step_violated)
        self.target_limit_violation_streak = self.target_limit_violation_streak + 1 if step_violated else 0
        self.target_limit_violation_max_consecutive = max(
            self.target_limit_violation_max_consecutive,
            self.target_limit_violation_streak,
        )
        self.target_limit_max_delta = max(self.target_limit_max_delta, max_delta)
        for index in range(int(delta.numel())):
            self.target_limit_joint_violation_counts[index] += int(bool(violation[index].item()))
            self.target_limit_joint_max_deltas[index] = max(
                self.target_limit_joint_max_deltas[index],
                float(delta[index].detach().cpu()),
            )

    def update(self, step: int) -> None:
        if self.context is None:
            return

        foot_pos = self.asset.data.body_pos_w[0, self.all_asset_ids, :]
        foot_vel = self.asset.data.body_lin_vel_w[0, self.all_asset_ids, :]
        contact_force_vectors = self._contact_force_vectors()
        contact_force = torch.linalg.norm(contact_force_vectors, dim=-1)
        upward_force = torch.clamp(contact_force_vectors[:, 2], min=0.0)
        upward_ratio = upward_force / torch.clamp(contact_force, min=1.0e-6)
        contact = contact_force > 5.0
        top_load_contact = (upward_force > 5.0) & (upward_ratio >= 0.45)
        approach = self.context["approach"]
        lateral = self.context["lateral"]
        origin_xy = self.context["origin_xy"][0]
        top_z = self.context["top_z"][0]
        low_z = self.context["low_z"][0]
        half_width = float(self.context["half_width"])

        rel_xy = foot_pos[:, :2] - origin_xy.unsqueeze(0)
        outward = torch.sum(rel_xy * approach.unsqueeze(0), dim=1)
        lateral_pos = torch.sum(rel_xy * lateral.unsqueeze(0), dim=1)
        inside_top = (torch.abs(outward) <= half_width + 0.04) & (torch.abs(lateral_pos) <= half_width + 0.04)
        top_height = (foot_pos[:, 2] >= top_z - 0.04) & (foot_pos[:, 2] <= top_z + 0.09)
        top_contact = top_load_contact & inside_top & top_height
        front_top_contact = top_contact[:2]
        rear_top_contact = top_contact[2:]
        front_wall_contact = (
            contact[:2]
            & (torch.abs(outward[:2] - half_width) <= 0.14)
            & (foot_pos[:2, 2] > low_z + 0.04)
            & (foot_pos[:2, 2] < top_z - 0.07)
        )

        if not self.initial_geometry_checked:
            foot_height_error = torch.abs(foot_pos[:, 2] - low_z)
            all_feet_outside = bool(torch.all(outward >= half_width + 0.02).item())
            self.initial_foot_height_error_max = float(torch.max(foot_height_error).detach().cpu())
            self.initial_no_top_contact = not bool(torch.any(top_contact).item())
            self.initial_all_feet_outside_platform = all_feet_outside
            self.initial_geometry_valid = bool(
                self.initial_foot_height_error_max <= 0.16
                and self.initial_no_top_contact
                and all_feet_outside
            )
            self.initial_geometry_checked = True

        rear_pos = foot_pos[2:]
        base_pos = self.asset.data.root_pos_w[0]
        heading_quat = math_utils.yaw_quat(self.asset.data.root_quat_w[0].unsqueeze(0))[0]
        rear_pos_b = math_utils.quat_apply_inverse(
            heading_quat.unsqueeze(0).repeat(2, 1), rear_pos - base_pos.unsqueeze(0)
        )
        rear_y = rear_pos_b[:, 1]
        rear_width = float(torch.abs(rear_y[0] - rear_y[1]).detach().cpu())
        rear_min_abs_y = float(torch.min(torch.abs(rear_y)).detach().cpu())

        root_rel_xy = base_pos[:2] - origin_xy
        root_outward = float(torch.sum(root_rel_xy * approach).detach().cpu())
        root_edge_margin = half_width - root_outward
        rear_edge_margin = float((half_width - torch.max(outward[2:])).detach().cpu())
        all_feet_edge_margin = float((half_width - torch.max(outward)).detach().cpu())
        root_h_top = float((base_pos[2] - top_z).detach().cpu())
        self.first_root_outward = root_outward if self.first_root_outward is None else self.first_root_outward
        self.last_root_outward = root_outward
        self.last_root_h_top = root_h_top
        self.root_h_top_max = max(self.root_h_top_max, root_h_top)

        projected_gravity = self.asset.data.projected_gravity_b[0]
        root_ang_vel = self.asset.data.root_ang_vel_b[0]
        forward_w = math_utils.quat_apply(
            heading_quat.unsqueeze(0),
            torch.tensor([[1.0, 0.0, 0.0]], device=foot_pos.device, dtype=foot_pos.dtype),
        )[0, :2]
        target_forward = -approach
        heading_cos = torch.clamp(torch.sum(forward_w * target_forward), -1.0, 1.0)
        heading_error = float(torch.acos(heading_cos).detach().cpu())

        front_lifted = bool(torch.any(foot_pos[:2, 2] > low_z + 0.08).item())
        front_step_contact = front_top_contact | front_wall_contact
        any_front_step_contact = bool(torch.any(front_step_contact).item())
        any_front_top = bool(torch.any(front_top_contact).item())
        any_rear_top = bool(torch.any(rear_top_contact).item())
        all_feet_inside = bool(torch.all(inside_top).item())
        stable_top = (
            all_feet_inside
            and all_feet_edge_margin >= 0.02
            and bool(torch.all(foot_pos[:, 2] >= top_z - 0.10).item())
            and int(torch.sum(contact).item()) >= 2
            and root_h_top >= 0.28
            and abs(float(projected_gravity[0].detach().cpu())) <= 0.42
            and abs(float(projected_gravity[1].detach().cpu())) <= 0.32
        )

        self.front_lift_streak = self.front_lift_streak + 1 if front_lifted else 0
        self.front_edge_contact_streak = self.front_edge_contact_streak + 1 if any_front_step_contact else 0
        self.front_support_streak = self.front_support_streak + 1 if any_front_top else 0
        self.rear_clear_streak = self.rear_clear_streak + 1 if any_rear_top else 0
        self.top_stable_streak = self.top_stable_streak + 1 if stable_top else 0
        self.top_stable_max = max(self.top_stable_max, self.top_stable_streak)
        if self.front_lift_streak >= 2:
            self._set_phase(1, step)
        if self.front_edge_contact_streak >= 2:
            self._set_phase(2, step)
        if self.front_support_streak >= 2:
            self._set_phase(3, step)
        if self.rear_clear_streak >= 2:
            self._set_phase(4, step)
        if self.top_stable_streak >= 10:
            self._set_phase(5, step)

        for index in range(2):
            if bool(front_step_contact[index].item()) and self.front_first_step_contact_steps[index] is None:
                self.front_first_step_contact_steps[index] = int(step)
                impact_speed = abs(float(foot_vel[index, 2].detach().cpu()))
                self.front_impact_speed_max = max(self.front_impact_speed_max, impact_speed)
                normal_speed = max(
                    0.0,
                    -float(torch.sum(foot_vel[index, :2] * approach).detach().cpu()),
                )
                self.front_impact_normal_speed_max = max(self.front_impact_normal_speed_max, normal_speed)
            self.front_top_contact_streaks[index] = (
                self.front_top_contact_streaks[index] + 1 if bool(front_top_contact[index].item()) else 0
            )
            self.rear_top_contact_streaks[index] = (
                self.rear_top_contact_streaks[index] + 1 if bool(rear_top_contact[index].item()) else 0
            )
            if (
                self.front_top_contact_streaks[index] >= self.contact_confirm_steps
                and self.front_first_contact_steps[index] is None
            ):
                self.front_first_contact_steps[index] = int(step - self.contact_confirm_steps + 1)
            if (
                self.rear_top_contact_streaks[index] >= self.contact_confirm_steps
                and self.rear_first_contact_steps[index] is None
            ):
                self.rear_first_contact_steps[index] = int(step - self.contact_confirm_steps + 1)

        front_bilateral_top = bool(torch.all(front_top_contact).item())
        rear_bilateral_top = bool(torch.all(rear_top_contact).item())
        self.front_bilateral_top_streak = (
            self.front_bilateral_top_streak + 1 if front_bilateral_top else 0
        )
        self.rear_bilateral_top_streak = (
            self.rear_bilateral_top_streak + 1 if rear_bilateral_top else 0
        )
        if (
            self.front_bilateral_top_step is None
            and self.front_bilateral_top_streak >= self.contact_confirm_steps
        ):
            self.front_bilateral_top_step = int(step - self.contact_confirm_steps + 1)
        rear_contact_steps = [int(value) for value in self.rear_first_contact_steps if value is not None]
        if self.first_rear_top_step is None and rear_contact_steps:
            self.first_rear_top_step = min(rear_contact_steps)
        if (
            self.second_rear_top_step is None
            and self.rear_bilateral_top_streak >= self.contact_confirm_steps
        ):
            self.second_rear_top_step = int(step - self.contact_confirm_steps + 1)

        rear_on_platform = bool(
            self.second_rear_top_step is not None
            and rear_bilateral_top
            and rear_edge_margin >= self.rear_on_platform_min_rear_edge_margin
            and root_edge_margin >= self.rear_on_platform_min_root_edge_margin
            and root_h_top >= self.rear_on_platform_min_root_h_top
            and rear_width >= self.rear_on_platform_min_rear_width
            and rear_min_abs_y >= self.rear_on_platform_min_rear_abs_y
            and abs(float(projected_gravity[0].detach().cpu()))
            <= self.rear_on_platform_max_pitch_metric
            and abs(float(projected_gravity[1].detach().cpu()))
            <= self.rear_on_platform_max_roll_metric
        )
        self.rear_on_platform_hold_streak = (
            self.rear_on_platform_hold_streak + 1 if rear_on_platform else 0
        )
        self.rear_on_platform_hold_max = max(
            self.rear_on_platform_hold_max,
            self.rear_on_platform_hold_streak,
        )
        if (
            self.rear_on_platform_hold_step is None
            and self.rear_on_platform_hold_streak >= self.rear_on_platform_hold_confirm_steps
        ):
            self.rear_on_platform_hold_step = int(
                step - self.rear_on_platform_hold_confirm_steps + 1
            )

        rear_advanced = bool(
            self.second_rear_top_step is not None
            and torch.all(inside_top[2:] & top_height[2:]).item()
            and torch.all(rear_top_contact).item()
            and rear_edge_margin >= self.rear_advance_margin
        )
        self.rear_advance_streak = self.rear_advance_streak + 1 if rear_advanced else 0
        if (
            self.rear_advance_step is None
            and self.rear_advance_streak >= self.rear_advance_confirm_steps
        ):
            self.rear_advance_step = int(step - self.rear_advance_confirm_steps + 1)

        front_single = bool(
            self.phase <= 3
            and torch.any(front_step_contact).item()
            and not torch.all(front_step_contact).item()
        )
        self.front_single_contact_streak = self.front_single_contact_streak + 1 if front_single else 0
        self.front_single_contact_max = max(self.front_single_contact_max, self.front_single_contact_streak)
        rear_edge_contact = bool(
            torch.any(contact[2:] & (torch.abs(outward[2:] - half_width) <= 0.12)).item()
        ) and self.phase >= 3
        self.rear_edge_streak = self.rear_edge_streak + 1 if rear_edge_contact else 0
        self.rear_edge_max = max(self.rear_edge_max, self.rear_edge_streak)

        rear_contact = contact[2:]
        rear_slip = torch.linalg.norm(foot_vel[2:, :2], dim=1)
        rear_slip_contact = float(
            torch.max(torch.where(rear_contact, rear_slip, torch.zeros_like(rear_slip))).detach().cpu()
        )
        front_slip = torch.linalg.norm(foot_vel[:2, :2], dim=1)
        front_slip_top = float(
            torch.max(torch.where(front_top_contact, front_slip, torch.zeros_like(front_slip))).detach().cpu()
        )
        rear_slip_top = float(
            torch.max(torch.where(rear_top_contact, rear_slip, torch.zeros_like(rear_slip))).detach().cpu()
        )
        sample = {
            "rear_width": rear_width,
            "rear_min_abs_y": rear_min_abs_y,
            "rear_bilateral_contact": float(torch.all(rear_contact).item()),
            "rear_slip_contact": rear_slip_contact,
            "rear_bilateral_top_contact": float(torch.all(rear_top_contact).item()),
            "rear_slip_top": rear_slip_top,
            "front_bilateral_top_contact": float(torch.all(front_top_contact).item()),
            "front_single_step_contact": float(front_single),
            "front_wall_contact": float(torch.any(front_wall_contact).item()),
            "front_slip_top": front_slip_top,
            "roll_metric": abs(float(projected_gravity[1].detach().cpu())),
            "pitch_metric": abs(float(projected_gravity[0].detach().cpu())),
            "roll_rate": abs(float(root_ang_vel[0].detach().cpu())),
            "yaw_rate": abs(float(root_ang_vel[2].detach().cpu())),
            "heading_error": heading_error,
            "root_outward": root_outward,
            "root_edge_margin": root_edge_margin,
            "rear_edge_margin": rear_edge_margin,
            "all_feet_edge_margin": all_feet_edge_margin,
            "root_h_top": root_h_top,
        }
        self.phase_samples[self.PHASES[self.phase]].append(sample)
        if self.phase < 3:
            self.pre_support_samples.append(sample)
        if 1 <= self.phase <= 4:
            self.critical_samples.append(sample)
        if (
            self.front_bilateral_top_step is not None
            and step >= self.front_bilateral_top_step + self.front_support_grace_steps
            and (self.rear_advance_step is None or step <= self.rear_advance_step)
        ):
            self.front_support_after_grace_samples.append(sample)
            if bool(torch.any(front_top_contact).item()):
                self.front_top_slip_after_grace.append(front_slip_top)
        if self.second_rear_top_step is not None and step >= self.second_rear_top_step:
            self.rear_support_after_second_samples.append(sample)
            if bool(torch.any(rear_top_contact).item()):
                self.rear_top_slip_after_second.append(rear_slip_top)
        if rear_on_platform:
            self.rear_on_platform_hold_samples.append(sample)

    @staticmethod
    def _contact_lag(first_steps: list[int | None]) -> int | None:
        if any(step is None for step in first_steps):
            return None
        return int(abs(int(first_steps[0]) - int(first_steps[1])))

    @staticmethod
    def _ordered_dwell(start_step: int | None, end_step: int | None) -> int | None:
        if start_step is None or end_step is None or end_step < start_step:
            return None
        return int(end_step - start_step)

    def summary(self) -> dict[str, object]:
        approach = self.phase_samples["approach"]
        critical = self.critical_samples
        front_support = self.phase_samples["front_top_support"] + self.phase_samples["rear_clear"]
        top_hold = self.phase_samples["top_hold"]
        critical_width = [sample["rear_width"] for sample in critical]
        critical_center = [sample["rear_min_abs_y"] for sample in critical]
        approach_width_median = _safe_quantile([sample["rear_width"] for sample in approach], 0.50)
        critical_width_q05 = _safe_quantile(critical_width, 0.05)
        width_drop = None
        if approach_width_median is not None and critical_width_q05 is not None:
            width_drop = approach_width_median - critical_width_q05
        all_samples = [sample for phase in self.PHASES for sample in self.phase_samples[phase]]
        event_sequence_valid = bool(
            self.front_bilateral_top_step is not None
            and self.first_rear_top_step is not None
            and self.second_rear_top_step is not None
            and self.rear_advance_step is not None
            and self.front_bilateral_top_step <= self.first_rear_top_step
            and self.first_rear_top_step <= self.second_rear_top_step
            and self.second_rear_top_step <= self.rear_advance_step
        )
        target_fraction_by_joint = {
            name: (
                self.target_limit_joint_violation_counts[index] / self.target_limit_sample_steps
                if self.target_limit_sample_steps > 0
                else None
            )
            for index, name in enumerate(self.target_joint_names)
        }
        target_max_by_joint = {
            name: self.target_limit_joint_max_deltas[index]
            for index, name in enumerate(self.target_joint_names)
        }
        target_contract = {}
        if self.target_limit_contract_valid:
            clip_cpu = self.target_action_term._clip[0].detach().cpu()
            target_contract = {
                name: [float(clip_cpu[index, 0]), float(clip_cpu[index, 1])]
                for index, name in enumerate(self.target_joint_names)
            }
        target_scale = (
            [float(value) for value in self.target_affine_scale.detach().cpu().tolist()]
            if self.target_affine_scale is not None
            else None
        )
        target_offset = (
            [float(value) for value in self.target_affine_offset.detach().cpu().tolist()]
            if self.target_affine_offset is not None
            else None
        )
        geometry_samples = len(all_samples)
        top_hold_success = self.phase >= 5 and self.top_stable_max >= 25 and not self.terminated_early
        rear_on_platform_hold_success = bool(
            self.rear_on_platform_hold_step is not None
            and self.rear_on_platform_hold_max >= self.rear_on_platform_hold_confirm_steps
            and not self.terminated_early
        )
        strict_full_climb_success = bool(top_hold_success and event_sequence_valid)
        context_reset_valid = bool(self.context and self.context.get("reset_valid", False))
        summary: dict[str, object] = {
            "schema_version": 7,
            "reset_context_valid": context_reset_valid,
            "reset_valid": bool(
                context_reset_valid and self.initial_geometry_checked and self.initial_geometry_valid
            ),
            "reset_low_patch_verified": bool(
                self.context and self.context.get("low_patch_verified", False)
            ),
            "reset_step_height": (
                float(self.context["step_height"][0].detach().cpu()) if self.context else None
            ),
            "reset_low_patch_xy_error": (
                float(self.context["low_patch_xy_error"][0].detach().cpu()) if self.context else None
            ),
            "reset_root_height_above_low": (
                float(self.context["root_height_above_low"][0].detach().cpu()) if self.context else None
            ),
            "initial_geometry_checked": self.initial_geometry_checked,
            "initial_geometry_valid": self.initial_geometry_valid,
            "initial_foot_height_error_max": self.initial_foot_height_error_max,
            "initial_no_top_contact": self.initial_no_top_contact,
            "initial_all_feet_outside_platform": self.initial_all_feet_outside_platform,
            "platform_width": float(self.context["platform_width"]) if self.context else None,
            "reset_edge_clearance": (
                float(self.context["edge_clearance"][0].detach().cpu()) if self.context else None
            ),
            "yaw_offset_deg": float(self.context["yaw_offset_deg"]) if self.context else None,
            "samples": self.target_limit_sample_steps,
            "geometry_samples": geometry_samples,
            "approach_samples": len(approach),
            "front_lift_samples": len(self.phase_samples["front_lift"]),
            "front_edge_contact_samples": len(self.phase_samples["front_edge_contact"]),
            "front_top_support_samples": len(self.phase_samples["front_top_support"]),
            "rear_clear_samples": len(self.phase_samples["rear_clear"]),
            "top_hold_samples": len(self.phase_samples["top_hold"]),
            "front_lift_reached": self.phase >= 1,
            "front_edge_contact_reached": self.phase >= 2,
            "front_top_support_reached": self.phase >= 3,
            "front_support_reached": self.phase >= 3,
            "rear_clear_reached": self.phase >= 4,
            "rear_advance_reached": self.rear_advance_step is not None,
            "rear_on_platform_hold_success": rear_on_platform_hold_success,
            "rear_on_platform_hold_step": self.rear_on_platform_hold_step,
            "rear_on_platform_hold_max_consecutive": self.rear_on_platform_hold_max,
            "rear_on_platform_hold_confirm_steps": self.rear_on_platform_hold_confirm_steps,
            "rear_on_platform_min_root_edge_margin": self.rear_on_platform_min_root_edge_margin,
            "rear_on_platform_min_rear_edge_margin": self.rear_on_platform_min_rear_edge_margin,
            "rear_on_platform_min_root_h_top": self.rear_on_platform_min_root_h_top,
            "rear_on_platform_min_rear_width": self.rear_on_platform_min_rear_width,
            "rear_on_platform_min_rear_abs_y": self.rear_on_platform_min_rear_abs_y,
            "rear_on_platform_max_pitch_metric": self.rear_on_platform_max_pitch_metric,
            "rear_on_platform_max_roll_metric": self.rear_on_platform_max_roll_metric,
            "rear_on_platform_hold_samples": len(self.rear_on_platform_hold_samples),
            "front_bilateral_top_contact_rate_rear_on_platform": _safe_mean(
                [
                    sample["front_bilateral_top_contact"]
                    for sample in self.rear_on_platform_hold_samples
                ]
            ),
            "event_sequence_valid": event_sequence_valid,
            "top_hold_reached": self.phase >= 5,
            "top_hold_max_consecutive": self.top_stable_max,
            "kinematic_top_hold_success": top_hold_success,
            "strict_full_climb_success": strict_full_climb_success,
            "full_climb_success": rear_on_platform_hold_success,
            "terminated_early": self.terminated_early,
            "termination_step": self.termination_step,
            "approach_rear_width_median": approach_width_median,
            "critical_rear_width_q05": critical_width_q05,
            "critical_rear_min_abs_y_q05": _safe_quantile(critical_center, 0.05),
            "critical_rear_width_drop": width_drop,
            "critical_width_violation_rate": _safe_mean([float(value < 0.20) for value in critical_width]),
            "critical_center_violation_rate": _safe_mean([float(value < 0.04) for value in critical_center]),
            "critical_rear_bilateral_contact_rate": _safe_mean(
                [sample["rear_bilateral_contact"] for sample in critical]
            ),
            "critical_rear_slip_q95": _safe_quantile(
                [sample["rear_slip_contact"] for sample in critical], 0.95
            ),
            "front_wall_contact_rate_pre_support": _safe_mean(
                [sample["front_wall_contact"] for sample in self.pre_support_samples]
            ),
            "front_bilateral_top_contact_rate_support": _safe_mean(
                [sample["front_bilateral_top_contact"] for sample in front_support]
            ),
            "front_slip_top_q95": _safe_quantile(
                [sample["front_slip_top"] for sample in front_support], 0.95
            ),
            "front_single_contact_max_consecutive": self.front_single_contact_max,
            "front_step_contact_lag_steps": self._contact_lag(self.front_first_step_contact_steps),
            "front_top_contact_lag_steps": self._contact_lag(self.front_first_contact_steps),
            "rear_top_contact_lag_steps": self._contact_lag(self.rear_first_contact_steps),
            "front_bilateral_top_step": self.front_bilateral_top_step,
            "first_rear_top_step": self.first_rear_top_step,
            "second_rear_top_step": self.second_rear_top_step,
            "rear_advance_step": self.rear_advance_step,
            "front_support_to_first_rear_top_steps": self._ordered_dwell(
                self.front_bilateral_top_step,
                self.first_rear_top_step,
            ),
            "first_to_second_rear_top_steps": self._ordered_dwell(
                self.first_rear_top_step,
                self.second_rear_top_step,
            ),
            "second_rear_top_to_rear_advance_steps": self._ordered_dwell(
                self.second_rear_top_step,
                self.rear_advance_step,
            ),
            "front_support_grace_steps": self.front_support_grace_steps,
            "front_support_after_grace_samples": len(self.front_support_after_grace_samples),
            "front_bilateral_top_contact_rate_after_grace": _safe_mean(
                [sample["front_bilateral_top_contact"] for sample in self.front_support_after_grace_samples]
            ),
            "front_top_slip_after_grace_q95": _safe_quantile(self.front_top_slip_after_grace, 0.95),
            "rear_support_after_second_samples": len(self.rear_support_after_second_samples),
            "rear_bilateral_top_contact_rate_after_second": _safe_mean(
                [sample["rear_bilateral_top_contact"] for sample in self.rear_support_after_second_samples]
            ),
            "rear_top_slip_after_second_q95": _safe_quantile(self.rear_top_slip_after_second, 0.95),
            "rear_edge_dwell_max_consecutive": self.rear_edge_max,
            "target_limit_contract_valid": self.target_limit_contract_valid,
            "target_limit_contract": target_contract,
            "target_action_order_valid": self.target_action_order_valid,
            "target_action_joint_names": self.target_joint_names,
            "target_action_asset_joint_ids": self.target_joint_ids,
            "target_action_scale": target_scale,
            "target_action_offset": target_offset,
            "action_samples": self.target_limit_sample_steps,
            "target_limit_sample_steps": self.target_limit_sample_steps,
            "target_limit_action_input_valid": self.target_limit_action_input_valid,
            "target_limit_invalid_action_steps": self.target_limit_invalid_action_steps,
            "target_limit_violation_fraction": (
                self.target_limit_violation_count / self.target_limit_joint_samples
                if self.target_limit_joint_samples > 0
                else None
            ),
            "target_limit_step_violation_rate": (
                self.target_limit_step_violation_count / self.target_limit_sample_steps
                if self.target_limit_sample_steps > 0
                else None
            ),
            "target_limit_max_delta": self.target_limit_max_delta if self.target_limit_sample_steps > 0 else None,
            "target_limit_violation_max_consecutive": self.target_limit_violation_max_consecutive,
            "target_limit_violation_fraction_by_joint": target_fraction_by_joint,
            "target_limit_max_delta_by_joint": target_max_by_joint,
            "top_contact_min_upward_force": 5.0,
            "top_contact_min_upward_ratio": 0.45,
            "top_contact_height_lower_tolerance": 0.04,
            "top_contact_height_upper_tolerance": 0.09,
            "front_impact_speed_max": self.front_impact_speed_max,
            "front_impact_normal_speed_max": self.front_impact_normal_speed_max,
            "roll_metric_q95": _safe_quantile([sample["roll_metric"] for sample in critical], 0.95),
            "pitch_metric_q95": _safe_quantile([sample["pitch_metric"] for sample in critical], 0.95),
            "roll_rate_q95": _safe_quantile([sample["roll_rate"] for sample in critical], 0.95),
            "yaw_rate_q95": _safe_quantile([sample["yaw_rate"] for sample in critical], 0.95),
            "heading_error_q95": _safe_quantile([sample["heading_error"] for sample in critical], 0.95),
            "root_outward_first": self.first_root_outward,
            "root_outward_last": self.last_root_outward,
            "top_hold_root_edge_margin_q05": _safe_quantile(
                [sample["root_edge_margin"] for sample in top_hold], 0.05
            ),
            "top_hold_rear_edge_margin_q05": _safe_quantile(
                [sample["rear_edge_margin"] for sample in top_hold], 0.05
            ),
            "top_hold_all_feet_edge_margin_q05": _safe_quantile(
                [sample["all_feet_edge_margin"] for sample in top_hold], 0.05
            ),
            "root_h_top_last": self.last_root_h_top,
            "root_h_top_max": self.root_h_top_max if geometry_samples else None,
        }
        return summary


def _rear_width_metric_sample(env):
    """Return rear-foot width plus root progress diagnostics for high-step play videos."""
    unwrapped = env.unwrapped
    asset = unwrapped.scene["robot"]
    if not hasattr(unwrapped, "_play_rear_width_foot_ids"):
        unwrapped._play_rear_width_foot_ids = asset.find_bodies(["RL_foot", "RR_foot"])[0]

    rear_foot_ids = unwrapped._play_rear_width_foot_ids
    rear_pos_w = asset.data.body_pos_w[:, rear_foot_ids, :]
    base_pos_w = asset.data.root_pos_w
    heading_quat = math_utils.yaw_quat(asset.data.root_quat_w)

    base_pos_rep = base_pos_w[:, None, :].repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 3)
    heading_rep = heading_quat[:, None, :].repeat(1, rear_pos_w.shape[1], 1).reshape(-1, 4)
    rear_pos_b = math_utils.quat_apply_inverse(heading_rep, rear_pos_w.reshape(-1, 3) - base_pos_rep)
    rear_pos_b = rear_pos_b.reshape(unwrapped.num_envs, rear_pos_w.shape[1], 3)
    rear_y = rear_pos_b[:, :, 1]
    rear_width = torch.abs(rear_y[:, 0] - rear_y[:, 1])
    rear_min_abs_y = torch.min(torch.abs(rear_y), dim=1).values
    terrain_origins = unwrapped.scene.env_origins
    root_distance_to_origin = torch.norm(asset.data.root_pos_w[:, :2] - terrain_origins[:, :2], dim=1)
    root_height_above_origin = asset.data.root_pos_w[:, 2] - terrain_origins[:, 2]
    return rear_width, rear_min_abs_y, root_distance_to_origin, root_height_above_origin


def _update_highstep_gap_camera(env, mode: str):
    """Set an optional debug camera that makes rear-foot lateral width easier to inspect."""
    if mode == "none" or not hasattr(env.unwrapped, "viewport_camera_controller"):
        return

    robot = env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0]
    root_quat = robot.data.root_quat_w[0]

    if mode == "top":
        eye = root_pos + torch.tensor([0.0, 0.0, 5.2], dtype=torch.float32, device=env.device)
        lookat = root_pos + torch.tensor([0.0, 0.0, 0.15], dtype=torch.float32, device=env.device)
    else:
        if mode == "rear_top":
            eye_offset = torch.tensor([-2.2, 0.0, 1.65], dtype=torch.float32, device=env.device)
        else:
            eye_offset = torch.tensor([-1.2, -2.1, 1.45], dtype=torch.float32, device=env.device)
        lookat_offset = torch.tensor([0.45, 0.0, 0.25], dtype=torch.float32, device=env.device)
        eye = math_utils.transform_points(eye_offset.unsqueeze(0), pos=root_pos.unsqueeze(0), quat=root_quat.unsqueeze(0)).squeeze(0)
        lookat = math_utils.transform_points(
            lookat_offset.unsqueeze(0), pos=root_pos.unsqueeze(0), quat=root_quat.unsqueeze(0)
        ).squeeze(0)

    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=eye.detach().cpu().numpy(), lookat=lookat.detach().cpu().numpy()
    )
    if not getattr(env.unwrapped, "_highstep_camera_binding_reported", False):
        print(
            "[HIGHSTEP_CAMERA] "
            f"mode={mode} env=0 root={root_pos.detach().cpu().tolist()} "
            f"eye={eye.detach().cpu().tolist()} lookat={lookat.detach().cpu().tolist()}",
            flush=True,
        )
        env.unwrapped._highstep_camera_binding_reported = True

class _MoEActor(nn.Module):
    """Deterministic MoE actor head: softmax routing over expert MLPs.

    - No randomness inside the actor mean; all exploration still comes from base policy's Normal(mean, std).
    - Supports top-k sparse routing by zeroing non-topk logits before softmax (still deterministic).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: list[int],
        act: type[nn.Module],
        num_experts: int = 4,
        topk: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self._in_features = int(in_dim)     # NEW: 供导出器探测
        self._out_features = int(out_dim)   # NEW: 供导出器探测
        assert num_experts >= 1
        assert 1 <= topk <= num_experts
        self.num_experts = num_experts
        self.topk = topk
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.experts = nn.ModuleList([_make_mlp(in_dim, hidden, out_dim, act) for _ in range(num_experts)])
        # 一个小 gating MLP（与基类 actor 的规模同一量级即可）
        gate_hidden = max(64, (hidden[0] if hidden else 128) // 2)
        self.gate = _make_mlp(in_dim, [gate_hidden], num_experts, act)
    # --- 以下三个方法是“顺序模块”兼容探针 ---
    class _FakeLayer:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    def __len__(self):                      # NEW
        # 让导出器可以 len(self.actor)
        return 2

    def __getitem__(self, idx):             # NEW
        # 导出器会读 [0].in_features，可能也会读 [-1].out_features
        if idx in (0, -2):
            return self._FakeLayer(in_features=self._in_features)
        if idx in (-1, 1):
            return self._FakeLayer(out_features=self._out_features)
        raise IndexError(idx)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [B, E]
        if self.topk < self.num_experts:
            # 稀疏 top-k：非 top-k 位置置为 -inf，再 softmax
            topk_vals, topk_idx = logits.topk(self.topk, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, topk_idx, True)
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits / self.temperature.clamp(min=1e-6), dim=-1)  # [B, E]
        means = torch.stack([e(x) for e in self.experts], dim=1)              # [B, E, A]
        out = torch.einsum("be,bea->ba", probs, means)                        # [B, A]
        return out

# -------- Policy that reuses ALL rsl-rl logic and just swaps the actor --------
try:
    from rsl_rl.modules.actor_critic import ActorCritic as _BaseActorCritic
except Exception:
    import rsl_rl.modules.actor_critic as _ac_mod
    _BaseActorCritic = _ac_mod.ActorCritic

import torch
import torch.nn as nn

class ActorCriticMoE(_BaseActorCritic):
    """RSL-RL v3 兼容：基于基类的 MoE 策略。
    - 复用基类：obs 归一化 / log_std / act() / evaluate() / evaluate_actions() / 导出等
    - 仅替换 actor 的 MLP 为 MoE 头（确定性均值；探索仍由基类的 Normal(mean, std) 完成）
    """
    def __init__(
        self,
        # ★ v3 签名：先给 obs、obs_groups，再给 num_actions 与其余 cfg ★
        obs,                      # dict 或 TensorDict：来自环境的一个样本观测（含各组）
        obs_groups: dict,         # 形如 {'actor': ['policy'], 'critic': ['critic']}
        num_actions: int,
        *,
        actor_hidden_dims = [256, 256],
        critic_hidden_dims = [512, 256],
        activation = "elu",
        init_noise_std = 0.8,
        noise_std_type = "scalar",
        actor_obs_normalization = True,
        critic_obs_normalization = True,
        # MoE 相关
        num_experts: int = 4,
        topk: int = 1,
        moe_temperature: float = 1.0,
        **kwargs,
    ):
        # 先让基类按 v3 流程把一切搭好（包含 obs 归一化、log_std 等）
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims = actor_hidden_dims,
            critic_hidden_dims = critic_hidden_dims,
            activation = activation,
            init_noise_std = init_noise_std,
            noise_std_type = noise_std_type,
            actor_obs_normalization = actor_obs_normalization,
            critic_obs_normalization = critic_obs_normalization,
            **kwargs,
        )

        act = dict(relu=nn.ReLU, elu=nn.ELU, gelu=nn.GELU)[activation.lower()]

        # 1) 从基类已构建好的 actor MLP 里取首层 Linear 的 in_features（最稳妥）
        base_actor = self.actor
        actor_in_dim = None
        for m in base_actor.modules():
            if isinstance(m, nn.Linear):
                actor_in_dim = m.in_features
                break

        # 2) 兜底：如果意外没取到（几乎不会发生），再用 obs_groups 的 'policy' 写法；若还没有就取第一个键
        if actor_in_dim is None:
            def _lastdim(t):
                return int(t.shape[-1])
            keys_for_actor = obs_groups.get("policy", None)
            if keys_for_actor is None and "actor" in obs_groups:  # 兼容别处用过的命名
                keys_for_actor = obs_groups["actor"]
            if keys_for_actor is None:
                keys_for_actor = ["policy"] if (isinstance(obs, dict) and "policy" in obs) else [next(iter(obs.keys()))]
            actor_in_dim = sum(_lastdim(obs[k]) for k in keys_for_actor)

        # 3) 构建 MoE 头并替换
        self.actor = _MoEActor(
            in_dim = actor_in_dim,
            out_dim = num_actions,
            hidden = list(actor_hidden_dims),
            act    = act,
            num_experts = num_experts,
            topk        = topk,
            temperature = float(moe_temperature),
        )

        # 4) （可选）继续给 MoE 线性层加谱归一化，与你现有风格一致
        try:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    _sn(m, n_power_iterations=1)
        except Exception:
            pass

        # （可选）给 MoE 里的 Linear 上谱归一化，与你之前的 SN 风格一致
        try:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    _sn(m, n_power_iterations=1)
        except Exception:
            pass

    # 便于监控 gating 的平均熵（可 wandb.log）
# 让 Isaac Lab 用 getattr(...) 能找到：rsl_rl.modules.actor_critic.ActorCriticMoE

# === NEW: debug helper to print root pose & target height ===
def _print_root_and_target(env):
    """
    Print root_pos_w (x,y,z), configured target_height, and (if available)
    ground_z from height scanner plus adjusted target = ground_z + target_height.
    """
    try:
        asset = env.unwrapped.scene["robot"]
        rp = asset.data.root_pos_w[0].detach().cpu().numpy()
        msg = f"[ROOT_POS_W] x={rp[0]:+.3f}  y={rp[1]:+.3f}  z={rp[2]:+.3f}"
    except Exception as e:
        print(f"[WARN] Cannot read root_pos_w: {e}", flush=True)
        return

    # read target_height from env config if present
    th = None
    try:
        th = float(env.unwrapped.cfg.rewards.base_height_l2.params["target_height"])
        msg += f"  | target_height(cfg)={th:+.3f}"
    except Exception:
        pass

    # try to read ground estimate from a RayCaster named "height_scanner_base"
    try:
        sensor = env.unwrapped.scene.sensors.get("height_scanner_base", None)
        if sensor is not None:
            z_hits = sensor.data.ray_hits_w[0, :, 2]
            import torch
            valid = torch.isfinite(z_hits)
            if valid.any():
                ground_z = z_hits[valid].mean().item()
                msg += f"  | ground_z≈{ground_z:+.3f}"
                if th is not None:
                    msg += f"  | adjusted≈{ground_z + th:+.3f}"
    except Exception as e:
        msg += f"  | ground_z=N/A ({e})"

    print(msg, flush=True)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.eval_effort_limits is not None and not (
        args_cli.print_rear_width_metrics and args_cli.front_step_eval_reset
    ):
        raise ValueError(
            "--eval_effort_limits requires --print_rear_width_metrics and --front_step_eval_reset "
            "so the request, runtime verification, and applied-torque metrics reach HIGHSTEP_EVAL_JSON"
        )
    eval_effort_limit_request = _apply_eval_effort_limit_override(
        env_cfg, args_cli.eval_effort_limits
    )
    # if args_cli.debug:
    #     print("\n==== [env_cfg 配置结构] ====\n")
    #     print_dict(env_cfg.to_dict(), nesting=4)
    # with open("env_cfg_debug.json", "w") as f:
    #     json.dump(env_cfg.to_dict(), f, indent=4)
    agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    terrain_schedule_cfg_for_manifest = getattr(getattr(env_cfg, "curriculum", None), "terrain_levels", None)
    support_schedule_cfg_for_manifest = getattr(
        getattr(env_cfg, "curriculum", None), "highstep_action_score", None
    )
    command_schedule_cfg_for_manifest = getattr(
        getattr(env_cfg, "curriculum", None), "command_levels", None
    )
    reward_stage_definition_for_manifest = _highstep_reward_stage_definition(env_cfg)
    if args_cli.seed is not None:
        seed = int(args_cli.seed)
        env_cfg.seed = seed
        agent_cfg.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    if args_cli.disable_action_prior:
        from robot_lab.tasks.locomotion.velocity import mdp as velocity_mdp

        if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
            old_action_cfg = env_cfg.actions.joint_pos
            joint_names = getattr(old_action_cfg, "joint_names", getattr(env_cfg, "joint_names", None))
            if joint_names is None:
                raise ValueError("--disable_action_prior requires env_cfg.joint_names or actions.joint_pos.joint_names.")

            env_cfg.actions.joint_pos = velocity_mdp.JointPositionActionCfg(
                asset_name=getattr(old_action_cfg, "asset_name", "robot") or "robot",
                joint_names=joint_names,
                scale=getattr(old_action_cfg, "scale", 0.1),
                offset=getattr(old_action_cfg, "offset", 0.0),
                preserve_order=getattr(old_action_cfg, "preserve_order", False),
                use_default_offset=getattr(old_action_cfg, "use_default_offset", True),
                clip=getattr(old_action_cfg, "clip", None),
            )
            print("[INFO] Disabled action prior for play: using plain JointPositionActionCfg.")
        else:
            print("[WARN] --disable_action_prior requested, but env_cfg.actions.joint_pos was not found.")

    # =========================================================================
    # 🌟 修改点 1：拦截并覆盖 agent_cfg (针对 symmetric_ppo_cfg)
    # =========================================================================
    if args_cli.agent == "symmetric_ppo_cfg":
        print("[INFO] Using Symmetric PPO Algorithm and Config for Playback!")
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo_cfg import ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg

        agent_cfg = ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg()
        agent_cfg.class_name = "SymmetricOnPolicyRunner"

        # 如果需要重新应用 CLI 参数覆盖，可以取消下面这行的注释
        # if hasattr(cli_args, 'update_rsl_rl_cfg'):
        #     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # =========================================================================

    if args_cli.moe:
        import rsl_rl.modules.actor_critic as ac
        ac.ActorCriticMoE = ActorCriticMoE
        import rsl_rl.runners.on_policy_runner as _opr
        _opr.ActorCriticMoE = ActorCriticMoE
        agent_cfg.policy.class_name = "ActorCriticMoE"
        # 2) 强制对齐训练时的网络维度 / 激活 / 探索噪声
        #    （Hydra 风格的 'agent.policy.*' 在这个脚本里会被忽略，所以在代码里直接改 cfg）
        agent_cfg.policy.actor_hidden_dims = [512, 256, 128]
        agent_cfg.policy.critic_hidden_dims = [512, 256, 128]
        agent_cfg.policy.activation = "elu"
        agent_cfg.policy.init_noise_std = 0.8
    # make a smaller scene for play
    env_cfg.scene.num_envs = args_cli.num_envs
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        if args_cli.play_terrain_level is not None or args_cli.play_terrain_type is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = max(
                int(env_cfg.scene.terrain.terrain_generator.num_rows),
                int(args_cli.play_terrain_level or 0) + 1,
            )
            env_cfg.scene.terrain.terrain_generator.num_cols = max(
                int(env_cfg.scene.terrain.terrain_generator.num_cols),
                20,
            )
            env_cfg.scene.terrain.terrain_generator.curriculum = True
        else:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    # Nominal play is deterministic; the random-force scenario keeps the full configured DR stack.
    for group_name in ("policy", "estimator", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    if not args_cli.keep_play_randomization:
        for event_name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_mass",
            "randomize_com_positions",
            "randomize_reset_joints",
            "randomize_actuator_gains",
            "randomize_joint_friction",
            "randomize_screw_joints",
            "randomize_reset_base",
            "randomize_highstep_foot_under_hip_reset",
            "randomize_apply_external_force_torque",
            "randomize_limb_external_force_torque",
            "randomize_highstep_joint_observation_bias",
            "randomize_push_robot",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
    if args_cli.eval_action_delay_steps is not None:
        delay_steps = max(0, int(args_cli.eval_action_delay_steps))
        action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        if action_cfg is not None and hasattr(action_cfg, "min_action_delay_steps"):
            action_cfg.min_action_delay_steps = delay_steps
            action_cfg.max_action_delay_steps = delay_steps
        elif delay_steps > 0:
            raise ValueError(
                f"Requested eval action delay={delay_steps}, but this task has no delayed action term."
            )
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels = None

    fixed_velocity_command = tuple(float(v) for v in args_cli.fixed_velocity_command) if args_cli.fixed_velocity_command else None
    if fixed_velocity_command is not None:
        env_cfg.commands.base_velocity.debug_vis = True
        env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
        for group_name in ["policy", "estimator", "critic"]:
            if hasattr(env_cfg.observations, group_name):
                obs_group = getattr(env_cfg.observations, group_name)
                if hasattr(obs_group, "velocity_commands") and obs_group.velocity_commands is not None:
                    old_history_len = getattr(obs_group.velocity_commands, "history_length", 0)
                    old_flatten = getattr(obs_group.velocity_commands, "flatten_history_dim", False)

                    setattr(obs_group, "velocity_commands", ObsTerm(
                        func=lambda env, cmd=fixed_velocity_command: torch.tensor(
                            cmd, device=env.device, dtype=torch.float32
                        ).unsqueeze(0).repeat(env.num_envs, 1),
                        history_length=old_history_len,
                        flatten_history_dim=old_flatten,
                    ))

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True

         # =========================================================================
        # 🌟 新增：彻底禁用环境自带的随机速度指令重采样，防止与键盘冲突
        # =========================================================================
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
        # =========================================================================

        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
            # sim_device 默认即可；需要的话可传 env_cfg.sim.device
        )
        controller = Se2Keyboard(kb_cfg)  # ← 用配置类构造

        # # 返回形状 [1, 3] 的 (vx, vy, wz)
        # env_cfg.observations.policy.velocity_commands = ObsTerm(
        #     func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        # )

        # =========================================================================
        # 🌟 核心修复：必须将键盘指令同步注入到 Policy, Estimator 和 Critic 组！
        # 否则 VAE (Estimator) 会吃到 0 指令，导致 Latent 向量与 Policy 观测冲突，机器狗原地抽搐！
        # =========================================================================
        for group_name in ["policy", "estimator", "critic"]:
            if hasattr(env_cfg.observations, group_name):
                obs_group = getattr(env_cfg.observations, group_name)
                if hasattr(obs_group, "velocity_commands") and obs_group.velocity_commands is not None:
                    # 获取原配置中的历史长度设置，防止被覆盖丢失
                    old_history_len = getattr(obs_group.velocity_commands, "history_length", 0)
                    old_flatten = getattr(obs_group.velocity_commands, "flatten_history_dim", False)

                    setattr(obs_group, "velocity_commands", ObsTerm(
                        func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
                        history_length=old_history_len,       # 把历史长度加回来！
                        flatten_history_dim=old_flatten       # 保持原有的展平设置
                    ))
        # =========================================================================


        # Register the R callback only after the wrapped environment and the
        # deterministic front-step placement are ready.  The callback itself
        # merely queues the reset; performing env.reset() from an input event
        # can race the simulation step and used to lose the selected platform
        # placement.


    if args_cli.se2_gamepad:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True

        # 游戏手柄配置
        se2_gamepad_cfg = Se2GamepadCfg(
            v_x_sensitivity=2.0,
            v_y_sensitivity=1.5,
            omega_z_sensitivity=3.0,
            dead_zone=0.15,
        )
        se2_controller = Se2Gamepad(se2_gamepad_cfg)

        # =========================================================================
        # 🌟 核心修复：同样为手柄同步注入所有观测组
        # =========================================================================
        for group_name in ["policy", "estimator", "critic"]:
            if hasattr(env_cfg.observations, group_name):
                obs_group = getattr(env_cfg.observations, group_name)
                if hasattr(obs_group, "velocity_commands") and obs_group.velocity_commands is not None:
                    old_history_len = getattr(obs_group.velocity_commands, "history_length", 0)
                    old_flatten = getattr(obs_group.velocity_commands, "flatten_history_dim", False)

                    setattr(obs_group, "velocity_commands", ObsTerm(
                        func=lambda env: se2_controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
                        history_length=old_history_len,
                        flatten_history_dim=old_flatten
                    ))
        # =========================================================================

        # 重置环境回调
        def reset_env_callback():
            print("[INFO] Resetting environment...")
            return env.reset()[0]  # 返回新的观测

        se2_controller.add_callback(7, reset_env_callback)  # Start按钮

        # 退出应用回调
        def exit_app_callback():
            print("[INFO] Exiting application...")
            exit(0)

        se2_controller.add_callback(6, exit_app_callback)  # Back/Select按钮


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        # 如果传入的是完整的绝对路径，且文件存在，直接使用
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # 否则（比如只传了 model_7900.pt），统统交给 get_checkpoint_path 去 logs 目录里智能搜索
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    _apply_play_terrain_selection(env, args_cli.play_terrain_level, args_cli.play_terrain_type)

    # RslRlVecEnvWrapper immediately resets the task.  Freeze the restored
    # checkpoint schedule before that reset so evaluation never observes a
    # process-local update zero and never advances schedules as if play steps
    # were PPO learning updates.
    is_highstep_task = "highstep" in str(args_cli.task or "").lower()
    if args_cli.allow_legacy_highstep_schedule_fallback and not is_highstep_task:
        raise ValueError("--allow_legacy_highstep_schedule_fallback requires a high-step task")
    if args_cli.highstep_v18_teacher_bootstrap_profile and not (
        args_cli.task
        == "RobotLab-Isaac-Velocity-HighstepActionScoreTeacherV18Bootstrap-ArcdogAdjustableLeg-v0"
        and not args_cli.allow_legacy_highstep_schedule_fallback
    ):
        raise ValueError("v1.8 Teacher bootstrap profile flag requires the exact v1.8 Teacher task")
    checkpoint_iteration_hint = None
    schedule_update_hint = 0.0
    schedule_source_hint = {"method": "not_highstep"}
    schedule_manifest = None
    schedule_manifest_path = None
    if is_highstep_task:
        checkpoint_iteration_hint = _read_rsl_checkpoint_iteration(resume_path)
        if args_cli.highstep_v18_teacher_bootstrap_profile:
            schedule_update_hint = float(checkpoint_iteration_hint)
            schedule_source_hint = {
                "method": "v18_teacher_bootstrap_profile_checkpoint_anchor",
                "source_manifest_intentionally_not_imported": True,
            }
        else:
            schedule_update_hint, schedule_source_hint = resolve_loaded_schedule_update(
                resume_path,
                checkpoint_iteration_hint,
                checkpoint_load_mode="full",
                schedule_resume_mode="preserve",
                allow_legacy_checkpoint_fallback=args_cli.allow_legacy_highstep_schedule_fallback,
            )
        install_global_update(
            env,
            schedule_update_hint,
            source=str(schedule_source_hint.get("method", "unknown")),
            resume_mode="preserve",
            runner_iteration=checkpoint_iteration_hint,
            advance_with_local_steps=False,
        )
        print(
            "[INFO] Installed frozen pre-wrapper evaluation schedule: "
            f"update={schedule_update_hint}, checkpoint_iter={checkpoint_iteration_hint}."
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model

    # =========================================================================
    # 🌟 修改点 2：动态切换 RunnerClass (针对 SymmetricOnPolicyRunner)
    # =========================================================================
    if agent_cfg.class_name == "SymmetricOnPolicyRunner":
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo import SymmetricOnPolicyRunner
        runner = SymmetricOnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
            config=env_cfg  # 传入自定义需要的 config 参数
        )
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # =========================================================================

    # Evaluation never needs optimizer state.  Recovery stage B intentionally
    # replaces the legacy whole-policy optimizer with a two-row optimizer, so
    # loading an older model_900 optimizer here would be both unnecessary and
    # structurally incompatible.  Policy weights remain loaded exactly.
    runner.load(resume_path, load_optimizer=False)

    if is_highstep_task:
        checkpoint_iteration = int(getattr(runner, "current_learning_iteration", 0))
        if checkpoint_iteration != checkpoint_iteration_hint:
            raise RuntimeError(
                "Checkpoint iteration changed between pre-wrapper metadata read and runner.load: "
                f"{checkpoint_iteration_hint} != {checkpoint_iteration}"
            )
        if args_cli.highstep_v18_teacher_bootstrap_profile:
            schedule_update_at_anchor = float(checkpoint_iteration)
            schedule_source = {
                "method": "v18_teacher_bootstrap_profile_checkpoint_anchor",
                "source_manifest_intentionally_not_imported": True,
            }
        else:
            schedule_update_at_anchor, schedule_source = resolve_loaded_schedule_update(
                resume_path,
                checkpoint_iteration,
                checkpoint_load_mode="full",
                schedule_resume_mode="preserve",
                allow_legacy_checkpoint_fallback=args_cli.allow_legacy_highstep_schedule_fallback,
            )
        install_global_update(
            env,
            schedule_update_at_anchor,
            source=str(schedule_source.get("method", "unknown")),
            resume_mode="preserve",
            runner_iteration=checkpoint_iteration,
            advance_with_local_steps=False,
        )
        action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        source_manifest_path, source_manifest = load_source_manifest(resume_path)
        if source_manifest is not None and not args_cli.highstep_v18_teacher_bootstrap_profile:
            current_definition = schedule_definition_from_configs(
                action_cfg=action_cfg,
                terrain_term_cfg=terrain_schedule_cfg_for_manifest,
                support_term_cfg=support_schedule_cfg_for_manifest,
                command_term_cfg=command_schedule_cfg_for_manifest,
                command_curriculum_enabled=command_schedule_cfg_for_manifest is not None,
                reward_stage_definition=reward_stage_definition_for_manifest,
            )
            assert_schedule_definition_compatible(source_manifest, current_definition)
        schedule_manifest = build_schedule_manifest(
            context="play",
            task=args_cli.task,
            env=env,
            action_cfg=action_cfg,
            terrain_term_cfg=terrain_schedule_cfg_for_manifest,
            support_term_cfg=support_schedule_cfg_for_manifest,
            terrain_curriculum_enabled=False,
            support_metric_enabled=support_schedule_cfg_for_manifest is not None,
            checkpoint_path=resume_path,
            checkpoint_iteration=checkpoint_iteration,
            checkpoint_load_mode="full",
            requested_resume_mode="preserve",
            resolved_resume_mode="preserve",
            runner_iteration_at_anchor=checkpoint_iteration,
            schedule_update_at_anchor=schedule_update_at_anchor,
            schedule_source=schedule_source,
            command_term_cfg=command_schedule_cfg_for_manifest,
            command_curriculum_enabled=command_schedule_cfg_for_manifest is not None,
            reward_stage_definition=reward_stage_definition_for_manifest,
        )
        schedule_manifest["evaluation_code_sha256"] = _evaluation_code_sha256()
        schedule_manifest["source_manifest_checked"] = (
            source_manifest is not None and not args_cli.highstep_v18_teacher_bootstrap_profile
        )
        schedule_manifest["v18_teacher_bootstrap_profile"] = bool(
            args_cli.highstep_v18_teacher_bootstrap_profile
        )
        schedule_manifest["source_manifest_path_checked"] = (
            str(source_manifest_path) if source_manifest_path is not None else None
        )
        try:
            terrain = env.unwrapped.scene.terrain
            terrain_levels = getattr(terrain, "terrain_levels", None)
            terrain_types = getattr(terrain, "terrain_types", None)
            expected_type_column = (
                _terrain_column_for_name(
                    env.unwrapped.cfg.scene.terrain.terrain_generator,
                    args_cli.play_terrain_type,
                )
                if args_cli.play_terrain_type is not None
                else None
            )
            schedule_manifest["play_selection"] = {
                "env0_terrain_level": int(terrain_levels[0].item()) if terrain_levels is not None else None,
                "env0_terrain_type": int(terrain_types[0].item()) if terrain_types is not None else None,
                "requested_terrain_level": args_cli.play_terrain_level,
                "requested_terrain_type": args_cli.play_terrain_type,
                "requested_terrain_type_column": expected_type_column,
            }
        except (AttributeError, IndexError, TypeError):
            schedule_manifest["play_selection"] = {"unavailable": True}
        manifest_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{SCHEDULE_MANIFEST_NAME}"
        )
        schedule_manifest_path = write_manifest(
            os.path.join(log_dir, "eval_schedule_manifests", manifest_filename), schedule_manifest
        )
        print(f"[INFO] Highstep evaluation schedule manifest: {schedule_manifest_path}")
        print(
            "[INFO] Highstep evaluation schedule active: "
            + json.dumps(
                {
                    "update": schedule_manifest["schedule_update_at_anchor"],
                    "prior_scale": schedule_manifest["action_prior"]["actual_prior_scale"],
                    "terrain_curriculum_enabled": schedule_manifest["terrain_schedule"]["curriculum_enabled"],
                    "terrain_stage": schedule_manifest["terrain_schedule"]["configured_stage_index"],
                    "terrain_max_level": schedule_manifest["terrain_schedule"]["configured_allowed_max_level"],
                    "support_blend": schedule_manifest["support_bottleneck"]["actual_blend"],
                },
                sort_keys=True,
            )
        )

    # Any evaluation reset must happen after the checkpoint schedule is
    # installed, otherwise reset-time high-step metrics see update zero.
    if args_cli.reset_after_play_terrain_selection and (
        args_cli.play_terrain_level is not None or args_cli.play_terrain_type is not None
    ):
        print("[INFO] Resetting environment after applying play terrain selection.")
        env.reset()
    if args_cli.front_step_eval_reset:
        platform_width = _front_step_platform_width(
            env,
            args_cli.play_terrain_type,
            args_cli.front_step_eval_platform_width,
        )
        _apply_front_step_eval_reset(
            env,
            args_cli.front_step_eval_side,
            args_cli.front_step_eval_distance,
            args_cli.front_step_eval_edge_gap,
            platform_width,
            args_cli.front_step_eval_lateral_offset,
            args_cli.front_step_eval_yaw_offset_deg,
            args_cli.play_terrain_type,
        )

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # --- De-parametrize all layers before TorchScript export ---
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    def _deparametrize_all(m):
        # 遍历所有子模块，若存在任何参数化(如 spectral_norm)则移除
        for mod in m.modules():
            if is_parametrized(mod):
                # 逐个把所有被参数化的参数（通常是 "weight"）恢复成普通参数
                if hasattr(mod, "parametrizations"):
                    for pname in list(mod.parametrizations.keys()):
                        try:
                            remove_parametrizations(mod, pname, leave_parametrized=False)
                        except Exception:
                            pass

    # Export may mutate parametrized modules, so automated evaluation bypasses it entirely.
    if not args_cli.skip_policy_export:
        _deparametrize_all(policy_nn)

    # =========================================================================
    # 🌟 核心修改点：新增导出包含 Estimator 的完整 Student 策略
    # =========================================================================
    class StudentWrapper(torch.nn.Module):
        def __init__(self, policy_nn):
            super().__init__()
            self.actor = policy_nn.actor
            self.estimator = getattr(policy_nn, "estimator", None)

        def forward(self, obs: torch.Tensor):
            if self.estimator is not None:
                est_out = self.estimator(obs)
                # 兼容 VAE 返回值 (mu 通常是第3个返回值)
                if isinstance(est_out, tuple) and len(est_out) >= 3:
                    mu = est_out[2]
                elif isinstance(est_out, torch.Tensor):
                    mu = est_out
                else:
                    mu = torch.zeros((obs.shape[0], 64), device=obs.device)

                safe_mu = torch.clamp(mu, min=-1.0, max=1.0)
                actor_input = torch.cat([obs, safe_mu.detach()], dim=-1)
            else:
                actor_input = obs
            return self.actor(actor_input)

    if not args_cli.skip_policy_export:
        try:
            print("[INFO] Wrapping policy with StudentWrapper to include Estimator...")
            student_model = StudentWrapper(policy_nn).to(env.unwrapped.device)
            student_model.eval()

            obs_dict = env.get_observations()

            # 🌟 修复点：TensorDict 无法使用 isinstance(..., dict) 判断，必须安全提取纯 Tensor
            if hasattr(obs_dict, "keys") and "policy" in obs_dict.keys():
                dummy_obs = obs_dict["policy"]
            elif hasattr(obs_dict, "policy"):
                dummy_obs = obs_dict.policy
            else:
                dummy_obs = obs_dict

            # 确保提取出来的是纯 torch.Tensor，防止 Tracer 再次报错
            if not isinstance(dummy_obs, torch.Tensor):
                print(f"[WARN] dummy_obs is still not a pure Tensor, type is {type(dummy_obs)}. Attempting to convert...")
                if hasattr(dummy_obs, "contiguous"):
                    dummy_obs = dummy_obs.contiguous()

            print(f"[Debug Proof] Extracted dummy_obs type: {type(dummy_obs)}, shape: {dummy_obs.shape}")

            os.makedirs(export_model_dir, exist_ok=True)

            jit_path = os.path.join(export_model_dir, "policy_student.pt")
            traced_script_module = torch.jit.trace(student_model, dummy_obs)
            traced_script_module.save(jit_path)
            print(f"[Debug Proof] ✅ Successfully exported COMPLETE Student Policy (Estimator+Actor) to {jit_path}")

        except Exception as e:
            print(f"[WARN] Failed to export Student policy. Error: {e}")
    # =========================================================================

    # =========================================================================
    # 🌟 修改点：加入 try-except，防止 VAE 的字典输入结构导致 JIT/ONNX 导出崩溃
    # =========================================================================
    if not args_cli.skip_policy_export:
        try:
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx", verbose=True)
        except Exception as e:
            print(f"[WARN] Failed to export policy to JIT/ONNX. This is common for custom architectures like VAE. Error: {e}")
    # =========================================================================

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    manual_reset_requested = False
    if args_cli.keyboard:
        def request_manual_reset():
            nonlocal manual_reset_requested
            manual_reset_requested = True
            print("[INFO] 'R' key pressed: high-step respawn queued.", flush=True)

        controller.add_callback("R", request_manual_reset)
        print(
            "[INFO] Manual controls: arrows/numpad move, Z/X yaw, L stops commands, "
            "R respawns at the selected high-step start.",
            flush=True,
        )
    _update_highstep_gap_camera(env, args_cli.highstep_gap_camera)
    # # --- 构建观测切片索引：名字 -> slice(start, end) ---
    # def build_group_index_map(obs_mgr, group_name="policy"):
    #     names = obs_mgr._group_obs_term_names[group_name]
    #     shapes = obs_mgr._group_obs_term_dim[group_name]  # e.g. (171,), (3,), ...
    #     idx_map, start = {}, 0
    #     for name, shape in zip(names, shapes):
    #         # shape 可能是 int 或 tuple，做个通用乘积
    #         if isinstance(shape, (list, tuple)):
    #             n = 1
    #             for s in shape:
    #                 n *= int(s)
    #         else:
    #             n = int(shape)
    #         idx_map[name] = slice(start, start + n)
    #         start += n
    #     return idx_map

    # # 在获取到第一帧 obs 之后构建一次映射
    # obs_mgr = env.unwrapped.observation_manager
    # idx_map = build_group_index_map(obs_mgr, group_name="policy")

    # # 取出并打印某个 env 的 height_scan（这里以 env_id = 0 为例）
    # env_id = 0
    # hs = obs[env_id, idx_map["height_scan"]].detach().cpu().numpy()
    # print(f"[height_scan] env#{env_id} len={hs.size}:")
    # print(hs)

    # # 如果你想按网格显示（171=9*19 很常见），可 reshape 看看
    # try:
    #     hs_grid = hs.reshape(9, 19)   # 若你的配置不是 9x19，把 9,19 换成你的行列
    #     print("[height_scan as grid 9x19]:")
    #     print(hs_grid)
    # except Exception:
    #     pass

    timestep = 0
    rear_width_stats = {
        "count": 0,
        "width_sum": 0.0,
        "width_min": float("inf"),
        "min_abs_y_sum": 0.0,
        "min_abs_y_min": float("inf"),
        "root_distance_first": None,
        "root_distance_last": None,
        "root_height_sum": 0.0,
        "root_height_max": float("-inf"),
    }
    highstep_eval_tracker = None
    if args_cli.print_rear_width_metrics and args_cli.front_step_eval_reset:
        highstep_eval_tracker = _HighstepEvalTracker(env)
    effort_feasibility_tracker = None
    if eval_effort_limit_request is not None:
        effort_feasibility_tracker = _EvalEffortFeasibilityTracker(
            env, eval_effort_limit_request
        )
    debug_print = False
    if args_cli.debug and not debug_print:
        # print obs & action dim
        print("\n========== [OBSERVATION / ACTION SHAPE INFO] ==========", flush=True)
        try:
            # print observation vector shape
            if isinstance(obs, dict):
                flat_obs_shape = sum([v.numel() for v in obs.values()])
                print(f"[OBS] Total flattened shape: {flat_obs_shape} (from {len(obs)} components)", flush=True)
            else:
                print(f"[OBS] shape: {tuple(obs.shape)}", flush=True)

            # print action vector shape
            action_tensor = env.unwrapped.action_manager.action
            print(f"[ACTION] shape: {tuple(action_tensor.shape)}", flush=True)
        except Exception as e:
            print(f"[WARN] Cannot access shape info: {e}", flush=True)
        print("========================================================\n", flush=True)

        # print obs group -> term list
        print("\n========== [OBS GROUP MEMBERS LIST & INFO] ==========", flush=True)
        try:
            obs_mgr = env.unwrapped.observation_manager
            for group_name, term_names in obs_mgr._group_obs_term_names.items():
                print(f"[OBS GROUP] {group_name}: {term_names}", flush=True)
                for idx, name in enumerate(term_names):
                    term_cfg = obs_mgr._group_obs_term_cfgs[group_name][idx]
                    shape = obs_mgr._group_obs_term_dim[group_name][idx]
                    func_name = getattr(term_cfg.func, '__name__', str(term_cfg.func))
                    noise_type = type(term_cfg.noise).__name__ if term_cfg.noise else None
                    # print detailed info
                    print(f"  [OBS NAME] {name}", flush=True)
                    print(f"    [FUNC]        {func_name}", flush=True)
                    print(f"    [SHAPE]       {shape}", flush=True)
                    print(f"    [HISTORY]     len={term_cfg.history_length} flatten={term_cfg.flatten_history_dim}", flush=True)
                    print(f"    [CLIP]        {term_cfg.clip}", flush=True)
                    print(f"    [SCALE]       {term_cfg.scale}", flush=True)
                    print(f"    [NOISE]       {noise_type}", flush=True)
                    # 专门处理 joint_pos 观测项
                    if name == "joint_pos":
                        # 打印额外的配置信息
                        print(f"    [SPECIFIC CONFIG FOR joint_pos]", flush=True)
                        # 获取关节名称
                        if hasattr(term_cfg, 'params') and 'asset_cfg' in term_cfg.params:
                            asset_cfg = term_cfg.params['asset_cfg']
                            print(f"      [JOINT_NAMES] {asset_cfg.joint_names}", flush=True)
                            # try:
                            #     asset = env.unwrapped.scene[asset_cfg.name]
                            #     joint_names = asset.joint_names
                            #     print(f"      [JOINT_NAMES] {joint_names}", flush=True)
                            # except Exception as e:
                            #     print(f"      [ERROR] Failed to get joint names: {str(e)}", flush=True)

        except Exception as e:
            print(f"[WARN] Observation manager terms not accessible: {e}", flush=True)
        print("======================================================\n", flush=True)

        # print action space vector and mapped targets
        print("\n====== [Action & Target Mapping] ======", flush=True)
        idx = 0
        for group_name, term in env.unwrapped.action_manager._terms.items():
            print(f"[ACTION GROUP] {group_name}", flush=True)

            # 获取关节名称
            joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]

            # 1. 获取网络输出的原始动作 (Raw Action)
            raw_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()

            # 2. 获取真正传给机器人的映射后目标位置 (Mapped Target Pos)
            # 在 IsaacLab 中，处理后的目标动作通常存在 processed_actions 或类似属性中
            mapped_targets = None
            if hasattr(term, "processed_actions"):
                mapped_targets = term.processed_actions[0].cpu().numpy()
            elif hasattr(term, "target_joint_pos"): # 兼容不同版本的 IsaacLab/Orbit
                mapped_targets = term.target_joint_pos[0].cpu().numpy()

            # 遍历打印对比
            for i, val in enumerate(raw_actions):
                joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"

                if mapped_targets is not None:
                    target_val = mapped_targets[i]
                    print(f"  {joint_name:>14s} | Raw Action: {val:>+7.4f}  ==>  Mapped Target: {target_val:>+7.4f}", flush=True)
                else:
                    # 如果找不到 processed_actions 属性，尝试打印 scale 帮助分析
                    scale_val = term.action_scale[i].item() if hasattr(term, "action_scale") else "unknown"
                    print(f"  {joint_name:>14s} | Raw Action: {val:>+7.4f}  (Scale: {scale_val})", flush=True)

            idx += term.action_dim
        print("=======================================\n", flush=True)

        debug_print = True
        time.sleep(0.1)  # avoid stdout loss
    # # 导出后切 JIT
    # jit_path = os.path.join(export_model_dir, "policy.pt")
    # del runner           # 不再需要含 critic 的 runner
    # torch.cuda.empty_cache() # 回收显存

    # policy_jit = torch.jit.load(jit_path, map_location=env.unwrapped.device)
    # policy_jit.eval()

    # # =========================================================================
    # # 🌟 新增：用于控制 debug 打印频率的计时器，防止刷屏影响查看
    # # =========================================================================
    # last_debug_print_time = time.time()
    # debug_print_interval = 0.1  # 每 0.1 秒打印一次，你可以根据需要调大或调小

    # simulate environment
    while simulation_app.is_running():

        if args_cli.keyboard and manual_reset_requested:
            # Handle the reset on the simulation thread.  Re-selecting the
            # terrain before env.reset() preserves the requested level/type;
            # the explicit placement afterwards preserves the corridor pose.
            # State used by the high-step curriculum is first allocated while
            # policy stepping is in inference mode.  Reset must use the same
            # mode: mutating an inference tensor outside it is a hard PyTorch
            # error and previously made the R shortcut close the application.
            with torch.inference_mode():
                controller.reset()
                _apply_play_terrain_selection(env, args_cli.play_terrain_level, args_cli.play_terrain_type)
                obs, _ = env.reset()
                if args_cli.front_step_eval_reset:
                    platform_width = _front_step_platform_width(
                        env,
                        args_cli.play_terrain_type,
                        args_cli.front_step_eval_platform_width,
                    )
                    _apply_front_step_eval_reset(
                        env,
                        args_cli.front_step_eval_side,
                        args_cli.front_step_eval_distance,
                        args_cli.front_step_eval_edge_gap,
                        platform_width,
                        args_cli.front_step_eval_lateral_offset,
                        args_cli.front_step_eval_yaw_offset_deg,
                        args_cli.play_terrain_type,
                    )
                obs = env.get_observations()
            manual_reset_requested = False
            print("[INFO] High-step respawn complete; command reset to zero.", flush=True)

        # # =========================================================================
        # # 🌟 修改点：加入时间节流 (Throttling) 判断
        # # =========================================================================
        # current_time = time.time()
        # should_print_debug = (current_time - last_debug_print_time) >= debug_print_interval
        # if should_print_debug:
        #     last_debug_print_time = current_time

        # print action space vector
        # if args_cli.debug and args_cli.keyboard and should_print_debug:
        if args_cli.debug and args_cli.keyboard:
            # print action space vector and mapped targets
            print("\n====== [Action & Target Mapping] ======", flush=True)
            idx = 0
            for group_name, term in env.unwrapped.action_manager._terms.items():
                print(f"[ACTION GROUP] {group_name}", flush=True)

                # 获取关节名称
                joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]

                # 1. 获取网络输出的原始动作 (Raw Action)
                raw_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()

                # 2. 获取真正传给机器人的映射后目标位置 (Mapped Target Pos)
                # 在 IsaacLab 中，处理后的目标动作通常存在 processed_actions 或类似属性中
                mapped_targets = None
                if hasattr(term, "processed_actions"):
                    mapped_targets = term.processed_actions[0].cpu().numpy()
                elif hasattr(term, "target_joint_pos"): # 兼容不同版本的 IsaacLab/Orbit
                    mapped_targets = term.target_joint_pos[0].cpu().numpy()

                # 遍历打印对比
                for i, val in enumerate(raw_actions):
                    joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"

                    if mapped_targets is not None:
                        target_val = mapped_targets[i]
                        print(f"  {joint_name:>14s} | Raw Action: {val:>+7.4f}  ==>  Mapped Target: {target_val:>+7.4f}", flush=True)
                    else:
                        # 如果找不到 processed_actions 属性，尝试打印 scale 帮助分析
                        scale_val = term.action_scale[i].item() if hasattr(term, "action_scale") else "unknown"
                        print(f"  {joint_name:>14s} | Raw Action: {val:>+7.4f}  (Scale: {scale_val})", flush=True)

                idx += term.action_dim
            print("=======================================\n", flush=True)

            # === NEW: also print root_pos_w & (optional) ground/adjusted target
            _print_root_and_target(env)

            # # =========================================================================
            # # 🌟 核心修复：专门用于实时查看 vel_command_obs 的多组 Debug 面板
            # # =========================================================================
            # print("\n====== [REAL-TIME COMMAND DEBUG] ======", flush=True)
            # try:
            #     # 1. 打印键盘/手柄原始指令 (Controller Output) - 证明按键是否生效
            #     raw_cmd = controller.advance().squeeze().cpu().numpy() if args_cli.keyboard else se2_controller.advance().squeeze().cpu().numpy()
            #     print(f"  [Keyboard Cmd]    (vx, vy, wz): [{raw_cmd[0]:+.4f}, {raw_cmd[1]:+.4f}, {raw_cmd[2]:+.4f}]", flush=True)

            #     # 2. 提取并打印各组的观测指令
            #     obs_mgr = env.unwrapped.observation_manager

            #     def extract_active_frame(flat_array):
            #         if len(flat_array) > 0 and len(flat_array) % 3 == 0:
            #             frames = flat_array.reshape(-1, 3)
            #             max_norm = -1
            #             active = frames[0]
            #             for f in frames:
            #                 norm = np.linalg.norm(f)
            #                 if norm > max_norm:
            #                     max_norm = norm
            #                     active = f
            #             return active.tolist()
            #         elif len(flat_array) >= 3:
            #             return flat_array[:3].tolist()
            #         return flat_array.tolist()

            #     def get_cmd_from_group(g_name):
            #         if g_name in obs_mgr._group_obs_term_names:
            #             t_names = obs_mgr._group_obs_term_names[g_name]
            #             if "velocity_commands" in t_names:
            #                 idx = t_names.index("velocity_commands")
            #                 start_idx = 0
            #                 for i in range(idx):
            #                     shape = obs_mgr._group_obs_term_dim[g_name][i]
            #                     dim = 1
            #                     if isinstance(shape, (list, tuple)):
            #                         for s in shape: dim *= int(s)
            #                     else: dim = int(shape)
            #                     start_idx += dim

            #                 shape_cmd = obs_mgr._group_obs_term_dim[g_name][idx]
            #                 dim_cmd = 1
            #                 if isinstance(shape_cmd, (list, tuple)):
            #                     for s in shape_cmd: dim_cmd *= int(s)
            #                 else: dim_cmd = int(shape_cmd)

            #                 group_obs = obs[g_name] if (hasattr(obs, "keys") and g_name in obs.keys()) else (obs[g_name] if isinstance(obs, dict) else obs)
            #                 if isinstance(group_obs, torch.Tensor):
            #                     vel_obs_block = group_obs[0, start_idx : start_idx + dim_cmd].cpu().numpy().flatten()
            #                     return extract_active_frame(vel_obs_block)
            #         return None

            #     policy_cmd = get_cmd_from_group("policy")
            #     estimator_cmd = get_cmd_from_group("estimator")

            #     if policy_cmd:
            #         print(f"  [Policy Input]    (vx, vy, wz): [{policy_cmd[0]:+.4f}, {policy_cmd[1]:+.4f}, {policy_cmd[2]:+.4f}]", flush=True)
            #     # 🌟 核心证明点：如果这里打印的也是 +1.0000，说明 VAE 大脑分裂被治愈了！
            #     if estimator_cmd:
            #         print(f"  [Estimator Input] (vx, vy, wz): [{estimator_cmd[0]:+.4f}, {estimator_cmd[1]:+.4f}, {estimator_cmd[2]:+.4f}]", flush=True)

            #     # 3. 打印机器人的真实物理速度 (Actual Robot Velocity) - 证明机器人真的在按指令运动！
            #     try:
            #         robot = env.unwrapped.scene["robot"]
            #         lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
            #         ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
            #         print(f"  [Actual Robot]    (vx, vy, wz): [{lin_vel[0]:+.4f}, {lin_vel[1]:+.4f}, {ang_vel[2]:+.4f}]", flush=True)
            #     except Exception as e:
            #         print(f"  [Actual Robot]    N/A ({e})", flush=True)

            # except Exception as e:
            #     print(f"  [WARN] Failed to extract debug info: {e}", flush=True)
            # print("=======================================\n", flush=True)
            # # =========================================================================

            # # 取出并打印某个 env 的 height_scan（这里以 env_id = 0 为例）
            # env_id = 0
            # hs = obs[env_id, idx_map["height_scan"]].detach().cpu().numpy()
            # def print_height_scan_col_major(grid, precision=6, sep=" "):
            #     """
            #     按列打印：先 [0][0] [1][0] ... [8][0]，换行；
            #     然后 [0][1] [1][1] ... [8][1]，以此类推。
            #     """
            #     rows, cols = grid.shape
            #     fmt = f"{{:+.{precision}f}}"
            #     for c in range(cols):
            #         line = sep.join(fmt.format(float(grid[r, cols-1-c])) for r in range(rows))
            #         print(line, flush=True)

            # # 已有的 hs -> (9, 19)
            # hs_grid = hs.reshape(9, 19)
            # print_height_scan_col_major(hs_grid, precision=3)



        # if args_cli.debug and args_cli.se2_gamepad and should_print_debug:
        if args_cli.debug and args_cli.se2_gamepad:
            print("\n====== [Observatiion Information] ======", flush=True)
            idx = 0
            obs_mgr = env.unwrapped.observation_manager
            # "asset_cfg"= SceneEntityCfg("robot", joint_names=".*", preserve_order=True)
            asset_print = env.unwrapped.scene["robot"]  # 假设资产名为"robot"
            default_joint_pose = asset_print.data.default_joint_pos
            joint_names = asset_print.joint_names
            # default_joint_pose = env.unwrapped.cfg.scene[SceneEntityCfg("robot", joint_names=".*", preserve_order=True)].data.default_joint_pos[:, asset_cfg.joint_ids]
            for group_name, term_names in obs_mgr._group_obs_term_names.items():
                if group_name == "policy":
                    group_data = obs_mgr._obs_buffer[group_name].data
                    if isinstance(group_data, torch.Tensor):
                        joint_pos_rel_values = group_data.flatten()[9:27]
                        # 如果组数据是张量
                        # print(f"  Type: Tensor")
                        # print(f"  Shape: {group_data.shape}")
                        # # 打印部分值（避免打印过大张量）
                        # print(f"  base_ang_vel: {group_data.flatten()[0:3].tolist()}")
                        # print(f"  projected_gravity: {group_data.flatten()[3:6].tolist()}")
                        # print(f"  vel_command_obs: {group_data.flatten()[6:9].tolist()}")
                        # print(f"  joint_pos_rel: {group_data.flatten()[9:27].tolist()}")
                        # print(f"  joint_vel: {group_data.flatten()[27:45].tolist()}")
                        # print(f"  actions: {group_data.flatten()[45:63].tolist()}")
                        # print(f"  pose_command: {group_data.flatten()[63:70].tolist()}")
                        # for i, name in enumerate(joint_names):
                        #     default_joint_pose_val = default_joint_pose[0, i].item()
                        #     current_val = default_joint_pose_val + joint_pos_rel_values[i]
                        #     print(f"  {name:<25} | {current_val:10.6f}")

                        print(f"  base_ang_vel: {group_data.flatten()[0:3].tolist()}")
                        print(f"  projected_gravity: {group_data.flatten()[3:6].tolist()}")
                        print(f"  vel_command_obs: {group_data.flatten()[6:9].tolist()}")
                        print(f"  joint_pos_rel: {group_data.flatten()[9:27].tolist()}")
                        print(f"  joint_vel: {group_data.flatten()[27:45].tolist()}")
                        print(f"  actions: {group_data.flatten()[45:57].tolist()}")
                        print(f"  pose_command: {group_data.flatten()[57:64].tolist()}")
                    # if term_names == "action":
            # act_mgr = env.unwrapped.action_manager
            # act_data = act_mgr[]
            # for group_name, term in env.unwrapped.action_manager._terms.items():
            #     print(f"[ACTION GROUP] {group_name}", flush=True)
            #     joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]
            #     term_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
            #     for i, val in enumerate(term_actions):
            #         joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            #         print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
            #     idx += term.action_dim
            # print("=====================================\n", flush=True)

        # =========================================================================
        # 🌟 新增：将键盘/手柄的指令同步给 CommandManager，让绿色箭头动起来！
        # =========================================================================
        if args_cli.keyboard or args_cli.se2_gamepad or fixed_velocity_command is not None:
            try:
                # 获取环境中的 base_velocity 指令项
                cmd_term = env.unwrapped.command_manager._terms.get("base_velocity")
                if cmd_term is not None:
                    # 获取当前控制器的最新指令
                    if args_cli.keyboard:
                        cur_cmd = controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32)
                    elif args_cli.se2_gamepad:
                        cur_cmd = se2_controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32)
                    else:
                        cur_cmd = torch.tensor(
                            fixed_velocity_command, device=env.device, dtype=torch.float32
                        ).unsqueeze(0).repeat(env.num_envs, 1)

                    # 同步给底层的命令管理器 (仅用于可视化箭头等，不影响网络实际吃到的指令)
                    if hasattr(cmd_term, "command"):
                        cmd_term.command[:] = cur_cmd
                    if hasattr(cmd_term, "vel_command_b"):
                        cmd_term.vel_command_b[:] = cur_cmd
            except Exception as e:
                pass
        # =========================================================================

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            _update_highstep_gap_camera(env, args_cli.highstep_gap_camera)
            # agent stepping
            actions = policy(obs)
            if highstep_eval_tracker is not None:
                actions_for_audit = actions
                if getattr(env, "clip_actions", None) is not None:
                    actions_for_audit = torch.clamp(actions, -env.clip_actions, env.clip_actions)
                highstep_eval_tracker.audit_policy_actions(actions_for_audit, timestep)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if effort_feasibility_tracker is not None:
                effort_feasibility_tracker.sample_applied_torque()
            _update_highstep_gap_camera(env, args_cli.highstep_gap_camera)
            terminated_this_step = bool(torch.any(dones).item())
            if args_cli.print_rear_width_metrics:
                if highstep_eval_tracker is not None:
                    if terminated_this_step:
                        highstep_eval_tracker.mark_terminated(timestep)
                    else:
                        highstep_eval_tracker.update(timestep)
                rear_width, rear_min_abs_y, root_distance, root_height = _rear_width_metric_sample(env)
                width_val = float(rear_width.mean().detach().cpu())
                min_abs_y_val = float(rear_min_abs_y.mean().detach().cpu())
                root_distance_val = float(root_distance.mean().detach().cpu())
                root_height_val = float(root_height.mean().detach().cpu())
                rear_width_stats["count"] += 1
                rear_width_stats["width_sum"] += width_val
                rear_width_stats["width_min"] = min(rear_width_stats["width_min"], width_val)
                rear_width_stats["min_abs_y_sum"] += min_abs_y_val
                rear_width_stats["min_abs_y_min"] = min(rear_width_stats["min_abs_y_min"], min_abs_y_val)
                if rear_width_stats["root_distance_first"] is None:
                    rear_width_stats["root_distance_first"] = root_distance_val
                rear_width_stats["root_distance_last"] = root_distance_val
                rear_width_stats["root_height_sum"] += root_height_val
                rear_width_stats["root_height_max"] = max(rear_width_stats["root_height_max"], root_height_val)
                interval = max(1, int(args_cli.rear_width_metric_interval))
                if timestep % interval == 0:
                    print(
                        "[PLAY_METRIC] "
                        f"step={timestep} rear_width={width_val:.4f} rear_min_abs_y={min_abs_y_val:.4f} "
                        f"root_dist_to_origin={root_distance_val:.4f} root_h_rel_origin={root_height_val:.4f}",
                        flush=True,
                    )
        timestep += 1
        if args_cli.print_rear_width_metrics and terminated_this_step:
            print(f"[HIGHSTEP_EVAL] terminated_early step={timestep - 1}", flush=True)
            break
        if args_cli.video and timestep == args_cli.video_length:
            break
        if args_cli.play_max_steps is not None and timestep >= int(args_cli.play_max_steps):
            break
        # if args_cli.keyboard:
        #     rsl_rl_utils.camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.print_rear_width_metrics and rear_width_stats["count"] > 0:
        count = rear_width_stats["count"]
        print(
            "[PLAY_METRIC_SUMMARY] "
            f"samples={count} "
            f"rear_width_mean={rear_width_stats['width_sum'] / count:.4f} "
            f"rear_width_min={rear_width_stats['width_min']:.4f} "
            f"rear_min_abs_y_mean={rear_width_stats['min_abs_y_sum'] / count:.4f} "
            f"rear_min_abs_y_min={rear_width_stats['min_abs_y_min']:.4f} "
            f"root_dist_first={rear_width_stats['root_distance_first']:.4f} "
            f"root_dist_last={rear_width_stats['root_distance_last']:.4f} "
            f"root_h_rel_origin_mean={rear_width_stats['root_height_sum'] / count:.4f} "
            f"root_h_rel_origin_max={rear_width_stats['root_height_max']:.4f}",
            flush=True,
        )
    if highstep_eval_tracker is not None:
        highstep_eval_summary = highstep_eval_tracker.summary()
        if effort_feasibility_tracker is not None:
            highstep_eval_summary.update(effort_feasibility_tracker.summary())
        else:
            highstep_eval_summary.update(
                {
                    "eval_effort_limit_override_requested": False,
                    "eval_effort_limit_requested_nm": None,
                    "eval_effort_limit_runtime_observed": False,
                    "eval_effort_limit_runtime_match": None,
                }
            )
        highstep_eval_summary.update(eval_schedule_fields(schedule_manifest))
        highstep_eval_summary.update(_runtime_eval_action_delay(env, args_cli.eval_action_delay_steps))
        highstep_eval_summary.update(
            {
                "loop_steps": int(timestep),
                "requested_play_max_steps": (
                    int(args_cli.play_max_steps) if args_cli.play_max_steps is not None else None
                ),
                "fixed_velocity_command": (
                    list(fixed_velocity_command) if fixed_velocity_command is not None else None
                ),
                "keep_play_randomization": bool(args_cli.keep_play_randomization),
                "front_step_eval_side": args_cli.front_step_eval_side,
                "front_step_eval_edge_gap": float(args_cli.front_step_eval_edge_gap),
                "checkpoint_sha256": (
                    schedule_manifest.get("checkpoint_sha256") if schedule_manifest is not None else None
                ),
                "evaluation_code_sha256": (
                    schedule_manifest.get("evaluation_code_sha256") if schedule_manifest is not None else None
                ),
            }
        )
        highstep_eval_summary["schedule_manifest_path"] = (
            str(schedule_manifest_path) if schedule_manifest_path is not None else None
        )
        if schedule_manifest is not None:
            unwrapped = env.unwrapped
            action_steps_per_update = int(
                schedule_manifest["action_prior"].get("num_steps_per_update", 24) or 24
            )
            runtime_clock = schedule_clock_state(unwrapped, action_steps_per_update)
            highstep_eval_summary.update(
                {
                    "schedule_runtime_global_update": runtime_clock["global_update"],
                    "schedule_runtime_frozen": not runtime_clock["advance_with_local_steps"],
                    "schedule_runtime_source": runtime_clock["source"],
                    "schedule_runtime_resume_mode": runtime_clock["resume_mode"],
                    "schedule_runtime_runner_iteration_at_anchor": runtime_clock[
                        "runner_iteration_at_anchor"
                    ],
                    "schedule_runtime_local_common_step": runtime_clock["local_common_step"],
                }
            )
            runtime_prior_scale = None
            for action_term in getattr(unwrapped.action_manager, "_terms", {}).values():
                if hasattr(action_term, "_last_phased_highstep_prior_scale"):
                    runtime_prior_scale = float(action_term._last_phased_highstep_prior_scale)
                    break
            runtime_support_blend = getattr(unwrapped, "_highstep_support_bottleneck_blend", None)
            if runtime_support_blend is not None:
                runtime_support_blend = float(runtime_support_blend)
            expected_prior_scale = float(schedule_manifest["action_prior"]["actual_prior_scale"])
            expected_support_blend = float(schedule_manifest["support_bottleneck"]["actual_blend"])
            runtime_match = runtime_schedule_match(
                schedule_manifest,
                runtime_prior_scale=runtime_prior_scale,
                runtime_support_blend=runtime_support_blend,
            )
            schedule_clock_runtime_match = bool(
                abs(
                    float(runtime_clock["global_update"])
                    - float(schedule_manifest["schedule_update_at_anchor"])
                )
                <= 1.0e-9
                and runtime_clock["advance_with_local_steps"] is False
                and runtime_clock["source"] == schedule_manifest["schedule_source"].get("method")
                and runtime_clock["resume_mode"] == "preserve"
                and runtime_clock["runner_iteration_at_anchor"]
                == schedule_manifest["runner_iteration_at_anchor"]
            )
            highstep_eval_summary.update(
                {
                    **runtime_match,
                    "schedule_clock_runtime_match": schedule_clock_runtime_match,
                    "schedule_valid": bool(
                        runtime_match["schedule_runtime_match"] and schedule_clock_runtime_match
                    ),
                    "action_prior_scale": (
                        runtime_prior_scale if runtime_prior_scale is not None else expected_prior_scale
                    ),
                    "action_prior_scale_manifest": expected_prior_scale,
                    "support_bottleneck_blend": (
                        runtime_support_blend if runtime_support_blend is not None else expected_support_blend
                    ),
                    "support_bottleneck_blend_manifest": expected_support_blend,
                }
            )
        print(
            "[HIGHSTEP_EVAL_JSON] " + json.dumps(highstep_eval_summary, sort_keys=True),
            flush=True,
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
