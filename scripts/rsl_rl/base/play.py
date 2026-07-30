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
import tempfile

from isaaclab.app import AppLauncher
from isaaclab.utils.dict import print_dict
# from isaaclab.managers import SceneEntityCfg

# import json

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args
from play_joint_recorder import PlayJointRecorder
from highstep_recorded_command_replay import RecordedCommandReplay
from highstep_phase_residual_runtime import (
    PhaseResidualReference,
    PhaseResidualReferenceController,
)
from highstep_phase_residual_dataset import PhaseResidualDatasetCollector
from highstep_phase_residual_dagger import PhaseResidualDaggerCollector
from highstep_phase_residual_inference import (
    PhaseResidualHybridController,
    PhaseResidualPreroll,
)
from highstep_phase_direct_action_inference import PhaseDirectActionController
from highstep_fixed_motion_direct_action import (
    FixedMotionStudentController,
    OracleDatasetCollector,
    SafeMappedTargetAdapter,
)
from highstep_fixed_motion_raw_action import (
    RawActionDaggerCollector,
    RawActionStudentController,
    RawPassthroughCollector,
    load_phase_only_deployment_authority,
    load_one_shot_deployment_authority,
)
from highstep_b300_hybrid_latent import CanonicalTensorCollector, DaggerTensorCollector

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Optional exact output directory for play video; existing non-empty directories fail closed.",
)
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
parser.add_argument(
    "--record_joint_data",
    "--record-joint-data",
    action="store_true",
    default=False,
    help=(
        "Record env0 policy/raw actions, the exact 16 mapped joint targets shown by --debug, "
        "and measured joint state for offline Teacher/Student comparison."
    ),
)
parser.add_argument(
    "--record_student_tensors",
    "--record-student-tensors",
    action="store_true",
    default=False,
    help=(
        "With --record_joint_data, also save the exact pre-step 570-D Student/estimator "
        "observations, raw/clamped 64-D estimator mu, normalized 634-D actor input, and "
        "3-D velocity prediction to student_tensors.pt."
    ),
)
parser.add_argument(
    "--joint_record_output",
    "--joint-record-output",
    type=str,
    default=None,
    help=(
        "Exact new output directory for --record_joint_data. The default is a timestamped "
        "directory under logs/play_joint_records. Existing directories are never overwritten."
    ),
)
parser.add_argument(
    "--joint_record_label",
    "--joint-record-label",
    type=str,
    default=None,
    help="Short Teacher/Student label stored in the recording manifest and default directory name.",
)
parser.add_argument(
    "--phase_residual_reference",
    type=str,
    default=None,
    help=(
        "Independent fixed-condition experiment only: bypass the policy and replay a hash-bound "
        "16-joint safe reference while preserving and exactly cancelling the live production prior."
    ),
)
parser.add_argument(
    "--phase_residual_reference_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --phase_residual_reference.",
)
parser.add_argument(
    "--phase_residual_reference_blend_steps",
    type=int,
    default=25,
    help="Quintic current-q to reference-start blend length for the independent replay experiment.",
)
parser.add_argument(
    "--phase_residual_reference_restore_source_state",
    action="store_true",
    default=False,
    help=(
        "Independent fixed-condition experiment only: restore the hash-bound source root/joint "
        "pose and velocity, use zero blend, and continue from the following reference frame."
    ),
)
parser.add_argument(
    "--phase_residual_dataset_output",
    type=str,
    default=None,
    help="Exact new output directory for the independent 15-rollout Teacher dataset.",
)
parser.add_argument(
    "--phase_residual_selected_reference_manifest",
    type=str,
    default=None,
    help="Frozen selected-reference manifest required by --phase_residual_dataset_output.",
)
parser.add_argument(
    "--phase_residual_selected_reference_manifest_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --phase_residual_selected_reference_manifest.",
)
parser.add_argument(
    "--phase_residual_source_timing_amendment",
    type=str,
    default=None,
    help="Frozen source-timing amendment required by Teacher dataset collection.",
)
parser.add_argument(
    "--phase_residual_source_timing_amendment_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --phase_residual_source_timing_amendment.",
)
parser.add_argument(
    "--phase_residual_training_manifest",
    type=str,
    default=None,
    help="Frozen supervised-training manifest for the independent hybrid behavior gate.",
)
parser.add_argument(
    "--phase_residual_training_manifest_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --phase_residual_training_manifest.",
)
parser.add_argument(
    "--phase_residual_wandb_verification",
    type=str,
    default=None,
    help="Read-only W&B remote-verification record required before the behavior gate.",
)
parser.add_argument(
    "--phase_residual_wandb_verification_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --phase_residual_wandb_verification.",
)
parser.add_argument(
    "--phase_residual_behavior_output",
    type=str,
    default=None,
    help="Exact new output directory for one 15-rollout fixed-condition behavior gate.",
)
parser.add_argument(
    "--phase_residual_preroll_manifest",
    type=str,
    default=None,
    help="Frozen zero-command Teacher pre-roll manifest required by the behavior gate.",
)
parser.add_argument(
    "--phase_residual_preroll_manifest_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --phase_residual_preroll_manifest.",
)
parser.add_argument(
    "--phase_residual_disable_residual",
    action="store_true",
    default=False,
    help="Run the exact same behavior gate with the residual forced to zero.",
)
parser.add_argument(
    "--phase_direct_action",
    action="store_true",
    default=False,
    help=(
        "Independent fixed-condition branch only: interpret the supplied training/W&B "
        "manifests as a phase-indexed complete safe-target model rather than a residual model."
    ),
)
parser.add_argument(
    "--fixed_motion_direct_action_mode",
    choices=("oracle", "student", "dagger"),
    default=None,
    help="Independent user-approved fixed-motion branch mode.",
)
parser.add_argument("--fixed_motion_preregistration", type=str, default=None)
parser.add_argument("--fixed_motion_preregistration_sha256", type=str, default=None)
parser.add_argument("--fixed_motion_output", type=str, default=None)
parser.add_argument("--fixed_motion_training_manifest", type=str, default=None)
parser.add_argument("--fixed_motion_training_manifest_sha256", type=str, default=None)
parser.add_argument("--fixed_motion_wandb_verification", type=str, default=None)
parser.add_argument("--fixed_motion_wandb_verification_sha256", type=str, default=None)
parser.add_argument(
    "--fixed_motion_raw_action_mode",
    choices=("smoke", "student", "dagger"),
    default=None,
    help="Independent user-approved fixed-motion raw-action branch mode.",
)
parser.add_argument("--fixed_motion_raw_preregistration", type=str, default=None)
parser.add_argument("--fixed_motion_raw_preregistration_sha256", type=str, default=None)
parser.add_argument("--fixed_motion_raw_output", type=str, default=None)
parser.add_argument("--fixed_motion_raw_training_manifest", type=str, default=None)
parser.add_argument("--fixed_motion_raw_training_manifest_sha256", type=str, default=None)
parser.add_argument("--fixed_motion_raw_wandb_verification", type=str, default=None)
parser.add_argument("--fixed_motion_raw_wandb_verification_sha256", type=str, default=None)
parser.add_argument("--fixed_motion_raw_dagger_round", type=int, default=None)
parser.add_argument(
    "--b300_hybrid_canonical_output",
    type=str,
    default=None,
    help="Collect exactly one hash-bound B300 canonical tensor trajectory.",
)
parser.add_argument("--b300_hybrid_collection_preregistration", type=str, default=None)
parser.add_argument("--b300_hybrid_collection_preregistration_sha256", type=str, default=None)
parser.add_argument(
    "--b300_hybrid_behavior_preregistration",
    type=str,
    default=None,
    help="Permit only the preregistered B300 Student task to reuse the frozen Teacher reset/command trace.",
)
parser.add_argument("--b300_hybrid_behavior_preregistration_sha256", type=str, default=None)
parser.add_argument("--b300_hybrid_dagger_output", type=str, default=None)
parser.add_argument("--b300_hybrid_dagger_stage_manifest", type=str, default=None)
parser.add_argument("--b300_hybrid_dagger_stage_manifest_sha256", type=str, default=None)
parser.add_argument("--b300_hybrid_narrow_route_amendment", type=str, default=None)
parser.add_argument("--b300_hybrid_narrow_route_amendment_sha256", type=str, default=None)
parser.add_argument(
    "--b300_hybrid_initial_joint_delta_sign",
    type=int,
    choices=(-1, 0, 1),
    default=0,
    help="Preregistered small DAgger reset perturbation sign; zero outside DAgger.",
)
parser.add_argument("--fixed_motion_phase_only_deployment_preregistration", type=str, default=None)
parser.add_argument("--fixed_motion_phase_only_deployment_preregistration_sha256", type=str, default=None)
parser.add_argument("--fixed_motion_one_shot_preregistration", type=str, default=None)
parser.add_argument("--fixed_motion_one_shot_preregistration_sha256", type=str, default=None)
parser.add_argument(
    "--phase_residual_dagger_output",
    type=str,
    default=None,
    help=(
        "Independent branch only: record same-state Teacher labels while the hybrid candidate "
        "drives the fixed 15-environment behavior rollout. Requires the normal behavior gate."
    ),
)
parser.add_argument(
    "--phase_residual_dagger_preregistration",
    type=str,
    default=None,
    help="Frozen DAgger round preregistration required by --phase_residual_dagger_output.",
)
parser.add_argument(
    "--phase_residual_dagger_preregistration_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --phase_residual_dagger_preregistration.",
)
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
    "--highstep_0707_legacy_student_profile",
    action="store_true",
    default=False,
    help=(
        "Play the frozen 0707 deployed Student checkpoint with the hash-bound 0707 "
        "Student environment and the historical non-recovery Stage-2 agent contract."
    ),
)
parser.add_argument(
    "--keep_play_randomization",
    action="store_true",
    default=False,
    help="Keep configured play-time reset/force randomization for robustness eval. Default keeps normal deterministic play.",
)
parser.add_argument(
    "--training_distribution",
    action="store_true",
    default=False,
    help=(
        "Run play with the task's unmodified training terrain, reset, observation-noise, event, "
        "command and curriculum distribution. Intended for automatic multi-env visual inspection."
    ),
)
parser.add_argument(
    "--training_distribution_env_snapshot",
    type=str,
    default=None,
    help="Hash-bound params/env.yaml used to prove --training_distribution matches the saved training config.",
)
parser.add_argument(
    "--training_distribution_env_sha256",
    type=str,
    default=None,
    help="Expected SHA256 of --training_distribution_env_snapshot.",
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
    "--recorded_command_trace",
    type=str,
    default=None,
    help="Hash-bound PlayJointRecorder CSV whose one episode supplies the exact command timeline.",
)
parser.add_argument(
    "--recorded_command_trace_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --recorded_command_trace.",
)
parser.add_argument(
    "--recorded_command_episode_id",
    type=int,
    default=None,
    help="Exact source episode to replay from --recorded_command_trace.",
)
parser.add_argument(
    "--recorded_command_source_manifest",
    type=str,
    default=None,
    help="Source PlayJointRecorder manifest binding task, checkpoint and fixed-condition reset.",
)
parser.add_argument(
    "--recorded_command_source_manifest_sha256",
    type=str,
    default=None,
    help="Required SHA256 for --recorded_command_source_manifest.",
)
parser.add_argument(
    "--highstep_gap_camera",
    type=str,
    choices=("none", "rear_top", "top", "side_top", "static_side_top", "static_robot_side"),
    default="none",
    help=(
        "Debug-only camera for highstep videos. static_side_top is a one-shot terrain-fixed "
        "view; static_robot_side is positioned from the post-reset robot pose once and then "
        "remains world-fixed. Existing modes retain their historical robot-follow behavior."
    ),
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
parser.add_argument(
    "--highstep_v1123_frozen_preflight_output",
    type=str,
    default=None,
    help="Write one zero-training v1.12.3 reward-latch preflight record.",
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

from highstep_manual_respawn import ManualHighstepStartGate, restore_default_joint_state

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
    approach_boundary_half_width = half_width
    stair_height_bounds = None
    generator_cfg = unwrapped.cfg.scene.terrain.terrain_generator
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) if generator_cfg is not None else {}
    terrain_cfg = sub_terrains.get(terrain_type) if terrain_type is not None else None
    if terrain_type == "pyramid_stairs":
        if terrain_cfg is None or generator_cfg is None:
            raise RuntimeError("pyramid_stairs reset requires its frozen terrain configuration.")
        terrain_size = tuple(float(value) for value in generator_cfg.size)
        border_width = float(terrain_cfg.border_width)
        side_axis = 0 if side.startswith("x") else 1
        approach_boundary_half_width = 0.5 * terrain_size[side_axis] - border_width
        step_width = float(terrain_cfg.step_width)
        center_width = float(terrain_cfg.platform_width)
        num_steps_x = (terrain_size[0] - 2.0 * border_width - center_width) // (2.0 * step_width) + 1
        num_steps_y = (terrain_size[1] - 2.0 * border_width - center_width) // (2.0 * step_width) + 1
        num_steps = int(min(num_steps_x, num_steps_y))
        stair_height_bounds = (
            (num_steps + 1) * float(terrain_cfg.step_height_range[0]),
            (num_steps + 1) * float(terrain_cfg.step_height_range[1]),
        )
    target_distance = (
        approach_boundary_half_width + float(edge_gap) if distance is None else float(distance)
    )
    edge_clearance = target_distance - approach_boundary_half_width
    if edge_clearance < 0.15:
        raise ValueError(
            "Invalid front-step reset: base would start inside or too close to the platform. "
            f"distance={target_distance:.3f}, approach_boundary_half_width="
            f"{approach_boundary_half_width:.3f}, "
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
            elif terrain_type == "pyramid_stairs":
                # MeshPyramidStairsTerrain has an exact flat outer border.  The
                # randomly sampled flat patches are only witnesses of its Z;
                # using their XY directly can make a valid deterministic start
                # fail depending on the random patch locations.
                low_z = torch.where(
                    valid_mask, patch_z, torch.full_like(patch_z, float("inf"))
                ).amin(dim=1)
                ground_witness = valid_mask & (torch.abs(patch_z - low_z[:, None]) <= 1.0e-4)
                if not torch.all(torch.any(ground_witness, dim=1)):
                    raise RuntimeError("Pyramid-stairs reset has no verified outer-ground witness.")
                local_target = target_xy - env_origins[:, :2]
                target_outward = torch.sum(
                    local_target * approach.unsqueeze(0), dim=1
                )
                on_outer_ground = target_outward >= approach_boundary_half_width + 0.15
                terrain_size_tensor = torch.tensor(
                    generator_cfg.size, device=device, dtype=dtype
                )
                inside_terrain = torch.all(
                    torch.abs(local_target) <= 0.5 * terrain_size_tensor - 0.20,
                    dim=1,
                )
                if not torch.all(on_outer_ground & inside_terrain):
                    raise RuntimeError(
                        "Analytic pyramid-stairs reset target is not on the verified outer ground."
                    )
                selected_patches = selected_patches.clone()
                selected_patches[:, :2] = target_xy
                selected_patches[:, 2] = low_z
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
    actual_clearance = root_outward - approach_boundary_half_width
    step_height = env_origins[:, 2] - selected_patches[:, 2]
    root_height_above_low = positions[:, 2] - selected_patches[:, 2]
    if stair_height_bounds is None:
        terrain_height_valid = (step_height >= 0.28) & (step_height <= 0.40)
    else:
        terrain_height_valid = (
            (step_height >= stair_height_bounds[0] - 0.02)
            & (step_height <= stair_height_bounds[1] + 0.02)
        )
    reset_valid = bool(
        torch.all(actual_clearance >= 0.15).item()
        and torch.all(terrain_height_valid).item()
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
        "approach_boundary_half_width": approach_boundary_half_width,
        "target_distance": target_distance,
        "edge_clearance": actual_clearance.clone(),
        "low_patch_verified": True,
        "low_patch_source": (
            "analytic_box_ground"
            if terrain_type in {"box", "box_hard"}
            else "analytic_pyramid_outer_ground"
            if terrain_type == "pyramid_stairs"
            else "sampled_flat_patch"
        ),
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


def _apply_manual_highstep_respawn(
    env,
    side: str,
    distance: float | None,
    edge_gap: float,
    platform_width: float,
    lateral_offset: float,
    yaw_offset_deg: float,
    terrain_type: str | None,
) -> None:
    """Apply the complete deterministic state used by keyboard high-step play.

    ``env.reset()`` clears the action and manager state, but deterministic play
    disables the reset-joint randomization event.  Without this explicit write,
    pressing R therefore retained the previous episode's q/dq.  Restore q/dq,
    apply the deterministic root placement, then rebuild observation history
    from that exact state.
    """

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    env_ids, joint_pos, joint_vel = restore_default_joint_state(robot)
    _apply_front_step_eval_reset(
        env,
        side,
        distance,
        edge_gap,
        platform_width,
        lateral_offset,
        yaw_offset_deg,
        terrain_type,
    )
    unwrapped.sim.forward()
    unwrapped.observation_manager.reset(env_ids)
    unwrapped.observation_manager.compute(update_history=True)
    print(
        "[MANUAL_HIGHSTEP_RESPAWN] "
        + json.dumps(
            {
                "joint_position_source": "asset_default_joint_pos",
                "joint_velocity_source": "explicit_zero",
                "joint_count": int(joint_pos.shape[-1]),
                "max_abs_joint_velocity": float(torch.max(torch.abs(joint_vel)).item()),
                "observation_history_reprimed": True,
                "action_manager_reset_by_env_reset": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _phase_residual_batch_rear_on_platform(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env rear-top contact and the primary rear-hold predicate.

    This is the vectorized form of the existing env0 gate used only to reject
    a bad 15-rollout Teacher dataset.  It does not change the canonical gate.
    """
    unwrapped = env.unwrapped
    context = getattr(unwrapped, "_front_step_eval_context", None)
    if not isinstance(context, dict):
        raise RuntimeError("phase-residual batch gate requires front-step reset context")
    asset = unwrapped.scene["robot"]
    sensor = unwrapped.scene.sensors["contact_forces"]
    rear_asset_ids = [
        int(asset.find_bodies([name])[0][0]) for name in ("RL_foot", "RR_foot")
    ]
    rear_contact_ids = [
        int(sensor.find_bodies([name])[0][0]) for name in ("RL_foot", "RR_foot")
    ]
    rear_pos = asset.data.body_pos_w[:, rear_asset_ids, :]
    force_data = sensor.data
    if hasattr(force_data, "net_forces_w") and force_data.net_forces_w is not None:
        forces = force_data.net_forces_w[:, rear_contact_ids, :]
    else:
        forces = force_data.net_forces_w_history[:, 0, rear_contact_ids, :]
    force_norm = torch.linalg.norm(forces, dim=-1)
    upward = torch.clamp(forces[:, :, 2], min=0.0)
    upward_ratio = upward / torch.clamp(force_norm, min=1.0e-6)

    approach = context["approach"].reshape(1, 1, 2)
    lateral = context["lateral"].reshape(1, 1, 2)
    origin_xy = context["origin_xy"].reshape(unwrapped.num_envs, 1, 2)
    top_z = context["top_z"].reshape(unwrapped.num_envs, 1)
    half_width = float(context["half_width"])
    rear_rel = rear_pos[:, :, :2] - origin_xy
    rear_outward = torch.sum(rear_rel * approach, dim=-1)
    rear_lateral = torch.sum(rear_rel * lateral, dim=-1)
    inside_top = (
        (torch.abs(rear_outward) <= half_width + 0.04)
        & (torch.abs(rear_lateral) <= half_width + 0.04)
    )
    top_height = (
        (rear_pos[:, :, 2] >= top_z - 0.04)
        & (rear_pos[:, :, 2] <= top_z + 0.09)
    )
    rear_top = torch.all(
        (upward > 5.0) & (upward_ratio >= 0.45) & inside_top & top_height,
        dim=1,
    )

    base_pos = asset.data.root_pos_w
    root_rel = base_pos[:, :2] - context["origin_xy"]
    root_outward = torch.sum(root_rel * context["approach"].reshape(1, 2), dim=1)
    root_edge_margin = half_width - root_outward
    rear_edge_margin = half_width - torch.max(rear_outward, dim=1).values
    root_h_top = base_pos[:, 2] - context["top_z"]
    heading = math_utils.yaw_quat(asset.data.root_quat_w)
    heading_repeated = heading[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
    rear_body = math_utils.quat_apply_inverse(
        heading_repeated,
        (rear_pos - base_pos[:, None, :]).reshape(-1, 3),
    ).reshape(unwrapped.num_envs, 2, 3)
    rear_y = rear_body[:, :, 1]
    rear_width = torch.abs(rear_y[:, 0] - rear_y[:, 1])
    rear_min_abs_y = torch.min(torch.abs(rear_y), dim=1).values
    projected_gravity = asset.data.projected_gravity_b
    rear_on_platform = (
        rear_top
        & (rear_edge_margin >= 0.04)
        & (root_edge_margin >= 0.20)
        & (root_h_top >= 0.28)
        & (rear_width >= 0.18)
        & (rear_min_abs_y >= 0.04)
        & (torch.abs(projected_gravity[:, 0]) <= 0.42)
        & (torch.abs(projected_gravity[:, 1]) <= 0.32)
    )
    return rear_top, rear_on_platform


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


_HIGHSTEP_0707_STUDENT_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-"
    "ArcdogAdjustableLeg-v0"
)
_HIGHSTEP_0707_STUDENT_CHECKPOINT_SHA256 = (
    "7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284"
)
_HIGHSTEP_0707_STUDENT_AGENT_SHA256 = (
    "bb01ed1c7a8574363e9cea0d9d1dc4a0828d8453a097715ae2e4b49fa51be9ca"
)


def _apply_highstep_0707_legacy_student_profile(agent_cfg, task_name: str):
    """Restore the inference-relevant agent contract saved with the 0707 Student.

    The v1.8 bootstrap task is intentionally reused only for its SHA-bound 0707
    environment.  Its recovery runner configuration belongs to a later workflow,
    so manual replay must switch back to the historical, non-recovery Stage-2
    contract before constructing the runner.
    """
    if task_name != _HIGHSTEP_0707_STUDENT_TASK:
        raise ValueError(
            "--highstep_0707_legacy_student_profile requires the exact hash-bound "
            f"0707 Student task {_HIGHSTEP_0707_STUDENT_TASK!r}"
        )

    agent_cfg.experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student"
    )
    agent_cfg.max_iterations = 4900
    agent_cfg.save_interval = 100
    agent_cfg.empirical_normalization = False
    agent_cfg.obs_groups = {
        "policy": ["policy"],
        "estimator": ["estimator"],
        "critic": ["critic"],
    }

    policy = agent_cfg.policy
    policy.distill_stage = 2
    policy.student_recovery_stage = "none"
    policy.student_actor_warmup_updates = 1400
    policy.student_vae_epochs = 4
    policy.student_low_speed_threshold = 0.10
    policy.student_prior_fade_speed = 0.45
    policy.student_vel_loss_coef = 10.0
    policy.student_latent_loss_coef = 50.0
    policy.student_teacher_action_loss_coef = 20.0
    policy.student_prior_box_loss_coef = 5.0
    policy.student_recon_loss_coef = 0.5
    policy.student_kl_loss_coef = 0.1
    policy.student_post_prior_mode = "highstep"
    policy.student_highstep_phase_loss_scale = 2.0
    policy.student_highstep_rear_box_loss_scale = 1.5
    policy.student_highstep_rear_hip_loss_scale = 0.0
    policy.student_highstep_rear_hip_min_abs = 0.0

    algorithm = agent_cfg.algorithm
    algorithm.num_learning_epochs = 5
    algorithm.num_mini_batches = 4
    algorithm.learning_rate = 1.0e-4
    algorithm.schedule = "adaptive"
    algorithm.gamma = 0.99
    algorithm.lam = 0.95
    algorithm.entropy_coef = 0.0015
    algorithm.desired_kl = 0.006
    algorithm.max_grad_norm = 1.0
    algorithm.value_loss_coef = 1.0
    algorithm.use_clipped_value_loss = True
    algorithm.clip_param = 0.2

    print(
        "[0707_REPLAY] Restored historical deployed Student agent contract "
        f"(agent_yaml_sha256={_HIGHSTEP_0707_STUDENT_AGENT_SHA256}).",
        flush=True,
    )
    return agent_cfg


def _recording_tensor_values(value, *, env_index: int = 0) -> list[float]:
    """Convert one env row from a runtime tensor-like value to plain floats."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim > 1:
            tensor = tensor[env_index]
        return [float(item) for item in tensor.flatten().cpu().tolist()]
    array = np.asarray(value)
    if array.ndim > 1:
        array = array[env_index]
    return [float(item) for item in array.reshape(-1).tolist()]


def _joint_recording_schema(env) -> tuple[list[str], list[dict[str, object]]]:
    """Resolve the exact action-order joint contract used by the live manager."""
    terms = getattr(env.unwrapped.action_manager, "_terms", None)
    if not isinstance(terms, dict) or not terms:
        raise RuntimeError("--record_joint_data requires an initialized Isaac Lab action manager")
    joint_names: list[str] = []
    term_schema: list[dict[str, object]] = []
    for term_name, term in terms.items():
        action_dim = int(term.action_dim)
        names = list(getattr(term, "_joint_names", []))
        if len(names) != action_dim:
            raise RuntimeError(
                f"action term {term_name!r} exposes {action_dim} actions but {len(names)} joint names"
            )
        joint_names.extend(str(name) for name in names)
        term_schema.append(
            {
                "term_name": str(term_name),
                "term_class": type(term).__name__,
                "action_dim": action_dim,
                "joint_names": [str(name) for name in names],
            }
        )
    if len(joint_names) != 16:
        raise RuntimeError(
            "--record_joint_data is fail-closed to the high-step 16-action contract; "
            f"the active task exposes {len(joint_names)} actions"
        )
    if len(set(joint_names)) != len(joint_names):
        raise RuntimeError("action-order joint names are not unique")
    return joint_names, term_schema


def _joint_recording_snapshot(env, policy_actions, joint_names: list[str]) -> dict[str, list[float]]:
    """Capture env0 values after ``env.step`` using the same mapped tensor as --debug."""
    unwrapped = env.unwrapped
    action_manager = unwrapped.action_manager
    mapped_target: list[float] = []
    for term_name, term in action_manager._terms.items():
        mapped = getattr(term, "processed_actions", None)
        if mapped is None:
            mapped = getattr(term, "target_joint_pos", None)
        if mapped is None:
            raise RuntimeError(
                f"action term {term_name!r} exposes neither processed_actions nor target_joint_pos"
            )
        mapped_target.extend(_recording_tensor_values(mapped))

    robot = unwrapped.scene["robot"]
    joint_id_by_name = {name: index for index, name in enumerate(robot.joint_names)}
    missing = [name for name in joint_names if name not in joint_id_by_name]
    if missing:
        raise RuntimeError(f"recorded action joints are absent from robot joint state: {missing}")
    joint_ids = [joint_id_by_name[name] for name in joint_names]

    command = unwrapped.command_manager.get_command("base_velocity")
    return {
        "policy_raw": _recording_tensor_values(policy_actions),
        "action_manager_raw": _recording_tensor_values(action_manager.action),
        "mapped_target": mapped_target,
        "joint_pos": _recording_tensor_values(robot.data.joint_pos[:, joint_ids]),
        "joint_vel": _recording_tensor_values(robot.data.joint_vel[:, joint_ids]),
        "command": _recording_tensor_values(command)[:3],
        "root_pos": _recording_tensor_values(robot.data.root_pos_w)[:3],
        "root_quat_wxyz": _recording_tensor_values(robot.data.root_quat_w)[:4],
        "root_lin_vel_b": _recording_tensor_values(robot.data.root_lin_vel_b)[:3],
        "root_ang_vel_b": _recording_tensor_values(robot.data.root_ang_vel_b)[:3],
    }


def _recording_observation_group(observations, key: str) -> torch.Tensor:
    if hasattr(observations, "keys") and key in observations.keys():
        value = observations[key]
    elif isinstance(observations, dict) and key in observations:
        value = observations[key]
    else:
        raise RuntimeError(f"Student tensor recording lacks observation group {key!r}")
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Student observation group {key!r} is not a tensor")
    return value


def _student_tensor_recording_snapshot(observations, policy_nn) -> dict[str, torch.Tensor]:
    """Capture env0's exact deterministic Student inference inputs before ``env.step``."""
    estimator = getattr(policy_nn, "estimator", None)
    if estimator is None or not hasattr(estimator, "encode"):
        raise RuntimeError(
            "--record_student_tensors requires a Student policy with an encoder-based estimator"
        )
    policy_keys = list(getattr(policy_nn, "policy_keys", ()))
    estimator_keys = list(getattr(policy_nn, "estimator_keys", ()))
    if len(policy_keys) != 1 or len(estimator_keys) != 1:
        raise RuntimeError(
            "Student tensor recording requires one policy and one estimator observation group; "
            f"got policy={policy_keys}, estimator={estimator_keys}"
        )
    policy_obs = _recording_observation_group(observations, policy_keys[0])
    estimator_obs = _recording_observation_group(observations, estimator_keys[0])
    if policy_obs.ndim != 2 or estimator_obs.ndim != 2:
        raise RuntimeError(
            f"Student tensor observation rank changed: policy={tuple(policy_obs.shape)}, "
            f"estimator={tuple(estimator_obs.shape)}"
        )
    if policy_obs.shape[0] < 1 or estimator_obs.shape[0] < 1:
        raise RuntimeError("Student tensor recording has no env0 observation")
    with torch.no_grad():
        mu, _, velocity_pred = estimator.encode(estimator_obs)
        clamped_mu = torch.clamp(mu, min=-1.0, max=1.0)
        actor_input = torch.cat((policy_obs, clamped_mu), dim=-1)
        actor_normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        if actor_normalizer is not None:
            actor_input = actor_normalizer(actor_input)
    expected = {
        "student_obs_570": (policy_obs, 570),
        "estimator_obs_570": (estimator_obs, 570),
        "latent_raw_mu_64": (mu, 64),
        "latent_clamped_mu_64": (clamped_mu, 64),
        "actor_input_634": (actor_input, 634),
        "velocity_pred_3": (velocity_pred, 3),
    }
    result: dict[str, torch.Tensor] = {}
    for name, (tensor, width) in expected.items():
        if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != width:
            raise RuntimeError(
                f"Student tensor contract changed for {name}: got {tuple(tensor.shape)}, "
                f"expected (num_envs, {width})"
            )
        result[name] = tensor[0].detach().cpu().clone()
    return result


def _joint_recording_output_dir(label: str, requested: str | None) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    safe_label = "".join(character if character.isalnum() or character in "-_" else "_" for character in label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / "logs" / "play_joint_records" / f"{timestamp}_{safe_label}_pid{os.getpid()}").resolve()


_TRAINING_DISTRIBUTION_ALLOWED_CONFIG_DIFFS = (
    ("scene", "num_envs"),
    ("scene", "terrain", "num_envs"),
)


def _drop_nested_mapping_value(mapping: dict, path: tuple[str, ...]) -> None:
    parent = mapping
    for name in path[:-1]:
        value = parent.get(name)
        if not isinstance(value, dict):
            return
        parent = value
    parent.pop(path[-1], None)


def _first_mapping_differences(left, right, *, prefix: str = "", limit: int = 12) -> list[str]:
    if limit <= 0:
        return []
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(
                    _first_mapping_differences(
                        left[key], right[key], prefix=path, limit=limit - len(differences)
                    )
                )
            if len(differences) >= limit:
                break
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix or "<root>"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _first_mapping_differences(
                    left_item,
                    right_item,
                    prefix=f"{prefix}[{index}]",
                    limit=limit - len(differences),
                )
            )
            if len(differences) >= limit:
                break
        return differences
    return [] if left == right else [prefix or "<root>"]


def _prepare_optional_config_leaves_for_restore(obj, data: dict, *, prefix: str = "") -> None:
    """Give strict configclass merging concrete types for saved optional leaves."""
    from collections.abc import Mapping

    for key, saved_value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(obj, dict):
            if key not in obj:
                continue
            current_value = obj[key]
        else:
            if not hasattr(obj, key):
                continue
            current_value = getattr(obj, key)
        if isinstance(saved_value, Mapping):
            if current_value is None:
                raise RuntimeError(
                    "cannot reconstruct a saved optional config object from the current task class: "
                    + path
                )
            _prepare_optional_config_leaves_for_restore(current_value, saved_value, prefix=path)
        elif current_value is None and saved_value is not None:
            if isinstance(obj, dict):
                obj[key] = saved_value
            else:
                setattr(obj, key, saved_value)


def _verify_training_distribution_config(env_cfg, snapshot_path: str, expected_sha256: str) -> dict[str, object]:
    """Restore and prove the saved training cfg, allowing only env count to differ."""
    snapshot = Path(snapshot_path).expanduser().resolve()
    if not snapshot.is_file():
        raise RuntimeError(f"training distribution env snapshot is missing: {snapshot}")
    actual_sha256 = _sha256_file(snapshot)
    if actual_sha256 != str(expected_sha256):
        raise RuntimeError(
            "training distribution env snapshot SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    import yaml
    from isaaclab.utils.io import dump_yaml

    # The task class can legitimately drift after a run has completed.  Merely
    # comparing today's defaults against params/env.yaml would either reject an
    # exact historical inspection or, worse, run today's distribution.  Restore
    # the serialized config first, then apply the one user-authorized override.
    requested_num_envs = int(env_cfg.scene.num_envs)
    # Isaac Lab's own dump contains Python tuple/slice tags (not accepted by
    # FullLoader).  UnsafeLoader is restricted here to this exact SHA-bound,
    # local training artifact; a changed file is rejected above before parsing.
    saved_config = yaml.load(snapshot.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)
    if not isinstance(saved_config, dict):
        raise RuntimeError("training distribution env snapshot did not decode to a mapping")
    # Optional scalar fields can be ``None`` in today's task defaults but
    # concrete in the training artifact.  Isaac Lab's strict merge rejects the
    # type transition, so materialize only those saved leaves first.  Missing
    # config objects remain fail-closed rather than being replaced by dicts.
    _prepare_optional_config_leaves_for_restore(env_cfg, saved_config)
    try:
        env_cfg.from_dict(saved_config)
    except Exception as exc:
        raise RuntimeError(
            "could not restore the hash-bound training environment snapshot into the active task"
        ) from exc
    env_cfg.scene.num_envs = requested_num_envs
    env_cfg.scene.terrain.num_envs = requested_num_envs

    with tempfile.NamedTemporaryFile(prefix="highstep_auto_env_", suffix=".yaml", delete=False) as handle:
        runtime_path = Path(handle.name)
    try:
        dump_yaml(str(runtime_path), env_cfg)
        saved_tree = yaml.load(snapshot.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        runtime_tree = yaml.load(runtime_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    finally:
        runtime_path.unlink(missing_ok=True)
    if not isinstance(saved_tree, dict) or not isinstance(runtime_tree, dict):
        raise RuntimeError("training distribution env snapshot did not decode to a mapping")
    for path in _TRAINING_DISTRIBUTION_ALLOWED_CONFIG_DIFFS:
        _drop_nested_mapping_value(saved_tree, path)
        _drop_nested_mapping_value(runtime_tree, path)
    differences = _first_mapping_differences(saved_tree, runtime_tree)
    if differences:
        raise RuntimeError(
            "runtime task config does not match the saved training distribution; "
            f"first_differences={differences}"
        )
    projection = json.dumps(runtime_tree, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "snapshot_path": str(snapshot),
        "snapshot_sha256": actual_sha256,
        "runtime_projection_sha256": hashlib.sha256(projection).hexdigest(),
        "allowed_override": "scene.num_envs and scene.terrain.num_envs only",
        "config_match": True,
    }


def _numeric_distribution(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().float().flatten().cpu().numpy()
    return {
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "mean": float(np.mean(array)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _training_distribution_initial_snapshot(env, binding: dict[str, object]) -> dict[str, object]:
    """Summarize the untouched training reset distribution before the first policy step."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    env_origins = unwrapped.scene.env_origins
    root_pos = robot.data.root_pos_w
    root_rel = root_pos - env_origins
    quat = robot.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
    )
    command = unwrapped.command_manager.get_command("base_velocity")

    terrain = unwrapped.scene.terrain
    level_values, level_counts = torch.unique(terrain.terrain_levels.detach(), return_counts=True)
    type_values, type_counts = torch.unique(terrain.terrain_types.detach(), return_counts=True)
    generator_cfg = unwrapped.cfg.scene.terrain.terrain_generator
    terrain_name_by_column: dict[int, str] = {}
    if generator_cfg is not None and getattr(generator_cfg, "sub_terrains", None):
        names = list(generator_cfg.sub_terrains.keys())
        proportions = np.asarray(
            [float(generator_cfg.sub_terrains[name].proportion) for name in names], dtype=np.float64
        )
        proportions /= proportions.sum()
        cumulative = np.cumsum(proportions)
        for column in range(int(generator_cfg.num_cols)):
            index = int(np.min(np.where(column / int(generator_cfg.num_cols) + 0.001 < cumulative)[0]))
            terrain_name_by_column[column] = names[index]
    type_count_by_name: dict[str, int] = {}
    for value, count in zip(type_values.cpu().tolist(), type_counts.cpu().tolist()):
        label = terrain_name_by_column.get(int(value), f"column_{int(value)}")
        type_count_by_name[label] = type_count_by_name.get(label, 0) + int(count)

    return {
        "kind": "highstep_training_distribution_initial_snapshot",
        "num_envs": int(unwrapped.num_envs),
        "binding": binding,
        "terrain_level_counts": {
            str(int(value)): int(count)
            for value, count in zip(level_values.cpu().tolist(), level_counts.cpu().tolist())
        },
        "terrain_type_counts": dict(sorted(type_count_by_name.items())),
        "env_origin_x": _numeric_distribution(env_origins[:, 0]),
        "env_origin_y": _numeric_distribution(env_origins[:, 1]),
        "env_origin_z": _numeric_distribution(env_origins[:, 2]),
        "root_relative_x": _numeric_distribution(root_rel[:, 0]),
        "root_relative_y": _numeric_distribution(root_rel[:, 1]),
        "root_relative_z": _numeric_distribution(root_rel[:, 2]),
        "root_yaw_rad": _numeric_distribution(yaw),
        "command_vx": _numeric_distribution(command[:, 0]),
        "command_vy": _numeric_distribution(command[:, 1]),
        "command_wz": _numeric_distribution(command[:, 2]),
    }


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
        # Keep the order explicit: the v1.12.2 gate is FL-only and must never
        # inherit an implementation-defined body ordering from a multi-pattern
        # ``find_bodies`` call.
        self.front_asset_ids = [
            int(self.asset.find_bodies([name])[0][0]) for name in ("FL_foot", "FR_foot")
        ]
        self.rear_asset_ids = [
            int(self.asset.find_bodies([name])[0][0]) for name in ("RL_foot", "RR_foot")
        ]
        self.all_asset_ids = self.front_asset_ids + self.rear_asset_ids
        self.front_contact_ids = [
            int(self.contact_sensor.find_bodies([name])[0][0])
            for name in ("FL_foot", "FR_foot")
        ]
        self.rear_contact_ids = [
            int(self.contact_sensor.find_bodies([name])[0][0])
            for name in ("RL_foot", "RR_foot")
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
        self.fl_forbidden_surface_contact = False
        self.fl_forbidden_surface_first_step: int | None = None
        self.fl_forbidden_surface_contact_samples = 0
        self.fl_forbidden_surface_contact_streak = 0
        self.fl_forbidden_surface_contact_max_consecutive = 0
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
        # A top-surface support force is allowed.  Any FL force at the near
        # vertical face, including the narrow band immediately under the top
        # lip, is a monotonic failure event even if the policy later recovers.
        # The height/force test excludes an ordinary upward top load while
        # retaining a vertical or downward underside load near the edge.
        front_wall_contact = (
            contact[:2]
            & (torch.abs(outward[:2] - half_width) <= 0.14)
            & (foot_pos[:2, 2] > low_z + 0.04)
            & (foot_pos[:2, 2] < top_z + 0.03)
            & ((foot_pos[:2, 2] < top_z - 0.01) | (upward_ratio[:2] < 0.45))
        )
        fl_forbidden_contact_now = bool(front_wall_contact[0].item())
        if fl_forbidden_contact_now:
            self.fl_forbidden_surface_contact = True
            self.fl_forbidden_surface_contact_samples += 1
            self.fl_forbidden_surface_contact_streak += 1
            if self.fl_forbidden_surface_first_step is None:
                self.fl_forbidden_surface_first_step = int(step)
        else:
            self.fl_forbidden_surface_contact_streak = 0
        self.fl_forbidden_surface_contact_max_consecutive = max(
            self.fl_forbidden_surface_contact_max_consecutive,
            self.fl_forbidden_surface_contact_streak,
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
        rear_fore_aft_error = float(
            torch.abs(rear_pos_b[0, 0] - rear_pos_b[1, 0]).detach().cpu()
        )
        rear_lateral_center_error = float(torch.abs(rear_y[0] + rear_y[1]).detach().cpu())

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
            "rear_fore_aft_error": rear_fore_aft_error,
            "rear_lateral_center_error": rear_lateral_center_error,
            "rear_bilateral_contact": float(torch.all(rear_contact).item()),
            "rear_slip_contact": rear_slip_contact,
            "rear_bilateral_top_contact": float(torch.all(rear_top_contact).item()),
            "rear_slip_top": rear_slip_top,
            "front_bilateral_top_contact": float(torch.all(front_top_contact).item()),
            "front_single_step_contact": float(front_single),
            "front_wall_contact": float(torch.any(front_wall_contact).item()),
            "fl_forbidden_surface_contact": float(fl_forbidden_contact_now),
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
        critical_fore_aft = [sample["rear_fore_aft_error"] for sample in critical]
        critical_lateral_center = [sample["rear_lateral_center_error"] for sample in critical]
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
            "schema_version": 8,
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
            "fl_vertical_riser_or_top_lip_underside_contact": (
                self.fl_forbidden_surface_contact
            ),
            "fl_vertical_riser_or_top_lip_underside_first_step": (
                self.fl_forbidden_surface_first_step
            ),
            "fl_vertical_riser_or_top_lip_underside_contact_samples": (
                self.fl_forbidden_surface_contact_samples
            ),
            "fl_vertical_riser_or_top_lip_underside_max_consecutive": (
                self.fl_forbidden_surface_contact_max_consecutive
            ),
            "terminated_early": self.terminated_early,
            "termination_step": self.termination_step,
            "approach_rear_width_median": approach_width_median,
            "critical_rear_width_q05": critical_width_q05,
            "critical_rear_min_abs_y_q05": _safe_quantile(critical_center, 0.05),
            "critical_rear_width_drop": width_drop,
            "approach_rear_fore_aft_error_q95": _safe_quantile(
                [sample["rear_fore_aft_error"] for sample in approach], 0.95
            ),
            "approach_rear_lateral_center_error_q95": _safe_quantile(
                [sample["rear_lateral_center_error"] for sample in approach], 0.95
            ),
            "critical_rear_fore_aft_error_q95": _safe_quantile(critical_fore_aft, 0.95),
            "critical_rear_lateral_center_error_q95": _safe_quantile(
                critical_lateral_center, 0.95
            ),
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


class _HighstepV1123FrozenPreflightTracker:
    """Read-only proof that reward is dense before and latched after release."""

    def __init__(self, env, eval_tracker: _HighstepEvalTracker):
        self.env = env.unwrapped
        self.eval_tracker = eval_tracker
        self.asset = self.env.scene["robot"]
        self.frames = []
        self.reference = []
        self.first_crossing_step = None

    @staticmethod
    def _scalar(value):
        return float(value[0].detach().cpu())

    def sample(self, step: int) -> None:
        debug = getattr(self.env, "_highstep_v1123_preflight_frame", None)
        context = self.eval_tracker.context
        if not isinstance(debug, dict) or context is None:
            raise RuntimeError("v1.12.3 frozen preflight reward frame was not captured")
        fl_id = self.eval_tracker.front_asset_ids[0]
        fl_pos = self.asset.data.body_pos_w[0, fl_id, :]
        approach = context["approach"]
        lateral = context["lateral"]
        origin_xy = context["origin_xy"][0]
        top_z = float(context["top_z"][0].detach().cpu())
        low_z = float(context["low_z"][0].detach().cpu())
        half_width = float(context["half_width"])
        rel_xy = fl_pos[:2] - origin_xy
        outward = float(torch.sum(rel_xy * approach).detach().cpu())
        lateral_pos = float(torch.sum(rel_xy * lateral).detach().cpu())
        fl_z = float(fl_pos[2].detach().cpu())

        fl_force = self.eval_tracker._contact_force_vectors()[0]
        force_norm = float(torch.linalg.norm(fl_force).detach().cpu())
        upward = max(0.0, float(fl_force[2].detach().cpu()))
        upward_ratio = upward / max(force_norm, 1.0e-6)
        inside_top = abs(outward) <= half_width + 0.04 and abs(lateral_pos) <= half_width + 0.04
        allowed_top_support = bool(
            upward > 5.0
            and upward_ratio >= 0.45
            and inside_top
            and top_z - 0.04 <= fl_z <= top_z + 0.09
        )
        first_forbidden = self.eval_tracker.fl_forbidden_surface_first_step
        no_prior_forbidden = first_forbidden is None or first_forbidden == int(step)
        command_x = float(
            self.env.command_manager.get_command("base_velocity")[0, 0].detach().cpu()
        )
        in_reference = bool(
            abs(outward - half_width) <= 0.30
            and abs(lateral_pos) <= half_width + 0.04
            and low_z + 0.02 < fl_z < top_z + 0.10
            and top_z - low_z >= 0.06
            and command_x >= 0.08
            and not allowed_top_support
            and no_prior_forbidden
            and fl_z <= top_z + 0.04
        )
        crossing = self._scalar(debug["fl_clearance"]) >= 0.04
        if crossing and self.first_crossing_step is None:
            self.first_crossing_step = int(step)
        row = {
            "step": int(step),
            "final_product": self._scalar(debug["final_product"]),
            "active_latch": self._scalar(debug["active_latch"]),
            "lift_score": self._scalar(debug["lift_score"]),
            "retraction_score": self._scalar(debug["retraction_score"]),
            "fl_clearance": self._scalar(debug["fl_clearance"]),
            "fl_body_x": self._scalar(debug["fl_body_x"]),
            "fl_world_z": self._scalar(debug["fl_world_z"]),
            "crossing": crossing,
            "in_reference_window": in_reference,
        }
        self.frames.append(row)
        if in_reference:
            self.reference.append(row)

    @staticmethod
    def _stats(rows):
        values = np.asarray([float(row["final_product"]) for row in rows], dtype=np.float64)
        return {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "q50": float(np.quantile(values, 0.50)) if values.size else None,
            "q95": float(np.quantile(values, 0.95)) if values.size else None,
            "nonzero_ratio": float(np.mean(values > 1.0e-12)) if values.size else None,
        }

    def summary(self):
        post = [
            row for row in self.frames
            if self.first_crossing_step is not None and int(row["step"]) >= self.first_crossing_step
        ]
        return {
            "schema_version": 1,
            "kind": "highstep_v1123_single_seed_frozen_preflight",
            "reference_window_definition": (
                "Post-step FL frames from first entry into the near-riser swing corridor through "
                "the first forbidden contact: |FL_outward-near_edge|<=0.30 m, lateral inside "
                "platform+0.04 m, low_z+0.02<FL_z<top_z+0.10, physical platform height>=0.06 m, "
                "cmd_x>=0.08, no allowed FL top support or prior forbidden contact, and "
                "FL_z<=top_z+0.04 m."
            ),
            "loop_frames": len(self.frames),
            "reference_window": self._stats(self.reference),
            "reference_steps": [int(row["step"]) for row in self.reference],
            "first_forbidden_contact_step": self.eval_tracker.fl_forbidden_surface_first_step,
            "first_crossing_step": self.first_crossing_step,
            "post_first_crossing_frame_count": len(post),
            "post_first_crossing_all_strict_zero": bool(
                post and all(float(row["final_product"]) == 0.0 for row in post)
            ),
            "post_first_crossing_nonzero_steps": [
                int(row["step"]) for row in post if float(row["final_product"]) != 0.0
            ],
            "training_calls": {
                "runner_learn": 0,
                "backward": 0,
                "optimizer_step": 0,
                "checkpoint_write": 0,
            },
            "frames": self.frames,
        }


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

    if mode == "static_side_top":
        # This mode is deliberately narrow: configure once in world/terrain
        # coordinates, then never consult the robot root again.  Existing
        # camera modes retain their historical follow behavior below.
        if getattr(env.unwrapped, "_highstep_static_camera_configured", False):
            return
        origin = env.unwrapped.scene.env_origins[0]
        camera_device = env.unwrapped.device
        eye = origin + torch.tensor(
            [-2.45, -2.35, 2.35], dtype=torch.float32, device=camera_device
        )
        lookat = origin + torch.tensor(
            [-0.85, 0.0, 0.30], dtype=torch.float32, device=camera_device
        )
        env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
        env.unwrapped.viewport_camera_controller.update_view_location(
            eye=eye.detach().cpu().numpy(), lookat=lookat.detach().cpu().numpy()
        )
        env.unwrapped._highstep_static_camera_configured = True
        print(
            "[HIGHSTEP_CAMERA] "
            f"mode=static_side_top env=0 world_origin={origin.detach().cpu().tolist()} "
            f"eye={eye.detach().cpu().tolist()} lookat={lookat.detach().cpu().tolist()}",
            flush=True,
        )
        return

    robot = env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0]
    root_quat = robot.data.root_quat_w[0]

    if mode == "static_robot_side":
        if getattr(env.unwrapped, "_highstep_static_robot_camera_configured", False):
            return
        eye_offset = torch.tensor(
            [-1.2, -2.1, 1.45], dtype=torch.float32, device=env.device
        )
        lookat_offset = torch.tensor(
            [0.45, 0.0, 0.25], dtype=torch.float32, device=env.device
        )
        eye = math_utils.transform_points(
            eye_offset.unsqueeze(0), pos=root_pos.unsqueeze(0), quat=root_quat.unsqueeze(0)
        ).squeeze(0)
        lookat = math_utils.transform_points(
            lookat_offset.unsqueeze(0), pos=root_pos.unsqueeze(0), quat=root_quat.unsqueeze(0)
        ).squeeze(0)
        env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
        env.unwrapped.viewport_camera_controller.update_view_location(
            eye=eye.detach().cpu().numpy(), lookat=lookat.detach().cpu().numpy()
        )
        env.unwrapped._highstep_static_robot_camera_configured = True
        print(
            "[HIGHSTEP_CAMERA] "
            f"mode=static_robot_side env=0 root={root_pos.detach().cpu().tolist()} "
            f"eye={eye.detach().cpu().tolist()} lookat={lookat.detach().cpu().tolist()}",
            flush=True,
        )
        return

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
    recorded_command_fields = {
        "--recorded_command_trace": args_cli.recorded_command_trace,
        "--recorded_command_trace_sha256": args_cli.recorded_command_trace_sha256,
        "--recorded_command_episode_id": args_cli.recorded_command_episode_id,
        "--recorded_command_source_manifest": args_cli.recorded_command_source_manifest,
        "--recorded_command_source_manifest_sha256": (
            args_cli.recorded_command_source_manifest_sha256
        ),
    }
    recorded_command_present = {
        name: value is not None for name, value in recorded_command_fields.items()
    }
    if any(recorded_command_present.values()) and not all(recorded_command_present.values()):
        missing = [name for name, present in recorded_command_present.items() if not present]
        raise ValueError(
            "recorded command replay requires all binding arguments; missing " + str(missing)
        )
    recorded_command_replay = None
    recorded_command_source_manifest_path = None
    recorded_command_source_manifest_sha256 = None
    recorded_command_source_manifest = None
    b300_hybrid_behavior = bool(args_cli.b300_hybrid_behavior_preregistration)
    if b300_hybrid_behavior != bool(
        args_cli.b300_hybrid_behavior_preregistration_sha256
    ):
        raise ValueError("B300 hybrid behavior preregistration path and SHA must be supplied together")
    b300_hybrid_behavior_preregistration = None
    if b300_hybrid_behavior:
        prereg_path = Path(args_cli.b300_hybrid_behavior_preregistration).expanduser().resolve()
        if not prereg_path.is_file():
            raise FileNotFoundError(f"B300 hybrid behavior preregistration not found: {prereg_path}")
        prereg_sha = _sha256_file(prereg_path)
        if prereg_sha != args_cli.b300_hybrid_behavior_preregistration_sha256:
            raise RuntimeError("B300 hybrid behavior preregistration SHA256 mismatch")
        b300_hybrid_behavior_preregistration = json.loads(prereg_path.read_text(encoding="utf-8"))
        if not (
            b300_hybrid_behavior_preregistration.get("kind")
            == "highstep_b300_canonical_hybrid_prior_latent_preregistration"
            and b300_hybrid_behavior_preregistration.get("workflow_id")
            == "highstep_b300_canonical_hybrid_prior_latent_20260718"
            and args_cli.task
            == "RobotLab-Isaac-Velocity-HighstepB300CanonicalHybridStudentNoPrior-ArcdogAdjustableLeg-v0"
        ):
            raise RuntimeError("B300 hybrid behavior authority/task mismatch")
    b300_hybrid_narrow = bool(args_cli.b300_hybrid_narrow_route_amendment)
    if b300_hybrid_narrow != bool(args_cli.b300_hybrid_narrow_route_amendment_sha256):
        raise ValueError("B300 narrow route path and SHA must be supplied together")
    if b300_hybrid_narrow:
        if not b300_hybrid_behavior:
            raise RuntimeError("B300 narrow route requires the B300 Student behavior authority")
        route_path = Path(args_cli.b300_hybrid_narrow_route_amendment).expanduser().resolve()
        if _sha256_file(route_path) != args_cli.b300_hybrid_narrow_route_amendment_sha256:
            raise RuntimeError("B300 narrow route SHA mismatch")
        route = json.loads(route_path.read_text(encoding="utf-8"))
        requested_scenario = {
            "name": next(
                (
                    str(row["name"])
                    for row in route.get("scenarios", [])
                    if abs(float(row["gap_m"]) - float(args_cli.front_step_eval_edge_gap)) <= 1.0e-9
                    and abs(float(row["lateral_m"]) - float(args_cli.front_step_eval_lateral_offset)) <= 1.0e-9
                    and abs(float(row["yaw_deg"]) - float(args_cli.front_step_eval_yaw_offset_deg)) <= 1.0e-9
                    and int(row["joint_delta_sign"]) == int(args_cli.b300_hybrid_initial_joint_delta_sign)
                ),
                "",
            ),
            "gap_m": float(args_cli.front_step_eval_edge_gap),
            "lateral_m": float(args_cli.front_step_eval_lateral_offset),
            "yaw_deg": float(args_cli.front_step_eval_yaw_offset_deg),
            "joint_delta_sign": int(args_cli.b300_hybrid_initial_joint_delta_sign),
        }
        if not (
            route.get("kind") == "highstep_b300_hybrid_dagger_route_amendment"
            and route.get("workflow_id") == "highstep_b300_canonical_hybrid_prior_latent_20260718"
            and route.get("preregistration_sha256")
            == args_cli.b300_hybrid_behavior_preregistration_sha256
            and requested_scenario["name"]
        ):
            raise RuntimeError("B300 narrow scenario is outside frozen route")
    if all(recorded_command_present.values()):
        if args_cli.fixed_velocity_command is not None:
            raise ValueError(
                "fixed_velocity_command and recorded command replay are mutually exclusive"
            )
        incompatible = {
            "--keyboard": args_cli.keyboard,
            "--se2_gamepad": args_cli.se2_gamepad,
            "--training_distribution": args_cli.training_distribution,
            "--phase_residual_reference": args_cli.phase_residual_reference is not None,
            "--phase_residual_dataset_output": args_cli.phase_residual_dataset_output is not None,
            "--phase_residual_behavior_output": args_cli.phase_residual_behavior_output is not None,
        }
        active_incompatible = [name for name, active in incompatible.items() if active]
        if active_incompatible:
            raise ValueError(
                "recorded command replay is a narrow single-preview path; remove incompatible "
                f"options: {active_incompatible}"
            )
        recorded_command_replay = RecordedCommandReplay(
            args_cli.recorded_command_trace,
            expected_sha256=args_cli.recorded_command_trace_sha256,
            episode_id=args_cli.recorded_command_episode_id,
        )
        recorded_command_source_manifest_path = Path(
            args_cli.recorded_command_source_manifest
        ).expanduser().resolve()
        if not recorded_command_source_manifest_path.is_file():
            raise FileNotFoundError(
                "recorded command source manifest not found: "
                f"{recorded_command_source_manifest_path}"
            )
        recorded_command_source_manifest_sha256 = _sha256_file(
            recorded_command_source_manifest_path
        )
        if (
            recorded_command_source_manifest_sha256
            != args_cli.recorded_command_source_manifest_sha256
        ):
            raise RuntimeError(
                "recorded command source manifest SHA256 mismatch: "
                f"expected={args_cli.recorded_command_source_manifest_sha256} "
                f"actual={recorded_command_source_manifest_sha256}"
            )
        recorded_command_source_manifest = json.loads(
            recorded_command_source_manifest_path.read_text(encoding="utf-8")
        )
        source_metadata = recorded_command_source_manifest.get("metadata", {})
        required_contract = {
            "task": (
                source_metadata.get("task") if b300_hybrid_behavior else args_cli.task
            ),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "terrain_level": args_cli.play_terrain_level,
            "terrain_type": args_cli.play_terrain_type,
            "front_step_eval_reset": args_cli.front_step_eval_reset,
            "front_step_eval_side": args_cli.front_step_eval_side,
            "front_step_eval_edge_gap": (
                source_metadata.get("front_step_eval_edge_gap")
                if b300_hybrid_narrow else float(args_cli.front_step_eval_edge_gap)
            ),
            "front_step_eval_lateral_offset": (
                source_metadata.get("front_step_eval_lateral_offset")
                if b300_hybrid_narrow else float(args_cli.front_step_eval_lateral_offset)
            ),
            "front_step_eval_yaw_offset_deg": (
                source_metadata.get("front_step_eval_yaw_offset_deg")
                if b300_hybrid_narrow else float(args_cli.front_step_eval_yaw_offset_deg)
            ),
            "eval_action_delay_steps": args_cli.eval_action_delay_steps,
        }
        mismatches = {
            name: {"source": source_metadata.get(name), "requested": requested}
            for name, requested in required_contract.items()
            if source_metadata.get(name) != requested
        }
        if mismatches:
            raise RuntimeError(
                "recorded command replay contract differs from source manifest: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if not args_cli.reset_after_play_terrain_selection:
            raise ValueError(
                "recorded command replay requires --reset_after_play_terrain_selection"
            )
        fixed_motion_data_only = (
            args_cli.fixed_motion_direct_action_mode == "oracle"
            or args_cli.fixed_motion_raw_action_mode is not None
            or args_cli.b300_hybrid_canonical_output is not None
            or b300_hybrid_behavior
            or args_cli.b300_hybrid_dagger_output is not None
        )
        if (not args_cli.video or not args_cli.record_joint_data) and not fixed_motion_data_only:
            raise ValueError("recorded command preview requires --video and --record_joint_data")
        if args_cli.highstep_gap_camera != "static_side_top":
            raise ValueError(
                "recorded command preview requires --highstep_gap_camera static_side_top"
            )
        if int(args_cli.num_envs or 0) != 1:
            raise ValueError("recorded command preview requires --num_envs 1")
        if not fixed_motion_data_only and int(args_cli.video_length) != recorded_command_replay.frame_count:
            raise ValueError(
                "recorded command video_length must equal the frozen episode frame count"
            )
        if (
            args_cli.play_max_steps is None
            or int(args_cli.play_max_steps) != recorded_command_replay.frame_count
        ):
            raise ValueError(
                "recorded command play_max_steps must equal the frozen episode frame count"
            )
        print(
            "[RECORDED_COMMAND_REPLAY] "
            + json.dumps(recorded_command_replay.summary(), sort_keys=True),
            flush=True,
        )

    training_distribution_binding = None
    if args_cli.training_distribution:
        incompatible = {
            "--keyboard": args_cli.keyboard,
            "--se2_gamepad": args_cli.se2_gamepad,
            "--debug": args_cli.debug,
            "--real-time": args_cli.real_time,
            "--record_joint_data": args_cli.record_joint_data,
            "--play_terrain_level": args_cli.play_terrain_level is not None,
            "--play_terrain_type": args_cli.play_terrain_type is not None,
            "--reset_after_play_terrain_selection": args_cli.reset_after_play_terrain_selection,
            "--front_step_eval_reset": args_cli.front_step_eval_reset,
            "--fixed_velocity_command": args_cli.fixed_velocity_command is not None,
            "--recorded_command_trace": args_cli.recorded_command_trace is not None,
            "--eval_action_delay_steps": args_cli.eval_action_delay_steps is not None,
            "--eval_effort_limits": args_cli.eval_effort_limits is not None,
            "--print_rear_width_metrics": args_cli.print_rear_width_metrics,
            "--highstep_gap_camera": args_cli.highstep_gap_camera != "none",
            "--seed": args_cli.seed is not None,
        }
        active_incompatible = [name for name, active in incompatible.items() if active]
        if active_incompatible:
            raise ValueError(
                "--training_distribution must preserve the saved training distribution; "
                f"remove incompatible options: {active_incompatible}"
            )
        if args_cli.num_envs is None or int(args_cli.num_envs) <= 0:
            raise ValueError("--training_distribution requires a positive explicit --num_envs")
        if not args_cli.training_distribution_env_snapshot or not args_cli.training_distribution_env_sha256:
            raise ValueError(
                "--training_distribution requires a hash-bound --training_distribution_env_snapshot "
                "and --training_distribution_env_sha256"
            )
    elif args_cli.training_distribution_env_snapshot or args_cli.training_distribution_env_sha256:
        raise ValueError("training distribution snapshot arguments require --training_distribution")

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
    if args_cli.highstep_0707_legacy_student_profile:
        agent_cfg = _apply_highstep_0707_legacy_student_profile(agent_cfg, args_cli.task)
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

    if args_cli.training_distribution:
        training_distribution_binding = _verify_training_distribution_config(
            env_cfg,
            args_cli.training_distribution_env_snapshot,
            args_cli.training_distribution_env_sha256,
        )
        print(
            "[TRAINING_DISTRIBUTION_BINDING] "
            + json.dumps(training_distribution_binding, sort_keys=True),
            flush=True,
        )

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
    # Manual play keeps its historical reduced/deterministic scene.  Automatic
    # distribution inspection preserves the saved training cfg verbatim; the
    # user-selected number of parallel envs is its only allowed config change.
    env_cfg.scene.num_envs = args_cli.num_envs
    if not args_cli.training_distribution:
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

    if not args_cli.training_distribution:
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
    if not args_cli.training_distribution:
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.command_levels = None

    fixed_velocity_command = tuple(float(v) for v in args_cli.fixed_velocity_command) if args_cli.fixed_velocity_command else None
    one_shot_obs_command_step = {"value": 0}
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

    if recorded_command_replay is not None:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True
        env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
        for group_name in ("policy", "estimator", "critic"):
            obs_group = getattr(env_cfg.observations, group_name, None)
            if obs_group is None:
                continue
            velocity_term = getattr(obs_group, "velocity_commands", None)
            if velocity_term is None:
                continue
            old_history_len = getattr(velocity_term, "history_length", 0)
            old_flatten = getattr(velocity_term, "flatten_history_dim", False)
            setattr(
                obs_group,
                "velocity_commands",
                ObsTerm(
                    func=lambda env, replay=recorded_command_replay: torch.tensor(
                        replay.command(), device=env.device, dtype=torch.float32
                    ).unsqueeze(0).repeat(env.num_envs, 1),
                    history_length=old_history_len,
                    flatten_history_dim=old_flatten,
                ),
            )

    if args_cli.fixed_motion_one_shot_preregistration is not None:
        # Bind command history before the environment is constructed.  This
        # makes reset-time 10-frame command history deterministic (all zeros)
        # and then exposes the current internal-clock command on every policy
        # inference step.  It does not read or replay an external trajectory.
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True
        env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
        for group_name in ("policy", "estimator", "critic"):
            obs_group = getattr(env_cfg.observations, group_name, None)
            if obs_group is None:
                continue
            velocity_term = getattr(obs_group, "velocity_commands", None)
            if velocity_term is None:
                continue
            old_history_len = getattr(velocity_term, "history_length", 0)
            old_flatten = getattr(velocity_term, "flatten_history_dim", False)
            setattr(
                obs_group,
                "velocity_commands",
                ObsTerm(
                    func=lambda env, clock=one_shot_obs_command_step: torch.tensor(
                        [
                            0.7200000286 if 21 <= int(clock["value"]) < 80 else 0.0,
                            0.0,
                            0.0,
                        ],
                        device=env.device,
                        dtype=torch.float32,
                    ).unsqueeze(0).repeat(env.num_envs, 1),
                    history_length=old_history_len,
                    flatten_history_dim=old_flatten,
                ),
            )

    manual_highstep_respawn_contract = bool(
        args_cli.keyboard
        and args_cli.front_step_eval_reset
        and not args_cli.keep_play_randomization
        and recorded_command_replay is None
    )
    manual_highstep_start_gate = (
        ManualHighstepStartGate() if manual_highstep_respawn_contract else None
    )

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

        def _manual_keyboard_command() -> torch.Tensor:
            command = controller.advance()
            if manual_highstep_start_gate is not None:
                command = manual_highstep_start_gate.filter_command(command)
            return command

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
                        func=lambda env: _manual_keyboard_command().unsqueeze(0).to(
                            env.device, dtype=torch.float32
                        ),
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

    if args_cli.highstep_0707_legacy_student_profile:
        actual_checkpoint_sha256 = _sha256_file(resume_path)
        if actual_checkpoint_sha256 != _HIGHSTEP_0707_STUDENT_CHECKPOINT_SHA256:
            raise RuntimeError(
                "0707 deployed Student checkpoint SHA256 mismatch: "
                f"{actual_checkpoint_sha256} != {_HIGHSTEP_0707_STUDENT_CHECKPOINT_SHA256}"
            )
        print(
            "[0707_REPLAY] Verified deployed Student checkpoint SHA256: "
            f"{actual_checkpoint_sha256}",
            flush=True,
        )

    if recorded_command_replay is not None:
        actual_checkpoint_sha256 = _sha256_file(resume_path)
        expected_checkpoint_sha256 = recorded_command_source_manifest["metadata"].get(
            "checkpoint_sha256"
        )
        if b300_hybrid_behavior:
            if (
                b300_hybrid_behavior_preregistration.get("teacher_sha256")
                != expected_checkpoint_sha256
            ):
                raise RuntimeError("B300 frozen command source is not the preregistered Teacher")
            checkpoint_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            recovery = (
                checkpoint_payload.get("infos", {})
                .get("robot_lab_algorithm_checkpoint_state", {})
                .get("student_recovery", {})
            )
            if not (
                recovery.get("stage") == "B300_CANONICAL_HYBRID"
                and recovery.get("preregistration_sha256")
                == args_cli.b300_hybrid_behavior_preregistration_sha256
                and int(recovery.get("effective_update_count", -1)) >= 1
            ):
                raise RuntimeError("B300 Student checkpoint recovery binding is invalid")
            print(
                "[B300_HYBRID_BEHAVIOR_BINDING] "
                + json.dumps(
                    {
                        "checkpoint": os.path.realpath(resume_path),
                        "checkpoint_sha256": actual_checkpoint_sha256,
                        "effective_updates": int(recovery["effective_update_count"]),
                        "preregistration_sha256": args_cli.b300_hybrid_behavior_preregistration_sha256,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elif actual_checkpoint_sha256 != expected_checkpoint_sha256:
            raise RuntimeError(
                "recorded command checkpoint SHA256 mismatch: "
                f"expected={expected_checkpoint_sha256} actual={actual_checkpoint_sha256}"
            )

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.highstep_v1123_frozen_preflight_output:
        env.unwrapped._highstep_v1123_preflight_capture = True

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

    # RecordVideo captures an initial frame as soon as it is wrapped.  Install
    # the world-fixed view before that capture so frame zero cannot be an empty
    # or default-camera image.  The one-shot guard prevents every later call
    # from changing eye/lookat.
    if args_cli.video and args_cli.highstep_gap_camera == "static_side_top":
        _update_highstep_gap_camera(env, args_cli.highstep_gap_camera)

    # wrap for video recording
    if args_cli.video:
        if args_cli.video_output_dir:
            video_folder = Path(args_cli.video_output_dir).expanduser().resolve()
            if video_folder.exists() and any(video_folder.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite non-empty video output: {video_folder}"
                )
        else:
            video_folder = Path(log_dir) / "videos" / "play"
        video_kwargs = {
            "video_folder": str(video_folder),
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
            terrain_curriculum_enabled=bool(
                args_cli.training_distribution and terrain_schedule_cfg_for_manifest is not None
            ),
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
    if recorded_command_replay is not None:
        source_joint_names = list(
            recorded_command_source_manifest.get("joint_names_in_action_order", [])
        )
        if len(source_joint_names) != 16:
            raise RuntimeError("recorded reset state requires the frozen 16-joint action order")
        reset_joint_pos, reset_joint_vel = recorded_command_replay.pre_reset_joint_state(
            source_joint_names
        )
        reset_robot = env.unwrapped.scene["robot"]
        reset_joint_id_by_name = {
            name: index for index, name in enumerate(reset_robot.joint_names)
        }
        missing_reset_joints = [
            name for name in source_joint_names if name not in reset_joint_id_by_name
        ]
        if missing_reset_joints:
            raise RuntimeError(
                f"recorded reset joints are absent from the live robot: {missing_reset_joints}"
            )
        reset_joint_ids = [reset_joint_id_by_name[name] for name in source_joint_names]
        reset_device = reset_robot.data.joint_pos.device
        reset_dtype = reset_robot.data.joint_pos.dtype
        reset_robot.write_joint_state_to_sim(
            torch.tensor(reset_joint_pos, device=reset_device, dtype=reset_dtype).reshape(1, 16)
            + (
                torch.tensor(
                    [0.010, -0.010, 0.010, -0.010, -0.008, 0.008, -0.008, 0.008,
                     0.006, -0.006, 0.006, -0.006, 0.001, -0.001, 0.001, -0.001],
                    device=reset_device,
                    dtype=reset_dtype,
                ).reshape(1, 16)
                * int(args_cli.b300_hybrid_initial_joint_delta_sign)
            ),
            torch.tensor(
                reset_joint_vel, device=reset_device, dtype=reset_dtype
            ).reshape(1, 16),
            joint_ids=reset_joint_ids,
        )
        env.unwrapped.sim.forward()
        print(
            "[RECORDED_RESET_STATE] "
            + json.dumps(
                {
                    "restored": True,
                    "joint_count": len(source_joint_names),
                    "source_control_step": recorded_command_replay.summary()[
                        "pre_reset_source_control_step"
                    ],
                    "source_episode_id": recorded_command_replay.summary()[
                        "pre_reset_source_episode_id"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
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
        if manual_highstep_respawn_contract:
            _apply_manual_highstep_respawn(
                env,
                args_cli.front_step_eval_side,
                args_cli.front_step_eval_distance,
                args_cli.front_step_eval_edge_gap,
                platform_width,
                args_cli.front_step_eval_lateral_offset,
                args_cli.front_step_eval_yaw_offset_deg,
                args_cli.play_terrain_type,
            )
            print(
                "RESET RECOVERING / 请勿操作 | "
                f"minimum={manual_highstep_start_gate.min_hold_steps} control steps "
                f"({manual_highstep_start_gate.min_hold_steps * float(env.unwrapped.step_dt):.2f} s) "
                f"and {manual_highstep_start_gate.stable_steps_required} consecutive stable steps; "
                "keyboard motion input is ignored.",
                flush=True,
            )
        else:
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
    phase_residual_reference = None
    phase_residual_reference_controller = None
    phase_residual_dataset_collector = None
    phase_residual_dataset_joint_ids = None
    phase_residual_dataset_rear_top_streak = None
    phase_residual_dataset_rear_confirmed = None
    phase_residual_dataset_hold_streak = None
    phase_residual_dataset_hold_success = None
    phase_residual_hybrid_controller = None
    phase_residual_behavior_output_dir = None
    phase_residual_behavior_rear_top_streak = None
    phase_residual_behavior_rear_confirmed = None
    phase_residual_behavior_hold_streak = None
    phase_residual_behavior_hold_success = None
    phase_residual_behavior_terminated = None
    phase_residual_behavior_joint_ids = None
    phase_residual_dagger_collector = None
    phase_residual_dataset_settle_steps = 0
    phase_residual_timing_amendment_path = None
    phase_residual_timing_amendment_sha256 = None
    phase_controller_inference_path = None
    fixed_motion_oracle_collector = None
    fixed_motion_target_adapter = None
    fixed_motion_student_controller = None
    fixed_motion_action_term = None
    fixed_motion_pending_teacher_target = None
    fixed_motion_current_vx = torch.zeros(env.num_envs, device=env.device)
    fixed_motion_rear_top_streak = None
    fixed_motion_rear_confirmed = None
    fixed_motion_hold_streak = None
    fixed_motion_hold_success = None
    fixed_motion_terminated = None
    fixed_motion_failure = None
    raw_motion_collector = None
    raw_motion_student_controller = None
    raw_motion_dagger_collector = None
    raw_motion_action_term = None
    raw_motion_current_vx = torch.zeros(env.num_envs, device=env.device)
    raw_motion_rear_top_streak = None
    raw_motion_rear_confirmed = None
    raw_motion_hold_streak = None
    raw_motion_hold_success = None
    raw_motion_terminated = None
    raw_motion_failure = None
    b300_hybrid_collector = None
    b300_hybrid_dagger_collector = None
    b300_hybrid_current_command = torch.zeros((env.num_envs, 3), device=env.device)
    if args_cli.b300_hybrid_canonical_output is not None:
        if not (
            args_cli.b300_hybrid_collection_preregistration
            and args_cli.b300_hybrid_collection_preregistration_sha256
            and recorded_command_replay is not None
            and args_cli.front_step_eval_reset
            and int(env.num_envs) == 1
            and int(args_cli.play_max_steps or -1) == 138
            and int(args_cli.eval_action_delay_steps or 0) == 0
        ):
            raise ValueError(
                "B300 canonical collection requires hash-bound preregistration, recorded "
                "command replay, frozen reset, num_envs=1, 138 steps and delay=0"
            )
        b300_hybrid_collector = CanonicalTensorCollector(
            args_cli.b300_hybrid_canonical_output,
            args_cli.b300_hybrid_collection_preregistration,
            args_cli.b300_hybrid_collection_preregistration_sha256,
            policy_module=policy_nn,
            env=env,
        )
    if args_cli.b300_hybrid_dagger_output is not None:
        if not (
            args_cli.b300_hybrid_dagger_stage_manifest
            and args_cli.b300_hybrid_dagger_stage_manifest_sha256
            and b300_hybrid_behavior
            and recorded_command_replay is not None
            and int(env.num_envs) == 1
            and int(args_cli.play_max_steps or -1) == 138
            and int(args_cli.eval_action_delay_steps or 0) == 0
        ):
            raise ValueError("B300 DAgger requires exact behavior authority, stage manifest, one env, 138 steps and delay=0")
        b300_hybrid_dagger_collector = DaggerTensorCollector(
            args_cli.b300_hybrid_dagger_output,
            args_cli.b300_hybrid_dagger_stage_manifest,
            args_cli.b300_hybrid_dagger_stage_manifest_sha256,
            policy_module=policy_nn,
            env=env,
        )
    fixed_motion_args = (
        args_cli.fixed_motion_preregistration,
        args_cli.fixed_motion_preregistration_sha256,
        args_cli.fixed_motion_output,
    )
    if args_cli.fixed_motion_direct_action_mode is not None:
        if not all(value is not None for value in fixed_motion_args):
            raise ValueError("fixed-motion mode requires preregistration/SHA and output")
        if not args_cli.front_step_eval_reset:
            raise ValueError("fixed-motion mode requires the frozen front-step reset")
        action_terms = getattr(env.unwrapped.action_manager, "_terms", {})
        fixed_motion_action_term = action_terms.get("joint_pos") if isinstance(action_terms, dict) else None
        if fixed_motion_action_term is None:
            raise RuntimeError("fixed-motion mode requires the production joint_pos action term")
        if args_cli.fixed_motion_direct_action_mode == "oracle":
            if int(env.num_envs) != 1:
                raise ValueError("fixed-motion safe oracle requires exactly one environment")
            fixed_motion_oracle_collector = OracleDatasetCollector(
                args_cli.fixed_motion_output,
                args_cli.fixed_motion_preregistration,
                args_cli.fixed_motion_preregistration_sha256,
                action_term=fixed_motion_action_term,
                device=env.device,
                dtype=env.unwrapped.scene["robot"].data.joint_pos.dtype,
            )
            fixed_motion_target_adapter = SafeMappedTargetAdapter(fixed_motion_action_term, 1)
        else:
            student_args = (
                args_cli.fixed_motion_training_manifest,
                args_cli.fixed_motion_training_manifest_sha256,
                args_cli.fixed_motion_wandb_verification,
                args_cli.fixed_motion_wandb_verification_sha256,
            )
            if not all(value is not None for value in student_args):
                raise ValueError("fixed-motion Student/DAgger mode requires training and W&B manifests with SHAs")
            if int(env.num_envs) != 15:
                raise ValueError("fixed-motion Student/DAgger gate requires exactly 15 environments")
            fixed_motion_student_controller = FixedMotionStudentController(
                training_manifest_path=args_cli.fixed_motion_training_manifest,
                training_manifest_sha256=args_cli.fixed_motion_training_manifest_sha256,
                wandb_verification_path=args_cli.fixed_motion_wandb_verification,
                wandb_verification_sha256=args_cli.fixed_motion_wandb_verification_sha256,
                preregistration_path=args_cli.fixed_motion_preregistration,
                preregistration_sha256=args_cli.fixed_motion_preregistration_sha256,
                environment_checkpoint_path=resume_path,
                action_term=fixed_motion_action_term,
                num_envs=15,
                device=env.device,
                dtype=env.unwrapped.scene["robot"].data.joint_pos.dtype,
            )
        fixed_motion_rear_top_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
        fixed_motion_rear_confirmed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        fixed_motion_hold_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
        fixed_motion_hold_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        fixed_motion_terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if args_cli.fixed_motion_raw_action_mode is not None:
        required = (
            args_cli.fixed_motion_raw_preregistration,
            args_cli.fixed_motion_raw_preregistration_sha256,
            args_cli.fixed_motion_raw_output,
        )
        if not all(value is not None for value in required):
            raise ValueError("raw-action mode requires preregistration/SHA and output")
        if not args_cli.front_step_eval_reset or int(env.num_envs) != 1:
            raise ValueError("raw-action fixed-motion runs require frozen reset and num_envs=1")
        terms = getattr(env.unwrapped.action_manager, "_terms", {})
        raw_motion_action_term = terms.get("joint_pos") if isinstance(terms, dict) else None
        if raw_motion_action_term is None:
            raise RuntimeError("raw-action mode requires the production joint_pos action term")
        robot_dtype = env.unwrapped.scene["robot"].data.joint_pos.dtype
        phase_only_action_term = None
        one_shot_enabled = False
        if args_cli.fixed_motion_phase_only_deployment_preregistration is not None:
            if args_cli.fixed_motion_phase_only_deployment_preregistration_sha256 is None:
                raise ValueError("phase-only deployment preregistration requires its SHA256")
            _, phase_only_authority = load_phase_only_deployment_authority(
                args_cli.fixed_motion_phase_only_deployment_preregistration,
                args_cli.fixed_motion_phase_only_deployment_preregistration_sha256,
            )
            if phase_only_authority["checkpoint_sha256"] != _sha256_file(
                phase_only_authority["checkpoint"]
            ):
                raise RuntimeError("phase-only deployment checkpoint binding changed")
            phase_only_action_term = raw_motion_action_term
        if args_cli.fixed_motion_one_shot_preregistration is not None:
            if args_cli.fixed_motion_one_shot_preregistration_sha256 is None:
                raise ValueError("one-shot deployment preregistration requires its SHA256")
            _, one_shot_authority = load_one_shot_deployment_authority(
                args_cli.fixed_motion_one_shot_preregistration,
                args_cli.fixed_motion_one_shot_preregistration_sha256,
            )
            if int(args_cli.play_max_steps or -1) != 138:
                raise ValueError("one-shot deployment requires exactly 138 play steps")
            one_shot_enabled = True
        if args_cli.fixed_motion_raw_action_mode == "smoke":
            raw_motion_collector = RawPassthroughCollector(
                args_cli.fixed_motion_raw_output,
                args_cli.fixed_motion_raw_preregistration,
                args_cli.fixed_motion_raw_preregistration_sha256,
                device=env.device,
                dtype=robot_dtype,
            )
        else:
            student_required = (
                args_cli.fixed_motion_raw_training_manifest,
                args_cli.fixed_motion_raw_training_manifest_sha256,
                args_cli.fixed_motion_raw_wandb_verification,
                args_cli.fixed_motion_raw_wandb_verification_sha256,
            )
            if not all(value is not None for value in student_required):
                raise ValueError("raw Student/DAgger mode requires training and W&B manifests with SHAs")
            raw_motion_student_controller = RawActionStudentController(
                training_manifest_path=args_cli.fixed_motion_raw_training_manifest,
                training_manifest_sha256=args_cli.fixed_motion_raw_training_manifest_sha256,
                wandb_verification_path=args_cli.fixed_motion_raw_wandb_verification,
                wandb_verification_sha256=args_cli.fixed_motion_raw_wandb_verification_sha256,
                preregistration_path=args_cli.fixed_motion_raw_preregistration,
                preregistration_sha256=args_cli.fixed_motion_raw_preregistration_sha256,
                environment_checkpoint_path=resume_path,
                num_envs=1,
                device=env.device,
                dtype=robot_dtype,
                phase_only_action_term=phase_only_action_term,
                one_shot=one_shot_enabled,
            )
            if args_cli.fixed_motion_raw_action_mode == "dagger":
                if args_cli.fixed_motion_raw_dagger_round is None:
                    raise ValueError("raw DAgger mode requires --fixed_motion_raw_dagger_round")
                raw_motion_dagger_collector = RawActionDaggerCollector(
                    args_cli.fixed_motion_raw_output,
                    args_cli.fixed_motion_raw_preregistration,
                    args_cli.fixed_motion_raw_preregistration_sha256,
                    args_cli.fixed_motion_raw_dagger_round,
                )
        raw_motion_rear_top_streak = torch.zeros(1, device=env.device, dtype=torch.int64)
        raw_motion_rear_confirmed = torch.zeros(1, device=env.device, dtype=torch.bool)
        raw_motion_hold_streak = torch.zeros(1, device=env.device, dtype=torch.int64)
        raw_motion_hold_success = torch.zeros(1, device=env.device, dtype=torch.bool)
        raw_motion_terminated = torch.zeros(1, device=env.device, dtype=torch.bool)
    if args_cli.phase_residual_reference_sha256 and not args_cli.phase_residual_reference:
        raise ValueError(
            "--phase_residual_reference_sha256 requires --phase_residual_reference"
        )
    if args_cli.phase_residual_reference:
        if not args_cli.phase_residual_reference_sha256:
            raise ValueError(
                "--phase_residual_reference requires --phase_residual_reference_sha256"
            )
        if not args_cli.front_step_eval_reset or int(env.num_envs) != 1:
            raise ValueError(
                "--phase_residual_reference requires --front_step_eval_reset and --num_envs 1"
            )
        record_joint_names, _ = _joint_recording_schema(env)
        action_terms = getattr(env.unwrapped.action_manager, "_terms", {})
        action_term = action_terms.get("joint_pos") if isinstance(action_terms, dict) else None
        if action_term is None:
            raise RuntimeError("phase-residual replay requires the joint_pos action term")
        robot = env.unwrapped.scene["robot"]
        joint_id_by_name = {name: index for index, name in enumerate(robot.joint_names)}
        joint_ids = [joint_id_by_name[name] for name in record_joint_names]
        phase_residual_reference = PhaseResidualReference(
            args_cli.phase_residual_reference,
            expected_sha256=args_cli.phase_residual_reference_sha256,
        )
        restore_source_state = bool(args_cli.phase_residual_reference_restore_source_state)
        if restore_source_state:
            if int(args_cli.phase_residual_reference_blend_steps) != 0:
                raise ValueError(
                    "--phase_residual_reference_restore_source_state requires "
                    "--phase_residual_reference_blend_steps 0"
                )
            if not phase_residual_reference.has_initial_full_state:
                raise RuntimeError(
                    "source-state replay requested but the reference has no complete root/joint state"
                )
            assert phase_residual_reference.initial_root_pose_cpu is not None
            assert phase_residual_reference.initial_root_velocity_world_cpu is not None
            assert phase_residual_reference.initial_measured_joint_vel_cpu is not None
            state_device = robot.data.joint_pos.device
            state_dtype = robot.data.joint_pos.dtype
            root_pose = phase_residual_reference.initial_root_pose_cpu.to(
                device=state_device, dtype=state_dtype
            ).reshape(1, 7)
            root_velocity = phase_residual_reference.initial_root_velocity_world_cpu.to(
                device=state_device, dtype=state_dtype
            ).reshape(1, 6)
            joint_position = phase_residual_reference.initial_measured_joint_pos_cpu.to(
                device=state_device, dtype=state_dtype
            ).reshape(1, 16)
            joint_velocity = phase_residual_reference.initial_measured_joint_vel_cpu.to(
                device=state_device, dtype=state_dtype
            ).reshape(1, 16)
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            robot.write_joint_state_to_sim(joint_position, joint_velocity, joint_ids=joint_ids)
            env.unwrapped.sim.forward()
            obs = env.get_observations()
            print(
                "[PHASE_RESIDUAL_SOURCE_STATE] "
                + json.dumps(
                    {
                        "restored": True,
                        "start_reference_index": 1,
                        "initial_joint_projection_count": (
                            phase_residual_reference.initial_measured_projection_count
                        ),
                        "initial_joint_projection_max_abs": (
                            phase_residual_reference.initial_measured_projection_max_abs
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        phase_residual_reference_controller = PhaseResidualReferenceController(
            phase_residual_reference,
            action_term=action_term,
            initial_joint_pos=robot.data.joint_pos[:, joint_ids],
            joint_names=record_joint_names,
            blend_steps=args_cli.phase_residual_reference_blend_steps,
            start_at_next_frame=restore_source_state,
        )
        print(
            "[PHASE_RESIDUAL_REFERENCE] "
            + json.dumps(
                {
                    "path": str(phase_residual_reference.path),
                    "sha256": phase_residual_reference.sha256,
                    "frames": phase_residual_reference.frame_count,
                    "blend_steps": int(args_cli.phase_residual_reference_blend_steps),
                    "restore_source_state": restore_source_state,
                    "start_reference_index": 1 if restore_source_state else 0,
                    "initial_joint_projection_count": (
                        phase_residual_reference.initial_measured_projection_count
                    ),
                    "initial_joint_projection_max_abs": (
                        phase_residual_reference.initial_measured_projection_max_abs
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dataset_args = (
        args_cli.phase_residual_dataset_output,
        args_cli.phase_residual_selected_reference_manifest,
        args_cli.phase_residual_selected_reference_manifest_sha256,
    )
    if any(value is not None for value in dataset_args) and not all(
        value is not None for value in dataset_args
    ):
        raise ValueError(
            "phase-residual dataset collection requires output, selected-reference manifest, "
            "and selected-reference manifest SHA together"
        )
    behavior_args = (
        args_cli.phase_residual_training_manifest,
        args_cli.phase_residual_training_manifest_sha256,
        args_cli.phase_residual_wandb_verification,
        args_cli.phase_residual_wandb_verification_sha256,
        args_cli.phase_residual_behavior_output,
        args_cli.phase_residual_preroll_manifest,
        args_cli.phase_residual_preroll_manifest_sha256,
    )
    if any(value is not None for value in behavior_args) and not all(
        value is not None for value in behavior_args
    ):
        raise ValueError(
            "phase-residual behavior gate requires training manifest/SHA, W&B verification/SHA, "
            "pre-roll manifest/SHA, and a new output directory together"
        )
    timing_args = (
        args_cli.phase_residual_source_timing_amendment,
        args_cli.phase_residual_source_timing_amendment_sha256,
    )
    if any(value is not None for value in timing_args) and not all(
        value is not None for value in timing_args
    ):
        raise ValueError("phase-residual source timing amendment and SHA must be supplied together")
    if (args_cli.phase_residual_dataset_output or args_cli.phase_residual_behavior_output) and not all(
        value is not None for value in timing_args
    ):
        raise ValueError("phase-residual dataset/behavior execution requires the frozen timing amendment")
    if all(value is not None for value in timing_args) and not (
        args_cli.phase_residual_dataset_output or args_cli.phase_residual_behavior_output
    ):
        raise ValueError("phase-residual timing amendment was provided without a dataset/behavior run")
    if args_cli.phase_residual_disable_residual and not args_cli.phase_residual_behavior_output:
        raise ValueError("--phase_residual_disable_residual requires a behavior gate")
    if args_cli.phase_direct_action and not args_cli.phase_residual_behavior_output:
        raise ValueError("--phase_direct_action requires the canonical fixed-condition behavior gate")
    if args_cli.phase_direct_action and args_cli.phase_residual_disable_residual:
        raise ValueError("direct-action and residual-disabled reference-only modes are mutually exclusive")
    dagger_args = (
        args_cli.phase_residual_dagger_output,
        args_cli.phase_residual_dagger_preregistration,
        args_cli.phase_residual_dagger_preregistration_sha256,
    )
    if any(value is not None for value in dagger_args) and not all(
        value is not None for value in dagger_args
    ):
        raise ValueError("phase-residual DAgger requires output, preregistration, and SHA together")
    if args_cli.phase_residual_dagger_output and not args_cli.phase_residual_behavior_output:
        raise ValueError("phase-residual DAgger requires the canonical behavior gate")
    if args_cli.phase_residual_dataset_output:
        if phase_residual_reference_controller is not None:
            raise ValueError("Teacher dataset collection cannot run reference replay")
        if int(env.num_envs) != 15:
            raise ValueError("phase-residual Teacher dataset collection requires --num_envs 15")
        phase_residual_dataset_settle_steps = 296
        if int(args_cli.play_max_steps or -1) != 416:
            raise ValueError("phase-residual Teacher dataset collection requires --play_max_steps 416")
        if not args_cli.front_step_eval_reset or args_cli.keep_play_randomization:
            raise ValueError(
                "phase-residual Teacher dataset requires deterministic front-step reset and no random events"
            )
        if args_cli.eval_action_delay_steps != 0:
            raise ValueError("phase-residual Teacher dataset requires action delay 0")
        if fixed_velocity_command != (0.72, 0.0, 0.0):
            raise ValueError("phase-residual Teacher dataset requires fixed command [0.72, 0, 0]")
        if args_cli.play_terrain_type != "box_hard" or int(args_cli.play_terrain_level or -1) != 9:
            raise ValueError("phase-residual Teacher dataset requires box_hard terrain level 9")
        selected_path = Path(args_cli.phase_residual_selected_reference_manifest).expanduser().resolve()
        selected_actual_sha = _sha256_file(selected_path)
        if selected_actual_sha != args_cli.phase_residual_selected_reference_manifest_sha256:
            raise RuntimeError(
                "selected-reference manifest SHA mismatch before dataset collection: "
                f"expected={args_cli.phase_residual_selected_reference_manifest_sha256} "
                f"actual={selected_actual_sha}"
            )
        selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_fixed = selected_payload.get("fixed_condition", {})
        if (
            selected_fixed.get("terrain_type") != "box_hard"
            or int(selected_fixed.get("terrain_level", -1)) != 9
            or selected_fixed.get("command") != [0.72, 0.0, 0.0]
        ):
            raise RuntimeError("selected-reference fixed-condition contract changed")
        prereg_binding = selected_payload.get("preregistration", {})
        prereg_path = Path(prereg_binding.get("path", "")).expanduser().resolve()
        if _sha256_file(prereg_path) != prereg_binding.get("sha256"):
            raise RuntimeError("selected-reference preregistration binding changed")
        prereg_payload = json.loads(prereg_path.read_text(encoding="utf-8"))
        teacher_binding = prereg_payload.get("unchanged_contract", {})
        if _sha256_file(resume_path) != teacher_binding.get("teacher_sha256"):
            raise RuntimeError("dataset checkpoint is not the frozen Teacher")
        if args_cli.task != "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123-ArcdogAdjustableLeg-v0":
            raise RuntimeError("dataset task is not the frozen production-prior Teacher task")

        phase_residual_timing_amendment_path = Path(
            args_cli.phase_residual_source_timing_amendment
        ).expanduser().resolve()
        phase_residual_timing_amendment_sha256 = _sha256_file(
            phase_residual_timing_amendment_path
        )
        if (
            phase_residual_timing_amendment_sha256
            != args_cli.phase_residual_source_timing_amendment_sha256
        ):
            raise RuntimeError("source-timing amendment SHA mismatch")
        timing_payload = json.loads(
            phase_residual_timing_amendment_path.read_text(encoding="utf-8")
        )
        timing_contract = timing_payload.get("corrected_timing_contract", {})
        if (
            timing_payload.get("status") != "frozen_before_dataset_retry"
            or timing_payload.get("selected_reference_manifest", {}).get("sha256")
            != selected_actual_sha
            or int(timing_contract.get("settle_steps_before_dataset", -1)) != 296
            or int(timing_contract.get("dataset_pre_active_steps", -1)) != 10
            or int(timing_contract.get("dataset_steps", -1)) != 120
            or int(timing_contract.get("total_play_steps", -1)) != 416
        ):
            raise RuntimeError("source-timing amendment contract changed")

        selected_reference_binding = selected_payload.get("selected_reference", {})
        dataset_reference = PhaseResidualReference(
            selected_reference_binding["path"],
            expected_sha256=selected_reference_binding["sha256"],
        )
        dataset_joint_names, _ = _joint_recording_schema(env)
        dataset_action_terms = getattr(env.unwrapped.action_manager, "_terms", {})
        dataset_action_term = (
            dataset_action_terms.get("joint_pos") if isinstance(dataset_action_terms, dict) else None
        )
        if dataset_action_term is None:
            raise RuntimeError("phase-residual Teacher dataset requires the joint_pos action term")
        dataset_robot = env.unwrapped.scene["robot"]
        dataset_joint_id_by_name = {
            name: index for index, name in enumerate(dataset_robot.joint_names)
        }
        phase_residual_dataset_joint_ids = [
            dataset_joint_id_by_name[name] for name in dataset_joint_names
        ]
        phase_residual_dataset_collector = PhaseResidualDatasetCollector(
            output_dir=args_cli.phase_residual_dataset_output,
            selected_manifest_path=selected_path,
            selected_manifest_sha256=selected_actual_sha,
            reference=dataset_reference,
            action_term=dataset_action_term,
            joint_names=dataset_joint_names,
            num_envs=int(env.num_envs),
        )
        phase_residual_dataset_rear_top_streak = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int64
        )
        phase_residual_dataset_rear_confirmed = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        phase_residual_dataset_hold_streak = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int64
        )
        phase_residual_dataset_hold_success = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        print(
            "[PHASE_RESIDUAL_DATASET] "
            + json.dumps(
                {
                    "output": str(phase_residual_dataset_collector.output_dir),
                    "rollouts": int(env.num_envs),
                    "settle_steps": phase_residual_dataset_settle_steps,
                    "dataset_steps": int(args_cli.play_max_steps) - phase_residual_dataset_settle_steps,
                    "selected_reference_manifest_sha256": selected_actual_sha,
                    "source_timing_amendment_sha256": phase_residual_timing_amendment_sha256,
                    "reference_sha256": dataset_reference.sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args_cli.phase_residual_behavior_output:
        if phase_residual_reference_controller is not None or phase_residual_dataset_collector is not None:
            raise ValueError("phase-residual behavior gate is mutually exclusive with replay/dataset modes")
        if int(env.num_envs) != 15 or int(args_cli.play_max_steps or -1) != 416:
            raise ValueError("phase-residual behavior gate requires --num_envs 15 --play_max_steps 416")
        if args_cli.print_rear_width_metrics:
            raise ValueError(
                "phase-residual 15-rollout behavior gate forbids the env0-only early-break metric path"
            )
        if not args_cli.front_step_eval_reset or args_cli.keep_play_randomization:
            raise ValueError(
                "phase-residual behavior gate requires deterministic front-step reset and no random events"
            )
        if (
            args_cli.eval_action_delay_steps != 0
            or fixed_velocity_command != (0.72, 0.0, 0.0)
            or args_cli.play_terrain_type != "box_hard"
            or int(args_cli.play_terrain_level or -1) != 9
            or args_cli.front_step_eval_side != "x-"
            or abs(float(args_cli.front_step_eval_edge_gap) - 0.55) > 1.0e-9
            or abs(float(args_cli.front_step_eval_lateral_offset)) > 1.0e-9
            or abs(float(args_cli.front_step_eval_yaw_offset_deg)) > 1.0e-9
        ):
            raise ValueError("phase-residual behavior gate fixed-condition contract changed")
        if args_cli.task != "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123-ArcdogAdjustableLeg-v0":
            raise RuntimeError("phase-residual behavior gate is not using the frozen Teacher environment")

        phase_residual_timing_amendment_path = Path(
            args_cli.phase_residual_source_timing_amendment
        ).expanduser().resolve()
        phase_residual_timing_amendment_sha256 = _sha256_file(
            phase_residual_timing_amendment_path
        )
        if phase_residual_timing_amendment_sha256 != args_cli.phase_residual_source_timing_amendment_sha256:
            raise RuntimeError("phase-residual behavior timing amendment SHA mismatch")
        timing_payload = json.loads(
            phase_residual_timing_amendment_path.read_text(encoding="utf-8")
        )
        timing_contract = timing_payload.get("corrected_timing_contract", {})
        if (
            timing_payload.get("status") != "frozen_before_dataset_retry"
            or int(timing_contract.get("settle_steps_before_dataset", -1)) != 296
            or int(timing_contract.get("dataset_pre_active_steps", -1)) != 10
            or int(timing_contract.get("dataset_steps", -1)) != 120
            or int(timing_contract.get("total_play_steps", -1)) != 416
        ):
            raise RuntimeError("phase-residual behavior timing contract changed")

        training_path = Path(args_cli.phase_residual_training_manifest).expanduser().resolve()
        if _sha256_file(training_path) != args_cli.phase_residual_training_manifest_sha256:
            raise RuntimeError("phase-residual behavior training manifest SHA mismatch")
        training_payload = json.loads(training_path.read_text(encoding="utf-8"))
        selected_path = Path(
            training_payload["selected_reference_manifest_path"]
        ).expanduser().resolve()
        if _sha256_file(selected_path) != training_payload["selected_reference_manifest_sha256"]:
            raise RuntimeError("phase-residual behavior selected manifest SHA mismatch")
        selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_reference = selected_payload["selected_reference"]
        behavior_reference = PhaseResidualReference(
            selected_reference["path"], expected_sha256=selected_reference["sha256"]
        )
        behavior_preroll = PhaseResidualPreroll(
            args_cli.phase_residual_preroll_manifest,
            args_cli.phase_residual_preroll_manifest_sha256,
            expected_reference_sha256=behavior_reference.sha256,
            expected_timing_sha256=phase_residual_timing_amendment_sha256,
        )
        behavior_joint_names, _ = _joint_recording_schema(env)
        behavior_terms = getattr(env.unwrapped.action_manager, "_terms", {})
        behavior_term = behavior_terms.get("joint_pos") if isinstance(behavior_terms, dict) else None
        if behavior_term is None:
            raise RuntimeError("phase-residual behavior gate requires the joint_pos action term")
        behavior_robot = env.unwrapped.scene["robot"]
        behavior_joint_id_by_name = {
            name: index for index, name in enumerate(behavior_robot.joint_names)
        }
        behavior_joint_ids = [behavior_joint_id_by_name[name] for name in behavior_joint_names]
        phase_residual_behavior_joint_ids = behavior_joint_ids
        controller_kwargs = {
            "training_manifest_path": training_path,
            "training_manifest_sha256": args_cli.phase_residual_training_manifest_sha256,
            "wandb_verification_path": args_cli.phase_residual_wandb_verification,
            "wandb_verification_sha256": args_cli.phase_residual_wandb_verification_sha256,
            "reference": behavior_reference,
            "preroll": behavior_preroll,
            "environment_checkpoint_path": resume_path,
            "action_term": behavior_term,
            "initial_joint_pos": behavior_robot.data.joint_pos[:, behavior_joint_ids],
            "joint_names": behavior_joint_names,
            "settle_steps": 296,
            "pre_active_steps": 10,
        }
        if args_cli.phase_direct_action:
            phase_residual_hybrid_controller = PhaseDirectActionController(
                **controller_kwargs
            )
            phase_controller_inference_path = Path(__file__).resolve().with_name(
                "highstep_phase_direct_action_inference.py"
            )
        else:
            phase_residual_hybrid_controller = PhaseResidualHybridController(
                **controller_kwargs,
                residual_enabled=not args_cli.phase_residual_disable_residual,
            )
            phase_controller_inference_path = Path(__file__).resolve().with_name(
                "highstep_phase_residual_inference.py"
            )
        phase_residual_behavior_output_dir = Path(
            args_cli.phase_residual_behavior_output
        ).expanduser().resolve()
        if phase_residual_behavior_output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite behavior output: {phase_residual_behavior_output_dir}"
            )
        phase_residual_behavior_output_dir.mkdir(parents=True, exist_ok=False)
        if args_cli.phase_residual_dagger_output:
            if args_cli.phase_direct_action:
                driver_mode = "phase_direct_action"
            else:
                driver_mode = (
                    "reference_only"
                    if args_cli.phase_residual_disable_residual
                    else "residual_enabled"
                )
            phase_residual_dagger_collector = PhaseResidualDaggerCollector(
                output_dir=args_cli.phase_residual_dagger_output,
                preregistration_path=args_cli.phase_residual_dagger_preregistration,
                preregistration_sha256=(
                    args_cli.phase_residual_dagger_preregistration_sha256
                ),
                training_manifest_path=training_path,
                training_manifest_sha256=(
                    args_cli.phase_residual_training_manifest_sha256
                ),
                joint_names=behavior_joint_names,
                num_envs=int(env.num_envs),
                driver_mode=driver_mode,
                authority_profile=(
                    "direct_action" if args_cli.phase_direct_action else "residual"
                ),
            )
        start_manifest = {
            "schema_version": 1,
            "kind": "highstep_phase_residual_behavior_start",
            "status": "running",
            "mode": getattr(
                phase_residual_hybrid_controller,
                "controller_mode",
                "residual_enabled" if not args_cli.phase_residual_disable_residual else "reference_only",
            ),
            "controller": phase_residual_hybrid_controller.summary(),
            "source_timing_amendment_path": str(phase_residual_timing_amendment_path),
            "source_timing_amendment_sha256": phase_residual_timing_amendment_sha256,
            "play_code_sha256": _sha256_file(Path(__file__).resolve()),
            "inference_code_sha256": _sha256_file(phase_controller_inference_path),
        }
        (phase_residual_behavior_output_dir / "start_manifest.json").write_text(
            json.dumps(start_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        phase_residual_behavior_rear_top_streak = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int64
        )
        phase_residual_behavior_rear_confirmed = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        phase_residual_behavior_hold_streak = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.int64
        )
        phase_residual_behavior_hold_success = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        phase_residual_behavior_terminated = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        print(
            "[PHASE_RESIDUAL_BEHAVIOR_START] "
            + json.dumps(start_manifest, sort_keys=True),
            flush=True,
        )
    if args_cli.training_distribution:
        initial_distribution = _training_distribution_initial_snapshot(
            env, training_distribution_binding
        )
        print(
            "[TRAINING_DISTRIBUTION_INITIAL_JSON] "
            + json.dumps(initial_distribution, sort_keys=True),
            flush=True,
        )
    manual_reset_requested = False
    if args_cli.keyboard:
        def request_manual_reset():
            nonlocal manual_reset_requested
            manual_reset_requested = True
            print(
                "RESET RECOVERING / 请勿操作 | R pressed; full recovery wait restarted and "
                "keyboard motion input is ignored.",
                flush=True,
            )

        controller.add_callback("R", request_manual_reset)
        print(
            "[INFO] Manual controls: arrows/numpad move, Z/X yaw, L stops commands, "
            "R respawns at the selected high-step start.",
            flush=True,
        )
    if args_cli.video or args_cli.highstep_gap_camera in {
        "static_side_top",
        "static_robot_side",
    }:
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
    joint_recorder = None
    if args_cli.record_student_tensors and not args_cli.record_joint_data:
        raise ValueError("--record_student_tensors requires --record_joint_data")
    if args_cli.joint_record_output and not args_cli.record_joint_data:
        raise ValueError("--joint_record_output requires --record_joint_data")
    if args_cli.record_joint_data:
        record_joint_names, action_term_schema = _joint_recording_schema(env)
        inferred_role = "student" if "student" in str(args_cli.task or "").lower() else "teacher"
        record_label = args_cli.joint_record_label or inferred_role
        record_output = _joint_recording_output_dir(record_label, args_cli.joint_record_output)
        schedule_path_text = str(schedule_manifest_path) if schedule_manifest_path is not None else None
        joint_recorder = PlayJointRecorder(
            record_output,
            joint_names=record_joint_names,
            step_dt=dt,
            record_student_tensors=bool(args_cli.record_student_tensors),
            metadata={
                "label": record_label,
                "task": args_cli.task,
                "checkpoint_path": str(Path(resume_path).resolve()),
                "checkpoint_sha256": _sha256_file(resume_path),
                "seed": args_cli.seed,
                "num_envs": int(env.num_envs),
                "recorded_env_index": 0,
                "debug_enabled": bool(args_cli.debug),
                "student_tensor_recording_enabled": bool(args_cli.record_student_tensors),
                "terrain_type": args_cli.play_terrain_type,
                "terrain_level": args_cli.play_terrain_level,
                "front_step_eval_reset": bool(args_cli.front_step_eval_reset),
                "front_step_eval_side": args_cli.front_step_eval_side,
                "front_step_eval_edge_gap": float(args_cli.front_step_eval_edge_gap),
                "front_step_eval_lateral_offset": float(args_cli.front_step_eval_lateral_offset),
                "front_step_eval_yaw_offset_deg": float(args_cli.front_step_eval_yaw_offset_deg),
                "eval_action_delay_steps": args_cli.eval_action_delay_steps,
                "schedule_manifest_path": schedule_path_text,
                "schedule_manifest_sha256": (
                    _sha256_file(schedule_manifest_path) if schedule_manifest_path is not None else None
                ),
                "play_code_sha256": _sha256_file(Path(__file__).resolve()),
                "recorder_code_sha256": _sha256_file(
                    Path(__file__).resolve().with_name("play_joint_recorder.py")
                ),
                "action_terms": action_term_schema,
                "recorded_command_replay": (
                    recorded_command_replay.summary()
                    if recorded_command_replay is not None
                    else None
                ),
                "recorded_command_source_manifest_path": (
                    str(recorded_command_source_manifest_path)
                    if recorded_command_source_manifest_path is not None
                    else None
                ),
                "recorded_command_source_manifest_sha256": (
                    recorded_command_source_manifest_sha256
                ),
            },
        )
        if args_cli.record_student_tensors:
            _student_tensor_recording_snapshot(obs, policy_nn)
        print(
            f"[JOINT_RECORD] recording env0 to {joint_recorder.csv_path}\n"
            f"[JOINT_RECORD] manifest: {joint_recorder.manifest_path}",
            flush=True,
        )
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
    v1123_preflight_tracker = None
    if args_cli.highstep_v1123_frozen_preflight_output:
        if highstep_eval_tracker is None:
            raise ValueError(
                "--highstep_v1123_frozen_preflight_output requires the fixed front-step metric path"
            )
        v1123_preflight_tracker = _HighstepV1123FrozenPreflightTracker(
            env, highstep_eval_tracker
        )
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
                if manual_highstep_start_gate is not None:
                    manual_highstep_start_gate.reset()
                if raw_motion_student_controller is not None:
                    raw_motion_student_controller.reset()
                _apply_play_terrain_selection(env, args_cli.play_terrain_level, args_cli.play_terrain_type)
                obs, _ = env.reset()
                if args_cli.front_step_eval_reset:
                    platform_width = _front_step_platform_width(
                        env,
                        args_cli.play_terrain_type,
                        args_cli.front_step_eval_platform_width,
                    )
                    if manual_highstep_respawn_contract:
                        _apply_manual_highstep_respawn(
                            env,
                            args_cli.front_step_eval_side,
                            args_cli.front_step_eval_distance,
                            args_cli.front_step_eval_edge_gap,
                            platform_width,
                            args_cli.front_step_eval_lateral_offset,
                            args_cli.front_step_eval_yaw_offset_deg,
                            args_cli.play_terrain_type,
                        )
                    else:
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
            if joint_recorder is not None:
                joint_recorder.mark_reset()
            manual_reset_requested = False
            print(
                "RESET RECOVERING / 请勿操作 | deterministic q/dq and high-step pose restored; "
                f"wait at least {manual_highstep_start_gate.min_hold_steps} control steps "
                f"({manual_highstep_start_gate.min_hold_steps * float(env.unwrapped.step_dt):.2f} s) "
                "and the original stability gate.",
                flush=True,
            )

        if manual_highstep_start_gate is not None and manual_highstep_start_gate.blocked:
            robot_state = env.unwrapped.scene["robot"].data
            gate_opened = manual_highstep_start_gate.observe(
                robot_state.joint_pos[0],
                robot_state.joint_vel[0],
                robot_state.root_lin_vel_b[0],
                robot_state.root_ang_vel_b[0],
            )
            if gate_opened:
                print(
                    "RESET READY / 现在可以推动方向键 | "
                    "[MANUAL_HIGHSTEP_READY] Stable start reached; keyboard commands are now enabled. "
                    + json.dumps(manual_highstep_start_gate.last_metrics, sort_keys=True),
                    flush=True,
                )
            elif manual_highstep_start_gate.steps_since_reset % 25 == 0:
                print(
                    "[MANUAL_HIGHSTEP_SETTLING] Keyboard commands remain zero. "
                    + json.dumps(manual_highstep_start_gate.last_metrics, sort_keys=True),
                    flush=True,
                )

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
        if recorded_command_replay is not None:
            recorded_command_replay.set_step(timestep)
        one_shot_command_active = (
            raw_motion_student_controller is not None
            and raw_motion_student_controller.one_shot_phase is not None
        )
        if (
            args_cli.keyboard
            or args_cli.se2_gamepad
            or fixed_velocity_command is not None
            or recorded_command_replay is not None
            or one_shot_command_active
        ):
            try:
                # 获取环境中的 base_velocity 指令项
                cmd_term = env.unwrapped.command_manager._terms.get("base_velocity")
                if cmd_term is not None:
                    # 获取当前控制器的最新指令
                    if one_shot_command_active:
                        one_shot_obs_command_step["value"] = timestep
                        cur_cmd = torch.tensor(
                            [raw_motion_student_controller.one_shot_command(timestep), 0.0, 0.0],
                            device=env.device,
                            dtype=torch.float32,
                        ).unsqueeze(0).repeat(env.num_envs, 1)
                    elif args_cli.keyboard:
                        cur_cmd = _manual_keyboard_command().unsqueeze(0).to(
                            env.device, dtype=torch.float32
                        )
                    elif args_cli.se2_gamepad:
                        cur_cmd = se2_controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32)
                    elif recorded_command_replay is not None:
                        cur_cmd = torch.tensor(
                            recorded_command_replay.command(),
                            device=env.device,
                            dtype=torch.float32,
                        ).unsqueeze(0).repeat(env.num_envs, 1)
                    else:
                        command_value = fixed_velocity_command
                        if phase_residual_dataset_collector is not None:
                            dataset_step = timestep - phase_residual_dataset_settle_steps
                            if dataset_step < 10:
                                command_value = (0.0, 0.0, 0.0)
                        if (
                            phase_residual_hybrid_controller is not None
                            and not phase_residual_hybrid_controller.command_is_active(timestep)
                        ):
                            command_value = (0.0, 0.0, 0.0)
                        cur_cmd = torch.tensor(
                            command_value, device=env.device, dtype=torch.float32
                        ).unsqueeze(0).repeat(env.num_envs, 1)

                    # 同步给底层的命令管理器 (仅用于可视化箭头等，不影响网络实际吃到的指令)
                    if hasattr(cmd_term, "command"):
                        cmd_term.command[:] = cur_cmd
                    if hasattr(cmd_term, "vel_command_b"):
                        cmd_term.vel_command_b[:] = cur_cmd
                    if args_cli.fixed_motion_direct_action_mode is not None:
                        fixed_motion_current_vx = cur_cmd[:, 0].detach().clone()
                    if args_cli.fixed_motion_raw_action_mode is not None:
                        raw_motion_current_vx = cur_cmd[:, 0].detach().clone()
                    if b300_hybrid_collector is not None or b300_hybrid_dagger_collector is not None:
                        b300_hybrid_current_command = cur_cmd.detach().clone()
                    if one_shot_command_active:
                        # The deployed controller assembles the 570-D observation
                        # from the command for this inference step.  Refresh here
                        # after advancing the internal clock so Isaac uses the
                        # same current-step contract instead of a one-step-old
                        # command at the 21/80 transition boundaries.
                        obs = env.get_observations()
                    if (
                        phase_residual_dataset_collector is not None
                        or phase_residual_hybrid_controller is not None
                    ):
                        # Ensure the current policy observation contains the
                        # source-equivalent command transition on this step.
                        obs = env.get_observations()
            except Exception:
                if (
                    phase_residual_dataset_collector is not None
                    or phase_residual_hybrid_controller is not None
                    or recorded_command_replay is not None
                ):
                    raise
        # =========================================================================

        start_time = time.time()
        pending_student_tensors = None
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.highstep_gap_camera not in {
                "static_side_top",
                "static_robot_side",
            }:
                _update_highstep_gap_camera(env, args_cli.highstep_gap_camera)
            if joint_recorder is not None and args_cli.record_student_tensors:
                pending_student_tensors = _student_tensor_recording_snapshot(obs, policy_nn)
            # agent stepping
            if b300_hybrid_dagger_collector is not None:
                actions = policy(obs)
                b300_hybrid_dagger_collector.prepare(obs, actions, b300_hybrid_current_command)
            elif b300_hybrid_collector is not None:
                actions = policy(obs)
                b300_hybrid_collector.prepare(obs, actions, b300_hybrid_current_command)
            elif raw_motion_collector is not None:
                teacher_raw_action = policy(obs)
                actions = raw_motion_collector.prepare(obs, raw_motion_current_vx, teacher_raw_action)
            elif raw_motion_student_controller is not None:
                actions = raw_motion_student_controller.action(obs, raw_motion_current_vx)
                if raw_motion_dagger_collector is not None:
                    assert raw_motion_student_controller.last_sample is not None
                    raw_motion_dagger_collector.prepare(
                        raw_motion_student_controller.last_sample,
                        policy(obs),
                    )
            elif fixed_motion_oracle_collector is not None:
                assert fixed_motion_action_term is not None
                assert fixed_motion_target_adapter is not None
                fixed_motion_oracle_collector.prepare(obs, fixed_motion_current_vx)
                teacher_raw_action = policy(obs)
                fixed_motion_action_term.process_actions(teacher_raw_action)
                fixed_motion_pending_teacher_target = fixed_motion_action_term.processed_actions.detach().clone()
                actions = fixed_motion_target_adapter.raw_for_target(fixed_motion_pending_teacher_target)
            elif fixed_motion_student_controller is not None:
                actions = fixed_motion_student_controller.action(obs, fixed_motion_current_vx)
            elif phase_residual_hybrid_controller is not None:
                dagger_teacher_target = None
                if (
                    phase_residual_dagger_collector is not None
                    and timestep >= phase_residual_hybrid_controller.settle_steps
                ):
                    # Teacher is queried at the candidate's exact pre-step
                    # physical state.  This probe never drives physics: the
                    # candidate call immediately below overwrites the live
                    # action term before env.step().
                    teacher_probe_action = policy(obs)
                    dagger_action_term = env.unwrapped.action_manager._terms["joint_pos"]
                    dagger_action_term.process_actions(teacher_probe_action)
                    dagger_teacher_target = getattr(
                        dagger_action_term, "processed_actions", None
                    )
                    if not isinstance(dagger_teacher_target, torch.Tensor):
                        raise RuntimeError("DAgger Teacher probe exposes no processed target")
                    dagger_teacher_target = dagger_teacher_target.detach().clone()
                actions = phase_residual_hybrid_controller.action_for_step(obs, timestep)
                if phase_residual_dagger_collector is not None and dagger_teacher_target is not None:
                    assert phase_residual_behavior_joint_ids is not None
                    dagger_robot = env.unwrapped.scene["robot"]
                    phase_residual_dagger_collector.prepare_before_step(
                        sample=phase_residual_hybrid_controller.dagger_sample(),
                        teacher_mapped_target=dagger_teacher_target,
                        joint_pos=dagger_robot.data.joint_pos[
                            :, phase_residual_behavior_joint_ids
                        ],
                        joint_vel=dagger_robot.data.joint_vel[
                            :, phase_residual_behavior_joint_ids
                        ],
                    )
            elif phase_residual_reference_controller is None:
                dataset_step = timestep - phase_residual_dataset_settle_steps
                if phase_residual_dataset_collector is not None and dataset_step >= 0:
                    phase_residual_dataset_collector.prepare_observation(obs, dataset_step)
                actions = policy(obs)
            else:
                actions = phase_residual_reference_controller.action_for_step(timestep)
            if highstep_eval_tracker is not None:
                actions_for_audit = actions
                if getattr(env, "clip_actions", None) is not None:
                    actions_for_audit = torch.clamp(actions, -env.clip_actions, env.clip_actions)
                highstep_eval_tracker.audit_policy_actions(actions_for_audit, timestep)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if b300_hybrid_collector is not None:
                b300_hybrid_collector.record_after_step()
            if b300_hybrid_dagger_collector is not None:
                b300_hybrid_dagger_collector.record_after_step()
            if raw_motion_collector is not None:
                assert raw_motion_action_term is not None
                raw_motion_collector.record_after_step(raw_motion_action_term)
            if raw_motion_dagger_collector is not None:
                raw_motion_dagger_collector.record_after_step()
            if args_cli.fixed_motion_raw_action_mode is not None:
                assert raw_motion_terminated is not None
                assert raw_motion_rear_top_streak is not None
                assert raw_motion_rear_confirmed is not None
                assert raw_motion_hold_streak is not None
                assert raw_motion_hold_success is not None
                raw_motion_terminated |= dones.detach().reshape(-1).bool()
                rear_top, rear_on_platform = _phase_residual_batch_rear_on_platform(env)
                raw_motion_rear_top_streak = torch.where(rear_top, raw_motion_rear_top_streak + 1, torch.zeros_like(raw_motion_rear_top_streak))
                raw_motion_rear_confirmed |= raw_motion_rear_top_streak >= 2
                hold_now = raw_motion_rear_confirmed & rear_on_platform
                raw_motion_hold_streak = torch.where(hold_now, raw_motion_hold_streak + 1, torch.zeros_like(raw_motion_hold_streak))
                raw_motion_hold_success |= raw_motion_hold_streak >= 25
            if fixed_motion_oracle_collector is not None:
                assert fixed_motion_target_adapter is not None
                assert fixed_motion_pending_teacher_target is not None
                fixed_motion_audit = fixed_motion_target_adapter.audit()
                if fixed_motion_audit["exact"] is not True:
                    raise RuntimeError("safe oracle target changed in production action channel: " + json.dumps(fixed_motion_audit, sort_keys=True))
                fixed_motion_oracle_collector.record(fixed_motion_pending_teacher_target)
            elif fixed_motion_student_controller is not None:
                fixed_motion_audit = fixed_motion_student_controller.audit()
                if fixed_motion_audit["exact"] is not True:
                    raise RuntimeError("Student safe target changed in production action channel: " + json.dumps(fixed_motion_audit, sort_keys=True))
            if args_cli.fixed_motion_direct_action_mode is not None:
                assert fixed_motion_terminated is not None
                assert fixed_motion_rear_top_streak is not None
                assert fixed_motion_rear_confirmed is not None
                assert fixed_motion_hold_streak is not None
                assert fixed_motion_hold_success is not None
                fixed_motion_terminated |= dones.detach().reshape(-1).bool()
                rear_top, rear_on_platform = _phase_residual_batch_rear_on_platform(env)
                fixed_motion_rear_top_streak = torch.where(rear_top, fixed_motion_rear_top_streak + 1, torch.zeros_like(fixed_motion_rear_top_streak))
                fixed_motion_rear_confirmed |= fixed_motion_rear_top_streak >= 2
                hold_now = fixed_motion_rear_confirmed & rear_on_platform
                fixed_motion_hold_streak = torch.where(hold_now, fixed_motion_hold_streak + 1, torch.zeros_like(fixed_motion_hold_streak))
                fixed_motion_hold_success |= fixed_motion_hold_streak >= 25
            if (
                phase_residual_dagger_collector is not None
                and timestep >= phase_residual_hybrid_controller.settle_steps
            ):
                phase_residual_dagger_collector.record_after_step(dones)
            if phase_residual_reference_controller is not None:
                reference_audit = phase_residual_reference_controller.audit_processed_target()
                if reference_audit["processed_target_exact"] is not True:
                    raise RuntimeError(
                        "phase-residual reference target changed inside the live action term: "
                        + json.dumps(reference_audit, sort_keys=True)
                    )
            if phase_residual_hybrid_controller is not None:
                hybrid_audit = phase_residual_hybrid_controller.audit_processed_target()
                if hybrid_audit["processed_target_exact"] is not True:
                    raise RuntimeError(
                        "phase-residual hybrid target changed inside the live action term: "
                        + json.dumps(hybrid_audit, sort_keys=True)
                    )
            terminated_this_step = bool(torch.any(dones).item())
            if phase_residual_hybrid_controller is not None:
                assert phase_residual_behavior_terminated is not None
                phase_residual_behavior_terminated |= dones.detach().reshape(-1).bool()
                if timestep >= phase_residual_hybrid_controller.settle_steps:
                    assert phase_residual_behavior_rear_top_streak is not None
                    assert phase_residual_behavior_rear_confirmed is not None
                    assert phase_residual_behavior_hold_streak is not None
                    assert phase_residual_behavior_hold_success is not None
                    rear_top, rear_on_platform = _phase_residual_batch_rear_on_platform(env)
                    phase_residual_behavior_rear_top_streak = torch.where(
                        rear_top,
                        phase_residual_behavior_rear_top_streak + 1,
                        torch.zeros_like(phase_residual_behavior_rear_top_streak),
                    )
                    phase_residual_behavior_rear_confirmed |= (
                        phase_residual_behavior_rear_top_streak >= 2
                    )
                    hold_now = phase_residual_behavior_rear_confirmed & rear_on_platform
                    phase_residual_behavior_hold_streak = torch.where(
                        hold_now,
                        phase_residual_behavior_hold_streak + 1,
                        torch.zeros_like(phase_residual_behavior_hold_streak),
                    )
                    phase_residual_behavior_hold_success |= (
                        phase_residual_behavior_hold_streak >= 25
                    )
            if phase_residual_dataset_collector is not None:
                assert phase_residual_dataset_joint_ids is not None
                dataset_term = env.unwrapped.action_manager._terms["joint_pos"]
                dataset_target = getattr(dataset_term, "processed_actions", None)
                if not isinstance(dataset_target, torch.Tensor):
                    raise RuntimeError("dataset action term exposes no processed mapped target")
                dataset_robot = env.unwrapped.scene["robot"]
                dataset_step = timestep - phase_residual_dataset_settle_steps
                if dataset_step < 0:
                    phase_residual_dataset_collector.prime_action_history(dataset_target)
                else:
                    phase_residual_dataset_collector.record_after_step(
                        teacher_mapped_target=dataset_target,
                        joint_pos=dataset_robot.data.joint_pos[:, phase_residual_dataset_joint_ids],
                        joint_vel=dataset_robot.data.joint_vel[:, phase_residual_dataset_joint_ids],
                        dones=dones,
                    )
                    rear_top, rear_on_platform = _phase_residual_batch_rear_on_platform(env)
                    phase_residual_dataset_rear_top_streak = torch.where(
                        rear_top,
                        phase_residual_dataset_rear_top_streak + 1,
                        torch.zeros_like(phase_residual_dataset_rear_top_streak),
                    )
                    phase_residual_dataset_rear_confirmed |= (
                        phase_residual_dataset_rear_top_streak >= 2
                    )
                    hold_now = phase_residual_dataset_rear_confirmed & rear_on_platform
                    phase_residual_dataset_hold_streak = torch.where(
                        hold_now,
                        phase_residual_dataset_hold_streak + 1,
                        torch.zeros_like(phase_residual_dataset_hold_streak),
                    )
                    phase_residual_dataset_hold_success |= (
                        phase_residual_dataset_hold_streak >= 25
                    )
            if joint_recorder is not None:
                joint_recorder.record(
                    control_step=timestep,
                    terminated_after_step=terminated_this_step,
                    student_tensors=pending_student_tensors,
                    **_joint_recording_snapshot(env, actions, record_joint_names),
                )
                if terminated_this_step:
                    joint_recorder.mark_reset()
            if effort_feasibility_tracker is not None:
                effort_feasibility_tracker.sample_applied_torque()
            if args_cli.highstep_gap_camera not in {
                "static_side_top",
                "static_robot_side",
            }:
                _update_highstep_gap_camera(env, args_cli.highstep_gap_camera)
            if args_cli.print_rear_width_metrics:
                if highstep_eval_tracker is not None:
                    if terminated_this_step:
                        highstep_eval_tracker.mark_terminated(timestep)
                    else:
                        highstep_eval_tracker.update(timestep)
                    if v1123_preflight_tracker is not None:
                        v1123_preflight_tracker.sample(timestep)
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

    if b300_hybrid_collector is not None:
        canonical_result = b300_hybrid_collector.finalize()
        print(
            "[B300_HYBRID_CANONICAL_JSON] " + json.dumps(canonical_result, sort_keys=True),
            flush=True,
        )
    if b300_hybrid_dagger_collector is not None:
        dagger_result = b300_hybrid_dagger_collector.finalize()
        print("[B300_HYBRID_DAGGER_RESULT] " + json.dumps(dagger_result, sort_keys=True), flush=True)

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
    highstep_eval_summary = None
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
    phase_residual_dataset_failure = None
    phase_residual_behavior_failure = None
    if phase_residual_dataset_collector is not None:
        assert phase_residual_dataset_hold_success is not None
        dataset_result = phase_residual_dataset_collector.finalize(
            metadata={
                "task": args_cli.task,
                "checkpoint_path": str(Path(resume_path).resolve()),
                "checkpoint_sha256": _sha256_file(resume_path),
                "seed": args_cli.seed,
                "fixed_velocity_command": list(fixed_velocity_command),
                "terrain_type": args_cli.play_terrain_type,
                "terrain_level": args_cli.play_terrain_level,
                "front_edge_gap": float(args_cli.front_step_eval_edge_gap),
                "action_delay_steps": args_cli.eval_action_delay_steps,
                "settle_steps_before_dataset": phase_residual_dataset_settle_steps,
                "dataset_pre_active_steps": 10,
                "source_timing_amendment_path": str(
                    phase_residual_timing_amendment_path
                ),
                "source_timing_amendment_sha256": (
                    phase_residual_timing_amendment_sha256
                ),
                "schedule_manifest_path": (
                    str(schedule_manifest_path) if schedule_manifest_path is not None else None
                ),
                "schedule_manifest_sha256": (
                    _sha256_file(schedule_manifest_path) if schedule_manifest_path is not None else None
                ),
                "play_code_sha256": _sha256_file(Path(__file__).resolve()),
                "dataset_code_sha256": _sha256_file(
                    Path(__file__).resolve().with_name("highstep_phase_residual_dataset.py")
                ),
                "env0_canonical_summary": highstep_eval_summary,
            },
            env_hold_success=phase_residual_dataset_hold_success.detach().cpu().tolist(),
            env_full_climb_success=phase_residual_dataset_hold_success.detach().cpu().tolist(),
        )
        print(
            "[PHASE_RESIDUAL_DATASET_JSON] " + json.dumps(dataset_result, sort_keys=True),
            flush=True,
        )
        if dataset_result["valid"] is not True:
            phase_residual_dataset_failure = (
                "phase-residual Teacher dataset failed closed: "
                f"terminated={dataset_result['terminated_env_ids']} "
                f"hold={sum(dataset_result['env_hold_success'])}/15"
            )
    if phase_residual_dagger_collector is not None:
        dagger_result = phase_residual_dagger_collector.finalize(
            metadata={
                "task": args_cli.task,
                "checkpoint_path": str(Path(resume_path).resolve()),
                "checkpoint_sha256": _sha256_file(resume_path),
                "seed": args_cli.seed,
                "fixed_velocity_command": list(fixed_velocity_command),
                "terrain_type": args_cli.play_terrain_type,
                "terrain_level": args_cli.play_terrain_level,
                "front_edge_gap": float(args_cli.front_step_eval_edge_gap),
                "lateral_offset": float(args_cli.front_step_eval_lateral_offset),
                "yaw_offset_deg": float(args_cli.front_step_eval_yaw_offset_deg),
                "action_delay_steps": args_cli.eval_action_delay_steps,
                "settle_steps": phase_residual_hybrid_controller.settle_steps,
                "pre_active_steps": phase_residual_hybrid_controller.pre_active_steps,
                "source_timing_amendment_path": str(
                    phase_residual_timing_amendment_path
                ),
                "source_timing_amendment_sha256": (
                    phase_residual_timing_amendment_sha256
                ),
                "play_code_sha256": _sha256_file(Path(__file__).resolve()),
                "dagger_code_sha256": _sha256_file(
                    Path(__file__).resolve().with_name(
                        "highstep_phase_residual_dagger.py"
                    )
                ),
            }
        )
        print(
            "[PHASE_RESIDUAL_DAGGER_JSON] " + json.dumps(dagger_result, sort_keys=True),
            flush=True,
        )
    if phase_residual_hybrid_controller is not None:
        assert phase_residual_behavior_output_dir is not None
        assert phase_residual_behavior_hold_success is not None
        assert phase_residual_behavior_terminated is not None
        hold = phase_residual_behavior_hold_success.detach().cpu().bool()
        terminated = phase_residual_behavior_terminated.detach().cpu().bool()
        valid = ~terminated
        reset_context = getattr(env.unwrapped, "_front_step_eval_context", {})
        reset_valid = bool(reset_context.get("reset_valid", False))
        valid_count = int(torch.sum(valid).item()) if reset_valid else 0
        hold_count = int(torch.sum(hold & valid).item()) if reset_valid else 0
        passed = valid_count == 15 and hold_count == 15
        behavior_result = {
            "schema_version": 1,
            "kind": "highstep_fixed_condition_phase_residual_behavior_gate",
            "status": "passed" if passed else "failed_behavior_gate",
            "mode": getattr(
                phase_residual_hybrid_controller,
                "controller_mode",
                "residual_enabled"
                if phase_residual_hybrid_controller.residual_enabled
                else "reference_only",
            ),
            "passed": passed,
            "valid_count": valid_count,
            "full_climb_count": hold_count,
            "rear_hold_count": hold_count,
            "required": {"valid": 15, "full_climb": 15, "rear_hold": 15},
            "terminated_env_ids": [
                int(index) for index, value in enumerate(terminated.tolist()) if value
            ],
            "rear_hold_by_env": [bool(value) for value in hold.tolist()],
            "reset_valid": reset_valid,
            "fixed_condition": {
                "terrain_type": args_cli.play_terrain_type,
                "terrain_level": args_cli.play_terrain_level,
                "front_edge_gap": float(args_cli.front_step_eval_edge_gap),
                "lateral_offset": float(args_cli.front_step_eval_lateral_offset),
                "yaw_offset_deg": float(args_cli.front_step_eval_yaw_offset_deg),
                "command": list(fixed_velocity_command),
                "action_delay_steps": args_cli.eval_action_delay_steps,
                "settle_steps": phase_residual_hybrid_controller.settle_steps,
                "pre_active_steps": phase_residual_hybrid_controller.pre_active_steps,
            },
            "controller": phase_residual_hybrid_controller.summary(),
            "source_timing_amendment_path": str(phase_residual_timing_amendment_path),
            "source_timing_amendment_sha256": phase_residual_timing_amendment_sha256,
            "play_code_sha256": _sha256_file(Path(__file__).resolve()),
            "inference_code_sha256": _sha256_file(phase_controller_inference_path),
            "env0_canonical_summary": highstep_eval_summary,
        }
        result_path = phase_residual_behavior_output_dir / "behavior_result.json"
        temporary_result = result_path.with_suffix(".json.tmp")
        temporary_result.write_text(
            json.dumps(behavior_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_result, result_path)
        behavior_result["result_path"] = str(result_path)
        behavior_result["result_sha256"] = _sha256_file(result_path)
        print(
            "[PHASE_RESIDUAL_BEHAVIOR_JSON] " + json.dumps(behavior_result, sort_keys=True),
            flush=True,
        )
        if not passed:
            phase_residual_behavior_failure = (
                "phase-residual behavior gate failed closed: "
                f"valid={valid_count}/15 full={hold_count}/15 rear_hold={hold_count}/15"
            )
    if v1123_preflight_tracker is not None:
        preflight_output = Path(args_cli.highstep_v1123_frozen_preflight_output).resolve()
        preflight_output.parent.mkdir(parents=True, exist_ok=True)
        preflight_output.write_text(
            json.dumps(v1123_preflight_tracker.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[HIGHSTEP_V1123_PREFLIGHT] {preflight_output}", flush=True)

    if args_cli.fixed_motion_direct_action_mode is not None:
        assert fixed_motion_hold_success is not None and fixed_motion_terminated is not None
        hold_values = fixed_motion_hold_success.detach().cpu().bool()
        terminated_values = fixed_motion_terminated.detach().cpu().bool()
        behavior = {
            "valid_count": int(torch.sum(~terminated_values).item()),
            "full_climb_count": int(torch.sum(hold_values).item()),
            "rear_hold_count": int(torch.sum(hold_values).item()),
            "full_climb": bool(hold_values[0].item()) if len(hold_values) == 1 else None,
            "rear_hold": bool(hold_values[0].item()) if len(hold_values) == 1 else None,
            "terminated_count": int(torch.sum(terminated_values).item()),
        }
        if fixed_motion_oracle_collector is not None:
            oracle_result = fixed_motion_oracle_collector.finalize(behavior)
            print("[FIXED_MOTION_ORACLE_JSON] " + json.dumps(oracle_result, sort_keys=True), flush=True)
            if oracle_result["passed"] is not True:
                fixed_motion_failure = "fixed-motion safe oracle failed; training is forbidden"
        else:
            assert fixed_motion_student_controller is not None
            passed = bool(
                behavior["valid_count"] == 15
                and behavior["full_climb_count"] >= 12
                and behavior["rear_hold_count"] >= 12
                and fixed_motion_student_controller.limit_violations == 0
            )
            behavior.update({
                "schema_version": 1,
                "kind": "highstep_fixed_motion_student_behavior_gate",
                "status": "passed_pending_user_visual_review" if passed else "failed_behavior_gate",
                "passed": passed,
                "target_limit_violations": fixed_motion_student_controller.limit_violations,
                "required": {"valid": 15, "full_climb": 12, "rear_hold": 12, "safe": 15},
            })
            result_path = Path(args_cli.fixed_motion_output).expanduser().resolve() / "behavior_result.json"
            result_path.parent.mkdir(parents=True, exist_ok=False)
            result_path.write_text(json.dumps(behavior, indent=2, sort_keys=True) + "\n")
            print("[FIXED_MOTION_BEHAVIOR_JSON] " + json.dumps(behavior, sort_keys=True), flush=True)

    if args_cli.fixed_motion_raw_action_mode is not None:
        assert raw_motion_hold_success is not None and raw_motion_terminated is not None
        hold = bool(raw_motion_hold_success[0].detach().cpu().item())
        terminated = bool(raw_motion_terminated[0].detach().cpu().item())
        raw_behavior = {
            "valid": not terminated,
            "full_climb": hold,
            "rear_hold": hold,
            "terminated": terminated,
        }
        if raw_motion_collector is not None:
            raw_smoke_result = raw_motion_collector.finalize(raw_behavior)
            print("[FIXED_MOTION_RAW_SMOKE_JSON] " + json.dumps(raw_smoke_result, sort_keys=True), flush=True)
            if raw_smoke_result["passed"] is not True:
                raw_motion_failure = "raw-action passthrough smoke failed; training is forbidden"
        else:
            assert raw_motion_student_controller is not None
            raw_behavior.update({
                "schema_version": 1,
                "kind": "highstep_fixed_motion_raw_student_behavior",
                "status": "completed",
                "illegal_outputs": raw_motion_student_controller.illegal_outputs,
            })
            output = Path(args_cli.fixed_motion_raw_output).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=raw_motion_dagger_collector is not None)
            (output / "behavior_result.json").write_text(json.dumps(raw_behavior, indent=2, sort_keys=True) + "\n")
            if raw_motion_dagger_collector is not None:
                dagger_result = raw_motion_dagger_collector.finalize(raw_behavior)
                print("[FIXED_MOTION_RAW_DAGGER_JSON] " + json.dumps(dagger_result, sort_keys=True), flush=True)
            print("[FIXED_MOTION_RAW_BEHAVIOR_JSON] " + json.dumps(raw_behavior, sort_keys=True), flush=True)

    if joint_recorder is not None:
        joint_recorder.close("normal")
        print(
            f"[JOINT_RECORD] completed: {joint_recorder.csv_path}\n"
            f"[JOINT_RECORD] final manifest: {joint_recorder.manifest_path}",
            flush=True,
        )

    # close the simulator
    env.close()
    if phase_residual_dataset_failure is not None:
        raise RuntimeError(phase_residual_dataset_failure)
    if phase_residual_behavior_failure is not None:
        raise RuntimeError(phase_residual_behavior_failure)
    if fixed_motion_failure is not None:
        raise RuntimeError(fixed_motion_failure)
    if raw_motion_failure is not None:
        raise RuntimeError(raw_motion_failure)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
