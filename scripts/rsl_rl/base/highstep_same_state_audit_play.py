#!/usr/bin/env python3
"""Run one preregistered high-step same-state audit trajectory.

This is deliberately a separate launcher from ``play.py``.  It creates exactly
one Robust Student-no-prior environment, drives it only with the qualified B500
Student, and calls ``SameStateAuditSession`` before every environment step.  The
Teacher and protected model_900 anchor are read-only forwards in that same
Student state; neither owns nor steps an environment.

The three base trajectories are fixed by ``base_suite_plan_payload()``.  Run one
at a time, for example::

    /home/lxq/miniconda3/envs/env_isaaclab/bin/python \
      scripts/rsl_rl/base/highstep_same_state_audit_play.py \
      --headless --trajectory seed11_nominal \
      --output-root tmp/highstep_student_recovery_v11_20260712/same_state_base3

No training, export, video, randomization, or policy/checkpoint override is
implemented here.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
PLAY_PATH = Path(__file__).with_name("play.py").resolve()
PLAY_SHA256 = "94588cba3b01e2b95cfdfff6d3b1f555823466a054b1eecf09cba531596b29d0"
AUDIT_TOOL_PATH = ROOT / "tools/highstep_same_state_audit.py"
ENV_CFG_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/highstep_env_cfg.py"
)
VELOCITY_CFG_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/velocity_env_cfg.py"
)
OBSERVATIONS_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/observations.py"
)
EVENTS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/events.py"
ACTIONS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"
AGENT_CFG_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py"
)
VAE_PPO_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
CLI_ARGS_PATH = ROOT / "scripts/rsl_rl/cli_args.py"
SCHEDULE_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"
)

TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0"
B500_CHECKPOINT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_23-55-22_student_recovery_v111_B500_20260712_235515/model_498.pt"
)
B500_SHA256 = "f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159"
B500_ENV_CONFIG = B500_CHECKPOINT.parent / "params/env.yaml"
B500_AGENT_CONFIG = B500_CHECKPOINT.parent / "params/agent.yaml"
B500_ENV_CONFIG_SHA256 = "d7f74dad52201deb5ac122ea9e899ae764f291e68fd73728cdd3dc7ac9969c38"
B500_AGENT_CONFIG_SHA256 = "db2db92fd700a5b400b0e2169e84a6c4f68d541b6753550d961be790146bb235"

NUM_ENVS = 1
TERRAIN_TYPE = "box"
TERRAIN_LEVEL = 9
FIXED_COMMAND = (0.45, 0.0, 0.0)
FRONT_STEP_SIDE = "x-"
FRONT_STEP_EDGE_GAP = 0.55
ACTION_DELAY_STEPS = 0
MAX_STEPS = 600
AUDIT_DRIVER_ACTION_SOURCE = "student"

DISABLED_RANDOM_EVENTS = (
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
)

REAL_GAIN_CONTRACT = {
    "legs_hip": (55.0, 1.5),
    "legs_thigh": (65.0, 1.5),
    "legs_calf": (80.0, 2.5),
}

ACTION_JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)

OBSERVATION_CONTRACT = {
    "policy": {
        "names": (
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ),
        "dims": (30, 30, 30, 160, 160, 160),
        "history_lengths": (10, 10, 10, 10, 10, 10),
        "scales": (0.25, 1.0, 1.0, 1.0, 0.05, 1.0),
        "functions": (
            "isaaclab.envs.mdp.observations:base_ang_vel",
            "isaaclab.envs.mdp.observations:projected_gravity",
            "fixed_velocity_command_override",
            "robot_lab.tasks.locomotion.velocity.mdp.observations:joint_pos_rel_with_persistent_bias",
            "isaaclab.envs.mdp.observations:joint_vel_rel",
            "isaaclab.envs.mdp.observations:last_action",
        ),
    },
    "estimator": {
        "names": (
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ),
        "dims": (30, 30, 30, 160, 160, 160),
        "history_lengths": (10, 10, 10, 10, 10, 10),
        "scales": (0.25, 1.0, 1.0, 1.0, 0.05, 1.0),
        "functions": (
            "isaaclab.envs.mdp.observations:base_ang_vel",
            "isaaclab.envs.mdp.observations:projected_gravity",
            "fixed_velocity_command_override",
            "robot_lab.tasks.locomotion.velocity.mdp.observations:joint_pos_rel_with_persistent_bias",
            "isaaclab.envs.mdp.observations:joint_vel_rel",
            "isaaclab.envs.mdp.observations:last_action",
        ),
    },
    "critic": {
        "names": (
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
            "height_scan",
        ),
        "dims": (3, 3, 3, 3, 16, 16, 16, 102),
        "history_lengths": (0, 0, 0, 0, 0, 0, 0, 0),
        "scales": (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "functions": (
            "isaaclab.envs.mdp.observations:base_lin_vel",
            "isaaclab.envs.mdp.observations:base_ang_vel",
            "isaaclab.envs.mdp.observations:projected_gravity",
            "fixed_velocity_command_override",
            "isaaclab.envs.mdp.observations:joint_pos_rel",
            "isaaclab.envs.mdp.observations:joint_vel_rel",
            "isaaclab.envs.mdp.observations:last_action",
            "isaaclab.envs.mdp.observations:height_scan",
        ),
    },
}

PLAY_CONTRACT_NODES = (
    "_terrain_column_for_name",
    "_apply_play_terrain_selection",
    "_front_step_side_vectors",
    "_front_step_platform_width",
    "_apply_front_step_eval_reset",
    "_safe_quantile",
    "_safe_mean",
    "_highstep_reward_stage_definition",
    "_runtime_eval_action_delay",
    "_HighstepEvalTracker",
)


@dataclass(frozen=True)
class Trajectory:
    run_id: str
    seed: int
    scenario: str
    lateral_offset_m: float
    yaw_offset_deg: float


TRAJECTORIES = {
    "seed11_nominal": Trajectory("seed11_nominal", 11, "nominal", 0.0, 0.0),
    "seed11_left_offset": Trajectory("seed11_left_offset", 11, "left_offset", 0.12, 4.0),
    "seed22_right_offset": Trajectory("seed22_right_offset", 22, "right_offset", -0.12, -4.0),
}


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _select_trajectory(name: str) -> Trajectory:
    try:
        return TRAJECTORIES[name]
    except KeyError as error:
        raise ValueError(f"trajectory must be one of {sorted(TRAJECTORIES)}") from error


def _validate_fixed_cli(args: argparse.Namespace) -> Trajectory:
    trajectory = _select_trajectory(args.trajectory)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    if args.task != TASK:
        raise ValueError(f"same-state audit task is fixed to {TASK!r}")
    if int(args.num_envs) != NUM_ENVS:
        raise ValueError("same-state audit requires --num_envs 1")
    if checkpoint != B500_CHECKPOINT.resolve(strict=True):
        raise ValueError(f"same-state audit checkpoint is fixed to {B500_CHECKPOINT}")
    if sha256_file(checkpoint) != B500_SHA256:
        raise RuntimeError("qualified B500 checkpoint SHA256 mismatch")
    if not bool(args.headless):
        raise ValueError("same-state evidence launcher requires --headless")
    return trajectory


def _play_contract_ast() -> ast.Module:
    actual = sha256_file(PLAY_PATH)
    if actual != PLAY_SHA256:
        raise RuntimeError(f"play.py SHA256 changed: {actual} != {PLAY_SHA256}")
    tree = ast.parse(PLAY_PATH.read_text(encoding="utf-8"), filename=str(PLAY_PATH))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in PLAY_CONTRACT_NODES
    ]
    names = [node.name for node in selected]
    missing = sorted(set(PLAY_CONTRACT_NODES) - set(names))
    duplicate = sorted(name for name in set(names) if names.count(name) != 1)
    if missing or duplicate:
        raise RuntimeError(f"cannot bind exact core9 play helpers: missing={missing}, duplicate={duplicate}")
    return ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))


def _load_exact_play_contract(*, np: Any, torch: Any, math_utils: Any) -> SimpleNamespace:
    """Load only the SHA-bound core9 reset/tracker definitions, never play.py side effects."""
    namespace: dict[str, Any] = {
        "np": np,
        "torch": torch,
        "math_utils": math_utils,
        "Path": Path,
    }
    exec(compile(_play_contract_ast(), str(PLAY_PATH), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in PLAY_CONTRACT_NODES})


def _ensure_base_suite_plan(output_root: Path, audit_module: Any) -> tuple[Path, str]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "base3_suite_plan.json"
    expected = audit_module.base_suite_plan_payload()
    if plan_path.exists():
        actual = json.loads(plan_path.read_text(encoding="utf-8"))
        if actual != json.loads(json.dumps(expected)):
            raise RuntimeError(f"existing base3 suite plan differs from fixed plan: {plan_path}")
        if plan_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"existing base3 suite plan is writable: {plan_path}")
    else:
        _atomic_json(plan_path, expected, read_only=True)
    return plan_path, sha256_file(plan_path)


def _assert_real_gain_config(env_cfg: Any) -> dict[str, list[float]]:
    actuators = env_cfg.scene.robot.actuators
    observed: dict[str, list[float]] = {}
    for group, (expected_kp, expected_kd) in REAL_GAIN_CONTRACT.items():
        cfg = actuators[group]
        kp, kd = float(cfg.stiffness), float(cfg.damping)
        if (kp, kd) != (expected_kp, expected_kd):
            raise RuntimeError(
                f"real-gain contract mismatch for {group}: {(kp, kd)} != {(expected_kp, expected_kd)}"
            )
        observed[group] = [kp, kd]
    return observed


def _configure_fixed_environment(env_cfg: Any, trajectory: Trajectory, ObsTerm: Any, torch: Any) -> dict[str, Any]:
    if int(env_cfg.scene.num_envs) != NUM_ENVS:
        env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.scene.terrain.max_init_terrain_level = None
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is None:
        raise RuntimeError("fixed box-level audit requires a terrain generator")
    generator.num_rows = max(int(generator.num_rows), TERRAIN_LEVEL + 1)
    generator.num_cols = max(int(generator.num_cols), 20)
    generator.curriculum = True

    corruption: dict[str, bool] = {}
    for group_name in ("policy", "estimator", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is None:
            raise RuntimeError(f"missing observation group {group_name!r}")
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
        corruption[group_name] = bool(getattr(group, "enable_corruption", False))
        if corruption[group_name]:
            raise RuntimeError(f"observation corruption remained enabled for {group_name}")
        old_command = getattr(group, "velocity_commands", None)
        if old_command is None:
            raise RuntimeError(f"missing velocity_commands term in {group_name}")
        history = getattr(old_command, "history_length", 0)
        flatten = getattr(old_command, "flatten_history_dim", False)
        setattr(
            group,
            "velocity_commands",
            ObsTerm(
                func=lambda env, cmd=FIXED_COMMAND: torch.tensor(
                    cmd, device=env.device, dtype=torch.float32
                ).unsqueeze(0).repeat(env.num_envs, 1),
                history_length=history,
                flatten_history_dim=flatten,
            ),
        )

    disabled: list[str] = []
    events = env_cfg.events
    for event_name in DISABLED_RANDOM_EVENTS:
        if hasattr(events, event_name):
            setattr(events, event_name, None)
            disabled.append(event_name)
    for event_name in disabled:
        if getattr(events, event_name) is not None:
            raise RuntimeError(f"play-time random event remained enabled: {event_name}")

    action_cfg = env_cfg.actions.joint_pos
    if not hasattr(action_cfg, "min_action_delay_steps"):
        raise RuntimeError("Student action term does not expose delay control")
    action_cfg.min_action_delay_steps = ACTION_DELAY_STEPS
    action_cfg.max_action_delay_steps = ACTION_DELAY_STEPS
    if int(action_cfg.min_action_delay_steps) != 0 or int(action_cfg.max_action_delay_steps) != 0:
        raise RuntimeError("failed to force action delay to zero")

    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.resampling_time_range = (1_000_000.0, 1_000_000.0)
    random_events_still_enabled = [
        name for name in DISABLED_RANDOM_EVENTS if getattr(events, name, None) is not None
    ]
    return {
        "observation_corruption_enabled": corruption,
        "disabled_random_events": disabled,
        "play_random_events_enabled": random_events_still_enabled,
        "action_cfg": action_cfg,
    }


def _sync_fixed_command(env: Any, torch: Any) -> None:
    term = env.unwrapped.command_manager._terms.get("base_velocity")
    if term is None:
        raise RuntimeError("runtime has no base_velocity command term")
    command = torch.tensor(FIXED_COMMAND, device=env.device, dtype=torch.float32).unsqueeze(0)
    wrote = False
    for attribute in ("command", "vel_command_b"):
        value = getattr(term, attribute, None)
        if value is not None:
            value[:] = command
            wrote = True
    if not wrote:
        raise RuntimeError("cannot synchronize fixed command into command manager")
    observed = env.unwrapped.command_manager.get_command("base_velocity")
    if observed.shape != (1, 3) or not torch.equal(observed, command.to(observed)):
        raise RuntimeError("command manager does not expose the exact fixed [0.45,0,0] command")


def _assert_identity_normalization(runner: Any, observations: Any, torch: Any) -> dict[str, Any]:
    policy_module = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
    if policy_module is None:
        raise RuntimeError("runner exposes no policy module")
    policy_obs = observations["policy"]
    checked: list[str] = []
    for name in ("actor_obs_normalizer", "student_obs_normalizer"):
        normalizer = getattr(policy_module, name, None)
        if normalizer is None:
            continue
        normalized = normalizer(policy_obs)
        if not torch.equal(normalized, policy_obs):
            raise RuntimeError(f"manual-vs-runner audit requires identity normalization; {name} changed observations")
        checked.append(name)
    return {
        "agent_empirical_normalization": False,
        "runtime_identity_normalizers_checked": checked,
        "runner_manual_equality_checked_every_frame": True,
    }


def _flat_dim(value: Any) -> int:
    if isinstance(value, int):
        return value
    result = 1
    for item in value:
        result *= int(item)
    return result


def _assert_observation_semantics(env: Any, observations: Any) -> dict[str, Any]:
    """Bind tensor slices to the saved B500/Teacher semantic observation layout."""
    saved_hashes = {
        "b500_env_yaml": sha256_file(B500_ENV_CONFIG.resolve(strict=True)),
        "b500_agent_yaml": sha256_file(B500_AGENT_CONFIG.resolve(strict=True)),
    }
    expected_hashes = {
        "b500_env_yaml": B500_ENV_CONFIG_SHA256,
        "b500_agent_yaml": B500_AGENT_CONFIG_SHA256,
    }
    if saved_hashes != expected_hashes:
        raise RuntimeError(f"saved B500 config binding changed: {saved_hashes} != {expected_hashes}")

    manager = env.unwrapped.observation_manager
    evidence: dict[str, Any] = {
        "saved_config_sha256": saved_hashes,
        "groups": {},
        "teacher_critic_privileged_slice": "critic[:,3:]",
        "teacher_critic_privileged_dim": 159,
        "teacher_velocity_label_slice": "critic[:,:3]",
        "teacher_velocity_label_scale": 2.0,
    }
    for group_name, contract in OBSERVATION_CONTRACT.items():
        names = tuple(manager._group_obs_term_names[group_name])
        dims = tuple(_flat_dim(value) for value in manager._group_obs_term_dim[group_name])
        term_cfgs = tuple(manager._group_obs_term_cfgs[group_name])
        history_lengths = tuple(int(getattr(cfg, "history_length", 0) or 0) for cfg in term_cfgs)
        flatten_history = tuple(bool(getattr(cfg, "flatten_history_dim", False)) for cfg in term_cfgs)
        # ObservationTermCfg uses ``None`` for the identity scale.  Normalize
        # that representation before comparing with the numeric saved contract.
        scales = tuple(
            1.0 if getattr(cfg, "scale", None) is None else float(cfg.scale)
            for cfg in term_cfgs
        )
        functions = tuple(
            "fixed_velocity_command_override"
            if name == "velocity_commands"
            else f"{cfg.func.__module__}:{cfg.func.__name__}"
            for name, cfg in zip(names, term_cfgs)
        )
        if names != contract["names"]:
            raise RuntimeError(f"{group_name} observation term order mismatch: {names}")
        if dims != contract["dims"]:
            raise RuntimeError(f"{group_name} observation term dimensions mismatch: {dims}")
        if history_lengths != contract["history_lengths"]:
            raise RuntimeError(f"{group_name} observation history mismatch: {history_lengths}")
        if not all(flatten_history):
            raise RuntimeError(f"{group_name} observation history is not flattened: {flatten_history}")
        if len(scales) != len(contract["scales"]) or any(
            abs(actual - expected) > 1.0e-8
            for actual, expected in zip(scales, contract["scales"])
        ):
            raise RuntimeError(f"{group_name} observation scale mismatch: {scales}")
        if functions != contract["functions"]:
            raise RuntimeError(f"{group_name} observation function mismatch: {functions}")
        for term_name in ("joint_pos", "joint_vel"):
            term_index = names.index(term_name)
            asset_cfg = term_cfgs[term_index].params.get("asset_cfg")
            if asset_cfg is None or tuple(asset_cfg.joint_names) != ACTION_JOINT_ORDER:
                raise RuntimeError(
                    f"{group_name}.{term_name} joint order differs from the checkpoint contract"
                )
        total = sum(dims)
        tensor = observations[group_name]
        if tuple(tensor.shape) != (1, total):
            raise RuntimeError(
                f"{group_name} observation tensor shape mismatch: {tuple(tensor.shape)} != {(1, total)}"
            )
        group_evidence: dict[str, Any] = {
            "term_names": list(names),
            "flattened_term_dims": list(dims),
            "history_lengths": list(history_lengths),
            "flatten_history_dim": list(flatten_history),
            "scales": list(scales),
            "functions": list(functions),
            "total_dim": total,
        }
        if group_name == "critic":
            if total != 162 or total - dims[0] != 159:
                raise RuntimeError("critic semantic split must be 3-D velocity + 159-D privileged input")
        elif total != 570:
            raise RuntimeError(f"{group_name} must be the checkpoint-compatible 570-D history")
        evidence["groups"][group_name] = group_evidence
    return evidence


def _assert_suite_runtime_contract(
    plan_payload: Mapping[str, Any],
    runtime_checks: Mapping[str, Any],
    real_gains: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    action_cfg = runtime_checks["action_cfg"]
    observed = {
        "task": TASK,
        "num_envs": NUM_ENVS,
        "terrain_type": TERRAIN_TYPE,
        "terrain_level": TERRAIN_LEVEL,
        "fixed_velocity_command": list(FIXED_COMMAND),
        "front_step_eval_side": FRONT_STEP_SIDE,
        "front_step_eval_edge_gap": FRONT_STEP_EDGE_GAP,
        "play_max_steps": MAX_STEPS,
        "eval_action_delay_steps": ACTION_DELAY_STEPS,
        "observation_corruption_enabled": any(runtime_checks["observation_corruption_enabled"].values()),
        "play_random_events_enabled": bool(runtime_checks["play_random_events_enabled"]),
        "real_gain_center": {
            "hip": list(real_gains["legs_hip"]),
            "thigh": list(real_gains["legs_thigh"]),
            "calf": list(real_gains["legs_calf"]),
        },
        "student_action_prior_enabled": bool(
            hasattr(action_cfg, "front_reach_box_bias") or hasattr(action_cfg, "prior_start_update")
        ),
        "causal_replacement": None,
    }
    expected = plan_payload["runtime_contract"]
    if observed != expected:
        raise RuntimeError(f"runtime differs from immutable base3 suite plan: {observed} != {expected}")
    return observed


def _runtime_code_hashes() -> dict[str, str]:
    paths = {
        "same_state_runtime": Path(__file__).resolve(),
        "same_state_tool": AUDIT_TOOL_PATH,
        "play_core9_contract": PLAY_PATH,
        "highstep_env_cfg": ENV_CFG_PATH,
        "velocity_env_cfg": VELOCITY_CFG_PATH,
        "observations": OBSERVATIONS_PATH,
        "events": EVENTS_PATH,
        "actions": ACTIONS_PATH,
        "highstep_schedule": SCHEDULE_PATH,
        "agent_cfg": AGENT_CFG_PATH,
        "vae_ppo": VAE_PPO_PATH,
        "rsl_cli_args": CLI_ARGS_PATH,
        "b500_saved_env_config": B500_ENV_CONFIG,
        "b500_saved_agent_config": B500_AGENT_CONFIG,
    }
    return {str(path): sha256_file(path) for path in paths.values()}


def _ensure_runtime_binding(
    output_root: Path,
    suite_plan_path: Path,
    suite_plan_sha256: str,
) -> tuple[Path, str]:
    path = output_root.expanduser().resolve() / "base3_runtime_binding.json"
    payload = {
        "schema_version": 1,
        "kind": "highstep_same_state_base3_runtime_binding",
        "immutable_after_write": True,
        "suite_plan_path": str(suite_plan_path),
        "suite_plan_sha256": suite_plan_sha256,
        "checkpoint_path": str(B500_CHECKPOINT.resolve()),
        "checkpoint_sha256": B500_SHA256,
        "runtime_code_and_config_sha256": _runtime_code_hashes(),
        "core9_play_contract_nodes": list(PLAY_CONTRACT_NODES),
        "core9_play_sha256": PLAY_SHA256,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"base3 runtime binding changed between trajectories: {path}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"base3 runtime binding is writable: {path}")
    else:
        _atomic_json(path, payload, read_only=True)
    return path, sha256_file(path)


def _read_checkpoint_iteration(checkpoint: Path, torch: Any, checkpoint_iteration_from_mapping: Any) -> int:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    try:
        return int(checkpoint_iteration_from_mapping(payload))
    finally:
        del payload


def _run(args: argparse.Namespace, trajectory: Trajectory, simulation_app: Any) -> int:
    # Isaac/robot task imports are intentionally delayed until after AppLauncher.
    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab.utils.math as math_utils
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    import robot_lab.tasks  # noqa: F401
    from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.vae_ppo import (
        VAEActorCritic,
        VAEPPO,
    )
    import rsl_rl.algorithms.ppo as rsl_ppo
    import rsl_rl.modules.actor_critic as rsl_actor_critic
    import rsl_rl.runners.on_policy_runner as rsl_runner
    from robot_lab.tasks.locomotion.velocity.mdp.highstep_schedule import (
        assert_schedule_definition_compatible,
        build_schedule_manifest,
        checkpoint_iteration_from_mapping,
        eval_schedule_fields,
        install_global_update,
        load_source_manifest,
        resolve_loaded_schedule_update,
        runtime_schedule_match,
        schedule_clock_state,
        schedule_definition_from_configs,
    )

    # Register the same custom policy/algorithm classes as play.py.
    rsl_actor_critic.VAEActorCritic = VAEActorCritic
    rsl_ppo.VAEPPO = VAEPPO
    rsl_runner.VAEActorCritic = VAEActorCritic
    rsl_runner.VAEPPO = VAEPPO

    sys.path.insert(0, str(ROOT))
    from tools import highstep_same_state_audit as audit

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cli_args

    play = _load_exact_play_contract(np=np, torch=torch, math_utils=math_utils)
    plan_path, plan_sha = _ensure_base_suite_plan(Path(args.output_root), audit)
    runtime_binding_path, runtime_binding_sha = _ensure_runtime_binding(
        Path(args.output_root), plan_path, plan_sha
    )
    output_dir = Path(args.output_root).expanduser().resolve() / trajectory.run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite same-state trajectory: {output_dir}")

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=NUM_ENVS, use_fabric=not args.disable_fabric)
    agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, args)
    if bool(getattr(agent_cfg, "empirical_normalization", True)):
        raise RuntimeError("same-state manual-vs-runner contract requires empirical_normalization=False")
    seed = trajectory.seed
    env_cfg.seed = seed
    agent_cfg.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    terrain_schedule_cfg = getattr(getattr(env_cfg, "curriculum", None), "terrain_levels", None)
    support_schedule_cfg = getattr(getattr(env_cfg, "curriculum", None), "highstep_action_score", None)
    command_schedule_cfg = getattr(getattr(env_cfg, "curriculum", None), "command_levels", None)
    reward_stage_definition = play._highstep_reward_stage_definition(env_cfg)
    real_gains = _assert_real_gain_config(env_cfg)
    runtime_checks = _configure_fixed_environment(env_cfg, trajectory, ObsTerm, torch)
    suite_plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    runtime_contract = _assert_suite_runtime_contract(
        suite_plan_payload, runtime_checks, real_gains
    )

    raw_env = gym.make(TASK, cfg=env_cfg)
    play._apply_play_terrain_selection(raw_env, TERRAIN_LEVEL, TERRAIN_TYPE)

    checkpoint_iteration = _read_checkpoint_iteration(
        B500_CHECKPOINT, torch, checkpoint_iteration_from_mapping
    )
    schedule_update, schedule_source = resolve_loaded_schedule_update(
        B500_CHECKPOINT,
        checkpoint_iteration,
        checkpoint_load_mode="full",
        schedule_resume_mode="preserve",
        allow_legacy_checkpoint_fallback=False,
    )
    install_global_update(
        raw_env,
        schedule_update,
        source=str(schedule_source.get("method", "unknown")),
        resume_mode="preserve",
        runner_iteration=checkpoint_iteration,
        advance_with_local_steps=False,
    )
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise RuntimeError(f"B500 audit requires OnPolicyRunner, got {agent_cfg.class_name!r}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(B500_CHECKPOINT), load_optimizer=False)
    if int(runner.current_learning_iteration) != checkpoint_iteration:
        raise RuntimeError("runner iteration differs from the bound B500 checkpoint metadata")
    install_global_update(
        env,
        schedule_update,
        source=str(schedule_source.get("method", "unknown")),
        resume_mode="preserve",
        runner_iteration=checkpoint_iteration,
        advance_with_local_steps=False,
    )

    source_manifest_path, source_manifest = load_source_manifest(B500_CHECKPOINT)
    if source_manifest is None:
        raise RuntimeError("B500 schedule source manifest is required; legacy fallback is forbidden")
    current_schedule_definition = schedule_definition_from_configs(
        action_cfg=env_cfg.actions.joint_pos,
        terrain_term_cfg=terrain_schedule_cfg,
        support_term_cfg=support_schedule_cfg,
        command_term_cfg=command_schedule_cfg,
        command_curriculum_enabled=command_schedule_cfg is not None,
        reward_stage_definition=reward_stage_definition,
    )
    assert_schedule_definition_compatible(source_manifest, current_schedule_definition)
    schedule_manifest = build_schedule_manifest(
        context="same_state_audit",
        task=TASK,
        env=env,
        action_cfg=env_cfg.actions.joint_pos,
        terrain_term_cfg=terrain_schedule_cfg,
        support_term_cfg=support_schedule_cfg,
        terrain_curriculum_enabled=False,
        support_metric_enabled=support_schedule_cfg is not None,
        checkpoint_path=B500_CHECKPOINT,
        checkpoint_iteration=checkpoint_iteration,
        checkpoint_load_mode="full",
        requested_resume_mode="preserve",
        resolved_resume_mode="preserve",
        runner_iteration_at_anchor=checkpoint_iteration,
        schedule_update_at_anchor=schedule_update,
        schedule_source=schedule_source,
        command_term_cfg=command_schedule_cfg,
        command_curriculum_enabled=command_schedule_cfg is not None,
        reward_stage_definition=reward_stage_definition,
    )

    # Match core9 reset ordering exactly: wrapper reset, then deterministic front reset.
    env.reset()
    platform_width = play._front_step_platform_width(env, TERRAIN_TYPE, None)
    play._apply_front_step_eval_reset(
        env,
        FRONT_STEP_SIDE,
        None,
        FRONT_STEP_EDGE_GAP,
        platform_width,
        trajectory.lateral_offset_m,
        trajectory.yaw_offset_deg,
        TERRAIN_TYPE,
    )
    context = getattr(env.unwrapped, "_front_step_eval_context", None)
    if not context or not bool(context.get("reset_valid", False)):
        raise RuntimeError("core9 front-step reset did not produce a valid context")
    _sync_fixed_command(env, torch)
    observations = env.get_observations()

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observation_semantics = _assert_observation_semantics(env, observations)
    normalization = _assert_identity_normalization(runner, observations, torch)
    prior_contract = audit.PostPriorContract.from_teacher_env_config()
    models = audit.BoundAuditModels(device=env.unwrapped.device)
    preregistration = audit.AuditPreregistration(
        run_id=trajectory.run_id,
        seed=trajectory.seed,
        scenario=trajectory.scenario,
        suite_plan_path=str(plan_path),
        suite_plan_sha256=plan_sha,
        suite_kind="base3",
        lateral_offset_m=trajectory.lateral_offset_m,
        yaw_offset_deg=trajectory.yaw_offset_deg,
        replacement=None,
        all_replacement_trials=(),
    )
    session = audit.SameStateAuditSession(
        models=models,
        output_dir=output_dir,
        preregistration=preregistration,
        prior_contract=prior_contract,
        driver_action_source=AUDIT_DRIVER_ACTION_SOURCE,
    )
    resolved_env_config_path = output_dir / "resolved_env_cfg.yaml"
    resolved_agent_config_path = output_dir / "resolved_agent_cfg.yaml"
    dump_yaml(str(resolved_env_config_path), env_cfg)
    dump_yaml(str(resolved_agent_config_path), agent_cfg)
    adapter = audit.HighstepRuntimeAdapter(env, contract=prior_contract)
    tracker = play._HighstepEvalTracker(env)

    # The adapter derives masks from the frozen Teacher env YAML.  Verify that
    # those exact masks are non-empty and are the only masks used for targets.
    if adapter.height_sensor is None or adapter.front_ray_mask is None or adapter.rear_ray_mask is None:
        raise RuntimeError("Teacher-configured terrain prior ray masks are unavailable")
    ray_contract = {
        "teacher_env_config": prior_contract.source_env_config_path,
        "teacher_env_config_sha256": prior_contract.source_env_config_sha256,
        "front_x_min": prior_contract.front_x_min,
        "rear_x_max": prior_contract.rear_x_max,
        "max_abs_y": prior_contract.max_abs_y,
        "front_ray_count": int(torch.sum(adapter.front_ray_mask).item()),
        "rear_ray_count": int(torch.sum(adapter.rear_ray_mask).item()),
    }

    timestep = 0
    terminated = False
    try:
        while simulation_app.is_running() and timestep < MAX_STEPS:
            with torch.inference_mode():
                _sync_fixed_command(env, torch)
                observations = env.get_observations()
                action_to_apply = session.evaluate_runtime_frame(
                    step=timestep,
                    adapter=adapter,
                    runner_policy=policy,
                    observations=observations,
                )
                action_for_limit_audit = action_to_apply
                if getattr(env, "clip_actions", None) is not None:
                    action_for_limit_audit = torch.clamp(
                        action_to_apply, -env.clip_actions, env.clip_actions
                    )
                tracker.audit_policy_actions(action_for_limit_audit, timestep)
                observations, _, dones, _ = env.step(action_to_apply)
                terminated = bool(torch.any(dones).item())
                if terminated:
                    tracker.mark_terminated(timestep)
                else:
                    tracker.update(timestep)
            timestep += 1
            if terminated:
                break

        if timestep <= 0:
            raise RuntimeError("same-state audit produced no frames")
        outcome = tracker.summary()
        delay_audit = play._runtime_eval_action_delay(env, ACTION_DELAY_STEPS)
        if delay_audit["eval_action_delay_runtime_match"] is not True:
            raise RuntimeError(f"runtime action delay contract failed: {delay_audit}")
        outcome.update(delay_audit)
        outcome.update(eval_schedule_fields(schedule_manifest))
        runtime_clock = schedule_clock_state(env.unwrapped, 24)
        schedule_match = runtime_schedule_match(
            schedule_manifest,
            runtime_prior_scale=None,
            runtime_support_blend=getattr(env.unwrapped, "_highstep_support_bottleneck_blend", None),
        )
        schedule_clock_match = bool(
            abs(float(runtime_clock["global_update"]) - float(schedule_update)) <= 1.0e-9
            and runtime_clock["advance_with_local_steps"] is False
            and runtime_clock["source"] == schedule_source.get("method")
            and runtime_clock["resume_mode"] == "preserve"
            and runtime_clock["runner_iteration_at_anchor"] == checkpoint_iteration
        )
        if schedule_match["schedule_runtime_match"] is not True or not schedule_clock_match:
            raise RuntimeError(
                f"same-state schedule runtime contract failed: match={schedule_match}, clock={runtime_clock}"
            )
        rebound_path, rebound_sha = _ensure_runtime_binding(
            Path(args.output_root), plan_path, plan_sha
        )
        if rebound_path != runtime_binding_path or rebound_sha != runtime_binding_sha:
            raise RuntimeError("base3 runtime binding changed during the trajectory")
        outcome.update(
            {
                "loop_steps": timestep,
                "requested_play_max_steps": MAX_STEPS,
                "fixed_velocity_command": list(FIXED_COMMAND),
                "keep_play_randomization": False,
                "front_step_eval_side": FRONT_STEP_SIDE,
                "front_step_eval_edge_gap": FRONT_STEP_EDGE_GAP,
                "checkpoint_sha256": B500_SHA256,
                "terminated": terminated,
                "same_state_runtime_contract_verified": True,
                "same_state_runtime_binding_path": str(runtime_binding_path),
                "same_state_runtime_binding_sha256": runtime_binding_sha,
                "action_prior_enabled": False,
                "schedule_valid": True,
                "schedule_clock_runtime_match": schedule_clock_match,
                "schedule_clock": runtime_clock,
                "schedule_runtime_match_fields": schedule_match,
            }
        )
        run_summary = session.finalize(outcome=outcome)

        environment_manifest = {
            "schema_version": 1,
            "kind": "highstep_same_state_runtime_environment",
            "trajectory": trajectory.__dict__,
            "suite_plan_path": str(plan_path),
            "suite_plan_sha256": plan_sha,
            "runtime_binding_path": str(runtime_binding_path),
            "runtime_binding_sha256": runtime_binding_sha,
            "runtime_contract": runtime_contract,
            "runtime_checks": {
                "observation_corruption_enabled_by_group": runtime_checks[
                    "observation_corruption_enabled"
                ],
                "disabled_random_events": runtime_checks["disabled_random_events"],
                "play_random_events_still_enabled": runtime_checks[
                    "play_random_events_enabled"
                ],
            },
            "real_gain_contract": real_gains,
            "observation_semantics": observation_semantics,
            "normalization": normalization,
            "teacher_post_prior_contract": prior_contract.__dict__,
            "teacher_ray_mask_contract": ray_contract,
            "checkpoint": str(B500_CHECKPOINT.resolve()),
            "checkpoint_sha256": B500_SHA256,
            "checkpoint_iteration": checkpoint_iteration,
            "resolved_env_config": str(resolved_env_config_path),
            "resolved_env_config_sha256": sha256_file(resolved_env_config_path),
            "resolved_agent_config": str(resolved_agent_config_path),
            "resolved_agent_config_sha256": sha256_file(resolved_agent_config_path),
            "source_schedule_manifest": str(source_manifest_path),
            "schedule_manifest": schedule_manifest,
            "code_sha256": _runtime_code_hashes(),
            "core9_play_contract_nodes": list(PLAY_CONTRACT_NODES),
            "core9_play_sha256": PLAY_SHA256,
            "run_summary": str(output_dir / "run_summary.json"),
            "run_summary_frame_count": run_summary["frame_count"],
            "only_session_action_entered_env_step": True,
            "teacher_environment_steps": 0,
            "training_allowed": False,
        }
        _atomic_json(output_dir / "runtime_environment.json", environment_manifest)
        print(
            "[HIGHSTEP_SAME_STATE_AUDIT_JSON] "
            + json.dumps(
                {
                    "run_id": trajectory.run_id,
                    "frame_count": run_summary["frame_count"],
                    "phase_coverage": run_summary["phase_coverage"],
                    "outcome_full_climb_success": outcome.get("full_climb_success"),
                    "outcome_rear_hold_success": outcome.get("rear_on_platform_hold_success"),
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        env.close()


def _build_parser() -> argparse.ArgumentParser:
    # AppLauncher and cli_args imports are intentionally local so CPU-only tests
    # can import this file without importing Isaac Sim or torch.
    from isaaclab.app import AppLauncher

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cli_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, choices=tuple(TRAJECTORIES))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(checkpoint=str(B500_CHECKPOINT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    args = parser.parse_args(argv)
    trajectory = _validate_fixed_cli(args)
    # parse_rsl_rl_cfg consumes seed if present; it is never a user override.
    args.seed = trajectory.seed
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return _run(args, trajectory, simulation_app)
    except BaseException:
        # Isaac Sim unloads its logging plugins during close(); print the
        # original traceback first so a startup failure cannot look successful.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
