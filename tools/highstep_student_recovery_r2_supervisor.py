#!/usr/bin/env python3
"""Fail-closed v1.1.1 R2 highstep Student recovery supervisor.

This module owns the authority checks, immutable launch latch, discarded
fresh4/full+1 smoke, checkpoint-scope audit and preregistered behavior gates.
GPU work is performed only from :class:`R2Supervisor`; importing this module is
CPU-only and safe for unit tests.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
WORKFLOW_ID = "highstep_student_recovery_v11_20260712"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA256 = "7dbb47d76b5eafab0e425be7ac20b5a0e9487466c6512831600f22fb7509bbe5"
PREREGISTRATION = ROOT / "tmp/highstep_student_recovery_v11_20260712/r2_preregistration.json"
PREREGISTRATION_SHA256 = "36d39316f8fbba407899a14d1d659873f75b33423464f01ea92000e588c56fc5"
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_v11_20260712"

TRAIN_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPriorR2-"
    "ArcdogAdjustableLeg-v0"
)
EVAL_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-"
    "ArcdogAdjustableLeg-v0"
)
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_recovery_r2_Student"
)
PARENT_TEACHER_MANIFEST = (
    ROOT / "tmp/highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json"
)
PARENT_TEACHER_MANIFEST_SHA256 = (
    "8b560be8778f76510cb807b95622b2b989dadef8985e90dd9835931d9622b60e"
)
SCHEMA4_MONITOR = ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh"
R2_MONITOR_SNAPSHOT = STATE_ROOT / "authority/highstep_r2_schema4_monitor.sh"
PINNED_BASE3_ROOT = STATE_ROOT / "same_state_base3_retry5"
CANDIDATE_AUDIT_ADAPTER = ROOT / "scripts/rsl_rl/base/highstep_candidate_same_state_audit_play.py"
PINNED_B500_REVOLUTE_MAE = 0.20432990000048584
PINNED_B500_BOX_MAE = 0.62325989688119277

PROTECTED_ROOT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_04-41-42_robust_student_distill_20260712_044124/model_900.pt"
)
PROTECTED_ROOT_SHA256 = "9bbd5b597d9c195ecf9afb141152b8f0749dc599a54868674b299107fb40a229"
B500 = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_23-55-22_student_recovery_v111_B500_20260712_235515/model_498.pt"
)
B500_SHA256 = "f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA256 = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
TEACHER_ENV = TEACHER.parent / "params/env.yaml"
TEACHER_ENV_SHA256 = "f8aa66bbc417eb67a7889900e4a1781eb2852cb3d94432a1282963cc0909c009"

ALGORITHM_STATE_KEY = "robot_lab_algorithm_checkpoint_state"
R2_OPTIMIZER_NAMES = (
    "estimator.encoder.0.weight",
    "estimator.encoder.0.bias",
    "estimator.encoder.2.weight",
    "estimator.encoder.2.bias",
    "estimator.fc_mu.weight",
    "estimator.fc_mu.bias",
    "algorithm.r2_box_weight",
    "algorithm.r2_box_bias",
)
R2_LIVE_FULL_TENSORS = frozenset(R2_OPTIMIZER_NAMES[:6])
R2_ACTOR_ROW_TENSORS = frozenset(("actor.6.weight", "actor.6.bias"))
JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)
ACTION_SCALE = (0.1,) * 12 + (0.02,) * 4
ACTION_OFFSET = (0.0,) * 4 + (0.7,) * 4 + (-1.3,) * 4 + (0.03,) * 4


def now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def is_read_only(path: Path) -> bool:
    return not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def exact_numbers(value: Any, expected: Sequence[float], tolerance: float = 1.0e-6) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(expected)
        and all(finite_number(item) for item in value)
        and all(abs(float(item) - float(wanted)) <= tolerance for item, wanted in zip(value, expected, strict=True))
    )


def _same_scalar(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if finite_number(left) and finite_number(right):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def validate_r2_eval_schedule_pair(
    payload: Mapping[str, Any], checkpoint: Path
) -> dict[str, Any]:
    """Validate only the approved R2-train -> unchanged deployment-eval alias."""

    manifest_path = Path(str(payload.get("schedule_manifest_path", ""))).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError("R2 evaluation schedule manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("schedule_source")
    if not isinstance(source, Mapping) or source.get("method") != "runtime_snapshot_exact":
        raise RuntimeError("R2 evaluation did not use an exact runtime snapshot")
    source_manifest_path = Path(str(source.get("source_manifest", ""))).expanduser().resolve()
    source_runtime_path = Path(str(source.get("source_runtime_state", ""))).expanduser().resolve()
    if not source_manifest_path.is_file() or not source_runtime_path.is_file():
        raise RuntimeError("R2 evaluation source schedule sidecars are missing")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    checkpoint = checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    try:
        checkpoint_iteration = int(checkpoint.stem.removeprefix("model_"))
    except ValueError as error:
        raise RuntimeError("R2 checkpoint filename does not bind a runner iteration") from error
    runtime_snapshot = source.get("runtime_snapshot")
    if not isinstance(runtime_snapshot, Mapping):
        raise RuntimeError("R2 schedule source lacks a runtime snapshot")
    expected_lineage = {
        "parent_teacher_manifest_path": str(PARENT_TEACHER_MANIFEST.resolve()),
        "parent_teacher_manifest_sha256": PARENT_TEACHER_MANIFEST_SHA256,
        "selected_teacher_checkpoint_path": str(TEACHER.resolve()),
        "selected_teacher_checkpoint_sha256": TEACHER_SHA256,
    }
    code_sha = payload.get("evaluation_code_sha256")
    expected_code = {
        "actions.py": sha256_file(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"),
        "curriculums.py": sha256_file(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py"),
        "highstep_env_cfg.py": sha256_file(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py"),
        "highstep_schedule.py": sha256_file(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"),
        "play.py": sha256_file(ROOT / "scripts/rsl_rl/base/play.py"),
        "rewards.py": sha256_file(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py"),
    }
    source_checkpoint_value = source_manifest.get("checkpoint_path")
    source_checkpoint_valid = source_checkpoint_value is None and source_manifest.get("checkpoint_sha256") is None
    if source_checkpoint_value:
        source_checkpoint = Path(str(source_checkpoint_value)).expanduser().resolve()
        source_checkpoint_valid = bool(
            source_checkpoint.is_file()
            and source_manifest.get("checkpoint_sha256") == sha256_file(source_checkpoint)
        )
    valid = bool(
        isinstance(manifest, Mapping)
        and isinstance(manifest.get("schema_version"), int)
        and manifest.get("schema_version") >= 3
        and manifest.get("context") == "play"
        and manifest.get("task") == EVAL_TASK
        and Path(str(manifest.get("checkpoint_path"))).expanduser().resolve() == checkpoint
        and manifest.get("checkpoint_sha256") == checkpoint_sha
        and manifest.get("checkpoint_iteration") == checkpoint_iteration
        and manifest.get("runner_iteration_at_anchor") == checkpoint_iteration
        and manifest.get("checkpoint_load_mode") == "full"
        and manifest.get("schedule_resume_mode_requested") == "preserve"
        and manifest.get("schedule_resume_mode_resolved") == "preserve"
        and manifest.get("advance_with_local_steps") is False
        and manifest.get("runtime_state_required_for_preserve") is False
        and manifest.get("source_manifest_checked") is True
        and Path(str(manifest.get("source_manifest_path_checked"))).expanduser().resolve()
        == source_manifest_path
        and source.get("runtime_snapshot_checkpoint_sha256") == checkpoint_sha
        and source.get("runtime_snapshot_checkpoint_sha256_verified") is True
        and source.get("legacy_fallback_used") is False
        and runtime_snapshot.get("checkpoint_file") == checkpoint.name
        and runtime_snapshot.get("checkpoint_sha256") == checkpoint_sha
        and runtime_snapshot.get("runner_iteration") == checkpoint_iteration
        and source_manifest.get("context") == "train"
        and source_manifest.get("task") == TRAIN_TASK
        and source_manifest.get("runtime_state_required_for_preserve") is True
        and source_manifest.get("student_parent_lineage") == expected_lineage
        and source_checkpoint_valid
        and source_manifest.get("schedule_definition") == manifest.get("schedule_definition")
        and code_sha == expected_code
        and manifest.get("evaluation_code_sha256") == expected_code
        and payload.get("runtime_snapshot_checkpoint_sha256_verified") is True
        and payload.get("runtime_snapshot_checkpoint_sha256") == checkpoint_sha
        and payload.get("schedule_source_method") == "runtime_snapshot_exact"
        and payload.get("schedule_manifest_schema_version") == manifest.get("schema_version")
        and payload.get("schedule_runtime_runner_iteration_at_anchor") == checkpoint_iteration
        and _same_scalar(payload.get("global_update"), manifest.get("schedule_update_at_anchor"))
        and _same_scalar(
            payload.get("schedule_runtime_global_update"), manifest.get("schedule_update_at_anchor")
        )
        and manifest.get("action_prior", {}).get("enabled") is payload.get("action_prior_enabled")
        and _same_scalar(
            manifest.get("action_prior", {}).get("actual_prior_scale"),
            payload.get("action_prior_scale_manifest"),
        )
        and manifest.get("play_selection", {}).get("requested_terrain_level") == 9
        and manifest.get("play_selection", {}).get("requested_terrain_type") == "box"
        and manifest.get("play_selection", {}).get("env0_terrain_level") == 9
        and manifest.get("play_selection", {}).get("env0_terrain_type")
        == manifest.get("play_selection", {}).get("requested_terrain_type_column")
    )
    if not valid:
        raise RuntimeError("R2 evaluation schedule/task/lineage/runtime binding is invalid")
    return {
        "play_task": EVAL_TASK,
        "source_train_task": TRAIN_TASK,
        "checkpoint_sha256": checkpoint_sha,
        "source_manifest": str(source_manifest_path),
        "source_runtime_state": str(source_runtime_path),
        "student_parent_lineage": expected_lineage,
        "all_other_schema4_schedule_checks_preserved": True,
    }


def validate_probe_payload(payload: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    checkpoint_sha = sha256_file(checkpoint)
    samples = payload.get("samples")
    geometry = payload.get("geometry_samples")
    loop_steps = payload.get("loop_steps")
    terminated = payload.get("terminated_early")
    if terminated is False:
        termination_valid = geometry == samples and loop_steps == 600
    elif terminated is True:
        termination_valid = (
            finite_number(geometry)
            and finite_number(samples)
            and float(geometry) + 1 == float(samples)
            and finite_number(payload.get("termination_step"))
            and float(payload["termination_step"]) + 1 == float(loop_steps)
        )
    else:
        termination_valid = False
    target_contract = payload.get("target_limit_contract")
    target_contract_valid = bool(
        isinstance(target_contract, Mapping)
        and set(target_contract) == set(JOINT_ORDER)
        and all(exact_numbers(value, (-60.0, 60.0)) for value in target_contract.values())
    )
    valid = bool(
        isinstance(payload.get("schema_version"), int)
        and payload.get("schema_version") >= 7
        and payload.get("checkpoint_sha256") == checkpoint_sha
        and payload.get("reset_context_valid") is True
        and payload.get("reset_valid") is True
        and payload.get("initial_geometry_valid") is True
        and payload.get("initial_no_top_contact") is True
        and payload.get("initial_all_feet_outside_platform") is True
        and finite_number(samples)
        and float(samples) > 0
        and payload.get("requested_play_max_steps") == 600
        and loop_steps == samples
        and payload.get("action_samples") == samples
        and payload.get("target_limit_sample_steps") == samples
        and termination_valid
        and payload.get("eval_action_delay_requested") == 0
        and payload.get("eval_action_delay_steps_runtime") == 0
        and payload.get("eval_action_delay_runtime_match") is True
        and payload.get("keep_play_randomization") is False
        and exact_numbers(payload.get("fixed_velocity_command"), (0.45, 0.0, 0.0))
        and payload.get("front_step_eval_side") == "x-"
        and _same_scalar(payload.get("front_step_eval_edge_gap"), 0.55)
        and payload.get("target_action_order_valid") is True
        and payload.get("target_action_joint_names") == list(JOINT_ORDER)
        and payload.get("target_action_asset_joint_ids") == list(range(16))
        and exact_numbers(payload.get("target_action_scale"), ACTION_SCALE)
        and exact_numbers(payload.get("target_action_offset"), ACTION_OFFSET)
        and payload.get("target_limit_contract_valid") is True
        and target_contract_valid
        and payload.get("target_limit_action_input_valid") is True
        and payload.get("target_limit_invalid_action_steps") == 0
        and payload.get("schedule_valid") is True
        and payload.get("schedule_runtime_match") is True
        and payload.get("schedule_clock_runtime_match") is True
        and payload.get("schedule_runtime_frozen") is True
        and payload.get("schedule_frozen_for_evaluation") is True
        and payload.get("schedule_runtime_resume_mode") == "preserve"
        and payload.get("action_prior_runtime_match") is True
        and payload.get("support_runtime_match") is True
    )
    if not valid:
        raise RuntimeError("R2 probe play payload failed schema4 action/reset/runtime validity")
    schedule = validate_r2_eval_schedule_pair(payload, checkpoint)
    no_severe = bool(
        finite_number(payload.get("critical_rear_min_abs_y_q05"))
        and float(payload["critical_rear_min_abs_y_q05"]) >= 0.04
        and finite_number(payload.get("critical_rear_width_q05"))
        and float(payload["critical_rear_width_q05"]) >= 0.18
        and finite_number(payload.get("critical_center_violation_rate"))
        and float(payload["critical_center_violation_rate"]) <= 0.10
        and finite_number(payload.get("critical_width_violation_rate"))
        and float(payload["critical_width_violation_rate"]) <= 0.10
    )
    return {
        "valid": True,
        "full_climb": payload.get("full_climb_success") is True,
        "rear_hold": payload.get("rear_on_platform_hold_success") is True,
        "front_top_support": payload.get("front_top_support_reached") is True,
        "no_severe_inward": no_severe,
        "schedule_binding": schedule,
    }


def probe_play_command(
    checkpoint: Path, *, seed: int, lateral: str, yaw: str
) -> list[str]:
    return [
        str(PYTHON),
        "scripts/rsl_rl/base/play.py",
        "--task", EVAL_TASK,
        "--num_envs", "1",
        "--headless",
        "--seed", str(seed),
        "--checkpoint", str(checkpoint.resolve()),
        "--play_terrain_type", "box",
        "--play_terrain_level", "9",
        "--fixed_velocity_command", "0.45", "0.0", "0.0",
        "--reset_after_play_terrain_selection",
        "--front_step_eval_reset",
        "--front_step_eval_side", "x-",
        "--front_step_eval_edge_gap", "0.55",
        "--front_step_eval_lateral_offset", lateral,
        "--front_step_eval_yaw_offset_deg", yaw,
        "--print_rear_width_metrics",
        "--skip_policy_export",
        "--rear_width_metric_interval", "60",
        "--play_max_steps", "600",
        "--eval_action_delay_steps", "0",
    ]


def core9_matrix_was_fully_executed(output: Path, checkpoint: Path) -> bool:
    rows_path = output / "eval_runs.jsonl"
    if not rows_path.is_file():
        return False
    try:
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError:
        return False
    expected = {
        (seed, scenario)
        for seed in (11, 22, 33)
        for scenario in ("nominal", "left_offset", "right_offset")
    }
    return bool(
        len(rows) == 9
        and {(row.get("seed"), row.get("scenario")) for row in rows} == expected
        and all(
            isinstance(row.get("checkpoint"), str)
            and Path(row["checkpoint"]).resolve() == checkpoint.resolve()
            for row in rows
        )
    )


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def behavior_rank(summary: Mapping[str, Any]) -> tuple[int, int]:
    full = int(summary["full_count"])
    hold = int(summary["rear_hold_count"])
    return min(full, hold), full + hold


def final_selection_rank(summary: Mapping[str, Any], updates: int) -> tuple[int, ...]:
    rank = behavior_rank(summary)
    return (
        rank[0],
        rank[1],
        int(summary["no_severe_inward_count"]),
        int(summary["front_top_support_count"]),
        -int(updates),
    )


def gate_probe3(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    expected = {
        (11, "nominal"),
        (11, "left_offset"),
        (22, "right_offset"),
    }
    identities = {(int(row["seed"]), str(row["scenario"])) for row in rows}
    if len(rows) != 3 or identities != expected or not all(row.get("valid") is True for row in rows):
        return False, "R2-50 probe3 valid/matrix gate failed"
    nominal = next(row for row in rows if (row["seed"], row["scenario"]) == (11, "nominal"))
    front = sum(row.get("front_top_support") is True for row in rows)
    no_severe = sum(row.get("no_severe_inward") is True for row in rows)
    passed = bool(
        nominal.get("full_climb") is True
        and nominal.get("rear_hold") is True
        and front >= 2
        and no_severe >= 2
    )
    return passed, (
        "R2-50 probe3 catastrophic filter passed"
        if passed
        else "R2-50 probe3 catastrophic filter failed"
    )


def gate_100(summary: Mapping[str, Any]) -> tuple[bool, str]:
    passed = bool(
        summary.get("valid_count") == 9
        and int(summary.get("full_count", -1)) >= 4
        and int(summary.get("rear_hold_count", -1)) >= 4
        and int(summary.get("front_top_support_count", -1)) >= 7
    )
    return passed, "R2-100 gate passed" if passed else "R2-100 hard gate failed"


def gate_300(summary: Mapping[str, Any]) -> tuple[bool, str]:
    passed = bool(
        summary.get("valid_count") == 9
        and max(
            int(summary.get("full_count", -1)),
            int(summary.get("rear_hold_count", -1)),
        )
        >= 5
        and int(summary.get("no_severe_inward_count", -1)) >= 7
    )
    return passed, "R2-300 gate passed" if passed else "R2-300 hard gate failed"


def gate_500(summary: Mapping[str, Any]) -> tuple[bool, str]:
    passed = bool(
        summary.get("valid_count") == 9
        and int(summary.get("full_count", -1)) >= 6
        and int(summary.get("rear_hold_count", -1)) >= 6
    )
    return passed, "R2-500 gate passed" if passed else "R2-500 hard gate failed"


def gate_enter_1000(
    summary100: Mapping[str, Any],
    summary300: Mapping[str, Any],
    summary500: Mapping[str, Any],
) -> tuple[bool, str]:
    ranks = tuple(behavior_rank(item) for item in (summary100, summary300, summary500))
    passed = ranks[0] < ranks[1] < ranks[2]
    return passed, (
        f"R2 strict lexicographic improvement passed: {ranks}"
        if passed
        else f"R2 strict lexicographic improvement failed: {ranks}"
    )


def behavior_final_gate(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("valid_count") == 9
        and int(summary.get("full_count", -1)) >= 8
        and int(summary.get("rear_hold_count", -1)) >= 8
        and int(summary.get("no_severe_inward_count", -1)) >= 8
    )


def validate_authority() -> dict[str, Any]:
    expected_files = {
        SPEC: SPEC_SHA256,
        PREREGISTRATION: PREREGISTRATION_SHA256,
        PROTECTED_ROOT: PROTECTED_ROOT_SHA256,
        B500: B500_SHA256,
        TEACHER: TEACHER_SHA256,
        TEACHER_ENV: TEACHER_ENV_SHA256,
        PARENT_TEACHER_MANIFEST: PARENT_TEACHER_MANIFEST_SHA256,
    }
    for path, expected in expected_files.items():
        actual = sha256_file(path.resolve(strict=True))
        if actual != expected:
            raise RuntimeError(f"R2 authority SHA mismatch: {path}: {actual} != {expected}")
    if not is_read_only(PREREGISTRATION):
        raise RuntimeError("R2 preregistration must remain read-only")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if not (
        prereg.get("schema_version") == 1
        and prereg.get("kind") == "highstep_student_recovery_r2_preregistration"
        and prereg.get("workflow_id") == WORKFLOW_ID
        and prereg.get("route", {}).get("selected") == "R2"
        and prereg.get("route", {}).get("r1_excluded") is True
        and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA256
        and prereg.get("training_allowed") is False
    ):
        raise RuntimeError("R2 preregistration content/latch differs from approved v1.1.1")
    binding = prereg.get("checkpoint_binding", {})
    required = {
        "protected_student_root": (PROTECTED_ROOT, PROTECTED_ROOT_SHA256),
        "behavior_start_and_anchor": (B500, B500_SHA256),
        "frozen_teacher": (TEACHER, TEACHER_SHA256),
    }
    for name, (path, digest) in required.items():
        record = binding.get(name, {})
        if os.path.realpath(str(record.get("path", ""))) != os.path.realpath(path):
            raise RuntimeError(f"R2 preregistration path mismatch: {name}")
        if record.get("sha256") != digest:
            raise RuntimeError(f"R2 preregistration SHA mismatch: {name}")
    behavior = binding["behavior_start_and_anchor"]
    evaluation_path = Path(str(behavior.get("evaluation_manifest", ""))).resolve(strict=True)
    handoff_path = Path(str(behavior.get("capped_handoff", ""))).resolve(strict=True)
    if (
        sha256_file(evaluation_path) != behavior.get("evaluation_manifest_sha256")
        or sha256_file(handoff_path) != behavior.get("capped_handoff_sha256")
    ):
        raise RuntimeError("Qualified B500 evidence SHA changed")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    summaries = evaluation.get("checkpoint_summaries")
    selected = summaries[0] if isinstance(summaries, list) and len(summaries) == 1 else {}
    best = handoff.get("best_core9")
    if not (
        evaluation.get("evaluation_complete") is True
        and evaluation.get("matrix_complete") is True
        and evaluation.get("completed_runs") == 9
        and evaluation.get("selected_checkpoint") == str(B500)
        and selected.get("checkpoint") == str(B500)
        and selected.get("checkpoint_sha256") == B500_SHA256
        and selected.get("valid_count") == 9
        and _same_scalar(selected.get("full_climb_rate"), 6 / 9)
        and _same_scalar(selected.get("rear_on_platform_hold_rate"), 6 / 9)
        and _same_scalar(selected.get("front_top_support_rate"), 8 / 9)
        and handoff.get("status") == "stopped_by_v111_cap"
        and handoff.get("spec_sha256") == SPEC_SHA256
        and handoff.get("best_checkpoint") == str(B500)
        and handoff.get("best_checkpoint_sha256") == B500_SHA256
        and isinstance(best, Mapping)
        and best.get("valid_count") == 9
        and best.get("full_count") == 6
        and best.get("rear_hold_count") == 6
        and best.get("front_top_support_count") == 8
        and best.get("no_severe_inward_count") == 9
    ):
        raise RuntimeError("Qualified B500 behavior evidence semantics changed")
    if prereg.get("trainable_scope", {}).get("optimizer_parameter_tensor_count") != 8:
        raise RuntimeError("R2 optimizer tensor count changed in preregistration")
    route = prereg.get("route", {})
    for path_key, sha_key in (
        ("same_state_aggregate", "same_state_aggregate_sha256"),
        ("route_selection", "route_selection_sha256"),
    ):
        path = Path(str(route.get(path_key, ""))).resolve(strict=True)
        if sha256_file(path) != route.get(sha_key):
            raise RuntimeError(f"R2 evidence authority changed: {path_key}")
    helper_path = Path(
        str(prereg.get("permanent_contract", {}).get("shared_post_prior_helper", "")).split("::", 1)[0]
    ).resolve(strict=True)
    if sha256_file(helper_path) != prereg.get("permanent_contract", {}).get(
        "shared_post_prior_helper_file_sha256"
    ):
        raise RuntimeError("R2 shared post-prior helper changed")
    return prereg


def aggregate_action_mae(frame_paths: Sequence[Path]) -> dict[str, Any]:
    if len(frame_paths) != 3:
        raise RuntimeError("Teacher-action regression requires exactly three base3 frame traces")
    revolute_sum = 0.0
    box_sum = 0.0
    frame_count = 0
    trace_records: list[dict[str, Any]] = []
    for path in frame_paths:
        local_count = 0
        with path.resolve(strict=True).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {value}")
                    ),
                )
                error = record.get("action_error_student_minus_teacher")
                if not (
                    isinstance(error, list)
                    and len(error) == 16
                    and all(finite_number(value) for value in error)
                    and record.get("runner_manual_action_allclose") is True
                    and record.get("physical_state_unchanged_by_forward") is True
                    and record.get("observation_and_prior_inputs_unchanged_by_forward") is True
                    and record.get("causal_replacement_applied") is False
                    and finite_number(record.get("action_decomposition_residual_max_abs"))
                    and float(record["action_decomposition_residual_max_abs"]) <= 1.0e-6
                ):
                    raise RuntimeError(f"Invalid candidate same-state frame in {path}:{local_count + 1}")
                revolute_sum += sum(abs(float(value)) for value in error[:12])
                box_sum += sum(abs(float(value)) for value in error[12:16])
                local_count += 1
                frame_count += 1
        if local_count != 600:
            raise RuntimeError(f"Candidate same-state trace has {local_count} frames, expected 600: {path}")
        trace_records.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path), "frame_count": local_count}
        )
    if frame_count != 1800:
        raise RuntimeError("Candidate same-state aggregate is not the exact 3x600 frame matrix")
    return {
        "frame_count": frame_count,
        "all12_revolute_mae": revolute_sum / (frame_count * 12),
        "all4_box_mae": box_sum / (frame_count * 4),
        "traces": trace_records,
        "finite_parity_digest_decomposition_verified": True,
    }


def pinned_base3_reference_binding() -> dict[str, Any]:
    run_ids = ("seed11_nominal", "seed11_left_offset", "seed22_right_offset")
    summaries: list[dict[str, Any]] = []
    frames: list[Path] = []
    for run_id in run_ids:
        summary_path = (PINNED_BASE3_ROOT / run_id / "run_summary.json").resolve(strict=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frame_path = Path(str(summary.get("frames_path", ""))).resolve(strict=True)
        if summary.get("run_id") != run_id or summary.get("frames_sha256") != sha256_file(frame_path):
            raise RuntimeError(f"Pinned B500 reference binding changed: {run_id}")
        summaries.append(
            {
                "run_id": run_id,
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "frames_path": str(frame_path),
                "frames_sha256": sha256_file(frame_path),
            }
        )
        frames.append(frame_path)
    aggregate = aggregate_action_mae(frames)
    if not (
        math.isclose(
            aggregate["all12_revolute_mae"],
            PINNED_B500_REVOLUTE_MAE,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            aggregate["all4_box_mae"],
            PINNED_B500_BOX_MAE,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise RuntimeError("Pinned B500 Teacher-action reference MAEs changed")
    return {
        "run_bindings": summaries,
        "all12_revolute_mae": PINNED_B500_REVOLUTE_MAE,
        "all4_box_mae": PINNED_B500_BOX_MAE,
        "frame_count": 1800,
    }


def audit_model_scope(
    anchor_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    *,
    require_change: bool = True,
) -> dict[str, Any]:
    if set(anchor_state) != set(candidate_state):
        raise RuntimeError("R2 candidate/B500 model-state keys differ")
    changed_full: list[str] = []
    changed_rows: list[str] = []
    for key in sorted(anchor_state):
        before = anchor_state[key]
        after = candidate_state[key]
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            raise RuntimeError(f"R2 model state contains non-tensor value: {key}")
        if before.shape != after.shape or before.dtype != after.dtype:
            raise RuntimeError(f"R2 tensor contract changed: {key}")
        if not bool(torch.isfinite(after).all().item()):
            raise RuntimeError(f"R2 candidate tensor is non-finite: {key}")
        if key in R2_LIVE_FULL_TENSORS:
            if not torch.equal(before, after):
                changed_full.append(key)
        elif key in R2_ACTOR_ROW_TENSORS:
            if after.shape[0] != 16 or not torch.equal(before[:12], after[:12]):
                raise RuntimeError(f"R2 changed an unauthorized actor row: {key}")
            if not torch.equal(before[12:16], after[12:16]):
                changed_rows.append(key)
        elif not torch.equal(before, after):
            raise RuntimeError(f"R2 changed frozen Student tensor: {key}")
    if require_change and not (changed_full or changed_rows):
        raise RuntimeError("R2 update produced no permitted parameter change")
    return {
        "changed_live_tensors": changed_full,
        "changed_actor_box_row_tensors": changed_rows,
        "frozen_tensor_violation_count": 0,
        "finite": True,
    }


def _optimizer_audit(
    optimizer: Mapping[str, Any],
    *,
    effective_count: int,
    parameter_shapes: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(state, Mapping):
        raise RuntimeError("R2 Adam must contain exactly two parameter groups")
    expected_groups = (
        ("estimator_encoder_and_mu", 6, 1.0e-4),
        ("box_output_rows", 2, 1.0e-5),
    )
    parameter_ids: list[Any] = []
    for group, (name, count, learning_rate) in zip(groups, expected_groups, strict=True):
        ids = group.get("params")
        if (
            group.get("name") != name
            or not isinstance(ids, list)
            or len(ids) != count
            or float(group.get("lr", -1.0)) != learning_rate
            or tuple(float(value) for value in group.get("betas", ())) != (0.9, 0.999)
            or float(group.get("eps", -1.0)) != 1.0e-8
            or float(group.get("weight_decay", -1.0)) != 0.0
            or group.get("amsgrad") is not False
        ):
            raise RuntimeError(f"R2 Adam group changed: {name}")
        parameter_ids.extend(ids)
    if len(parameter_ids) != 8 or len(set(parameter_ids)) != 8 or set(state) != set(parameter_ids):
        raise RuntimeError("R2 Adam state does not own exactly the eight ordered tensors")
    if len(parameter_shapes) != 8:
        raise RuntimeError("R2 optimizer shape contract must contain eight tensors")
    expected_step = effective_count * 4
    steps: list[int] = []
    for parameter_id, expected_shape in zip(parameter_ids, parameter_shapes, strict=True):
        item = state[parameter_id]
        if not isinstance(item, Mapping):
            raise RuntimeError("R2 Adam parameter state is not a mapping")
        if set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError("R2 Adam parameter state has missing or unexpected fields")
        raw_step = item.get("step")
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise RuntimeError("R2 Adam step is not scalar")
            step_value = float(raw_step.item())
        elif isinstance(raw_step, (int, float)) and not isinstance(raw_step, bool):
            step_value = float(raw_step)
        else:
            raise RuntimeError("R2 Adam step has an invalid type")
        if not math.isfinite(step_value) or not step_value.is_integer():
            raise RuntimeError("R2 Adam step is non-finite or non-integral")
        step = int(step_value)
        if step != expected_step:
            raise RuntimeError(f"R2 Adam step mismatch: {step} != {expected_step}")
        steps.append(step)
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = item.get(moment_name)
            if (
                not isinstance(moment, torch.Tensor)
                or tuple(moment.shape) != tuple(expected_shape)
                or not bool(torch.isfinite(moment).all().item())
            ):
                raise RuntimeError(f"R2 Adam {moment_name} is invalid")
    return {
        "optimizer_parameter_tensor_count": 8,
        "optimizer_parameter_names": list(R2_OPTIMIZER_NAMES),
        "group_parameter_counts": [6, 2],
        "group_learning_rates": [1.0e-4, 1.0e-5],
        "adam_steps": steps,
        "adam_moments_finite": True,
    }


def audit_r2_checkpoint_payload(
    anchor_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    *,
    expected_count: int,
    expected_load_mode: str | None = None,
) -> dict[str, Any]:
    anchor_state = anchor_payload.get("model_state_dict")
    candidate_state = candidate_payload.get("model_state_dict")
    if not isinstance(anchor_state, Mapping) or not isinstance(candidate_state, Mapping):
        raise RuntimeError("R2 checkpoint lacks model_state_dict")
    scope = audit_model_scope(anchor_state, candidate_state)
    infos = candidate_payload.get("infos")
    extra = infos.get(ALGORITHM_STATE_KEY) if isinstance(infos, Mapping) else None
    if not isinstance(extra, Mapping):
        raise RuntimeError("R2 checkpoint lacks algorithm-owned state")
    recovery = extra.get("student_recovery")
    if not (
        extra.get("schema_version") == 1
        and extra.get("algorithm_class") == "VAEPPO"
        and extra.get("distill_stage") == 2
        and extra.get("student_distill_update_count") == expected_count
        and isinstance(recovery, Mapping)
        and recovery.get("schema_version") == 1
        and recovery.get("stage") == "R2"
        and recovery.get("effective_update_count") == expected_count
        and recovery.get("preregistration_sha256") == PREREGISTRATION_SHA256
        and tuple(recovery.get("optimizer_parameter_names") or ()) == R2_OPTIMIZER_NAMES
    ):
        raise RuntimeError("R2 checkpoint version/stage/count/preregistration binding mismatch")
    binding = recovery.get("binding_manifest")
    if not isinstance(binding, Mapping):
        raise RuntimeError("R2 checkpoint lacks binding_manifest")
    expected_binding = {
        "stage": "R2",
        "preregistration_path": os.path.realpath(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protected_student_root": os.path.realpath(PROTECTED_ROOT),
        "protected_student_root_sha256": PROTECTED_ROOT_SHA256,
        "initial_student_checkpoint": os.path.realpath(B500),
        "initial_student_sha256": B500_SHA256,
        "teacher_checkpoint": os.path.realpath(TEACHER),
        "teacher_sha256": TEACHER_SHA256,
        "teacher_env_yaml": os.path.realpath(TEACHER_ENV),
        "teacher_env_yaml_sha256": TEACHER_ENV_SHA256,
        "schedule_resume_mode": "preserve",
        "effective_update_count": expected_count,
        "optimizer_parameter_tensor_count": 8,
        "optimizer_parameter_names": list(R2_OPTIMIZER_NAMES),
        "trainable_live_tensors": list(R2_OPTIMIZER_NAMES[:6]),
        "materialized_actor_rows": [12, 13, 14, 15],
    }
    if expected_load_mode is not None:
        expected_binding["checkpoint_load_mode"] = expected_load_mode
    mismatches = {
        key: {"actual": binding.get(key), "expected": value}
        for key, value in expected_binding.items()
        if binding.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"R2 checkpoint immutable binding mismatch: {mismatches}")
    if binding.get("teacher_independent_storage") is not True or binding.get(
        "anchor_independent_storage"
    ) is not True:
        raise RuntimeError("R2 checkpoint does not prove independent Teacher/B500 modules")
    runtime = binding.get("runtime_contract")
    if not (
        isinstance(runtime, Mapping)
        and runtime.get("joint_order") == list(JOINT_ORDER)
        and exact_numbers(runtime.get("action_scale"), ACTION_SCALE)
        and exact_numbers(runtime.get("action_offset"), ACTION_OFFSET)
        and runtime.get("joint_pos_clip") == [[-60.0, 60.0] for _ in range(16)]
        and runtime.get("teacher_context_shape") == [3]
        and runtime.get("teacher_context_order")
        == ["height_delta", "command_x", "front_rear_delta"]
    ):
        raise RuntimeError("R2 checkpoint runtime action/context contract changed")
    prefix_error = binding.get("actor_prefix_equivalence_max_error")
    if (
        isinstance(prefix_error, bool)
        or not isinstance(prefix_error, (int, float))
        or not math.isfinite(float(prefix_error))
        or float(prefix_error) > 1.0e-6
    ):
        raise RuntimeError("R2 actor prefix equivalence audit failed")

    box_weight = recovery.get("box_weight")
    box_bias = recovery.get("box_bias")
    actor_weight = candidate_state.get("actor.6.weight")
    actor_bias = candidate_state.get("actor.6.bias")
    if not (
        isinstance(box_weight, torch.Tensor)
        and isinstance(box_bias, torch.Tensor)
        and isinstance(actor_weight, torch.Tensor)
        and isinstance(actor_bias, torch.Tensor)
        and torch.equal(box_weight.detach().cpu(), actor_weight[12:16].detach().cpu())
        and torch.equal(box_bias.detach().cpu(), actor_bias[12:16].detach().cpu())
    ):
        raise RuntimeError("R2 materialized box rows disagree with model_state_dict")
    optimizer = candidate_payload.get("optimizer_state_dict")
    extra_optimizer = extra.get("vae_optimizer_state_dict")
    if not isinstance(optimizer, Mapping) or not isinstance(extra_optimizer, Mapping):
        raise RuntimeError("R2 checkpoint lacks resumable Adam state")
    if not _nested_equal(optimizer, extra_optimizer):
        raise RuntimeError("R2 runner and algorithm Adam states disagree")
    shapes = [tuple(candidate_state[name].shape) for name in R2_OPTIMIZER_NAMES[:6]]
    shapes.extend((tuple(box_weight.shape), tuple(box_bias.shape)))
    adam = _optimizer_audit(
        optimizer,
        effective_count=expected_count,
        parameter_shapes=shapes,
    )
    return {
        "effective_update_count": expected_count,
        "scope": scope,
        "adam": adam,
        "teacher_binding_verified": True,
        "b500_anchor_binding_verified": True,
        "protected_root_binding_verified": True,
        "runner_and_algorithm_optimizer_state_equal": True,
    }


def audit_smoke_resume(
    audit4: Mapping[str, Any], audit5: Mapping[str, Any]
) -> dict[str, Any]:
    steps4 = list(audit4["adam"]["adam_steps"])
    steps5 = list(audit5["adam"]["adam_steps"])
    if len(steps4) != 8 or len(steps5) != 8:
        raise RuntimeError("R2 smoke Adam scope changed across full resume")
    if any(after != before + 4 for before, after in zip(steps4, steps5, strict=True)):
        raise RuntimeError("R2 smoke full resume did not continue each Adam step by four")
    if audit4.get("effective_update_count") != 4 or audit5.get("effective_update_count") != 5:
        raise RuntimeError("R2 smoke effective count continuity failed")
    return {
        "fresh_count": 4,
        "full_resume_count": 5,
        "adam_step_continuity_verified": True,
        "optimizer_parameter_tensor_count": 8,
    }


def critical_file_paths() -> tuple[Path, ...]:
    paths = {
        SPEC,
        PREREGISTRATION,
        ROOT / "tools/highstep_student_recovery_r2_supervisor.py",
        ROOT / "scripts/rsl_rl/base/train.py",
        ROOT / "scripts/rsl_rl/base/play.py",
        ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py",
        ROOT / "scripts/rsl_rl/cli_args.py",
        ROOT / "scripts/rsl_rl/base/highstep_candidate_same_state_audit_play.py",
        ROOT / "scripts/rsl_rl/base/highstep_same_state_audit_play.py",
        ROOT / "tools/highstep_same_state_audit.py",
        ROOT / "tools/highstep_student_recovery_supervisor.py",
        ROOT / "scripts/systemd/highstep-student-recovery-r2.service",
        ROOT / "tests/test_highstep_candidate_same_state_audit.py",
        SCHEMA4_MONITOR,
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/observations.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/events.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/velocity_env_cfg.py",
        ROOT / "source/robot_lab/robot_lab/terrains/config/rough.py",
        ROOT / (
            "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
            "Arcdog_adjustable_leg/highstep_env_cfg.py"
        ),
        ROOT / (
            "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
            "Arcdog_adjustable_leg/__init__.py"
        ),
        ROOT / (
            "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
            "Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py"
        ),
        ROOT / (
            "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
            "Arcdog_adjustable_leg/agents/vae_ppo.py"
        ),
    }
    paths.update(ROOT.glob("tests/test_*r2*.py"))
    paths.update(ROOT.glob("tests/test_highstep_teacher_prior_context.py"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"R2 implementation file is missing: {missing}")
    return tuple(sorted(paths, key=str))


def render_r2_monitor_source(source: str) -> str:
    """Add the sole approved R2-train -> deployment-eval task alias.

    The original schema4 monitor retains every other schedule, lineage, SHA,
    action-contract and runtime check byte-for-byte.
    """

    # The canonical monitor now delegates task aliases to one closed,
    # workflow-scoped helper.  If that helper already contains this exact R2
    # tuple, deriving a second copied monitor must be a byte-identical no-op.
    if (
        "from tools.highstep_core9_task_compat import source_authority_is_valid" in source
        and "source_authority_is_valid(" in source
    ):
        from tools.highstep_core9_task_compat import source_task_is_compatible

        if not source_task_is_compatible(
            workflow_id=WORKFLOW_ID,
            role="student",
            source_task=TRAIN_TASK,
            eval_task=EVAL_TASK,
        ):
            raise RuntimeError("canonical task helper is missing the exact approved R2 task pair")
        return source

    needle = '''    source_task_compatible = bool(
        source_manifest.get("task") == task
        or (
            role == "teacher_robust"
            and task
            == "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0"
            and source_manifest.get("task")
            == "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0"
        )
    )'''
    replacement = '''    source_task_compatible = bool(
        source_manifest.get("task") == task
        or (
            role == "teacher_robust"
            and task
            == "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0"
            and source_manifest.get("task")
            == "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0"
        )
        or (
            # v1.1.1 R2 trains with a privileged training-only context group,
            # then must be evaluated through the unchanged deployment task.
            role == "student"
            and task
            == "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0"
            and source_manifest.get("task")
            == "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPriorR2-ArcdogAdjustableLeg-v0"
        )
    )'''
    if source.count(needle) != 1:
        raise RuntimeError("Schema4 source-task compatibility block changed; refusing R2 patch")
    rendered = source.replace(needle, replacement)
    if rendered.count(TRAIN_TASK) != 1 or rendered.count(EVAL_TASK) < 1:
        raise RuntimeError("Derived R2 schema4 monitor does not contain the exact approved task pair")
    return rendered


def ensure_r2_monitor_snapshot() -> tuple[Path, str]:
    source = SCHEMA4_MONITOR.read_text(encoding="utf-8")
    rendered = render_r2_monitor_source(source)
    expected = rendered.encode("utf-8")
    R2_MONITOR_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    if R2_MONITOR_SNAPSHOT.exists():
        if R2_MONITOR_SNAPSHOT.read_bytes() != expected or not is_read_only(R2_MONITOR_SNAPSHOT):
            raise RuntimeError("Existing R2 schema4 monitor snapshot is stale or writable")
    else:
        temporary = R2_MONITOR_SNAPSHOT.with_name(
            f".{R2_MONITOR_SNAPSHOT.name}.{os.getpid()}.tmp"
        )
        temporary.write_bytes(expected)
        temporary.chmod(0o555)
        os.replace(temporary, R2_MONITOR_SNAPSHOT)
    return R2_MONITOR_SNAPSHOT, sha256_file(R2_MONITOR_SNAPSHOT)


class R2Supervisor:
    """Own one immutable R2 workflow; no implicit retry branch is permitted."""

    def __init__(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_path = STATE_ROOT / "state.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.preflight_path = STATE_ROOT / "r2_preflight_manifest.json"
        self.smoke_path = STATE_ROOT / "r2_smoke_audit.json"
        self.main_launch_path = STATE_ROOT / "r2_main_launch_manifest.json"
        self.active_process: subprocess.Popen[bytes] | None = None
        self.critical_hashes: dict[str, str] = {}
        existing: dict[str, Any] = {}
        if self.state_path.is_file():
            try:
                candidate = json.loads(self.state_path.read_text(encoding="utf-8"))
                if candidate.get("workflow_id") == WORKFLOW_ID:
                    existing = candidate
            except json.JSONDecodeError:
                pass
        self.state: dict[str, Any] = {
            **existing,
            "schema_version": 2,
            "workflow_id": WORKFLOW_ID,
            "route": "R2",
            "spec_sha256": SPEC_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "supervisor_pid": os.getpid(),
            "status": "starting",
            "phase": "r2_preflight",
            "checkpoint": str(B500),
            "effective_updates": 0,
            "stop_reason": None,
        }
        self.update_state()

    def update_state(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at"] = now_text()
        self.state["last_supervisor_check_at"] = time.time()
        atomic_json(self.state_path, self.state)
        atomic_json(
            self.heartbeat_path,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "timestamp": self.state["updated_at"],
                "status": self.state.get("status"),
                "phase": self.state.get("phase"),
                "checkpoint": self.state.get("checkpoint"),
                "effective_updates": self.state.get("effective_updates"),
                "active_pid": self.state.get("active_pid"),
            },
        )

    @staticmethod
    def finite(value: Any, default: float) -> float:
        return float(value) if finite_number(value) else default

    @staticmethod
    def _is_managed_gpu_job_argv(argv: Sequence[str]) -> bool:
        """Match real process argv tokens, never shell command text mentioning a job."""
        exact_suffixes = (
            "scripts/rsl_rl/base/train.py",
            "scripts/rsl_rl/base/play.py",
            "highstep_candidate_same_state_audit_play.py",
            "highstep_same_state_audit_play.py",
            "highstep_v13_core9_trace_play.py",
        )
        for token in argv:
            # A bash -c/-lc command body is one whitespace-containing argv token.
            # Treating substrings in that body as processes makes harmless status
            # commands trip the fail-closed concurrency guard.
            if not token or any(character.isspace() for character in token):
                continue
            normalized = token.replace("\\", "/").rstrip("/")
            if any(
                normalized == suffix or normalized.endswith(f"/{suffix}")
                for suffix in exact_suffixes
            ):
                return True
            if Path(normalized).name.startswith("highstep_centerline_guard_monitor"):
                return True
        return False

    @classmethod
    def active_gpu_jobs(cls) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv = [
                    token.decode(errors="replace")
                    for token in (proc / "cmdline").read_bytes().split(b"\0")
                    if token
                ]
            except OSError:
                continue
            if cls._is_managed_gpu_job_argv(argv):
                jobs.append({"pid": int(proc.name), "argv": " ".join(argv).strip()})
        return jobs

    def require_no_gpu_job(self) -> None:
        jobs = self.active_gpu_jobs()
        if jobs:
            raise RuntimeError(f"Another train/eval/play process is active: {jobs}")

    def preflight(self) -> dict[str, Any]:
        self.require_no_gpu_job()
        prereg = validate_authority()
        self.critical_hashes = {
            str(path.resolve()): sha256_file(path) for path in critical_file_paths()
        }
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True
        ).splitlines()
        manifest = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_r2_preflight",
            "workflow_id": WORKFLOW_ID,
            "created_at": now_text(),
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA256,
            "preregistration_path": str(PREREGISTRATION),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "preregistration_training_allowed_latch": prereg["training_allowed"],
            "main_training_authorized": False,
            "protected_root": str(PROTECTED_ROOT),
            "protected_root_sha256": PROTECTED_ROOT_SHA256,
            "behavior_start_and_anchor": str(B500),
            "behavior_start_and_anchor_sha256": B500_SHA256,
            "teacher": str(TEACHER),
            "teacher_sha256": TEACHER_SHA256,
            "train_task": TRAIN_TASK,
            "eval_task": EVAL_TASK,
            "critical_file_sha256": self.critical_hashes,
            "worktree_dirty": bool(status),
            "git_status_porcelain": status,
            "no_concurrent_train_eval_play": True,
        }
        atomic_json(self.preflight_path, manifest)
        self.update_state(
            status="preflight_passed",
            phase="r2_static_then_smoke",
            preflight_manifest=str(self.preflight_path),
        )
        return manifest

    def assert_code_unchanged(self) -> None:
        if not self.critical_hashes:
            raise RuntimeError("R2 critical code baseline was not captured")
        current = {
            str(path.resolve()): sha256_file(path) for path in critical_file_paths()
        }
        if current != self.critical_hashes:
            changed = sorted(set(current) | set(self.critical_hashes))
            changed = [key for key in changed if current.get(key) != self.critical_hashes.get(key)]
            raise RuntimeError(f"R2 implementation changed while active: {changed}")

    @staticmethod
    def _tree_progress(path: Path) -> tuple[int, int]:
        size = 0
        newest = 0
        if path.exists():
            for item in path.rglob("*"):
                if not item.is_file():
                    continue
                try:
                    metadata = item.stat()
                except OSError:
                    continue
                size += metadata.st_size
                newest = max(newest, metadata.st_mtime_ns)
        return size, newest

    @staticmethod
    def stop_process_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 30), (signal.SIGKILL, 5)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + grace
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(1)
            if process.poll() is not None:
                return

    def stop_active_process(self) -> None:
        if self.active_process is not None:
            self.stop_process_group(self.active_process)
            self.active_process = None

    def wait_process(
        self,
        process: subprocess.Popen[bytes],
        progress_root: Path,
        phase: str,
        stall_seconds: int = 900,
    ) -> None:
        last_progress = self._tree_progress(progress_root)
        last_progress_at = time.monotonic()
        last_code_check = 0.0
        self.active_process = process
        while process.poll() is None:
            now = time.monotonic()
            if now - last_code_check >= 30:
                self.assert_code_unchanged()
                last_code_check = now
            progress = self._tree_progress(progress_root)
            if progress != last_progress:
                last_progress = progress
                last_progress_at = now
            elif now - last_progress_at >= stall_seconds:
                self.stop_process_group(process)
                raise RuntimeError(f"{phase} made no observable progress for {stall_seconds}s")
            self.update_state(
                status="running",
                phase=phase,
                active_pid=process.pid,
                progress_bytes=progress[0],
            )
            time.sleep(5)
        self.active_process = None
        self.update_state(active_pid=None)
        if process.returncode != 0:
            raise RuntimeError(f"{phase} exited with return code {process.returncode}")

    @staticmethod
    def checkpoint_payload(path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Checkpoint payload is not a mapping: {path}")
        return payload

    @classmethod
    def checkpoint_effective_count(cls, path: Path) -> int:
        if path.resolve() == B500.resolve():
            return 0
        payload = cls.checkpoint_payload(path)
        extra = ((payload.get("infos") or {}).get(ALGORITHM_STATE_KEY) or {})
        recovery = extra.get("student_recovery") or {}
        count = recovery.get("effective_update_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= 1000:
            raise RuntimeError(f"Checkpoint has no valid R2 effective count: {path}")
        return count

    @classmethod
    def find_checkpoint_by_count(cls, run_dir: Path, expected_count: int) -> Path:
        matches: list[Path] = []
        for path in run_dir.glob("model_*.pt"):
            try:
                if cls.checkpoint_effective_count(path) == expected_count:
                    matches.append(path)
            except Exception:
                continue
        if not matches:
            raise RuntimeError(f"No R2 checkpoint count={expected_count} in {run_dir}")
        return max(matches, key=lambda path: path.stat().st_mtime_ns)

    def audit_checkpoint(
        self,
        checkpoint: Path,
        *,
        expected_count: int,
        expected_load_mode: str | None,
    ) -> dict[str, Any]:
        return audit_r2_checkpoint_payload(
            self.checkpoint_payload(B500),
            self.checkpoint_payload(checkpoint),
            expected_count=expected_count,
            expected_load_mode=expected_load_mode,
        )

    def run_train(
        self,
        *,
        label: str,
        checkpoint: Path,
        load_mode: str,
        updates: int,
        num_envs: int,
        smoke: bool,
    ) -> tuple[Path, Path, Path]:
        self.assert_code_unchanged()
        self.require_no_gpu_job()
        if not smoke:
            self.assert_main_training_authorized()
        source_count = self.checkpoint_effective_count(checkpoint)
        if load_mode == "full":
            # A checkpoint may never become a continuation source based only
            # on its count.  Re-audit scope, Adam and immutable bindings first.
            self.audit_checkpoint(
                checkpoint,
                expected_count=source_count,
                expected_load_mode=None,
            )
        expected_count = updates if load_mode == "weights_only" else source_count + updates
        if load_mode == "weights_only" and checkpoint.resolve() != B500.resolve():
            raise RuntimeError("Fresh R2 may only load the preregistered B500")
        if load_mode == "full" and source_count <= 0:
            raise RuntimeError("R2 full resume requires a versioned R2 checkpoint")
        maximum_effective_updates = int(
            getattr(self, "maximum_effective_updates", 1000)
        )
        if expected_count > maximum_effective_updates or updates <= 0:
            raise RuntimeError("R2 requested update budget is invalid")

        stage_result = STATE_ROOT / "training_stage_results" / f"{label}.json"
        if stage_result.is_file():
            if not is_read_only(stage_result):
                raise RuntimeError(f"R2 completed stage result is writable: {stage_result}")
            record = json.loads(stage_result.read_text(encoding="utf-8"))
            result = Path(str(record.get("checkpoint", ""))).resolve(strict=True)
            run_dir = Path(str(record.get("run_dir", ""))).resolve(strict=True)
            log_path = Path(str(record.get("train_log", ""))).resolve(strict=True)
            expected_record = bool(
                record.get("workflow_id") == WORKFLOW_ID
                and record.get("label") == label
                and record.get("smoke_discard_branch") is smoke
                and record.get("source_checkpoint") == str(checkpoint.resolve())
                and record.get("source_checkpoint_sha256") == sha256_file(checkpoint)
                and record.get("checkpoint_load_mode") == load_mode
                and record.get("requested_additional_updates") == updates
                and record.get("num_envs") == num_envs
                and record.get("effective_update_count") == expected_count
                and record.get("checkpoint_sha256") == sha256_file(result)
            )
            if not expected_record:
                raise RuntimeError(f"R2 completed stage result is stale: {stage_result}")
            self.audit_checkpoint(
                result,
                expected_count=expected_count,
                expected_load_mode=load_mode,
            )
            self.update_state(
                phase=f"reuse_train_{label}",
                checkpoint=str(result),
                effective_updates=expected_count,
                run_dir=str(run_dir),
                train_log=str(log_path),
            )
            return run_dir, result, log_path

        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"student_recovery_r2_{label}_{stamp}_{os.getpid()}"
        launch_dir = STATE_ROOT / "training" / run_name
        launch_dir.mkdir(parents=True, exist_ok=False)
        log_path = launch_dir / "train.log"
        command = [
            str(PYTHON),
            "-u",
            "scripts/rsl_rl/base/train.py",
            "--task",
            TRAIN_TASK,
            "--num_envs",
            str(num_envs),
            "--max_iterations",
            str(updates),
            "--seed",
            "42",
            "--headless",
            "--logger",
            "tensorboard",
            "--resume",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--highstep_resume_mode",
            "refine",
            "--highstep_checkpoint_load_mode",
            load_mode,
            "--highstep_schedule_resume_mode",
            "preserve",
            "--highstep_parent_teacher_manifest",
            str(PARENT_TEACHER_MANIFEST),
            "--run_name",
            run_name,
        ]
        atomic_json(
            launch_dir / "launch.json",
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "label": label,
                "smoke_discard_branch": smoke,
                "command": command,
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "source_effective_count": source_count,
                "checkpoint_load_mode": load_mode,
                "requested_additional_updates": updates,
                "expected_effective_count": expected_count,
                "num_envs": num_envs,
                "created_at": now_text(),
            },
        )
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("wb") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.wait_process(
                process,
                progress_root=launch_dir,
                phase=f"train_{label}",
                stall_seconds=900,
            )
        matches = sorted(
            EXPERIMENT_ROOT.glob(f"*_{run_name}"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one R2 run under {EXPERIMENT_ROOT} for {run_name}, got {matches}"
            )
        run_dir = matches[0]
        result = self.find_checkpoint_by_count(run_dir, expected_count)
        audit = self.audit_checkpoint(
            result,
            expected_count=expected_count,
            expected_load_mode=load_mode,
        )
        binding_path = run_dir / "params" / str(
            getattr(
                self,
                "training_binding_filename",
                "highstep_student_recovery_r2_binding.json",
            )
        )
        if not binding_path.is_file():
            raise RuntimeError("R2 run lacks train/runtime binding manifest")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        train_runtime = binding.get("train_runtime_contract")
        action_config = binding.get("action_config_contract")
        if not (
            binding.get("task") == TRAIN_TASK
            and binding.get("preregistration_sha256") == PREREGISTRATION_SHA256
            and binding.get("checkpoint_load_mode") == load_mode
            and binding.get("schedule_resume_mode") == "preserve"
            and isinstance(action_config, Mapping)
            and action_config.get("joint_order") == list(JOINT_ORDER)
            and action_config.get("scale_patterns")
            == {".*_box_joint": 0.02, ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1}
            and action_config.get("clip_patterns") == {".*": [-60.0, 60.0]}
            and action_config.get("use_default_offset") is True
            and action_config.get("preserve_order") is True
            and isinstance(train_runtime, Mapping)
            and train_runtime.get("validated") is True
            and train_runtime.get("joint_order") == list(JOINT_ORDER)
            and train_runtime.get("joint_ids") == list(range(16))
            and exact_numbers(train_runtime.get("action_scale"), ACTION_SCALE)
            and exact_numbers(train_runtime.get("action_offset"), ACTION_OFFSET)
            and exact_numbers(train_runtime.get("default_joint_pos"), ACTION_OFFSET)
            and train_runtime.get("joint_pos_clip") == [[-60.0, 60.0] for _ in range(16)]
            and train_runtime.get("teacher_context_shape") == [3]
            and train_runtime.get("teacher_context_order")
            == ["height_delta", "command_x", "front_rear_delta"]
            and train_runtime.get("teacher_context_runtime_finite") is True
            and train_runtime.get("policy_estimator_critic_input") is False
            and train_runtime.get("policy_observation_groups")
            == {
                "policy": ["policy"],
                "estimator": ["estimator"],
                "critic": ["critic"],
                "teacher_context": ["teacher_context"],
            }
            and train_runtime.get("policy_obs_dim") == 570
        ):
            raise RuntimeError("R2 run binding manifest disagrees with launch")
        atomic_json(
            launch_dir / "result.json",
            {
                "schema_version": 1,
                "completed": True,
                "run_dir": str(run_dir),
                "checkpoint": str(result),
                "checkpoint_sha256": sha256_file(result),
                "effective_update_count": expected_count,
                "checkpoint_audit": audit,
                "train_binding": str(binding_path),
                "train_binding_sha256": sha256_file(binding_path),
                "completed_at": now_text(),
            },
        )
        atomic_json(
            stage_result,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "label": label,
                "smoke_discard_branch": smoke,
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_load_mode": load_mode,
                "requested_additional_updates": updates,
                "num_envs": num_envs,
                "run_dir": str(run_dir),
                "train_log": str(log_path),
                "checkpoint": str(result),
                "checkpoint_sha256": sha256_file(result),
                "effective_update_count": expected_count,
                "checkpoint_audit": audit,
                "completed_at": now_text(),
            },
            read_only=True,
        )
        self.update_state(
            checkpoint=str(result),
            effective_updates=expected_count,
            run_dir=str(run_dir),
            train_log=str(log_path),
        )
        return run_dir, result, log_path

    @staticmethod
    def _audit_smoke_log(log_path: Path) -> dict[str, Any]:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        before = re.findall(r"R2_Buffer_Step_Before_Clear[^\n]*?24(?:\.0+)?", text)
        after = re.findall(r"R2_Buffer_Step_After_Clear[^\n]*?0(?:\.0+)?", text)
        if not before or not after:
            raise RuntimeError(
                "R2 smoke log does not prove storage step 24 before update and 0 after clear"
            )
        return {
            "buffer_step_before_clear": 24,
            "buffer_step_after_clear": 0,
            "before_evidence_count": len(before),
            "after_evidence_count": len(after),
        }

    def smoke(self) -> Path:
        if self.smoke_path.is_file():
            if not is_read_only(self.smoke_path):
                raise RuntimeError("Existing R2 smoke evidence is writable")
            existing = json.loads(self.smoke_path.read_text(encoding="utf-8"))
            checkpoint4 = Path(str(existing.get("fresh4_checkpoint", ""))).resolve(strict=True)
            checkpoint5 = Path(str(existing.get("full_resume5_checkpoint", ""))).resolve(strict=True)
            if not (
                existing.get("passed") is True
                and existing.get("discard_branch") is True
                and existing.get("fresh4_checkpoint_sha256") == sha256_file(checkpoint4)
                and existing.get("full_resume5_checkpoint_sha256") == sha256_file(checkpoint5)
            ):
                raise RuntimeError("Existing R2 smoke evidence is stale")
            audit4 = self.audit_checkpoint(
                checkpoint4, expected_count=4, expected_load_mode="weights_only"
            )
            audit5 = self.audit_checkpoint(
                checkpoint5, expected_count=5, expected_load_mode="full"
            )
            audit_smoke_resume(audit4, audit5)
            self.update_state(
                status="smoke_reused",
                phase="r2_main_launch_binding",
                checkpoint=str(B500),
                effective_updates=0,
                smoke_audit=str(self.smoke_path),
            )
            return checkpoint5
        self.update_state(status="running", phase="r2_smoke_fresh4")
        _, checkpoint4, log4 = self.run_train(
            label="smoke_fresh4",
            checkpoint=B500,
            load_mode="weights_only",
            updates=4,
            num_envs=256,
            smoke=True,
        )
        audit4 = self.audit_checkpoint(
            checkpoint4, expected_count=4, expected_load_mode="weights_only"
        )
        log_audit4 = self._audit_smoke_log(log4)
        self.update_state(status="running", phase="r2_smoke_full_resume1")
        _, checkpoint5, log5 = self.run_train(
            label="smoke_full_resume1",
            checkpoint=checkpoint4,
            load_mode="full",
            updates=1,
            num_envs=256,
            smoke=True,
        )
        audit5 = self.audit_checkpoint(
            checkpoint5, expected_count=5, expected_load_mode="full"
        )
        log_audit5 = self._audit_smoke_log(log5)
        continuity = audit_smoke_resume(audit4, audit5)
        payload = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_r2_discarded_smoke_audit",
            "workflow_id": WORKFLOW_ID,
            "passed": True,
            "discard_branch": True,
            "fresh4_checkpoint": str(checkpoint4),
            "fresh4_checkpoint_sha256": sha256_file(checkpoint4),
            "fresh4_audit": audit4,
            "fresh4_storage_audit": log_audit4,
            "full_resume5_checkpoint": str(checkpoint5),
            "full_resume5_checkpoint_sha256": sha256_file(checkpoint5),
            "full_resume5_audit": audit5,
            "full_resume5_storage_audit": log_audit5,
            "continuity": continuity,
            "main_checkpoint_must_restart_fresh_from_b500": True,
            "completed_at": now_text(),
        }
        atomic_json(self.smoke_path, payload, read_only=True)
        self.update_state(
            status="smoke_passed",
            phase="r2_main_launch_binding",
            checkpoint=str(B500),
            effective_updates=0,
            smoke_audit=str(self.smoke_path),
        )
        return checkpoint5

    def run_static_tests(self) -> Path:
        evidence = STATE_ROOT / "r2_static_evidence.json"
        if evidence.is_file():
            if not is_read_only(evidence):
                raise RuntimeError("Existing R2 static evidence is writable")
            existing = json.loads(evidence.read_text(encoding="utf-8"))
            bound_tests = existing.get("test_file_sha256")
            if not (
                existing.get("passed") is True
                and existing.get("py_compile_return_code") == 0
                and existing.get("pytest_return_code") == 0
                and existing.get("critical_file_sha256") == self.critical_hashes
                and isinstance(bound_tests, Mapping)
                and all(Path(path).is_file() and sha256_file(path) == digest for path, digest in bound_tests.items())
                and Path(str(existing.get("log", ""))).is_file()
                and sha256_file(existing["log"]) == existing.get("log_sha256")
            ):
                raise RuntimeError("Existing R2 static evidence is stale")
            self.update_state(status="static_reused", phase="r2_smoke", static_evidence=str(evidence))
            return evidence
        tests = sorted(ROOT.glob("tests/test_*.py"))
        if not tests:
            raise RuntimeError("No CPU/static tests were found")
        python_sources = [path for path in critical_file_paths() if path.suffix == ".py"]
        log_path = STATE_ROOT / "r2_static_tests.log"
        with log_path.open("wb") as stream:
            compile_command = [str(PYTHON), "-m", "py_compile", *map(str, python_sources)]
            stream.write(("COMMAND " + " ".join(compile_command) + "\n").encode())
            compiled = subprocess.run(
                compile_command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
            pytest_command = [str(PYTHON), "-m", "pytest", "-q", "tests"]
            stream.write(("\nCOMMAND " + " ".join(pytest_command) + "\n").encode())
            tested = subprocess.run(
                pytest_command,
                cwd=ROOT,
                env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if compiled.returncode != 0 or tested.returncode != 0:
            raise RuntimeError(f"R2 static/CPU tests failed: {log_path}")
        atomic_json(
            evidence,
            {
                "schema_version": 1,
                "passed": True,
                "tests": [str(path) for path in tests],
                "test_file_sha256": {str(path): sha256_file(path) for path in tests},
                "critical_file_sha256": self.critical_hashes,
                "py_compile_sources": [str(path) for path in python_sources],
                "py_compile_return_code": compiled.returncode,
                "pytest_return_code": tested.returncode,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
                "completed_at": now_text(),
            },
            read_only=True,
        )
        self.update_state(status="static_passed", phase="r2_smoke", static_evidence=str(evidence))
        return evidence

    def write_main_launch_manifest(self, static_evidence: Path) -> Path:
        self.assert_code_unchanged()
        if not self.smoke_path.is_file() or not is_read_only(self.smoke_path):
            raise RuntimeError("R2 smoke evidence is missing or writable")
        smoke = json.loads(self.smoke_path.read_text(encoding="utf-8"))
        if smoke.get("passed") is not True or smoke.get("discard_branch") is not True:
            raise RuntimeError("R2 smoke evidence did not pass as a discarded branch")
        monitor_snapshot, monitor_snapshot_sha = ensure_r2_monitor_snapshot()
        pinned_reference = pinned_base3_reference_binding()
        payload = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_r2_main_launch_authorization",
            "workflow_id": WORKFLOW_ID,
            "created_at": now_text(),
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA256,
            "preregistration_path": str(PREREGISTRATION),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "preregistration_training_allowed_latch": False,
            "latch_interpretation": "pre-main latch; preregistration remains immutable",
            "static_passed": True,
            "static_evidence": str(static_evidence),
            "static_evidence_sha256": sha256_file(static_evidence),
            "smoke_passed": True,
            "smoke_discarded": True,
            "smoke_evidence": str(self.smoke_path),
            "smoke_evidence_sha256": sha256_file(self.smoke_path),
            "critical_file_sha256": self.critical_hashes,
            "schema4_monitor_source": str(SCHEMA4_MONITOR),
            "schema4_monitor_source_sha256": sha256_file(SCHEMA4_MONITOR),
            "r2_monitor_snapshot": str(monitor_snapshot),
            "r2_monitor_snapshot_sha256": monitor_snapshot_sha,
            "only_schedule_task_alias": {"source_train_task": TRAIN_TASK, "eval_task": EVAL_TASK},
            "pinned_b500_teacher_action_reference": pinned_reference,
            "behavior_start": str(B500),
            "behavior_start_sha256": B500_SHA256,
            "main_initial_load": "B500 weights_only schedule preserve fresh R2 Adam count=0",
            "main_training_authorized": True,
            "maximum_effective_updates": 1000,
            "threshold_changes_allowed": False,
        }
        if self.main_launch_path.exists():
            self.assert_main_training_authorized()
            return self.main_launch_path
        atomic_json(self.main_launch_path, payload, read_only=True)
        self.assert_main_training_authorized()
        return self.main_launch_path

    def assert_main_training_authorized(self) -> dict[str, Any]:
        if not self.main_launch_path.is_file() or not is_read_only(self.main_launch_path):
            raise RuntimeError("Immutable R2 main launch manifest is absent or writable")
        payload = json.loads(self.main_launch_path.read_text(encoding="utf-8"))
        required = bool(
            payload.get("kind") == "highstep_student_recovery_r2_main_launch_authorization"
            and payload.get("workflow_id") == WORKFLOW_ID
            and payload.get("spec_sha256") == SPEC_SHA256
            and payload.get("preregistration_sha256") == PREREGISTRATION_SHA256
            and payload.get("preregistration_training_allowed_latch") is False
            and payload.get("static_passed") is True
            and payload.get("smoke_passed") is True
            and payload.get("smoke_discarded") is True
            and payload.get("main_training_authorized") is True
            and payload.get("threshold_changes_allowed") is False
            and payload.get("critical_file_sha256") == self.critical_hashes
        )
        if not required:
            raise RuntimeError("R2 main launch authorization content is stale or invalid")
        for path_key, sha_key in (
            ("static_evidence", "static_evidence_sha256"),
            ("smoke_evidence", "smoke_evidence_sha256"),
            ("r2_monitor_snapshot", "r2_monitor_snapshot_sha256"),
        ):
            path = Path(str(payload.get(path_key, ""))).resolve(strict=True)
            if not is_read_only(path) or sha256_file(path) != payload.get(sha_key):
                raise RuntimeError(f"R2 main launch bound evidence changed: {path_key}")
        if payload.get("only_schedule_task_alias") != {
            "source_train_task": TRAIN_TASK,
            "eval_task": EVAL_TASK,
        }:
            raise RuntimeError("R2 main launch schedule task alias changed")
        if payload.get("schema4_monitor_source_sha256") != sha256_file(SCHEMA4_MONITOR):
            raise RuntimeError("R2 main launch schema4 monitor source changed")
        if payload.get("pinned_b500_teacher_action_reference") != pinned_base3_reference_binding():
            raise RuntimeError("R2 main launch pinned B500 action reference changed")
        self.assert_code_unchanged()
        return payload

    @staticmethod
    def _single_eval_payload(log_path: Path) -> dict[str, Any]:
        payloads: list[dict[str, Any]] = []
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[HIGHSTEP_EVAL_JSON]" not in line:
                continue
            try:
                value = json.loads(line.split("[HIGHSTEP_EVAL_JSON]", 1)[1].strip())
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed HIGHSTEP_EVAL_JSON in {log_path}") from error
            if isinstance(value, dict):
                payloads.append(value)
        if len(payloads) != 1:
            raise RuntimeError(f"Expected exactly one HIGHSTEP_EVAL_JSON in {log_path}")
        return payloads[0]

    def run_probe3(self, checkpoint: Path) -> list[dict[str, Any]]:
        self.assert_main_training_authorized()
        self.audit_checkpoint(
            checkpoint,
            expected_count=50,
            expected_load_mode=None,
        )
        self.require_no_gpu_job()
        output = STATE_ROOT / "evaluations/r2_probe3_50"
        output.mkdir(parents=True, exist_ok=True)
        terminal_manifest = output / "probe3_manifest.json"
        if terminal_manifest.is_file():
            if not is_read_only(terminal_manifest):
                raise RuntimeError("Existing R2 probe3 terminal manifest is writable")
            terminal = json.loads(terminal_manifest.read_text(encoding="utf-8"))
            if (
                terminal.get("checkpoint") != str(checkpoint.resolve())
                or terminal.get("checkpoint_sha256") != sha256_file(checkpoint)
                or terminal.get("effective_update_count") != 50
            ):
                raise RuntimeError("Existing R2 probe3 terminal manifest is stale")
            rows = []
            expected_geometry = {
                (11, "nominal"): (0.0, 0.0),
                (11, "left_offset"): (0.12, 4.0),
                (22, "right_offset"): (-0.12, -4.0),
            }
            for stored in terminal.get("rows", []):
                log_path = Path(str(stored.get("log", ""))).resolve(strict=True)
                if stored.get("log_sha256") != sha256_file(log_path):
                    raise RuntimeError("Existing R2 probe3 log changed")
                payload = self._single_eval_payload(log_path)
                geometry = expected_geometry.get((stored.get("seed"), stored.get("scenario")))
                launch_path = log_path.with_suffix(".launch.json")
                launch = json.loads(launch_path.read_text(encoding="utf-8"))
                expected_command = probe_play_command(
                    checkpoint,
                    seed=int(stored["seed"]),
                    lateral=f"{geometry[0]:.2f}" if geometry is not None else "invalid",
                    yaw=f"{geometry[1]:.1f}" if geometry is not None else "invalid",
                )
                if (
                    geometry is None
                    or not is_read_only(launch_path)
                    or launch.get("command") != expected_command
                    or not _same_scalar(payload.get("yaw_offset_deg"), geometry[1])
                ):
                    raise RuntimeError("Existing R2 probe3 scenario geometry changed")
                validated = validate_probe_payload(payload, checkpoint)
                rows.append({"seed": stored["seed"], "scenario": stored["scenario"], **validated,
                             "log": str(log_path), "log_sha256": sha256_file(log_path)})
            passed, reason = gate_probe3(rows)
            if passed is not terminal.get("passed") or reason != terminal.get("reason"):
                raise RuntimeError("Existing R2 probe3 gate result changed")
            return rows
        plan_path = output / "probe3_plan.json"
        plan = {
            "schema_version": 1,
            "immutable_before_evaluation": True,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "effective_update_count": 50,
            "matrix": ["seed11 nominal", "seed11 left_offset", "seed22 right_offset"],
            "gate": (
                "valid=3/3 AND seed11 nominal full=true AND seed11 nominal rear_hold=true "
                "AND front_top>=2/3 AND no_severe>=2/3"
            ),
            "promotion_allowed": False,
            "main_launch_manifest": str(self.main_launch_path),
            "main_launch_manifest_sha256": sha256_file(self.main_launch_path),
        }
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan or not is_read_only(plan_path):
                raise RuntimeError("Existing R2 probe3 immutable plan is stale or writable")
        else:
            atomic_json(plan_path, plan, read_only=True)
        scenarios = (
            (11, "nominal", "0.00", "0.0"),
            (11, "left_offset", "0.12", "4.0"),
            (22, "right_offset", "-0.12", "-4.0"),
        )
        rows: list[dict[str, Any]] = []
        for seed, scenario, lateral, yaw in scenarios:
            self.require_no_gpu_job()
            log_path = output / f"seed{seed}_{scenario}.log"
            command = probe_play_command(
                checkpoint, seed=seed, lateral=lateral, yaw=yaw
            )
            launch_path = log_path.with_suffix(".launch.json")
            launch = {
                "schema_version": 1,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "seed": seed,
                "scenario": scenario,
                "lateral_offset_m": float(lateral),
                "yaw_offset_deg": float(yaw),
                "command": command,
            }
            if launch_path.exists():
                if not is_read_only(launch_path) or json.loads(
                    launch_path.read_text(encoding="utf-8")
                ) != launch:
                    raise RuntimeError("Existing R2 probe3 launch binding changed")
            else:
                atomic_json(launch_path, launch, read_only=True)
            if log_path.exists():
                payload = self._single_eval_payload(log_path)
                if not _same_scalar(payload.get("yaw_offset_deg"), float(yaw)):
                    raise RuntimeError("Existing R2 probe3 scenario geometry changed")
                validated = validate_probe_payload(payload, checkpoint)
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        **validated,
                        "log": str(log_path),
                        "log_sha256": sha256_file(log_path),
                    }
                )
                continue
            environment = os.environ.copy()
            environment.pop("DISPLAY", None)
            environment.pop("XAUTHORITY", None)
            with log_path.open("wb") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.wait_process(
                    process,
                    progress_root=output,
                    phase=f"probe3_seed{seed}_{scenario}",
                    stall_seconds=900,
                )
            payload = self._single_eval_payload(log_path)
            if not _same_scalar(payload.get("yaw_offset_deg"), float(yaw)):
                raise RuntimeError("R2 probe3 scenario geometry changed")
            validated = validate_probe_payload(payload, checkpoint)
            row = {
                "seed": seed,
                "scenario": scenario,
                **validated,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
            }
            rows.append(row)
            atomic_json(output / "probe3_rows_partial.json", {"rows": rows})
        passed, reason = gate_probe3(rows)
        manifest = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_r2_probe3",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "effective_update_count": 50,
            "valid_count": sum(row["valid"] is True for row in rows),
            "front_top_support_count": sum(row["front_top_support"] is True for row in rows),
            "no_severe_inward_count": sum(row["no_severe_inward"] is True for row in rows),
            "rows": rows,
            "passed": passed,
            "reason": reason,
            "completed_at": now_text(),
        }
        atomic_json(terminal_manifest, manifest, read_only=True)
        self.update_state(
            phase="r2_probe3_50",
            checkpoint=str(checkpoint),
            effective_updates=50,
            probe3={key: manifest[key] for key in ("valid_count", "front_top_support_count", "no_severe_inward_count", "passed", "reason")},
        )
        return rows

    @staticmethod
    def _configure_legacy_schema4_reader() -> Any:
        # When this supervisor is executed by absolute script path, Python puts
        # ROOT/tools (not ROOT) at sys.path[0].  Make the repository package
        # root explicit before importing the read-only legacy schema adapter.
        root_text = str(ROOT)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from tools import highstep_student_recovery_supervisor as legacy

        bindings = {
            "ROOT": ROOT,
            "TASK": EVAL_TASK,
            "WORKFLOW_ID": WORKFLOW_ID,
            "SPEC": SPEC,
            "SPEC_SHA256": SPEC_SHA256,
            "STATE_ROOT": STATE_ROOT,
            "STUDENT": B500,
            "STUDENT_SHA256": B500_SHA256,
            "TEACHER": TEACHER,
            "TEACHER_SHA256": TEACHER_SHA256,
            "PARENT_MANIFEST": PARENT_TEACHER_MANIFEST,
            "EXPERIMENT_ROOT": EXPERIMENT_ROOT,
        }
        for name, value in bindings.items():
            setattr(legacy, name, value)
        if any(getattr(legacy, name) != value for name, value in bindings.items()):
            raise RuntimeError("Failed to bind legacy schema4 reader to the R2 workflow")
        return legacy

    def _record_evaluation(
        self, label: str, checkpoint: Path, output: Path, summary: Mapping[str, Any]
    ) -> None:
        record = {
            "workflow_id": WORKFLOW_ID,
            "phase": label,
            "effective_updates": self.checkpoint_effective_count(checkpoint),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "output": str(output),
            "summary": dict(summary),
            "created_at": now_text(),
        }
        ledger = STATE_ROOT / "r2_evaluation_ledger.jsonl"
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.update_state(
            phase=label,
            checkpoint=str(checkpoint),
            effective_updates=record["effective_updates"],
            core9=dict(summary),
            evaluation_dir=str(output),
        )

    def run_core9(self, checkpoint: Path, label: str) -> tuple[Path, dict[str, Any]]:
        self.assert_main_training_authorized()
        expected_count = int(label.removeprefix("R2_"))
        self.audit_checkpoint(
            checkpoint,
            expected_count=expected_count,
            expected_load_mode=None,
        )
        self.require_no_gpu_job()
        monitor_path, monitor_sha = ensure_r2_monitor_snapshot()
        authorization = self.assert_main_training_authorized()
        if authorization.get("r2_monitor_snapshot_sha256") != monitor_sha:
            raise RuntimeError("R2 monitor snapshot differs from the immutable main launch binding")
        output = STATE_ROOT / "evaluations" / label
        if output.exists():
            legacy = self._configure_legacy_schema4_reader()
            try:
                launch = json.loads((output / "launch.json").read_text(encoding="utf-8"))
                if not (
                    launch.get("checkpoint") == str(checkpoint.resolve())
                    and launch.get("checkpoint_sha256") == sha256_file(checkpoint)
                    and launch.get("source_train_task") == TRAIN_TASK
                    and launch.get("eval_task") == EVAL_TASK
                    and launch.get("monitor_sha256") == monitor_sha
                ):
                    raise RuntimeError("stale launch")
                summary = legacy.Supervisor.read_evaluation(
                    self, output, expected_checkpoint=checkpoint
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
                if core9_matrix_was_fully_executed(output, checkpoint):
                    raise RuntimeError(
                        f"Completed {label} core9 failed schema/validity checks; hard gate is terminal and evaluation will not be retried"
                    ) from error
                # Preserve an incomplete infrastructure attempt; a read-only
                # evaluation may be repeated because no behavior gate existed.
                output = STATE_ROOT / "evaluations" / f"{label}_infra_recovery_{time.strftime('%Y%m%d_%H%M%S')}"
            else:
                summary = dict(summary)
                summary["r2_source_train_task"] = TRAIN_TASK
                summary["deployment_eval_task"] = EVAL_TASK
                summary["r2_schema4_monitor_sha256"] = monitor_sha
                self._record_evaluation(label, checkpoint, output, summary)
                return output, summary
        output.mkdir(parents=True)
        launcher_log = output / "supervisor_launcher.log"
        environment = os.environ.copy()
        environment.update(
            {
                "RECOVERY_SPEC_MODE": "1",
                "WORKFLOW_ID": WORKFLOW_ID,
                "LEDGER_FILE": str(STATE_ROOT / "r2_schema4_raw_ledger.jsonl"),
                "EVAL_CHECKPOINT_COUNT": "1",
                "EVAL_CHECKPOINT_PATH": str(checkpoint.resolve()),
                "EVAL_CHECKPOINT_STRIDE": "1",
                "EVAL_ALLOW_LEGACY_SCHEDULE_FALLBACK": "0",
                "POLL_SECONDS": "5",
                "PLAY_STALL_TIMEOUT_SECONDS": "900",
                "PLAY_WATCHDOG_POLL_SECONDS": "10",
                "MIN_FREE_DISK_GIB": "15",
            }
        )
        command = [
            "bash",
            str(monitor_path),
            "0",
            str(checkpoint.parent.resolve()),
            "/dev/null",
            str(output),
            EVAL_TASK,
            "student",
            str(PARENT_TEACHER_MANIFEST),
        ]
        atomic_json(
            output / "launch.json",
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "label": label,
                "command": command,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_train_task": TRAIN_TASK,
                "eval_task": EVAL_TASK,
                "monitor": str(monitor_path),
                "monitor_sha256": monitor_sha,
                "created_at": now_text(),
            },
        )
        with launcher_log.open("wb") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.wait_process(
                process,
                progress_root=output,
                phase=f"core9_{label}",
                stall_seconds=1200,
            )
        legacy = self._configure_legacy_schema4_reader()
        summary = legacy.Supervisor.read_evaluation(
            self,
            output,
            expected_checkpoint=checkpoint,
        )
        summary = dict(summary)
        summary["r2_source_train_task"] = TRAIN_TASK
        summary["deployment_eval_task"] = EVAL_TASK
        summary["r2_schema4_monitor_sha256"] = monitor_sha
        self._record_evaluation(label, checkpoint, output, summary)
        return output, summary

    def run_teacher_action_regression(
        self, checkpoint: Path
    ) -> tuple[Path, dict[str, Any]]:
        self.assert_main_training_authorized()
        count = self.checkpoint_effective_count(checkpoint)
        self.audit_checkpoint(checkpoint, expected_count=count, expected_load_mode=None)
        self.require_no_gpu_job()
        output = STATE_ROOT / "candidate_same_state" / f"R2_{count}"
        if output.exists():
            terminal_path = output / "teacher_action_regression.json"
            if terminal_path.is_file() and is_read_only(terminal_path):
                terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
                if not (
                    terminal.get("checkpoint") == str(checkpoint.resolve())
                    and terminal.get("checkpoint_sha256") == sha256_file(checkpoint)
                    and terminal.get("effective_update_count") == count
                    and terminal.get("threshold_multiplier") == 1.05
                ):
                    raise RuntimeError("Existing candidate action regression is stale")
                reference_paths = [Path(item["path"]) for item in terminal["reference"]["traces"]]
                candidate_paths = [Path(item["path"]) for item in terminal["candidate"]["traces"]]
                if aggregate_action_mae(reference_paths) != terminal["reference"]:
                    raise RuntimeError("Existing candidate regression B500 reference changed")
                if aggregate_action_mae(candidate_paths) != terminal["candidate"]:
                    raise RuntimeError("Existing candidate regression traces changed")
                return terminal_path, terminal
            raise RuntimeError(f"Candidate same-state output is partial and requires external recovery audit: {output}")
        output.mkdir(parents=True)
        candidate_sha = sha256_file(checkpoint)
        run_ids = ("seed11_nominal", "seed11_left_offset", "seed22_right_offset")
        plan = {
            "schema_version": 1,
            "kind": "highstep_r2_candidate_action_regression_plan",
            "immutable_before_evaluation": True,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": candidate_sha,
            "effective_update_count": count,
            "run_ids": list(run_ids),
            "reference_root": str(PINNED_BASE3_ROOT),
            "threshold_multiplier": 1.05,
            "adapter": str(CANDIDATE_AUDIT_ADAPTER),
            "adapter_sha256": sha256_file(CANDIDATE_AUDIT_ADAPTER),
            "main_launch_manifest": str(self.main_launch_path),
            "main_launch_manifest_sha256": sha256_file(self.main_launch_path),
        }
        atomic_json(output / "regression_plan.json", plan, read_only=True)
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
        environment["HIGHSTEP_R2_CANDIDATE_CHECKPOINT"] = str(checkpoint.resolve())
        environment["HIGHSTEP_R2_CANDIDATE_SHA256"] = candidate_sha
        environment["HIGHSTEP_CANDIDATE_STAGE"] = str(
            getattr(self, "candidate_stage", "R2")
        )
        for run_id in run_ids:
            self.require_no_gpu_job()
            log_path = output / f"{run_id}.log"
            command = [
                str(PYTHON),
                "-u",
                str(CANDIDATE_AUDIT_ADAPTER),
                "--trajectory",
                run_id,
                "--output-root",
                str(output),
                "--task",
                EVAL_TASK,
                "--num_envs",
                "1",
                "--checkpoint",
                str(checkpoint.resolve()),
                "--headless",
            ]
            atomic_json(
                output / f"{run_id}.launch.json",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": candidate_sha,
                    "command": command,
                    "adapter_sha256": sha256_file(CANDIDATE_AUDIT_ADAPTER),
                },
                read_only=True,
            )
            with log_path.open("wb") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.wait_process(
                    process,
                    output,
                    f"candidate_same_state_{run_id}",
                    stall_seconds=1200,
                )
            markers = [
                line
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if "[HIGHSTEP_CANDIDATE_SAME_STATE_AUDIT_JSON]" in line
            ]
            if len(markers) != 1:
                raise RuntimeError(f"Candidate adapter did not validate {run_id}")
        candidate_summaries = [output / run_id / "run_summary.json" for run_id in run_ids]
        candidate_frames: list[Path] = []
        for run_id, summary_path in zip(run_ids, candidate_summaries, strict=True):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            frames_path = Path(str(summary.get("frames_path", ""))).resolve(strict=True)
            if not (
                summary.get("run_id") == run_id
                and summary.get("frame_count") == 600
                and summary.get("runner_manual_action_allclose_all_frames") is True
                and summary.get("all_forward_state_digests_unchanged") is True
                and summary.get("all_observation_and_prior_inputs_unchanged") is True
                and finite_number(summary.get("action_decomposition_residual_global_max_abs"))
                and float(summary["action_decomposition_residual_global_max_abs"]) <= 1.0e-6
                and summary.get("checkpoint_binding", {}).get("student_checkpoint_sha256")
                == candidate_sha
                and summary.get("checkpoint_binding", {}).get(
                    "candidate_r2_extra_binding", {}
                ).get("stage")
                == str(getattr(self, "candidate_stage", "R2"))
                and summary.get("frames_sha256") == sha256_file(frames_path)
            ):
                raise RuntimeError(f"Candidate same-state summary failed: {summary_path}")
            candidate_frames.append(frames_path)
        bound_reference = pinned_base3_reference_binding()
        if self.assert_main_training_authorized().get(
            "pinned_b500_teacher_action_reference"
        ) != bound_reference:
            raise RuntimeError("Pinned B500 action reference changed during candidate audit")
        reference_frames = [
            Path(item["frames_path"]) for item in bound_reference["run_bindings"]
        ]
        reference = aggregate_action_mae(reference_frames)
        candidate = aggregate_action_mae(candidate_frames)
        revolute_limit = 1.05 * float(reference["all12_revolute_mae"])
        box_limit = 1.05 * float(reference["all4_box_mae"])
        passed = bool(
            float(candidate["all12_revolute_mae"]) <= revolute_limit
            and float(candidate["all4_box_mae"]) <= box_limit
        )
        manifest = {
            "schema_version": 1,
            "kind": "highstep_r2_candidate_teacher_action_regression",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": candidate_sha,
            "effective_update_count": count,
            "reference": reference,
            "candidate": candidate,
            "threshold_multiplier": 1.05,
            "revolute_mae_limit": revolute_limit,
            "box_mae_limit": box_limit,
            "all_frames_finite_parity_digest_decomposition": True,
            "passed": passed,
            "completed_at": now_text(),
        }
        manifest_path = output / "teacher_action_regression.json"
        atomic_json(manifest_path, manifest, read_only=True)
        self.update_state(
            phase="r2_teacher_action_regression",
            checkpoint=str(checkpoint),
            effective_updates=count,
            teacher_action_regression=manifest,
        )
        return manifest_path, manifest

    def finalize_behavior_candidate(
        self,
        *,
        checkpoint: Path,
        summary: dict[str, Any],
        evaluation_dir: Path,
    ) -> bool:
        regression_path, regression = self.run_teacher_action_regression(checkpoint)
        summary = dict(summary)
        summary["teacher_action_regression_audit"] = str(regression_path)
        summary["teacher_action_regression_audit_sha256"] = sha256_file(regression_path)
        summary["teacher_action_regression_audit_passed"] = regression["passed"]
        if regression["passed"] is not True:
            self.update_state(core9=summary)
            self.write_handoff(
                status="stopped_by_action_regression",
                reason="Behavior final gate passed but the preregistered Teacher-action regression failed",
                best_summary=summary,
            )
            return False
        legacy = self._configure_legacy_schema4_reader()
        artifacts = legacy.Supervisor.export_and_record_candidate(
            self, checkpoint, evaluation_dir
        )
        self.update_state(core9=summary)
        self.write_handoff(
            status="candidate_ready",
            reason="R2 behavior final gate and Teacher-action regression both passed; export/videos complete",
            best_summary=summary,
            candidate=True,
            candidate_artifacts=artifacts,
        )
        return True

    def write_handoff(
        self,
        *,
        status: str,
        reason: str,
        best_summary: Mapping[str, Any] | None = None,
        candidate: bool = False,
        candidate_artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "route": "R2",
            "status": status,
            "stop_reason": reason,
            "phase": self.state.get("phase"),
            "checkpoint": self.state.get("checkpoint", str(B500)),
            "effective_updates": self.state.get("effective_updates", 0),
            "best_core9": dict(best_summary) if best_summary is not None else self.state.get("core9"),
            "spec_sha256": SPEC_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "main_launch_manifest": (
                str(self.main_launch_path) if self.main_launch_path.is_file() else None
            ),
            "candidate": candidate,
            "candidate_artifacts": dict(candidate_artifacts) if candidate_artifacts is not None else None,
            "user_action_required": status not in {"main_authorized", "candidate_ready"},
            "completed_at": now_text(),
        }
        atomic_json(self.handoff_path, payload)
        self.update_state(
            status=status,
            stop_reason=reason,
            handoff=str(self.handoff_path),
            active_pid=None,
        )

    @staticmethod
    def _best_result(
        results: Sequence[tuple[Path, int, Mapping[str, Any], Path | None]]
    ) -> tuple[Path, int, Mapping[str, Any], Path | None]:
        return max(
            results,
            key=lambda item: final_selection_rank(item[2], item[1]),
        )

    def _stop_by_gate(
        self,
        *,
        reason: str,
        results: Sequence[tuple[Path, int, Mapping[str, Any], Path | None]],
    ) -> None:
        checkpoint, updates, summary, _ = self._best_result(results)
        self.update_state(
            checkpoint=str(checkpoint),
            effective_updates=updates,
            core9=dict(summary),
        )
        self.write_handoff(
            status="stopped_by_gate",
            reason=reason,
            best_summary=summary,
        )

    def run(self) -> None:
        if self.handoff_path.is_file():
            prior_handoff = json.loads(self.handoff_path.read_text(encoding="utf-8"))
            terminal_statuses = {
                "candidate_ready",
                "stopped_by_gate",
                "stopped_by_action_regression",
            }
            if (
                prior_handoff.get("workflow_id") == WORKFLOW_ID
                and prior_handoff.get("spec_sha256") == SPEC_SHA256
                and prior_handoff.get("preregistration_sha256") == PREREGISTRATION_SHA256
                and prior_handoff.get("status") in terminal_statuses
            ):
                self.update_state(
                    status=prior_handoff["status"],
                    phase="terminal_handoff_preserved",
                    stop_reason=prior_handoff.get("stop_reason"),
                    checkpoint=prior_handoff.get("checkpoint", str(B500)),
                    effective_updates=prior_handoff.get("effective_updates", 0),
                    handoff=str(self.handoff_path),
                )
                return
        self.preflight()
        if self.main_launch_path.is_file():
            self.assert_main_training_authorized()
        else:
            static_evidence = self.run_static_tests()
            self.smoke()
            self.write_main_launch_manifest(static_evidence)
        self.assert_main_training_authorized()
        baseline = {
            "valid_count": 9,
            "full_count": 6,
            "rear_hold_count": 6,
            "front_top_support_count": 8,
            "no_severe_inward_count": 9,
            "source": "preregistered qualified B500 core9",
        }
        results: list[tuple[Path, int, Mapping[str, Any], Path | None]] = [
            (B500, 0, baseline, None)
        ]

        _, checkpoint50, _ = self.run_train(
            label="main_50",
            checkpoint=B500,
            load_mode="weights_only",
            updates=50,
            num_envs=4096,
            smoke=False,
        )
        probe_rows = self.run_probe3(checkpoint50)
        allowed, reason = gate_probe3(probe_rows)
        if not allowed:
            self._stop_by_gate(reason=reason, results=results)
            return

        _, checkpoint100, _ = self.run_train(
            label="main_100",
            checkpoint=checkpoint50,
            load_mode="full",
            updates=50,
            num_envs=4096,
            smoke=False,
        )
        eval100, summary100 = self.run_core9(checkpoint100, "R2_100")
        results.append((checkpoint100, 100, summary100, eval100))
        if behavior_final_gate(summary100):
            self.finalize_behavior_candidate(
                checkpoint=checkpoint100,
                summary=summary100,
                evaluation_dir=eval100,
            )
            return
        allowed, reason = gate_100(summary100)
        if not allowed:
            self._stop_by_gate(reason=reason, results=results)
            return

        _, checkpoint300, _ = self.run_train(
            label="main_300",
            checkpoint=checkpoint100,
            load_mode="full",
            updates=200,
            num_envs=4096,
            smoke=False,
        )
        eval300, summary300 = self.run_core9(checkpoint300, "R2_300")
        results.append((checkpoint300, 300, summary300, eval300))
        if behavior_final_gate(summary300):
            self.finalize_behavior_candidate(
                checkpoint=checkpoint300,
                summary=summary300,
                evaluation_dir=eval300,
            )
            return
        allowed, reason = gate_300(summary300)
        if not allowed:
            self._stop_by_gate(reason=reason, results=results)
            return

        _, checkpoint500, _ = self.run_train(
            label="main_500",
            checkpoint=checkpoint300,
            load_mode="full",
            updates=200,
            num_envs=4096,
            smoke=False,
        )
        eval500, summary500 = self.run_core9(checkpoint500, "R2_500")
        results.append((checkpoint500, 500, summary500, eval500))
        if behavior_final_gate(summary500):
            self.finalize_behavior_candidate(
                checkpoint=checkpoint500,
                summary=summary500,
                evaluation_dir=eval500,
            )
            return
        allowed, reason = gate_500(summary500)
        if not allowed:
            self._stop_by_gate(reason=reason, results=results)
            return
        allowed, reason = gate_enter_1000(summary100, summary300, summary500)
        if not allowed:
            self._stop_by_gate(reason=reason, results=results)
            return

        _, checkpoint1000, _ = self.run_train(
            label="main_1000",
            checkpoint=checkpoint500,
            load_mode="full",
            updates=500,
            num_envs=4096,
            smoke=False,
        )
        eval1000, summary1000 = self.run_core9(checkpoint1000, "R2_1000")
        results.append((checkpoint1000, 1000, summary1000, eval1000))
        if behavior_final_gate(summary1000):
            self.finalize_behavior_candidate(
                checkpoint=checkpoint1000,
                summary=summary1000,
                evaluation_dir=eval1000,
            )
            return
        self._stop_by_gate(
            reason="R2 maximum 1000 effective updates reached without the final 8/9 gate",
            results=results,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (STATE_ROOT / "r2_supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("R2 supervisor lock is already held", file=sys.stderr)
        return 2
    supervisor = R2Supervisor()

    def handle_signal(signum: int, _frame: Any) -> None:
        supervisor.stop_active_process()
        supervisor.write_handoff(status="interrupted", reason=f"supervisor_signal_{signum}")
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        supervisor.run()
        return 0
    except KeyboardInterrupt:
        supervisor.stop_active_process()
        if supervisor.state.get("status") != "interrupted":
            supervisor.write_handoff(
                status="interrupted",
                reason="supervisor_keyboard_interrupt",
            )
        return 130
    except BaseException as error:
        supervisor.stop_active_process()
        supervisor.write_handoff(
            status="failed_closed",
            reason=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
