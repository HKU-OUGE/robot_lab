#!/usr/bin/env python3
"""Fail-closed checkpoint scope for the v1.10 E1400 continuation A/B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORKFLOW_ID = "highstep_e1400_continuation_ab_20260715"
SPEC_SHA = "3b526ddb88b60193d5fee1bb30f6473b5322df96eca393106ad680c6b245ffcf"
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
SOURCE = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "environment_curriculum_v18_Student/"
    "2026-07-14_21-03-31_v18_stage_a_E1400_20260714_210326/model_1394.pt"
)
SOURCE_SHA = "92bf3d0612f0f9ea1f85b379af8754a0df2710febb9b0067fd86e17487c810f9"
TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-"
    "ArcdogAdjustableLeg-v0"
)
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def checkpoint_scope(
    checkpoint: Path,
    expected_updates: int,
    *,
    branch: str,
    preregistration: Path,
    preregistration_sha256: str,
    run_dir: Path,
    dashboard: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    preregistration = preregistration.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    branch = branch.upper()
    if branch not in {"A", "B"} or expected_updates not in {1500, 1600, 1700}:
        raise RuntimeError("continuation checkpoint scope branch/update changed")
    checkpoint_sha = sha256_file(checkpoint)
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("E1400 source checkpoint changed")
    prereg = _json(preregistration)
    declaration = _json(dashboard)
    if not (
        sha256_file(preregistration) == preregistration_sha256
        and prereg.get("kind") == "highstep_e1400_continuation_ab_v110_runtime_preregistration"
        and prereg.get("workflow_id") == WORKFLOW_ID
        and prereg.get("authority", {}).get("version") == "v1.10"
        and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA
        and declaration.get("workflow_id") == WORKFLOW_ID
        and declaration.get("authority_version") == "v1.10"
        and declaration.get("spec_sha256") == SPEC_SHA
        and declaration.get("preregistration_sha256") == preregistration_sha256
        and Path(str(declaration.get("preregistration_path", ""))).resolve() == preregistration
    ):
        raise RuntimeError("continuation active authority changed")

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algorithm = candidate.get("infos", {}).get(ALGORITHM_KEY, {})
    recovery = algorithm.get("student_recovery", {})
    binding = recovery.get("binding_manifest", {})
    count = int(algorithm.get("student_distill_update_count", -1))
    candidate_iteration = int(candidate.get("iter", -1))
    expected_warmup = 1700 if branch == "A" else 1400
    if not (
        count == expected_updates
        and recovery.get("effective_update_count") == expected_updates
        and binding.get("effective_update_count") == expected_updates
        and recovery.get("stage") == "E1400_CONTINUATION"
        and binding.get("stage") == "E1400_CONTINUATION"
        and binding.get("continuation_branch") == branch
        and recovery.get("preregistration_sha256") == preregistration_sha256
        and binding.get("preregistration_sha256") == preregistration_sha256
        and binding.get("recovery_spec_sha256") == SPEC_SHA
        and binding.get("teacher_sha256") == TEACHER_SHA
        and binding.get("student_teacher_storage_independent") is True
        and binding.get("student_ppo_permanently_disabled") is True
        and binding.get("actor_body_frozen") is True
        and binding.get("critic_frozen") is True
        and binding.get("student_privileged_encoder_frozen") is True
        and binding.get("warmup_updates") == expected_warmup
        and binding.get("warmup_main_action_target") == "teacher_pre_prior"
        and binding.get("warmup_prior_box_loss_coefficient") == 0.0
        and binding.get("phase_scale") == 2.0
        and binding.get("rear_box_scale") == 1.5
    ):
        raise RuntimeError("continuation checkpoint binding changed")

    source_state = source["model_state_dict"]
    state = candidate["model_state_dict"]
    changed: list[str] = []
    forbidden: list[str] = []
    for name, before in source_state.items():
        after = state.get(name)
        if not torch.is_tensor(before) or not torch.is_tensor(after) or torch.equal(before, after):
            continue
        changed.append(name)
        if name.startswith("estimator."):
            continue
        if branch == "B" and name.endswith(("actor.6.weight", "actor.6.bias")):
            if torch.equal(before[:12], after[:12]):
                continue
        forbidden.append(name)
    if forbidden:
        raise RuntimeError(f"forbidden continuation parameter changes: {forbidden}")
    if branch == "A" and any(not name.startswith("estimator.") for name in changed):
        raise RuntimeError("estimator-only branch changed an actor tensor")

    optimizer = candidate.get("optimizer_state_dict")
    saved_optimizer = algorithm.get("vae_optimizer_state_dict")
    groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
    if not (
        _nested_equal(optimizer, saved_optimizer)
        and [len(group.get("params", [])) for group in groups] == [16, 2]
        and [group.get("v15_role") for group in groups] == ["estimator", "box_rows"]
        and [group.get("lr") for group in groups] == [1.0e-3, 1.0e-5]
        and len(optimizer.get("state", {})) >= 16
    ):
        raise RuntimeError("continuation full optimizer binding changed")

    schedule_path = run_dir / "params/highstep_schedule_manifest.json"
    runtime_path = run_dir / "params/highstep_runtime_state.json"
    environment_path = run_dir / "params/highstep_v18_environment_binding.json"
    schedule, runtime, environment = _json(schedule_path), _json(runtime_path), _json(environment_path)
    snapshots = [row for row in runtime.get("snapshots", []) if row.get("checkpoint_file") == checkpoint.name]
    if not (
        len(snapshots) == 1
        and snapshots[0].get("checkpoint_sha256") == checkpoint_sha
        and snapshots[0].get("runner_iteration") == candidate_iteration
        and snapshots[0].get("schedule_update") == float(expected_updates)
        and schedule.get("task") == TASK
        and schedule.get("checkpoint_load_mode") == "full"
        and schedule.get("schedule_resume_mode_resolved") == "preserve"
        and environment.get("task") == TASK
        and environment.get("source_env_yaml_sha256")
        == "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636"
    ):
        raise RuntimeError("continuation schedule/runtime/environment binding changed")

    return {
        "schema_version": 1,
        "kind": "highstep_e1400_continuation_checkpoint_scope",
        "workflow_id": WORKFLOW_ID,
        "branch": branch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "effective_updates": expected_updates,
        "runner_iteration": candidate_iteration,
        "run_dir": str(run_dir),
        "changed_parameter_names": changed,
        "forbidden_parameter_names": forbidden,
        "optimizer_group_sizes": [len(group["params"]) for group in groups],
        "optimizer_group_learning_rates": [group["lr"] for group in groups],
        "optimizer_state_entries": len(optimizer["state"]),
        "schedule_manifest": str(schedule_path),
        "schedule_manifest_sha256": sha256_file(schedule_path),
        "runtime_state": str(runtime_path),
        "runtime_state_sha256": sha256_file(runtime_path),
        "environment_binding": str(environment_path),
        "environment_binding_sha256": sha256_file(environment_path),
        "preregistration_sha256": preregistration_sha256,
    }
