#!/usr/bin/env python3
"""Narrow, fail-closed checkpoint scope audit for the v1.8 curriculum route.

This module intentionally does not modify or widen the historical v1.5 helper.
It accepts only the active v1.8 workflow and requires an explicit immutable
rebinding manifest when a completed checkpoint predates an infrastructure-only
preregistration amendment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORKFLOW_ID = "highstep_student_env_curriculum_v18_20260714"
SPEC_SHA = "e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4"
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
SOURCE = ROOT / (
    "tmp/highstep_historical_0707_exact_new_teacher_20260714/"
    "source_model_172300/model_172300.pt"
)
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"
STAGE_A_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-"
    "ArcdogAdjustableLeg-v0"
)
STAGE_B_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Robust-"
    "ArcdogAdjustableLeg-v0"
)


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
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _authority(
    checkpoint: Path,
    checkpoint_sha256: str,
    binding: Mapping[str, Any],
    preregistration: Path,
    preregistration_sha256: str,
    dashboard: Path,
    authority_rebinding: Path | None,
) -> dict[str, Any]:
    declaration = _json(dashboard)
    if not (
        declaration.get("workflow_id") == WORKFLOW_ID
        and declaration.get("authority_version") == "v1.8"
        and declaration.get("spec_sha256") == SPEC_SHA
        and Path(str(declaration.get("preregistration_path", ""))).resolve() == preregistration.resolve()
        and declaration.get("preregistration_sha256") == preregistration_sha256
        and sha256_file(preregistration) == preregistration_sha256
    ):
        raise RuntimeError("v1.8 active dashboard authority changed")
    current = _json(preregistration)
    if not (
        current.get("kind") == "highstep_student_environment_curriculum_v18_preregistration"
        and current.get("workflow_id") == WORKFLOW_ID
        and current.get("authority", {}).get("version") == "v1.8"
        and current.get("authority", {}).get("spec_sha256") == SPEC_SHA
    ):
        raise RuntimeError("v1.8 preregistration identity changed")

    bound_path = Path(str(binding.get("preregistration_path", "")))
    bound_sha = str(binding.get("preregistration_sha256", ""))
    if not bound_path.is_absolute() or sha256_file(bound_path) != bound_sha:
        raise RuntimeError("checkpoint-bound v1.8 preregistration changed")
    if bound_path.resolve() == preregistration.resolve() and bound_sha == preregistration_sha256:
        return {"mode": "direct", "checkpoint_preregistration_sha256": bound_sha}
    if authority_rebinding is None:
        raise RuntimeError("v1.8 checkpoint authority requires an explicit rebinding manifest")
    rebinding = _json(authority_rebinding)
    if not (
        sha256_file(authority_rebinding) == rebinding.get("self_sha256_excluded")
        if "self_sha256_excluded" in rebinding else True
    ):
        raise RuntimeError("v1.8 rebinding manifest self binding changed")
    if not (
        rebinding.get("kind") == "highstep_v18_checkpoint_scope_infrastructure_rebinding"
        and rebinding.get("workflow_id") == WORKFLOW_ID
        and rebinding.get("checkpoint") == str(checkpoint.resolve())
        and rebinding.get("checkpoint_sha256") == checkpoint_sha256
        and rebinding.get("checkpoint_preregistration_path") == str(bound_path.resolve())
        and rebinding.get("checkpoint_preregistration_sha256") == bound_sha
        and rebinding.get("runtime_preregistration_path") == str(preregistration.resolve())
        and rebinding.get("runtime_preregistration_sha256") == preregistration_sha256
        and rebinding.get("changed_training_semantics") is False
        and rebinding.get("old_v15_helper_authority_modified") is False
    ):
        raise RuntimeError("v1.8 checkpoint authority rebinding changed")
    return {
        "mode": "infrastructure_rebinding",
        "manifest": str(authority_rebinding.resolve()),
        "manifest_sha256": sha256_file(authority_rebinding),
        "checkpoint_preregistration_sha256": bound_sha,
        "runtime_preregistration_sha256": preregistration_sha256,
    }


def checkpoint_scope_v18(
    checkpoint: Path,
    expected_updates: int,
    *,
    preregistration: Path,
    preregistration_sha256: str,
    run_dir: Path,
    authority_rebinding: Path | None = None,
    dashboard: Path = DASHBOARD,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    checkpoint_sha = sha256_file(checkpoint)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algorithm = candidate.get("infos", {}).get(ALGORITHM_KEY, {})
    recovery = algorithm.get("student_recovery", {})
    binding = recovery.get("binding_manifest", {})
    count = int(algorithm.get("student_distill_update_count", -1))
    candidate_iteration = int(candidate.get("iter", -1))
    if not (
        count == expected_updates
        and recovery.get("effective_update_count") == expected_updates
        and binding.get("effective_update_count") == expected_updates
        and recovery.get("stage") == "ENV_CURRICULUM_V18"
        and binding.get("stage") == "ENV_CURRICULUM_V18"
    ):
        raise RuntimeError("v1.8 effective update/count binding changed")
    authority = _authority(
        checkpoint, checkpoint_sha, binding, preregistration,
        preregistration_sha256, dashboard, authority_rebinding,
    )
    if not (
        sha256_file(SOURCE) == TEACHER_SHA
        and binding.get("initial_student_sha256") == TEACHER_SHA
        and binding.get("teacher_sha256") == TEACHER_SHA
        and binding.get("recovery_spec_sha256") == SPEC_SHA
        and recovery.get("preregistration_sha256") == binding.get("preregistration_sha256")
        and binding.get("student_teacher_storage_independent") is True
        and binding.get("student_ppo_permanently_disabled") is True
        and binding.get("actor_body_frozen") is True
        and binding.get("critic_frozen") is True
        and binding.get("student_privileged_encoder_frozen") is True
        and binding.get("warmup_updates") == 1400
        and binding.get("warmup_main_action_target") == "teacher_pre_prior"
        and binding.get("warmup_prior_box_loss_coefficient") == 0.0
        and binding.get("phase_scale") == 2.0
        and binding.get("rear_box_scale") == 1.5
    ):
        raise RuntimeError("v1.8 checkpoint training-contract binding changed")

    changed: list[str] = []
    forbidden: list[str] = []
    source_state = source["model_state_dict"]
    state = candidate["model_state_dict"]
    for name, before in source_state.items():
        after = state.get(name)
        if not torch.is_tensor(before) or not torch.is_tensor(after) or torch.equal(before, after):
            continue
        changed.append(name)
        if name.startswith("estimator."):
            continue
        if expected_updates > 1400 and name.endswith(("actor.6.weight", "actor.6.bias")):
            if torch.equal(before[:12], after[:12]):
                continue
        forbidden.append(name)
    if forbidden or (expected_updates <= 1400 and any(not name.startswith("estimator.") for name in changed)):
        raise RuntimeError(f"forbidden v1.8 parameter changes: {forbidden or changed}")

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
        raise RuntimeError("v1.8 full optimizer binding changed")

    schedule_path = run_dir / "params/highstep_schedule_manifest.json"
    runtime_path = run_dir / "params/highstep_runtime_state.json"
    environment_path = run_dir / "params/highstep_student_environment_curriculum_v18_binding.json"
    schedule, runtime, environment = _json(schedule_path), _json(runtime_path), _json(environment_path)
    snapshots = [row for row in runtime.get("snapshots", []) if row.get("checkpoint_file") == checkpoint.name]
    if not (
        len(snapshots) == 1
        and snapshots[0].get("checkpoint_sha256") == checkpoint_sha
        and snapshots[0].get("runner_iteration") == candidate_iteration
        and snapshots[0].get("schedule_update") == float(expected_updates)
        and schedule.get("task") in {STAGE_A_TASK, STAGE_B_TASK}
        and environment.get("task") == schedule.get("task")
        and environment.get("workflow_id") == WORKFLOW_ID
        and environment.get("teacher_sha256") == TEACHER_SHA
        and environment.get("warmup_updates") == 1400
        and environment.get("phase_scale") == 2.0
        and environment.get("rear_box_scale") == 1.5
        and environment.get("warmup_prior_box_loss_coefficient") == 0.0
    ):
        raise RuntimeError("v1.8 schedule/runtime/environment checkpoint binding changed")
    return {
        "schema_version": 1,
        "kind": "highstep_v18_checkpoint_scope",
        "workflow_id": WORKFLOW_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "effective_updates": expected_updates,
        "runner_iteration": candidate_iteration,
        "run_dir": str(run_dir),
        "changed_parameter_names": changed,
        "forbidden_parameter_names": forbidden,
        "actor_fully_frozen_through_warmup": expected_updates <= 1400,
        "optimizer_group_sizes": [len(group["params"]) for group in groups],
        "optimizer_group_learning_rates": [group["lr"] for group in groups],
        "optimizer_state_entries": len(optimizer["state"]),
        "schedule_manifest": str(schedule_path),
        "schedule_manifest_sha256": sha256_file(schedule_path),
        "runtime_state": str(runtime_path),
        "runtime_state_sha256": sha256_file(runtime_path),
        "environment_binding": str(environment_path),
        "environment_binding_sha256": sha256_file(environment_path),
        "authority": authority,
        "old_v15_helper_authority_used": False,
    }
