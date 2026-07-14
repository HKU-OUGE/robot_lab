#!/usr/bin/env python3
"""Fail-closed autonomous v1.2 R3 highstep recovery supervisor."""

from __future__ import annotations

import copy
import fcntl
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import highstep_student_recovery_r2_supervisor as base  # noqa: E402


WORKFLOW_ID = "highstep_student_recovery_v12_20260713"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA256 = "053da1c6d30f9d5ed9aa4c2d5bedbab8d98130edb8a60f8e2657d4a8d0d99352"
ROUTE_REPORT = ROOT / "docs/robotlab_memory_zh/library/highstep_r2_failure_v12_route_report_20260713.md"
ROUTE_REPORT_SHA256 = "788b04bbf9b9e91dea0a9ceb2c23a52263f22bc9019bfd6da9371a3f25da63bf"
PREREGISTRATION = ROOT / "tmp/highstep_student_recovery_v12_20260713/r3_preregistration.json"
PREREGISTRATION_SHA256 = "13c184e4b6e2c3b514f52466dac1e27ff18256433716fd0c6b5d0890d93945e6"
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_v12_20260713"

TRAIN_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPriorR3-"
    "ArcdogAdjustableLeg-v0"
)
EVAL_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-"
    "ArcdogAdjustableLeg-v0"
)
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
    "student_recovery_r3_Student"
)
R250 = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
    "student_recovery_r2_Student/2026-07-13_03-37-24_student_recovery_r2_"
    "main_50_20260713_033718_2942949/model_49.pt"
)
R250_SHA256 = "e875424ed69eaaa474a9ac6849a6066dd264ce0eaf04ad63a418c3af1d66918c"
B500 = base.B500
B500_SHA256 = base.B500_SHA256
PROTECTED_ROOT = base.PROTECTED_ROOT
PROTECTED_ROOT_SHA256 = base.PROTECTED_ROOT_SHA256
TEACHER = base.TEACHER
TEACHER_SHA256 = base.TEACHER_SHA256
TEACHER_ENV = base.TEACHER_ENV
TEACHER_ENV_SHA256 = base.TEACHER_ENV_SHA256
PARENT_TEACHER_MANIFEST = base.PARENT_TEACHER_MANIFEST
SCHEMA4_MONITOR = base.SCHEMA4_MONITOR
MONITOR_SNAPSHOT = STATE_ROOT / "authority/highstep_r3_schema4_monitor.sh"
R3_OPTIMIZER_NAMES = ("actor.6.weight", "actor.6.bias")
ALGORITHM_STATE_KEY = base.ALGORITHM_STATE_KEY

# Rebind the reusable process, train, play, and schema4-reader machinery before
# any supervisor instance is constructed.  R2 terminal evidence is untouched.
for _name, _value in {
    "WORKFLOW_ID": WORKFLOW_ID,
    "SPEC": SPEC,
    "SPEC_SHA256": SPEC_SHA256,
    "PREREGISTRATION": PREREGISTRATION,
    "PREREGISTRATION_SHA256": PREREGISTRATION_SHA256,
    "STATE_ROOT": STATE_ROOT,
    "TRAIN_TASK": TRAIN_TASK,
    "EVAL_TASK": EVAL_TASK,
    "EXPERIMENT_ROOT": EXPERIMENT_ROOT,
    "B500": R250,
    "B500_SHA256": R250_SHA256,
    "PROTECTED_ROOT": PROTECTED_ROOT,
    "PROTECTED_ROOT_SHA256": PROTECTED_ROOT_SHA256,
    "TEACHER": TEACHER,
    "TEACHER_SHA256": TEACHER_SHA256,
    "TEACHER_ENV": TEACHER_ENV,
    "TEACHER_ENV_SHA256": TEACHER_ENV_SHA256,
    "R2_MONITOR_SNAPSHOT": MONITOR_SNAPSHOT,
}.items():
    setattr(base, _name, _value)

_original_pinned_base3_reference_binding = base.pinned_base3_reference_binding


def pinned_b500_reference_binding() -> dict[str, Any]:
    """Read the immutable base3 reference with its original B500 identity."""
    prior_path, prior_sha = base.B500, base.B500_SHA256
    try:
        base.B500, base.B500_SHA256 = B500, B500_SHA256
        return _original_pinned_base3_reference_binding()
    finally:
        base.B500, base.B500_SHA256 = prior_path, prior_sha


base.pinned_base3_reference_binding = pinned_b500_reference_binding


class TransientInfrastructureError(RuntimeError):
    """A stage may be restarted without changing its technical contract."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    return base.sha256_file(path)


def is_read_only(path: Path) -> bool:
    return base.is_read_only(path)


def now_text() -> str:
    return base.now_text()


def behavior_rank(summary: Mapping[str, Any]) -> tuple[int, int, int]:
    full = int(summary.get("full_count", -1))
    hold = int(summary.get("rear_hold_count", -1))
    front = int(summary.get("front_top_support_count", -1))
    return min(full, hold), full + hold, front


def final_selection_rank(summary: Mapping[str, Any], updates: int) -> tuple[int, ...]:
    full = int(summary.get("full_count", -1))
    hold = int(summary.get("rear_hold_count", -1))
    return (
        min(full, hold),
        full + hold,
        int(summary.get("no_severe_inward_count", -1)),
        int(summary.get("front_top_support_count", -1)),
        -int(updates),
    )


def final_gate(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("valid_count") == 9
        and int(summary.get("full_count", -1)) >= 8
        and int(summary.get("rear_hold_count", -1)) >= 8
        and int(summary.get("no_severe_inward_count", -1)) >= 8
    )


def core_gate(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("valid_count") == 9
        and int(summary.get("full_count", -1)) >= 7
        and int(summary.get("rear_hold_count", -1)) >= 7
        and int(summary.get("front_top_support_count", -1)) >= 7
        and int(summary.get("no_severe_inward_count", -1)) >= 8
    )


def probe_gate(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    expected = {(11, "nominal"), (11, "left_offset"), (22, "right_offset")}
    identities = {(int(row["seed"]), str(row["scenario"])) for row in rows}
    if len(rows) != 3 or identities != expected or not all(row.get("valid") is True for row in rows):
        return False, "R3-10 probe3 matrix/valid gate failed"
    by_identity = {(int(row["seed"]), str(row["scenario"])): row for row in rows}
    protected = [by_identity[(11, "nominal")], by_identity[(11, "left_offset")]]
    passed = bool(
        all(row.get("full_climb") is True and row.get("rear_hold") is True for row in protected)
        and sum(row.get("front_top_support") is True for row in rows) >= 2
        and sum(row.get("no_severe_inward") is True for row in rows) == 3
    )
    return passed, "R3-10 probe3 passed" if passed else "R3-10 probe3 hard gate failed"


def critical_file_paths() -> tuple[Path, ...]:
    paths = {
        SPEC,
        ROUTE_REPORT,
        PREREGISTRATION,
        ROOT / "tools/highstep_student_recovery_r3_supervisor.py",
        ROOT / "tools/highstep_student_recovery_r2_supervisor.py",
        ROOT / "tools/highstep_student_recovery_supervisor.py",
        ROOT / "scripts/systemd/highstep-student-recovery-r3.service",
        ROOT / "scripts/rsl_rl/base/train.py",
        ROOT / "scripts/rsl_rl/base/play.py",
        ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py",
        ROOT / "scripts/rsl_rl/base/highstep_candidate_same_state_audit_play.py",
        ROOT / "scripts/rsl_rl/base/highstep_same_state_audit_play.py",
        ROOT / "tools/highstep_same_state_audit.py",
        SCHEMA4_MONITOR,
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/observations.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__init__.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
        ROOT / "tests/test_highstep_student_recovery_r2.py",
        ROOT / "tests/test_highstep_candidate_same_state_audit.py",
    }
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"R3 implementation file missing: {missing}")
    return tuple(sorted(paths, key=str))


base.critical_file_paths = critical_file_paths


def ensure_monitor_snapshot() -> tuple[Path, str]:
    source = SCHEMA4_MONITOR.read_text(encoding="utf-8")
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
    replacement = needle[:-2] + f'''        or (
            # v1.2 R3 has a training-only teacher_context observation group.
            role == "student"
            and task == "{EVAL_TASK}"
            and source_manifest.get("task") == "{TRAIN_TASK}"
        )
    )'''
    if source.count(needle) != 1:
        raise RuntimeError("schema4 source-task block changed")
    rendered = source.replace(needle, replacement).encode("utf-8")
    MONITOR_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    if MONITOR_SNAPSHOT.exists():
        if MONITOR_SNAPSHOT.read_bytes() != rendered or not is_read_only(MONITOR_SNAPSHOT):
            raise RuntimeError("R3 monitor snapshot is stale or writable")
    else:
        temporary = MONITOR_SNAPSHOT.with_name(f".{MONITOR_SNAPSHOT.name}.{os.getpid()}.tmp")
        temporary.write_bytes(rendered)
        temporary.chmod(0o555)
        os.replace(temporary, MONITOR_SNAPSHOT)
    return MONITOR_SNAPSHOT, sha256_file(MONITOR_SNAPSHOT)


def _r3_optimizer_audit(
    optimizer: Mapping[str, Any], candidate_state: Mapping[str, torch.Tensor], count: int
) -> dict[str, Any]:
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state, Mapping):
        raise RuntimeError("R3 Adam must have exactly one parameter group")
    group = groups[0]
    parameter_ids = group.get("params")
    if not (
        group.get("name") == "full_action_head"
        and isinstance(parameter_ids, list)
        and len(parameter_ids) == 2
        and len(set(parameter_ids)) == 2
        and set(state) == set(parameter_ids)
        and float(group.get("lr", -1.0)) == 5.0e-6
        and tuple(float(v) for v in group.get("betas", ())) == (0.9, 0.999)
        and float(group.get("eps", -1.0)) == 1.0e-8
        and float(group.get("weight_decay", -1.0)) == 0.0
        and group.get("amsgrad") is False
    ):
        raise RuntimeError("R3 Adam contract changed")
    steps: list[int] = []
    for parameter_id, name in zip(parameter_ids, R3_OPTIMIZER_NAMES, strict=True):
        item = state[parameter_id]
        raw_step = item.get("step") if isinstance(item, Mapping) else None
        step = int(raw_step.item()) if isinstance(raw_step, torch.Tensor) else int(raw_step)
        if step != count * 4:
            raise RuntimeError(f"R3 Adam step mismatch: {step} != {count * 4}")
        steps.append(step)
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = item.get(moment_name)
            if not (
                isinstance(moment, torch.Tensor)
                and tuple(moment.shape) == tuple(candidate_state[name].shape)
                and bool(torch.isfinite(moment).all().item())
            ):
                raise RuntimeError(f"R3 Adam {moment_name} invalid for {name}")
    return {"parameter_names": list(R3_OPTIMIZER_NAMES), "adam_steps": steps, "finite": True}


def audit_r3_checkpoint_payload(
    anchor_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    *,
    expected_count: int,
    expected_load_mode: str | None,
) -> dict[str, Any]:
    anchor = anchor_payload.get("model_state_dict")
    candidate = candidate_payload.get("model_state_dict")
    if not isinstance(anchor, Mapping) or not isinstance(candidate, Mapping) or set(anchor) != set(candidate):
        raise RuntimeError("R3 candidate/anchor state keys differ")
    changed: list[str] = []
    for name in sorted(anchor):
        before, after = anchor[name], candidate[name]
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            raise RuntimeError(f"R3 state is not tensor-only at {name}")
        if before.shape != after.shape or before.dtype != after.dtype or not bool(torch.isfinite(after).all().item()):
            raise RuntimeError(f"R3 tensor contract/finite failure at {name}")
        if not torch.equal(before, after):
            if name not in R3_OPTIMIZER_NAMES:
                raise RuntimeError(f"R3 changed frozen tensor {name}")
            changed.append(name)
    if not changed:
        raise RuntimeError("R3 checkpoint produced no action-head change")
    infos = candidate_payload.get("infos")
    extra = infos.get(ALGORITHM_STATE_KEY) if isinstance(infos, Mapping) else None
    recovery = extra.get("student_recovery") if isinstance(extra, Mapping) else None
    if not (
        isinstance(extra, Mapping)
        and extra.get("schema_version") == 1
        and extra.get("algorithm_class") == "VAEPPO"
        and extra.get("distill_stage") == 2
        and extra.get("student_distill_update_count") == expected_count
        and isinstance(recovery, Mapping)
        and recovery.get("schema_version") == 1
        and recovery.get("stage") == "R3"
        and recovery.get("effective_update_count") == expected_count
        and recovery.get("preregistration_sha256") == PREREGISTRATION_SHA256
        and tuple(recovery.get("optimizer_parameter_names") or ()) == R3_OPTIMIZER_NAMES
    ):
        raise RuntimeError("R3 checkpoint stage/count/preregistration mismatch")
    binding = recovery.get("binding_manifest")
    expected = {
        "stage": "R3",
        "preregistration_path": os.path.realpath(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protected_student_root": os.path.realpath(PROTECTED_ROOT),
        "protected_student_root_sha256": PROTECTED_ROOT_SHA256,
        "behavior_reference_b500": os.path.realpath(B500),
        "behavior_reference_b500_sha256": B500_SHA256,
        "initial_student_checkpoint": os.path.realpath(R250),
        "initial_student_sha256": R250_SHA256,
        "teacher_checkpoint": os.path.realpath(TEACHER),
        "teacher_sha256": TEACHER_SHA256,
        "teacher_env_yaml": os.path.realpath(TEACHER_ENV),
        "teacher_env_yaml_sha256": TEACHER_ENV_SHA256,
        "schedule_resume_mode": "preserve",
        "effective_update_count": expected_count,
        "optimizer_parameter_tensor_count": 2,
        "optimizer_parameter_names": list(R3_OPTIMIZER_NAMES),
    }
    if expected_load_mode is not None:
        expected["checkpoint_load_mode"] = expected_load_mode
    mismatches = {
        key: {"actual": binding.get(key) if isinstance(binding, Mapping) else None, "expected": value}
        for key, value in expected.items()
        if not isinstance(binding, Mapping) or binding.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"R3 immutable checkpoint binding mismatch: {mismatches}")
    runtime = binding.get("runtime_contract")
    if not (
        binding.get("teacher_independent_storage") is True
        and binding.get("anchor_independent_storage") is True
        and isinstance(runtime, Mapping)
        and runtime.get("joint_order") == list(base.JOINT_ORDER)
        and base.exact_numbers(runtime.get("action_scale"), base.ACTION_SCALE)
        and base.exact_numbers(runtime.get("action_offset"), base.ACTION_OFFSET)
        and runtime.get("joint_pos_clip") == [[-60.0, 60.0] for _ in range(16)]
        and runtime.get("teacher_context_shape") == [3]
        and runtime.get("teacher_context_order") == ["height_delta", "command_x", "front_rear_delta"]
    ):
        raise RuntimeError("R3 runtime/action/independent-module contract changed")
    runner_optimizer = candidate_payload.get("optimizer_state_dict")
    algorithm_optimizer = extra.get("vae_optimizer_state_dict")
    if not (
        isinstance(runner_optimizer, Mapping)
        and isinstance(algorithm_optimizer, Mapping)
        and base._nested_equal(runner_optimizer, algorithm_optimizer)
    ):
        raise RuntimeError("R3 runner/algorithm Adam states disagree")
    return {
        "effective_update_count": expected_count,
        "changed_live_tensors": changed,
        "frozen_tensor_violation_count": 0,
        "adam": _r3_optimizer_audit(runner_optimizer, candidate, expected_count),
        "lineage_and_runtime_verified": True,
    }


class R3Supervisor(base.R2Supervisor):
    maximum_effective_updates = 100
    training_binding_filename = "highstep_student_recovery_r3_binding.json"
    candidate_stage = "R3"

    def __init__(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_path = STATE_ROOT / "state.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.preflight_path = STATE_ROOT / "r3_preflight_manifest.json"
        self.smoke_path = STATE_ROOT / "r3_smoke_audit.json"
        self.main_launch_path = STATE_ROOT / "r3_main_launch_manifest.json"
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
        self.state = {
            **existing,
            "schema_version": 2,
            "workflow_id": WORKFLOW_ID,
            "route": "R3",
            "spec_sha256": SPEC_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "supervisor_pid": os.getpid(),
            "status": "starting",
            "phase": existing.get("phase", "r3_preflight"),
            "checkpoint": existing.get("checkpoint", str(R250)),
            "effective_updates": existing.get("effective_updates", 0),
            "stop_reason": None,
        }
        self.update_state()

    @classmethod
    def checkpoint_effective_count(cls, path: Path) -> int:
        if path.resolve() == R250.resolve():
            return 0
        payload = cls.checkpoint_payload(path)
        extra = ((payload.get("infos") or {}).get(ALGORITHM_STATE_KEY) or {})
        recovery = extra.get("student_recovery") or {}
        count = recovery.get("effective_update_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= 100:
            raise RuntimeError(f"checkpoint has no valid R3 effective count: {path}")
        return count

    def audit_checkpoint(
        self, checkpoint: Path, *, expected_count: int, expected_load_mode: str | None
    ) -> dict[str, Any]:
        return audit_r3_checkpoint_payload(
            self.checkpoint_payload(R250),
            self.checkpoint_payload(checkpoint),
            expected_count=expected_count,
            expected_load_mode=expected_load_mode,
        )

    def preflight(self) -> dict[str, Any]:
        self.require_no_gpu_job()
        if sha256_file(SPEC) != SPEC_SHA256 or sha256_file(ROUTE_REPORT) != ROUTE_REPORT_SHA256:
            raise RuntimeError("R3 spec/route report authority SHA mismatch")
        if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256 or not is_read_only(PREREGISTRATION):
            raise RuntimeError("R3 preregistration SHA/read-only binding failed")
        prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
        if not (
            prereg.get("kind") == "highstep_student_recovery_r3_preregistration"
            and prereg.get("workflow_id") == WORKFLOW_ID
            and prereg.get("approved_version") == "v1.2"
            and prereg.get("route", {}).get("selected") == "R3"
            and prereg.get("training_allowed") is False
            and prereg.get("gates", {}).get("maximum_effective_updates") == 100
            and prereg.get("automation", {}).get("codex_model_if_ever_used") == "gpt-5.6-sol"
            and prereg.get("automation", {}).get("codex_reasoning_effort_if_ever_used") == "max"
            and prereg.get("automation", {}).get("ultra_forbidden") is True
        ):
            raise RuntimeError("R3 preregistration authority fields changed")
        for path, digest in (
            (R250, R250_SHA256), (B500, B500_SHA256),
            (PROTECTED_ROOT, PROTECTED_ROOT_SHA256), (TEACHER, TEACHER_SHA256),
            (TEACHER_ENV, TEACHER_ENV_SHA256),
        ):
            if sha256_file(path) != digest:
                raise RuntimeError(f"R3 bound checkpoint/config changed: {path}")
        self.critical_hashes = {str(path.resolve()): sha256_file(path) for path in critical_file_paths()}
        manifest = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_r3_preflight",
            "workflow_id": WORKFLOW_ID,
            "created_at": now_text(),
            "spec_sha256": SPEC_SHA256,
            "route_report_sha256": ROUTE_REPORT_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "start_and_anchor": str(R250),
            "start_and_anchor_sha256": R250_SHA256,
            "behavior_reference_b500_sha256": B500_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "train_task": TRAIN_TASK,
            "eval_task": EVAL_TASK,
            "critical_file_sha256": self.critical_hashes,
            "no_concurrent_train_eval_play": True,
            "main_training_authorized": False,
        }
        base.atomic_json(self.preflight_path, manifest)
        self.update_state(status="preflight_passed", phase="r3_static_then_smoke")
        return manifest

    @staticmethod
    def _audit_smoke_log(log_path: Path) -> dict[str, Any]:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        before = re.findall(r"R3_Buffer_Step_Before_Clear[^\n]*?24(?:\.0+)?", text)
        after = re.findall(r"R3_Buffer_Step_After_Clear[^\n]*?0(?:\.0+)?", text)
        if not before or not after:
            raise RuntimeError("R3 smoke did not prove storage fill/clear")
        return {"buffer_step_before_clear": 24, "buffer_step_after_clear": 0}

    def smoke(self) -> Path:
        if self.smoke_path.is_file():
            if not is_read_only(self.smoke_path):
                raise RuntimeError("existing R3 smoke evidence is writable")
            existing = json.loads(self.smoke_path.read_text(encoding="utf-8"))
            checkpoint4 = Path(existing["fresh4_checkpoint"]).resolve(strict=True)
            checkpoint5 = Path(existing["full_resume5_checkpoint"]).resolve(strict=True)
            audit4 = self.audit_checkpoint(checkpoint4, expected_count=4, expected_load_mode="weights_only")
            audit5 = self.audit_checkpoint(checkpoint5, expected_count=5, expected_load_mode="full")
            if any(b + 4 != a for b, a in zip(audit4["adam"]["adam_steps"], audit5["adam"]["adam_steps"], strict=True)):
                raise RuntimeError("R3 smoke full-resume Adam continuity failed")
            return checkpoint5
        _, checkpoint4, log4 = self.run_train(
            label="smoke_fresh4", checkpoint=R250, load_mode="weights_only",
            updates=4, num_envs=256, smoke=True,
        )
        audit4 = self.audit_checkpoint(checkpoint4, expected_count=4, expected_load_mode="weights_only")
        _, checkpoint5, log5 = self.run_train(
            label="smoke_full_resume1", checkpoint=checkpoint4, load_mode="full",
            updates=1, num_envs=256, smoke=True,
        )
        audit5 = self.audit_checkpoint(checkpoint5, expected_count=5, expected_load_mode="full")
        if any(b + 4 != a for b, a in zip(audit4["adam"]["adam_steps"], audit5["adam"]["adam_steps"], strict=True)):
            raise RuntimeError("R3 smoke full-resume Adam continuity failed")
        payload = {
            "schema_version": 1, "kind": "highstep_student_recovery_r3_discarded_smoke",
            "workflow_id": WORKFLOW_ID, "passed": True, "discard_branch": True,
            "fresh4_checkpoint": str(checkpoint4), "fresh4_checkpoint_sha256": sha256_file(checkpoint4),
            "fresh4_audit": audit4, "fresh4_storage_audit": self._audit_smoke_log(log4),
            "full_resume5_checkpoint": str(checkpoint5), "full_resume5_checkpoint_sha256": sha256_file(checkpoint5),
            "full_resume5_audit": audit5, "full_resume5_storage_audit": self._audit_smoke_log(log5),
            "main_checkpoint_must_restart_fresh_from_r2_50": True, "completed_at": now_text(),
        }
        base.atomic_json(self.smoke_path, payload, read_only=True)
        self.update_state(status="smoke_passed", phase="r3_main_launch_binding", checkpoint=str(R250), effective_updates=0)
        return checkpoint5

    def write_main_launch_manifest(self, static_evidence: Path) -> Path:
        self.assert_code_unchanged()
        if not self.smoke_path.is_file() or not is_read_only(self.smoke_path):
            raise RuntimeError("R3 discarded smoke evidence missing")
        monitor, monitor_sha = ensure_monitor_snapshot()
        pinned_reference = base.pinned_base3_reference_binding()
        payload = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_r3_main_launch_authorization",
            "workflow_id": WORKFLOW_ID,
            "created_at": now_text(),
            "spec_sha256": SPEC_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "preregistration_training_allowed_latch": False,
            "static_passed": True,
            "static_evidence": str(static_evidence),
            "static_evidence_sha256": sha256_file(static_evidence),
            "smoke_passed": True,
            "smoke_discarded": True,
            "smoke_evidence": str(self.smoke_path),
            "smoke_evidence_sha256": sha256_file(self.smoke_path),
            "critical_file_sha256": self.critical_hashes,
            "r2_monitor_snapshot": str(monitor),
            "r2_monitor_snapshot_sha256": monitor_sha,
            "only_schedule_task_alias": {"source_train_task": TRAIN_TASK, "eval_task": EVAL_TASK},
            "pinned_b500_teacher_action_reference": pinned_reference,
            "behavior_start": str(R250),
            "behavior_start_sha256": R250_SHA256,
            "main_initial_load": "R2-50 weights_only schedule preserve fresh R3 Adam count=0",
            "main_training_authorized": True,
            "maximum_effective_updates": 100,
            "threshold_changes_allowed": False,
        }
        if self.main_launch_path.exists():
            self.assert_main_training_authorized()
            return self.main_launch_path
        base.atomic_json(self.main_launch_path, payload, read_only=True)
        self.assert_main_training_authorized()
        return self.main_launch_path

    def assert_main_training_authorized(self) -> dict[str, Any]:
        if not self.main_launch_path.is_file() or not is_read_only(self.main_launch_path):
            raise RuntimeError("immutable R3 main launch manifest absent")
        payload = json.loads(self.main_launch_path.read_text(encoding="utf-8"))
        if not (
            payload.get("kind") == "highstep_student_recovery_r3_main_launch_authorization"
            and payload.get("workflow_id") == WORKFLOW_ID
            and payload.get("spec_sha256") == SPEC_SHA256
            and payload.get("preregistration_sha256") == PREREGISTRATION_SHA256
            and payload.get("static_passed") is True
            and payload.get("smoke_passed") is True
            and payload.get("smoke_discarded") is True
            and payload.get("main_training_authorized") is True
            and payload.get("maximum_effective_updates") == 100
            and payload.get("threshold_changes_allowed") is False
            and payload.get("critical_file_sha256") == self.critical_hashes
        ):
            raise RuntimeError("R3 main launch authorization changed")
        for path_key, sha_key in (
            ("static_evidence", "static_evidence_sha256"),
            ("smoke_evidence", "smoke_evidence_sha256"),
            ("r2_monitor_snapshot", "r2_monitor_snapshot_sha256"),
        ):
            path = Path(payload[path_key]).resolve(strict=True)
            if not is_read_only(path) or sha256_file(path) != payload[sha_key]:
                raise RuntimeError(f"R3 bound evidence changed: {path_key}")
        if payload.get("pinned_b500_teacher_action_reference") != base.pinned_base3_reference_binding():
            raise RuntimeError("R3 pinned B500 Teacher-action reference changed")
        self.assert_code_unchanged()
        return payload

    def run_probe3(self, checkpoint: Path) -> list[dict[str, Any]]:
        self.assert_main_training_authorized()
        self.audit_checkpoint(checkpoint, expected_count=10, expected_load_mode=None)
        output = STATE_ROOT / "evaluations/R3_probe3_10"
        output.mkdir(parents=True, exist_ok=True)
        terminal = output / "probe3_manifest.json"
        scenarios = (
            (11, "nominal", "0.00", "0.0"),
            (11, "left_offset", "0.12", "4.0"),
            (22, "right_offset", "-0.12", "-4.0"),
        )
        if terminal.is_file():
            if not is_read_only(terminal):
                raise RuntimeError("R3 probe terminal manifest writable")
            stored = json.loads(terminal.read_text(encoding="utf-8"))
            if stored.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise RuntimeError("R3 probe checkpoint changed")
            rows = list(stored.get("rows", []))
            passed, reason = probe_gate(rows)
            if stored.get("passed") is not passed or stored.get("reason") != reason:
                raise RuntimeError("R3 probe stored gate changed")
            return rows
        plan = {
            "schema_version": 1, "immutable_before_evaluation": True,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "effective_update_count": 10,
            "matrix": [f"seed{s} {n}" for s, n, _, _ in scenarios],
            "gate": "valid=3; nominal+left full/hold; front>=2; no_severe=3",
            "main_launch_manifest_sha256": sha256_file(self.main_launch_path),
        }
        plan_path = output / "probe3_plan.json"
        if plan_path.exists():
            if not is_read_only(plan_path) or json.loads(plan_path.read_text()) != plan:
                raise RuntimeError("R3 probe plan changed")
        else:
            base.atomic_json(plan_path, plan, read_only=True)
        rows: list[dict[str, Any]] = []
        for seed, scenario, lateral, yaw in scenarios:
            self.require_no_gpu_job()
            log = output / f"seed{seed}_{scenario}.log"
            command = base.probe_play_command(checkpoint, seed=seed, lateral=lateral, yaw=yaw)
            launch = log.with_suffix(".launch.json")
            launch_payload = {
                "schema_version": 1, "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint), "seed": seed,
                "scenario": scenario, "lateral_offset_m": float(lateral),
                "yaw_offset_deg": float(yaw), "command": command,
            }
            if launch.exists():
                if not is_read_only(launch) or json.loads(launch.read_text()) != launch_payload:
                    raise RuntimeError("R3 probe launch changed")
            else:
                base.atomic_json(launch, launch_payload, read_only=True)
            if not log.exists():
                environment = os.environ.copy(); environment.pop("DISPLAY", None); environment.pop("XAUTHORITY", None)
                with log.open("wb") as stream:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    self.wait_process(process, output, f"probe3_seed{seed}_{scenario}", stall_seconds=900)
            payload = self._single_eval_payload(log)
            if not base._same_scalar(payload.get("yaw_offset_deg"), float(yaw)):
                raise RuntimeError("R3 probe geometry changed")
            validated = base.validate_probe_payload(payload, checkpoint)
            rows.append({"seed": seed, "scenario": scenario, **validated, "log": str(log), "log_sha256": sha256_file(log)})
            base.atomic_json(output / "probe3_rows_partial.json", {"rows": rows})
        passed, reason = probe_gate(rows)
        manifest = {
            "schema_version": 1, "kind": "highstep_student_recovery_r3_probe3",
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "effective_update_count": 10, "rows": rows, "passed": passed, "reason": reason,
            "completed_at": now_text(),
        }
        base.atomic_json(terminal, manifest, read_only=True)
        self.update_state(phase="r3_probe3_10", probe3=manifest, checkpoint=str(checkpoint), effective_updates=10)
        return rows

    def run_core9(self, checkpoint: Path, label: str) -> tuple[Path, dict[str, Any]]:
        self.assert_main_training_authorized()
        expected_count = int(label.removeprefix("R3_"))
        self.audit_checkpoint(checkpoint, expected_count=expected_count, expected_load_mode=None)
        self.require_no_gpu_job()
        monitor, monitor_sha = ensure_monitor_snapshot()
        if self.assert_main_training_authorized().get("r2_monitor_snapshot_sha256") != monitor_sha:
            raise RuntimeError("R3 monitor differs from launch binding")
        output = STATE_ROOT / "evaluations" / label
        if output.exists():
            legacy = self._configure_legacy_schema4_reader()
            try:
                launch = json.loads((output / "launch.json").read_text(encoding="utf-8"))
                if not (
                    launch.get("checkpoint_sha256") == sha256_file(checkpoint)
                    and launch.get("source_train_task") == TRAIN_TASK
                    and launch.get("eval_task") == EVAL_TASK
                    and launch.get("monitor_sha256") == monitor_sha
                ):
                    raise RuntimeError("stale core9 launch")
                summary = legacy.Supervisor.read_evaluation(self, output, expected_checkpoint=checkpoint)
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
                if base.core9_matrix_was_fully_executed(output, checkpoint):
                    raise RuntimeError(f"completed {label} core9 failed validity; terminal") from error
                output = STATE_ROOT / "evaluations" / f"{label}_infra_recovery_{time.strftime('%Y%m%d_%H%M%S')}"
            else:
                summary = dict(summary)
                summary.update({"r3_source_train_task": TRAIN_TASK, "deployment_eval_task": EVAL_TASK, "r3_schema4_monitor_sha256": monitor_sha})
                self._record_evaluation(label, checkpoint, output, summary)
                return output, summary
        output.mkdir(parents=True)
        environment = os.environ.copy()
        environment.update({
            "RECOVERY_SPEC_MODE": "1", "WORKFLOW_ID": WORKFLOW_ID,
            "LEDGER_FILE": str(STATE_ROOT / "r3_schema4_raw_ledger.jsonl"),
            "EVAL_CHECKPOINT_COUNT": "1", "EVAL_CHECKPOINT_PATH": str(checkpoint.resolve()),
            "EVAL_CHECKPOINT_STRIDE": "1", "EVAL_ALLOW_LEGACY_SCHEDULE_FALLBACK": "0",
            "POLL_SECONDS": "5", "PLAY_STALL_TIMEOUT_SECONDS": "900",
            "PLAY_WATCHDOG_POLL_SECONDS": "10", "MIN_FREE_DISK_GIB": "15",
        })
        command = ["bash", str(monitor), "0", str(checkpoint.parent.resolve()), "/dev/null", str(output), EVAL_TASK, "student", str(PARENT_TEACHER_MANIFEST)]
        base.atomic_json(output / "launch.json", {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "label": label,
            "command": command, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint), "source_train_task": TRAIN_TASK,
            "eval_task": EVAL_TASK, "monitor": str(monitor), "monitor_sha256": monitor_sha,
            "created_at": now_text(),
        })
        with (output / "supervisor_launcher.log").open("wb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.wait_process(process, output, f"core9_{label}", stall_seconds=1200)
        legacy = self._configure_legacy_schema4_reader()
        summary = dict(legacy.Supervisor.read_evaluation(self, output, expected_checkpoint=checkpoint))
        summary.update({"r3_source_train_task": TRAIN_TASK, "deployment_eval_task": EVAL_TASK, "r3_schema4_monitor_sha256": monitor_sha})
        self._record_evaluation(label, checkpoint, output, summary)
        return output, summary

    def write_handoff(
        self, *, status: str, reason: str, best_summary: Mapping[str, Any] | None = None,
        candidate: bool = False, candidate_artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "route": "R3",
            "status": status, "stop_reason": reason, "phase": self.state.get("phase"),
            "checkpoint": self.state.get("checkpoint", str(R250)),
            "effective_updates": self.state.get("effective_updates", 0),
            "best_core9": dict(best_summary) if best_summary is not None else self.state.get("core9"),
            "spec_sha256": SPEC_SHA256, "preregistration_sha256": PREREGISTRATION_SHA256,
            "main_launch_manifest": str(self.main_launch_path) if self.main_launch_path.is_file() else None,
            "candidate": candidate,
            "candidate_artifacts": dict(candidate_artifacts) if candidate_artifacts is not None else None,
            "user_action_required": status not in {"candidate_ready"}, "completed_at": now_text(),
        }
        base.atomic_json(self.handoff_path, payload)
        self.update_state(status=status, stop_reason=reason, handoff=str(self.handoff_path), active_pid=None)

    @staticmethod
    def _best(results: Sequence[tuple[Path, int, Mapping[str, Any], Path | None]]) -> tuple[Path, int, Mapping[str, Any], Path | None]:
        return max(results, key=lambda item: final_selection_rank(item[2], item[1]))

    def _stop(self, reason: str, results: Sequence[tuple[Path, int, Mapping[str, Any], Path | None]]) -> None:
        checkpoint, updates, summary, _ = self._best(results)
        self.update_state(checkpoint=str(checkpoint), effective_updates=updates, core9=dict(summary))
        self.write_handoff(status="stopped_by_gate", reason=reason, best_summary=summary)

    def run(self) -> None:
        if self.handoff_path.is_file():
            handoff = json.loads(self.handoff_path.read_text(encoding="utf-8"))
            if (
                handoff.get("workflow_id") == WORKFLOW_ID
                and handoff.get("spec_sha256") == SPEC_SHA256
                and handoff.get("preregistration_sha256") == PREREGISTRATION_SHA256
                and handoff.get("status") in {"candidate_ready", "stopped_by_gate", "stopped_by_action_regression"}
            ):
                self.update_state(status=handoff["status"], phase="terminal_handoff_preserved", stop_reason=handoff.get("stop_reason"))
                return
        self.preflight()
        if self.main_launch_path.is_file():
            self.assert_main_training_authorized()
        else:
            static = self.run_static_tests()
            self.smoke()
            self.write_main_launch_manifest(static)
        self.assert_main_training_authorized()
        results: list[tuple[Path, int, Mapping[str, Any], Path | None]] = [
            (B500, 0, {"valid_count": 9, "full_count": 6, "rear_hold_count": 6, "front_top_support_count": 8, "no_severe_inward_count": 9, "source": "B500 core9"}, None),
            (R250, 0, {"valid_count": 9, "full_count": 6, "rear_hold_count": 6, "front_top_support_count": 6, "no_severe_inward_count": 9, "source": "R2-50 core9"}, None),
        ]
        _, checkpoint10, _ = self.run_train(label="main_10", checkpoint=R250, load_mode="weights_only", updates=10, num_envs=4096, smoke=False)
        rows = self.run_probe3(checkpoint10)
        passed, reason = probe_gate(rows)
        if not passed:
            self._stop(reason, results); return
        _, checkpoint25, _ = self.run_train(label="main_25", checkpoint=checkpoint10, load_mode="full", updates=15, num_envs=4096, smoke=False)
        eval25, summary25 = self.run_core9(checkpoint25, "R3_25")
        results.append((checkpoint25, 25, summary25, eval25))
        if final_gate(summary25):
            self.finalize_behavior_candidate(checkpoint=checkpoint25, summary=summary25, evaluation_dir=eval25); return
        if not core_gate(summary25):
            self._stop("R3-25 hard gate failed", results); return
        _, checkpoint50, _ = self.run_train(label="main_50", checkpoint=checkpoint25, load_mode="full", updates=25, num_envs=4096, smoke=False)
        eval50, summary50 = self.run_core9(checkpoint50, "R3_50")
        results.append((checkpoint50, 50, summary50, eval50))
        if final_gate(summary50):
            self.finalize_behavior_candidate(checkpoint=checkpoint50, summary=summary50, evaluation_dir=eval50); return
        if not core_gate(summary50) or behavior_rank(summary50) <= behavior_rank(summary25):
            self._stop("R3-50 gate or strict rank improvement failed", results); return
        _, checkpoint100, _ = self.run_train(label="main_100", checkpoint=checkpoint50, load_mode="full", updates=50, num_envs=4096, smoke=False)
        eval100, summary100 = self.run_core9(checkpoint100, "R3_100")
        results.append((checkpoint100, 100, summary100, eval100))
        if final_gate(summary100):
            self.finalize_behavior_candidate(checkpoint=checkpoint100, summary=summary100, evaluation_dir=eval100); return
        self._stop("R3 maximum 100 effective updates reached without final 8/9 gate; no R4", results)


def _failure_is_transient(supervisor: R3Supervisor, error: BaseException) -> bool:
    phase = str(supervisor.state.get("phase", ""))
    if not phase.startswith(("train_", "probe3_", "core9_", "candidate_same_state_")):
        return False
    text = f"{type(error).__name__}: {error}".lower()
    for path in sorted(STATE_ROOT.rglob("*.log"), key=lambda item: item.stat().st_mtime_ns)[-3:]:
        try:
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")[-24000:].lower()
        except OSError:
            pass
    permanent = (
        "sha256 mismatch", "binding mismatch", "contract changed", "frozen tensor",
        "preregistration", "unauthorized", "scope changed", "lineage does not bind",
        "schedule preserve", "hard gate", "completed r3_",
    )
    if any(marker in text for marker in permanent):
        return False
    transient = (
        "return code", "made no observable progress", "cuda out of memory", "outofmemoryerror",
        "resource temporarily unavailable", "connection reset", "input/output error",
        "no space left", "device is busy", "failed to create window", "x11",
    )
    return any(marker in text for marker in transient)


def main() -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (STATE_ROOT / "r3_supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("R3 supervisor lock already held", file=sys.stderr)
        return 2
    supervisor = R3Supervisor()

    def handle_signal(signum: int, _frame: Any) -> None:
        supervisor.stop_active_process()
        supervisor.write_handoff(status="interrupted", reason=f"supervisor_signal_{signum}")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        supervisor.run()
        return 0
    except KeyboardInterrupt:
        supervisor.stop_active_process()
        return 130
    except BaseException as error:
        supervisor.stop_active_process()
        if _failure_is_transient(supervisor, error):
            retry_path = STATE_ROOT / "transient_retries.json"
            retries = json.loads(retry_path.read_text()) if retry_path.is_file() else {}
            phase = str(supervisor.state.get("phase", "unknown"))
            attempt = int(retries.get(phase, 0)) + 1
            retries[phase] = attempt
            base.atomic_json(retry_path, retries)
            if attempt <= 3:
                supervisor.update_state(status="transient_retry_pending", stop_reason=f"attempt {attempt}/3: {type(error).__name__}: {error}")
                return 75
        supervisor.write_handoff(status="failed_closed", reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
