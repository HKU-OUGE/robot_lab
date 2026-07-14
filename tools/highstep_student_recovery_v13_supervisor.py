#!/usr/bin/env python3
"""Autonomous fail-closed supervisor for approved highstep recovery v1.3."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
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


WORKFLOW_ID = "highstep_student_recovery_v13_20260713"
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_v13_20260713"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA256 = "a1fc5e1283606b69bec7b7256c20abf53003601e9a127d303294a160cbb57223"
R4_PREREG = STATE_ROOT / "r4_preregistration.json"
R4_PREREG_SHA256 = "8f43112e94fbe032fd4d640292151210b692c3f853d460a98a0456fb0ed7b45e"
R5_PREREG = STATE_ROOT / "r5_preregistration.json"
R5_PREREG_SHA256 = "c222cbeafc5ca1dbd1cb94cdb93b588a25cc7b7d160af8385c4fdd8a14a48736"
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0"
TRACE_SCRIPT = ROOT / "scripts/rsl_rl/base/highstep_v13_core9_trace_play.py"
FIT_SCRIPT = ROOT / "tools/highstep_v13_r4_fit.py"
R5_SCRIPT = ROOT / "tools/highstep_v13_r5_train.py"
SERVICE = ROOT / "scripts/systemd/highstep-student-recovery-v13.service"
REPAIR_SERVICE = ROOT / "scripts/systemd/highstep-student-recovery-v13-repair.service"
REPAIR_SCRIPT = ROOT / "tools/highstep_v13_failure_wake_codex.py"
SCHEMA4_MONITOR = ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh"
PARENT_TEACHER_MANIFEST = base.PARENT_TEACHER_MANIFEST
B500 = base.B500
B500_SHA256 = base.B500_SHA256
TEACHER = base.TEACHER
TEACHER_SHA256 = base.TEACHER_SHA256
PROTECTED_ROOT = base.PROTECTED_ROOT
PROTECTED_ROOT_SHA256 = base.PROTECTED_ROOT_SHA256
R4_NAMES = ("actor.6.weight", "actor.6.bias")
R5_NAMES = ("actor.4.weight", "actor.4.bias", "actor.6.weight", "actor.6.bias")
ALGORITHM_STATE_KEY = base.ALGORITHM_STATE_KEY


for _name, _value in {
    "WORKFLOW_ID": WORKFLOW_ID,
    "SPEC": SPEC,
    "SPEC_SHA256": SPEC_SHA256,
    "PREREGISTRATION": R4_PREREG,
    "PREREGISTRATION_SHA256": R4_PREREG_SHA256,
    "STATE_ROOT": STATE_ROOT,
    "TRAIN_TASK": TASK,
    "EVAL_TASK": TASK,
    "EXPERIMENT_ROOT": STATE_ROOT / "r4_candidates",
    "B500": B500,
    "B500_SHA256": B500_SHA256,
}.items():
    setattr(base, _name, _value)


def sha256_file(path: os.PathLike[str] | str) -> str:
    return base.sha256_file(path)


def now_text() -> str:
    return base.now_text()


def is_read_only(path: Path) -> bool:
    return base.is_read_only(path)


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    base.atomic_json(path, payload, read_only=read_only)


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
        and int(summary.get("front_top_support_count", -1)) >= 8
        and int(summary.get("no_severe_inward_count", -1)) >= 8
    )


def behavior_rank(summary: Mapping[str, Any], budget: int) -> tuple[int, ...]:
    full = int(summary.get("full_count", -1))
    hold = int(summary.get("rear_hold_count", -1))
    return (
        min(full, hold), full + hold,
        int(summary.get("no_severe_inward_count", -1)),
        int(summary.get("front_top_support_count", -1)), -budget,
    )


def critical_file_paths() -> tuple[Path, ...]:
    paths = {
        SPEC, R4_PREREG, R5_PREREG, TRACE_SCRIPT, FIT_SCRIPT, R5_SCRIPT, SERVICE,
        REPAIR_SERVICE, REPAIR_SCRIPT,
        ROOT / "tools/highstep_student_recovery_v13_supervisor.py",
        ROOT / "tools/highstep_student_recovery_r2_supervisor.py",
        ROOT / "tools/highstep_student_recovery_supervisor.py",
        ROOT / "tools/highstep_same_state_audit.py",
        ROOT / "scripts/rsl_rl/base/highstep_same_state_audit_play.py",
        ROOT / "scripts/rsl_rl/base/highstep_candidate_same_state_audit_play.py",
        ROOT / "scripts/rsl_rl/base/play.py",
        ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py",
        ROOT / "scripts/rsl_rl/cli_args.py",
        ROOT / "tests/test_highstep_v13_r4.py",
        SCHEMA4_MONITOR,
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
    }
    paths.update(ROOT.glob("tests/test_*.py"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"v1.3 implementation file missing: {missing}")
    return tuple(sorted(paths, key=str))


base.critical_file_paths = critical_file_paths


def _checkpoint_scope(
    anchor_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    allowed: Sequence[str],
) -> dict[str, Any]:
    if set(anchor_state) != set(candidate_state):
        raise RuntimeError("v1.3 candidate/anchor state keys differ")
    changed: list[str] = []
    for name in anchor_state:
        before, after = anchor_state[name], candidate_state[name]
        if before.shape != after.shape or before.dtype != after.dtype or not bool(torch.isfinite(after).all().item()):
            raise RuntimeError(f"v1.3 tensor contract failed: {name}")
        if not torch.equal(before, after):
            if name not in allowed:
                raise RuntimeError(f"v1.3 changed frozen tensor: {name}")
            changed.append(name)
    if not changed:
        raise RuntimeError("v1.3 candidate produced no allowed parameter change")
    return {"changed_tensors": sorted(changed), "frozen_tensor_violation_count": 0}


class V13Supervisor(base.R2Supervisor):
    candidate_stage = "R4"

    def __init__(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_path = STATE_ROOT / "state.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.preflight_path = STATE_ROOT / "v13_preflight_manifest.json"
        self.smoke_path = STATE_ROOT / "unused_r2_smoke.json"
        self.main_launch_path = STATE_ROOT / "v13_main_launch_manifest.json"
        self.active_process: subprocess.Popen[bytes] | None = None
        self.critical_hashes: dict[str, str] = {}
        existing: dict[str, Any] = {}
        if self.state_path.is_file():
            try:
                value = json.loads(self.state_path.read_text())
                if value.get("workflow_id") == WORKFLOW_ID:
                    existing = value
            except json.JSONDecodeError:
                pass
        self.state = {
            **existing,
            "schema_version": 2,
            "workflow_id": WORKFLOW_ID,
            "route": existing.get("route", "R4"),
            "spec_sha256": SPEC_SHA256,
            "r4_preregistration_sha256": R4_PREREG_SHA256,
            "r5_preregistration_sha256": R5_PREREG_SHA256,
            "supervisor_pid": os.getpid(),
            "status": "starting",
            "phase": existing.get("phase", "v13_preflight"),
            "checkpoint": existing.get("checkpoint", str(B500)),
            "effective_updates": existing.get("effective_updates", 0),
            "stop_reason": None,
        }
        self.update_state()

    @classmethod
    def checkpoint_effective_count(cls, path: Path) -> int:
        payload = cls.checkpoint_payload(path)
        infos = payload.get("infos") or {}
        r5 = infos.get("highstep_v13_r5")
        if isinstance(r5, Mapping):
            return int(r5["effective_epochs"])
        return 0

    def audit_checkpoint(
        self,
        checkpoint: Path,
        *,
        expected_stage: str | None = None,
        expected_count: int | None = None,
        expected_load_mode: str | None = None,
    ) -> dict[str, Any]:
        anchor = self.checkpoint_payload(B500)
        candidate = self.checkpoint_payload(checkpoint)
        infos = candidate.get("infos") or {}
        if isinstance(infos.get("highstep_v13_r4"), Mapping):
            stage, allowed, extra = "R4", R4_NAMES, infos["highstep_v13_r4"]
            required_sha = R4_PREREG_SHA256
        elif isinstance(infos.get("highstep_v13_r5"), Mapping):
            stage, allowed, extra = "R5", R5_NAMES, infos["highstep_v13_r5"]
            required_sha = R5_PREREG_SHA256
        else:
            raise RuntimeError("v1.3 checkpoint lacks R4/R5 binding")
        if expected_stage is not None and stage != expected_stage:
            raise RuntimeError(f"v1.3 checkpoint stage mismatch: {stage} != {expected_stage}")
        if not (
            extra.get("stage") == stage
            and extra.get("workflow_id") == WORKFLOW_ID
            and extra.get("preregistration_sha256") == required_sha
            and extra.get("anchor_checkpoint_sha256") == B500_SHA256
            and extra.get("trainable_tensors") == list(allowed)
            and isinstance(extra.get("dataset_manifest"), str)
            and sha256_file(extra["dataset_manifest"]) == extra.get("dataset_manifest_sha256")
        ):
            raise RuntimeError(f"v1.3 {stage} checkpoint immutable binding failed")
        scope = _checkpoint_scope(anchor["model_state_dict"], candidate["model_state_dict"], allowed)
        if expected_count is not None and self.checkpoint_effective_count(checkpoint) != expected_count:
            raise RuntimeError("v1.3 checkpoint effective count changed")
        return {"stage": stage, "scope": scope, "binding_verified": True}

    def preflight(self) -> dict[str, Any]:
        self.require_no_gpu_job()
        for path, digest in (
            (SPEC, SPEC_SHA256), (R4_PREREG, R4_PREREG_SHA256),
            (R5_PREREG, R5_PREREG_SHA256), (B500, B500_SHA256),
            (TEACHER, TEACHER_SHA256), (PROTECTED_ROOT, PROTECTED_ROOT_SHA256),
        ):
            if sha256_file(path) != digest:
                raise RuntimeError(f"v1.3 authority SHA mismatch: {path}")
        if not is_read_only(R4_PREREG) or not is_read_only(R5_PREREG):
            raise RuntimeError("v1.3 preregistration is writable")
        r4 = json.loads(R4_PREREG.read_text())
        r5 = json.loads(R5_PREREG.read_text())
        if not (
            r4.get("workflow_id") == WORKFLOW_ID
            and r4.get("training_allowed") is False
            and len(r4.get("core9_matrix") or []) == 9
            and r5.get("workflow_id") == WORKFLOW_ID
            and r5.get("training_allowed") is False
            and r5.get("absolute_epoch_cap") == 5
        ):
            raise RuntimeError("v1.3 preregistration fields changed")
        for item in r4["terminal_evidence"].values():
            if sha256_file(item["path"]) != item["sha256"]:
                raise RuntimeError(f"v1.3 terminal evidence changed: {item['path']}")
        self.critical_hashes = {str(path.resolve()): sha256_file(path) for path in critical_file_paths()}
        payload = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_v13_preflight",
            "workflow_id": WORKFLOW_ID,
            "created_at": now_text(),
            "spec_sha256": SPEC_SHA256,
            "r4_preregistration_sha256": R4_PREREG_SHA256,
            "r5_preregistration_sha256": R5_PREREG_SHA256,
            "b500_sha256": B500_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "critical_file_sha256": self.critical_hashes,
            "no_concurrent_train_eval_play": True,
            "main_parameter_update_authorized": False,
        }
        atomic_json(self.preflight_path, payload)
        self.update_state(status="preflight_passed", phase="v13_static")
        return payload

    def run_static_tests(self) -> Path:
        # Static evidence is immutable, so a legitimate pre-launch code repair
        # must create a new SHA-addressed record instead of either overwriting
        # the old record or failing forever on its now-stale binding.
        binding_digest = hashlib.sha256(
            json.dumps(self.critical_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        evidence = STATE_ROOT / f"v13_static_evidence_{binding_digest}.json"
        log_path = STATE_ROOT / f"v13_static_tests_{binding_digest}.log"
        if evidence.is_file():
            if not is_read_only(evidence):
                raise RuntimeError("Existing v1.3 static evidence is writable")
            existing = json.loads(evidence.read_text(encoding="utf-8"))
            bound_tests = existing.get("test_file_sha256")
            if not (
                existing.get("passed") is True
                and existing.get("py_compile_return_code") == 0
                and existing.get("pytest_return_code") == 0
                and existing.get("critical_file_sha256") == self.critical_hashes
                and isinstance(bound_tests, Mapping)
                and all(
                    Path(path).is_file() and sha256_file(path) == digest
                    for path, digest in bound_tests.items()
                )
                and Path(str(existing.get("log", ""))).is_file()
                and sha256_file(existing["log"]) == existing.get("log_sha256")
            ):
                raise RuntimeError("Existing v1.3 SHA-addressed static evidence is invalid")
            self.update_state(status="static_reused", phase="v13_dataset", static_evidence=str(evidence))
            return evidence
        tests = sorted(ROOT.glob("tests/test_*.py"))
        if not tests:
            raise RuntimeError("No CPU/static tests were found")
        python_sources = [path for path in critical_file_paths() if path.suffix == ".py"]
        with log_path.open("wb") as stream:
            compile_command = [str(PYTHON), "-m", "py_compile", *map(str, python_sources)]
            stream.write(("COMMAND " + " ".join(compile_command) + "\n").encode())
            compiled = subprocess.run(
                compile_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
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
            raise RuntimeError(f"v1.3 static/CPU tests failed: {log_path}")
        atomic_json(
            evidence,
            {
                "schema_version": 1,
                "kind": "highstep_student_recovery_v13_static_evidence",
                "workflow_id": WORKFLOW_ID,
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
        self.update_state(status="static_passed", phase="v13_dataset", static_evidence=str(evidence))
        return evidence

    @staticmethod
    def _trace_validation(summary_path: Path, role: str) -> dict[str, Any]:
        summary = json.loads(summary_path.read_text())
        outcome = summary.get("outcome") or {}
        frames = Path(summary["frames_path"]).resolve(strict=True)
        if not (
            summary.get("frame_count") == 600
            and outcome.get("loop_steps") == 600
            and outcome.get("terminated") is False
            and summary.get("runner_manual_action_allclose_all_frames") is True
            and summary.get("all_forward_state_digests_unchanged") is True
            and summary.get("all_observation_and_prior_inputs_unchanged") is True
            and float(summary.get("action_decomposition_residual_global_max_abs", 1.0)) <= 1.0e-6
            and summary.get("frames_sha256") == sha256_file(frames)
        ):
            raise RuntimeError(f"v1.3 trace integrity failed: {summary_path}")
        succeeded = outcome.get("full_climb_success") is True and outcome.get("rear_on_platform_hold_success") is True
        if (role == "success_anchor") is not succeeded:
            raise RuntimeError(f"B500 trace behavior differs from preregistered role: {summary_path}")
        return {
            "frames": str(frames), "frames_sha256": sha256_file(frames),
            "summary": str(summary_path), "summary_sha256": sha256_file(summary_path),
            "full_climb_success": outcome.get("full_climb_success"),
            "rear_hold_success": outcome.get("rear_on_platform_hold_success"),
            "front_top_support": outcome.get("front_top_support_reached"),
        }

    def collect_trace(self, row: Mapping[str, Any]) -> dict[str, Any]:
        run_id = f"seed{int(row['seed'])}_{row['scenario']}"
        stage_result = STATE_ROOT / "trace_stage_results" / f"{run_id}.json"
        if stage_result.is_file():
            if not is_read_only(stage_result):
                raise RuntimeError("v1.3 trace stage result writable")
            record = json.loads(stage_result.read_text())
            validation = self._trace_validation(Path(record["summary"]), str(row["b500_role"]))
            if record.get("frames_sha256") != validation["frames_sha256"]:
                raise RuntimeError("v1.3 trace stage result changed")
            return record
        self.assert_code_unchanged()
        self.require_no_gpu_job()
        attempt_root = STATE_ROOT / "trace_attempts" / f"{run_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        attempt_root.mkdir(parents=True)
        log = attempt_root / "trace.log"
        command = [
            str(PYTHON), "-u", str(TRACE_SCRIPT), "--trajectory", run_id,
            "--output-root", str(attempt_root), "--task", TASK, "--num_envs", "1",
            "--checkpoint", str(B500), "--headless",
        ]
        atomic_json(attempt_root / "launch.json", {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "run_id": run_id,
            "command": command, "checkpoint_sha256": B500_SHA256,
            "trace_script_sha256": sha256_file(TRACE_SCRIPT), "created_at": now_text(),
        }, read_only=True)
        environment = os.environ.copy(); environment.pop("DISPLAY", None); environment.pop("XAUTHORITY", None)
        environment["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libstdc++.so.6"
        with log.open("wb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.wait_process(process, attempt_root, f"trace_{run_id}", stall_seconds=1200)
        summary_path = attempt_root / run_id / "run_summary.json"
        validation = self._trace_validation(summary_path, str(row["b500_role"]))
        Path(validation["frames"]).chmod(0o444)
        summary_path.chmod(0o444)
        record = {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "run_id": run_id,
            "seed": int(row["seed"]), "scenario": row["scenario"],
            "role": row["b500_role"], **validation,
            "log": str(log), "log_sha256": sha256_file(log), "completed_at": now_text(),
        }
        atomic_json(stage_result, record, read_only=True)
        return record

    def collect_dataset(self) -> Path:
        manifest = STATE_ROOT / "replay_dataset_manifest.json"
        if manifest.is_file():
            if not is_read_only(manifest):
                raise RuntimeError("v1.3 dataset manifest writable")
            payload = json.loads(manifest.read_text())
            if not (payload.get("matrix_complete") is True and payload.get("total_frames") == 5400):
                raise RuntimeError("v1.3 dataset manifest invalid")
            for item in payload["trajectories"]:
                if sha256_file(item["frames"]) != item["frames_sha256"]:
                    raise RuntimeError("v1.3 dataset trace changed")
            return manifest
        prereg = json.loads(R4_PREREG.read_text())
        records: list[dict[str, Any]] = []
        for row in prereg["core9_matrix"]:
            self.update_state(status="running", phase=f"trace_seed{row['seed']}_{row['scenario']}")
            records.append(self.collect_trace(row))
        if len(records) != 9 or sum(item["role"] == "success_anchor" for item in records) != 6:
            raise RuntimeError("v1.3 collected dataset role matrix changed")
        payload = {
            "schema_version": 1, "kind": "highstep_v13_b500_core9_replay_dataset",
            "workflow_id": WORKFLOW_ID, "preregistration": str(R4_PREREG),
            "preregistration_sha256": R4_PREREG_SHA256, "anchor_checkpoint": str(B500),
            "anchor_checkpoint_sha256": B500_SHA256, "matrix_complete": True,
            "total_frames": 5400, "trajectories": records, "created_at": now_text(),
        }
        atomic_json(manifest, payload, read_only=True)
        self.update_state(status="dataset_ready", phase="r4_launch_binding", dataset_manifest=str(manifest))
        return manifest

    def write_main_launch_manifest(self, static_evidence: Path, dataset_manifest: Path) -> Path:
        self.assert_code_unchanged()
        payload = {
            "schema_version": 1, "kind": "highstep_v13_parameter_update_authorization",
            "workflow_id": WORKFLOW_ID, "created_at": now_text(),
            "spec_sha256": SPEC_SHA256, "r4_preregistration_sha256": R4_PREREG_SHA256,
            "r5_preregistration_sha256": R5_PREREG_SHA256,
            "static_evidence": str(static_evidence), "static_evidence_sha256": sha256_file(static_evidence),
            "dataset_manifest": str(dataset_manifest), "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "critical_file_sha256": self.critical_hashes,
            "pinned_b500_teacher_action_reference": base.pinned_base3_reference_binding(),
            "parameter_update_authorized": True, "threshold_changes_allowed": False,
        }
        if self.main_launch_path.exists():
            self.assert_main_training_authorized(); return self.main_launch_path
        atomic_json(self.main_launch_path, payload, read_only=True)
        self.assert_main_training_authorized()
        return self.main_launch_path

    def _validate_original_launch_payload(self, payload: Mapping[str, Any]) -> None:
        if not (
            payload.get("kind") == "highstep_v13_parameter_update_authorization"
            and payload.get("workflow_id") == WORKFLOW_ID
            and payload.get("spec_sha256") == SPEC_SHA256
            and payload.get("r4_preregistration_sha256") == R4_PREREG_SHA256
            and payload.get("r5_preregistration_sha256") == R5_PREREG_SHA256
            and payload.get("parameter_update_authorized") is True
            and payload.get("threshold_changes_allowed") is False
        ):
            raise RuntimeError("v1.3 original launch authorization changed")
        for path_key, sha_key in (
            ("static_evidence", "static_evidence_sha256"),
            ("dataset_manifest", "dataset_manifest_sha256"),
        ):
            path = Path(str(payload[path_key])).resolve(strict=True)
            if not is_read_only(path) or sha256_file(path) != payload[sha_key]:
                raise RuntimeError(f"v1.3 original launch evidence changed: {path_key}")
        if payload.get("pinned_b500_teacher_action_reference") != base.pinned_base3_reference_binding():
            raise RuntimeError("v1.3 B500 action reference changed")

    def _repair_amendment_path(self) -> Path:
        digest = hashlib.sha256(
            json.dumps(self.critical_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return STATE_ROOT / f"v13_launch_repair_amendment_{digest}.json"

    def _r4_infeasible_boundary_evidence(self) -> dict[str, Any]:
        stage = STATE_ROOT / "r4_fit_stage_result.json"
        trigger = STATE_ROOT / "r5_trigger_manifest.json"
        if not stage.is_file() or not trigger.is_file() or not is_read_only(stage) or not is_read_only(trigger):
            raise RuntimeError("v1.3 post-R4 repair requires immutable R4/R5 boundary manifests")
        stage_payload = json.loads(stage.read_text())
        report = Path(str(stage_payload.get("report", ""))).resolve(strict=True)
        if not is_read_only(report) or sha256_file(report) != stage_payload.get("report_sha256"):
            raise RuntimeError("v1.3 R4 infeasibility report binding changed")
        report_payload = json.loads(report.read_text())
        candidates = report_payload.get("candidate_manifests") or []
        if not (
            report_payload.get("head_only_feasible") is False
            and report_payload.get("safe_candidate_count") == 0
            and report_payload.get("candidate_count") == 9
            and len(candidates) == 9
        ):
            raise RuntimeError("v1.3 post-R4 repair boundary is not fixed R4 infeasibility")
        candidate_bindings: list[dict[str, str]] = []
        for item in candidates:
            manifest = Path(str(item.get("manifest", ""))).resolve(strict=True)
            if not is_read_only(manifest) or sha256_file(manifest) != item.get("manifest_sha256"):
                raise RuntimeError("v1.3 unsafe R4 candidate manifest changed")
            candidate = json.loads(manifest.read_text())
            checkpoint = Path(str(candidate.get("checkpoint", ""))).resolve(strict=True)
            if sha256_file(checkpoint) != candidate.get("checkpoint_sha256"):
                raise RuntimeError("v1.3 unsafe R4 candidate checkpoint changed")
            candidate_bindings.append(
                {
                    "manifest": str(manifest),
                    "manifest_sha256": sha256_file(manifest),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
        trigger_payload = json.loads(trigger.read_text())
        if not (
            trigger_payload.get("training_authorized") is True
            and trigger_payload.get("r4_feasibility_sha256") == sha256_file(report)
            and trigger_payload.get("r5_preregistration_sha256") == R5_PREREG_SHA256
        ):
            raise RuntimeError("v1.3 R5 trigger binding changed")
        return {
            "r4_fit_stage_result": str(stage.resolve()),
            "r4_fit_stage_result_sha256": sha256_file(stage),
            "r4_feasibility_report": str(report),
            "r4_feasibility_report_sha256": sha256_file(report),
            "r5_trigger_manifest": str(trigger.resolve()),
            "r5_trigger_manifest_sha256": sha256_file(trigger),
            "candidate_bindings": candidate_bindings,
        }

    def ensure_launch_authorization(self, static_evidence: Path, dataset_manifest: Path) -> Path:
        if not self.main_launch_path.is_file() or not is_read_only(self.main_launch_path):
            raise RuntimeError("v1.3 immutable original launch authorization absent")
        original = json.loads(self.main_launch_path.read_text())
        self._validate_original_launch_payload(original)
        if original.get("critical_file_sha256") == self.critical_hashes:
            self.assert_main_training_authorized()
            return self.main_launch_path

        # A post-launch correctness amendment is legal only before any R4/R5
        # parameter artifact exists.  Failed logs are evidence and are kept.
        parameter_artifacts = [STATE_ROOT / "r4_fit_stage_result.json", STATE_ROOT / "r5_launch_manifest.json"]
        parameter_artifacts.extend(STATE_ROOT.glob("r4_fit_attempts/*/lambda_*/model_*.pt"))
        parameter_artifacts.extend(STATE_ROOT.glob("r5_*/**/model_*.pt"))
        existing_artifacts = [str(path) for path in parameter_artifacts if path.exists()]
        r5_checkpoints = list(STATE_ROOT.glob("r5_*/**/model_*.pt"))
        artifact_boundary = "pre_parameter"
        boundary_evidence: dict[str, Any] | None = None
        if existing_artifacts:
            if r5_checkpoints:
                raise RuntimeError(
                    "v1.3 code changed after R5 parameter artifacts existed; repair amendment forbidden: "
                    + ", ".join(map(str, r5_checkpoints))
                )
            boundary_evidence = self._r4_infeasible_boundary_evidence()
            artifact_boundary = "post_r4_infeasible_pre_r5"
        if Path(dataset_manifest).resolve() != Path(str(original["dataset_manifest"])).resolve():
            raise RuntimeError("v1.3 repair amendment attempted to change replay dataset")
        if sha256_file(dataset_manifest) != original["dataset_manifest_sha256"]:
            raise RuntimeError("v1.3 repair amendment dataset SHA changed")
        amendment = self._repair_amendment_path()
        payload = {
            "schema_version": 1,
            "kind": "highstep_v13_pre_parameter_correctness_repair_amendment",
            "workflow_id": WORKFLOW_ID,
            "created_at": now_text(),
            "original_launch_manifest": str(self.main_launch_path.resolve()),
            "original_launch_manifest_sha256": sha256_file(self.main_launch_path),
            "spec_sha256": SPEC_SHA256,
            "r4_preregistration_sha256": R4_PREREG_SHA256,
            "r5_preregistration_sha256": R5_PREREG_SHA256,
            "dataset_manifest": str(Path(dataset_manifest).resolve()),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "static_evidence": str(Path(static_evidence).resolve()),
            "static_evidence_sha256": sha256_file(static_evidence),
            "critical_file_sha256": self.critical_hashes,
            "repair_scope": (
                [
                    "cross-device B500 anchor reconstruction numerical integrity tolerance",
                    "deterministic child-log fail-fast classification",
                    "external fail-closed Codex recovery hook",
                ]
                if artifact_boundary == "pre_parameter"
                else [
                    "R5 direct-entry repository-root import binding",
                    "post-R4 pre-R5 immutable boundary authorization",
                ]
            ),
            "artifact_boundary": artifact_boundary,
            "r4_infeasible_boundary_evidence": boundary_evidence,
            "no_parameter_artifacts_at_amendment": artifact_boundary == "pre_parameter",
            "no_r5_parameter_artifacts_at_amendment": not r5_checkpoints,
            "threshold_changes_allowed": False,
            "parameter_update_authorized": True,
        }
        if amendment.exists():
            if not is_read_only(amendment):
                raise RuntimeError("v1.3 repair amendment is writable")
        else:
            atomic_json(amendment, payload, read_only=True)
        self.assert_main_training_authorized()
        return amendment

    def assert_main_training_authorized(self) -> dict[str, Any]:
        if not self.main_launch_path.is_file() or not is_read_only(self.main_launch_path):
            raise RuntimeError("v1.3 immutable launch authorization absent")
        payload = json.loads(self.main_launch_path.read_text())
        self._validate_original_launch_payload(payload)
        effective: Mapping[str, Any] = payload
        if payload.get("critical_file_sha256") != self.critical_hashes:
            amendment = self._repair_amendment_path()
            if not amendment.is_file() or not is_read_only(amendment):
                raise RuntimeError("v1.3 current-code repair amendment absent or writable")
            effective = json.loads(amendment.read_text())
            required = bool(
                effective.get("kind") == "highstep_v13_pre_parameter_correctness_repair_amendment"
                and effective.get("workflow_id") == WORKFLOW_ID
                and effective.get("original_launch_manifest_sha256") == sha256_file(self.main_launch_path)
                and effective.get("spec_sha256") == SPEC_SHA256
                and effective.get("r4_preregistration_sha256") == R4_PREREG_SHA256
                and effective.get("r5_preregistration_sha256") == R5_PREREG_SHA256
                and effective.get("dataset_manifest_sha256") == payload.get("dataset_manifest_sha256")
                and effective.get("critical_file_sha256") == self.critical_hashes
                and effective.get("threshold_changes_allowed") is False
                and effective.get("parameter_update_authorized") is True
            )
            boundary = effective.get("artifact_boundary", "pre_parameter")
            if boundary == "pre_parameter":
                required = required and effective.get("no_parameter_artifacts_at_amendment") is True
            elif boundary == "post_r4_infeasible_pre_r5":
                required = bool(
                    required
                    and effective.get("no_r5_parameter_artifacts_at_amendment") is True
                    and effective.get("r4_infeasible_boundary_evidence")
                    == self._r4_infeasible_boundary_evidence()
                )
            else:
                required = False
            if not required:
                raise RuntimeError("v1.3 current-code repair amendment changed")
            static_path = Path(str(effective["static_evidence"])).resolve(strict=True)
            if not is_read_only(static_path) or sha256_file(static_path) != effective["static_evidence_sha256"]:
                raise RuntimeError("v1.3 repair static evidence changed")
        self.assert_code_unchanged()
        return dict(effective)

    def fit_r4(self, dataset_manifest: Path) -> dict[str, Any]:
        self.assert_main_training_authorized()
        stage_result = STATE_ROOT / "r4_fit_stage_result.json"
        if stage_result.is_file():
            if not is_read_only(stage_result):
                raise RuntimeError("R4 fit stage result writable")
            record = json.loads(stage_result.read_text())
            report = Path(record["report"]).resolve(strict=True)
            if record.get("report_sha256") != sha256_file(report) or not is_read_only(report):
                raise RuntimeError("R4 feasibility report changed")
            payload = json.loads(report.read_text())
            for item in payload["candidate_manifests"]:
                if sha256_file(item["manifest"]) != item["manifest_sha256"]:
                    raise RuntimeError("R4 candidate manifest changed")
            self.update_state(r4_feasibility_report=str(report))
            return payload
        attempt_root = STATE_ROOT / "r4_fit_attempts" / time.strftime("%Y%m%d_%H%M%S")
        attempt_root.parent.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-u", str(FIT_SCRIPT), "--dataset-manifest", str(dataset_manifest), "--output-root", str(attempt_root)]
        log = attempt_root.with_suffix(".log")
        with log.open("wb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.wait_process(process, attempt_root, "r4_fixed_replay_fit", stall_seconds=600)
        report = attempt_root / "r4_offline_feasibility.json"
        if not report.is_file() or not is_read_only(report):
            raise RuntimeError("R4 fit did not produce immutable feasibility report")
        atomic_json(stage_result, {
            "schema_version": 1, "workflow_id": WORKFLOW_ID,
            "report": str(report), "report_sha256": sha256_file(report),
            "log": str(log), "log_sha256": sha256_file(log), "completed_at": now_text(),
        }, read_only=True)
        self.update_state(r4_feasibility_report=str(report))
        return json.loads(report.read_text())

    @staticmethod
    def _probe_gate(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, bool, str]:
        expected = {(11, "nominal"), (11, "left_offset"), (22, "right_offset")}
        identities = {(int(row["seed"]), str(row["scenario"])) for row in rows}
        if len(rows) != 3 or identities != expected or not all(row.get("valid") is True for row in rows):
            return False, False, "probe3 matrix/valid failed"
        by = {(int(row["seed"]), str(row["scenario"])): row for row in rows}
        protected = bool(
            by[(11, "nominal")].get("full_climb") is True
            and by[(11, "nominal")].get("rear_hold") is True
            and by[(11, "left_offset")].get("full_climb") is True
            and by[(11, "left_offset")].get("rear_hold") is True
            and sum(row.get("no_severe_inward") is True for row in rows) == 3
        )
        passed = bool(protected and by[(22, "right_offset")].get("front_top_support") is True)
        return passed, protected, "probe3 passed" if passed else ("protected success regression" if not protected else "right offset not improved")

    def run_probe3(self, checkpoint: Path, label: str, stage: str) -> tuple[list[dict[str, Any]], bool, bool, str]:
        self.assert_main_training_authorized(); self.audit_checkpoint(checkpoint, expected_stage=stage)
        output = STATE_ROOT / "evaluations" / f"{label}_probe3"
        terminal = output / "probe3_manifest.json"
        if terminal.is_file():
            payload = json.loads(terminal.read_text()); rows = payload["rows"]
            passed, protected, reason = self._probe_gate(rows)
            if payload.get("passed") is not passed or payload.get("protected_successes_preserved") is not protected:
                raise RuntimeError("stored v1.3 probe gate changed")
            return rows, passed, protected, reason
        output.mkdir(parents=True, exist_ok=True)
        scenarios = ((11, "nominal", "0.00", "0.0"), (11, "left_offset", "0.12", "4.0"), (22, "right_offset", "-0.12", "-4.0"))
        atomic_json(output / "probe3_plan.json", {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "label": label,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "gate": "valid3; nominal+left full/hold; right front support; no_severe3",
            "matrix": [f"seed{s}_{n}" for s, n, _, _ in scenarios],
        }, read_only=True)
        rows: list[dict[str, Any]] = []
        for seed, scenario, lateral, yaw in scenarios:
            self.require_no_gpu_job()
            attempt = 1 + len(list(output.glob(f"seed{seed}_{scenario}_attempt*.log")))
            log = output / f"seed{seed}_{scenario}_attempt{attempt}.log"
            command = base.probe_play_command(checkpoint, seed=seed, lateral=lateral, yaw=yaw)
            environment = os.environ.copy(); environment.pop("DISPLAY", None); environment.pop("XAUTHORITY", None)
            with log.open("wb") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.wait_process(process, output, f"{label}_probe_seed{seed}_{scenario}", stall_seconds=900)
            payload = self._single_eval_payload(log)
            validated = base.validate_probe_payload(payload, checkpoint)
            rows.append({"seed": seed, "scenario": scenario, **validated, "log": str(log), "log_sha256": sha256_file(log)})
        passed, protected, reason = self._probe_gate(rows)
        manifest = {
            "schema_version": 1, "kind": "highstep_v13_probe3", "stage": stage,
            "label": label, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "rows": rows, "passed": passed, "protected_successes_preserved": protected,
            "reason": reason, "completed_at": now_text(),
        }
        atomic_json(terminal, manifest, read_only=True)
        self.update_state(phase=f"{label}_probe3", checkpoint=str(checkpoint), probe3=manifest)
        return rows, passed, protected, reason

    @staticmethod
    def _configure_legacy_schema4_reader() -> Any:
        from tools import highstep_student_recovery_supervisor as legacy
        for name, value in {
            "ROOT": ROOT, "TASK": TASK, "WORKFLOW_ID": WORKFLOW_ID,
            "SPEC": SPEC, "SPEC_SHA256": SPEC_SHA256, "STATE_ROOT": STATE_ROOT,
            "STUDENT": B500, "STUDENT_SHA256": B500_SHA256,
            "TEACHER": TEACHER, "TEACHER_SHA256": TEACHER_SHA256,
            "PARENT_MANIFEST": PARENT_TEACHER_MANIFEST,
            "EXPERIMENT_ROOT": STATE_ROOT / "r4_candidates",
        }.items():
            setattr(legacy, name, value)
        return legacy

    def run_core9(self, checkpoint: Path, label: str, stage: str) -> tuple[Path, dict[str, Any]]:
        self.assert_main_training_authorized(); self.audit_checkpoint(checkpoint, expected_stage=stage); self.require_no_gpu_job()
        output = STATE_ROOT / "evaluations" / f"{label}_core9"
        if output.exists():
            legacy = self._configure_legacy_schema4_reader()
            try:
                launch = json.loads((output / "launch.json").read_text())
                if launch.get("checkpoint_sha256") != sha256_file(checkpoint) or launch.get("monitor_sha256") != sha256_file(SCHEMA4_MONITOR):
                    raise RuntimeError("stale v1.3 core9 launch")
                summary = dict(legacy.Supervisor.read_evaluation(self, output, expected_checkpoint=checkpoint))
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
                if base.core9_matrix_was_fully_executed(output, checkpoint):
                    raise RuntimeError(f"completed {label} core9 is invalid and terminal") from error
                output = STATE_ROOT / "evaluations" / f"{label}_core9_infra_{time.strftime('%Y%m%d_%H%M%S')}"
            else:
                self._record_evaluation(label, checkpoint, output, summary); return output, summary
        output.mkdir(parents=True)
        environment = os.environ.copy(); environment.update({
            "RECOVERY_SPEC_MODE": "1", "WORKFLOW_ID": WORKFLOW_ID,
            "LEDGER_FILE": str(STATE_ROOT / "v13_schema4_raw_ledger.jsonl"),
            "EVAL_CHECKPOINT_COUNT": "1", "EVAL_CHECKPOINT_PATH": str(checkpoint),
            "EVAL_CHECKPOINT_STRIDE": "1", "EVAL_ALLOW_LEGACY_SCHEDULE_FALLBACK": "0",
            "POLL_SECONDS": "5", "PLAY_STALL_TIMEOUT_SECONDS": "900",
            "PLAY_WATCHDOG_POLL_SECONDS": "10", "MIN_FREE_DISK_GIB": "15",
        })
        command = ["bash", str(SCHEMA4_MONITOR), "0", str(checkpoint.parent), "/dev/null", str(output), TASK, "student", str(PARENT_TEACHER_MANIFEST)]
        atomic_json(output / "launch.json", {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "label": label, "stage": stage,
            "command": command, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "monitor": str(SCHEMA4_MONITOR), "monitor_sha256": sha256_file(SCHEMA4_MONITOR), "created_at": now_text(),
        })
        with (output / "launcher.log").open("wb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.wait_process(process, output, f"{label}_core9", stall_seconds=1200)
        legacy = self._configure_legacy_schema4_reader()
        summary = dict(legacy.Supervisor.read_evaluation(self, output, expected_checkpoint=checkpoint))
        self._record_evaluation(label, checkpoint, output, summary)
        return output, summary

    def run_r5_stage(self, *, label: str, checkpoint: Path, load_mode: str, additional_epochs: int, total_epochs: int, dataset: Path) -> Path:
        self.assert_main_training_authorized()
        result_path = STATE_ROOT / "r5_stage_results" / f"{label}.json"
        if result_path.is_file():
            record = json.loads(result_path.read_text()); candidate = Path(record["checkpoint"]).resolve(strict=True)
            self.audit_checkpoint(candidate, expected_stage="R5")
            if record.get("checkpoint_sha256") != sha256_file(candidate) or record.get("effective_epochs") != total_epochs:
                raise RuntimeError("R5 stage result changed")
            return candidate
        trigger = STATE_ROOT / "r5_trigger_manifest.json"
        if not trigger.is_file() or not is_read_only(trigger):
            raise RuntimeError("R5 trigger manifest absent")
        output = STATE_ROOT / "r5_runs" / f"{label}_attempt_{time.strftime('%Y%m%d_%H%M%S')}"
        command = [str(PYTHON), "-u", str(R5_SCRIPT), "--dataset-manifest", str(dataset), "--checkpoint", str(checkpoint), "--load-mode", load_mode, "--additional-epochs", str(additional_epochs), "--output-dir", str(output)]
        log = output.with_suffix(".log"); log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.wait_process(process, log.parent, f"r5_train_{label}", stall_seconds=600)
        manifest = json.loads((output / "candidate_manifest.json").read_text())
        candidate = Path(manifest["checkpoint"]).resolve(strict=True)
        audit = self.audit_checkpoint(candidate, expected_stage="R5")
        if int(manifest["effective_epochs"]) != total_epochs:
            raise RuntimeError("R5 candidate epoch count changed")
        atomic_json(result_path, {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "label": label,
            "checkpoint": str(candidate), "checkpoint_sha256": sha256_file(candidate),
            "effective_epochs": total_epochs, "audit": audit, "log": str(log),
            "log_sha256": sha256_file(log), "completed_at": now_text(),
        }, read_only=True)
        return candidate

    def write_handoff(self, *, status: str, reason: str, best_summary: Mapping[str, Any] | None = None, candidate: bool = False, candidate_artifacts: Mapping[str, Any] | None = None) -> None:
        payload = {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "route": self.state.get("route"),
            "status": status, "stop_reason": reason, "phase": self.state.get("phase"),
            "checkpoint": self.state.get("checkpoint", str(B500)),
            "effective_updates": self.state.get("effective_updates", 0),
            "best_core9": dict(best_summary) if best_summary is not None else self.state.get("core9"),
            "spec_sha256": SPEC_SHA256, "r4_preregistration_sha256": R4_PREREG_SHA256,
            "r5_preregistration_sha256": R5_PREREG_SHA256, "candidate": candidate,
            "candidate_artifacts": dict(candidate_artifacts) if candidate_artifacts is not None else None,
            "user_action_required": status != "candidate_ready", "completed_at": now_text(),
        }
        atomic_json(self.handoff_path, payload)
        self.update_state(status=status, stop_reason=reason, handoff=str(self.handoff_path), active_pid=None)

    def _best(self, results: Sequence[tuple[Path, int, Mapping[str, Any], Path]]) -> tuple[Path, int, Mapping[str, Any], Path]:
        baseline = (B500, 0, {"valid_count": 9, "full_count": 6, "rear_hold_count": 6, "front_top_support_count": 8, "no_severe_inward_count": 9}, Path("/dev/null"))
        return max([baseline, *results], key=lambda item: behavior_rank(item[2], item[1]))

    def _finalize(self, checkpoint: Path, summary: Mapping[str, Any], evaluation: Path, stage: str) -> bool:
        self.candidate_stage = stage
        self.state["route"] = stage
        return self.finalize_behavior_candidate(checkpoint=checkpoint, summary=dict(summary), evaluation_dir=evaluation)

    def _run_r5(self, dataset: Path, r4_reason: str, results: list[tuple[Path, int, Mapping[str, Any], Path]]) -> None:
        self.update_state(route="R5", phase="r5_trigger")
        trigger = STATE_ROOT / "r5_trigger_manifest.json"
        if not trigger.exists():
            feasibility_path = Path(self.state["r4_feasibility_report"]).resolve(strict=True)
            atomic_json(trigger, {
                "schema_version": 1, "kind": "highstep_v13_r5_trigger", "workflow_id": WORKFLOW_ID,
                "reason": r4_reason, "r4_feasibility": str(feasibility_path),
                "r4_feasibility_sha256": sha256_file(feasibility_path),
                "dataset_manifest": str(dataset), "dataset_manifest_sha256": sha256_file(dataset),
                "r5_preregistration": str(R5_PREREG), "r5_preregistration_sha256": R5_PREREG_SHA256,
                "created_at": now_text(), "training_authorized": True,
            }, read_only=True)
        self.run_r5_stage(label="discarded_smoke1", checkpoint=B500, load_mode="weights_only", additional_epochs=1, total_epochs=1, dataset=dataset)
        source = B500; source_count = 0
        for label, additional, total in (("main1", 1, 1), ("main3", 2, 3), ("main5", 2, 5)):
            load_mode = "weights_only" if total == 1 else "full"
            checkpoint = self.run_r5_stage(label=label, checkpoint=source, load_mode=load_mode, additional_epochs=additional, total_epochs=total, dataset=dataset)
            _, passed, protected, reason = self.run_probe3(checkpoint, f"R5_{total}", "R5")
            if not protected:
                best = self._best(results); self.update_state(checkpoint=str(best[0]), core9=dict(best[2]))
                self.write_handoff(status="stopped_by_gate", reason=f"R5-{total} protected success regression: {reason}", best_summary=best[2]); return
            if passed:
                evaluation, summary = self.run_core9(checkpoint, f"R5_{total}", "R5")
                results.append((checkpoint, total, summary, evaluation))
                if final_gate(summary):
                    self._finalize(checkpoint, summary, evaluation, "R5"); return
            source, source_count = checkpoint, total
        best = self._best(results); self.update_state(checkpoint=str(best[0]), effective_updates=best[1], core9=dict(best[2]))
        self.write_handoff(status="stopped_by_gate", reason="v1.3 R5 absolute 5-epoch cap reached without final 8/9 gate", best_summary=best[2])

    def run(self) -> None:
        if self.handoff_path.is_file():
            handoff = json.loads(self.handoff_path.read_text())
            if handoff.get("workflow_id") == WORKFLOW_ID and handoff.get("status") in {"candidate_ready", "stopped_by_gate", "stopped_by_action_regression"}:
                self.update_state(status=handoff["status"], phase="terminal_handoff_preserved", stop_reason=handoff.get("stop_reason")); return
        self.preflight()
        static = self.run_static_tests()
        dataset = self.collect_dataset()
        if self.main_launch_path.is_file():
            self.ensure_launch_authorization(static, dataset)
        else:
            self.write_main_launch_manifest(static, dataset)
        feasibility = self.fit_r4(dataset)
        results: list[tuple[Path, int, Mapping[str, Any], Path]] = []
        safe = feasibility.get("safe_candidate_order") or []
        if safe:
            self.update_state(route="R4", phase="r4_behavior_candidates")
            for index, item in enumerate(safe[:3], start=1):
                checkpoint = Path(item["checkpoint"]).resolve(strict=True)
                _, passed, protected, reason = self.run_probe3(checkpoint, f"R4_candidate{index}", "R4")
                if not protected:
                    continue
                if passed:
                    evaluation, summary = self.run_core9(checkpoint, f"R4_candidate{index}", "R4")
                    results.append((checkpoint, index, summary, evaluation))
                    if final_gate(summary):
                        self._finalize(checkpoint, summary, evaluation, "R4"); return
        r4_reason = "R4 head-only offline infeasible" if not safe else "all at most three R4 behavior candidates failed final gate"
        self._run_r5(dataset, r4_reason, results)


def _is_transient(supervisor: V13Supervisor, error: BaseException) -> bool:
    phase = str(supervisor.state.get("phase", ""))
    if not phase.startswith(("trace_", "r4_", "R4_", "r5_", "R5_")):
        return False
    text = f"{type(error).__name__}: {error}".lower()
    if phase.startswith("trace_"):
        run_id = phase.removeprefix("trace_")
        logs = sorted(
            (STATE_ROOT / "trace_attempts").glob(f"{run_id}_*/trace.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if logs:
            with logs[0].open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 65536))
                text += "\n" + stream.read().decode("utf-8", errors="replace").lower()
    else:
        if phase == "r4_fixed_replay_fit":
            logs = list((STATE_ROOT / "r4_fit_attempts").glob("*.log"))
        else:
            logs = [
                path
                for path in STATE_ROOT.rglob("*.log")
                if path.name not in {"v13_supervisor_service.log"}
                and not path.name.startswith("v13_static_tests_")
            ]
        logs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if logs:
            with logs[0].open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 65536))
                text += "\n" + stream.read().decode("utf-8", errors="replace").lower()
    transient = (
        "cuda out of memory",
        "resource temporarily unavailable",
        "connection reset",
        "input/output error",
        "no space left",
    )
    if any(marker in text for marker in transient):
        return True
    permanent = (
        "sha mismatch",
        "binding",
        "contract",
        "frozen tensor",
        "preregistration",
        "not one of the fixed",
        "action reconstruction failed",
        "behavior differs",
        "gate",
        "scope",
        "traceback (most recent call last)",
    )
    if any(marker in text for marker in permanent):
        return False
    return any(marker in text for marker in ("return code", "no observable progress"))


def main() -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (STATE_ROOT / "v13_supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("v1.3 supervisor lock already held", file=sys.stderr); return 2
    supervisor = V13Supervisor()

    def handle_signal(signum: int, _frame: Any) -> None:
        supervisor.stop_active_process(); supervisor.write_handoff(status="interrupted", reason=f"supervisor_signal_{signum}"); raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal); signal.signal(signal.SIGTERM, handle_signal)
    try:
        supervisor.run(); return 0
    except KeyboardInterrupt:
        supervisor.stop_active_process(); return 130
    except BaseException as error:
        supervisor.stop_active_process()
        if _is_transient(supervisor, error):
            path = STATE_ROOT / "transient_retries.json"; retries = json.loads(path.read_text()) if path.is_file() else {}
            phase = str(supervisor.state.get("phase", "unknown")); attempt = int(retries.get(phase, 0)) + 1; retries[phase] = attempt; atomic_json(path, retries)
            if attempt <= 3:
                supervisor.update_state(status="transient_retry_pending", stop_reason=f"attempt {attempt}/3: {type(error).__name__}: {error}"); return 75
        supervisor.write_handoff(status="failed_closed", reason=f"{type(error).__name__}: {error}"); raise


if __name__ == "__main__":
    raise SystemExit(main())
