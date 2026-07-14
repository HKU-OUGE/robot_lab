#!/usr/bin/env python3
"""Spec-locked highstep Student recovery B supervisor (2026-07-12)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0"
WORKFLOW_ID = "highstep_student_recovery_20260712_v1"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA256 = "ae31b2da2d6c1a058a608da12c8ce500f98db28bab4dd32f4c0240b36ce9daff"
POLICY = ROOT / "tmp/highstep_student_recovery_policy_20260712.json"
STUDENT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_04-41-42_robust_student_distill_20260712_044124/model_900.pt"
)
STUDENT_SHA256 = "9bbd5b597d9c195ecf9afb141152b8f0749dc599a54868674b299107fb40a229"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA256 = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
PARENT_MANIFEST = ROOT / "tmp/highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json"
BASELINE_DIR = ROOT / "tmp/highstep_student_recovery_20260712/baseline_core9_retry2"
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_20260712"
CORRECTNESS_FIX_MANIFEST = STATE_ROOT / "correctness_fix_resume_manifest.json"
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student"
)
ALGORITHM_STATE_KEY = "robot_lab_algorithm_checkpoint_state"
CRITICAL_FILES = (
    SPEC,
    POLICY,
    CORRECTNESS_FIX_MANIFEST,
    ROOT / "tools/highstep_student_recovery_supervisor.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py",
    ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py",
    ROOT / "scripts/rsl_rl/base/train.py",
    ROOT / "scripts/rsl_rl/base/play.py",
    ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh",
    ROOT / "tests/test_highstep_student_recovery_stage_b.py",
    ROOT / "tests/test_vae_checkpoint_resume.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Supervisor:
    def __init__(self, baseline_dir: Path) -> None:
        self.baseline_dir = baseline_dir
        self.state_path = STATE_ROOT / "state.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.ledger_path = STATE_ROOT / "evaluation_ledger.jsonl"
        self.state: dict = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA256,
            "status": "starting",
            "phase": "preflight",
            "started_at": now_text(),
            "supervisor_pid": os.getpid(),
            "stage": "B",
            "student_source": str(STUDENT),
            "teacher_target": str(TEACHER),
            "effective_updates": 0,
            "checkpoint": str(STUDENT),
            "stop_reason": None,
        }
        self.evaluations: list[dict] = []
        self.critical_hashes: dict[str, str] = {}
        self.active_process: subprocess.Popen | None = None
        self.update_state()

    def update_state(self, **updates) -> None:
        self.state.update(updates)
        self.state["updated_at"] = now_text()
        self.state["last_supervisor_check_at"] = time.time()
        self.state["last_supervisor_pid"] = os.getpid()
        atomic_json(self.state_path, self.state)
        atomic_json(
            self.heartbeat_path,
            {
                "workflow_id": WORKFLOW_ID,
                "timestamp": self.state["updated_at"],
                "phase": self.state.get("phase"),
                "status": self.state.get("status"),
                "effective_updates": self.state.get("effective_updates"),
                "checkpoint": self.state.get("checkpoint"),
                "active_pid": self.state.get("active_pid"),
            },
        )

    def preflight(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        if sha256(SPEC) != SPEC_SHA256:
            raise RuntimeError("Formal recovery specification SHA256 changed")
        if sha256(STUDENT) != STUDENT_SHA256 or sha256(TEACHER) != TEACHER_SHA256:
            raise RuntimeError("Canonical Student/Teacher checkpoint SHA256 changed")
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        if policy.get("workflow_id") != WORKFLOW_ID or policy.get("spec_sha256") != SPEC_SHA256:
            raise RuntimeError("Recovery automation policy is not bound to the approved specification")

        if not CORRECTNESS_FIX_MANIFEST.is_file():
            raise RuntimeError("Explicit actor-prefix correctness-fix resume manifest is missing")
        correction = json.loads(CORRECTNESS_FIX_MANIFEST.read_text(encoding="utf-8"))
        archived_handoff = Path(str(correction.get("failed_attempt_handoff", "")))
        if not (
            correction.get("schema_version") == 1
            and correction.get("workflow_id") == WORKFLOW_ID
            and correction.get("spec_sha256") == SPEC_SHA256
            and correction.get("resume_authorized") is True
            and correction.get("resume_source_checkpoint") == str(STUDENT)
            and correction.get("resume_source_sha256") == STUDENT_SHA256
            and correction.get("invalid_failed_checkpoint_sha256")
            == "9a56654427b1687342a46633719c839048fac25ecb2b9f4869900beaf881ce41"
            and correction.get("fixed_vae_ppo_sha256")
            == sha256(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py")
            and correction.get("fixed_stage_b_test_sha256")
            == sha256(ROOT / "tests/test_highstep_student_recovery_stage_b.py")
            and archived_handoff.is_file()
            and correction.get("failed_attempt_handoff_sha256") == sha256(archived_handoff)
        ):
            raise RuntimeError("Actor-prefix correctness-fix resume manifest is invalid or stale")

        legacy_dir = STATE_ROOT / "superseded_legacy_state"
        legacy_dir.mkdir(exist_ok=True)
        for source in (
            ROOT / "tmp/highstep_goal_orchestrator/state.json",
            ROOT / "tmp/highstep_goal_orchestrator/handoff.json",
            ROOT / "tmp/highstep_automation_policy.json",
        ):
            if source.exists():
                shutil.copy2(source, legacy_dir / source.name)

        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True)
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
        self.critical_hashes = {str(path): sha256(path) for path in CRITICAL_FILES}
        manifest = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "created_at": now_text(),
            "spec_path": str(SPEC),
            "spec_sha256": sha256(SPEC),
            "git_head": head,
            "worktree_dirty": bool(status.strip()),
            "git_status_porcelain": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "critical_file_sha256": self.critical_hashes,
            "student_checkpoint": str(STUDENT),
            "student_sha256": sha256(STUDENT),
            "teacher_checkpoint": str(TEACHER),
            "teacher_sha256": sha256(TEACHER),
            "parent_teacher_manifest": str(PARENT_MANIFEST),
            "parent_teacher_manifest_sha256": sha256(PARENT_MANIFEST),
            "old_automation_service_must_remain_disabled": True,
            "legacy_state_snapshot": str(legacy_dir),
            "correctness_fix_resume_manifest": str(CORRECTNESS_FIX_MANIFEST),
            "correctness_fix_resume_manifest_sha256": sha256(CORRECTNESS_FIX_MANIFEST),
            "failed_attempt_handoff": str(archived_handoff),
            "failed_attempt_handoff_sha256": sha256(archived_handoff),
        }
        atomic_json(STATE_ROOT / "preflight_manifest.json", manifest)
        self.update_state(status="waiting", phase="baseline_core9", preflight_manifest=str(STATE_ROOT / "preflight_manifest.json"))

    def assert_code_unchanged(self) -> None:
        if not self.critical_hashes:
            raise RuntimeError("Critical code baseline was not captured")
        current = {str(path): sha256(path) for path in CRITICAL_FILES}
        if current != self.critical_hashes:
            changed = sorted(key for key in current if current.get(key) != self.critical_hashes.get(key))
            raise RuntimeError(f"Critical recovery code changed while the workflow was active: {changed}")

    @staticmethod
    def active_gpu_jobs() -> list[dict]:
        jobs = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if "scripts/rsl_rl/base/train.py" in argv or "scripts/rsl_rl/base/play.py" in argv:
                jobs.append({"pid": int(proc.name), "argv": argv.strip()})
        return jobs

    def require_no_gpu_job(self) -> None:
        jobs = self.active_gpu_jobs()
        if jobs:
            raise RuntimeError(f"Another train/play process is active: {jobs}")

    def wait_for_baseline(self) -> dict:
        manifest_path = self.baseline_dir / "evaluation_manifest.json"
        while True:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("completed_runs") == 9:
                    break
            matching_monitor = False
            for proc in Path("/proc").iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                except OSError:
                    continue
                if str(self.baseline_dir) in argv and "highstep_centerline_guard_monitor" in argv:
                    matching_monitor = True
                    break
            if not matching_monitor:
                rows = self.baseline_dir / "eval_runs.jsonl"
                count = len(rows.read_text().splitlines()) if rows.exists() else 0
                raise RuntimeError(f"Baseline monitor exited before completing core9 ({count}/9)")
            self.update_state(status="running", phase="baseline_core9", baseline_runs_completed=self._jsonl_count(self.baseline_dir / "eval_runs.jsonl"))
            time.sleep(10)
        summary = self.read_evaluation(self.baseline_dir, expected_checkpoint=STUDENT)
        if summary["valid_count"] < 9 or summary["pass_count"] < 2 or summary["full_count"] < 3 or summary["rear_hold_count"] < 3:
            raise RuntimeError(f"Baseline reproduction gate failed: {summary}")
        self.record_evaluation("baseline", 0, STUDENT, self.baseline_dir, summary)
        return summary

    @staticmethod
    def _jsonl_count(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())

    @staticmethod
    def tree_progress(path: Path) -> tuple[int, int]:
        size = 0
        newest = 0
        if path.exists():
            for item in path.rglob("*"):
                if item.is_file():
                    try:
                        stat = item.stat()
                    except OSError:
                        continue
                    size += stat.st_size
                    newest = max(newest, stat.st_mtime_ns)
        return size, newest

    def wait_process(self, process: subprocess.Popen, progress_root: Path, phase: str, stall_seconds: int) -> None:
        last_progress = self.tree_progress(progress_root)
        last_progress_at = time.monotonic()
        last_code_check = 0.0
        self.active_process = process
        while process.poll() is None:
            if time.monotonic() - last_code_check >= 30.0:
                self.assert_code_unchanged()
                last_code_check = time.monotonic()
            progress = self.tree_progress(progress_root)
            if progress != last_progress:
                last_progress = progress
                last_progress_at = time.monotonic()
            elif time.monotonic() - last_progress_at >= stall_seconds:
                self.update_state(status="stopping", stop_reason=f"{phase}_sustained_no_progress")
                self.stop_process_group(process)
                raise RuntimeError(f"{phase} made no observable progress for {stall_seconds}s")
            self.update_state(status="running", phase=phase, active_pid=process.pid, progress_bytes=progress[0])
            time.sleep(5)
        self.update_state(active_pid=None)
        self.active_process = None
        if process.returncode != 0:
            raise RuntimeError(f"{phase} exited with return code {process.returncode}")

    @staticmethod
    def stop_process_group(process: subprocess.Popen) -> None:
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

    def run_train(self, *, label: str, checkpoint: Path, load_mode: str, updates: int, num_envs: int) -> tuple[Path, Path]:
        self.assert_code_unchanged()
        self.require_no_gpu_job()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"student_recovery_{label}_{stamp}"
        launch_dir = STATE_ROOT / "training" / run_name
        launch_dir.mkdir(parents=True, exist_ok=False)
        log_path = launch_dir / "train.log"
        # model_900 has a SHA-bound runtime snapshot at schedule update 901.
        # B resets only its row optimizer/effective count, not the highstep
        # environment distribution reproduced by the baseline evaluation.
        schedule_mode = "preserve"
        command = [
            str(PYTHON), "-u", "scripts/rsl_rl/base/train.py",
            "--task", TASK,
            "--num_envs", str(num_envs),
            "--max_iterations", str(updates),
            "--seed", "42",
            "--headless",
            "--logger", "tensorboard",
            "--resume",
            "--checkpoint", str(checkpoint),
            "--highstep_resume_mode", "refine",
            "--highstep_checkpoint_load_mode", load_mode,
            "--highstep_schedule_resume_mode", schedule_mode,
            "--highstep_parent_teacher_manifest", str(PARENT_MANIFEST),
            "--run_name", run_name,
        ]
        atomic_json(
            launch_dir / "launch.json",
            {
                "workflow_id": WORKFLOW_ID,
                "label": label,
                "command": command,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_load_mode": load_mode,
                "requested_updates": updates,
                "num_envs": num_envs,
                "created_at": now_text(),
            },
        )
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.update_state(
                status="running",
                phase=f"train_{label}",
                active_pid=process.pid,
                train_log=str(log_path),
                source_checkpoint=str(checkpoint),
            )
            self.wait_process(process, launch_dir, f"train_{label}", stall_seconds=900)

        matches = sorted(EXPERIMENT_ROOT.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime_ns)
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one run directory for {run_name}, got {matches}")
        run_dir = matches[0]
        target_count = self.checkpoint_effective_count(checkpoint) + updates if load_mode == "full" else updates
        result_checkpoint = self.find_checkpoint_by_effective_count(run_dir, target_count)
        self.update_state(
            run_dir=str(run_dir),
            checkpoint=str(result_checkpoint),
            effective_updates=target_count,
        )
        return run_dir, result_checkpoint

    @staticmethod
    def checkpoint_payload(path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def checkpoint_effective_count(self, path: Path) -> int:
        if path.resolve() == STUDENT.resolve():
            return 0
        payload = self.checkpoint_payload(path)
        infos = payload.get("infos") or {}
        extra = infos.get(ALGORITHM_STATE_KEY) or {}
        recovery = extra.get("student_recovery") or {}
        count = recovery.get("effective_update_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"Checkpoint has no valid recovery effective count: {path}")
        return count

    def find_checkpoint_by_effective_count(self, run_dir: Path, expected: int) -> Path:
        matches = []
        for checkpoint in run_dir.glob("model_*.pt"):
            try:
                if self.checkpoint_effective_count(checkpoint) == expected:
                    matches.append(checkpoint)
            except Exception:
                continue
        if not matches:
            raise RuntimeError(f"No checkpoint with effective_update_count={expected} in {run_dir}")
        return max(matches, key=lambda path: path.stat().st_mtime_ns)

    def verify_row_only_delta(
        self,
        before: Path,
        after: Path,
        expected_count: int,
        *,
        require_both_row_tensors_changed: bool = False,
    ) -> dict:
        before_state = self.checkpoint_payload(before)["model_state_dict"]
        after_payload = self.checkpoint_payload(after)
        after_state = after_payload["model_state_dict"]
        if set(before_state) != set(after_state):
            raise RuntimeError("Student model state keys changed during stage B")
        changed = []
        changed_action_rows: list[int] = []
        for key in before_state:
            left = before_state[key]
            right = after_state[key]
            if key == "actor.6.weight":
                frozen_rows = [row for row in range(right.shape[0]) if row not in (2, 3)]
                if not torch.equal(left[frozen_rows], right[frozen_rows]):
                    raise RuntimeError("Frozen actor output weight row changed")
                if not torch.equal(left[2:4], right[2:4]):
                    changed.append("actor.6.weight[2:4]")
            elif key == "actor.6.bias":
                frozen_rows = [row for row in range(right.shape[0]) if row not in (2, 3)]
                if not torch.equal(left[frozen_rows], right[frozen_rows]):
                    raise RuntimeError("Frozen actor output bias row changed")
                if not torch.equal(left[2:4], right[2:4]):
                    changed.append("actor.6.bias[2:4]")
            elif not torch.equal(left, right):
                raise RuntimeError(f"Frozen policy tensor changed: {key}")
        if not changed:
            raise RuntimeError("Stage B produced no permitted rear-hip row change")
        for row in (2, 3):
            if (
                not torch.equal(before_state["actor.6.weight"][row], after_state["actor.6.weight"][row])
                or not torch.equal(before_state["actor.6.bias"][row], after_state["actor.6.bias"][row])
            ):
                changed_action_rows.append(row)
        if changed_action_rows != [2, 3]:
            raise RuntimeError(f"Both RL/RR hip output rows must change; got rows {changed_action_rows}")
        if require_both_row_tensors_changed and set(changed) != {
            "actor.6.weight[2:4]",
            "actor.6.bias[2:4]",
        }:
            raise RuntimeError(f"Smoke expected both rear hip row tensors to change, got {changed}")

        infos = after_payload.get("infos") or {}
        extra = infos.get(ALGORITHM_STATE_KEY)
        if not isinstance(extra, dict):
            raise RuntimeError("Recovery checkpoint lacks algorithm-owned state")
        recovery = extra.get("student_recovery") or {}
        binding = recovery.get("binding_manifest") or {}
        if recovery.get("effective_update_count") != expected_count:
            raise RuntimeError("Recovery checkpoint effective count mismatch")
        if binding.get("teacher_checkpoint") != str(TEACHER) or binding.get("teacher_sha256") != TEACHER_SHA256:
            raise RuntimeError("Recovery checkpoint Teacher binding mismatch")
        if binding.get("initial_student_checkpoint") != str(STUDENT) or binding.get("initial_student_sha256") != STUDENT_SHA256:
            raise RuntimeError("Recovery checkpoint Student binding mismatch")
        optimizer = after_payload.get("optimizer_state_dict") or {}
        groups = optimizer.get("param_groups") or []
        if len(groups) != 1 or len(groups[0].get("params", [])) != 2:
            raise RuntimeError("Recovery checkpoint optimizer does not own exactly two row tensors")
        for tensor in (recovery.get("rear_hip_weight"), recovery.get("rear_hip_bias")):
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise RuntimeError("Recovery checkpoint contains invalid rear hip row tensors")
        if not torch.equal(after_state["actor.6.weight"][2:4].cpu(), recovery["rear_hip_weight"].cpu()) or not torch.equal(
            after_state["actor.6.bias"][2:4].cpu(), recovery["rear_hip_bias"].cpu()
        ):
            raise RuntimeError("Checkpoint model_state and recovery row tensors disagree")
        return {
            "before": str(before),
            "after": str(after),
            "effective_update_count": expected_count,
            "changed_tensors": changed,
            "changed_action_rows": changed_action_rows,
            "frozen_tensor_violation_count": 0,
            "optimizer_parameter_tensor_count": 2,
            "teacher_binding_verified": True,
            "student_binding_verified": True,
            "finite": True,
        }

    @staticmethod
    def _nested_tensor_equal(left, right) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict) or isinstance(right, dict):
            return (
                isinstance(left, dict)
                and isinstance(right, dict)
                and set(left) == set(right)
                and all(Supervisor._nested_tensor_equal(left[key], right[key]) for key in left)
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return (
                isinstance(left, type(right))
                and len(left) == len(right)
                and all(Supervisor._nested_tensor_equal(a, b) for a, b in zip(left, right))
            )
        return left == right

    def _adam_checkpoint_audit(self, checkpoint: Path) -> tuple[dict, dict]:
        payload = self.checkpoint_payload(checkpoint)
        optimizer = payload.get("optimizer_state_dict")
        extra = ((payload.get("infos") or {}).get(ALGORITHM_STATE_KEY) or {})
        extra_optimizer = extra.get("vae_optimizer_state_dict")
        if not isinstance(optimizer, dict) or not isinstance(extra_optimizer, dict):
            raise RuntimeError(f"Smoke checkpoint is missing resumable Adam state: {checkpoint}")
        if not self._nested_tensor_equal(optimizer, extra_optimizer):
            raise RuntimeError("Runner and algorithm-owned Stage B Adam states disagree")
        groups = optimizer.get("param_groups") or []
        if len(groups) != 1 or len(groups[0].get("params", [])) != 2:
            raise RuntimeError("Stage B Adam must own exactly the two rear-hip row tensors")
        parameter_ids = list(groups[0]["params"])
        state = optimizer.get("state") or {}
        if set(state) != set(parameter_ids):
            raise RuntimeError("Stage B Adam moments are incomplete or include an unauthorized parameter")

        recovery = extra.get("student_recovery") or {}
        expected_shapes = [
            tuple(recovery.get("rear_hip_weight", torch.empty(0)).shape),
            tuple(recovery.get("rear_hip_bias", torch.empty(0)).shape),
        ]
        steps: list[float] = []
        for parameter_id, expected_shape in zip(parameter_ids, expected_shapes):
            moments = state[parameter_id]
            step = moments.get("step")
            step_value = float(step.item()) if isinstance(step, torch.Tensor) and step.numel() == 1 else float(step)
            if not math.isfinite(step_value) or step_value <= 0:
                raise RuntimeError("Stage B Adam has an invalid step counter")
            steps.append(step_value)
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = moments.get(name)
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape or not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Stage B Adam {name} is missing, non-finite, or has the wrong shape")
        return optimizer, {
            "checkpoint": str(checkpoint),
            "optimizer_parameter_tensor_count": 2,
            "adam_steps": steps,
            "adam_moments_finite": True,
            "runner_and_algorithm_optimizer_state_equal": True,
        }

    def smoke(self) -> Path:
        _, smoke4 = self.run_train(label="smoke4", checkpoint=STUDENT, load_mode="weights_only", updates=4, num_envs=256)
        audit4 = self.verify_row_only_delta(
            STUDENT, smoke4, 4, require_both_row_tensors_changed=True
        )
        _, smoke5 = self.run_train(label="smoke_resume1", checkpoint=smoke4, load_mode="full", updates=1, num_envs=256)
        audit5 = self.verify_row_only_delta(
            smoke4, smoke5, 5, require_both_row_tensors_changed=True
        )
        optimizer4, optimizer_audit4 = self._adam_checkpoint_audit(smoke4)
        optimizer5, optimizer_audit5 = self._adam_checkpoint_audit(smoke5)
        ids4 = optimizer4["param_groups"][0]["params"]
        ids5 = optimizer5["param_groups"][0]["params"]
        if len(ids4) != len(ids5):
            raise RuntimeError("Smoke full resume changed the Stage B Adam parameter scope")
        for before_id, after_id in zip(ids4, ids5):
            before_state = optimizer4["state"][before_id]
            after_state = optimizer5["state"][after_id]
            before_step = float(before_state["step"].item())
            after_step = float(after_state["step"].item())
            if not after_step > before_step:
                raise RuntimeError("Smoke full resume did not continue the existing Adam step counter")
            for name in ("exp_avg", "exp_avg_sq"):
                if tuple(before_state[name].shape) != tuple(after_state[name].shape):
                    raise RuntimeError(f"Smoke full resume changed Adam {name} shape")
        audit = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "passed": True,
            "teacher_sha256": TEACHER_SHA256,
            "student_sha256": STUDENT_SHA256,
            "initial_four_updates": audit4,
            "full_resume_fifth_update": audit5,
            "checkpoint_full_resume_verified": True,
            "adam_after_four_updates": optimizer_audit4,
            "adam_after_full_resume": optimizer_audit5,
            "adam_moments_and_step_continuity_verified": True,
            "completed_at": now_text(),
        }
        atomic_json(STATE_ROOT / "smoke_audit.json", audit)
        self.update_state(status="passed", phase="smoke", smoke_audit=str(STATE_ROOT / "smoke_audit.json"))
        return smoke5

    def evaluate(self, label: str, checkpoint: Path) -> tuple[Path, dict]:
        self.assert_code_unchanged()
        self.require_no_gpu_job()
        out_dir = STATE_ROOT / "evaluations" / label
        if out_dir.exists():
            out_dir = STATE_ROOT / "evaluations" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True)
        launcher_log = out_dir / "supervisor_launcher.log"
        environment = os.environ.copy()
        environment.update(
            {
                "RECOVERY_SPEC_MODE": "1",
                "WORKFLOW_ID": WORKFLOW_ID,
                "LEDGER_FILE": str(self.ledger_path),
                "EVAL_CHECKPOINT_COUNT": "1",
                "EVAL_CHECKPOINT_PATH": str(checkpoint),
                "EVAL_CHECKPOINT_STRIDE": "1",
                "EVAL_ALLOW_LEGACY_SCHEDULE_FALLBACK": "0",
                "POLL_SECONDS": "5",
                "PLAY_STALL_TIMEOUT_SECONDS": "600",
                "PLAY_WATCHDOG_POLL_SECONDS": "10",
                "MIN_FREE_DISK_GIB": "15",
            }
        )
        command = [
            "bash", "tmp/highstep_centerline_guard_monitor_20260709.sh",
            "0", str(checkpoint.parent), "/dev/null", str(out_dir), TASK, "student", str(PARENT_MANIFEST),
        ]
        self.update_state(eval_dir=str(out_dir), checkpoint=str(checkpoint))
        with launcher_log.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.wait_process(process, out_dir, f"eval_{label}", stall_seconds=900)
        summary = self.read_evaluation(out_dir, expected_checkpoint=checkpoint)
        self.record_evaluation(label, self.checkpoint_effective_count(checkpoint), checkpoint, out_dir, summary)
        return out_dir, summary

    @staticmethod
    def finite(value, default: float) -> float:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else default

    def read_evaluation(self, out_dir: Path, *, expected_checkpoint: Path | None = None) -> dict:
        manifest_path = out_dir / "evaluation_manifest.json"
        rows_path = out_dir / "eval_runs.jsonl"
        if not manifest_path.exists() or not rows_path.exists():
            raise RuntimeError(f"Evaluation artifacts are incomplete: {out_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not (
            manifest.get("evaluation_complete") is True
            and manifest.get("matrix_complete") is True
            and manifest.get("parent_teacher_manifest_valid") is True
            and manifest.get("student_source_lineage_valid") is True
            and manifest.get("task") == TASK
            and manifest.get("role") == "student"
            and manifest.get("workflow_id") == WORKFLOW_ID
            and manifest.get("evaluation_profile") == "core9"
        ):
            raise RuntimeError(
                "Evaluation manifest does not prove complete core9 parent/Student lineage: "
                f"{manifest_path}"
            )
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        checkpoint_paths = {row.get("checkpoint") for row in rows if isinstance(row.get("checkpoint"), str)}
        if len(checkpoint_paths) != 1:
            raise RuntimeError(f"Evaluation rows do not bind exactly one checkpoint: {checkpoint_paths}")
        checkpoint_path = Path(next(iter(checkpoint_paths))).resolve()
        if expected_checkpoint is not None and checkpoint_path != expected_checkpoint.resolve():
            raise RuntimeError(
                f"Evaluation checkpoint mismatch: {checkpoint_path} != {expected_checkpoint.resolve()}"
            )
        checkpoint_digest = sha256(checkpoint_path)
        expected_joint_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
            "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
        ]

        expected_code_sha = {
            "actions.py": sha256(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"),
            "curriculums.py": sha256(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py"),
            "highstep_env_cfg.py": sha256(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py"),
            "highstep_schedule.py": sha256(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"),
            "play.py": sha256(ROOT / "scripts/rsl_rl/base/play.py"),
            "rewards.py": sha256(ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py"),
        }

        def valid_row(row: dict) -> bool:
            item = row.get("eval")
            if not isinstance(item, dict):
                return False
            samples = self.finite(item.get("samples"), -1.0)
            geometry_samples = self.finite(item.get("geometry_samples"), -1.0)
            action_samples = self.finite(item.get("action_samples"), -1.0)
            loop_steps = self.finite(item.get("loop_steps"), -1.0)
            if item.get("terminated_early") is False:
                termination_valid = geometry_samples == samples and loop_steps == 600.0
            elif item.get("terminated_early") is True:
                termination_valid = (
                    geometry_samples + 1.0 == samples
                    and self.finite(item.get("termination_step"), -1.0) + 1.0 == loop_steps
                )
            else:
                termination_valid = False
            return bool(
                row.get("return_code") == 0
                and row.get("task") == TASK
                and row.get("role") == "student"
                and row.get("randomize") is False
                and row.get("action_delay_steps") == 0
                and item.get("schema_version") == 7
                and item.get("checkpoint_sha256") == checkpoint_digest
                and item.get("evaluation_code_sha256") == expected_code_sha
                and item.get("runtime_snapshot_checkpoint_sha256_verified") is True
                and item.get("reset_context_valid") is True
                and item.get("reset_valid") is True
                and item.get("initial_geometry_valid") is True
                and item.get("initial_no_top_contact") is True
                and item.get("initial_all_feet_outside_platform") is True
                and samples > 0
                and loop_steps == samples
                and action_samples == samples
                and self.finite(item.get("target_limit_sample_steps"), -1.0) == samples
                and termination_valid
                and item.get("requested_play_max_steps") == 600
                and item.get("eval_action_delay_requested") == 0
                and item.get("eval_action_delay_runtime_match") is True
                and item.get("keep_play_randomization") is False
                and item.get("schedule_valid") is True
                and item.get("schedule_runtime_match") is True
                and item.get("schedule_clock_runtime_match") is True
                and item.get("schedule_frozen_for_evaluation") is True
                and item.get("target_action_order_valid") is True
                and item.get("target_action_joint_names") == expected_joint_order
                and item.get("target_limit_contract_valid") is True
                and item.get("target_limit_action_input_valid") is True
                and item.get("target_limit_invalid_action_steps") == 0
            )

        valid_rows = [row for row in rows if valid_row(row)]
        evals = [row["eval"] for row in valid_rows]
        if len(rows) != 9 or len(valid_rows) != 9:
            raise RuntimeError(
                f"core9 is not complete and valid: raw={len(rows)} valid={len(valid_rows)} dir={out_dir}"
            )
        expected_matrix = {
            (seed, scenario)
            for seed in (11, 22, 33)
            for scenario in ("nominal", "left_offset", "right_offset")
        }
        actual_matrix = {(row.get("seed"), row.get("scenario")) for row in valid_rows}
        if actual_matrix != expected_matrix:
            raise RuntimeError(f"core9 seed/scenario matrix mismatch: {actual_matrix}")

        def row_pass(item: dict) -> bool:
            return bool(
                item.get("rear_on_platform_hold_success") is True
                and item.get("rear_clear_reached") is True
                and item.get("terminated_early") is False
                and self.finite(item.get("critical_rear_width_q05"), -1.0) >= 0.18
                and self.finite(item.get("critical_rear_min_abs_y_q05"), -1.0) >= 0.04
                and self.finite(item.get("critical_width_violation_rate"), 1.0) <= 0.10
                and self.finite(item.get("critical_center_violation_rate"), 1.0) <= 0.10
                and self.finite(item.get("rear_on_platform_hold_max_consecutive"), 0.0) >= 25
            )

        def values(key: str) -> list[float]:
            return [
                float(item[key])
                for item in evals
                if isinstance(item.get(key), (int, float))
                and not isinstance(item.get(key), bool)
                and math.isfinite(float(item[key]))
            ]

        no_severe = sum(
            1
            for item in evals
            if self.finite(item.get("critical_rear_min_abs_y_q05"), -1.0) >= 0.04
            and self.finite(item.get("critical_rear_width_q05"), -1.0) >= 0.18
            and self.finite(item.get("critical_center_violation_rate"), 1.0) <= 0.10
            and self.finite(item.get("critical_width_violation_rate"), 1.0) <= 0.10
        )
        full = sum(item.get("full_climb_success") is True for item in evals)
        rear_hold = sum(item.get("rear_on_platform_hold_success") is True for item in evals)
        front_top_support = sum(item.get("front_top_support_reached") is True for item in evals)
        first_rear_top = sum(item.get("first_rear_top_step") is not None for item in evals)
        remaining_failures = [item for item in evals if item.get("rear_on_platform_hold_success") is not True]
        stuck_second = sum(
            item.get("front_top_support_reached") is True
            and item.get("first_rear_top_step") is not None
            and item.get("second_rear_top_step") is None
            for item in remaining_failures
        )
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest,
            "valid_count": len(valid_rows),
            "pass_count": sum(row_pass(item) for item in evals),
            "full_count": full,
            "rear_hold_count": rear_hold,
            "teacher_reference_full_and_hold_count": 9,
            # This behavioral comparison is necessary but is not allowed to
            # stand in for the specification's independent Teacher-action
            # regression audit.
            "behavior_relative_teacher_full_hold_within_one": full >= 8 and rear_hold >= 8,
            "teacher_action_regression_audit_passed": False,
            "no_severe_inward_count": no_severe,
            "front_top_support_count": front_top_support,
            "front_top_support_rate": front_top_support / 9.0,
            "first_rear_top_count": first_rear_top,
            "first_rear_top_rate": first_rear_top / 9.0,
            "rear_abs_y_worst": min(values("critical_rear_min_abs_y_q05"), default=-1.0),
            "rear_width_worst": min(values("critical_rear_width_q05"), default=-1.0),
            "center_violation_rate_avg": sum(values("critical_center_violation_rate")) / max(1, len(values("critical_center_violation_rate"))),
            "width_violation_rate_avg": sum(values("critical_width_violation_rate")) / max(1, len(values("critical_width_violation_rate"))),
            "dwell_average": sum(values("rear_edge_dwell_max_consecutive")) / max(1, len(values("rear_edge_dwell_max_consecutive"))),
            "dwell_worst": max(values("rear_edge_dwell_max_consecutive"), default=9999.0),
            "remaining_failure_count": len(remaining_failures),
            "stuck_at_second_rear_count": stuck_second,
            "raw_runs": len(rows),
            "evaluation_manifest": str(manifest_path),
            "legacy_aggregate_decision": manifest.get("decision"),
        }

    def record_evaluation(self, label: str, effective_updates: int, checkpoint: Path, out_dir: Path, summary: dict) -> None:
        record = {
            "workflow_id": WORKFLOW_ID,
            "phase": label,
            "effective_updates": effective_updates,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "out_dir": str(out_dir),
            "summary": summary,
            "created_at": now_text(),
        }
        self.evaluations.append(record)
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.update_state(phase=label, effective_updates=effective_updates, checkpoint=str(checkpoint), core9=summary)

    @staticmethod
    def behavior_final_gate(summary: dict) -> bool:
        return (
            summary["valid_count"] == 9
            and summary["full_count"] >= 8
            and summary["rear_hold_count"] >= 8
            and summary["no_severe_inward_count"] >= 8
        )

    @staticmethod
    def final_gate(summary: dict) -> bool:
        return (
            Supervisor.behavior_final_gate(summary)
            and summary.get("teacher_action_regression_audit_passed") is True
        )

    def finish_or_hold_behavior_candidate(
        self,
        *,
        reason: str,
        checkpoint: Path,
        summary: dict,
        evaluation_dir: Path,
    ) -> bool:
        """Finish a fully audited candidate or stop before an unaudited export."""
        if not self.behavior_final_gate(summary):
            return False
        if self.final_gate(summary):
            self.finish_candidate(reason, checkpoint, summary, evaluation_dir)
        else:
            self.finish(
                status="awaiting_teacher_action_audit",
                reason=(
                    f"{reason}; independent Student-vs-Teacher action regression evidence "
                    "is still required before candidate export"
                ),
                best_checkpoint=checkpoint,
                best_summary=summary,
            )
        return True

    @staticmethod
    def rank(summary: dict) -> tuple:
        return (
            summary["full_count"],
            summary["rear_hold_count"],
            summary["no_severe_inward_count"],
            summary["front_top_support_count"],
            summary["first_rear_top_count"],
            summary["pass_count"],
            summary["rear_abs_y_worst"],
            summary["rear_width_worst"],
            -summary["dwell_average"],
            -summary["dwell_worst"],
        )

    @staticmethod
    def front_half_not_regressed(reference: dict, current: dict) -> bool:
        return (
            current["front_top_support_count"] >= reference["front_top_support_count"]
            and current["first_rear_top_count"] >= reference["first_rear_top_count"]
        )

    @staticmethod
    def geometry_not_materially_worse(reference: dict, current: dict) -> bool:
        return (
            current["rear_abs_y_worst"] + 0.005 >= reference["rear_abs_y_worst"]
            and current["rear_width_worst"] + 0.010 >= reference["rear_width_worst"]
            and current["center_violation_rate_avg"] <= reference["center_violation_rate_avg"] + 0.05
            and current["width_violation_rate_avg"] <= reference["width_violation_rate_avg"] + 0.05
        )

    @staticmethod
    def gate_300(baseline: dict, current: dict) -> tuple[bool, str]:
        if current["full_count"] < baseline["full_count"] or current["rear_hold_count"] < baseline["rear_hold_count"]:
            return False, "B300 full-climb/rear-hold count decreased"
        if not Supervisor.front_half_not_regressed(baseline, current):
            return False, "B300 pre-climb front/first-rear progression regressed"
        if not Supervisor.geometry_not_materially_worse(baseline, current):
            return False, "B300 rear inward-collapse metrics worsened"
        return True, "B300 has no material behavior regression"

    @staticmethod
    def gate_500_to_1000(baseline: dict, current: dict) -> tuple[bool, str]:
        if current["full_count"] < baseline["full_count"] or current["rear_hold_count"] < baseline["rear_hold_count"]:
            return False, "B500 full-climb/rear-hold count regressed"
        if not Supervisor.front_half_not_regressed(baseline, current):
            return False, "B500 pre-climb front/first-rear progression regressed"
        if not Supervisor.geometry_not_materially_worse(baseline, current):
            return False, "B500 rear inward-collapse geometry regressed"
        if current["full_count"] >= baseline["full_count"] + 1 or current["rear_hold_count"] >= baseline["rear_hold_count"] + 1:
            return True, "B500 added at least one full-climb/rear-hold success"
        geometric = (
            current["rear_abs_y_worst"] >= 0.04
            and current["rear_width_worst"] >= 0.18
            and current["dwell_average"] <= baseline["dwell_average"] - 30
            and current["dwell_worst"] <= baseline["dwell_worst"] - 60
        )
        return (geometric, "B500 geometric/dwell alternative gate passed" if geometric else "B500 behavior gate did not justify 1000")

    @staticmethod
    def gate_1000_to_2000(previous: dict, current: dict) -> tuple[bool, str]:
        if current["full_count"] < previous["full_count"] or current["rear_hold_count"] < previous["rear_hold_count"]:
            return False, "B1000 full-climb/rear-hold count regressed"
        if not Supervisor.front_half_not_regressed(previous, current):
            return False, "B1000 pre-climb front/first-rear progression regressed"
        if current["no_severe_inward_count"] < previous["no_severe_inward_count"]:
            return False, "B1000 severe rear inward-collapse count regressed"
        if not Supervisor.geometry_not_materially_worse(previous, current):
            return False, "B1000 rear inward-collapse geometry regressed"
        improvement = (
            current["full_count"] >= previous["full_count"] + 1
            or current["rear_hold_count"] >= previous["rear_hold_count"] + 1
            or current["first_rear_top_count"] >= previous["first_rear_top_count"] + 1
            or current["no_severe_inward_count"] >= previous["no_severe_inward_count"] + 1
            or (
                current["rear_abs_y_worst"] >= previous["rear_abs_y_worst"] + 0.005
                or current["rear_width_worst"] >= previous["rear_width_worst"] + 0.010
                or (
                    current["dwell_average"] <= previous["dwell_average"] - 30
                    and current["dwell_worst"] <= previous["dwell_worst"] - 60
                )
            )
        )
        return (improvement, "B1000 produced another behavior improvement" if improvement else "B1000 produced no further behavior improvement")

    @staticmethod
    def c_gate(baseline: dict, best: dict) -> tuple[bool, str]:
        failures = best["remaining_failure_count"]
        passed = (
            best["rear_abs_y_worst"] >= 0.04
            and best["rear_width_worst"] >= 0.18
            and best["full_count"] >= baseline["full_count"]
            and failures > 0
            and best["stuck_at_second_rear_count"] > failures / 2
        )
        return (passed, "B-to-C behavior classification passed" if passed else "B-to-C behavior classification failed")

    def export_and_record_candidate(self, checkpoint: Path, evaluation_dir: Path) -> dict:
        """Export the selected Student and retain all nine labeled core9 videos."""
        self.require_no_gpu_job()
        candidate_dir = STATE_ROOT / "candidate"
        videos_dir = candidate_dir / "videos"
        logs_dir = candidate_dir / "video_logs"
        videos_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        scenarios = (
            ("nominal", "0.00", "0.0"),
            ("left_offset", "0.12", "4.0"),
            ("right_offset", "-0.12", "-4.0"),
        )
        exported_dir = checkpoint.parent / "exported"
        exported_student = exported_dir / "policy_student.pt"
        previous_export_mtime = exported_student.stat().st_mtime_ns if exported_student.exists() else None
        saved_videos = []
        video_records: list[dict] = []
        export_record: dict | None = None
        export_requested = True
        for seed in (11, 22, 33):
            for scenario, lateral, yaw in scenarios:
                label = f"seed{seed}_{scenario}"
                log_path = logs_dir / f"{label}.log"
                source_video_dir = checkpoint.parent / "videos/play"
                before = {
                    path.resolve(): path.stat().st_mtime_ns
                    for path in source_video_dir.glob("*.mp4")
                } if source_video_dir.exists() else {}
                command = [
                    str(PYTHON), "scripts/rsl_rl/base/play.py",
                    "--task", TASK,
                    "--num_envs", "1",
                    "--headless",
                    "--enable_cameras",
                    "--video",
                    "--video_length", "600",
                    "--seed", str(seed),
                    "--checkpoint", str(checkpoint),
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
                    "--rear_width_metric_interval", "60",
                    "--play_max_steps", "600",
                    "--eval_action_delay_steps", "0",
                ]
                if not export_requested:
                    command.append("--skip_policy_export")
                launch_time_ns = time.time_ns()
                with log_path.open("wb") as log_file:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.wait_process(process, logs_dir, f"candidate_video_{label}", stall_seconds=900)
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                payloads = []
                for line in log_text.splitlines():
                    if "[HIGHSTEP_EVAL_JSON]" not in line:
                        continue
                    try:
                        payloads.append(json.loads(line.split("[HIGHSTEP_EVAL_JSON]", 1)[1].strip()))
                    except json.JSONDecodeError as error:
                        raise RuntimeError(f"Candidate video evaluation JSON is malformed: {label}") from error
                if len(payloads) != 1 or not isinstance(payloads[0], dict):
                    raise RuntimeError(f"Candidate video run has no evaluation payload: {label}")
                evaluation_payload = payloads[0]
                if export_requested:
                    if not exported_student.is_file() or exported_student.stat().st_size <= 0:
                        raise RuntimeError("This candidate run did not create policy_student.pt")
                    export_mtime = exported_student.stat().st_mtime_ns
                    if export_mtime < launch_time_ns or (
                        previous_export_mtime is not None and export_mtime <= previous_export_mtime
                    ):
                        raise RuntimeError("Candidate policy_student.pt was not freshly exported by this run")
                    if "Successfully exported COMPLETE Student Policy" not in log_text:
                        raise RuntimeError("play.py did not confirm this run's complete Student export")
                    try:
                        jit_policy = torch.jit.load(str(exported_student), map_location="cpu")
                        jit_policy.eval()
                        with torch.inference_mode():
                            jit_output = jit_policy(torch.zeros((1, 570), dtype=torch.float32))
                    except Exception as error:
                        raise RuntimeError("Fresh policy_student.pt cannot execute on the locked 570-D observation") from error
                    if not isinstance(jit_output, torch.Tensor) or tuple(jit_output.shape) != (1, 16) or not torch.isfinite(jit_output).all():
                        raise RuntimeError("Fresh policy_student.pt produced an invalid 16-action output")
                    preserved_export = candidate_dir / "policy_student.pt"
                    shutil.copy2(exported_student, preserved_export)
                    export_record = {
                        "source": str(exported_student),
                        "preserved_copy": str(preserved_export),
                        "sha256": sha256(preserved_export),
                        "size_bytes": preserved_export.stat().st_size,
                        "fresh_export_mtime_ns": export_mtime,
                        "jit_input_shape": [1, 570],
                        "jit_output_shape": [1, 16],
                        "jit_output_finite": True,
                    }
                    export_requested = False
                candidates = []
                for path in source_video_dir.glob("*.mp4"):
                    resolved = path.resolve()
                    if resolved not in before or path.stat().st_mtime_ns > before[resolved]:
                        candidates.append(path)
                if not candidates:
                    raise RuntimeError(f"Candidate video file was not created: {label}")
                source_video = max(candidates, key=lambda path: path.stat().st_mtime_ns)
                destination = videos_dir / f"{label}.mp4"
                shutil.copy2(source_video, destination)
                if destination.stat().st_size <= 0:
                    raise RuntimeError(f"Candidate video is empty: {label}")
                saved_videos.append(str(destination))
                video_records.append(
                    {
                        "label": label,
                        "seed": seed,
                        "scenario": scenario,
                        "video": str(destination),
                        "video_sha256": sha256(destination),
                        "evaluation": evaluation_payload,
                        "full_climb_success": evaluation_payload.get("full_climb_success") is True,
                        "rear_on_platform_hold_success": evaluation_payload.get("rear_on_platform_hold_success") is True,
                    }
                )

        expected_labels = {
            f"seed{seed}_{scenario}"
            for seed in (11, 22, 33)
            for scenario in ("nominal", "left_offset", "right_offset")
        }
        if export_record is None or len(video_records) != 9 or {item["label"] for item in video_records} != expected_labels:
            raise RuntimeError("Candidate artifacts do not contain one freshly labeled video for every core9 run")
        evaluation_copy = candidate_dir / "evaluation_manifest.json"
        shutil.copy2(evaluation_dir / "evaluation_manifest.json", evaluation_copy)
        evaluation_rows_copy = candidate_dir / "eval_runs.jsonl"
        shutil.copy2(evaluation_dir / "eval_runs.jsonl", evaluation_rows_copy)
        atomic_json(candidate_dir / "video_run_manifest.json", {"runs": video_records})

        concat_list = candidate_dir / "video_concat.txt"
        concat_list.write_text(
            "".join(f"file '{Path(path).as_posix()}'\n" for path in saved_videos),
            encoding="utf-8",
        )
        compilation = candidate_dir / "highstep_student_repeated_core9.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to build the candidate core9 compilation")
        completed = subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(compilation)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            completed = subprocess.run(
                [
                    ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-an", str(compilation),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if completed.returncode != 0 or not compilation.is_file() or compilation.stat().st_size <= 0:
            raise RuntimeError("ffmpeg failed to create the required core9 compilation")
        artifacts = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "teacher_checkpoint": str(TEACHER),
            "teacher_sha256": TEACHER_SHA256,
            "spec_sha256": SPEC_SHA256,
            "critical_file_sha256": self.critical_hashes,
            "student_policy_export": export_record,
            "evaluation_manifest": str(evaluation_copy),
            "evaluation_rows": str(evaluation_rows_copy),
            "core9_videos": saved_videos,
            "video_run_manifest": str(candidate_dir / "video_run_manifest.json"),
            "repeated_action_compilation": str(compilation),
            "repeated_action_compilation_sha256": sha256(compilation),
        }
        atomic_json(candidate_dir / "candidate_artifacts.json", artifacts)
        return artifacts

    def finish(
        self,
        *,
        status: str,
        reason: str,
        best_checkpoint: Path,
        best_summary: dict,
        candidate: bool = False,
        candidate_artifacts: dict | None = None,
    ) -> None:
        self.assert_code_unchanged()
        handoff = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA256,
            "status": status,
            "stage": "B",
            "stop_reason": reason,
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_sha256": sha256(best_checkpoint),
            "best_core9": best_summary,
            "student_source": str(STUDENT),
            "student_source_sha256": STUDENT_SHA256,
            "teacher_target": str(TEACHER),
            "teacher_target_sha256": TEACHER_SHA256,
            "trainable_action_rows": [2, 3],
            "frozen_scope": "all Student tensors except RL/RR hip final output rows",
            "candidate": candidate,
            "candidate_artifacts": candidate_artifacts,
            "evaluations": self.evaluations,
            "completed_at": now_text(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update_state(
            status=status,
            phase="candidate_ready" if candidate else "stopped",
            stop_reason=reason,
            checkpoint=str(best_checkpoint),
            core9=best_summary,
            handoff=str(self.handoff_path),
        )

    def finish_candidate(self, reason: str, checkpoint: Path, summary: dict, evaluation_dir: Path) -> None:
        artifacts = self.export_and_record_candidate(checkpoint, evaluation_dir)
        self.finish(
            status="candidate_ready",
            reason=reason,
            best_checkpoint=checkpoint,
            best_summary=summary,
            candidate=True,
            candidate_artifacts=artifacts,
        )

    def run(self) -> None:
        self.preflight()
        baseline = self.wait_for_baseline()
        self.require_no_gpu_job()
        self.update_state(status="running", phase="static_and_cpu_tests")
        test_log = STATE_ROOT / "static_tests.log"
        with test_log.open("wb") as stream:
            completed = subprocess.run(
                [str(PYTHON), "-m", "pytest", "-q", "tests/test_highstep_student_recovery_stage_b.py", "tests/test_vae_checkpoint_resume.py"],
                cwd=ROOT,
                env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError("Stage B static/CPU test gate failed")

        self.smoke()
        self.require_no_gpu_job()
        # Baseline remains a selectable behavior checkpoint.  A trained
        # checkpoint never becomes "best" merely because it is newer.
        b_results: list[tuple[Path, dict]] = [(STUDENT, baseline)]
        evaluation_dirs: dict[str, Path] = {str(STUDENT): self.baseline_dir}

        _, checkpoint = self.run_train(label="B300", checkpoint=STUDENT, load_mode="weights_only", updates=300, num_envs=4096)
        self.verify_row_only_delta(STUDENT, checkpoint, 300)
        eval300_dir, summary300 = self.evaluate("B300", checkpoint)
        b_results.append((checkpoint, summary300))
        evaluation_dirs[str(checkpoint)] = eval300_dir
        if self.finish_or_hold_behavior_candidate(
            reason="B300 reached final 8/9 behavior gate",
            checkpoint=checkpoint,
            summary=summary300,
            evaluation_dir=eval300_dir,
        ):
            return
        allowed, reason = self.gate_300(baseline, summary300)
        if not allowed:
            best_checkpoint, best_summary = max(b_results, key=lambda item: self.rank(item[1]))
            self.finish(status="stopped_by_gate", reason=reason, best_checkpoint=best_checkpoint, best_summary=best_summary)
            return

        _, checkpoint500 = self.run_train(label="B500", checkpoint=checkpoint, load_mode="full", updates=200, num_envs=4096)
        self.verify_row_only_delta(checkpoint, checkpoint500, 500)
        eval500_dir, summary500 = self.evaluate("B500", checkpoint500)
        b_results.append((checkpoint500, summary500))
        evaluation_dirs[str(checkpoint500)] = eval500_dir
        if self.finish_or_hold_behavior_candidate(
            reason="B500 reached final 8/9 behavior gate",
            checkpoint=checkpoint500,
            summary=summary500,
            evaluation_dir=eval500_dir,
        ):
            return
        allowed, reason = self.gate_500_to_1000(baseline, summary500)
        if not allowed:
            best_checkpoint, best_summary = max(b_results, key=lambda item: self.rank(item[1]))
            c_allowed, c_reason = self.c_gate(baseline, best_summary)
            suffix = f"; {c_reason}; exact post-prior target audit required before C" if c_allowed else f"; {c_reason}"
            self.finish(status="stopped_by_gate", reason=reason + suffix, best_checkpoint=best_checkpoint, best_summary=best_summary)
            return

        _, checkpoint1000 = self.run_train(label="B1000", checkpoint=checkpoint500, load_mode="full", updates=500, num_envs=4096)
        self.verify_row_only_delta(checkpoint500, checkpoint1000, 1000)
        eval1000_dir, summary1000 = self.evaluate("B1000", checkpoint1000)
        b_results.append((checkpoint1000, summary1000))
        evaluation_dirs[str(checkpoint1000)] = eval1000_dir
        if self.finish_or_hold_behavior_candidate(
            reason="B1000 reached final 8/9 behavior gate",
            checkpoint=checkpoint1000,
            summary=summary1000,
            evaluation_dir=eval1000_dir,
        ):
            return
        allowed, reason = self.gate_1000_to_2000(summary500, summary1000)
        if not allowed:
            best_checkpoint, best_summary = max(b_results, key=lambda item: self.rank(item[1]))
            c_allowed, c_reason = self.c_gate(baseline, best_summary)
            suffix = f"; {c_reason}; exact post-prior target audit required before C" if c_allowed else f"; {c_reason}"
            self.finish(status="stopped_by_gate", reason=reason + suffix, best_checkpoint=best_checkpoint, best_summary=best_summary)
            return

        _, checkpoint2000 = self.run_train(label="B2000", checkpoint=checkpoint1000, load_mode="full", updates=1000, num_envs=4096)
        self.verify_row_only_delta(checkpoint1000, checkpoint2000, 2000)
        eval2000_dir, summary2000 = self.evaluate("B2000", checkpoint2000)
        b_results.append((checkpoint2000, summary2000))
        evaluation_dirs[str(checkpoint2000)] = eval2000_dir
        best_checkpoint, best_summary = max(b_results, key=lambda item: self.rank(item[1]))
        if self.behavior_final_gate(best_summary):
            evaluation_dir = evaluation_dirs[str(best_checkpoint)]
            self.finish_or_hold_behavior_candidate(
                reason="B stage reached final 8/9 behavior gate",
                checkpoint=best_checkpoint,
                summary=best_summary,
                evaluation_dir=evaluation_dir,
            )
            return
        c_allowed, c_reason = self.c_gate(baseline, best_summary)
        suffix = "; exact post-prior target audit required before C" if c_allowed else ""
        self.finish(status="stopped_by_gate", reason=f"B2000 maximum reached without final gate; {c_reason}{suffix}", best_checkpoint=best_checkpoint, best_summary=best_summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    args = parser.parse_args()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_ROOT / "supervisor.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A highstep Student recovery supervisor already holds the lock", file=sys.stderr)
        return 2

    existing_state_path = STATE_ROOT / "state.json"
    if existing_state_path.exists():
        try:
            existing_state = json.loads(existing_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Existing recovery state is corrupt; refusing to start a new branch", file=sys.stderr)
            return 3
        if existing_state.get("workflow_id") == WORKFLOW_ID:
            print(
                "Existing recovery workflow state found; refusing an implicit retry. "
                "Inspect state.json/handoff.json and resume explicitly after audit.",
                file=sys.stderr,
            )
            return 3

    supervisor = Supervisor(args.baseline_dir.resolve())

    def handle_signal(signum, _frame):
        supervisor.update_state(status="stopping", stop_reason=f"supervisor_signal_{signum}")
        supervisor.stop_active_process()
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        supervisor.run()
        return 0
    except BaseException as error:
        supervisor.stop_active_process()
        atomic_json(
            supervisor.handoff_path,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "status": "failed_closed",
                "phase": supervisor.state.get("phase"),
                "error": f"{type(error).__name__}: {error}",
                "state": supervisor.state,
                "failed_at": now_text(),
            },
        )
        supervisor.update_state(status="failed_closed", stop_reason=f"{type(error).__name__}: {error}", handoff=str(supervisor.handoff_path), active_pid=None)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
