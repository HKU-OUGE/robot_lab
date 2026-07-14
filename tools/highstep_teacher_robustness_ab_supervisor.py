#!/usr/bin/env python3
"""Independent fail-closed supervisor for v1.11 Teacher robustness A/B."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

ROOT = Path("/home/lxq/Softwares/robot_lab")
WORK = ROOT / "tmp/highstep_teacher_robustness_ab_20260715"
WORKFLOW = "highstep_teacher_robustness_ab_20260715"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "445b322302f1e1b64a621208670c7fa8736d7f1a64dec2d000f6b629e936076a"
PREREG = WORK / "preregistration_v111.json"
PREREG_SHA = "1b2bd47960d9111cf1b98676ece01524d89a889fc4b55cfc1ed08174782f2a55"
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
SERVICE = "highstep-teacher-robustness-ab.service"

sys.path.insert(0, str(ROOT))
from tools.highstep_teacher_robustness_ab import (  # noqa: E402
    aggregate_teacher,
    build_matrix,
    improvement_decision,
    sha256_file,
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)
    if read_only:
        path.chmod(0o444)


def tagged_json(log: Path, prefix: str) -> dict[str, Any]:
    matches = []
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            matches.append(json.loads(line[len(prefix):]))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix.strip()} in {log}, got {len(matches)}")
    return matches[0]


class Supervisor:
    def __init__(self) -> None:
        self.lock = (WORK / "supervisor.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.active: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.state_path = WORK / "state.json"
        self.heartbeat_path = WORK / "heartbeat.json"
        self.handoff_path = WORK / "handoff.json"
        self._validate_authority()
        self.state = self._load_state()

    def _validate_authority(self) -> None:
        if sha256_file(SPEC) != SPEC_SHA or sha256_file(PREREG) != PREREG_SHA:
            raise RuntimeError("v1.11 spec/preregistration binding mismatch")
        if PREREG.stat().st_mode & 0o222:
            raise RuntimeError("v1.11 preregistration must be read-only")
        binding = Path(os.environ["HIGHSTEP_TEACHER_AB_RUNTIME_BINDING"]).resolve(strict=True)
        expected = os.environ["HIGHSTEP_TEACHER_AB_RUNTIME_BINDING_SHA256"]
        if sha256_file(binding) != expected or binding.stat().st_mode & 0o222:
            raise RuntimeError("v1.11 runtime binding mismatch")
        payload = json.loads(binding.read_text())
        checks = {
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "supervisor_sha256": sha256_file(Path(__file__)),
            "executor_sha256": sha256_file(ROOT / "scripts/rsl_rl/base/highstep_teacher_robustness_ab_play.py"),
            "contract_sha256": sha256_file(ROOT / "tools/highstep_teacher_robustness_ab.py"),
        }
        if payload.get("workflow_id") != WORKFLOW or any(payload.get(k) != v for k, v in checks.items()):
            raise RuntimeError("v1.11 runtime code hashes changed")
        prereg = json.loads(PREREG.read_text())
        for teacher in ("A", "B"):
            item = prereg["teachers"][teacher]
            if sha256_file(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
                raise RuntimeError(f"Teacher {teacher} checkpoint SHA mismatch")
        if len(build_matrix()) != 102:
            raise RuntimeError("v1.11 matrix cardinality changed")

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("workflow_id") != WORKFLOW or state.get("spec_sha256") != SPEC_SHA or state.get("preregistration_sha256") != PREREG_SHA:
                raise RuntimeError("stored v1.11 state authority mismatch")
            return state
        return {
            "schema_version": 1, "workflow_id": WORKFLOW, "authority_version": "v1.11",
            "spec_path": str(SPEC), "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG), "preregistration_sha256": PREREG_SHA,
            "status": "preflight", "phase": "preflight", "failure_class": "none",
            "supervisor_pid": os.getpid(), "active_pid": None, "current_teacher": None,
            "current_run_id": None, "A_completed": 0, "B_completed": 0, "total_completed": 0,
            "infrastructure_retry_count": 0, "invalid_infrastructure_count": 0,
            "training_allowed": False, "requires_user_action": False, "updated_at": now(),
        }

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["supervisor_pid"] = os.getpid()
        self.state["updated_at"] = now()
        atomic_json(self.state_path, self.state)
        atomic_json(self.heartbeat_path, {
            "schema_version": 1, "workflow_id": WORKFLOW,
            "status": self.state["status"], "phase": self.state["phase"],
            "supervisor_pid": os.getpid(), "active_pid": self.state.get("active_pid"),
            "current_teacher": self.state.get("current_teacher"),
            "current_run_id": self.state.get("current_run_id"),
            "total_completed": self.state.get("total_completed", 0),
            "written_at": now(), "written_epoch": time.time(),
        })

    def completed(self, teacher: str) -> list[dict[str, Any]]:
        rows = []
        for row in build_matrix():
            result = WORK / "evaluations" / teacher / str(row["run_id"]) / "result.json"
            if result.exists():
                payload = json.loads(result.read_text())
                if payload.get("teacher") != teacher or payload.get("run_id") != row["run_id"] or payload.get("valid") is not True:
                    raise RuntimeError(f"stored result failed validation: {result}")
                rows.append(payload)
        return rows

    def refresh_counts(self) -> None:
        a = len(self.completed("A"))
        b = len(self.completed("B"))
        self.state.update(A_completed=a, B_completed=b, total_completed=a + b)

    def _archive_failed_attempt(self, teacher: str, run_id: str, attempt: int, run_dir: Path) -> None:
        target = WORK / "failure_recovery" / f"{teacher}_{run_id}_attempt{attempt}_{time.strftime('%Y%m%d_%H%M%S')}"
        target.mkdir(parents=True, exist_ok=True)
        if run_dir.exists():
            for path in run_dir.iterdir():
                shutil.move(str(path), target / path.name)
        evidence = WORK / "evidence" / teacher
        for suffix in (".frames.jsonl", ".runtime.json"):
            path = evidence / f"{run_id}{suffix}"
            if path.exists():
                shutil.move(str(path), target / path.name)

    def run_one(self, teacher: str, row: Mapping[str, Any]) -> dict[str, Any]:
        run_id = str(row["run_id"])
        run_dir = WORK / "evaluations" / teacher / run_id
        result_path = run_dir / "result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        prereg = json.loads(PREREG.read_text())
        t = prereg["teachers"][teacher]
        for attempt in range(1, 4):
            run_dir.mkdir(parents=True, exist_ok=True)
            log = run_dir / "play.log"
            env = os.environ.copy()
            env.update({
                "HIGHSTEP_TEACHER_AB_ROOT": str(WORK),
                "HIGHSTEP_TEACHER_AB_RUN_ID": run_id,
                "HIGHSTEP_TEACHER_AB_TEACHER": teacher,
                "HIGHSTEP_TEACHER_AB_SNAPSHOT_MODE": t["snapshot_mode"],
                "HIGHSTEP_TEACHER_AB_PREREGISTRATION": str(PREREG),
                "HIGHSTEP_TEACHER_AB_PREREGISTRATION_SHA256": PREREG_SHA,
            })
            cmd = [
                str(PYTHON), "scripts/rsl_rl/base/highstep_teacher_robustness_ab_play.py",
                "--task", prereg["environment"]["task"], "--num_envs", "1", "--headless",
                "--seed", str(row["seed"]), "--checkpoint", t["checkpoint"],
                "--play_terrain_type", "box", "--play_terrain_level", "9",
                "--fixed_velocity_command", "0.45", "0.0", "0.0",
                "--reset_after_play_terrain_selection", "--front_step_eval_reset",
                "--front_step_eval_side", "x-", "--front_step_eval_edge_gap", "0.55",
                "--front_step_eval_lateral_offset", "0.0", "--front_step_eval_yaw_offset_deg", "0.0",
                "--print_rear_width_metrics", "--skip_policy_export", "--rear_width_metric_interval", "60",
                "--play_max_steps", "600", "--eval_action_delay_steps", "0",
                "--allow_legacy_highstep_schedule_fallback",
            ]
            started = time.time()
            with log.open("w", encoding="utf-8") as stream:
                self.active = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                self.update(status="evaluating", phase="teacher_ab_rollout", active_pid=self.active.pid,
                            current_teacher=teacher, current_run_id=run_id, current_row=dict(row),
                            current_attempt=attempt, current_started_epoch=started, latest_log=str(log))
                while self.active.poll() is None:
                    if self.stop_requested:
                        os.killpg(self.active.pid, signal.SIGINT)
                    self.update(status="evaluating", phase="teacher_ab_rollout", active_pid=self.active.pid)
                    time.sleep(3)
                rc = int(self.active.returncode or 0)
            self.active = None
            self.update(active_pid=None, phase="validating_rollout")
            if rc == 0:
                try:
                    canonical = tagged_json(log, "[HIGHSTEP_EVAL_JSON] ")
                    runtime = tagged_json(log, "[HIGHSTEP_TEACHER_AB_RUNTIME_JSON] ")
                    checks = [
                        canonical.get("checkpoint_sha256") == t["checkpoint_sha256"],
                        canonical.get("schedule_valid") is True,
                        canonical.get("reset_valid") is True,
                        canonical.get("action_prior_runtime_match") is True,
                        canonical.get("action_prior_enabled") is True,
                        abs(float(canonical.get("action_prior_scale_runtime", -1)) - 1.0) <= 1.0e-9,
                        canonical.get("eval_action_delay_runtime_match") is True,
                        canonical.get("eval_action_delay_steps_runtime") == 0,
                        canonical.get("keep_play_randomization") is False,
                        canonical.get("fixed_velocity_command") == [0.45, 0.0, 0.0],
                        canonical.get("loop_steps") == 600,
                        runtime.get("read_only_checks_passed") is True,
                        runtime.get("snapshot_sha256") == sha256_file(WORK / "snapshots" / f"{run_id}.json"),
                    ]
                    target = row["initial_rear_width_m"]
                    if target is not None:
                        checks.append(abs(float(runtime["initial_rear_width_m"]) - float(target)) <= 0.002)
                    if not all(checks):
                        raise RuntimeError(f"v1.11 runtime validity checks failed: {checks}")
                    result = {
                        **dict(row), "schema_version": 1, "kind": "highstep_v111_teacher_ab_result",
                        "workflow_id": WORKFLOW, "teacher": teacher, "valid": True,
                        "full_climb": canonical.get("full_climb_success") is True,
                        "rear_hold": canonical.get("rear_on_platform_hold_success") is True,
                        "recovery": runtime.get("recovery") is True,
                        "recovery_time_s": runtime.get("recovery_time_s"),
                        "rear_width_min_m": runtime.get("rear_width_min_m"),
                        "rear_min_abs_y_min_m": runtime.get("rear_min_abs_y_min_m"),
                        "centerline_crossed": runtime.get("centerline_crossed") is True,
                        "roll_metric_abs_max": runtime.get("roll_metric_abs_max"),
                        "yaw_deviation_abs_max_rad": runtime.get("yaw_deviation_abs_max_rad"),
                        "fell": runtime.get("fell") is True,
                        "impulse_applied": runtime.get("impulse_applied") is True,
                        "impulse_step": runtime.get("impulse_step"),
                        "snapshot": runtime["snapshot"], "snapshot_sha256": runtime["snapshot_sha256"],
                        "checkpoint_sha256_before": runtime["checkpoint_sha256_before"],
                        "checkpoint_sha256_after": runtime["checkpoint_sha256_after"],
                        "policy_tensor_sha256_before": runtime["policy_tensor_sha256_before"],
                        "policy_tensor_sha256_after": runtime["policy_tensor_sha256_after"],
                        "log": str(log), "log_sha256": sha256_file(log),
                        "runtime_evidence": str(WORK / "evidence" / teacher / f"{run_id}.runtime.json"),
                        "duration_seconds": time.time() - started, "completed_at": now(),
                    }
                    atomic_json(result_path, result, read_only=True)
                    self.refresh_counts()
                    self.update(phase="rollout_completed", current_duration_seconds=result["duration_seconds"])
                    return result
                except Exception as error:
                    rc = 90
                    self.state["last_error"] = f"{type(error).__name__}: {error}"
            self.state["invalid_infrastructure_count"] = int(self.state.get("invalid_infrastructure_count", 0)) + 1
            if attempt >= 3:
                raise RuntimeError(f"rollout infrastructure failed after 3 attempts: {teacher}/{run_id}, rc={rc}")
            self.state["infrastructure_retry_count"] = int(self.state.get("infrastructure_retry_count", 0)) + 1
            self._archive_failed_attempt(teacher, run_id, attempt, run_dir)
            self.update(status="infrastructure_retry", phase="archived_retryable_rollout", active_pid=None)
        raise AssertionError("unreachable")

    def run(self) -> None:
        self.update(status="running", phase="authority_verified", failure_class="none", active_pid=None)
        for teacher in ("A", "B"):
            for row in build_matrix():
                if self.stop_requested:
                    raise KeyboardInterrupt
                self.run_one(teacher, row)
        old_rows = self.completed("A")
        new_rows = self.completed("B")
        old = aggregate_teacher(old_rows)
        new = aggregate_teacher(new_rows)
        decision = improvement_decision(old, new)
        manifest = {
            "schema_version": 1, "kind": "highstep_v111_teacher_robustness_ab_decision",
            "workflow_id": WORKFLOW, "spec_sha256": SPEC_SHA, "preregistration_sha256": PREREG_SHA,
            "teacher_A": old, "teacher_B": new, "decision": decision,
            "completed_at": now(),
        }
        decision_path = WORK / "manifests" / "teacher_robustness_ab_decision.json"
        atomic_json(decision_path, manifest, read_only=True)
        handoff = {
            "schema_version": 1, "workflow_id": WORKFLOW, "status": decision["status"],
            "decision_manifest": str(decision_path), "decision_manifest_sha256": sha256_file(decision_path),
            "training_started": False, "v110_resumed": False, "next_action": "report Teacher A/B comparison only",
            "written_at": now(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update(status=decision["status"], phase=decision["status"], active_pid=None,
                    current_teacher=None, current_run_id=None, decision=decision,
                    decision_manifest=str(decision_path), decision_manifest_sha256=sha256_file(decision_path),
                    requires_user_action=False)
        subprocess.run(["systemctl", "--user", "disable", SERVICE], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self, *_args: Any) -> None:
        self.stop_requested = True


def main() -> int:
    supervisor: Supervisor | None = None
    try:
        supervisor = Supervisor()
        signal.signal(signal.SIGTERM, supervisor.stop)
        signal.signal(signal.SIGINT, supervisor.stop)
        supervisor.run()
        return 0
    except BlockingIOError:
        return 3
    except KeyboardInterrupt:
        if supervisor is not None:
            supervisor.update(status="paused", phase="paused_at_safe_boundary", active_pid=None)
        return 0
    except Exception as error:
        if supervisor is not None:
            supervisor.update(status="infrastructure_failed", phase="infrastructure_failed", failure_class="infrastructure",
                              active_pid=None, last_error=f"{type(error).__name__}: {error}", requires_user_action=True)
            atomic_json(supervisor.handoff_path, {
                "schema_version": 1, "workflow_id": WORKFLOW, "status": "infrastructure_failed",
                "error": f"{type(error).__name__}: {error}", "training_started": False,
                "next_action": "repair infrastructure and resume only missing rollout", "written_at": now(),
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
