#!/usr/bin/env python3
"""Independent v1.9 supervisor for the read-only oracle prior-delta A/B."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.highstep_oracle_prior_delta import (
    counts_from_eval_rows,
    load_json_lines,
    oracle_recovery_decision,
    sha256_file,
)
from tools.highstep_oracle_prior_delta_monitor import materialize


ROOT = Path("/home/lxq/Softwares/robot_lab")
STATE_ROOT = ROOT / "tmp/highstep_oracle_prior_delta_ab_20260714"
WORKFLOW_ID = "highstep_oracle_prior_delta_ab_20260714"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA256 = "5446b9102c58f12561d10b563bf6e98ff6f79b53f3432383f026b1b356973bf7"
PREREG_BASE = STATE_ROOT / "preregistration_v19.json"
PREREG_BASE_SHA256 = "5579552b1d876fd1ac8a130c65227f6a2636f97efc97851552197acb6920e589"
PREREG = STATE_ROOT / "preregistration_v19_runtime_rebinding.json"
PREREG_SHA256 = "9fc7c4ae1a5a215ee116beaf8074d6f824d76277b1fd6fb923b9a8f893abfe19"
CHECKPOINT = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_environment_curriculum_v18_Student/2026-07-14_21-03-31_v18_stage_a_E1400_20260714_210326/model_1394.pt"
CHECKPOINT_SHA256 = "92bf3d0612f0f9ea1f85b379af8754a0df2710febb9b0067fd86e17487c810f9"
TEACHER = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt"
TEACHER_SHA256 = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
RUN_DIR = CHECKPOINT.parent
TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-ArcdogAdjustableLeg-v0"
PARENT = ROOT / "tmp/highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json"
MONITOR = STATE_ROOT / "infrastructure/highstep_oracle_prior_delta_core9.sh"
MONITOR_SHA256 = "18dd8cd62881ca537a2b09efcca7a8c52cb1854350aaee1cef0fc63aef65c0c1"
RUNTIME_BINDING = STATE_ROOT / "manifests/runtime_code_rebinding.json"
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(0o444)


class Supervisor:
    def __init__(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.lock_handle = (STATE_ROOT / "supervisor.lock").open("a+")
        fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = STATE_ROOT / "state.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.active: subprocess.Popen[str] | None = None
        self._validate_authority()
        self.state = self._load_or_initialize_state()

    def _validate_authority(self) -> None:
        bindings = (
            (SPEC, SPEC_SHA256, "spec"),
            (PREREG, PREREG_SHA256, "preregistration"),
            (CHECKPOINT, CHECKPOINT_SHA256, "Student checkpoint"),
            (TEACHER, TEACHER_SHA256, "Teacher checkpoint"),
        )
        for path, expected, label in bindings:
            actual = sha256_file(path.resolve(strict=True))
            if actual != expected:
                raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")
        prereg = json.loads(PREREG.read_text())
        if prereg.get("workflow_id") != WORKFLOW_ID or prereg.get("training_allowed") is not False:
            raise RuntimeError("v1.9 preregistration workflow/training scope changed")
        if (
            prereg.get("base_preregistration_path") != str(PREREG_BASE)
            or prereg.get("base_preregistration_sha256") != PREREG_BASE_SHA256
            or sha256_file(PREREG_BASE) != PREREG_BASE_SHA256
            or prereg.get("repaired_oracle_play_wrapper_sha256")
            != "0f5d072328ea415a33efe0988251843514d2bb820d35b9b0f2ba8ae03c156207"
        ):
            raise RuntimeError("v1.9 base A/B preregistration binding changed")
        if PREREG.stat().st_mode & 0o222:
            raise RuntimeError("v1.9 preregistration must be read-only")
        if materialize(MONITOR) != MONITOR_SHA256:
            raise RuntimeError("v1.9 derived monitor SHA mismatch")
        binding = json.loads(RUNTIME_BINDING.resolve(strict=True).read_text())
        if (
            binding.get("workflow_id") != WORKFLOW_ID
            or binding.get("spec_sha256") != SPEC_SHA256
            or binding.get("preregistration_sha256") != PREREG_SHA256
            or binding.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or binding.get("teacher_checkpoint_sha256") != TEACHER_SHA256
            or binding.get("monitor_sha256") != MONITOR_SHA256
            or binding.get("supervisor_sha256") != sha256_file(Path(__file__))
        ):
            raise RuntimeError("v1.9 runtime code rebinding manifest changed")
        if RUNTIME_BINDING.stat().st_mode & 0o222:
            raise RuntimeError("v1.9 runtime code rebinding manifest must be read-only")

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("workflow_id") != WORKFLOW_ID or state.get("spec_sha256") != SPEC_SHA256 or state.get("preregistration_sha256") != PREREG_SHA256:
                raise RuntimeError("stored v1.9 state authority binding changed")
            return state
        state = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "authority_version": "v1.9",
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA256,
            "preregistration_path": str(PREREG),
            "preregistration_sha256": PREREG_SHA256,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "teacher_checkpoint": str(TEACHER),
            "teacher_checkpoint_sha256": TEACHER_SHA256,
            "status": "ready",
            "phase": "ready",
            "active_pid": None,
            "supervisor_pid": os.getpid(),
            "training_allowed": False,
            "updated_at": now(),
        }
        atomic_json(self.state_path, state)
        return state

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["supervisor_pid"] = os.getpid()
        self.state["updated_at"] = now()
        atomic_json(self.state_path, self.state)
        heartbeat = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "status": self.state["status"],
            "phase": self.state["phase"],
            "supervisor_pid": os.getpid(),
            "active_pid": self.state.get("active_pid"),
            "checkpoint": str(CHECKPOINT),
            "training_allowed": False,
            "written_at": now(),
        }
        atomic_json(self.heartbeat_path, heartbeat)

    def _run_mode(self, mode: str) -> dict[str, Any]:
        root = STATE_ROOT / "evaluations" / mode
        terminal = root / "mode_result.json"
        if terminal.exists():
            return json.loads(terminal.read_text())
        root.mkdir(parents=True, exist_ok=True)
        evidence = STATE_ROOT / "evidence"
        environment = os.environ.copy()
        environment.update(
            {
                "RECOVERY_SPEC_MODE": "1",
                "WORKFLOW_ID": WORKFLOW_ID,
                "LEDGER_FILE": str(STATE_ROOT / "core9_ledger.jsonl"),
                "EVAL_CHECKPOINT_COUNT": "1",
                "EVAL_CHECKPOINT_PATH": str(CHECKPOINT),
                "EVAL_CHECKPOINT_STRIDE": "1",
                "EVAL_ALLOW_LEGACY_SCHEDULE_FALLBACK": "0",
                "POLL_SECONDS": "5",
                "PLAY_STALL_TIMEOUT_SECONDS": "900",
                "PLAY_WATCHDOG_POLL_SECONDS": "10",
                "MIN_FREE_DISK_GIB": "15",
                "HIGHSTEP_ORACLE_PRIOR_DELTA_MODE": mode,
                "HIGHSTEP_ORACLE_PRIOR_DELTA_EVIDENCE_ROOT": str(evidence),
                "HIGHSTEP_ORACLE_PRIOR_DELTA_PREREGISTRATION": str(PREREG),
                "HIGHSTEP_ORACLE_PRIOR_DELTA_PREREGISTRATION_SHA256": PREREG_SHA256,
                "HIGHSTEP_V18_PREREGISTRATION_PATH": str(
                    ROOT / "tmp/highstep_student_env_curriculum_v18_20260714/preregistration_v18_12.json"
                ),
                "HIGHSTEP_V18_PREREGISTRATION_SHA256": "fd38dd7e3270fd257c211ca8c23d15d9719cbef3e817dd1281b73f122563fac1",
            }
        )
        command = [
            "bash", str(MONITOR), "0", str(RUN_DIR), "/dev/null", str(root),
            TASK, "student", str(PARENT),
        ]
        log_path = root / "launcher.log"
        with log_path.open("a", encoding="utf-8") as log:
            self.active = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.update(status="evaluating", phase=f"oracle_ab_{mode}", active_pid=self.active.pid)
            while self.active.poll() is None:
                self.update(status="evaluating", phase=f"oracle_ab_{mode}", active_pid=self.active.pid)
                time.sleep(5)
            return_code = int(self.active.returncode or 0)
        self.active = None
        self.update(status="evaluating", phase=f"oracle_ab_{mode}_validating", active_pid=None)
        if return_code != 0:
            raise RuntimeError(f"oracle {mode} core9 monitor failed with rc={return_code}")

        manifest = json.loads((root / "evaluation_manifest.json").read_text())
        rows = load_json_lines(root / "eval_runs.jsonl")
        if manifest.get("evaluation_complete") is not True or manifest.get("matrix_complete") is not True or manifest.get("workflow_id") != WORKFLOW_ID:
            raise RuntimeError(f"oracle {mode} core9 manifest is incomplete")
        for row in rows:
            item = row.get("eval")
            if not isinstance(item, Mapping) or item.get("checkpoint_sha256") != CHECKPOINT_SHA256:
                raise RuntimeError(f"oracle {mode} evaluation checkpoint binding failed")
        counts = counts_from_eval_rows(rows)

        summaries = []
        for seed in (11, 22, 33):
            for scenario in ("nominal", "left_offset", "right_offset"):
                path = evidence / mode / f"seed{seed}_{scenario}.summary.json"
                summary = json.loads(path.read_text())
                if (
                    summary.get("mode") != mode
                    or summary.get("seed") != seed
                    or summary.get("scenario") != scenario
                    or summary.get("all_read_only_and_action_contract_checks_passed") is not True
                    or summary.get("preregistration_sha256") != PREREG_SHA256
                    or summary.get("student_checkpoint_sha256_after") != CHECKPOINT_SHA256
                    or summary.get("teacher_checkpoint_sha256_after") != TEACHER_SHA256
                    or float(summary.get("protected_nonbox_max_abs_change", 1.0)) != 0.0
                ):
                    raise RuntimeError(f"oracle {mode} runtime evidence failed: {path}")
                summaries.append({"path": str(path), "sha256": sha256_file(path), "frame_count": summary["frame_count"]})
        payload = {
            "schema_version": 1,
            "kind": "highstep_v19_oracle_prior_delta_mode_result",
            "workflow_id": WORKFLOW_ID,
            "mode": mode,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "teacher_checkpoint": str(TEACHER),
            "teacher_checkpoint_sha256": TEACHER_SHA256,
            "counts": counts,
            "evaluation_manifest": str(root / "evaluation_manifest.json"),
            "evaluation_manifest_sha256": sha256_file(root / "evaluation_manifest.json"),
            "runtime_evidence": summaries,
            "completed_at": now(),
        }
        atomic_json(terminal, payload, read_only=True)
        return payload

    def run(self) -> None:
        self.update(status="running", phase="authority_verified", active_pid=None)
        control = self._run_mode("control")
        oracle = self._run_mode("oracle_prior_delta")
        decision = oracle_recovery_decision(control["counts"], oracle["counts"])
        payload = {
            "schema_version": 1,
            "kind": "highstep_v19_oracle_prior_delta_ab_decision",
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA256,
            "preregistration_sha256": PREREG_SHA256,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "teacher_checkpoint": str(TEACHER),
            "teacher_checkpoint_sha256": TEACHER_SHA256,
            "control": control,
            "oracle": oracle,
            "decision": decision,
            "completed_at": now(),
        }
        decision_path = STATE_ROOT / "manifests/oracle_prior_delta_ab_decision.json"
        atomic_json(decision_path, payload, read_only=True)
        status = str(decision["status"])
        next_action = (
            "write_residual_optimizer_budget_preregistration_then_static_binding_smoke"
            if decision["recovered"]
            else "none; residual training forbidden"
        )
        handoff = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "status": status,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "decision_manifest": str(decision_path),
            "decision_manifest_sha256": sha256_file(decision_path),
            "residual_training_allowed": bool(decision["recovered"]),
            "next_action": next_action,
            "written_at": now(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update(
            status=status,
            phase=status,
            active_pid=None,
            control_counts=control["counts"],
            oracle_counts=oracle["counts"],
            decision=decision,
            decision_manifest=str(decision_path),
            decision_manifest_sha256=sha256_file(decision_path),
            requires_user_action=False,
        )

    def terminate_child(self) -> None:
        if self.active is not None and self.active.poll() is None:
            os.killpg(self.active.pid, signal.SIGINT)


def main() -> int:
    supervisor: Supervisor | None = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("v1.9 oracle supervisor lock is already held", file=sys.stderr)
        return 3
    except Exception as error:
        if supervisor is not None:
            supervisor.update(
                status="evaluation_infrastructure_retryable",
                phase="evaluation_infrastructure_retryable",
                active_pid=None,
                last_error=f"{type(error).__name__}: {error}",
            )
            atomic_json(
                supervisor.handoff_path,
                {
                    "schema_version": 1,
                    "workflow_id": WORKFLOW_ID,
                    "status": "evaluation_infrastructure_retryable",
                    "error": f"{type(error).__name__}: {error}",
                    "next_action": "systemd restart resumes only missing A/B mode",
                    "written_at": now(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
