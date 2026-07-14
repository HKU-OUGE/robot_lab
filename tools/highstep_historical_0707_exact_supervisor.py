#!/usr/bin/env python3
"""Autonomous supervisor for the evidence-frozen historical 0707 Stage-2 route."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping


ROOT = Path("/home/lxq/Softwares/robot_lab")
LEGACY_PATH = ROOT / "tools/highstep_0707_exact_new_teacher_supervisor.py"
_spec = importlib.util.spec_from_file_location(
    "highstep_zero_scale_supervisor_reused_mechanics", LEGACY_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load frozen highstep supervisor mechanics")
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
prior = legacy.prior
base = legacy.base

WORKFLOW_ID = "highstep_historical_0707_exact_new_teacher_20260714"
STATE_ROOT = ROOT / "tmp/highstep_historical_0707_exact_new_teacher_20260714"
SPEC_SHA = "140fd81d4f6877d25e72f3f1e05799cb6771dfdfc9775fe84a46ecf7d7a6a917"
OLD_SPEC_SHA = "eff246af70dbfca7b3a6661f0a29f71bb1aa3b769a944351431f8b589663437a"
OLD_PREREG = STATE_ROOT / "preregistration.json"
OLD_PREREG_SHA = "150675a9aed90e32c16f58f5f08c315db960cfdcc7ec00f22500521d9c1c306d"
PREREG = Path(os.environ.get(
    "HIGHSTEP_HISTORICAL_0707_EXACT_PREREGISTRATION_PATH",
    STATE_ROOT / "preregistration_v171.json",
))
PREREG_SHA = os.environ.get("HIGHSTEP_HISTORICAL_0707_EXACT_PREREGISTRATION_SHA256", "")
REBINDING = Path(os.environ.get(
    "HIGHSTEP_HISTORICAL_0707_EXACT_REBINDING_PATH",
    STATE_ROOT / "v171_infrastructure_amendment.json",
))
REBINDING_SHA = os.environ.get("HIGHSTEP_HISTORICAL_0707_EXACT_REBINDING_SHA256", "")
E100_CHECKPOINT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "historical_0707_exact_new_teacher_Student/"
    "2026-07-14_03-56-25_historical_0707_exact_E100_20260714_035620/model_99.pt"
)
E100_CHECKPOINT_SHA = "5ab0254d99cf2009dac781d217326d2c768807e41989f91ae6b6d9ed8480da62"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
SOURCE = STATE_ROOT / "source_model_172300/model_172300.pt"
SOURCE_SCHEDULE = STATE_ROOT / "source_model_172300/highstep_schedule_manifest.json"
SOURCE_RUNTIME = STATE_ROOT / "source_model_172300/highstep_runtime_state.json"
SOURCE_SCHEDULE_SHA = "fc8de4ab2b5c30999b451a2e96a70dc2daeed67e3798adbe052e6086504a2aea"
SOURCE_RUNTIME_SHA = "b0d413f2edd1d9cc7cec01467112027c0b580614c537b0382770777e260668e4"
CORRECTION = ROOT / (
    "tmp/highstep_historical_0707_exact_20260714/historical_evidence_correction.json"
)
CORRECTION_SHA = "eb561df82d5f116577497eb7c3f553bb82ae81493258599212da59d4d0c6a175"
ZERO_SCALE_STOP = ROOT / (
    "tmp/highstep_0707_exact_new_teacher_20260713/zero_scale_ablation_stop_manifest.json"
)
SAVE_POINTS = (100, 300, 500, 700, 900, 1400)
CORE9_POINTS = set(SAVE_POINTS)
TRAIN_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorHistorical0707Exact-"
    "ArcdogAdjustableLeg-v0"
)
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_historical_0707_exact_new_teacher_Student"
)

# Rebind all globals used by the already-tested stage/evaluation/W&B mechanics.
for module in (legacy, prior, base):
    module.WORKFLOW_ID = WORKFLOW_ID
    module.STATE_ROOT = STATE_ROOT
    module.SPEC_SHA = SPEC_SHA
    module.PREREG = PREREG
    module.PREREG_SHA = PREREG_SHA
legacy.TEACHER = TEACHER
legacy.TEACHER_SHA = TEACHER_SHA
legacy.SOURCE = SOURCE
legacy.SOURCE_SCHEDULE = SOURCE_SCHEDULE
legacy.SOURCE_RUNTIME = SOURCE_RUNTIME
legacy.SOURCE_SCHEDULE_SHA = SOURCE_SCHEDULE_SHA
legacy.SOURCE_RUNTIME_SHA = SOURCE_RUNTIME_SHA
legacy.SAVE_POINTS = SAVE_POINTS
legacy.CORE9_POINTS = CORE9_POINTS
legacy.TRAIN_TASK = TRAIN_TASK
legacy.EXPERIMENT_ROOT = EXPERIMENT_ROOT
prior.SOURCE_E300 = SOURCE
prior.SOURCE_E300_SHA = TEACHER_SHA
prior.SAVE_POINTS = SAVE_POINTS
prior.CORE9_POINTS = CORE9_POINTS
base.SOURCE = SOURCE
base.TEACHER = TEACHER
base.TRAIN_TASK = TRAIN_TASK
base.EXPERIMENT_ROOT = EXPERIMENT_ROOT
base.SAVE_POINTS = SAVE_POINTS
base.CORE9_POINTS = CORE9_POINTS
base.TRAIN_RUN_PREFIX = "historical_0707_exact"
base.PREREG_ENV_PREFIX = "HIGHSTEP_HISTORICAL_0707_EXACT"
base.WARMUP_UPDATES = 1400
base.ABSOLUTE_CAP = 1400


def atomic_json(path: Path, payload: Mapping[str, Any], read_only: bool = False) -> None:
    base.atomic_json(path, payload, read_only=read_only)


class Supervisor(legacy.Supervisor):
    """Historical 0707 authority; no zero-scale or post-prior-main fallback."""

    def update(self, **values: Any) -> None:
        values.setdefault("authority_version", "v1.7.1")
        values.setdefault("failure_class", self.state.get("failure_class", "none"))
        super().update(**values)
        heartbeat = json.loads(self.heartbeat_path.read_text())
        heartbeat.update({
            "authority_version": self.state["authority_version"],
            "failure_class": self.state["failure_class"],
        })
        atomic_json(self.heartbeat_path, heartbeat)

    @staticmethod
    def classify_failure(phase: str, message: str) -> str:
        text = f"{phase} {message}".lower()
        if "wandb" in text or "sync" in text:
            return "wandb_infrastructure_retryable"
        if "video" in text or "ffprobe" in text or "camera" in text:
            return "video_infrastructure_retryable"
        if any(token in text for token in ("imitation", "probe", "core9", "directional", "eval")):
            return "evaluation_infrastructure_retryable"
        if any(token in text for token in (
            "binding", "checkpoint", "optimizer", "teacher", "preregistration",
            "spec", "schedule", "tensor", "contract", "critical code",
        )):
            return "training_mechanism_error"
        return "external_fault_requires_user"

    def wait(self, process: subprocess.Popen[bytes], root: Path, phase: str, stall: int) -> None:
        if process.poll() is not None:
            raise RuntimeError(f"{phase} child exited before PID verification")
        status = (Path("/proc") / str(process.pid) / "status").read_text()
        parent_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
        parent_pid = int(parent_line.split()[1])
        process_group = os.getpgid(process.pid)
        if parent_pid != os.getpid() or process_group != process.pid:
            raise RuntimeError(
                f"{phase} child ownership mismatch: ppid={parent_pid}, pgid={process_group}"
            )
        self.update(status="running", phase=phase, active_pid=process.pid, failure_class="none")
        state = json.loads(self.state_path.read_text())
        heartbeat = json.loads(self.heartbeat_path.read_text())
        if any(item.get("active_pid") != process.pid or item.get("phase") != phase
               for item in (state, heartbeat)):
            raise RuntimeError(f"{phase} state/heartbeat PID witness mismatch")
        witness = STATE_ROOT / "launch_verifications" / f"{phase}.json"
        payload = {
            "schema_version": 1,
            "kind": "highstep_v171_child_pid_verification",
            "workflow_id": WORKFLOW_ID,
            "phase": phase,
            "supervisor_pid": os.getpid(),
            "child_pid": process.pid,
            "child_parent_pid": parent_pid,
            "child_process_group": process_group,
            "state_active_pid": state["active_pid"],
            "heartbeat_active_pid": heartbeat["active_pid"],
            "verified_at": base.now(),
        }
        if witness.exists():
            suffix = 2
            while witness.with_name(f"{witness.stem}_attempt{suffix}.json").exists():
                suffix += 1
            witness = witness.with_name(f"{witness.stem}_attempt{suffix}.json")
        atomic_json(witness, payload, read_only=True)
        super().wait(process, root, phase, stall)

    def preflight(self) -> None:
        if not PREREG_SHA or not REBINDING_SHA:
            raise RuntimeError("historical v1.7.1 authority SHA environment is missing")
        for path, expected in (
            (base.SPEC, SPEC_SHA),
            (PREREG, PREREG_SHA),
            (REBINDING, REBINDING_SHA),
            (OLD_PREREG, OLD_PREREG_SHA),
            (TEACHER, TEACHER_SHA),
            (SOURCE, TEACHER_SHA),
            (SOURCE_SCHEDULE, SOURCE_SCHEDULE_SHA),
            (SOURCE_RUNTIME, SOURCE_RUNTIME_SHA),
            (CORRECTION, CORRECTION_SHA),
        ):
            if base.sha256_file(path) != expected:
                raise RuntimeError(f"historical 0707 authority SHA mismatch: {path}")
        prereg = json.loads(PREREG.read_text())
        authority = prereg.get("authority", {})
        if not (
            prereg.get("kind") == "highstep_historical_0707_exact_v171_preregistration"
            and authority.get("version") == "v1.7.1"
            and authority.get("spec_sha256") == SPEC_SHA
            and prereg.get("historical_evidence_correction_sha256") == CORRECTION_SHA
        ):
            raise RuntimeError("historical v1.7.1 preregistration identity changed")
        resume = prereg.get("resume_rebinding", {})
        if not (
            resume.get("source_preregistration_sha256") == OLD_PREREG_SHA
            and resume.get("source_spec_sha256") == OLD_SPEC_SHA
            and resume.get("source_checkpoint") == str(E100_CHECKPOINT)
            and resume.get("source_checkpoint_sha256") == E100_CHECKPOINT_SHA
            and resume.get("source_effective_updates") == 100
            and resume.get("preserve_optimizer") is True
            and resume.get("training_contract_changed") is False
            and resume.get("audit_path") == str(REBINDING)
            and resume.get("audit_sha256") == REBINDING_SHA
        ):
            raise RuntimeError("historical v1.7.1 E100 resume rebinding changed")
        amendment = json.loads(REBINDING.read_text())
        if not (
            amendment.get("kind") == "highstep_v171_infrastructure_amendment"
            and amendment.get("training_contract_changed") is False
            and amendment.get("source_checkpoint_sha256") == E100_CHECKPOINT_SHA
            and amendment.get("old_services_disabled") is True
            and amendment.get("dashboard_workflow_id") == WORKFLOW_ID
        ):
            raise RuntimeError("historical v1.7.1 infrastructure amendment changed")
        if not ZERO_SCALE_STOP.is_file():
            raise RuntimeError("zero-scale ablation has not reached its safe archival boundary")
        archived = json.loads(ZERO_SCALE_STOP.read_text())
        if not (
            archived.get("historical_classification") == "zero_scale_ablation"
            and archived.get("resume_for_delivery_allowed") is False
            and archived.get("optimizer_hyperparameter_switch_in_place_allowed") is False
        ):
            raise RuntimeError("zero-scale archival boundary changed")
        changed = {
            name for name, expected in prereg.get("code_sha256", {}).items()
            if base.sha256_file(Path(name)) != expected
        }
        if changed:
            raise RuntimeError(f"historical 0707 preregistered code changed: {sorted(changed)}")
        self.require_idle()
        smoke_path = STATE_ROOT / "smoke/smoke_manifest.json"
        smoke = json.loads(smoke_path.read_text())
        if not (
            smoke.get("passed") is True
            and smoke.get("start_checkpoint_sha256") == TEACHER_SHA
            and smoke.get("end_effective_updates") in (1, 2, 3, 4, 5)
            and smoke.get("warmup_updates") == 1400
            and smoke.get("warmup_main_target") == "teacher_pre_prior"
            and smoke.get("phase_scale") == 2.0
            and smoke.get("rear_box_scale") == 1.5
            and smoke.get("phase_weight_active_from_update_zero") is True
            and smoke.get("rear_box_weight_active_from_update_zero") is True
            and smoke.get("warmup_prior_box_loss_zero") is True
            and smoke.get("actor_unchanged") is True
            and smoke.get("teacher_student_storage_independent") is True
            and smoke.get("full_checkpoint_restore_verified") is True
        ):
            raise RuntimeError("historical 0707 1-5 update smoke is missing or invalid")
        critical = [
            Path(__file__), LEGACY_PATH, legacy.BASE_PATH,
            ROOT / "tools/highstep_student_recovery_v15_supervisor.py",
            ROOT / "tools/highstep_wandb_stage_gate.py",
            ROOT / "tools/highstep_video_visibility_qc.py",
            base.CANDIDATE_AUDIT, base.IMITATION_GATE,
            ROOT / "tools/highstep_core9_task_compat.py",
            ROOT / "scripts/rsl_rl/base/train.py", ROOT / "scripts/rsl_rl/base/play.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__init__.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
            ROOT / "tmp/highstep_dashboard_state.py",
            ROOT / "tmp/highstep_train_dashboard.py",
            ROOT / "tmp/highstep_status_dashboard.py",
        ]
        self.critical_hashes = {
            str(path.resolve()): base.sha256_file(path.resolve()) for path in critical
        }
        if self.critical_hashes != prereg.get("code_sha256"):
            raise RuntimeError("historical 0707 critical code set differs from preregistration")
        preflight = STATE_ROOT / "preflight_v171_manifest.json"
        payload = {
            "schema_version": 1,
            "kind": "highstep_historical_0707_exact_v171_preflight",
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "authority_rebinding": str(REBINDING),
            "authority_rebinding_sha256": REBINDING_SHA,
            "source_e100_checkpoint": str(E100_CHECKPOINT),
            "source_e100_checkpoint_sha256": E100_CHECKPOINT_SHA,
            "historical_evidence_correction_sha256": CORRECTION_SHA,
            "zero_scale_ablation_stop_manifest": str(ZERO_SCALE_STOP),
            "zero_scale_ablation_stop_manifest_sha256": base.sha256_file(ZERO_SCALE_STOP),
            "teacher_sha256": TEACHER_SHA,
            "smoke_manifest": str(smoke_path),
            "smoke_manifest_sha256": base.sha256_file(smoke_path),
            "critical_code_sha256": self.critical_hashes,
            "warmup_updates": 1400,
            "warmup_main_target": "teacher_pre_prior",
            "phase_scale": 2.0,
            "rear_box_scale": 1.5,
            "warmup_prior_box_loss": 0.0,
            "fresh_from_teacher": False,
            "resume_from_complete_e100_with_optimizer": True,
            "training_contract_changed": False,
            "completed_at": base.now(),
        }
        if preflight.exists():
            stored = json.loads(preflight.read_text())
            stable = {k: v for k, v in payload.items() if k != "completed_at"}
            old = {k: v for k, v in stored.items() if k != "completed_at"}
            if stable != old or not base.is_read_only(preflight):
                raise RuntimeError("stored historical 0707 preflight changed")
        else:
            atomic_json(preflight, payload, read_only=True)
        self.update(
            status="preflight_passed",
            phase="ready_for_stage_100",
            effective_updates=100,
            checkpoint=str(E100_CHECKPOINT),
            code_sha256=self.critical_hashes,
            stop_reason=None,
            last_error=None,
            failure_class="none",
        )

    def train_stage(self, stage: int, source: Path, previous: int):
        """Adopt the completed v1.7 E100 artifact once; never retrain it."""
        training_result = STATE_ROOT / "stages" / f"E{stage}" / "training_result.json"
        if stage != 100 or training_result.is_file():
            return super().train_stage(stage, source, previous)
        if base.sha256_file(E100_CHECKPOINT) != E100_CHECKPOINT_SHA:
            raise RuntimeError("completed E100 checkpoint SHA mismatch")
        if self.checkpoint_count(E100_CHECKPOINT) != 100:
            raise RuntimeError("completed E100 checkpoint update count mismatch")
        launch = json.loads((STATE_ROOT / "stages/E100/launch.json").read_text())
        run_name = launch["command"][launch["command"].index("--run_name") + 1]
        matches = list(EXPERIMENT_ROOT.glob(f"*_{run_name}"))
        if matches != [E100_CHECKPOINT.parent]:
            raise RuntimeError(f"completed E100 run identity mismatch: {matches}")
        _, wandb_manifest = self.wandb_files(100, SOURCE, 100)
        from tools.highstep_v15_imitation_gate import checkpoint_scope
        scope_output = STATE_ROOT / "stages/E100/checkpoint_scope.json"
        if not scope_output.exists():
            atomic_json(scope_output, checkpoint_scope(E100_CHECKPOINT, 100), read_only=True)
        train_log = STATE_ROOT / "stages/E100/train.log"
        atomic_json(training_result, {
            "schema_version": 1,
            "kind": "highstep_v171_adopted_completed_e100_training_result",
            "checkpoint": str(E100_CHECKPOINT),
            "checkpoint_sha256": E100_CHECKPOINT_SHA,
            "effective_updates": 100,
            "run_dir": str(E100_CHECKPOINT.parent),
            "train_log": str(train_log),
            "train_log_sha256": base.sha256_file(train_log),
            "checkpoint_scope": str(scope_output),
            "checkpoint_scope_sha256": base.sha256_file(scope_output),
            "wandb_stage_manifest": str(wandb_manifest),
            "source_v17_training_was_not_restarted": True,
            "adopted_at": base.now(),
        }, read_only=True)
        self.update(
            checkpoint=str(E100_CHECKPOINT), effective_updates=100,
            run_dir=str(E100_CHECKPOINT.parent), phase="E100_complete_adopted",
        )
        return E100_CHECKPOINT.parent, E100_CHECKPOINT, wandb_manifest

    def _e700_zero_scale_comparison(self, imitation: Mapping[str, Any]) -> dict[str, Any]:
        zero_path = ROOT / (
            "tmp/highstep_0707_exact_new_teacher_20260713/stages/E700/"
            "imitation/imitation_gate.json"
        )
        zero = json.loads(zero_path.read_text())
        phases = ("first_rear_top", "second_rear_top", "rear_hold")
        rows = []
        for phase in phases:
            baseline = zero["phase_metrics"][phase]
            candidate = imitation["phase_metrics"][phase]
            baseline_latent = float(baseline["latent_mae"])
            candidate_latent = float(candidate["latent_mae"])
            baseline_nonbox = sum(float(x) for x in baseline["joint_mae"][:12]) / 12.0
            candidate_nonbox = sum(float(x) for x in candidate["joint_mae"][:12]) / 12.0
            rows.append({
                "phase": phase,
                "zero_scale_latent_mae": baseline_latent,
                "historical_latent_mae": candidate_latent,
                "latent_ratio": candidate_latent / baseline_latent,
                "zero_scale_nonbox_action_mae": baseline_nonbox,
                "historical_nonbox_action_mae": candidate_nonbox,
                "nonbox_action_ratio": candidate_nonbox / baseline_nonbox,
                "both_improved_at_least_5pct": (
                    candidate_latent <= 0.95 * baseline_latent
                    and candidate_nonbox <= 0.95 * baseline_nonbox
                ),
            })
        aggregate_latent_ratio = (
            sum(row["historical_latent_mae"] for row in rows)
            / sum(row["zero_scale_latent_mae"] for row in rows)
        )
        aggregate_nonbox_ratio = (
            sum(row["historical_nonbox_action_mae"] for row in rows)
            / sum(row["zero_scale_nonbox_action_mae"] for row in rows)
        )
        improved_phase_count = sum(row["both_improved_at_least_5pct"] for row in rows)
        no_phase_regression_over_10pct = all(
            row["latent_ratio"] <= 1.10 and row["nonbox_action_ratio"] <= 1.10
            for row in rows
        )
        passed = bool(
            aggregate_latent_ratio <= 0.95
            and aggregate_nonbox_ratio <= 0.95
            and improved_phase_count >= 2
            and no_phase_regression_over_10pct
        )
        payload = {
            "schema_version": 1,
            "kind": "historical_0707_exact_E700_vs_zero_scale_E700",
            "workflow_id": WORKFLOW_ID,
            "zero_scale_imitation_gate": str(zero_path),
            "zero_scale_imitation_gate_sha256": base.sha256_file(zero_path),
            "criteria_preregistered_before_historical_E700": {
                "aggregate_latent_ratio_max": 0.95,
                "aggregate_nonbox_action_ratio_max": 0.95,
                "phases_with_both_metrics_improved_5pct_min": 2,
                "per_phase_metric_regression_ratio_max": 1.10,
            },
            "phase_rows": rows,
            "aggregate_latent_ratio": aggregate_latent_ratio,
            "aggregate_nonbox_action_ratio": aggregate_nonbox_ratio,
            "improved_phase_count": improved_phase_count,
            "no_phase_regression_over_10pct": no_phase_regression_over_10pct,
            "passed": passed,
            "failed_action": "stop_and_isolate_new_teacher_latent_observability",
            "completed_at": base.now(),
        }
        output = STATE_ROOT / "stages/E700/e700_zero_scale_comparison.json"
        if output.exists():
            if json.loads(output.read_text()) != payload or not base.is_read_only(output):
                raise RuntimeError("stored historical E700 comparison changed")
        else:
            atomic_json(output, payload, read_only=True)
        return payload

    def decision(
        self,
        stage: int,
        imitation: Mapping[str, Any],
        probe: Mapping[str, Any],
        core: Mapping[str, Any],
        directional: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = super().decision(stage, imitation, probe, core, directional)
        if stage < 700 and result["final_numeric_gate_passed"]:
            result.update({
                "early_numeric_gate_observed": True,
                "final_numeric_gate_passed": False,
                "stop": False,
                "classification": "behavior_success_pending_mandatory_E700_comparison",
                "reason": "continue_to_mandatory_E700_zero_scale_comparison",
            })
        if stage != 700:
            if stage == 1400 and not result["final_numeric_gate_passed"]:
                result.update({
                    "stop": True,
                    "terminal_status": "stopped_by_gate",
                    "classification": "behavior_not_successful_at_absolute_cap",
                    "reason": "absolute_1400_cap_without_stable_climb",
                })
            return result
        comparison = self._e700_zero_scale_comparison(imitation)
        result["e700_zero_scale_comparison"] = comparison
        if not comparison["passed"]:
            result.update({
                "stop": True,
                "terminal_status": "latent_observability_isolation_required",
                "classification": "historical_route_no_clear_E700_improvement",
                "reason": "E700_no_clear_latent_and_nonbox_improvement_vs_zero_scale",
                "E900_E1400_allowed": False,
            })
        return result


def main() -> int:
    supervisor = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("historical 0707 supervisor already owns the workflow lock")
        return 2
    except Exception as error:
        if supervisor is not None:
            failure_class = supervisor.classify_failure(
                str(supervisor.state.get("phase", "unknown")), str(error)
            )
            supervisor.record_terminal(
                "mechanism_error_requires_repair"
                if failure_class == "training_mechanism_error"
                else "infrastructure_error_requires_retry",
                f"{type(error).__name__}: {error}",
            )
            supervisor.update(failure_class=failure_class, last_error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
