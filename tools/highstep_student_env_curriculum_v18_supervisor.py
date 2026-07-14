#!/usr/bin/env python3
"""Autonomous two-stage Student environment curriculum supervisor (spec v1.8)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

try:
    from tools.highstep_v18_checkpoint_scope import checkpoint_scope_v18
    from tools.highstep_v18_core9_monitor import CANONICAL as CANONICAL_MONITOR
    from tools.highstep_v18_core9_monitor import CANONICAL_SHA as CANONICAL_MONITOR_SHA
    from tools.highstep_v18_core9_monitor import expected_rendered
except ModuleNotFoundError:  # Direct execution sets sys.path[0] to tools/.
    from highstep_v18_checkpoint_scope import checkpoint_scope_v18
    from highstep_v18_core9_monitor import CANONICAL as CANONICAL_MONITOR
    from highstep_v18_core9_monitor import CANONICAL_SHA as CANONICAL_MONITOR_SHA
    from highstep_v18_core9_monitor import expected_rendered


ROOT = Path("/home/lxq/Softwares/robot_lab")
MECHANICS_PATH = ROOT / "tools/highstep_historical_0707_exact_supervisor.py"
_spec = importlib.util.spec_from_file_location("highstep_v18_reused_mechanics", MECHANICS_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load highstep supervisor mechanics")
mechanics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mechanics)
legacy = mechanics.legacy
prior = mechanics.prior
base = mechanics.base

WORKFLOW_ID = "highstep_student_env_curriculum_v18_20260714"
STATE_ROOT = ROOT / "tmp/highstep_student_env_curriculum_v18_20260714"
SPEC_SHA = "e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4"
PREREG = Path(os.environ.get(
    "HIGHSTEP_V18_PREREGISTRATION_PATH", STATE_ROOT / "preregistration.json"
))
PREREG_SHA = os.environ.get("HIGHSTEP_V18_PREREGISTRATION_SHA256", "")
MEMORY = ROOT / (
    "docs/robotlab_memory_zh/library/"
    "highstep_v18_student_environment_curriculum_decision_20260714.md"
)
MEMORY_SHA = "605ded098845f0868405b1f0756d4aa3f76fb1134079f5975d3a536b8ec48821"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
SOURCE = ROOT / (
    "tmp/highstep_historical_0707_exact_new_teacher_20260714/"
    "source_model_172300/model_172300.pt"
)
SOURCE_SCHEDULE = SOURCE.with_name("highstep_schedule_manifest.json")
SOURCE_RUNTIME = SOURCE.with_name("highstep_runtime_state.json")
SOURCE_SCHEDULE_SHA = "fc8de4ab2b5c30999b451a2e96a70dc2daeed67e3798adbe052e6086504a2aea"
SOURCE_RUNTIME_SHA = "b0d413f2edd1d9cc7cec01467112027c0b580614c537b0382770777e260668e4"
OLD_STATE = ROOT / "tmp/highstep_historical_0707_exact_new_teacher_20260714/state.json"
OLD_HANDOFF = ROOT / "tmp/highstep_historical_0707_exact_new_teacher_20260714/handoff.json"
OLD_E700_DIAGNOSTIC = ROOT / (
    "tmp/highstep_historical_0707_exact_new_teacher_20260714/diagnostics/"
    "E700_frozen_training_buffer/diagnostic_manifest.json"
)
OLD_E700_DIAGNOSTIC_SHA = "47fc72bd803fa44b51170e405dfc32f57db89ba91e6190af1986d99ee4738e5d"

STAGE_A_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-"
    "ArcdogAdjustableLeg-v0"
)
STAGE_B_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Robust-"
    "ArcdogAdjustableLeg-v0"
)
STAGE_A_PROFILE = STATE_ROOT / "profiles/stage_a_0707_student_bootstrap.json"
STAGE_A_PROFILE_SHA = "39ef546bb271426e7d2cc878b5685a32ecbf510fadd9b97ea8b44ada826168b6"
STAGE_B_PROFILE = STATE_ROOT / "profiles/stage_b_current_robust.json"
STAGE_B_PROFILE_SHA = "d06123905d5f9f5da749bb5d6f447c4d354ccae0bfa9b3238a6ea01a87b743a6"
TEACHER_PROFILE = STATE_ROOT / "profiles/teacher_0707_bootstrap.json"
TEACHER_PROFILE_SHA = "102431cceeccb8456a1494289b8c6dffd29c5984d4aae6dfd0f1be70d1d2d1ee"
STAGE_A_ENV_SHA = "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636"
STAGE_B_ENV_SHA = "61d70655405a49ad8fd72377aed3e193b0d8435898ca1fd39316d08f1989a729"
STAGE_A_POINTS = (100, 300, 500, 700, 900, 1400)
STAGE_B_RELATIVE_POINTS = (100, 300, 500, 900, 1400, 1800, 2400)
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "environment_curriculum_v18_Student"
)
E100_RECOVERY_MANIFEST = Path(os.environ.get(
    "HIGHSTEP_V18_E100_RECOVERY_MANIFEST",
    STATE_ROOT / "failure_recovery/e100_checkpoint_recovery_manifest_v4.json",
))
E300_RECOVERY_MANIFEST = Path(os.environ.get(
    "HIGHSTEP_V18_E300_RECOVERY_MANIFEST",
    STATE_ROOT / "failure_recovery/e300_checkpoint_recovery_manifest_v1.json",
))
E100_CORE9_REUSE_MANIFEST = STATE_ROOT / "failure_recovery/e100_core9_readonly_reuse_manifest.json"
E100_CORE9_REUSE_MANIFEST_SHA = "e3e3465963cd636fb3af5adbdc60811fd2fa5792502c624ce4bce15ad22849c7"
V18_MONITOR = STATE_ROOT / "infrastructure/highstep_v18_core9_monitor_v3.sh"
V18_MONITOR_SHA = "642a2b28123950e7901eaee1271b83a7e6f6231fbf343178e7e54bbb58a46e47"

# Every inherited helper reads these module globals. Rebind them before a
# Supervisor instance is constructed; per-stage task/profile binding is done
# again immediately before each train/eval operation.
for module in (mechanics, legacy, prior, base):
    module.WORKFLOW_ID = WORKFLOW_ID
    module.STATE_ROOT = STATE_ROOT
    module.SPEC_SHA = SPEC_SHA
    module.PREREG = PREREG
    module.PREREG_SHA = PREREG_SHA
base.SOURCE = SOURCE
base.TEACHER = TEACHER
base.EXPERIMENT_ROOT = EXPERIMENT_ROOT
base.SAVE_POINTS = STAGE_A_POINTS
base.CORE9_POINTS = set(STAGE_A_POINTS)
base.TRAIN_RUN_PREFIX = "v18_stage_a"
base.PREREG_ENV_PREFIX = "HIGHSTEP_V18"
base.WARMUP_UPDATES = 1400
base.ABSOLUTE_CAP = 2500
base.MONITOR = V18_MONITOR
prior.SOURCE_E300 = SOURCE
prior.SOURCE_E300_SHA = TEACHER_SHA


def atomic_json(path: Path, payload: Mapping[str, Any], read_only: bool = False) -> None:
    base.atomic_json(path, payload, read_only=read_only)


class Supervisor(mechanics.Supervisor):
    """Fresh bootstrap, then a full-checkpoint environment-only transition."""

    def __init__(self) -> None:
        self.course_stage = "preflight"
        self.environment_profile = "none"
        self.environment_source_sha256 = "none"
        super().__init__()

    def update(self, **values: Any) -> None:
        values.setdefault("authority_version", "v1.8")
        values.setdefault("course_stage", self.course_stage)
        values.setdefault("environment_profile", self.environment_profile)
        values.setdefault("environment_source_sha256", self.environment_source_sha256)
        values.setdefault("failure_class", self.state.get("failure_class", "none"))
        base.Supervisor.update(self, **values)

    def wait(self, process: subprocess.Popen[bytes], root: Path, phase: str, stall: int) -> None:
        """Require a real child PID and persist an immutable launch witness."""
        if process.poll() is not None:
            raise RuntimeError(f"{phase} child exited before PID verification")
        status = (Path("/proc") / str(process.pid) / "status").read_text()
        parent = int(next(line for line in status.splitlines() if line.startswith("PPid:")).split()[1])
        group = os.getpgid(process.pid)
        if parent != os.getpid() or group != process.pid:
            raise RuntimeError(f"{phase} child ownership mismatch: ppid={parent}, pgid={group}")
        self.update(status="running", phase=phase, active_pid=process.pid, failure_class="none")
        witness = STATE_ROOT / "launch_verifications" / f"{phase}.json"
        if witness.exists():
            index = 2
            while witness.with_name(f"{witness.stem}_attempt{index}.json").exists():
                index += 1
            witness = witness.with_name(f"{witness.stem}_attempt{index}.json")
        atomic_json(witness, {
            "schema_version": 1,
            "kind": "highstep_v18_child_pid_verification",
            "workflow_id": WORKFLOW_ID,
            "phase": phase,
            "course_stage": self.course_stage,
            "supervisor_pid": os.getpid(),
            "child_pid": process.pid,
            "child_parent_pid": parent,
            "child_process_group": group,
            "environment_profile": self.environment_profile,
            "environment_source_sha256": self.environment_source_sha256,
            "verified_at": base.now(),
        }, read_only=True)
        base.Supervisor.wait(self, process, root, phase, stall)

    @staticmethod
    def classify_failure(phase: str, message: str) -> str:
        text = f"{phase} {message}".lower()
        if "wandb" in text or "sync" in text or "network" in text:
            return "wandb_infrastructure_retryable"
        if any(token in text for token in ("probe", "core9", "directional", "eval", "video")):
            return "evaluation_infrastructure_retryable"
        if any(token in text for token in (
            "authority", "binding", "checkpoint", "optimizer", "teacher", "preregistration",
            "spec", "schedule", "tensor", "contract", "profile", "critical code",
        )):
            return "training_mechanism_error"
        return "external_fault_requires_user"

    def _bind_stage(self, course_stage: str) -> None:
        if course_stage == "A":
            task, profile, digest, env_digest, prefix = (
                STAGE_A_TASK, STAGE_A_PROFILE, STAGE_A_PROFILE_SHA, STAGE_A_ENV_SHA, "v18_stage_a"
            )
        elif course_stage == "B":
            task, profile, digest, env_digest, prefix = (
                STAGE_B_TASK, STAGE_B_PROFILE, STAGE_B_PROFILE_SHA, STAGE_B_ENV_SHA, "v18_stage_b"
            )
        else:
            raise RuntimeError(f"unknown v1.8 course stage: {course_stage}")
        self.course_stage = course_stage
        self.environment_profile = str(profile)
        self.environment_source_sha256 = env_digest
        for module in (base, prior, legacy, mechanics):
            module.TRAIN_TASK = task
            module.EVAL_TASK = task
            module.EXPERIMENT_ROOT = EXPERIMENT_ROOT
        base.ENVIRONMENT_PROFILE_MANIFEST = str(profile)
        base.ENVIRONMENT_PROFILE_MANIFEST_SHA256 = digest
        base.TRAIN_RUN_PREFIX = prefix
        self.update()

    def preflight(self) -> None:
        if not PREREG_SHA:
            raise RuntimeError("v1.8 preregistration SHA environment is missing")
        required = {
            base.SPEC: SPEC_SHA,
            PREREG: PREREG_SHA,
            MEMORY: MEMORY_SHA,
            TEACHER: TEACHER_SHA,
            SOURCE: TEACHER_SHA,
            SOURCE_SCHEDULE: SOURCE_SCHEDULE_SHA,
            SOURCE_RUNTIME: SOURCE_RUNTIME_SHA,
            STAGE_A_PROFILE: STAGE_A_PROFILE_SHA,
            STAGE_B_PROFILE: STAGE_B_PROFILE_SHA,
            TEACHER_PROFILE: TEACHER_PROFILE_SHA,
            OLD_E700_DIAGNOSTIC: OLD_E700_DIAGNOSTIC_SHA,
            CANONICAL_MONITOR: CANONICAL_MONITOR_SHA,
            V18_MONITOR: V18_MONITOR_SHA,
            E100_CORE9_REUSE_MANIFEST: E100_CORE9_REUSE_MANIFEST_SHA,
        }
        for path, expected in required.items():
            if base.sha256_file(path) != expected:
                raise RuntimeError(f"v1.8 authority SHA mismatch: {path}")
        if V18_MONITOR.read_text() != expected_rendered() or V18_MONITOR.stat().st_mode & 0o222:
            raise RuntimeError("v1.8 core9 monitor is not the exact read-only canonical derivation")
        prereg = json.loads(PREREG.read_text())
        base_prereg_path = prereg.get("base_preregistration")
        if isinstance(base_prereg_path, str):
            inherited_path = Path(base_prereg_path).resolve(strict=True)
            if base.sha256_file(inherited_path) != prereg.get("base_preregistration_sha256"):
                raise RuntimeError("v1.8 inherited preregistration SHA mismatch")
            inherited = json.loads(inherited_path.read_text())
            merged_code = dict(inherited.get("code_sha256", {}))
            merged_code.update(prereg.get("code_sha256", {}))
            merged_code.update(prereg.get("code_sha256_overrides", {}))
            prereg = {**inherited, **prereg, "code_sha256": merged_code}
        if not (
            prereg.get("kind") == "highstep_student_environment_curriculum_v18_preregistration"
            and prereg.get("authority", {}).get("version") == "v1.8"
            and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA
            and prereg.get("teacher", {}).get("checkpoint_sha256") == TEACHER_SHA
            and prereg.get("environment_schedule", {}).get("stage_a", {}).get("source_env_yaml_sha256") == STAGE_A_ENV_SHA
            and prereg.get("environment_schedule", {}).get("stage_b", {}).get("source_env_yaml_sha256") == STAGE_B_ENV_SHA
        ):
            raise RuntimeError("v1.8 preregistration identity changed")
        old_state = json.loads(OLD_STATE.read_text())
        old_handoff = json.loads(OLD_HANDOFF.read_text())
        if not (
            old_state.get("workflow_id") == "highstep_historical_0707_exact_new_teacher_20260714"
            and old_state.get("effective_updates") == 700
            and old_state.get("status") in {"latent_observability_isolation_required", "stopped_by_gate"}
            and old_handoff.get("workflow_id") == old_state.get("workflow_id")
        ):
            raise RuntimeError("v1.7.1 E700 historical failure boundary changed")
        self.require_idle()
        teacher_preflight = STATE_ROOT / "preflight/teacher_bootstrap_preflight.json"
        teacher_evidence = json.loads(teacher_preflight.read_text())
        if not (
            teacher_evidence.get("passed") is True
            and teacher_evidence.get("checkpoint_sha256") == TEACHER_SHA
            and teacher_evidence.get("full_climb_count") >= 2
            and teacher_evidence.get("rear_hold_count") >= 2
            and all(row.get("schedule_valid") is True for row in teacher_evidence.get("rows", []))
            and base.is_read_only(teacher_preflight)
        ):
            raise RuntimeError("v1.8 Teacher bootstrap preflight is missing or invalid")
        smoke_path = STATE_ROOT / "smoke/smoke_manifest.json"
        smoke = json.loads(smoke_path.read_text())
        if not (
            smoke.get("passed") is True
            and smoke.get("preregistration_sha256")
            == prereg.get("smoke_source_preregistration_sha256", PREREG_SHA)
            and smoke.get("start_checkpoint_sha256") == TEACHER_SHA
            and smoke.get("end_effective_updates") in (1, 2, 3, 4, 5)
            and smoke.get("environment_source_sha256") == STAGE_A_ENV_SHA
            and smoke.get("warmup_updates") == 1400
            and smoke.get("warmup_main_target") == "teacher_pre_prior"
            and smoke.get("phase_scale") == 2.0
            and smoke.get("rear_box_scale") == 1.5
            and smoke.get("warmup_prior_box_loss_zero") is True
            and smoke.get("actor_unchanged") is True
            and smoke.get("teacher_student_storage_independent") is True
            and smoke.get("full_checkpoint_restore_verified") is True
            and base.is_read_only(smoke_path)
        ):
            raise RuntimeError("v1.8 1-5 update smoke is missing or invalid")
        critical = [Path(name) for name in prereg.get("code_sha256", {})]
        self.critical_hashes = {
            str(path.resolve()): base.sha256_file(path.resolve()) for path in critical
        }
        if self.critical_hashes != prereg.get("code_sha256"):
            raise RuntimeError("v1.8 critical code differs from preregistration")
        payload = {
            "schema_version": 1,
            "kind": "highstep_student_environment_curriculum_v18_preflight",
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "teacher_sha256": TEACHER_SHA,
            "stage_a_environment_sha256": STAGE_A_ENV_SHA,
            "stage_b_environment_sha256": STAGE_B_ENV_SHA,
            "fresh_student_from_teacher": True,
            "v171_optimizer_resume_allowed": False,
            "critical_code_sha256": self.critical_hashes,
            "teacher_preflight_sha256": base.sha256_file(teacher_preflight),
            "smoke_manifest_sha256": base.sha256_file(smoke_path),
            "completed_at": base.now(),
        }
        output = STATE_ROOT / "preflight" / f"preflight_{PREREG_SHA[:12]}_manifest.json"
        if output.exists():
            old = json.loads(output.read_text())
            if ({k: v for k, v in old.items() if k != "completed_at"}
                    != {k: v for k, v in payload.items() if k != "completed_at"}):
                raise RuntimeError("stored v1.8 preflight changed")
        else:
            atomic_json(output, payload, read_only=True)
        self.update(
            status="preflight_passed", phase="ready_for_stage_A_E100",
            effective_updates=0, checkpoint=str(TEACHER), stop_reason=None,
            last_error=None, failure_class="none",
        )

    def wandb_files(self, stage: int, source: Path, additional: int):
        course = self.course_stage
        config_path = STATE_ROOT / "wandb_stage_configs" / f"Stage{course}_E{stage}_v18_2.json"
        run_name = (
            f"{WORKFLOW_ID}_student_environment_curriculum_"
            f"Stage{course}_E{stage}_attempt1"
        )
        if not config_path.exists():
            base.atomic_json(config_path, {
                "workflow_id": WORKFLOW_ID,
                "route": "student_environment_curriculum",
                "course_stage": course,
                "stage": f"E{stage}",
                "attempt": "attempt1",
                "run_name": run_name,
                "group": WORKFLOW_ID,
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "start_checkpoint": str(source.resolve()),
                "start_checkpoint_sha256": base.sha256_file(source),
                "teacher_checkpoint": str(TEACHER),
                "teacher_sha256": TEACHER_SHA,
                "environment_profile": self.environment_profile,
                "environment_source_sha256": self.environment_source_sha256,
                "frozen_tensors": [
                    "teacher_actor.*", "teacher_priv_encoder.*", "critic.*", "priv_encoder.*",
                    "actor.* through update 1400", "actor body and rows 0:12 permanently",
                ],
                "trainable_tensors": ["estimator.*"] if stage <= 1400 else [
                    "estimator.*", "actor.6.weight[12:16]", "actor.6.bias[12:16]",
                ],
                "optimizer": {
                    "class": "Adam", "estimator_lr": 0.001, "box_rows_lr": 0.00001,
                    "ppo": "permanently_disabled",
                },
                "budget": {
                    "additional_updates": additional, "effective_update_target": stage,
                    "absolute_cap": 2500,
                },
                "training_task": base.TRAIN_TASK,
                "historical_sync": False,
            }, read_only=True)
        manifest = STATE_ROOT / "wandb_stages" / run_name / "stage_manifest.json"
        if not manifest.exists():
            manifest = base.wandb_gate.prepare_stage(config_path, STATE_ROOT)
        return config_path, manifest

    def train_stage(self, stage: int, source: Path, previous: int, course_stage: str):
        self._bind_stage(course_stage)
        base.V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE = ""
        base.V18_STAGE_TRANSITION_MANIFEST = ""
        base.V18_STAGE_TRANSITION_MANIFEST_SHA256 = ""
        if course_stage == "A":
            base.SCHEDULE_RESUME_MODE = "reset" if previous == 0 else "preserve"
        else:
            previous_result = STATE_ROOT / "stages" / f"E{previous}" / "stage_result.json"
            previous_course = json.loads(previous_result.read_text()).get("course_stage")
            if previous_course == "A":
                transition = STATE_ROOT / "stage_transition/stage_a_to_b_diff_manifest.json"
                base.SCHEDULE_RESUME_MODE = "reset"
                base.V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE = str(previous)
                base.V18_STAGE_TRANSITION_MANIFEST = str(transition)
                base.V18_STAGE_TRANSITION_MANIFEST_SHA256 = base.sha256_file(transition)
            else:
                base.SCHEDULE_RESUME_MODE = "preserve"
        if course_stage == "A" and stage == 100 and previous == 0 and E100_RECOVERY_MANIFEST.exists():
            recovery = json.loads(E100_RECOVERY_MANIFEST.read_text())
            checkpoint = Path(str(recovery.get("selected_checkpoint", ""))).resolve(strict=True)
            run_dir = Path(str(recovery.get("selected_run_dir", ""))).resolve(strict=True)
            rebinding = Path(str(recovery.get("authority_rebinding_manifest", ""))).resolve(strict=True)
            if not (
                recovery.get("kind") == "highstep_v18_e100_infrastructure_recovery"
                and recovery.get("workflow_id") == WORKFLOW_ID
                and recovery.get("selected_checkpoint_sha256") == base.sha256_file(checkpoint)
                and recovery.get("effective_updates") == 100
                and recovery.get("repeat_training_forbidden") is True
                and recovery.get("changed_training_semantics") is False
                and base.is_read_only(E100_RECOVERY_MANIFEST)
                and base.is_read_only(rebinding)
            ):
                raise RuntimeError("v1.8 E100 recovery manifest changed")
            scope = checkpoint_scope_v18(
                checkpoint, 100, preregistration=PREREG,
                preregistration_sha256=PREREG_SHA, run_dir=run_dir,
                authority_rebinding=rebinding,
            )
            stage_root = STATE_ROOT / "stages/E100"
            authority_suffix = PREREG_SHA[:12]
            scope_output = stage_root / f"checkpoint_scope_v18_recovered_{authority_suffix}.json"
            if not scope_output.exists():
                atomic_json(scope_output, scope, read_only=True)
            elif json.loads(scope_output.read_text()) != scope:
                raise RuntimeError("stored recovered v1.8 checkpoint scope changed")
            wandb_manifest = Path(str(recovery.get("wandb_stage_manifest", ""))).resolve(strict=True)
            training_result = stage_root / f"training_result_{authority_suffix}.json"
            payload = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": base.sha256_file(checkpoint),
                "effective_updates": 100,
                "run_dir": str(run_dir),
                "train_log": recovery["selected_train_log"],
                "train_log_sha256": recovery["selected_train_log_sha256"],
                "checkpoint_scope": str(scope_output),
                "checkpoint_scope_sha256": base.sha256_file(scope_output),
                "wandb_stage_manifest": str(wandb_manifest),
                "recovered_after_infrastructure_failure": True,
                "repeated_e100_training_skipped": True,
                "completed_at": recovery["selected_checkpoint_completed_at"],
            }
            if not training_result.exists():
                atomic_json(training_result, payload, read_only=True)
            elif json.loads(training_result.read_text()) != payload:
                raise RuntimeError("stored recovered v1.8 E100 training result changed")
            self.update(
                checkpoint=str(checkpoint), effective_updates=100,
                run_dir=str(run_dir), failure_class="none", last_error=None,
            )
            return run_dir, checkpoint, wandb_manifest
        if course_stage == "A" and stage == 300 and previous == 100 and E300_RECOVERY_MANIFEST.exists():
            recovery = json.loads(E300_RECOVERY_MANIFEST.read_text())
            checkpoint = Path(str(recovery.get("selected_checkpoint", ""))).resolve(strict=True)
            run_dir = Path(str(recovery.get("selected_run_dir", ""))).resolve(strict=True)
            train_log = Path(str(recovery.get("selected_train_log", ""))).resolve(strict=True)
            rebinding = Path(str(recovery.get("authority_rebinding_manifest", ""))).resolve(strict=True)
            interrupted = Path(str(recovery.get("interrupted_run_forbidden", ""))).resolve(strict=True)
            if not (
                recovery.get("kind") == "highstep_v18_e300_infrastructure_recovery"
                and recovery.get("workflow_id") == WORKFLOW_ID
                and recovery.get("effective_updates") == 300
                and recovery.get("previous_effective_updates") == 100
                and recovery.get("selected_checkpoint_sha256") == base.sha256_file(checkpoint)
                and recovery.get("selected_train_log_sha256") == base.sha256_file(train_log)
                and recovery.get("repeat_e300_training_forbidden") is True
                and recovery.get("changed_training_semantics") is False
                and checkpoint.parent == run_dir
                and interrupted != run_dir
                and base.is_read_only(E300_RECOVERY_MANIFEST)
                and base.is_read_only(rebinding)
            ):
                raise RuntimeError("v1.8 E300 recovery manifest changed")
            scope = checkpoint_scope_v18(
                checkpoint, 300, preregistration=PREREG,
                preregistration_sha256=PREREG_SHA, run_dir=run_dir,
                authority_rebinding=rebinding,
            )
            stage_root = STATE_ROOT / "stages/E300"
            authority_suffix = PREREG_SHA[:12]
            scope_output = stage_root / f"checkpoint_scope_v18_recovered_{authority_suffix}.json"
            if not scope_output.exists():
                atomic_json(scope_output, scope, read_only=True)
            elif json.loads(scope_output.read_text()) != scope:
                raise RuntimeError("stored recovered v1.8 E300 checkpoint scope changed")
            wandb_manifest = Path(str(recovery.get("wandb_stage_manifest", ""))).resolve(strict=True)
            training_result = stage_root / f"training_result_{authority_suffix}.json"
            payload = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": base.sha256_file(checkpoint),
                "effective_updates": 300,
                "run_dir": str(run_dir),
                "train_log": str(train_log),
                "train_log_sha256": base.sha256_file(train_log),
                "checkpoint_scope": str(scope_output),
                "checkpoint_scope_sha256": base.sha256_file(scope_output),
                "wandb_stage_manifest": str(wandb_manifest),
                "recovered_after_posttrain_scope_failure": True,
                "repeated_e300_training_skipped": True,
                "completed_at": recovery["selected_checkpoint_completed_at"],
            }
            if not training_result.exists():
                atomic_json(training_result, payload, read_only=True)
            elif json.loads(training_result.read_text()) != payload:
                raise RuntimeError("stored recovered v1.8 E300 training result changed")
            self.update(
                checkpoint=str(checkpoint), effective_updates=300,
                run_dir=str(run_dir), failure_class="none", last_error=None,
            )
            return run_dir, checkpoint, wandb_manifest
        return base.Supervisor.train_stage(self, stage, source, previous)

    def checkpoint_scope(self, checkpoint: Path, stage: int, run_dir: Path) -> dict[str, Any]:
        return checkpoint_scope_v18(
            checkpoint, stage, preregistration=PREREG,
            preregistration_sha256=PREREG_SHA, run_dir=run_dir,
        )

    def probe(self, stage: int, checkpoint: Path):
        return base.Supervisor.probe(self, stage, checkpoint)

    def core9(self, stage: int, checkpoint: Path, run_dir: Path):
        previous = os.environ.get("HIGHSTEP_V18_CORE9_REUSE_MANIFEST")
        if stage == 100:
            os.environ["HIGHSTEP_V18_CORE9_REUSE_MANIFEST"] = str(E100_CORE9_REUSE_MANIFEST)
        try:
            return base.Supervisor.core9(self, stage, checkpoint, run_dir)
        finally:
            if previous is None:
                os.environ.pop("HIGHSTEP_V18_CORE9_REUSE_MANIFEST", None)
            else:
                os.environ["HIGHSTEP_V18_CORE9_REUSE_MANIFEST"] = previous

    def directional(self, stage: int, checkpoint: Path, core: Mapping[str, Any]):
        return prior.Supervisor.directional(self, stage, checkpoint, core)

    def diagnostic_record(self, stage: int, checkpoint: Path) -> dict[str, Any]:
        """Behavior is authoritative; same-state imitation remains diagnostic-only."""
        path = STATE_ROOT / "stages" / f"E{stage}" / "same_state_diagnostic_status.json"
        if not path.exists():
            atomic_json(path, {
                "schema_version": 1,
                "kind": "highstep_v18_same_state_diagnostic_status",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": base.sha256_file(checkpoint),
                "passed": False,
                "failed_element_count": 0,
                "phase_metrics": {},
                "classification": "not_a_behavior_gate",
                "candidate_driven_behavior_is_authoritative": True,
            }, read_only=True)
        return json.loads(path.read_text())

    def _stage_b_targets(self, stage_a_pass: int) -> list[int]:
        targets = sorted({stage_a_pass + delta for delta in STAGE_B_RELATIVE_POINTS
                          if stage_a_pass + delta <= 2500})
        if not targets or targets[-1] < 2500:
            targets.append(2500)
        return targets

    def _transition_manifest(self, checkpoint: Path, stage_a_pass: int) -> Path:
        output = STATE_ROOT / "stage_transition/stage_a_to_b_diff_manifest.json"
        payload = {
            "schema_version": 1,
            "kind": "highstep_v18_environment_only_stage_transition",
            "workflow_id": WORKFLOW_ID,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": base.sha256_file(checkpoint),
            "source_effective_updates": stage_a_pass,
            "resume_mode": "full_checkpoint_with_original_optimizer",
            "effective_updates_reset": False,
            "only_changed_training_mechanism": "student_environment_profile",
            "from": {"task": STAGE_A_TASK, "profile_sha256": STAGE_A_PROFILE_SHA, "env_sha256": STAGE_A_ENV_SHA},
            "to": {"task": STAGE_B_TASK, "profile_sha256": STAGE_B_PROFILE_SHA, "env_sha256": STAGE_B_ENV_SHA},
            "unchanged": [
                "Teacher", "Student tensors", "optimizer", "effective update", "warmup=1400",
                "pre-prior action target", "phase_scale=2.0", "rear_box_scale=1.5",
                "warmup prior-box loss=0", "losses", "learning rates", "freeze scope",
                "network", "reward", "action_scale", "joint_pos.clip", "action contract",
            ],
            "created_at": base.now(),
        }
        if output.exists():
            old = json.loads(output.read_text())
            if ({k: v for k, v in old.items() if k != "created_at"}
                    != {k: v for k, v in payload.items() if k != "created_at"}):
                raise RuntimeError("v1.8 Stage A/B transition manifest changed")
        else:
            atomic_json(output, payload, read_only=True)
        return output

    def _result(self, course: str, stage: int, checkpoint: Path, run_dir: Path,
                wandb_manifest: Path) -> dict[str, Any]:
        self._bind_stage(course)
        diagnostic = self.diagnostic_record(stage, checkpoint)
        probe = self.probe(stage, checkpoint)
        core = self.core9(stage, checkpoint, run_dir)
        directional = None if course == "A" else self.directional(stage, checkpoint, core)
        counts = core["counts"]
        if course == "A":
            passed = bool(
                counts["valid"] == 9 and counts["full_climb"] >= 7
                and counts["rear_hold"] >= 7 and counts["no_severe_inward"] >= 8
            )
            reason = "stage_a_bootstrap_gate_passed" if passed else "continue_stage_a"
            decision = {
                "classification": "bootstrap_behavior_success" if passed else "bootstrap_behavior_not_yet_successful",
                "final_numeric_gate_passed": passed,
                "stop": passed or stage >= 1400,
                "reason": reason if passed else (
                    "stage_a_E1400_cap_without_bootstrap_gate" if stage >= 1400 else reason
                ),
                "passed_sides": [],
                "confirmed_safety_risk": counts["no_severe_inward"] < 8,
            }
            direction_score = 0
        else:
            passed_sides = [
                side for side, value in directional["corridors"].items()
                if value["stable_behavior_passed"]
                and value["counts"]["no_severe_inward"] >= 8
            ]
            passed = bool(passed_sides)
            decision = {
                "classification": "directional_behavior_success" if passed else "robust_behavior_not_yet_successful",
                "final_numeric_gate_passed": passed,
                "stop": passed or stage >= 2500,
                "reason": "stable_climb_pending_user_visual_review" if passed else (
                    "absolute_2500_cap_without_stable_climb" if stage >= 2500 else "continue_stage_b"
                ),
                "passed_sides": passed_sides,
                "confirmed_safety_risk": False,
            }
            direction_score = max(
                min(value["counts"]["full_climb"], value["counts"]["rear_hold"])
                for value in directional["corridors"].values()
            )
        rank = [int(passed), direction_score, counts["full_climb"] + counts["rear_hold"],
                probe["behavior_score"], counts["no_severe_inward"], -stage]
        return {
            "schema_version": 1,
            "kind": "highstep_student_environment_curriculum_v18_stage_result",
            "course_stage": course,
            "stage": stage,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": base.sha256_file(checkpoint),
            "same_state_diagnostic": diagnostic,
            "probe": probe,
            "core9": core,
            "directional": directional,
            "decision": decision,
            "selection_rank": rank,
            "wandb_stage_manifest": str(wandb_manifest),
            "environment_profile": self.environment_profile,
            "environment_source_sha256": self.environment_source_sha256,
            "behavior_evidence_is_candidate_driven": True,
            "completed_at": base.now(),
        }

    @staticmethod
    def wandb_gate_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "course_stage": result["course_stage"],
            "probe": result["probe"]["counts"],
            "core9": result["core9"]["counts"],
            "directional": None if result["directional"] is None else {
                side: result["directional"]["corridors"][side]["counts"] for side in ("left", "right")
            },
            "classification": result["decision"]["classification"],
            "environment_source_sha256": result["environment_source_sha256"],
        }

    def _run_point(self, course: str, stage: int, source: Path, previous: int) -> dict[str, Any]:
        self._bind_stage(course)
        result_path = STATE_ROOT / "stages" / f"E{stage}" / "stage_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            checkpoint = Path(result["checkpoint"]).resolve(strict=True)
            if result.get("course_stage") != course or result["checkpoint_sha256"] != base.sha256_file(checkpoint):
                raise RuntimeError(f"stored v1.8 E{stage} result changed")
            wandb_manifest = Path(result["wandb_stage_manifest"])
        else:
            run_dir, checkpoint, wandb_manifest = self.train_stage(stage, source, previous, course)
            result = self._result(course, stage, checkpoint, run_dir, wandb_manifest)
            atomic_json(result_path, result, read_only=True)
        if (json.loads(wandb_manifest.read_text()).get("sync_status") != "synced"
                or not self.wandb_offline_coverage_complete(wandb_manifest)):
            self.finalize_wandb(
                stage, Path(result["checkpoint"]), result_path, wandb_manifest,
                self.wandb_gate_metrics(result), result["decision"]["reason"],
            )
        self.update(
            checkpoint=result["checkpoint"], effective_updates=stage,
            behavior_probe=result["probe"]["counts"], core9=result["core9"]["counts"],
            directional=None if result["directional"] is None else {
                side: result["directional"]["corridors"][side]["counts"] for side in ("left", "right")
            },
            classification=result["decision"]["classification"], decision=result["decision"],
        )
        return result

    def run(self) -> None:
        self.preflight()
        source, previous = SOURCE, 0
        stage_a_result = None
        for stage in STAGE_A_POINTS:
            self.assert_code()
            result = self._run_point("A", stage, source, previous)
            source, previous = Path(result["checkpoint"]), stage
            if result["decision"]["final_numeric_gate_passed"]:
                stage_a_result = result
                break
            if stage == 1400:
                self.record_terminal("stopped_by_gate", result["decision"]["reason"], result)
                return
        if stage_a_result is None:
            raise RuntimeError("Stage A ended without a deterministic gate decision")
        transition = self._transition_manifest(source, previous)
        self.update(
            status="stage_a_passed", phase="stage_a_to_b_transition_complete",
            stage_transition_manifest=str(transition),
            stage_transition_manifest_sha256=base.sha256_file(transition),
        )
        best = stage_a_result
        for stage in self._stage_b_targets(previous):
            self.assert_code()
            result = self._run_point("B", stage, source, previous)
            source, previous = Path(result["checkpoint"]), stage
            if tuple(result["selection_rank"]) > tuple(best["selection_rank"]):
                best = result
            if result["decision"]["final_numeric_gate_passed"]:
                side = result["decision"]["passed_sides"][0]
                self._bind_stage("B")
                video_manifest = prior.Supervisor.videos(self, stage, source, side)
                self.update(
                    video_manifest=str(video_manifest),
                    video_manifest_sha256=base.sha256_file(video_manifest),
                )
                self.record_terminal(
                    "student_directional_candidate_pending_user_visual_review",
                    result["decision"]["reason"], best,
                )
                return
            if stage >= 2500:
                self.record_terminal("stopped_by_gate", result["decision"]["reason"], best)
                return


def main() -> int:
    supervisor: Supervisor | None = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("v1.8 supervisor already owns the workflow lock", file=sys.stderr)
        return 2
    except Exception as error:
        if supervisor is not None:
            failure_class = supervisor.classify_failure(
                str(supervisor.state.get("phase", "unknown")), str(error)
            )
            status = (
                "mechanism_error_requires_repair"
                if failure_class == "training_mechanism_error"
                else "infrastructure_error_requires_retry"
            )
            supervisor.record_terminal(status, f"{type(error).__name__}: {error}")
            supervisor.update(failure_class=failure_class, last_error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
