#!/usr/bin/env python3
"""Autonomous supervisor for the spec-v1.6 0707_exact_new_teacher route."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


ROOT = Path("/home/lxq/Softwares/robot_lab")
BASE_PATH = ROOT / "tools/highstep_student_recovery_v152_supervisor.py"
_spec = importlib.util.spec_from_file_location("highstep_v152_base_for_0707_exact", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load v1.5.2 supervisor base")
prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior)
base = prior.base

WORKFLOW_ID = "highstep_0707_exact_new_teacher_20260713"
STATE_ROOT = ROOT / "tmp/highstep_0707_exact_new_teacher_20260713"
SPEC_SHA = "4baed191f98f9b746eec9181b3f31bcdd16e3cc147726b676d9949d7e1fe4425"
PREREG = STATE_ROOT / "preregistration.json"
PREREG_SHA = os.environ.get("HIGHSTEP_0707_EXACT_PREREGISTRATION_SHA256", "")
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
SOURCE = STATE_ROOT / "source_model_172300/model_172300.pt"
SOURCE_SCHEDULE = STATE_ROOT / "source_model_172300/highstep_schedule_manifest.json"
SOURCE_RUNTIME = STATE_ROOT / "source_model_172300/highstep_runtime_state.json"
SOURCE_SCHEDULE_SHA = "01d5e63351426ca8d955492665a891419e8368c7e6b3d3e408dc1c08bb109b8c"
SOURCE_RUNTIME_SHA = "b0d413f2edd1d9cc7cec01467112027c0b580614c537b0382770777e260668e4"
SAVE_POINTS = (100, 300, 500, 700, 900, 1200, 1400, 1800, 2500)
CORE9_POINTS = set(SAVE_POINTS)
TRAIN_TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior0707Exact-ArcdogAdjustableLeg-v0"
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_0707_exact_new_teacher_Student"
)
OLD_STATE = ROOT / "tmp/highstep_student_recovery_v152_20260713/state.json"
ARCHIVE = STATE_ROOT / "old_route_archive_manifest.json"
INFRA_REBINDING_TEXT = os.environ.get("HIGHSTEP_0707_EXACT_INFRA_REBINDING_PATH", "")
INFRA_REBINDING = Path(INFRA_REBINDING_TEXT) if INFRA_REBINDING_TEXT else None
INFRA_REBINDING_SHA = os.environ.get("HIGHSTEP_0707_EXACT_INFRA_REBINDING_SHA256", "")
PREVIOUS_WANDB_GATE_SHA = "be7a2ab4fa8a2d400c0f8ec82b92c8a8f0bf9d93771eef6d890965caf87e9b15"
CANONICAL_HELPER_SHA = "771cad7496fbe1d31933b222d94881c2e8ed0054c70262d66cd682c91a255e51"

# Rebind every authority-bearing global used by the inherited mechanics.
prior.WORKFLOW_ID = WORKFLOW_ID
prior.STATE_ROOT = STATE_ROOT
prior.SPEC_SHA = SPEC_SHA
prior.PREREG = PREREG
prior.PREREG_SHA = PREREG_SHA
prior.SOURCE_E300 = SOURCE
prior.SOURCE_E300_SHA = TEACHER_SHA
prior.SAVE_POINTS = SAVE_POINTS
prior.CORE9_POINTS = CORE9_POINTS
base.WORKFLOW_ID = WORKFLOW_ID
base.STATE_ROOT = STATE_ROOT
base.SPEC_SHA = SPEC_SHA
base.PREREG = PREREG
base.PREREG_SHA = PREREG_SHA
base.SOURCE = SOURCE
base.TEACHER = TEACHER
base.TRAIN_TASK = TRAIN_TASK
base.EXPERIMENT_ROOT = EXPERIMENT_ROOT
base.SAVE_POINTS = SAVE_POINTS
base.CORE9_POINTS = CORE9_POINTS
base.TRAIN_RUN_PREFIX = "0707_exact"
base.PREREG_ENV_PREFIX = "HIGHSTEP_0707_EXACT"
base.WARMUP_UPDATES = 1200
base.ABSOLUTE_CAP = 2500


def atomic_json(path: Path, payload: Mapping[str, Any], read_only: bool = False) -> None:
    base.atomic_json(path, payload, read_only=read_only)


def validate_infrastructure_rebinding(
    path: Path | None,
    expected_sha256: str,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize only the approved W&B multi-fragment infrastructure repair."""
    if path is None or not expected_sha256:
        raise RuntimeError("0707 exact infrastructure rebinding authority is missing")
    path = path.resolve(strict=True)
    if base.sha256_file(path) != expected_sha256 or not base.is_read_only(path):
        raise RuntimeError("0707 exact infrastructure rebinding manifest changed or is writable")
    payload = json.loads(path.read_text())
    if not (
        payload.get("schema_version") == 1
        and payload.get("kind") == "highstep_0707_exact_wandb_multifragment_rebinding"
        and payload.get("workflow_id") == WORKFLOW_ID
        and payload.get("spec_sha256") == SPEC_SHA
        and payload.get("preregistration_sha256") == PREREG_SHA
    ):
        raise RuntimeError("0707 exact infrastructure rebinding identity changed")
    scope = payload.get("scope", {})
    protected = (
        "spec_changed", "teacher_changed", "loss_changed", "warmup_changed",
        "freeze_scope_changed", "optimizer_changed", "learning_rate_changed",
        "schedule_changed", "behavior_gate_changed", "training_semantics_changed",
    )
    if scope.get("wandb_multifragment_sync_only") is not True or any(
        scope.get(name) is not False for name in protected
    ):
        raise RuntimeError("0707 exact infrastructure rebinding exceeds approved scope")
    recovery = payload.get("recovery_checkpoint", {})
    if not (
        recovery.get("path") == str((EXPERIMENT_ROOT / "2026-07-14_00-21-31_0707_exact_E300_shutdown_resume_20260714_002124/model_297.pt").resolve())
        and recovery.get("sha256") == "ddb068c9be49a618fe3792578b58c8e77d11f14f75d5d0002eb1bed5c650d971"
        and recovery.get("effective_updates") == 300
        and recovery.get("interrupted_e500_checkpoint_allowed") is False
    ):
        raise RuntimeError("0707 exact infrastructure recovery checkpoint binding changed")
    changes = {
        str(item.get("path")): item for item in payload.get("code_changes", [])
        if isinstance(item, Mapping) and item.get("path")
    }
    preregistered = preregistration.get("code_sha256", {})
    permitted = {
        str((ROOT / "tools/highstep_student_recovery_v15_supervisor.py").resolve()),
        str(Path(__file__).resolve()),
    }
    declared = set(payload.get("preregistered_authority_changes", []))
    if declared != permitted:
        raise RuntimeError("0707 exact preregistered authority change set widened")
    for name in permitted:
        item = changes.get(name, {})
        if item.get("old_sha256") != preregistered.get(name):
            raise RuntimeError(f"0707 exact old code authority mismatch: {name}")
        if item.get("new_sha256") != base.sha256_file(Path(name)):
            raise RuntimeError(f"0707 exact rebound code SHA mismatch: {name}")
    gate_name = str((ROOT / "tools/highstep_wandb_stage_gate.py").resolve())
    gate_change = changes.get(gate_name, {})
    if not (
        gate_change.get("old_sha256") == PREVIOUS_WANDB_GATE_SHA
        and gate_change.get("new_sha256") == base.sha256_file(Path(gate_name))
    ):
        raise RuntimeError("0707 exact W&B gate rebinding changed")
    helper = payload.get("unchanged_authority", {}).get("canonical_task_helper", {})
    if not (
        helper.get("sha256") == CANONICAL_HELPER_SHA
        and base.sha256_file(Path(str(helper.get("path")))) == CANONICAL_HELPER_SHA
    ):
        raise RuntimeError("0707 exact canonical helper authority changed")
    return payload


class Supervisor(prior.Supervisor):
    """Exact-0707 authority with behavior-first directional delivery."""

    def preflight(self) -> None:
        if not PREREG_SHA:
            raise RuntimeError("0707 exact preregistration SHA environment is missing")
        for path, expected in (
            (base.SPEC, SPEC_SHA), (PREREG, PREREG_SHA), (TEACHER, TEACHER_SHA),
            (SOURCE, TEACHER_SHA), (SOURCE_SCHEDULE, SOURCE_SCHEDULE_SHA),
            (SOURCE_RUNTIME, SOURCE_RUNTIME_SHA),
        ):
            if base.sha256_file(path) != expected:
                raise RuntimeError(f"0707 exact authority SHA mismatch: {path}")
        prereg = json.loads(PREREG.read_text())
        if not (
            prereg.get("kind") == "highstep_0707_exact_new_teacher_preregistration"
            and prereg.get("authority", {}).get("version") == "v1.6.1"
            and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA
        ):
            raise RuntimeError("0707 exact preregistration identity changed")
        changed = {
            name for name, expected in prereg.get("code_sha256", {}).items()
            if base.sha256_file(Path(name)) != expected
        }
        rebinding = None
        if changed:
            rebinding = validate_infrastructure_rebinding(
                INFRA_REBINDING, INFRA_REBINDING_SHA, prereg
            )
            if changed != set(rebinding["preregistered_authority_changes"]):
                raise RuntimeError("0707 exact code changes exceed infrastructure rebinding")
        archive = json.loads(ARCHIVE.read_text())
        if not (
            archive.get("old_workflow_status_preserved") is True
            and archive.get("resume_old_e_checkpoints_allowed") is False
        ):
            raise RuntimeError("old route archive boundary changed")
        self.require_idle()
        smoke_path = STATE_ROOT / "smoke/smoke_manifest.json"
        smoke = json.loads(smoke_path.read_text())
        if not (
            smoke.get("passed") is True
            and smoke.get("start_checkpoint_sha256") == TEACHER_SHA
            and smoke.get("end_effective_updates") in (1, 2, 3, 4, 5)
            and smoke.get("warmup_main_target") == "teacher_pre_prior"
            and smoke.get("warmup_prior_box_loss_zero") is True
            and smoke.get("warmup_post_prior_main_loss_zero") is True
            and smoke.get("actor_unchanged") is True
            and smoke.get("teacher_student_storage_independent") is True
            and smoke.get("full_checkpoint_restore_verified") is True
        ):
            raise RuntimeError("0707 exact 1-5 update smoke is missing or invalid")
        critical = [
            Path(__file__), BASE_PATH,
            ROOT / "tools/highstep_student_recovery_v15_supervisor.py",
            ROOT / "tools/highstep_wandb_stage_gate.py",
            base.CANDIDATE_AUDIT, base.IMITATION_GATE,
            ROOT / "scripts/rsl_rl/base/train.py", ROOT / "scripts/rsl_rl/base/play.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        ]
        self.critical_hashes = {
            str(path.resolve()): base.sha256_file(path.resolve()) for path in critical
        }
        preflight = STATE_ROOT / "preflight_manifest.json"
        payload = {
            "schema_version": 1,
            "kind": "highstep_0707_exact_new_teacher_preflight",
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "teacher_sha256": TEACHER_SHA,
            "smoke_manifest": str(smoke_path),
            "smoke_manifest_sha256": base.sha256_file(smoke_path),
            "old_route_archive_manifest": str(ARCHIVE),
            "old_route_archive_manifest_sha256": base.sha256_file(ARCHIVE),
            "critical_code_sha256": self.critical_hashes,
            "target_semantics": "teacher_pre_prior_during_warmup",
            "warmup_updates": 1200,
            "behavior_evidence_driver": "candidate_student_only",
            "completed_at": base.now(),
        }
        if preflight.exists() and rebinding is None:
            stored = json.loads(preflight.read_text())
            stable = {key: value for key, value in payload.items() if key != "completed_at"}
            old = {key: value for key, value in stored.items() if key != "completed_at"}
            if stable != old or not base.is_read_only(preflight):
                raise RuntimeError("stored 0707 exact preflight changed")
        elif not preflight.exists():
            atomic_json(preflight, payload, read_only=True)
        else:
            if not base.is_read_only(preflight):
                raise RuntimeError("stored 0707 exact preflight became writable")
            stored = json.loads(preflight.read_text())
            if not (
                stored.get("spec_sha256") == SPEC_SHA
                and stored.get("preregistration_sha256") == PREREG_SHA
                and stored.get("teacher_sha256") == TEACHER_SHA
            ):
                raise RuntimeError("stored 0707 exact preflight authority changed")
            amendment = STATE_ROOT / "preflight_wandb_multifragment_rebinding.json"
            amendment_payload = {
                "schema_version": 1,
                "kind": "highstep_0707_exact_preflight_infrastructure_rebinding",
                "workflow_id": WORKFLOW_ID,
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "previous_preflight": str(preflight),
                "previous_preflight_sha256": base.sha256_file(preflight),
                "rebinding_manifest": str(INFRA_REBINDING.resolve()),
                "rebinding_manifest_sha256": INFRA_REBINDING_SHA,
                "critical_code_sha256": self.critical_hashes,
                "training_semantics_changed": False,
            }
            if amendment.exists():
                if (
                    json.loads(amendment.read_text()) != amendment_payload
                    or not base.is_read_only(amendment)
                ):
                    raise RuntimeError("stored infrastructure preflight rebinding changed")
            else:
                atomic_json(amendment, amendment_payload, read_only=True)
        self.update(
            status="preflight_passed", phase="ready_for_stage_100", effective_updates=0,
            checkpoint=str(TEACHER), code_sha256=self.critical_hashes,
            stop_reason=None, last_error=None,
        )

    def run(self) -> None:
        self.preflight()
        source, previous = SOURCE, 0
        best = None
        for stage in SAVE_POINTS:
            self.assert_code()
            result_path = STATE_ROOT / "stages" / f"E{stage}" / "stage_result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                checkpoint = Path(result["checkpoint"]).resolve(strict=True)
                if result["checkpoint_sha256"] != base.sha256_file(checkpoint):
                    raise RuntimeError(f"stage result E{stage} checkpoint changed")
                wandb_manifest = Path(result["wandb_stage_manifest"]).resolve(strict=True)
                if (
                    json.loads(wandb_manifest.read_text()).get("sync_status") != "synced"
                    or not self.wandb_offline_coverage_complete(wandb_manifest)
                ):
                    self.finalize_wandb(
                        stage, checkpoint, result_path, wandb_manifest,
                        self.wandb_gate_metrics(result), result["decision"]["reason"],
                    )
            else:
                run_dir, checkpoint, wandb_manifest = self.train_stage(stage, source, previous)
                imitation = self.imitation(stage, checkpoint)
                probe = self.probe(stage, checkpoint)
                core = self.core9(stage, checkpoint, run_dir)
                directional = self.directional(stage, checkpoint, core)
                decision = self.decision(stage, imitation, probe, core, directional)
                direction_score = max(
                    min(value["counts"]["full_climb"], value["counts"]["rear_hold"])
                    for value in directional["corridors"].values()
                )
                rank = [
                    int(decision["final_numeric_gate_passed"]), direction_score,
                    core["counts"]["full_climb"] + core["counts"]["rear_hold"],
                    probe["behavior_score"], -imitation["failed_element_count"], -stage,
                ]
                result = {
                    "schema_version": 1, "kind": "highstep_0707_exact_stage_result",
                    "stage": stage, "checkpoint": str(checkpoint),
                    "checkpoint_sha256": base.sha256_file(checkpoint),
                    "imitation": imitation, "probe": probe, "core9": core,
                    "directional": directional, "decision": decision,
                    "selection_rank": rank, "wandb_stage_manifest": str(wandb_manifest),
                    "behavior_evidence_is_candidate_driven": True,
                    "completed_at": base.now(),
                }
                atomic_json(result_path, result, read_only=True)
                self.finalize_wandb(
                    stage, checkpoint, result_path, wandb_manifest,
                    self.wandb_gate_metrics(result), decision["reason"],
                )
            best = result if best is None or tuple(result["selection_rank"]) > tuple(best["selection_rank"]) else best
            decision = result["decision"]
            self.update(
                checkpoint=result["checkpoint"], effective_updates=stage,
                imitation_gate={"passed": result["imitation"]["passed"], "failed_elements": result["imitation"]["failed_element_count"]},
                behavior_probe=result["probe"]["counts"], core9=result["core9"]["counts"],
                directional={side: result["directional"]["corridors"][side]["counts"] for side in ("left", "right")},
                classification=decision["classification"], decision=decision,
                best_checkpoint=best["checkpoint"], best_checkpoint_sha256=best["checkpoint_sha256"],
            )
            if decision["final_numeric_gate_passed"]:
                side = decision["passed_sides"][0] if decision["passed_sides"] else "left"
                video_manifest = self.videos(stage, Path(result["checkpoint"]), side)
                self.update(video_manifest=str(video_manifest), video_manifest_sha256=base.sha256_file(video_manifest))
                self.record_terminal("student_directional_candidate_pending_user_visual_review", decision["reason"], best)
                return
            if decision.get("stop") is True:
                self.record_terminal(
                    decision.get("terminal_status", "stopped_by_gate"),
                    decision["reason"],
                    best,
                )
                return
            source, previous = Path(result["checkpoint"]), stage
        self.record_terminal(
            "stopped_by_gate",
            f"absolute_{base.ABSOLUTE_CAP}_cap_without_stable_climb",
            best,
        )


def main() -> int:
    supervisor = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("0707 exact supervisor already owns the workflow lock")
        return 2
    except Exception as error:
        if supervisor is not None:
            supervisor.record_terminal(
                "mechanism_error_requires_repair",
                f"{type(error).__name__}: {error}",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
