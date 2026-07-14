#!/usr/bin/env python3
"""Bounded E1400 continuation A/B supervisor for spec v1.10."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch

try:
    from tools.highstep_e1400_continuation_checkpoint_scope import checkpoint_scope
except ModuleNotFoundError:
    from highstep_e1400_continuation_checkpoint_scope import checkpoint_scope


ROOT = Path("/home/lxq/Softwares/robot_lab")
V18_PATH = ROOT / "tools/highstep_student_env_curriculum_v18_supervisor.py"
_spec = importlib.util.spec_from_file_location("highstep_v110_reused_v18", V18_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load v1.8 supervisor mechanics")
v18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v18)
mechanics = v18.mechanics
legacy = v18.legacy
prior = v18.prior
base = v18.base

WORKFLOW_ID = "highstep_e1400_continuation_ab_20260715"
STATE_ROOT = ROOT / "tmp/highstep_e1400_continuation_ab_20260715"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "3b526ddb88b60193d5fee1bb30f6473b5322df96eca393106ad680c6b245ffcf"
BASE_PREREG = STATE_ROOT / "preregistration_v110.json"
BASE_PREREG_SHA = "b570e454b7f0068786f43faf0685f084580f6858268c79c8bb5cde69e088f202"
PREREG = Path(os.environ.get(
    "HIGHSTEP_E1400_CONTINUATION_PREREGISTRATION_PATH",
    STATE_ROOT / "preregistration_v110_runtime.json",
))
PREREG_SHA = os.environ.get("HIGHSTEP_E1400_CONTINUATION_PREREGISTRATION_SHA256", "")
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
SOURCE = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "environment_curriculum_v18_Student/"
    "2026-07-14_21-03-31_v18_stage_a_E1400_20260714_210326/model_1394.pt"
)
SOURCE_SHA = "92bf3d0612f0f9ea1f85b379af8754a0df2710febb9b0067fd86e17487c810f9"
SOURCE_SCOPE = ROOT / "tmp/highstep_student_env_curriculum_v18_20260714/stages/E1400/checkpoint_scope.json"
SOURCE_SCOPE_SHA = "62b574be1d47f3a0483aab5fd7222463f7a8020fc255400e8e6e48389e02fe5b"
SOURCE_HANDOFF = ROOT / "tmp/highstep_student_env_curriculum_v18_20260714/handoff.json"
SOURCE_HANDOFF_SHA = "5a6510fa02faee9185d2a5056267e0e5be18c567a8397ee379ad5a0508f8f0c7"
ORACLE_DECISION = ROOT / (
    "tmp/highstep_oracle_prior_delta_ab_20260714/manifests/"
    "oracle_prior_delta_ab_decision.json"
)
ORACLE_DECISION_SHA = "546f0a3396e2d0e65bedd35904123c68847c61cf35c0c815ac5a66b30b644b70"
ORACLE_HANDOFF = ROOT / "tmp/highstep_oracle_prior_delta_ab_20260714/handoff.json"
ORACLE_HANDOFF_SHA = "db4c7ac6bb3b76860c430ca5ed148c9ac84b2904dfd6658570f454aa7a8a5ea9"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
PROFILE = ROOT / "tmp/highstep_student_env_curriculum_v18_20260714/profiles/stage_a_0707_student_bootstrap.json"
PROFILE_SHA = "39ef546bb271426e7d2cc878b5685a32ecbf510fadd9b97ea8b44ada826168b6"
ENV_SHA = "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636"
TRAIN_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-"
    "ArcdogAdjustableLeg-v0"
)
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "environment_curriculum_v18_Student"
)
SAVE_POINTS = (1500, 1600, 1700)
PARENT_TEACHER_MANIFEST = ROOT / (
    "tmp/highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json"
)
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"

for module in (v18, mechanics, legacy, prior, base):
    module.WORKFLOW_ID = WORKFLOW_ID
    module.STATE_ROOT = STATE_ROOT
    module.SPEC_SHA = SPEC_SHA
    module.PREREG = PREREG
    module.PREREG_SHA = PREREG_SHA
base.SOURCE = SOURCE
base.TEACHER = TEACHER
base.TRAIN_TASK = TRAIN_TASK
base.EVAL_TASK = TRAIN_TASK
base.EXPERIMENT_ROOT = EXPERIMENT_ROOT
base.SAVE_POINTS = SAVE_POINTS
base.CORE9_POINTS = set(SAVE_POINTS)
base.PREREG_ENV_PREFIX = "HIGHSTEP_E1400_CONTINUATION"
base.ABSOLUTE_CAP = 1700
base.PARENT_TEACHER_MANIFEST = PARENT_TEACHER_MANIFEST
prior.SOURCE_E300 = SOURCE
prior.SOURCE_E300_SHA = SOURCE_SHA


def atomic_json(path: Path, payload: Mapping[str, Any], read_only: bool = False) -> None:
    base.atomic_json(path, payload, read_only=read_only)


class Supervisor(v18.Supervisor):
    def __init__(self) -> None:
        self.branch = "none"
        self.branch_root = STATE_ROOT
        super().__init__()

    def update(self, **values: Any) -> None:
        values.setdefault("authority_version", "v1.10")
        values.setdefault("branch", self.branch)
        values.setdefault("failure_class", self.state.get("failure_class", "none"))
        base.Supervisor.update(self, **values)

    @staticmethod
    def _sha(path: Path) -> str:
        return base.sha256_file(path)

    def _bind_branch(self, branch: str) -> None:
        branch = branch.upper()
        if branch not in {"A", "B"}:
            raise RuntimeError(f"invalid continuation branch: {branch}")
        self.branch = branch
        self.branch_root = STATE_ROOT / "branches" / branch
        os.environ["HIGHSTEP_E1400_CONTINUATION_BRANCH"] = branch
        for module in (base, prior, legacy, mechanics, v18):
            module.WORKFLOW_ID = WORKFLOW_ID
            module.SPEC_SHA = SPEC_SHA
            module.PREREG = PREREG
            module.PREREG_SHA = PREREG_SHA
        base.STATE_ROOT = self.branch_root
        base.SOURCE = SOURCE
        base.TEACHER = TEACHER
        base.TRAIN_TASK = TRAIN_TASK
        base.EVAL_TASK = TRAIN_TASK
        base.EXPERIMENT_ROOT = EXPERIMENT_ROOT
        base.SAVE_POINTS = SAVE_POINTS
        base.CORE9_POINTS = set(SAVE_POINTS)
        base.TRAIN_RUN_PREFIX = f"v110_branch_{branch.lower()}"
        base.PREREG_ENV_PREFIX = "HIGHSTEP_E1400_CONTINUATION"
        base.WARMUP_UPDATES = 1700 if branch == "A" else 1400
        base.ABSOLUTE_CAP = 1700
        base.SCHEDULE_RESUME_MODE = "preserve"
        base.ENVIRONMENT_PROFILE_MANIFEST = str(PROFILE)
        base.ENVIRONMENT_PROFILE_MANIFEST_SHA256 = PROFILE_SHA
        base.V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE = ""
        base.V18_STAGE_TRANSITION_MANIFEST = ""
        base.V18_STAGE_TRANSITION_MANIFEST_SHA256 = ""
        self.course_stage = "A"
        self.environment_profile = str(PROFILE)
        self.environment_source_sha256 = ENV_SHA
        self.update(branch=branch, environment_profile=str(PROFILE), environment_source_sha256=ENV_SHA)

    def preflight(self) -> None:
        if not PREREG_SHA:
            raise RuntimeError("v1.10 runtime preregistration SHA is missing")
        required = {
            SPEC: SPEC_SHA,
            BASE_PREREG: BASE_PREREG_SHA,
            PREREG: PREREG_SHA,
            SOURCE: SOURCE_SHA,
            SOURCE_SCOPE: SOURCE_SCOPE_SHA,
            SOURCE_HANDOFF: SOURCE_HANDOFF_SHA,
            ORACLE_DECISION: ORACLE_DECISION_SHA,
            ORACLE_HANDOFF: ORACLE_HANDOFF_SHA,
            TEACHER: TEACHER_SHA,
            PROFILE: PROFILE_SHA,
        }
        for path, expected in required.items():
            if self._sha(path) != expected:
                raise RuntimeError(f"v1.10 authority SHA mismatch: {path}")
        prereg = json.loads(PREREG.read_text())
        if not (
            prereg.get("kind") == "highstep_e1400_continuation_ab_v110_runtime_preregistration"
            and prereg.get("workflow_id") == WORKFLOW_ID
            and prereg.get("authority", {}).get("version") == "v1.10"
            and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA
            and prereg.get("base_preregistration") == str(BASE_PREREG)
            and prereg.get("base_preregistration_sha256") == BASE_PREREG_SHA
            and prereg.get("source", {}).get("checkpoint_sha256") == SOURCE_SHA
            and prereg.get("budget", {}).get("matched_save_points") == list(SAVE_POINTS)
        ):
            raise RuntimeError("v1.10 runtime preregistration identity changed")
        dashboard = json.loads(DASHBOARD.read_text())
        if not (
            dashboard.get("workflow_id") == WORKFLOW_ID
            and dashboard.get("authority_version") == "v1.10"
            and dashboard.get("spec_sha256") == SPEC_SHA
            and dashboard.get("preregistration_path") == str(PREREG)
            and dashboard.get("preregistration_sha256") == PREREG_SHA
        ):
            raise RuntimeError("v1.10 dashboard authority changed")
        source = torch.load(SOURCE, map_location="cpu", weights_only=False)
        algorithm = source.get("infos", {}).get(ALGORITHM_KEY, {})
        recovery = algorithm.get("student_recovery", {})
        optimizer = source.get("optimizer_state_dict", {})
        if not (
            algorithm.get("student_distill_update_count") == 1400
            and recovery.get("stage") == "ENV_CURRICULUM_V18"
            and recovery.get("effective_update_count") == 1400
            and recovery.get("preregistration_sha256")
            == "fd38dd7e3270fd257c211ca8c23d15d9719cbef3e817dd1281b73f122563fac1"
            and [group.get("lr") for group in optimizer.get("param_groups", [])]
            == [1.0e-3, 1.0e-5]
        ):
            raise RuntimeError("E1400 source is not the complete full checkpoint")
        smoke = prereg.get("smoke", {})
        smoke_path = Path(str(smoke.get("manifest", "")))
        if not (
            smoke_path.is_absolute()
            and self._sha(smoke_path) == smoke.get("manifest_sha256")
            and json.loads(smoke_path.read_text()).get("passed") is True
        ):
            raise RuntimeError("v1.10 continuation smoke evidence missing")
        critical = {str(Path(path).resolve()): digest for path, digest in prereg.get("code_sha256", {}).items()}
        actual = {path: self._sha(Path(path)) for path in critical}
        if actual != critical:
            raise RuntimeError("v1.10 critical code differs from preregistration")
        self.critical_hashes = actual
        self.require_idle()
        self.update(
            status="preflight_passed", phase="ready_for_matched_E1500",
            checkpoint=str(SOURCE), effective_updates=1400, code_sha256=actual,
            failure_class="none", last_error=None,
        )

    def checkpoint_scope(self, checkpoint: Path, stage: int, run_dir: Path) -> dict[str, Any]:
        return checkpoint_scope(
            checkpoint, stage, branch=self.branch, preregistration=PREREG,
            preregistration_sha256=PREREG_SHA, run_dir=run_dir, dashboard=DASHBOARD,
        )

    def diagnostic_record(self, stage: int, checkpoint: Path) -> dict[str, Any]:
        path = self.branch_root / "stages" / f"E{stage}" / "same_state_diagnostic_status.json"
        if not path.exists():
            atomic_json(path, {
                "schema_version": 1,
                "kind": "highstep_v110_same_state_diagnostic_status",
                "branch": self.branch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": self._sha(checkpoint),
                "classification": "not_a_behavior_gate",
                "candidate_driven_behavior_is_authoritative": True,
            }, read_only=True)
        return json.loads(path.read_text())

    def _result(self, stage: int, checkpoint: Path, run_dir: Path,
                wandb_manifest: Path) -> dict[str, Any]:
        diagnostic = self.diagnostic_record(stage, checkpoint)
        probe = base.Supervisor.probe(self, stage, checkpoint)
        core = base.Supervisor.core9(self, stage, checkpoint, run_dir)
        counts = core["counts"]
        passed = bool(
            counts["valid"] == 9 and counts["full_climb"] >= 7
            and counts["rear_hold"] >= 7 and counts["no_severe_inward"] >= 8
        )
        return {
            "schema_version": 1,
            "kind": "highstep_e1400_continuation_stage_result",
            "workflow_id": WORKFLOW_ID,
            "branch": self.branch,
            "stage": stage,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": self._sha(checkpoint),
            "same_state_diagnostic": diagnostic,
            "probe": probe,
            "core9": core,
            "decision": {
                "stage_a_gate_passed": passed,
                "confirmed_safety_risk": counts["no_severe_inward"] < 8,
                "reason": "stage_a_gate_passed" if passed else "continue_to_next_preregistered_point",
            },
            "selection_rank": [
                min(counts["full_climb"], counts["rear_hold"]),
                counts["full_climb"] + counts["rear_hold"],
                counts["front_top_support"],
                counts["no_severe_inward"],
            ],
            "wandb_stage_manifest": str(wandb_manifest),
            "environment_profile": str(PROFILE),
            "environment_source_sha256": ENV_SHA,
            "behavior_evidence_is_candidate_driven": True,
            "completed_at": base.now(),
        }

    @staticmethod
    def _wandb_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "branch": result["branch"],
            "core9": result["core9"]["counts"],
            "probe": result["probe"]["counts"],
            "stage_a_gate_passed": result["decision"]["stage_a_gate_passed"],
            "environment_source_sha256": ENV_SHA,
        }

    def _run_point(self, branch: str, stage: int, source: Path, previous: int) -> dict[str, Any]:
        self._bind_branch(branch)
        stage_root = self.branch_root / "stages" / f"E{stage}"
        result_path = stage_root / "stage_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            checkpoint = Path(result["checkpoint"]).resolve(strict=True)
            if not (
                result.get("branch") == branch
                and result.get("stage") == stage
                and result.get("checkpoint_sha256") == self._sha(checkpoint)
            ):
                raise RuntimeError(f"stored v1.10 {branch}/E{stage} result changed")
            wandb_manifest = Path(result["wandb_stage_manifest"])
        else:
            run_dir, checkpoint, wandb_manifest = base.Supervisor.train_stage(
                self, stage, source, previous
            )
            result = self._result(stage, checkpoint, run_dir, wandb_manifest)
            atomic_json(result_path, result, read_only=True)
        if (
            json.loads(wandb_manifest.read_text()).get("sync_status") != "synced"
            or not self.wandb_offline_coverage_complete(wandb_manifest)
        ):
            self.finalize_wandb(
                stage, Path(result["checkpoint"]), result_path, wandb_manifest,
                self._wandb_metrics(result), result["decision"]["reason"],
            )
        self.update(
            status="running", phase=f"matched_E{stage}_{branch}_complete",
            checkpoint=result["checkpoint"], effective_updates=stage,
            active_pid=None, branch=branch,
        )
        return result

    def _terminal(self, status: str, reason: str, payload: Mapping[str, Any]) -> None:
        self.update(
            status=status, phase=status, active_pid=None, stop_reason=reason,
            selected_branch=payload.get("selected_branch"),
            selected_checkpoint=payload.get("selected_checkpoint"),
            matched_results=payload.get("matched_results"),
            requires_user_action=status == "continuation_branch_selected_pending_stage_b_authority",
        )
        atomic_json(STATE_ROOT / "handoff.json", {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "status": status,
            "reason": reason,
            "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG),
            "preregistration_sha256": PREREG_SHA,
            **dict(payload),
            "automatic_stage_b_started": False,
            "automatic_real_robot_deployment_performed": False,
            "written_at": base.now(),
        }, read_only=True)

    def run(self) -> None:
        self.preflight()
        sources = {"A": SOURCE, "B": SOURCE}
        previous = {"A": 1400, "B": 1400}
        all_results: dict[str, dict[str, Any]] = {"A": {}, "B": {}}
        for stage in SAVE_POINTS:
            matched: dict[str, Any] = {}
            for branch in ("A", "B"):
                self.assert_code()
                result = self._run_point(branch, stage, sources[branch], previous[branch])
                sources[branch] = Path(result["checkpoint"])
                previous[branch] = stage
                all_results[branch][str(stage)] = result
                matched[branch] = result
            decision_path = STATE_ROOT / "matched_decisions" / f"E{stage}.json"
            passed = [branch for branch in ("A", "B") if matched[branch]["decision"]["stage_a_gate_passed"]]
            selected = max(("A", "B"), key=lambda branch: tuple(matched[branch]["selection_rank"]))
            decision = {
                "schema_version": 1,
                "kind": "highstep_e1400_continuation_matched_decision",
                "workflow_id": WORKFLOW_ID,
                "stage": stage,
                "results": {
                    branch: {
                        "checkpoint": matched[branch]["checkpoint"],
                        "checkpoint_sha256": matched[branch]["checkpoint_sha256"],
                        "counts": matched[branch]["core9"]["counts"],
                        "rank": matched[branch]["selection_rank"],
                        "passed": matched[branch]["decision"]["stage_a_gate_passed"],
                    }
                    for branch in ("A", "B")
                },
                "passed_branches": passed,
                "selected_by_preregistered_rank": selected,
                "continue": not passed and stage < 1700,
                "completed_at": base.now(),
            }
            atomic_json(decision_path, decision, read_only=True)
            if passed:
                selected = max(passed, key=lambda branch: tuple(matched[branch]["selection_rank"]))
                self._terminal(
                    "continuation_branch_selected_pending_stage_b_authority",
                    f"matched E{stage} Stage-A gate passed",
                    {
                        "selected_branch": selected,
                        "selected_checkpoint": matched[selected]["checkpoint"],
                        "selected_checkpoint_sha256": matched[selected]["checkpoint_sha256"],
                        "matched_results": decision["results"],
                    },
                )
                return
        final = {
            branch: all_results[branch]["1700"] for branch in ("A", "B")
        }
        selected = max(("A", "B"), key=lambda branch: tuple(final[branch]["selection_rank"]))
        self._terminal(
            "continuation_ab_stopped_by_gate",
            "neither branch passed the preregistered Stage-A gate by E1700",
            {
                "selected_branch": None,
                "selected_checkpoint": None,
                "best_diagnostic_branch": selected,
                "best_diagnostic_checkpoint": final[selected]["checkpoint"],
                "best_diagnostic_checkpoint_sha256": final[selected]["checkpoint_sha256"],
                "matched_results": {
                    branch: {
                        "checkpoint": final[branch]["checkpoint"],
                        "checkpoint_sha256": final[branch]["checkpoint_sha256"],
                        "counts": final[branch]["core9"]["counts"],
                        "rank": final[branch]["selection_rank"],
                    }
                    for branch in ("A", "B")
                },
            },
        )


def main() -> int:
    supervisor: Supervisor | None = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("v1.10 continuation supervisor already owns the workflow lock", file=sys.stderr)
        return 2
    except Exception as error:
        if supervisor is not None:
            failure_class = supervisor.classify_failure(
                str(supervisor.state.get("phase", "unknown")), str(error)
            )
            supervisor.update(
                status="mechanism_error_requires_repair",
                phase="mechanism_error_requires_repair",
                active_pid=None,
                failure_class=failure_class,
                last_error=f"{type(error).__name__}: {error}",
                requires_user_action=False,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
