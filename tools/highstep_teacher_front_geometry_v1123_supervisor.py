#!/usr/bin/env python3
"""Fail-closed v1.12.3 paired Teacher reward-geometry A/B supervisor."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import yaml

from tools import highstep_teacher_front_placement_v1122_supervisor as base


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORK = ROOT / "tmp/highstep_teacher_front_geometry_v1123_20260716"
WORKFLOW = "highstep_teacher_front_geometry_v1123_20260716"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = os.environ.get("HIGHSTEP_V1123_SPEC_SHA256", "")
PREREG = Path(os.environ.get("HIGHSTEP_V1123_PREREGISTRATION", ""))
PREREG_SHA = os.environ.get("HIGHSTEP_V1123_PREREGISTRATION_SHA256", "")
RUNTIME = Path(os.environ.get("HIGHSTEP_V1123_RUNTIME_BINDING", ""))
RUNTIME_SHA = os.environ.get("HIGHSTEP_V1123_RUNTIME_BINDING_SHA256", "")
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
SOURCE = ROOT / (
    "tmp/highstep_teacher_front_placement_v1122_20260716/"
    "source_model_173200_v1122/model_173200.pt"
)
SOURCE_SHA = "962fd3ce3983e4a478092be8f7a636e87ed872b79496c4fa3134191838d9b80d"
SOURCE_ITERATION = 173200
SOURCE_SCHEDULE = SOURCE.parent / "params/highstep_schedule_manifest.json"
SOURCE_RUNTIME = SOURCE.parent / "params/highstep_runtime_state.json"
PREFLIGHT = WORK / "frozen_preflight_manifest.json"
PREFLIGHT_SHA = "26b716cde40e1045520808b18c63a2b0f0fb0b24a13545820f7639c1c81094ce"
TASK_A = "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123Control-ArcdogAdjustableLeg-v0"
TASK_B = "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123-ArcdogAdjustableLeg-v0"
EXPERIMENT_A = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_control_Teacher"
EXPERIMENT_B = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher"
WANDB_ENTITY = "xinqili551-the-university-of-hong-kong"
WANDB_PROJECT = "isaaclab"
EVAL_SEEDS = tuple(range(1101, 1116))
SERVICE = "highstep-teacher-front-geometry-v1123.service"


def configure_base(*, task: str, experiment: Path, updates: int) -> None:
    base.ROOT = ROOT
    base.WORK = WORK
    base.WORKFLOW = WORKFLOW
    base.SPEC = SPEC
    base.SPEC_SHA = SPEC_SHA
    base.PREREG = PREREG
    base.PREREG_SHA = PREREG_SHA
    base.DASHBOARD = DASHBOARD
    base.SOURCE = SOURCE
    base.SOURCE_SHA = SOURCE_SHA
    base.SOURCE_ITERATION = SOURCE_ITERATION
    base.REBOUND_SOURCE = SOURCE
    base.REBOUND_SCHEDULE = SOURCE_SCHEDULE
    base.REBOUND_RUNTIME = SOURCE_RUNTIME
    base.TASK = task
    base.EXPERIMENT = experiment
    base.WANDB_ENTITY = WANDB_ENTITY
    base.WANDB_PROJECT = WANDB_PROJECT
    base.EVAL_SEEDS = EVAL_SEEDS
    base.FORMAL_NUM_ENVS = 4096
    base.SEED = 42
    base.FORMAL_UPDATES = updates
    base.TARGET_FINAL_ITERATION = SOURCE_ITERATION + updates - 1
    base.SERVICE = SERVICE


class Supervisor(base.Supervisor):
    def __init__(self) -> None:
        WORK.mkdir(parents=True, exist_ok=True)
        self.lock = (WORK / "supervisor.lock").open("a+", encoding="utf-8")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = WORK / "state.json"
        self.heartbeat_path = WORK / "heartbeat.json"
        self.handoff_path = WORK / "handoff.json"
        self.active = None
        self.stop_requested = False
        self.current_stage: dict[str, Any] | None = None
        self._validate_authority()
        self.state = base.read_json(self.state_path)
        if not (
            self.state.get("workflow_id") == WORKFLOW
            and self.state.get("spec_sha256") == SPEC_SHA
            and self.state.get("preregistration_sha256") == PREREG_SHA
        ):
            raise RuntimeError("v1.12.3 state authority mismatch")

    def _validate_authority(self) -> None:
        if not all((SPEC_SHA, PREREG_SHA, RUNTIME_SHA)):
            raise RuntimeError("v1.12.3 service authority environment is incomplete")
        checks = (
            (SPEC, SPEC_SHA, False),
            (PREREG, PREREG_SHA, True),
            (RUNTIME, RUNTIME_SHA, True),
            (SOURCE, SOURCE_SHA, True),
            (PREFLIGHT, PREFLIGHT_SHA, True),
        )
        for path, expected, read_only in checks:
            if not path.is_file() or base.sha256_file(path) != expected:
                raise RuntimeError(f"v1.12.3 authority artifact mismatch: {path}")
            if read_only and path.stat().st_mode & 0o222:
                raise RuntimeError(f"v1.12.3 authority artifact is writable: {path}")
        prereg = base.read_json(PREREG)
        if not (
            prereg.get("workflow_id") == WORKFLOW
            and prereg.get("authority", {}).get("version") == "v1.12.3"
            and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA
            and prereg.get("source", {}).get("checkpoint_sha256") == SOURCE_SHA
            and prereg.get("single_semantic_variable")
            == "left_front_precontact_reward_geometry_contract"
            and prereg.get("frozen_preflight", {}).get("sha256") == PREFLIGHT_SHA
        ):
            raise RuntimeError("v1.12.3 paired preregistration content mismatch")
        for path_text, expected in prereg.get("code_sha256", {}).items():
            path = Path(path_text)
            if not path.is_file() or base.sha256_file(path) != expected:
                raise RuntimeError(f"v1.12.3 bound code changed: {path}")
        runtime = base.read_json(RUNTIME)
        dashboard = base.read_json(DASHBOARD)
        if not (
            runtime.get("workflow_id") == WORKFLOW
            and runtime.get("spec_sha256") == SPEC_SHA
            and runtime.get("preregistration_sha256") == PREREG_SHA
            and runtime.get("supervisor_sha256") == base.sha256_file(Path(__file__))
            and dashboard.get("workflow_id") == WORKFLOW
            and dashboard.get("spec_sha256") == SPEC_SHA
            and dashboard.get("preregistration_sha256") == PREREG_SHA
            and dashboard.get("state_path") == str(self.state_path)
        ):
            raise RuntimeError("v1.12.3 runtime/dashboard binding mismatch")
        if base.checkpoint_scope(SOURCE)["iteration"] != SOURCE_ITERATION:
            raise RuntimeError("protected source iteration changed")

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["supervisor_pid"] = os.getpid()
        self.state["updated_at"] = base.now()
        base.atomic_json(self.state_path, self.state)
        base.atomic_json(
            self.heartbeat_path,
            {
                "schema_version": 2,
                "workflow_id": WORKFLOW,
                "authority_version": "v1.12.3",
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "status": self.state.get("status"),
                "phase": self.state.get("phase"),
                "supervisor_pid": os.getpid(),
                "active_pid": self.state.get("active_pid"),
                "active_child_kind": self.state.get("active_child_kind"),
                "current_iteration": self.state.get("current_iteration"),
                "effective_updates": self.state.get("effective_updates"),
                "wandb_run_id": self.state.get("wandb_run_id"),
                "wandb_run_url": self.state.get("wandb_run_url"),
                "wandb_sync_status": self.state.get("wandb_sync_status"),
                "wandb_remote_latest_step": self.state.get("wandb_remote_latest_step"),
                "failure_class": self.state.get("failure_class", "none"),
                "written_at": base.now(),
                "written_epoch": time.time(),
            },
        )

    def _required_wandb_config(self) -> dict[str, Any]:
        if self.current_stage is None:
            raise RuntimeError("W&B config requested without an active paired stage")
        stage = self.current_stage
        return {
            "workflow_id": WORKFLOW,
            "authority_version": "v1.12.3",
            "stage_id": stage["id"],
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "source_checkpoint_sha256": SOURCE_SHA,
            "single_semantic_variable": "left_front_precontact_reward_geometry_contract",
            "left_front_reward_weight": stage["weight"],
            "num_envs": 4096,
            "seed": 42,
            "additional_updates": stage["updates"],
            "source_iteration": SOURCE_ITERATION,
            "target_final_iteration": stage["target"],
            "checkpoint_load_mode": "full",
            "schedule_resume_mode": "preserve",
            "initial_distribution_claim": (
                "same_fresh_initial_training_distribution_not_process_exact_resume"
            ),
            "task": stage["task"],
        }

    def _verify_wandb_config(self, stage: dict[str, Any]) -> Path:
        self.current_stage = stage
        runtime = base.read_json(RUNTIME)
        record = runtime.get("wandb_configs", {}).get(stage["id"], {})
        path = Path(record.get("path", ""))
        if not (
            path.is_file()
            and base.sha256_file(path) == record.get("sha256")
            and not (path.stat().st_mode & 0o222)
        ):
            raise RuntimeError(f"immutable W&B config invalid for {stage['id']}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {"wandb_version": 1}
        expected.update(
            {key: {"value": value} for key, value in self._required_wandb_config().items()}
        )
        if payload != expected:
            raise RuntimeError(f"W&B config content mismatch for {stage['id']}")
        return path

    def smoke(self) -> Path:
        marker = WORK / "smoke/smoke_result.json"
        configure_base(task=TASK_B, experiment=EXPERIMENT_B, updates=3)
        if marker.exists():
            payload = base.read_json(marker)
            output = Path(payload.get("output_checkpoint", ""))
            if payload.get("passed") is True and output.is_file():
                return output
            raise RuntimeError("stored v1.12.3 smoke changed")
        source_before = base.sha256_file(SOURCE)
        source_scope = base.checkpoint_scope(SOURCE)
        name = f"v1123_geometry_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
        rc, run_dir = self._launch_training(
            checkpoint=SOURCE,
            iterations=3,
            num_envs=64,
            run_name=name,
            phase="v1123_smoke_3_updates_full_resume",
            logger="tensorboard",
        )
        output = base.latest_checkpoint([run_dir] if run_dir else [])
        if rc != 0 or run_dir is None or output is None:
            raise RuntimeError(f"v1.12.3 smoke failed rc={rc} run={run_dir}")
        output_scope = base.checkpoint_scope(output)
        schedule = base.read_json(run_dir / "params/highstep_schedule_manifest.json")
        if not (
            output_scope["iteration"] == SOURCE_ITERATION + 2
            and output_scope["model_tensor_signature_sha256"]
            == source_scope["model_tensor_signature_sha256"]
            and output_scope["optimizer_state_entries"] == source_scope["optimizer_state_entries"]
            and output_scope["optimizer_step_min"] > source_scope["optimizer_step_min"]
            and schedule.get("checkpoint_load_mode") == "full"
            and schedule.get("schedule_source", {}).get("method") == "runtime_snapshot_exact"
            and base.sha256_file(SOURCE) == source_before
        ):
            raise RuntimeError("v1.12.3 smoke tensor/optimizer/schedule binding failed")
        base.atomic_json(
            marker,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "passed": True,
                "updates": 3,
                "num_envs": 64,
                "source_sha256_before": source_before,
                "source_sha256_after": base.sha256_file(SOURCE),
                "source_scope": source_scope,
                "output_checkpoint": str(output),
                "output_checkpoint_sha256": base.sha256_file(output),
                "output_scope": output_scope,
                "schedule_manifest": str(run_dir / "params/highstep_schedule_manifest.json"),
                "completed_at": base.now(),
            },
            read_only=True,
        )
        return output

    def _wait_remote(self, stage: dict[str, Any], checkpoint: Path) -> Path:
        deadline = time.monotonic() + 300.0
        remote = None
        last_error = None
        while time.monotonic() < deadline:
            try:
                remote = self._poll_wandb(child_alive=False)
                if remote.get("state") == "finished" and int(
                    remote.get("remote_latest_step") or -1
                ) >= stage["target"]:
                    break
            except Exception as error:
                last_error = error
            time.sleep(10)
        if remote is None or not (
            remote.get("state") == "finished"
            and int(remote.get("remote_latest_step") or -1) >= stage["target"]
        ):
            raise RuntimeError(f"W&B remote finalization failed for {stage['id']}: {last_error or remote}")
        samples = self.state.get("wandb_live_step_samples") or []
        steps = [int(row["remote_latest_step"]) for row in samples]
        if len(steps) < 2 or any(b <= a for a, b in zip(steps, steps[1:])):
            raise RuntimeError(f"W&B live step growth not verified for {stage['id']}: {steps}")
        marker = WORK / "wandb" / f"{stage['id']}_remote_gate.json"
        base.atomic_json(
            marker,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "stage_id": stage["id"],
                "passed": True,
                "run_id": remote["run_id"],
                "url": remote["url"],
                "remote_state": remote["state"],
                "remote_latest_step": remote["remote_latest_step"],
                "live_step_samples": samples,
                "output_checkpoint": str(checkpoint),
                "output_checkpoint_sha256": base.sha256_file(checkpoint),
                "verified_at": base.now(),
            },
            read_only=True,
        )
        return marker

    def train_stage(self, stage: dict[str, Any]) -> dict[str, Any]:
        configure_base(task=stage["task"], experiment=stage["experiment"], updates=stage["updates"])
        self.current_stage = stage
        config_path = self._verify_wandb_config(stage)
        run_id = f"v1123{stage['id'].lower().replace('-', '')}{int(time.time())}{os.getpid()}"
        run_name = f"v1123_{stage['id']}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.update(
            status="running",
            phase=f"{stage['id']}_preparing",
            active_stage=stage["id"],
            wandb_run_id=run_id,
            wandb_run_url=None,
            wandb_sync_status="online_launch_pending",
            wandb_remote_latest_step=None,
            wandb_live_step_samples=[],
            failure_class="none",
            formal_training_started=True,
            training_started=True,
        )
        rc, run_dir = self._launch_training(
            checkpoint=SOURCE,
            iterations=stage["updates"],
            num_envs=4096,
            run_name=run_name,
            phase=f"{stage['id']}_continuous_training",
            logger="wandb",
            wandb_run_id=run_id,
            wandb_config=config_path,
        )
        if rc != 0 or run_dir is None:
            raise RuntimeError(f"{stage['id']} training failed rc={rc} run={run_dir}")
        final = run_dir / f"model_{stage['target']}.pt"
        if not final.is_file():
            raise RuntimeError(f"{stage['id']} final checkpoint missing: {final}")
        intermediate = None
        if stage["id"] == "B-E300":
            intermediate = run_dir / "model_173299.pt"
            if not intermediate.is_file():
                raise RuntimeError("continuous B run did not save B-E100")
        scope = base.checkpoint_scope(final)
        source_scope = base.checkpoint_scope(SOURCE)
        if not (
            scope["iteration"] == stage["target"]
            and scope["model_tensor_signature_sha256"] == source_scope["model_tensor_signature_sha256"]
            and scope["optimizer_state_entries"] == source_scope["optimizer_state_entries"]
            and scope["optimizer_step_min"] > source_scope["optimizer_step_min"]
            and base.sha256_file(SOURCE) == SOURCE_SHA
        ):
            raise RuntimeError(f"{stage['id']} tensor/optimizer binding failed")
        schedule = base.read_json(run_dir / "params/highstep_schedule_manifest.json")
        distribution = schedule.get("fresh_initial_training_distribution")
        if not isinstance(distribution, dict) or not distribution.get("joint_sha256"):
            raise RuntimeError(f"{stage['id']} missing fresh initial terrain evidence")
        remote_gate = self._wait_remote(stage, final)
        result = {
            "stage_id": stage["id"],
            "run_dir": str(run_dir),
            "child_pid": self.state.get(f"{stage['id']}_child_pid"),
            "wandb_run_id": run_id,
            "wandb_url": self.state.get("wandb_run_url"),
            "final_checkpoint": str(final),
            "final_checkpoint_sha256": base.sha256_file(final),
            "intermediate_checkpoint": str(intermediate) if intermediate else None,
            "intermediate_checkpoint_sha256": base.sha256_file(intermediate) if intermediate else None,
            "fresh_initial_distribution": distribution,
            "remote_gate": str(remote_gate),
            "remote_gate_sha256": base.sha256_file(remote_gate),
        }
        marker = WORK / "training" / f"{stage['id']}_result.json"
        base.atomic_json(marker, result, read_only=True)
        return result

    def _pid_witness(self, process, phase: str, log: Path) -> None:
        super()._pid_witness(process, phase, log)
        if self.current_stage is not None and self.current_stage["id"] in {"A-E100", "B-E300"}:
            self.update(**{f"{self.current_stage['id']}_child_pid": process.pid})

    @staticmethod
    def _safety_pass(summary_path: Path) -> bool:
        summary = base.read_json(summary_path)
        for rollout in summary.get("rollouts", []):
            result_path = Path(rollout["log"]).with_name("result.json")
            row = base.read_json(result_path)
            canonical = row.get("canonical", {})
            if (
                row.get("terminated_early") is True
                or canonical.get("target_limit_step_violation_rate") != 0.0
                or canonical.get("critical_width_violation_rate") != 0.0
                or canonical.get("critical_center_violation_rate") != 0.0
            ):
                return False
        return True

    def _update_remote_summaries(self, stages: list[dict[str, Any]], evaluations: dict[str, Path]) -> None:
        import wandb

        for stage in stages:
            run_id = stage["wandb_run_id"]
            required = {
                "paired_ab_gate": "evaluation_complete",
                "fresh_initial_distribution_joint_sha256": stage[
                    "fresh_initial_distribution"
                ]["joint_sha256"],
            }
            if stage["stage_id"] == "A-E100":
                required["A_E100_evaluation_sha256"] = base.sha256_file(evaluations["A-E100"])
            else:
                required["B_E100_evaluation_sha256"] = base.sha256_file(evaluations["B-E100"])
                required["B_E300_evaluation_sha256"] = base.sha256_file(evaluations["B-E300"])
            path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}"
            run = wandb.Api(timeout=30).run(path)
            run.summary.update(required)
            run.update()
            actual = dict(wandb.Api(timeout=30).run(path).summary)
            if any(actual.get(key) != value for key, value in required.items()):
                raise RuntimeError(f"W&B final summary readback failed for {stage['stage_id']}")

    def run(self) -> None:
        self.update(
            status="running",
            phase="v1123_authority_verified",
            requires_user_action=False,
            training_started=False,
            student_training_started=False,
            automatic_real_robot_deployment=False,
        )
        self.smoke()
        stage_a = {
            "id": "A-E100", "task": TASK_A, "experiment": EXPERIMENT_A,
            "updates": 100, "target": 173299, "weight": 0.0,
        }
        stage_b = {
            "id": "B-E300", "task": TASK_B, "experiment": EXPERIMENT_B,
            "updates": 300, "target": 173499, "weight": 0.10,
        }
        a = self.train_stage(stage_a)
        b = self.train_stage(stage_b)
        if a["fresh_initial_distribution"] != b["fresh_initial_distribution"]:
            raise RuntimeError("paired A/B fresh initial terrain distribution mismatch")

        configure_base(task=TASK_B, experiment=EXPERIMENT_B, updates=300)
        checkpoints = {
            "A-E100": Path(a["final_checkpoint"]),
            "B-E100": Path(b["intermediate_checkpoint"]),
            "B-E300": Path(b["final_checkpoint"]),
        }
        evaluations = {
            label: self.evaluate(checkpoint, label) for label, checkpoint in checkpoints.items()
        }
        summaries = {label: base.read_json(path) for label, path in evaluations.items()}
        gates = {}
        for label, summary in summaries.items():
            gates[label] = bool(
                summary["valid_count"] == 15
                and summary["fl_riser_contact_count"] == 0
                and summary["full_climb_count"] == 15
                and summary["rear_hold_count"] == 15
                and self._safety_pass(evaluations[label])
            )
        selected = next((label for label in ("B-E100", "B-E300") if gates[label]), None)
        decision = (
            "teacher_candidate_pending_user_visual_review"
            if selected is not None
            else "v1123_paired_behavior_gate_failed"
        )
        self._update_remote_summaries([a, b], evaluations)
        handoff = {
            "schema_version": 2,
            "workflow_id": WORKFLOW,
            "authority_version": "v1.12.3",
            "status": decision,
            "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG),
            "preregistration_sha256": PREREG_SHA,
            "source_checkpoint": str(SOURCE),
            "source_checkpoint_sha256": SOURCE_SHA,
            "same_fresh_initial_training_distribution": True,
            "process_exact_resume_claimed": False,
            "training_results": [a, b],
            "evaluation_summaries": {
                label: {
                    "path": str(path),
                    "sha256": base.sha256_file(path),
                    "valid": summaries[label]["valid_count"],
                    "fl_forbidden_contact": summaries[label]["fl_riser_contact_count"],
                    "full_climb": summaries[label]["full_climb_count"],
                    "rear_hold": summaries[label]["rear_hold_count"],
                    "safety_pass": self._safety_pass(path),
                    "gate_pass": gates[label],
                }
                for label, path in evaluations.items()
            },
            "selected_checkpoint": selected,
            "student_training_started": False,
            "automatic_real_robot_deployment": False,
            "next_action": (
                "User visual review only; do not start Student or deploy."
                if selected else "Stop; do not append E500 or change variables."
            ),
            "written_at": base.now(),
        }
        base.atomic_json(self.handoff_path, handoff)
        self.update(
            status=decision,
            phase=decision,
            active_pid=None,
            active_child_kind=None,
            selected_checkpoint=selected,
            evaluation_summaries=handoff["evaluation_summaries"],
            requires_user_action=True,
            failure_class="none" if selected else "behavior_gate_failed",
            last_error=None,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", SERVICE],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> int:
    supervisor = None
    try:
        configure_base(task=TASK_B, experiment=EXPERIMENT_B, updates=300)
        supervisor = Supervisor()
        signal.signal(signal.SIGINT, supervisor.stop)
        signal.signal(signal.SIGTERM, supervisor.stop)
        supervisor.run()
        return 0
    except Exception as error:
        if supervisor is not None:
            supervisor.update(
                status="infrastructure_blocked",
                phase="infrastructure_blocked",
                active_pid=None,
                active_child_kind=None,
                failure_class="infrastructure_fault",
                last_error=f"{type(error).__name__}: {error}",
                requires_user_action=True,
            )
        print(f"[V1123_SUPERVISOR_FATAL] {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
