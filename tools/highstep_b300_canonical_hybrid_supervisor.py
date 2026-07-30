#!/usr/bin/env python3
"""Fail-closed supervisor for B300 canonical hybrid-prior latent distillation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import yaml


ROOT = Path("/home/lxq/Softwares/robot_lab")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import highstep_wandb_stage_gate as wandb_gate

PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
WORKFLOW = "highstep_b300_canonical_hybrid_prior_latent_20260718"
FLOW = ROOT / "tmp/highstep_b300_canonical_hybrid_prior_latent_20260718"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_b300_canonical_hybrid_prior_latent_spec_20260718.md"
SPEC_SHA = "aff25865dadf6d191984c454d050517f1a8b793b381fe150d79abc2b54e78f8c"
PREREG = FLOW / "preregistration.json"
PREREG_SHA = "485661618424ed8f7ce6fbdd75fd76d115a2e54ef955fc39c9a439a9a921fd8e"
TEACHER = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
TEACHER_SHA = "d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431"
DATASET = FLOW / "canonical_tensor_trajectory_attempt2/canonical_tensor_trajectory.pt"
DATASET_SHA = "5b53cd6e14e7760199c18a6036e3a9fd41d549890b76e80b6607a4eee9d2f9cd"
ROUTE_AMENDMENT = FLOW / "dagger_route_correction_v11.json"
ROUTE_AMENDMENT_SHA = "9cb98b8f4832819b51d95a1cd2fe4e10223ee390ac3d435f7c5ccc1707a4d8f1"
ROUTE_CORRECTION = ROOT / "docs/robotlab_memory_zh/library/highstep_b300_canonical_hybrid_route_correction_v11_20260718.md"
ROUTE_CORRECTION_SHA = "4cf28171965277a287f53be57f6229122cb4bedae11d555b1d0d0d188a66cf1e"
DIAGNOSIS = FLOW / "diagnostics/E300_same_state_causal_diagnosis.json"
DIAGNOSIS_SHA = "3fb502e5382470ab880d6c9bcdca79543ec2c30528318d9340ef94f56256daea"
RUNTIME_REBINDING = FLOW / "manifests/runtime_code_rebinding8.json"
PARENT = ROOT / "tmp/highstep_be300_clamp_gradient_repair_v114_20260718/teacher_parent_lineage_v114.json"
TRACE = ROOT / "logs/play_joint_records/20260717_072307_teacher_b300_box_hard_seed11_pid877984/joint_trace.csv"
SOURCE_MANIFEST = TRACE.parent / "manifest.json"
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
TRAIN = ROOT / "scripts/rsl_rl/base/train.py"
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
MERGE = ROOT / "tools/highstep_b300_hybrid_dagger_merge.py"
TASK = "RobotLab-Isaac-Velocity-HighstepB300CanonicalHybridStudentNoPrior-ArcdogAdjustableLeg-v0"
EXPERIMENT = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_canonical_hybrid_student_no_prior_Student"
STAGES = (100, 300, 500, 700)
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"
REPAIRS = (
    FLOW / "manifests/policy_preregistration_binding_repair_rebinding1.json",
    FLOW / "manifests/fresh_schedule_reset_repair_rebinding2.json",
    FLOW / "manifests/dagger_dataset_scope_repair_rebinding3.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, value: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


class Supervisor:
    def __init__(self) -> None:
        self.state_path = FLOW / "state.json"
        self.heartbeat_path = FLOW / "heartbeat.json"
        self.handoff_path = FLOW / "handoff.json"
        self.state: dict[str, Any] = {}
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text())
        self.route = json.loads(ROUTE_AMENDMENT.read_text())
        self.active: subprocess.Popen[bytes] | None = None
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)

    def _signal(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def update(self, **changes: Any) -> None:
        self.state.update({
            "schema_version": 1,
            "workflow_id": WORKFLOW,
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG),
            "preregistration_sha256": PREREG_SHA,
            "supervisor_pid": os.getpid(),
            "service": "highstep-b300-canonical-hybrid.service",
            "route_correction_path": str(ROUTE_CORRECTION),
            "route_correction_sha256": ROUTE_CORRECTION_SHA,
            "route_preregistration_path": str(ROUTE_AMENDMENT),
            "route_preregistration_sha256": ROUTE_AMENDMENT_SHA,
            "updated_at": now(),
            "updated_at_epoch": time.time(),
            **changes,
        })
        atomic_json(self.state_path, self.state)
        atomic_json(self.heartbeat_path, self.state)
        handoff = {
            key: self.state.get(key)
            for key in (
                "schema_version", "workflow_id", "status", "phase", "spec_path", "spec_sha256",
                "preregistration_path", "preregistration_sha256", "checkpoint",
                "checkpoint_sha256", "effective_updates", "active_pid", "wandb_run_id",
                "wandb_run_url", "wandb_sync_status", "centerline_counts", "last_error",
            )
        }
        handoff["next_step"] = self.state.get("next_step")
        handoff["updated_at"] = self.state["updated_at"]
        atomic_json(self.handoff_path, handoff)

    def assert_authority(self) -> None:
        expected = {
            SPEC: SPEC_SHA, PREREG: PREREG_SHA, TEACHER: TEACHER_SHA,
            DATASET: DATASET_SHA, ROUTE_AMENDMENT: ROUTE_AMENDMENT_SHA,
            ROUTE_CORRECTION: ROUTE_CORRECTION_SHA, DIAGNOSIS: DIAGNOSIS_SHA,
        }
        for path, expected_sha in expected.items():
            if not path.is_file() or sha(path) != expected_sha:
                raise RuntimeError(f"authority SHA mismatch: {path}")
        prereg = json.loads(PREREG.read_text())
        if not (
            prereg.get("workflow_id") == WORKFLOW
            and prereg.get("teacher_sha256") == TEACHER_SHA
            and prereg.get("canonical_tensor_dataset_sha256") == DATASET_SHA
        ):
            raise RuntimeError("preregistration contract mismatch")
        if not (
            self.route.get("status")
            == "frozen_before_D1_collection_after_E300_diagnosis"
            and self.route.get("source_checkpoint_sha256")
            == "e7a3eec3cc0848b4af7e9a8608bc281ad94c64ec1e299f69cedcbe34bf04d388"
            and self.route.get("source_effective_updates") == 300
            and self.route.get("collection_trajectories_per_round") == 1
            and self.route.get("single_changed_variable", {}).get("name")
            == "supervision_state_distribution"
            and self.route.get("diagnosis_manifest_sha256") == DIAGNOSIS_SHA
            and self.route.get("route_correction_amendment_sha256")
            == ROUTE_CORRECTION_SHA
        ):
            raise RuntimeError("route correction preregistration mismatch")
        repairs = [json.loads(path.read_text()) for path in REPAIRS]
        for file_name, frozen_sha in prereg["critical_training_code"].items():
            expected_sha = frozen_sha
            for repair in repairs:
                if repair.get("affected_file") == file_name and repair.get("old_sha256") == expected_sha:
                    expected_sha = repair.get("new_sha256")
            if sha(Path(file_name)) != expected_sha:
                raise RuntimeError(f"unbound critical code change: {file_name}")
        runtime = json.loads(RUNTIME_REBINDING.read_text())
        if not (
            runtime.get("kind") == "highstep_b300_hybrid_runtime_code_rebinding"
            and runtime.get("workflow_id") == WORKFLOW
            and runtime.get("spec_sha256") == SPEC_SHA
            and runtime.get("preregistration_sha256") == PREREG_SHA
            and runtime.get("route_amendment_sha256") == ROUTE_AMENDMENT_SHA
        ):
            raise RuntimeError("runtime code rebinding contract mismatch")
        for file_name, expected_sha in runtime.get("code_sha256", {}).items():
            if sha(Path(file_name)) != expected_sha:
                raise RuntimeError(f"runtime code rebinding changed: {file_name}")
        dashboard = json.loads(DASHBOARD.read_text())
        if not (
            dashboard.get("workflow_id") == WORKFLOW
            and Path(dashboard.get("state_path", "")).resolve() == self.state_path.resolve()
            and dashboard.get("spec_sha256") == SPEC_SHA
            and dashboard.get("preregistration_sha256") == PREREG_SHA
        ):
            raise RuntimeError("dashboard authority is not bound to B300 hybrid")

    @staticmethod
    def active_jobs() -> list[dict[str, Any]]:
        jobs = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                argv = [item.decode(errors="replace") for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
            except OSError:
                continue
            if any(Path(item).name in {"train.py", "play.py"} for item in argv):
                jobs.append({"pid": int(proc.name), "argv": " ".join(argv)})
        return jobs

    def require_idle(self) -> None:
        jobs = self.active_jobs()
        if jobs:
            raise RuntimeError(f"another train/eval/play process is active: {jobs}")

    def run_child(
        self, command: Sequence[str], log: Path, phase: str, *, env: Mapping[str, str] | None = None
    ) -> None:
        self.require_idle()
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
        environment.update({
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{ROOT / 'source/robot_lab'}:{ROOT}",
            "LD_PRELOAD": "/lib/x86_64-linux-gnu/libstdc++.so.6",
            "HIGHSTEP_B300_HYBRID_PREREGISTRATION_PATH": str(PREREG),
            "HIGHSTEP_B300_HYBRID_PREREGISTRATION_SHA256": PREREG_SHA,
        })
        if env:
            environment.update(env)
        with log.open("wb") as stream:
            process = subprocess.Popen(
                list(command), cwd=ROOT, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            self.active = process
            while process.poll() is None:
                if self.stop_requested:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=60)
                    raise RuntimeError("supervisor stop requested at atomic child boundary")
                self.update(status="running", phase=phase, active_pid=process.pid)
                time.sleep(3)
            self.active = None
            self.update(active_pid=None)
        if process.returncode != 0:
            raise RuntimeError(f"{phase} exited {process.returncode}; log={log}")

    @staticmethod
    def checkpoint_payload(path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError("checkpoint is not a mapping")
        return payload

    @classmethod
    def checkpoint_count(cls, path: Path) -> int:
        return int(cls.checkpoint_payload(path)["infos"][ALGORITHM_KEY]["student_distill_update_count"])

    def checkpoint_scope(
        self, path: Path, expected: int, *, training_dataset_sha256: str = DATASET_SHA
    ) -> dict[str, Any]:
        payload = self.checkpoint_payload(path)
        algorithm = payload.get("infos", {}).get(ALGORITHM_KEY, {})
        recovery = algorithm.get("student_recovery", {})
        binding = recovery.get("binding_manifest", {})
        required = (
            int(algorithm.get("student_distill_update_count", -1)) == expected
            and int(recovery.get("effective_update_count", -1)) == expected
            and recovery.get("preregistration_sha256") == PREREG_SHA
            and binding.get("teacher_sha256") == TEACHER_SHA
            and binding.get("student_teacher_storage_independent") is True
            and binding.get("student_ppo_permanently_disabled") is True
            and binding.get("actor_body_frozen") is True
            and binding.get("critic_frozen") is True
            and binding.get("student_privileged_encoder_frozen") is True
            and binding.get("frozen_nonbox_action_rows") == list(range(12))
            and binding.get("trainable_action_rows_from_update_zero") == [12, 13, 14, 15]
            and binding.get("warmup_updates") == 0
            and binding.get("warmup_main_action_target")
            == "teacher_pre_prior_nonbox_plus_post_prior_policy_unit_box"
            and binding.get("student_actor_latent_clamp_backward") == "straight_through"
            and binding.get("canonical_tensor_dataset_sha256") == training_dataset_sha256
            and isinstance(algorithm.get("vae_optimizer_state_dict"), Mapping)
        )
        if not required:
            raise RuntimeError(f"checkpoint scope mismatch at E{expected}")
        return {
            "checkpoint": str(path.resolve()), "checkpoint_sha256": sha(path),
            "effective_updates": expected, "optimizer_state_entries": len(algorithm["vae_optimizer_state_dict"]["state"]),
            "frozen_nonbox_rows": list(range(12)), "trainable_box_rows": [12, 13, 14, 15],
        }

    def find_checkpoint(self, run_dir: Path, expected: int) -> Path | None:
        candidates = []
        for path in run_dir.glob("model_*.pt"):
            try:
                if self.checkpoint_count(path) == expected:
                    candidates.append(path)
            except Exception:
                pass
        return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None

    def wandb_files(self, stage: int, source: Path, previous: int) -> tuple[Path, Path, Path]:
        additional = stage - previous
        run_name = f"{WORKFLOW}_B_E{stage}_attempt1"
        config_path = FLOW / "wandb_stage_configs" / f"E{stage}.json"
        native_path = FLOW / "wandb_stage_configs" / f"E{stage}.wandb.yaml"
        start = 0 if previous == 0 else int(self.checkpoint_payload(source)["iter"])
        config = {
            "workflow_id": WORKFLOW, "route": "B", "stage": f"E{stage}", "attempt": "attempt1",
            "run_name": run_name, "group": WORKFLOW, "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA, "start_checkpoint": str(source.resolve()),
            "start_checkpoint_sha256": sha(source), "teacher_checkpoint": str(TEACHER),
            "teacher_sha256": TEACHER_SHA,
            "frozen_tensors": ["teacher.*", "critic.*", "student_priv_encoder.*", "actor rows 0:12 and actor body"],
            "trainable_tensors": ["estimator.*", "actor.final.weight[12:16]", "actor.final.bias[12:16]"],
            "optimizer": {"class": "Adam", "estimator_lr": 1e-3, "box_rows_lr": 1e-5, "fresh_at_E100": True, "preserve_after_E100": True},
            "budget": {"additional_updates": additional, "effective_update_target": stage, "absolute_cap": 700},
            "training_task": TASK, "historical_sync": False, "wandb_mode": "online",
            "canonical_dataset": str(DATASET), "canonical_dataset_sha256": DATASET_SHA,
            "mixed_target": "pre_prior[0:12]+post_prior_policy_units[12:16]",
            "expected_history_step_start": start, "expected_history_step_end": start + additional - 1,
            "expected_history_unique_steps": additional,
        }
        if not config_path.exists():
            atomic_json(config_path, config, read_only=True)
            native = {"wandb_version": 1, **{key: {"value": value} for key, value in config.items()}}
            native_path.parent.mkdir(parents=True, exist_ok=True)
            native_path.write_text(yaml.safe_dump(native, sort_keys=True, allow_unicode=True))
            native_path.chmod(0o444)
        elif json.loads(config_path.read_text()) != config:
            raise RuntimeError(f"stored W&B config changed at E{stage}")
        manifest = FLOW / "wandb_stages" / run_name / "stage_manifest.json"
        if not manifest.exists():
            manifest = wandb_gate.prepare_stage(config_path, FLOW)
        return config_path, native_path, manifest

    def train_stage(self, stage: int, source: Path, previous: int) -> tuple[Path, Path, Path]:
        stage_root = FLOW / "stages" / f"E{stage}"
        result_path = stage_root / "training_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            checkpoint = Path(result["checkpoint"]).resolve(strict=True)
            self.checkpoint_scope(checkpoint, stage)
            return Path(result["run_dir"]), checkpoint, Path(result["wandb_stage_manifest"])
        additional = stage - previous
        _, native, wandb_manifest = self.wandb_files(stage, source, previous)
        wb = json.loads(wandb_manifest.read_text())
        run_name = wb["run_name"]
        command = [
            str(PYTHON), "-u", str(TRAIN.relative_to(ROOT)), "--task", TASK,
            "--num_envs", "4096", "--max_iterations", str(additional), "--seed", "42",
            "--headless", "--logger", "wandb", "--log_project_name", "isaaclab",
            "--resume", "--checkpoint", str(source.resolve()), "--highstep_resume_mode", "refine",
            "--highstep_checkpoint_load_mode", "weights_only" if previous == 0 else "full",
            "--highstep_schedule_resume_mode", "reset" if previous == 0 else "preserve",
            "--highstep_parent_teacher_manifest", str(PARENT), "--run_name", run_name,
        ]
        launch = {
            "schema_version": 1, "stage": stage, "previous_effective_updates": previous,
            "additional_updates": additional, "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": sha(source), "command": command,
            "wandb_stage_manifest": str(wandb_manifest), "created_at": now(),
        }
        atomic_json(stage_root / "launch.json", launch, read_only=True)
        environment = wandb_gate.process_environment(wandb_manifest)
        environment.update({
            "WANDB_MODE": "online", "WANDB_X_DISABLE_STATS": "true", "WANDB_DISABLE_GIT": "true",
            "ROBOT_LAB_WANDB_CONFIG_PATH": str(native.resolve()),
            "ROBOT_LAB_WANDB_CONFIG_SHA256": sha(native),
        })
        self.update(
            status="running", phase=f"training_E{stage}", wandb_run_id=wb["run_id"],
            wandb_run_url=wb["run_url"], wandb_sync_status="online_recording",
            next_step=f"complete E{stage} then run 15 centerline evaluations",
        )
        self.run_child(command, stage_root / "train.log", f"training_E{stage}", env=environment)
        matches = [path for path in EXPERIMENT.glob(f"*_{run_name}") if path.is_dir()]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one E{stage} run directory, got {matches}")
        run_dir = matches[0]
        checkpoint = self.find_checkpoint(run_dir, stage)
        if checkpoint is None:
            raise RuntimeError(f"E{stage} ended without full effective checkpoint")
        scope = self.checkpoint_scope(checkpoint, stage)
        atomic_json(stage_root / "checkpoint_scope.json", scope, read_only=True)
        result = {
            "schema_version": 1, "kind": "b300_hybrid_training_result", "stage": stage,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint),
            "effective_updates": stage, "run_dir": str(run_dir.resolve()),
            "schedule_manifest": str((run_dir / "params/highstep_schedule_manifest.json").resolve()),
            "runtime_state": str((run_dir / "params/highstep_runtime_state.json").resolve()),
            "wandb_stage_manifest": str(wandb_manifest.resolve()), "completed_at": now(),
        }
        atomic_json(result_path, result, read_only=True)
        self.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint), effective_updates=stage)
        return run_dir, checkpoint, wandb_manifest

    def eval_command(
        self,
        checkpoint: Path,
        *,
        scenario: Mapping[str, Any] | None = None,
        narrow: bool = False,
        video_dir: Path | None = None,
        trace_dir: Path | None = None,
    ) -> list[str]:
        scenario = scenario or {
            "name": "nominal", "gap_m": 0.55, "lateral_m": 0.0,
            "yaw_deg": 0.0, "joint_delta_sign": 0,
        }
        command = [
            str(PYTHON), "-u", str(PLAY.relative_to(ROOT)), "--task", TASK, "--num_envs", "1",
            "--seed", "11", "--checkpoint", str(checkpoint), "--play_terrain_level", "9",
            "--play_terrain_type", "box_hard", "--reset_after_play_terrain_selection",
            "--front_step_eval_reset", "--front_step_eval_side", "x-",
            "--front_step_eval_edge_gap", str(scenario["gap_m"]),
            "--front_step_eval_lateral_offset", str(scenario["lateral_m"]),
            "--front_step_eval_yaw_offset_deg", str(scenario["yaw_deg"]),
            "--b300_hybrid_initial_joint_delta_sign", str(scenario["joint_delta_sign"]),
            "--eval_action_delay_steps", "0", "--recorded_command_trace", str(TRACE),
            "--recorded_command_trace_sha256", sha(TRACE), "--recorded_command_episode_id", "4",
            "--recorded_command_source_manifest", str(SOURCE_MANIFEST),
            "--recorded_command_source_manifest_sha256", sha(SOURCE_MANIFEST),
            "--b300_hybrid_behavior_preregistration", str(PREREG),
            "--b300_hybrid_behavior_preregistration_sha256", PREREG_SHA,
            "--play_max_steps", "138", "--highstep_gap_camera", "static_side_top",
            "--print_rear_width_metrics", "--skip_policy_export", "--headless",
        ]
        if video_dir is not None and trace_dir is not None:
            command += [
                "--video", "--video_length", "138", "--video_output_dir", str(video_dir),
                "--record_joint_data", "--joint_record_output", str(trace_dir),
                "--joint_record_label", video_dir.parent.name,
            ]
        if narrow:
            command += [
                "--b300_hybrid_narrow_route_amendment", str(ROUTE_AMENDMENT),
                "--b300_hybrid_narrow_route_amendment_sha256", ROUTE_AMENDMENT_SHA,
            ]
        return command

    @staticmethod
    def parse_eval(log: Path) -> dict[str, Any]:
        lines = [line for line in log.read_text(errors="replace").splitlines() if line.startswith("[HIGHSTEP_EVAL_JSON] ")]
        if len(lines) != 1:
            raise RuntimeError(f"evaluation log has {len(lines)} result rows: {log}")
        payload = json.loads(lines[0].split(" ", 1)[1])
        valid = bool(
            payload.get("reset_valid") is True and payload.get("schedule_valid") is True
            and payload.get("runtime_snapshot_checkpoint_sha256_verified") is True
            and payload.get("eval_action_delay_runtime_match") is True
            and payload.get("target_action_order_valid") is True
            and payload.get("target_limit_action_input_valid") is True
            and int(payload.get("target_limit_invalid_action_steps", -1)) == 0
            and int(payload.get("samples", -1)) == 138
        )
        safe = bool(valid and float(payload.get("target_limit_violation_fraction", 1.0)) == 0.0)
        return {
            "valid": valid, "safe": safe,
            "full_climb": payload.get("full_climb_success") is True,
            "rear_hold": payload.get("rear_on_platform_hold_success") is True,
            "terminated_early": payload.get("terminated_early"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
        }

    def _evaluate_matrix(
        self,
        *,
        label: str,
        checkpoint: Path,
        scenarios: Sequence[Mapping[str, Any]],
        root: Path,
        narrow: bool = False,
    ) -> tuple[dict[str, Any], Path]:
        manifest = root / "result.json"
        if manifest.exists():
            return json.loads(manifest.read_text()), manifest
        rows = []
        for index, scenario in enumerate(scenarios, start=1):
            row = None
            # Reuse a completed, fully valid attempt before launching anything.
            # Infrastructure-invalid logs remain immutable evidence and do not
            # force an otherwise valid scene to be rerun.
            for attempt in range(1, 4):
                log = root / f"run{index:02d}_attempt{attempt}.log"
                if not log.exists():
                    continue
                try:
                    candidate = self.parse_eval(log)
                except Exception:
                    continue
                if candidate["valid"] and candidate["checkpoint_sha256"] == sha(checkpoint):
                    candidate.update({
                        "run": index, "scenario": dict(scenario), "attempt": attempt,
                        "log": str(log.resolve()), "log_sha256": sha(log),
                    })
                    row = candidate
                    break
            if row is not None:
                rows.append(row)
                continue
            for attempt in range(1, 4):
                log = root / f"run{index:02d}_attempt{attempt}.log"
                phase = f"{label}_run{index:02d}_attempt{attempt}"
                self.update(
                    status="evaluating", phase=phase,
                    evaluation_completed=index - 1, evaluation_total=len(scenarios),
                )
                if not log.exists():
                    try:
                        self.run_child(
                            self.eval_command(checkpoint, scenario=scenario, narrow=narrow),
                            log,
                            phase,
                        )
                    except RuntimeError:
                        continue
                try:
                    row = self.parse_eval(log)
                except Exception:
                    continue
                if row["valid"]:
                    row.update({
                        "run": index, "scenario": dict(scenario), "attempt": attempt,
                        "log": str(log.resolve()), "log_sha256": sha(log),
                    })
                    break
            if row is None or not row["valid"]:
                raise RuntimeError(f"{label} run {index} remained infrastructure-invalid")
            if row["checkpoint_sha256"] != sha(checkpoint):
                raise RuntimeError(f"{label} run {index} checkpoint binding mismatch")
            rows.append(row)
        counts = {
            "valid": sum(row["valid"] for row in rows),
            "full_climb": sum(row["full_climb"] for row in rows),
            "rear_hold": sum(row["rear_hold"] for row in rows),
            "safe": sum(row["safe"] for row in rows),
        }
        result = {
            "schema_version": 1, "kind": "b300_hybrid_behavior15", "label": label,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint),
            "rows": rows, "counts": counts,
            "passed": counts["valid"] == 15 and counts["full_climb"] >= 12 and counts["rear_hold"] >= 12 and counts["safe"] == 15,
            "completed_at": now(),
        }
        atomic_json(manifest, result, read_only=True)
        self.update(centerline_counts=counts, evaluation_completed=15, evaluation_total=15)
        return result, manifest

    def evaluate15(self, stage: int, checkpoint: Path) -> tuple[dict[str, Any], Path]:
        nominal = {
            "name": "nominal", "gap_m": 0.55, "lateral_m": 0.0,
            "yaw_deg": 0.0, "joint_delta_sign": 0,
        }
        return self._evaluate_matrix(
            label=f"E{stage}_centerline",
            checkpoint=checkpoint,
            scenarios=[nominal] * 15,
            root=FLOW / "stages" / f"E{stage}" / "centerline15",
        )

    def collect_dagger_round(self, round_id: int, checkpoint: Path) -> Path:
        root = FLOW / "dagger" / f"round{round_id}" / "collection"
        manifests: list[Path] = []
        teacher_env = TEACHER.parent / "params/env.yaml"
        for index, scenario in enumerate(self.route["scenarios"], start=1):
            run_root = root / f"run{index:02d}_{scenario['name']}"
            result_manifest = run_root / "tensor" / "manifest.json"
            if result_manifest.exists():
                manifests.append(result_manifest)
                continue
            stage_manifest = run_root / "collection_stage.json"
            stage_payload = {
                "schema_version": 1,
                "kind": "highstep_b300_hybrid_dagger_collection_stage",
                "workflow_id": WORKFLOW,
                "round": round_id,
                "scenario": dict(scenario),
                "route_amendment": str(ROUTE_AMENDMENT),
                "route_amendment_sha256": ROUTE_AMENDMENT_SHA,
                "preregistration_sha256": PREREG_SHA,
                "teacher_checkpoint": str(TEACHER),
                "teacher_sha256": TEACHER_SHA,
                "teacher_env_config": str(teacher_env),
                "teacher_env_config_sha256": sha(teacher_env),
                "student_checkpoint": str(checkpoint.resolve()),
                "student_checkpoint_sha256": sha(checkpoint),
                "student_drives_physics": True,
                "teacher_labels_same_pre_step_state": True,
                "delay_steps": 0,
            }
            if not stage_manifest.exists():
                atomic_json(stage_manifest, stage_payload, read_only=True)
            elif json.loads(stage_manifest.read_text()) != stage_payload:
                raise RuntimeError(f"DAgger collection stage changed: {stage_manifest}")
            command = self.eval_command(checkpoint, scenario=scenario, narrow=True) + [
                "--b300_hybrid_dagger_output", str(run_root / "tensor"),
                "--b300_hybrid_dagger_stage_manifest", str(stage_manifest),
                "--b300_hybrid_dagger_stage_manifest_sha256", sha(stage_manifest),
            ]
            self.update(
                status="collecting_dagger", phase=f"D{round_id}_collect_run{index:02d}",
                dagger_round=round_id, dagger_collection_completed=index - 1,
                dagger_collection_total=len(self.route["scenarios"]),
            )
            self.run_child(command, run_root / "collection.log", f"D{round_id}_collect_run{index:02d}")
            if not result_manifest.exists():
                raise RuntimeError(f"DAgger collection produced no manifest: {run_root}")
            manifests.append(result_manifest)

        aggregate_root = FLOW / "dagger" / f"round{round_id}" / "aggregate"
        aggregate_manifest = aggregate_root / "manifest.json"
        if not aggregate_manifest.exists():
            command = [
                sys.executable, str(MERGE),
                "--route-amendment", str(ROUTE_AMENDMENT),
                "--route-amendment-sha256", ROUTE_AMENDMENT_SHA,
                "--round", str(round_id),
                "--canonical-manifest", str(FLOW / "canonical_tensor_trajectory_attempt2/manifest.json"),
                "--canonical-manifest-sha256", "b2794b7dfa872d3d5f297ef08a4bdeebcd6289181907978f49f4c229a8ac53a7",
                "--output", str(aggregate_root),
            ]
            if round_id > 1:
                prior = FLOW / "dagger" / f"round{round_id - 1}" / "aggregate/manifest.json"
                command += [
                    "--prior-aggregate-manifest", str(prior),
                    "--prior-aggregate-manifest-sha256", sha(prior),
                ]
            for manifest in manifests:
                command += ["--dagger-manifest", str(manifest), "--dagger-manifest-sha256", sha(manifest)]
            self.run_child(command, aggregate_root.parent / "merge.log", f"D{round_id}_merge")
        aggregate = json.loads(aggregate_manifest.read_text())
        if not (
            aggregate.get("round") == round_id
            and aggregate.get("unique_episode_count")
            == 1 + int(self.route["collection_trajectories_per_round"]) * round_id
            and aggregate.get("sample_count")
            == 138 * (
                1 + int(self.route["collection_trajectories_per_round"]) * round_id
            )
        ):
            raise RuntimeError(f"DAgger aggregate scope mismatch at round {round_id}")
        return aggregate_manifest

    def dagger_wandb_files(
        self, round_id: int, source: Path, effective_start: int, aggregate_manifest: Path
    ) -> tuple[Path, Path, Path]:
        run_name = f"{WORKFLOW}_D_D{round_id}_attempt1"
        config_path = FLOW / "wandb_stage_configs" / f"D{round_id}.json"
        native_path = FLOW / "wandb_stage_configs" / f"D{round_id}.wandb.yaml"
        aggregate = json.loads(aggregate_manifest.read_text())
        start_step = int(self.checkpoint_payload(source)["iter"])
        updates = int(self.route["training_updates_per_round"])
        config = {
            "workflow_id": WORKFLOW, "route": "D", "stage": f"D{round_id}", "attempt": "attempt1",
            "run_name": run_name, "group": WORKFLOW, "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA, "route_amendment_sha256": ROUTE_AMENDMENT_SHA,
            "start_checkpoint": str(source.resolve()), "start_checkpoint_sha256": sha(source),
            "teacher_checkpoint": str(TEACHER), "teacher_sha256": TEACHER_SHA,
            "training_dataset_manifest": str(aggregate_manifest.resolve()),
            "training_dataset_manifest_sha256": sha(aggregate_manifest),
            "training_dataset_sha256": aggregate["dataset_sha256"],
            "unique_episode_count": aggregate["unique_episode_count"],
            "student_drives_physics": True, "teacher_labels_same_pre_step_state": True,
            "frozen_tensors": ["teacher.*", "critic.*", "student_priv_encoder.*", "actor rows 0:12 and actor body"],
            "trainable_tensors": ["estimator.*", "actor.final.weight[12:16]", "actor.final.bias[12:16]"],
            "optimizer": {"class": "Adam", "resume": "full_preserve", "estimator_lr": 1e-3, "box_rows_lr": 1e-5},
            "budget": {"additional_updates": updates, "effective_update_start": effective_start, "effective_update_target": effective_start + updates},
            "training_task": TASK, "historical_sync": False, "wandb_mode": "online",
            "mixed_target": "pre_prior[0:12]+post_prior_policy_units[12:16]",
            "expected_history_step_start": start_step,
            "expected_history_step_end": start_step + updates - 1,
            "expected_history_unique_steps": updates,
        }
        if not config_path.exists():
            atomic_json(config_path, config, read_only=True)
            native = {"wandb_version": 1, **{key: {"value": value} for key, value in config.items()}}
            native_path.parent.mkdir(parents=True, exist_ok=True)
            native_path.write_text(yaml.safe_dump(native, sort_keys=True, allow_unicode=True))
            native_path.chmod(0o444)
        elif json.loads(config_path.read_text()) != config:
            raise RuntimeError(f"stored W&B config changed at D{round_id}")
        manifest = FLOW / "wandb_stages" / run_name / "stage_manifest.json"
        if not manifest.exists():
            manifest = wandb_gate.prepare_stage(config_path, FLOW)
        return config_path, native_path, manifest

    def train_dagger_round(
        self, round_id: int, source: Path, effective_start: int, aggregate_manifest: Path
    ) -> tuple[Path, Path, Path, int]:
        root = FLOW / "dagger" / f"round{round_id}" / "training"
        result_path = root / "training_result.json"
        aggregate = json.loads(aggregate_manifest.read_text())
        target = effective_start + int(self.route["training_updates_per_round"])
        if result_path.exists():
            result = json.loads(result_path.read_text())
            checkpoint = Path(result["checkpoint"]).resolve(strict=True)
            self.checkpoint_scope(checkpoint, target, training_dataset_sha256=aggregate["dataset_sha256"])
            return Path(result["run_dir"]), checkpoint, Path(result["wandb_stage_manifest"]), target
        _, native, wandb_manifest = self.dagger_wandb_files(
            round_id, source, effective_start, aggregate_manifest
        )
        wb = json.loads(wandb_manifest.read_text())
        command = [
            str(PYTHON), "-u", str(TRAIN.relative_to(ROOT)), "--task", TASK,
            "--num_envs", "4096", "--max_iterations", str(self.route["training_updates_per_round"]),
            "--seed", "42", "--headless", "--logger", "wandb", "--log_project_name", "isaaclab",
            "--resume", "--checkpoint", str(source.resolve()), "--highstep_resume_mode", "refine",
            "--highstep_checkpoint_load_mode", "full", "--highstep_schedule_resume_mode", "preserve",
            "--highstep_parent_teacher_manifest", str(PARENT), "--run_name", wb["run_name"],
        ]
        atomic_json(root / "launch.json", {
            "schema_version": 1, "round": round_id, "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": sha(source), "effective_update_start": effective_start,
            "effective_update_target": target, "aggregate_manifest": str(aggregate_manifest.resolve()),
            "aggregate_manifest_sha256": sha(aggregate_manifest), "command": command,
            "wandb_stage_manifest": str(wandb_manifest), "created_at": now(),
        }, read_only=True)
        environment = wandb_gate.process_environment(wandb_manifest)
        environment.update({
            "WANDB_MODE": "online", "WANDB_X_DISABLE_STATS": "true", "WANDB_DISABLE_GIT": "true",
            "ROBOT_LAB_WANDB_CONFIG_PATH": str(native.resolve()),
            "ROBOT_LAB_WANDB_CONFIG_SHA256": sha(native),
            "HIGHSTEP_B300_HYBRID_TRAINING_DATASET_MANIFEST": str(aggregate_manifest.resolve()),
            "HIGHSTEP_B300_HYBRID_TRAINING_DATASET_MANIFEST_SHA256": sha(aggregate_manifest),
        })
        self.update(
            status="running", phase=f"training_D{round_id}", dagger_round=round_id,
            wandb_run_id=wb["run_id"], wandb_run_url=wb["run_url"],
            wandb_sync_status="online_recording",
        )
        self.run_child(command, root / "train.log", f"training_D{round_id}", env=environment)
        matches = [path for path in EXPERIMENT.glob(f"*_{wb['run_name']}") if path.is_dir()]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one D{round_id} run directory, got {matches}")
        run_dir = matches[0]
        checkpoint = self.find_checkpoint(run_dir, target)
        if checkpoint is None:
            raise RuntimeError(f"D{round_id} ended without complete checkpoint E{target}")
        scope = self.checkpoint_scope(
            checkpoint, target, training_dataset_sha256=aggregate["dataset_sha256"]
        )
        atomic_json(root / "checkpoint_scope.json", scope, read_only=True)
        result = {
            "schema_version": 1, "kind": "b300_hybrid_dagger_training_result",
            "round": round_id, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha(checkpoint), "effective_updates": target,
            "run_dir": str(run_dir.resolve()), "wandb_stage_manifest": str(wandb_manifest.resolve()),
            "aggregate_manifest": str(aggregate_manifest.resolve()),
            "aggregate_manifest_sha256": sha(aggregate_manifest), "completed_at": now(),
        }
        atomic_json(result_path, result, read_only=True)
        self.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint), effective_updates=target)
        return run_dir, checkpoint, wandb_manifest, target

    def finalize_wandb(self, stage: int, checkpoint: Path, behavior: Mapping[str, Any], behavior_manifest: Path, wandb_manifest: Path) -> None:
        summary_path = FLOW / "wandb_stage_summaries" / f"E{stage}.json"
        summary = {
            "output_checkpoint": str(checkpoint.resolve()), "output_checkpoint_sha256": sha(checkpoint),
            "effective_updates": stage,
            "gate_metrics": {"centerline15": dict(behavior["counts"])},
            "gate_conclusion": "passed" if behavior["passed"] else "continue",
            "manifest_path": str(behavior_manifest.resolve()), "manifest_sha256": sha(behavior_manifest),
        }
        if not summary_path.exists():
            atomic_json(summary_path, summary, read_only=True)
        manifest = json.loads(wandb_manifest.read_text())
        config = json.loads((wandb_manifest.parent / "config.snapshot.json").read_text())
        expected_steps = set(range(config["expected_history_step_start"], config["expected_history_step_end"] + 1))
        while True:
            try:
                import wandb
                remote = wandb.Api(timeout=60).run(f"{manifest['entity']}/{manifest['project']}/{manifest['run_id']}")
                steps = wandb_gate.verified_remote_history_steps(remote, expected_steps)
                run = wandb.init(
                    entity=manifest["entity"], project=manifest["project"], id=manifest["run_id"],
                    name=manifest["run_name"], group=manifest["group"], resume="must", mode="online",
                    settings=wandb.Settings(init_timeout=60),
                )
                if run is None:
                    raise RuntimeError("wandb.init returned no run")
                run.config.update(config, allow_val_change=False)
                run.summary.update({**summary, "historical_sync": False})
                run.finish()
                remote = wandb.Api(timeout=60).run(f"{manifest['entity']}/{manifest['project']}/{manifest['run_id']}")
                steps = wandb_gate.verified_remote_history_steps(remote, expected_steps)
                remote_summary = wandb_gate.plain_value(dict(remote.summary))
                if any(remote_summary.get(key) != wandb_gate.plain_value(value) for key, value in {**summary, "historical_sync": False}.items()):
                    raise RuntimeError("remote W&B summary verification failed")
                manifest.update({
                    "sync_status": "synced", "summary_path": str(summary_path.resolve()),
                    "summary_sha256": sha(summary_path), "remote_history_verification": {
                        "expected_unique_steps": len(expected_steps), "remote_unique_steps": len(steps),
                        "step_min": min(expected_steps), "step_max": max(expected_steps), "missing_steps": 0,
                    }, "verified_at": now(),
                })
                manifest.pop("sync_error", None)
                atomic_json(wandb_manifest, manifest)
                self.update(wandb_sync_status="synced")
                return
            except Exception as error:
                self.update(status="pending_sync", phase=f"wandb_sync_E{stage}", wandb_sync_status="pending_remote_verification", last_error=f"{type(error).__name__}: {error}")
                for _ in range(20):
                    if self.stop_requested:
                        raise RuntimeError("stop requested during W&B verification")
                    self.update()
                    time.sleep(3)

    def finalize_dagger_wandb(
        self,
        round_id: int,
        effective_updates: int,
        checkpoint: Path,
        behavior: Mapping[str, Any],
        behavior_manifest: Path,
        wandb_manifest: Path,
    ) -> None:
        summary_path = FLOW / "wandb_stage_summaries" / f"D{round_id}.json"
        summary = {
            "output_checkpoint": str(checkpoint.resolve()),
            "output_checkpoint_sha256": sha(checkpoint),
            "effective_updates": effective_updates,
            "dagger_round": round_id,
            "gate_metrics": {"narrow15": dict(behavior["counts"])},
            "gate_conclusion": "passed" if behavior["passed"] else "continue",
            "manifest_path": str(behavior_manifest.resolve()),
            "manifest_sha256": sha(behavior_manifest),
        }
        if not summary_path.exists():
            atomic_json(summary_path, summary, read_only=True)
        manifest = json.loads(wandb_manifest.read_text())
        config = json.loads((wandb_manifest.parent / "config.snapshot.json").read_text())
        expected_steps = set(range(
            config["expected_history_step_start"], config["expected_history_step_end"] + 1
        ))
        while True:
            try:
                import wandb
                remote = wandb.Api(timeout=60).run(
                    f"{manifest['entity']}/{manifest['project']}/{manifest['run_id']}"
                )
                steps = wandb_gate.verified_remote_history_steps(remote, expected_steps)
                run = wandb.init(
                    entity=manifest["entity"], project=manifest["project"],
                    id=manifest["run_id"], name=manifest["run_name"],
                    group=manifest["group"], resume="must", mode="online",
                    settings=wandb.Settings(init_timeout=60),
                )
                if run is None:
                    raise RuntimeError("wandb.init returned no run")
                run.config.update(config, allow_val_change=False)
                run.summary.update({**summary, "historical_sync": False})
                run.finish()
                remote = wandb.Api(timeout=60).run(
                    f"{manifest['entity']}/{manifest['project']}/{manifest['run_id']}"
                )
                steps = wandb_gate.verified_remote_history_steps(remote, expected_steps)
                remote_summary = wandb_gate.plain_value(dict(remote.summary))
                if any(
                    remote_summary.get(key) != wandb_gate.plain_value(value)
                    for key, value in {**summary, "historical_sync": False}.items()
                ):
                    raise RuntimeError("remote DAgger W&B summary verification failed")
                manifest.update({
                    "sync_status": "synced", "summary_path": str(summary_path.resolve()),
                    "summary_sha256": sha(summary_path),
                    "remote_history_verification": {
                        "expected_unique_steps": len(expected_steps),
                        "remote_unique_steps": len(steps),
                        "step_min": min(expected_steps), "step_max": max(expected_steps),
                        "missing_steps": 0,
                    },
                    "verified_at": now(),
                })
                manifest.pop("sync_error", None)
                atomic_json(wandb_manifest, manifest)
                self.update(wandb_sync_status="synced")
                return
            except Exception as error:
                self.update(
                    status="pending_sync", phase=f"wandb_sync_D{round_id}",
                    wandb_sync_status="pending_remote_verification",
                    last_error=f"{type(error).__name__}: {error}",
                )
                for _ in range(20):
                    if self.stop_requested:
                        raise RuntimeError("stop requested during DAgger W&B verification")
                    self.update()
                    time.sleep(3)

    def generate_videos(
        self,
        label: str,
        checkpoint: Path,
        scenarios: Sequence[Mapping[str, Any]],
    ) -> Path:
        root = FLOW / "visual_review" / label
        clips = []
        for index, scenario in enumerate(scenarios, start=1):
            run = root / f"run{index:02d}"
            delivery = run / f"student_{label}_run{index:02d}.mp4"
            if not delivery.exists():
                phase = f"{label}_video_run{index:02d}"
                self.update(status="recording_video", phase=phase, video_completed=index-1, video_total=15)
                self.run_child(
                    self.eval_command(
                        checkpoint, scenario=scenario, narrow=True,
                        video_dir=run / "raw_video", trace_dir=run / "joint_trace",
                    ),
                    run / "run.log", phase,
                )
                videos = sorted((run / "raw_video").rglob("*.mp4"))
                if len(videos) != 1:
                    raise RuntimeError(f"expected one video for run {index}, found {videos}")
                delivery.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(videos[0], delivery)
            clips.append(delivery)
        concat = root / "concat.txt"
        concat.write_text("".join(f"file '{path}'\n" for path in clips))
        montage = root / f"student_{label}_batch15_montage.mp4"
        if not montage.exists():
            self.run_child(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(montage)], root / "montage.log", f"{label}_video_montage")
        return montage

    def run(self) -> int:
        lock_path = FLOW / "supervisor.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + "\n"); lock.flush()
            self.assert_authority()
            self.require_idle()
            dagger_source = Path(self.route["source_checkpoint"]).resolve(strict=True)
            if sha(dagger_source) != self.route["source_checkpoint_sha256"]:
                raise RuntimeError("route-correction E300 checkpoint SHA mismatch")
            self.checkpoint_scope(dagger_source, 300)
            diagnosis = json.loads(DIAGNOSIS.read_text())
            if not (
                diagnosis.get("status") == "completed_read_only"
                and diagnosis.get("checkpoint_sha256") == sha(dagger_source)
                and diagnosis.get("behavior_counts", {}).get("full_climb") == 0
                and diagnosis.get("behavior_counts", {}).get("rear_hold") == 0
            ):
                raise RuntimeError("E300 causal diagnosis binding mismatch")
            nominal = dict(self.route["scenarios"][0])
            evaluation_matrix = [dict(nominal) for _ in range(int(self.route["gate"]["evaluation_runs"]))]
            dagger_effective = 300
            self.update(
                status="running", phase="route_correction_preflight_complete",
                active_pid=None, checkpoint=str(dagger_source),
                checkpoint_sha256=sha(dagger_source), effective_updates=300,
                next_step="collect one nominal Student-driven D1 trajectory",
            )
            for round_id in range(1, int(self.route["maximum_rounds"]) + 1):
                aggregate_manifest = self.collect_dagger_round(round_id, dagger_source)
                _, dagger_checkpoint, dagger_wb, dagger_effective = self.train_dagger_round(
                    round_id, dagger_source, dagger_effective, aggregate_manifest
                )
                dagger_behavior, dagger_behavior_manifest = self._evaluate_matrix(
                    label=f"D{round_id}_nominal15",
                    checkpoint=dagger_checkpoint,
                    scenarios=evaluation_matrix,
                    root=FLOW / "dagger" / f"round{round_id}" / "nominal15",
                    narrow=True,
                )
                self.finalize_dagger_wandb(
                    round_id, dagger_effective, dagger_checkpoint,
                    dagger_behavior, dagger_behavior_manifest, dagger_wb,
                )
                atomic_json(
                    FLOW / "dagger" / f"round{round_id}" / "round_result.json",
                    {
                        "schema_version": 2, "round": round_id,
                        "checkpoint": str(dagger_checkpoint.resolve()),
                        "checkpoint_sha256": sha(dagger_checkpoint),
                        "effective_updates": dagger_effective,
                        "aggregate_manifest": str(aggregate_manifest.resolve()),
                        "aggregate_manifest_sha256": sha(aggregate_manifest),
                        "behavior_manifest": str(dagger_behavior_manifest.resolve()),
                        "behavior_manifest_sha256": sha(dagger_behavior_manifest),
                        "counts": dagger_behavior["counts"],
                        "passed": dagger_behavior["passed"], "completed_at": now(),
                    },
                    read_only=True,
                )
                if dagger_behavior["passed"]:
                    label = f"D{round_id}_E{dagger_effective}"
                    montage = self.generate_videos(
                        label, dagger_checkpoint, evaluation_matrix
                    )
                    self.update(
                        status="student_fixed_motion_candidate_pending_user_visual_review",
                        phase="pending_user_visual_review",
                        checkpoint=str(dagger_checkpoint.resolve()),
                        checkpoint_sha256=sha(dagger_checkpoint),
                        effective_updates=dagger_effective,
                        dagger_round=round_id,
                        narrow_counts=dagger_behavior["counts"],
                        montage=str(montage.resolve()),
                        next_step="wait for user visual review; do not deploy",
                    )
                    return 0
                dagger_source = dagger_checkpoint
            self.update(
                status="stopped_by_gate", phase="D3_nominal_gate_failed",
                checkpoint=str(dagger_source.resolve()),
                checkpoint_sha256=sha(dagger_source),
                effective_updates=dagger_effective,
                next_step="report D3 result; do not add training or branches",
            )
            return 0


def main() -> int:
    supervisor = Supervisor()
    try:
        return supervisor.run()
    except BlockingIOError:
        return 0
    except Exception as error:
        supervisor.update(
            status="infrastructure_error_requires_repair", phase="fail_closed",
            active_pid=None, last_error=f"{type(error).__name__}: {error}",
            next_step="repair only the concrete infrastructure fault, then resume from latest complete checkpoint",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
