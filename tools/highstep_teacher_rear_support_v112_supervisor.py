#!/usr/bin/env python3
"""Fail-closed long-run supervisor for the v1.12 rear-support Teacher route."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Mapping

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORK = ROOT / "tmp/highstep_teacher_rear_support_v112_20260715"
WORKFLOW = "highstep_teacher_rear_support_v112_20260715"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
PREREG = Path(
    os.environ.get(
        "HIGHSTEP_V112_PREREGISTRATION",
        WORK / "preregistration_v112.json",
    )
)
SPEC_SHA = os.environ.get("HIGHSTEP_V112_SPEC_SHA256", "")
PREREG_SHA = os.environ.get("HIGHSTEP_V112_PREREGISTRATION_SHA256", "")
RUNTIME_BINDING = Path(os.environ.get("HIGHSTEP_V112_RUNTIME_BINDING", ""))
RUNTIME_BINDING_SHA = os.environ.get("HIGHSTEP_V112_RUNTIME_BINDING_SHA256", "")
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
WANDB = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/wandb")
WANDB_ENTITY = "xinqili551-the-university-of-hong-kong"
WANDB_PROJECT = "isaaclab"
CANONICAL_SOURCE = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
SOURCE_DIR = WORK / "source_model_172300_v112"
SOURCE = SOURCE_DIR / "model_172300.pt"
SOURCE_SCHEDULE = SOURCE_DIR / "highstep_schedule_manifest.json"
SOURCE_RUNTIME = SOURCE_DIR / "highstep_runtime_state.json"
SOURCE_MIGRATION = SOURCE_DIR / "source_migration_manifest.json"
LEGACY_ALGORITHM_STATE_MIGRATION = (
    WORK / "manifests/verified_legacy_teacher_algorithm_state_migration.json"
)
SOURCE_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
SOURCE_SCHEDULE_SHA = "3cb611b1aed8b014c289b60791b0782f56172b47f86a6901daf77f825c80403f"
SOURCE_RUNTIME_SHA = "14ffddc8f522b1e6b416386df277e06d528d9f170802c4506baaaf102274a294"
SOURCE_MIGRATION_SHA = "909d206f7e2f44fdb1478897bddf936fcf5f3149b62a2e27b5339c996be4f77d"
LEGACY_ALGORITHM_STATE_MIGRATION_SHA = (
    "b280fb6d0709f405fb1f5336bd0509f1b100d9796e4871a891d3213ba9be1512"
)
TASK = "RobotLab-Isaac-Velocity-HighstepRearSupportV112-ArcdogAdjustableLeg-v0"
EXPERIMENT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_rear_support_v112_Teacher"
)
SERVICE = "highstep-teacher-rear-support-v112.service"
SOURCE_ITERATION = 172300
ADDITIONAL_UPDATES = 6000
TARGET_FINAL_ITERATION = SOURCE_ITERATION + ADDITIONAL_UPDATES - 1
EVAL_OFFSETS = (500, 1000, 2000, 4000, 6000)
MAX_INFRA_RETRIES = 8


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)
    if read_only:
        path.chmod(0o444)


def checkpoint_scope(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "iter"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"incomplete Teacher checkpoint: {path}")
    optimizer = payload["optimizer_state_dict"]
    if not isinstance(optimizer, dict) or not optimizer.get("state") or len(optimizer.get("param_groups", [])) != 1:
        raise RuntimeError(f"Teacher optimizer state is incomplete: {path}")
    iteration = int(payload["iter"])
    if iteration < SOURCE_ITERATION:
        raise RuntimeError(f"Teacher checkpoint iteration regressed: {iteration}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "iteration": iteration,
        "model_tensor_count": len(payload["model_state_dict"]),
        "optimizer_state_entries": len(optimizer["state"]),
        "optimizer_param_groups": len(optimizer["param_groups"]),
    }


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def tagged_eval(log: Path) -> dict[str, Any]:
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("[HIGHSTEP_EVAL_JSON] "):
            rows.append(json.loads(line[len("[HIGHSTEP_EVAL_JSON] ") :]))
    if len(rows) != 1:
        raise RuntimeError(f"expected one canonical evaluation row in {log}, got {len(rows)}")
    return rows[0]


def no_severe_inward(outcome: Mapping[str, Any]) -> bool:
    """Preserve the canonical fail-closed centerline safety definition."""
    return bool(
        float(outcome.get("critical_rear_min_abs_y_q05", -1.0)) >= 0.04
        and float(outcome.get("critical_rear_width_q05", -1.0)) >= 0.18
        and float(outcome.get("critical_center_violation_rate", 1.0)) <= 0.10
        and float(outcome.get("critical_width_violation_rate", 1.0)) <= 0.10
    )


class Supervisor:
    def __init__(self) -> None:
        WORK.mkdir(parents=True, exist_ok=True)
        self.lock = (WORK / "supervisor.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = WORK / "state.json"
        self.heartbeat_path = WORK / "heartbeat.json"
        self.handoff_path = WORK / "handoff.json"
        self.active: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self._validate_authority()
        self.state = self._load_state()

    def _validate_authority(self) -> None:
        if not SPEC_SHA or not PREREG_SHA or not RUNTIME_BINDING_SHA:
            raise RuntimeError("v1.12 service authority environment is incomplete")
        if sha256_file(SPEC) != SPEC_SHA or sha256_file(PREREG) != PREREG_SHA:
            raise RuntimeError("v1.12 spec/preregistration binding mismatch")
        if PREREG.stat().st_mode & 0o222:
            raise RuntimeError("v1.12 preregistration must be read-only")
        if not RUNTIME_BINDING.is_file() or sha256_file(RUNTIME_BINDING) != RUNTIME_BINDING_SHA:
            raise RuntimeError("v1.12 runtime rebinding mismatch")
        if RUNTIME_BINDING.stat().st_mode & 0o222:
            raise RuntimeError("v1.12 runtime rebinding must be read-only")
        if sha256_file(CANONICAL_SOURCE) != SOURCE_SHA or sha256_file(SOURCE) != SOURCE_SHA:
            raise RuntimeError("protected/canonical model_172300 SHA mismatch")
        for path, expected in (
            (SOURCE_SCHEDULE, SOURCE_SCHEDULE_SHA),
            (SOURCE_RUNTIME, SOURCE_RUNTIME_SHA),
            (SOURCE_MIGRATION, SOURCE_MIGRATION_SHA),
            (LEGACY_ALGORITHM_STATE_MIGRATION, LEGACY_ALGORITHM_STATE_MIGRATION_SHA),
        ):
            if not path.is_file() or sha256_file(path) != expected or path.stat().st_mode & 0o222:
                raise RuntimeError(f"v1.12 source schedule migration changed: {path}")
        source_scope = checkpoint_scope(SOURCE)
        if source_scope["iteration"] != SOURCE_ITERATION:
            raise RuntimeError("protected model_172300 iteration mismatch")

        prereg = json.loads(PREREG.read_text())
        if not (
            prereg.get("workflow_id") == WORKFLOW
            and prereg.get("authority", {}).get("version") == "v1.12"
            and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA
            and prereg.get("single_semantic_variable", {}).get("name")
            == "rear_support_motion_contract"
            and Path(prereg.get("checkpoint", {}).get("path", "")).resolve()
            == SOURCE.resolve()
            and prereg.get("checkpoint", {}).get("sha256") == SOURCE_SHA
            and prereg.get("checkpoint", {}).get("legacy_algorithm_state_migration_path")
            == str(LEGACY_ALGORITHM_STATE_MIGRATION)
            and prereg.get("checkpoint", {}).get("legacy_algorithm_state_migration_sha256")
            == LEGACY_ALGORITHM_STATE_MIGRATION_SHA
            and prereg.get("training", {}).get("additional_updates") == ADDITIONAL_UPDATES
            and prereg.get("training", {}).get("target_final_iteration")
            == TARGET_FINAL_ITERATION
        ):
            raise RuntimeError("v1.12 preregistration identity changed")
        for path_text, digest in prereg.get("code_sha256", {}).items():
            path = Path(path_text)
            if sha256_file(path) != digest:
                raise RuntimeError(f"v1.12 bound code changed: {path}")

        binding = json.loads(RUNTIME_BINDING.read_text())
        if not (
            binding.get("workflow_id") == WORKFLOW
            and binding.get("spec_sha256") == SPEC_SHA
            and binding.get("preregistration_sha256") == PREREG_SHA
            and binding.get("supervisor_sha256") == sha256_file(Path(__file__))
            and binding.get("source_checkpoint_sha256") == SOURCE_SHA
            and binding.get("source_schedule_sha256") == SOURCE_SCHEDULE_SHA
            and binding.get("source_runtime_sha256") == SOURCE_RUNTIME_SHA
            and binding.get("source_migration_sha256") == SOURCE_MIGRATION_SHA
            and binding.get("legacy_algorithm_state_migration_sha256")
            == LEGACY_ALGORITHM_STATE_MIGRATION_SHA
        ):
            raise RuntimeError("v1.12 runtime binding content mismatch")

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if not (
                state.get("workflow_id") == WORKFLOW
                and state.get("spec_sha256") == SPEC_SHA
                and state.get("preregistration_sha256") == PREREG_SHA
            ):
                raise RuntimeError("stored v1.12 state authority mismatch")
            return state
        return {
            "schema_version": 1,
            "workflow_id": WORKFLOW,
            "authority_version": "v1.12",
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG),
            "preregistration_sha256": PREREG_SHA,
            "status": "preflight",
            "phase": "preflight",
            "failure_class": "none",
            "supervisor_pid": os.getpid(),
            "active_pid": None,
            "source_checkpoint": str(SOURCE),
            "source_checkpoint_sha256": SOURCE_SHA,
            "source_iteration": SOURCE_ITERATION,
            "target_final_iteration": TARGET_FINAL_ITERATION,
            "effective_updates": 0,
            "infrastructure_retry_count": 0,
            "requires_user_action": False,
            "updated_at": now(),
        }

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["supervisor_pid"] = os.getpid()
        self.state["updated_at"] = now()
        atomic_json(self.state_path, self.state)
        atomic_json(
            self.heartbeat_path,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "status": self.state.get("status"),
                "phase": self.state.get("phase"),
                "supervisor_pid": os.getpid(),
                "active_pid": self.state.get("active_pid"),
                "effective_updates": self.state.get("effective_updates", 0),
                "checkpoint": self.state.get("checkpoint"),
                "wandb_run_id": self.state.get("wandb_run_id"),
                "written_at": now(),
                "written_epoch": time.time(),
            },
        )

    def _current_iteration_from_log(self, log: Path) -> int | None:
        if not log.exists():
            return None
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 256 * 1024))
            text = stream.read().decode(errors="replace")
        matches = re.findall(r"Learning iteration\s+(\d+)\s*/", text)
        return int(matches[-1]) if matches else None

    def _wait(self, process: subprocess.Popen[str], log: Path, phase: str) -> int:
        if process.poll() is not None:
            raise RuntimeError(f"{phase} child exited before PID verification")
        status = (Path("/proc") / str(process.pid) / "status").read_text()
        parent = int(
            next(line for line in status.splitlines() if line.startswith("PPid:"))
            .split()[1]
        )
        if parent != os.getpid():
            raise RuntimeError(f"{phase} child parent mismatch: {parent}")
        witness = WORK / "launch_verifications" / f"{phase}_{int(time.time())}.json"
        atomic_json(
            witness,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "phase": phase,
                "supervisor_pid": os.getpid(),
                "child_pid": process.pid,
                "child_parent_pid": parent,
                "log": str(log),
                "verified_at": now(),
            },
            read_only=True,
        )
        while process.poll() is None:
            if self.stop_requested:
                os.killpg(process.pid, signal.SIGINT)
            iteration = self._current_iteration_from_log(log)
            values: dict[str, Any] = {
                "status": "running",
                "phase": phase,
                "active_pid": process.pid,
                "latest_log": str(log),
                "progress_bytes": log.stat().st_size if log.exists() else 0,
            }
            if iteration is not None and iteration >= SOURCE_ITERATION:
                values["current_iteration"] = iteration
                values["effective_updates"] = iteration - SOURCE_ITERATION + 1
            self.update(**values)
            time.sleep(5)
        rc = int(process.returncode or 0)
        self.active = None
        self.update(active_pid=None)
        return rc

    def _train_command(
        self,
        *,
        checkpoint: Path,
        iterations: int,
        num_envs: int,
        run_name: str,
        logger: str,
    ) -> list[str]:
        command = [
            str(PYTHON),
            "-u",
            "scripts/rsl_rl/base/train.py",
            "--task",
            TASK,
            "--num_envs",
            str(num_envs),
            "--max_iterations",
            str(iterations),
            "--seed",
            "42",
            "--headless",
            "--logger",
            logger,
            "--log_project_name",
            "isaaclab",
            "--resume",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--highstep_resume_mode",
            "refine",
            "--highstep_checkpoint_load_mode",
            "full",
            "--highstep_schedule_resume_mode",
            "preserve",
            "--allow_legacy_highstep_schedule_fallback",
            "--run_name",
            run_name,
        ]
        if checkpoint.resolve() == SOURCE.resolve() and sha256_file(checkpoint) == SOURCE_SHA:
            command.extend(
                [
                    "--verified_legacy_teacher_algorithm_state_manifest",
                    str(LEGACY_ALGORITHM_STATE_MIGRATION),
                    "--verified_legacy_teacher_algorithm_state_manifest_sha256",
                    LEGACY_ALGORITHM_STATE_MIGRATION_SHA,
                ]
            )
        return command

    def _launch_training(
        self,
        *,
        checkpoint: Path,
        iterations: int,
        num_envs: int,
        run_name: str,
        phase: str,
        logger: str,
        wandb_run_id: str | None = None,
    ) -> tuple[int, Path | None]:
        log = WORK / "logs" / f"{run_name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = self._train_command(
            checkpoint=checkpoint,
            iterations=iterations,
            num_envs=num_envs,
            run_name=run_name,
            logger=logger,
        )
        launch = WORK / "launches" / f"{run_name}.json"
        if not launch.exists():
            atomic_json(
                launch,
                {
                    "schema_version": 1,
                    "workflow_id": WORKFLOW,
                    "phase": phase,
                    "command": command,
                    "source_checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                    "iterations": iterations,
                    "num_envs": num_envs,
                    "wandb_run_id": wandb_run_id,
                    "created_at": now(),
                },
                read_only=True,
            )
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "WANDB_RUN_GROUP": WORKFLOW,
                "WANDB_NAME": run_name,
            }
        )
        if logger == "wandb":
            env.update(
                {
                    "WANDB_MODE": "offline",
                    "WANDB_RUN_ID": str(wandb_run_id),
                    "WANDB_RESUME": "allow",
                }
            )
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[V112_SUPERVISOR_LAUNCH] {json.dumps({'time': now(), 'command': command})}\n")
            stream.flush()
            self.active = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.update(
                status="running",
                phase=phase,
                active_pid=self.active.pid,
                latest_log=str(log),
                wandb_run_id=wandb_run_id,
            )
            rc = self._wait(self.active, log, phase)
        run_dirs = sorted(EXPERIMENT.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime)
        return rc, run_dirs[-1] if run_dirs else None

    def smoke(self) -> None:
        marker = WORK / "smoke/smoke_result.json"
        if marker.exists():
            payload = json.loads(marker.read_text())
            if payload.get("passed") is not True:
                raise RuntimeError("stored v1.12 smoke did not pass")
            self.update(phase="smoke_reused")
            return
        source_before = sha256_file(SOURCE)
        run_name = f"v112_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
        rc, run_dir = self._launch_training(
            checkpoint=SOURCE,
            iterations=3,
            num_envs=64,
            run_name=run_name,
            phase="smoke_3_updates",
            logger="tensorboard",
        )
        if rc != 0 or run_dir is None:
            raise RuntimeError(f"v1.12 smoke failed: rc={rc}, run_dir={run_dir}")
        checkpoint = latest_checkpoint(run_dir)
        if checkpoint is None:
            raise RuntimeError("v1.12 smoke produced no checkpoint")
        scope = checkpoint_scope(checkpoint)
        if scope["iteration"] < SOURCE_ITERATION + 2 or sha256_file(SOURCE) != source_before:
            raise RuntimeError("v1.12 smoke checkpoint/source integrity failed")
        atomic_json(
            marker,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "passed": True,
                "source_checkpoint_sha256_before": source_before,
                "source_checkpoint_sha256_after": sha256_file(SOURCE),
                "run_dir": str(run_dir),
                "checkpoint_scope": scope,
                "completed_at": now(),
            },
            read_only=True,
        )
        self.update(phase="smoke_passed", smoke_checkpoint=str(checkpoint))

    def _existing_training_source(self) -> tuple[Path, int]:
        latest: tuple[int, Path] | None = None
        for run_dir in EXPERIMENT.glob("*_v112_long_*"):
            checkpoint = latest_checkpoint(run_dir)
            if checkpoint is None:
                continue
            try:
                scope = checkpoint_scope(checkpoint)
            except Exception:
                continue
            iteration = int(scope["iteration"])
            if latest is None or iteration > latest[0]:
                latest = (iteration, checkpoint)
        if latest is None:
            return SOURCE, SOURCE_ITERATION
        return latest[1], latest[0]

    def long_train(self) -> tuple[Path, Path]:
        complete = WORK / "training/long_training_result.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            checkpoint = Path(record["checkpoint"]).resolve(strict=True)
            run_dir = Path(record["run_dir"]).resolve(strict=True)
            scope = checkpoint_scope(checkpoint)
            if scope["sha256"] != record["checkpoint_sha256"] or scope["iteration"] < TARGET_FINAL_ITERATION:
                raise RuntimeError("stored v1.12 long-training result changed")
            return run_dir, checkpoint

        wandb_run_id = str(self.state.get("wandb_run_id") or f"v112{int(time.time()) % 100000000:08d}")
        retry = int(self.state.get("infrastructure_retry_count", 0))
        while retry <= MAX_INFRA_RETRIES:
            checkpoint, iteration = self._existing_training_source()
            if iteration >= TARGET_FINAL_ITERATION:
                run_dir = checkpoint.parent
                break
            remaining = TARGET_FINAL_ITERATION - iteration + 1
            run_name = f"v112_long_attempt{retry + 1}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.update(
                status="running",
                phase=f"long_training_attempt{retry + 1}",
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256_file(checkpoint),
                current_iteration=iteration,
                effective_updates=max(0, iteration - SOURCE_ITERATION + 1),
                remaining_iterations=remaining,
                wandb_run_id=wandb_run_id,
            )
            rc, run_dir = self._launch_training(
                checkpoint=checkpoint,
                iterations=remaining,
                num_envs=4096,
                run_name=run_name,
                phase=f"long_training_attempt{retry + 1}",
                logger="wandb",
                wandb_run_id=wandb_run_id,
            )
            if run_dir is not None:
                candidate = latest_checkpoint(run_dir)
                if candidate is not None:
                    scope = checkpoint_scope(candidate)
                    self.update(
                        checkpoint=str(candidate),
                        checkpoint_sha256=scope["sha256"],
                        current_iteration=scope["iteration"],
                        effective_updates=scope["iteration"] - SOURCE_ITERATION + 1,
                    )
                    if scope["iteration"] >= TARGET_FINAL_ITERATION:
                        checkpoint = candidate
                        break
            if rc == 0:
                raise RuntimeError("v1.12 long training exited cleanly before target checkpoint")
            retry += 1
            self.update(
                status="infrastructure_retry",
                phase="long_training_retry_boundary",
                infrastructure_retry_count=retry,
                failure_class="training_infrastructure_retryable",
                last_error=f"training child rc={rc}; resuming latest complete checkpoint",
            )
        else:
            raise RuntimeError("v1.12 long training exhausted infrastructure retries")

        scope = checkpoint_scope(checkpoint)
        if scope["iteration"] < TARGET_FINAL_ITERATION:
            raise RuntimeError("v1.12 long training target not reached")
        atomic_json(
            complete,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "source_checkpoint": str(SOURCE),
                "canonical_source_checkpoint": str(CANONICAL_SOURCE),
                "source_checkpoint_sha256": SOURCE_SHA,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": scope["sha256"],
                "checkpoint_scope": scope,
                "effective_updates": scope["iteration"] - SOURCE_ITERATION + 1,
                "wandb_run_id": wandb_run_id,
                "wandb_sync_status": "pending_sync",
                "completed_at": now(),
            },
            read_only=True,
        )
        self.update(
            status="post_training",
            phase="long_training_complete",
            checkpoint=str(checkpoint),
            checkpoint_sha256=scope["sha256"],
            current_iteration=scope["iteration"],
            effective_updates=scope["iteration"] - SOURCE_ITERATION + 1,
            remaining_iterations=0,
            failure_class="none",
        )
        return run_dir, checkpoint

    def _checkpoint_for_offset(self, run_dir: Path, offset: int) -> Path | None:
        target = SOURCE_ITERATION + offset - 1
        candidates: list[tuple[int, Path]] = []
        run_dirs = {run_dir, *EXPERIMENT.glob("*_v112_long_*")}
        for candidate_dir in run_dirs:
            exact = candidate_dir / f"model_{target}.pt"
            if exact.exists():
                return exact
            for path in candidate_dir.glob("model_*.pt"):
                match = re.fullmatch(r"model_(\d+)\.pt", path.name)
                if match and int(match.group(1)) <= target:
                    candidates.append((int(match.group(1)), path))
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def evaluate(self, run_dir: Path, final_checkpoint: Path) -> Path:
        result_path = WORK / "evaluations/posttrain_summary.json"
        if result_path.exists():
            return result_path
        rows = []
        for offset in EVAL_OFFSETS:
            checkpoint = self._checkpoint_for_offset(run_dir, offset)
            if checkpoint is None and offset == ADDITIONAL_UPDATES:
                checkpoint = final_checkpoint
            if checkpoint is None:
                continue
            for seed in (11, 22, 33):
                out = WORK / "evaluations" / f"E{offset}" / f"seed{seed}"
                out.mkdir(parents=True, exist_ok=True)
                log = out / "play.log"
                stored = out / "result.json"
                if stored.exists():
                    rows.append(json.loads(stored.read_text()))
                    continue
                command = [
                    str(PYTHON),
                    "scripts/rsl_rl/base/play.py",
                    "--task",
                    TASK,
                    "--num_envs",
                    "1",
                    "--headless",
                    "--seed",
                    str(seed),
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--play_terrain_type",
                    "box",
                    "--play_terrain_level",
                    "9",
                    "--fixed_velocity_command",
                    "0.45",
                    "0.0",
                    "0.0",
                    "--reset_after_play_terrain_selection",
                    "--front_step_eval_reset",
                    "--front_step_eval_side",
                    "x-",
                    "--front_step_eval_edge_gap",
                    "0.55",
                    "--front_step_eval_lateral_offset",
                    "0.0",
                    "--front_step_eval_yaw_offset_deg",
                    "0.0",
                    "--print_rear_width_metrics",
                    "--skip_policy_export",
                    "--rear_width_metric_interval",
                    "60",
                    "--play_max_steps",
                    "600",
                    "--eval_action_delay_steps",
                    "0",
                    "--allow_legacy_highstep_schedule_fallback",
                ]
                with log.open("w", encoding="utf-8") as stream:
                    self.active = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=os.environ.copy(),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                    rc = self._wait(self.active, log, f"posttrain_eval_E{offset}_seed{seed}")
                if rc != 0:
                    raise RuntimeError(f"v1.12 posttrain evaluation failed: E{offset}/seed{seed}, rc={rc}")
                canonical = tagged_eval(log)
                valid = bool(
                    canonical.get("checkpoint_sha256") == sha256_file(checkpoint)
                    and canonical.get("schedule_valid") is True
                    and canonical.get("reset_valid") is True
                    and canonical.get("action_prior_enabled") is True
                    and canonical.get("eval_action_delay_steps_runtime") == 0
                )
                if not valid:
                    raise RuntimeError(f"v1.12 invalid evaluation evidence: E{offset}/seed{seed}")
                row = {
                    "schema_version": 1,
                    "workflow_id": WORKFLOW,
                    "offset": offset,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "valid": True,
                    "full_climb": canonical.get("full_climb_success") is True,
                    "rear_hold": canonical.get("rear_on_platform_hold_success") is True,
                    "no_severe_inward": no_severe_inward(canonical),
                    "rear_width_min": canonical.get("rear_width_min"),
                    "rear_min_abs_y_min": canonical.get("rear_min_abs_y_min"),
                    "approach_rear_fore_aft_error_q95": canonical.get(
                        "approach_rear_fore_aft_error_q95"
                    ),
                    "approach_rear_lateral_center_error_q95": canonical.get(
                        "approach_rear_lateral_center_error_q95"
                    ),
                    "critical_rear_fore_aft_error_q95": canonical.get(
                        "critical_rear_fore_aft_error_q95"
                    ),
                    "critical_rear_lateral_center_error_q95": canonical.get(
                        "critical_rear_lateral_center_error_q95"
                    ),
                    "log": str(log),
                    "log_sha256": sha256_file(log),
                    "completed_at": now(),
                }
                atomic_json(stored, row, read_only=True)
                rows.append(row)
        by_offset: dict[str, Any] = {}
        for offset in EVAL_OFFSETS:
            selected = [row for row in rows if row["offset"] == offset]
            if not selected:
                continue
            by_offset[str(offset)] = {
                "valid": sum(row["valid"] for row in selected),
                "full_climb": sum(row["full_climb"] for row in selected),
                "rear_hold": sum(row["rear_hold"] for row in selected),
                "no_severe_inward": sum(row["no_severe_inward"] for row in selected),
                "checkpoint": selected[0]["checkpoint"],
                "checkpoint_sha256": selected[0]["checkpoint_sha256"],
            }
        ranked = sorted(
            by_offset.items(),
            key=lambda item: (
                min(item[1]["full_climb"], item[1]["rear_hold"]),
                item[1]["full_climb"] + item[1]["rear_hold"],
                item[1]["no_severe_inward"],
                -int(item[0]),
            ),
            reverse=True,
        )
        summary = {
            "schema_version": 1,
            "workflow_id": WORKFLOW,
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "by_offset": by_offset,
            "selected_behavior_best": ranked[0][1] if ranked else None,
            "status": "teacher_rear_support_long_run_complete_pending_motion_review",
            "student_training_started": False,
            "automatic_real_robot_deployment": False,
            "completed_at": now(),
        }
        atomic_json(result_path, summary, read_only=True)
        return result_path

    def sync_wandb(self, final_checkpoint: Path, evaluation: Path) -> Path:
        marker = WORK / "wandb/wandb_sync_verification.json"
        if marker.exists():
            payload = json.loads(marker.read_text())
            if not (
                payload.get("sync_status") == "synced"
                and payload.get("checkpoint_sha256") == sha256_file(final_checkpoint)
                and int(payload.get("remote_unique_steps", 0)) >= ADDITIONAL_UPDATES
            ):
                raise RuntimeError("stored v1.12 W&B sync verification changed")
            return marker

        run_id = str(self.state.get("wandb_run_id") or "")
        if not run_id:
            raise RuntimeError("v1.12 W&B run id is missing")
        fragments = sorted((ROOT / "wandb").glob(f"offline-run-*-{run_id}"))
        if not fragments:
            raise RuntimeError(f"no local W&B fragment found for {run_id}")
        self.update(
            status="wandb_sync",
            phase="wandb_multifragment_sync",
            active_pid=None,
            wandb_fragment_count=len(fragments),
        )
        fragment_records = []
        for index, fragment in enumerate(fragments, start=1):
            log = WORK / "wandb" / f"sync_fragment_{index}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [str(WANDB), "sync", str(fragment)],
                cwd=ROOT,
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                check=False,
            )
            log.write_text(completed.stdout + "\n[stderr]\n" + completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"W&B fragment sync failed for {fragment.name}: rc={completed.returncode}"
                )
            files = {
                str(path.relative_to(fragment)): sha256_file(path)
                for path in sorted(fragment.rglob("*"))
                if path.is_file()
            }
            fragment_records.append(
                {
                    "path": str(fragment),
                    "files_sha256": files,
                    "sync_log": str(log),
                    "sync_log_sha256": sha256_file(log),
                }
            )

        verification_script = """
import json
import wandb
api = wandb.Api(timeout=90)
run = api.run(%r)
rows = list(run.scan_history(page_size=1000))
steps = sorted({int(row[\"_step\"]) for row in rows if row.get(\"_step\") is not None})
run.config.update(%r, allow_val_change=True)
run.summary.update(%r)
run.update()
print(json.dumps({
    \"url\": run.url,
    \"name\": run.name,
    \"group\": run.group,
    \"remote_rows\": len(rows),
    \"remote_unique_steps\": len(steps),
    \"remote_step_min\": min(steps) if steps else None,
    \"remote_step_max\": max(steps) if steps else None,
}, sort_keys=True))
""" % (
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}",
            {
                "workflow_id": WORKFLOW,
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "source_checkpoint": str(SOURCE),
                "canonical_source_checkpoint": str(CANONICAL_SOURCE),
                "source_checkpoint_sha256": SOURCE_SHA,
                "single_semantic_variable": "rear_support_motion_contract",
                "additional_updates": ADDITIONAL_UPDATES,
                "training_task": TASK,
            },
            {
                "output_checkpoint": str(final_checkpoint),
                "output_checkpoint_sha256": sha256_file(final_checkpoint),
                "effective_updates": ADDITIONAL_UPDATES,
                "evaluation_manifest": str(evaluation),
                "evaluation_manifest_sha256": sha256_file(evaluation),
                "gate_conclusion": "teacher_rear_support_long_run_complete_pending_motion_review",
            },
        )
        completed = subprocess.run(
            [str(PYTHON), "-c", verification_script],
            cwd=ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )
        remote_log = WORK / "wandb/remote_verification.log"
        remote_log.write_text(completed.stdout + "\n[stderr]\n" + completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"W&B remote verification failed: rc={completed.returncode}")
        try:
            remote = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as error:
            raise RuntimeError("W&B remote verification returned no JSON") from error
        if (
            remote.get("group") != WORKFLOW
            or int(remote.get("remote_unique_steps", 0)) < ADDITIONAL_UPDATES
        ):
            raise RuntimeError(f"W&B remote history incomplete: {remote}")
        payload = {
            "schema_version": 2,
            "workflow_id": WORKFLOW,
            "run_id": run_id,
            "sync_status": "synced",
            "fragments": fragment_records,
            "remote": remote,
            "remote_unique_steps": remote["remote_unique_steps"],
            "checkpoint": str(final_checkpoint),
            "checkpoint_sha256": sha256_file(final_checkpoint),
            "evaluation": str(evaluation),
            "evaluation_sha256": sha256_file(evaluation),
            "completed_at": now(),
        }
        atomic_json(marker, payload, read_only=True)
        self.update(
            status="post_training",
            phase="wandb_synced",
            wandb_sync_status="synced",
            wandb_sync_manifest=str(marker),
            wandb_remote_url=remote.get("url"),
        )
        return marker

    def run(self) -> None:
        self.update(status="running", phase="authority_verified", failure_class="none")
        self.smoke()
        run_dir, final_checkpoint = self.long_train()
        evaluation = self.evaluate(run_dir, final_checkpoint)
        wandb_sync = self.sync_wandb(final_checkpoint, evaluation)
        handoff = {
            "schema_version": 1,
            "workflow_id": WORKFLOW,
            "status": "teacher_rear_support_long_run_complete_pending_motion_review",
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "source_checkpoint": str(SOURCE),
            "source_checkpoint_sha256": SOURCE_SHA,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "evaluation_summary": str(evaluation),
            "evaluation_summary_sha256": sha256_file(evaluation),
            "wandb_sync_manifest": str(wandb_sync),
            "wandb_sync_manifest_sha256": sha256_file(wandb_sync),
            "next_action": "review rear-only motion evidence before any Student distillation",
            "written_at": now(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update(
            status="teacher_rear_support_long_run_complete_pending_motion_review",
            phase="teacher_rear_support_long_run_complete_pending_motion_review",
            active_pid=None,
            evaluation_summary=str(evaluation),
            evaluation_summary_sha256=sha256_file(evaluation),
            requires_user_action=True,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", SERVICE],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

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
            supervisor.update(
                status="paused",
                phase="paused_at_safe_checkpoint_boundary",
                active_pid=None,
            )
        return 0
    except Exception as error:
        if supervisor is not None:
            supervisor.update(
                status="infrastructure_failed",
                phase="infrastructure_failed",
                failure_class="infrastructure",
                active_pid=None,
                last_error=f"{type(error).__name__}: {error}",
                requires_user_action=True,
            )
            atomic_json(
                supervisor.handoff_path,
                {
                    "schema_version": 1,
                    "workflow_id": WORKFLOW,
                    "status": "infrastructure_failed",
                    "error": f"{type(error).__name__}: {error}",
                    "next_action": "repair infrastructure and resume the latest complete v1.12 checkpoint",
                    "written_at": now(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
