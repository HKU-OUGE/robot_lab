#!/usr/bin/env python3
"""Fail-closed v1.12.2 FL-only finite Teacher-stage supervisor."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any, Mapping

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORK = ROOT / "tmp/highstep_teacher_front_placement_v1122_20260716"
WORKFLOW = "highstep_teacher_front_placement_v1122_20260716"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
PREREG = Path(os.environ.get("HIGHSTEP_V1122_PREREGISTRATION", ""))
SPEC_SHA = os.environ.get("HIGHSTEP_V1122_SPEC_SHA256", "")
PREREG_SHA = os.environ.get("HIGHSTEP_V1122_PREREGISTRATION_SHA256", "")
RUNTIME_BINDING = Path(os.environ.get("HIGHSTEP_V1122_RUNTIME_BINDING", ""))
RUNTIME_BINDING_SHA = os.environ.get("HIGHSTEP_V1122_RUNTIME_BINDING_SHA256", "")
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
SOURCE = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_rear_support_v112_Teacher/"
    "2026-07-15_10-52-45_v112_long_attempt1_20260715_105240/model_173200.pt"
)
SOURCE_PARAMS = SOURCE.parent / "params"
SOURCE_SCHEDULE = SOURCE_PARAMS / "highstep_schedule_manifest.json"
SOURCE_RUNTIME = SOURCE_PARAMS / "highstep_runtime_state.json"
REBOUND_SOURCE_DIR = WORK / "source_model_173200_v1122"
REBOUND_SOURCE = REBOUND_SOURCE_DIR / "model_173200.pt"
REBOUND_SCHEDULE = REBOUND_SOURCE_DIR / "params/highstep_schedule_manifest.json"
REBOUND_RUNTIME = REBOUND_SOURCE_DIR / "params/highstep_runtime_state.json"
SOURCE_SHA = "962fd3ce3983e4a478092be8f7a636e87ed872b79496c4fa3134191838d9b80d"
TASK = "RobotLab-Isaac-Velocity-HighstepRearSupportFrontPlacementV1121-ArcdogAdjustableLeg-v0"
EXPERIMENT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_rear_support_front_placement_v1121_Teacher"
)
SERVICE = "highstep-teacher-front-placement-v1122.service"
WANDB_ENTITY = "xinqili551-the-university-of-hong-kong"
WANDB_PROJECT = "isaaclab"
SOURCE_ITERATION = 173200
SMOKE_UPDATES = 3
FORMAL_UPDATES = 100
TARGET_FINAL_ITERATION = SOURCE_ITERATION + FORMAL_UPDATES - 1
FORMAL_NUM_ENVS = 4096
SEED = 42
SAVE_INTERVAL = 100
EVAL_SEEDS = tuple(range(1101, 1116))
MAX_TRAIN_ATTEMPTS = 3


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    if read_only:
        path.chmod(0o444)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object: {path}")
    return value


def _optimizer_steps(optimizer: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for state in optimizer.get("state", {}).values():
        step = state.get("step") if isinstance(state, Mapping) else None
        if hasattr(step, "item"):
            step = step.item()
        if isinstance(step, (int, float)) and not isinstance(step, bool):
            values.append(float(step))
    return values


def checkpoint_scope(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "iter", "infos"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"incomplete Teacher checkpoint: {path}")
    model = payload["model_state_dict"]
    optimizer = payload["optimizer_state_dict"]
    infos = payload["infos"]
    if not isinstance(model, dict) or not model:
        raise RuntimeError(f"empty Teacher model state: {path}")
    if not isinstance(optimizer, dict) or not optimizer.get("state"):
        raise RuntimeError(f"empty Teacher optimizer state: {path}")
    if len(optimizer.get("param_groups", [])) != 1:
        raise RuntimeError(f"unexpected Teacher optimizer groups: {path}")
    algorithm = (
        infos.get("robot_lab_algorithm_checkpoint_state")
        if isinstance(infos, dict)
        else None
    )
    if not (
        isinstance(algorithm, dict)
        and algorithm.get("schema_version") == 1
        and algorithm.get("algorithm_class") == "VAEPPO"
        and algorithm.get("distill_stage") == 1
        and isinstance(algorithm.get("vae_optimizer_state_dict"), dict)
        and len(algorithm["vae_optimizer_state_dict"].get("param_groups", [])) == 1
    ):
        raise RuntimeError(f"Teacher algorithm-owned checkpoint state is incomplete: {path}")
    steps = _optimizer_steps(optimizer)
    if len(steps) != len(optimizer["state"]):
        raise RuntimeError(f"Teacher optimizer step counters are incomplete: {path}")
    tensor_signature = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(model.items())
        if isinstance(value, torch.Tensor)
    }
    signature_sha = hashlib.sha256(
        json.dumps(tensor_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    group = optimizer["param_groups"][0]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "iteration": int(payload["iter"]),
        "model_tensor_count": len(model),
        "model_tensor_signature_sha256": signature_sha,
        "optimizer_state_entries": len(optimizer["state"]),
        "optimizer_param_groups": len(optimizer["param_groups"]),
        "optimizer_step_min": min(steps),
        "optimizer_step_max": max(steps),
        "optimizer_lr": float(group["lr"]),
        "algorithm_class": algorithm["algorithm_class"],
        "distill_stage": algorithm["distill_stage"],
        "vae_optimizer_state_entries": len(
            algorithm["vae_optimizer_state_dict"].get("state", {})
        ),
        "vae_optimizer_param_groups": len(
            algorithm["vae_optimizer_state_dict"].get("param_groups", [])
        ),
    }


def latest_checkpoint(run_dirs: list[Path], *, at_most: int | None = None) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for run_dir in run_dirs:
        for path in run_dir.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if not match:
                continue
            iteration = int(match.group(1))
            if at_most is None or iteration <= at_most:
                candidates.append((iteration, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def tagged_eval(log: Path) -> dict[str, Any]:
    rows = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[HIGHSTEP_EVAL_JSON] "):
            rows.append(json.loads(line[len("[HIGHSTEP_EVAL_JSON] ") :]))
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError(f"expected one canonical evaluation row in {log}, got {len(rows)}")
    return rows[0]


class Supervisor:
    def __init__(self) -> None:
        WORK.mkdir(parents=True, exist_ok=True)
        self.lock = (WORK / "supervisor.lock").open("a+", encoding="utf-8")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = WORK / "state.json"
        self.heartbeat_path = WORK / "heartbeat.json"
        self.handoff_path = WORK / "handoff.json"
        self.active: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self._prepare_source_rebinding()
        self._validate_authority()
        self.state = self._load_state()

    def _prepare_source_rebinding(self) -> None:
        """Create a byte-identical checkpoint mirror beside the approved sidecars.

        The canonical E1000 checkpoint and its old v1.12 sidecars remain
        untouched.  The new sidecar changes only the schedule-definition
        inventory by adding the already-approved FL reward stage; its runtime
        snapshot is byte-for-byte equivalent as JSON to E1000's exact record.
        """
        if not PREREG.is_file():
            raise RuntimeError("v1.12.2 finite preregistration is missing")
        prereg = read_json(PREREG)
        rebound = prereg.get("source_rebinding", {})
        if not (
            rebound.get("checkpoint") == str(REBOUND_SOURCE)
            and rebound.get("checkpoint_sha256") == SOURCE_SHA
            and rebound.get("schedule_manifest") == str(REBOUND_SCHEDULE)
            and rebound.get("runtime_state") == str(REBOUND_RUNTIME)
            and rebound.get("schedule_reset") is False
            and rebound.get("optimizer_reset") is False
        ):
            raise RuntimeError("v1.12.2 source-rebinding authority is incomplete")
        for path, expected in (
            (REBOUND_SCHEDULE, rebound.get("schedule_manifest_sha256")),
            (REBOUND_RUNTIME, rebound.get("runtime_state_sha256")),
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"v1.12.2 task-only source sidecar changed: {path}")
            if path.stat().st_mode & 0o222:
                raise RuntimeError(f"v1.12.2 task-only source sidecar must be read-only: {path}")
        if REBOUND_SOURCE.exists():
            if not REBOUND_SOURCE.is_file() or sha256_file(REBOUND_SOURCE) != SOURCE_SHA:
                raise RuntimeError("existing v1.12.2 rebound checkpoint is invalid")
        else:
            REBOUND_SOURCE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE, REBOUND_SOURCE)
            if sha256_file(REBOUND_SOURCE) != SOURCE_SHA:
                raise RuntimeError("v1.12.2 rebound checkpoint copy failed SHA verification")
            REBOUND_SOURCE.chmod(0o444)

    def _validate_authority(self) -> None:
        if not SPEC_SHA or not PREREG_SHA or not RUNTIME_BINDING_SHA:
            raise RuntimeError("v1.12.2 service authority environment is incomplete")
        if sha256_file(SPEC) != SPEC_SHA:
            raise RuntimeError("v1.12.2 formal spec SHA mismatch")
        if not PREREG.is_file() or sha256_file(PREREG) != PREREG_SHA:
            raise RuntimeError("v1.12.2 finite preregistration SHA mismatch")
        if PREREG.stat().st_mode & 0o222:
            raise RuntimeError("v1.12.2 finite preregistration must be read-only")
        if not RUNTIME_BINDING.is_file() or sha256_file(RUNTIME_BINDING) != RUNTIME_BINDING_SHA:
            raise RuntimeError("v1.12.2 runtime binding SHA mismatch")
        if RUNTIME_BINDING.stat().st_mode & 0o222:
            raise RuntimeError("v1.12.2 runtime binding must be read-only")
        if sha256_file(SOURCE) != SOURCE_SHA:
            raise RuntimeError("protected v1.12 E1000 source checkpoint changed")

        prereg = read_json(PREREG)
        authority = prereg.get("authority", {})
        source = prereg.get("source", {})
        rebound = prereg.get("source_rebinding", {})
        stage = prereg.get("finite_training_stage", {})
        evaluation = prereg.get("evaluation_gate", {})
        if not (
            prereg.get("workflow_id") == WORKFLOW
            and authority.get("version") == "v1.12.2"
            and authority.get("spec_sha256") == SPEC_SHA
            and source.get("checkpoint") == str(SOURCE)
            and source.get("checkpoint_sha256") == SOURCE_SHA
            and rebound.get("checkpoint") == str(REBOUND_SOURCE)
            and rebound.get("checkpoint_sha256") == SOURCE_SHA
            and rebound.get("schedule_manifest_sha256") == sha256_file(REBOUND_SCHEDULE)
            and rebound.get("runtime_state_sha256") == sha256_file(REBOUND_RUNTIME)
            and stage.get("stage_id") == "FLR-E100"
            and stage.get("additional_updates") == FORMAL_UPDATES
            and stage.get("target_final_iteration") == TARGET_FINAL_ITERATION
            and stage.get("num_envs") == FORMAL_NUM_ENVS
            and stage.get("seed") == SEED
            and stage.get("save_interval") == SAVE_INTERVAL
            and evaluation.get("seeds") == list(EVAL_SEEDS)
            and evaluation.get("valid_required") == len(EVAL_SEEDS)
            and evaluation.get("fl_contacts_allowed") == 0
        ):
            raise RuntimeError("v1.12.2 finite preregistration content mismatch")
        if sha256_file(SOURCE_SCHEDULE) != source.get("schedule_manifest_sha256"):
            raise RuntimeError("v1.12.2 source schedule manifest changed")
        if sha256_file(SOURCE_RUNTIME) != source.get("runtime_state_sha256"):
            raise RuntimeError("v1.12.2 source runtime state changed")
        if sha256_file(REBOUND_SOURCE) != SOURCE_SHA:
            raise RuntimeError("v1.12.2 task-only rebound checkpoint changed")
        for path_text, expected in prereg.get("code_sha256", {}).items():
            path = Path(path_text)
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"v1.12.2 bound runtime code changed: {path}")

        binding = read_json(RUNTIME_BINDING)
        if not (
            binding.get("workflow_id") == WORKFLOW
            and binding.get("spec_sha256") == SPEC_SHA
            and binding.get("preregistration_sha256") == PREREG_SHA
            and binding.get("source_checkpoint_sha256") == SOURCE_SHA
            and binding.get("supervisor_sha256") == sha256_file(Path(__file__))
            and binding.get("task") == TASK
            and binding.get("stage_id") == "FLR-E100"
        ):
            raise RuntimeError("v1.12.2 runtime binding content mismatch")

        dashboard = read_json(DASHBOARD)
        if not (
            dashboard.get("workflow_id") == WORKFLOW
            and dashboard.get("spec_sha256") == SPEC_SHA
            and dashboard.get("preregistration_path") == str(PREREG)
            and dashboard.get("preregistration_sha256") == PREREG_SHA
            and dashboard.get("state_path") == str(self.state_path)
        ):
            raise RuntimeError("v1.12.2 dashboard authority mismatch")
        scope = checkpoint_scope(SOURCE)
        if scope["iteration"] != SOURCE_ITERATION:
            raise RuntimeError("protected v1.12 E1000 source iteration changed")

    def _load_state(self) -> dict[str, Any]:
        state = read_json(self.state_path)
        if not (
            state.get("workflow_id") == WORKFLOW
            and state.get("authority_version") == "v1.12.2"
            and state.get("spec_sha256") == SPEC_SHA
            and state.get("preregistration_path") == str(PREREG)
            and state.get("preregistration_sha256") == PREREG_SHA
        ):
            raise RuntimeError("stored v1.12.2 state authority mismatch")
        return state

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["supervisor_pid"] = os.getpid()
        self.state["updated_at"] = now()
        atomic_json(self.state_path, self.state)
        atomic_json(
            self.heartbeat_path,
            {
                "schema_version": 2,
                "workflow_id": WORKFLOW,
                "authority_version": "v1.12.2",
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "status": self.state.get("status"),
                "phase": self.state.get("phase"),
                "supervisor_pid": os.getpid(),
                "active_pid": self.state.get("active_pid"),
                "active_child_kind": self.state.get("active_child_kind"),
                "checkpoint": self.state.get("checkpoint"),
                "current_iteration": self.state.get("current_iteration"),
                "effective_updates": self.state.get("effective_updates", 0),
                "wandb_run_id": self.state.get("wandb_run_id"),
                "wandb_run_url": self.state.get("wandb_run_url"),
                "wandb_sync_status": self.state.get("wandb_sync_status"),
                "wandb_remote_latest_step": self.state.get("wandb_remote_latest_step"),
                "failure_class": self.state.get("failure_class", "none"),
                "written_at": now(),
                "written_epoch": time.time(),
            },
        )

    @staticmethod
    def _current_iteration_from_log(log: Path) -> int | None:
        if not log.exists():
            return None
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 256 * 1024))
            text = stream.read().decode(errors="replace")
        matches = re.findall(r"Learning iteration\s+(\d+)\s*/", text)
        return int(matches[-1]) if matches else None

    def _pid_witness(self, process: subprocess.Popen[str], phase: str, log: Path) -> None:
        if process.poll() is not None:
            raise RuntimeError(f"{phase} child exited before PID verification")
        status_path = Path("/proc") / str(process.pid) / "status"
        status = status_path.read_text(encoding="utf-8")
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
                "child_process_group": os.getpgid(process.pid),
                "log": str(log),
                "verified_at": now(),
            },
            read_only=True,
        )

    def _required_wandb_config(self) -> dict[str, Any]:
        prereg = read_json(PREREG)
        return {
            "workflow_id": WORKFLOW,
            "authority_version": "v1.12.2",
            "stage_id": "FLR-E100",
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "code_sha256": prereg["code_sha256"],
            "source_checkpoint_sha256": SOURCE_SHA,
            "single_semantic_variable": prereg["single_semantic_variable"],
            "optimizer": prereg["optimizer_binding"],
            "iteration_budget": prereg["finite_training_stage"],
            "task": TASK,
        }

    def _poll_wandb(self, *, child_alive: bool) -> dict[str, Any]:
        import wandb

        run_id = str(self.state.get("wandb_run_id") or "")
        if not run_id:
            raise RuntimeError("W&B run id is missing")
        api = wandb.Api(timeout=20)
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        config = dict(run.config)
        required = self._required_wandb_config()
        mismatches = {
            key: {"actual": config.get(key), "expected": value}
            for key, value in required.items()
            if config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"W&B remote config binding mismatch: {mismatches}")
        if run.group != WORKFLOW:
            raise RuntimeError(f"W&B remote group mismatch: {run.group!r}")
        latest = getattr(run, "lastHistoryStep", None)
        if latest is not None:
            latest = int(latest)
        samples = list(self.state.get("wandb_live_step_samples") or [])
        if child_alive and latest is not None and (
            not samples or int(samples[-1]["remote_latest_step"]) != latest
        ):
            samples.append(
                {
                    "observed_at": now(),
                    "remote_latest_step": latest,
                    "child_pid": self.state.get("active_pid"),
                }
            )
        self.update(
            wandb_run_url=run.url,
            wandb_sync_status=(
                "online_remote_progress_verified" if latest is not None else "online_run_visible"
            ),
            wandb_remote_latest_step=latest,
            wandb_remote_state=run.state,
            wandb_live_step_samples=samples,
            wandb_last_poll_error=None,
        )
        return {
            "run_id": run.id,
            "url": run.url,
            "group": run.group,
            "state": run.state,
            "remote_latest_step": latest,
            "config_verified": True,
        }

    def _wait(
        self,
        process: subprocess.Popen[str],
        log: Path,
        phase: str,
        *,
        monitor_wandb: bool = False,
    ) -> int:
        self._pid_witness(process, phase, log)
        next_remote_poll = time.monotonic() + 10.0
        signal_sent = False
        while process.poll() is None:
            if self.stop_requested and not signal_sent:
                os.killpg(process.pid, signal.SIGINT)
                signal_sent = True
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
                values["effective_updates"] = min(
                    FORMAL_UPDATES, iteration - SOURCE_ITERATION + 1
                )
            self.update(**values)
            if monitor_wandb and time.monotonic() >= next_remote_poll:
                try:
                    self._poll_wandb(child_alive=True)
                except Exception as error:
                    self.update(
                        wandb_sync_status="online_remote_poll_retryable",
                        wandb_last_poll_error=f"{type(error).__name__}: {error}",
                    )
                next_remote_poll = time.monotonic() + 20.0
            time.sleep(5)
        rc = int(process.returncode or 0)
        self.active = None
        self.update(active_pid=None, active_child_kind=None)
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
        return [
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
            str(SEED),
            "--headless",
            "--logger",
            logger,
            "--log_project_name",
            WANDB_PROJECT,
            "--resume",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--highstep_resume_mode",
            "refine",
            "--highstep_checkpoint_load_mode",
            "full",
            "--highstep_schedule_resume_mode",
            "preserve",
            "--run_name",
            run_name,
        ]

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
        wandb_config: Path | None = None,
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
                "wandb_config": str(wandb_config) if wandb_config else None,
                "created_at": now(),
            },
            read_only=True,
        )
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1"})
        monitor_wandb = logger == "wandb"
        if monitor_wandb:
            if not wandb_run_id or not wandb_config:
                raise RuntimeError("online W&B launch lacks run id or immutable config")
            env.update(
                {
                    "WANDB_MODE": "online",
                    "WANDB_RUN_ID": wandb_run_id,
                    "WANDB_RESUME": "allow",
                    "WANDB_USERNAME": WANDB_ENTITY,
                    "WANDB_RUN_GROUP": WORKFLOW,
                    "ROBOT_LAB_WANDB_CONFIG_PATH": str(wandb_config),
                    "ROBOT_LAB_WANDB_CONFIG_SHA256": sha256_file(wandb_config),
                    "WANDB_INIT_TIMEOUT": "120",
                    "WANDB__SERVICE_WAIT": "300",
                }
            )
        with log.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n[V1122_SUPERVISOR_LAUNCH] "
                + json.dumps({"time": now(), "command": command})
                + "\n"
            )
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
                active_child_kind="train",
                latest_log=str(log),
                checkpoint=str(checkpoint),
                wandb_run_id=wandb_run_id,
            )
            rc = self._wait(
                self.active,
                log,
                phase,
                monitor_wandb=monitor_wandb,
            )
        run_dirs = sorted(
            EXPERIMENT.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime
        )
        return rc, run_dirs[-1] if run_dirs else None

    def smoke(self) -> Path:
        marker = WORK / "smoke/smoke_result.json"
        if marker.exists():
            payload = read_json(marker)
            checkpoint = Path(payload.get("output_checkpoint", ""))
            if not (
                payload.get("passed") is True
                and checkpoint.is_file()
                and sha256_file(checkpoint) == payload.get("output_checkpoint_sha256")
            ):
                raise RuntimeError("stored v1.12.2 smoke evidence changed")
            self.update(phase="smoke_reused", smoke_checkpoint=str(checkpoint))
            return checkpoint

        source_before = sha256_file(SOURCE)
        source_scope = checkpoint_scope(SOURCE)
        run_name = f"v1122_flr_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
        rc, run_dir = self._launch_training(
            checkpoint=REBOUND_SOURCE,
            iterations=SMOKE_UPDATES,
            num_envs=64,
            run_name=run_name,
            phase="smoke_3_updates_full_resume",
            logger="tensorboard",
        )
        if rc != 0 or run_dir is None:
            raise RuntimeError(f"v1.12.2 full-resume smoke failed: rc={rc}, run={run_dir}")
        checkpoint = latest_checkpoint([run_dir])
        if checkpoint is None:
            raise RuntimeError("v1.12.2 smoke produced no checkpoint")
        output_scope = checkpoint_scope(checkpoint)
        expected_iteration = SOURCE_ITERATION + SMOKE_UPDATES - 1
        manifest = read_json(run_dir / "params/highstep_schedule_manifest.json")
        runtime = read_json(run_dir / "params/highstep_runtime_state.json")
        if not (
            output_scope["iteration"] == expected_iteration
            and output_scope["model_tensor_signature_sha256"]
            == source_scope["model_tensor_signature_sha256"]
            and output_scope["optimizer_state_entries"]
            == source_scope["optimizer_state_entries"]
            and output_scope["optimizer_step_min"] > source_scope["optimizer_step_min"]
            and output_scope["algorithm_class"] == source_scope["algorithm_class"]
            and output_scope["distill_stage"] == source_scope["distill_stage"]
            and manifest.get("checkpoint_load_mode") == "full"
            and manifest.get("schedule_resume_mode_resolved") == "preserve"
            and manifest.get("schedule_source", {}).get("method") == "runtime_snapshot_exact"
            and manifest.get("command_curriculum", {}).get("state_restored") is True
            and manifest.get("moving_best_state", {}).get("action_score_best_restored") is True
            and runtime.get("schema_version") == 2
            and sha256_file(SOURCE) == source_before
        ):
            raise RuntimeError("v1.12.2 smoke full-resume/tensor/schedule binding failed")
        atomic_json(
            marker,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "passed": True,
                "updates": SMOKE_UPDATES,
                "num_envs": 64,
                "source_checkpoint_sha256_before": source_before,
                "source_checkpoint_sha256_after": sha256_file(SOURCE),
                "rebound_source_checkpoint": str(REBOUND_SOURCE),
                "rebound_source_checkpoint_sha256": sha256_file(REBOUND_SOURCE),
                "rebound_schedule_manifest_sha256": sha256_file(REBOUND_SCHEDULE),
                "rebound_runtime_state_sha256": sha256_file(REBOUND_RUNTIME),
                "source_scope": source_scope,
                "output_checkpoint": str(checkpoint),
                "output_checkpoint_sha256": sha256_file(checkpoint),
                "output_scope": output_scope,
                "schedule_manifest": str(run_dir / "params/highstep_schedule_manifest.json"),
                "runtime_state": str(run_dir / "params/highstep_runtime_state.json"),
                "completed_at": now(),
            },
            read_only=True,
        )
        self.update(
            phase="smoke_passed",
            smoke_checkpoint=str(checkpoint),
            smoke_checkpoint_sha256=sha256_file(checkpoint),
        )
        return checkpoint

    @staticmethod
    def _play_command(checkpoint: Path, seed: int, *, video: bool = False) -> list[str]:
        command = [
            str(PYTHON),
            "-u",
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
        ]
        if video:
            command.extend(
                [
                    "--video",
                    "--video_length",
                    "600",
                    "--highstep_gap_camera",
                    "side_top",
                ]
            )
        return command

    def _run_eval_child(self, command: list[str], log: Path, phase: str) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
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
            self.update(
                status="running",
                phase=phase,
                active_pid=self.active.pid,
                active_child_kind="play",
                latest_log=str(log),
            )
            return self._wait(self.active, log, phase)

    def evaluate(self, checkpoint: Path, label: str) -> Path:
        summary_path = WORK / "evaluations" / label / "summary.json"
        if summary_path.exists():
            payload = read_json(summary_path)
            if not (
                payload.get("valid_count") == len(EVAL_SEEDS)
                and payload.get("checkpoint_sha256") == sha256_file(checkpoint)
            ):
                raise RuntimeError(f"stored {label} evaluation evidence changed")
            return summary_path

        rows: list[dict[str, Any]] = []
        checkpoint_sha = sha256_file(checkpoint)
        for index, seed in enumerate(EVAL_SEEDS, start=1):
            out = WORK / "evaluations" / label / f"rollout_{index:02d}_seed{seed}"
            stored = out / "result.json"
            log = out / "play.log"
            if stored.exists():
                row = read_json(stored)
                if row.get("checkpoint_sha256") != checkpoint_sha:
                    raise RuntimeError(f"stored {label} rollout checkpoint changed: {stored}")
                rows.append(row)
                continue
            command = self._play_command(checkpoint, seed)
            rc = self._run_eval_child(
                command, log, f"{label}_rollout_{index:02d}_seed{seed}"
            )
            if rc != 0:
                raise RuntimeError(f"{label} rollout {index}/15 failed: rc={rc}")
            canonical = tagged_eval(log)
            required_fl_value = canonical.get(
                "fl_vertical_riser_or_top_lip_underside_contact"
            )
            valid = bool(
                canonical.get("schema_version", 0) >= 8
                and canonical.get("checkpoint_sha256") == checkpoint_sha
                and canonical.get("schedule_valid") is True
                and canonical.get("reset_valid") is True
                and canonical.get("action_prior_enabled") is True
                and canonical.get("eval_action_delay_steps_runtime") == 0
                and isinstance(required_fl_value, bool)
            )
            if not valid:
                raise RuntimeError(f"invalid fail-closed {label} rollout {index}/15")
            row = {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "label": label,
                "rollout_index": index,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "valid": True,
                "fl_forbidden_contact": required_fl_value is True,
                "fl_forbidden_contact_first_step": canonical.get(
                    "fl_vertical_riser_or_top_lip_underside_first_step"
                ),
                "full_climb": canonical.get("full_climb_success") is True,
                "rear_hold": canonical.get("rear_on_platform_hold_success") is True,
                "terminated_early": canonical.get("terminated_early") is True,
                "front_wall_contact_rate_pre_support": canonical.get(
                    "front_wall_contact_rate_pre_support"
                ),
                "critical_rear_width_q05": canonical.get("critical_rear_width_q05"),
                "critical_rear_min_abs_y_q05": canonical.get(
                    "critical_rear_min_abs_y_q05"
                ),
                "canonical": canonical,
                "log": str(log),
                "log_sha256": sha256_file(log),
                "completed_at": now(),
            }
            atomic_json(stored, row, read_only=True)
            rows.append(row)
        summary = {
            "schema_version": 1,
            "workflow_id": WORKFLOW,
            "label": label,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "seeds": list(EVAL_SEEDS),
            "valid_count": sum(row["valid"] for row in rows),
            "fl_riser_contact_count": sum(row["fl_forbidden_contact"] for row in rows),
            "full_climb_count": sum(row["full_climb"] for row in rows),
            "rear_hold_count": sum(row["rear_hold"] for row in rows),
            "rollouts": [
                {
                    key: row[key]
                    for key in (
                        "rollout_index",
                        "seed",
                        "valid",
                        "fl_forbidden_contact",
                        "full_climb",
                        "rear_hold",
                        "log",
                        "log_sha256",
                    )
                }
                for row in rows
            ],
            "completed_at": now(),
        }
        if summary["valid_count"] != len(EVAL_SEEDS):
            raise RuntimeError(f"{label} did not produce 15/15 valid rollouts")
        atomic_json(summary_path, summary, read_only=True)
        self.update(
            phase=f"{label}_complete",
            **{f"{label}_summary": str(summary_path)},
        )
        return summary_path

    def _write_wandb_config(self) -> Path:
        path = WORK / "wandb/flr_E100_config_rebinding4.yaml"
        values = self._required_wandb_config()
        payload: dict[str, Any] = {"wandb_version": 1}
        payload.update({key: {"value": value} for key, value in values.items()})
        import yaml

        if path.exists():
            existing = yaml.safe_load(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError("immutable W&B config changed")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        path.chmod(0o444)
        return path

    def _formal_run_dirs(self) -> list[Path]:
        return sorted(
            EXPERIMENT.glob("*_v1122_flr_E100_attempt*"),
            key=lambda path: path.stat().st_mtime,
        )

    def _final_remote_training_gate(self, checkpoint: Path) -> Path:
        marker = WORK / "wandb/remote_training_gate.json"
        if marker.exists():
            payload = read_json(marker)
            if not (
                payload.get("passed") is True
                and payload.get("output_checkpoint_sha256") == sha256_file(checkpoint)
                and payload.get("remote_latest_step", -1) >= TARGET_FINAL_ITERATION
            ):
                raise RuntimeError("stored W&B remote training gate changed")
            return marker

        deadline = time.monotonic() + 300.0
        last_error: Exception | None = None
        remote: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                remote = self._poll_wandb(child_alive=False)
                if (
                    remote.get("state") == "finished"
                    and int(remote.get("remote_latest_step") or -1)
                    >= TARGET_FINAL_ITERATION
                ):
                    break
            except Exception as error:
                last_error = error
                self.update(
                    wandb_sync_status="remote_finalization_retryable",
                    wandb_last_poll_error=f"{type(error).__name__}: {error}",
                )
            time.sleep(10)
        if remote is None or not (
            remote.get("state") == "finished"
            and int(remote.get("remote_latest_step") or -1) >= TARGET_FINAL_ITERATION
        ):
            raise RuntimeError(f"W&B remote finalization incomplete: {last_error or remote}")
        samples = list(self.state.get("wandb_live_step_samples") or [])
        live_steps = [int(sample["remote_latest_step"]) for sample in samples]
        if len(live_steps) < 2 or any(
            later <= earlier for earlier, later in zip(live_steps, live_steps[1:])
        ):
            raise RuntimeError(
                f"W&B real-time step growth was not verified during the active child: {live_steps}"
            )
        stage_manifest = WORK / "training/flr_E100_training_manifest.json"
        output_scope = checkpoint_scope(checkpoint)
        atomic_json(
            stage_manifest,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "stage_id": "FLR-E100",
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "source_checkpoint": str(SOURCE),
                "source_checkpoint_sha256": SOURCE_SHA,
                "output_checkpoint": str(checkpoint),
                "output_checkpoint_sha256": sha256_file(checkpoint),
                "output_scope": output_scope,
                "effective_updates": FORMAL_UPDATES,
                "wandb": remote,
                "wandb_live_step_samples": samples,
                "formal_run_dirs": [str(path) for path in self._formal_run_dirs()],
                "remote_gate_passed_before_candidate_evaluation": True,
                "completed_at": now(),
            },
            read_only=True,
        )

        import wandb

        api = wandb.Api(timeout=30)
        run = api.run(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}/{self.state['wandb_run_id']}"
        )
        run.summary.update(
            {
                "output_checkpoint_sha256": sha256_file(checkpoint),
                "effective_updates": FORMAL_UPDATES,
                "training_remote_gate": "passed_pending_candidate_evaluation",
                "manifest_path": str(stage_manifest),
            }
        )
        run.update()
        payload = {
            "schema_version": 1,
            "workflow_id": WORKFLOW,
            "passed": True,
            "run_id": remote["run_id"],
            "url": remote["url"],
            "group": remote["group"],
            "remote_state": remote["state"],
            "remote_latest_step": remote["remote_latest_step"],
            "live_step_samples": samples,
            "output_checkpoint": str(checkpoint),
            "output_checkpoint_sha256": sha256_file(checkpoint),
            "stage_manifest": str(stage_manifest),
            "stage_manifest_sha256": sha256_file(stage_manifest),
            "verified_at": now(),
        }
        atomic_json(marker, payload, read_only=True)
        self.update(
            status="post_training",
            phase="wandb_remote_training_gate_passed",
            wandb_sync_status="remote_training_gate_passed",
            wandb_remote_latest_step=remote["remote_latest_step"],
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            effective_updates=FORMAL_UPDATES,
        )
        return marker

    def formal_train(self) -> tuple[Path, Path]:
        marker = WORK / "training/flr_E100_result.json"
        if marker.exists():
            payload = read_json(marker)
            checkpoint = Path(payload.get("output_checkpoint", ""))
            if not (
                checkpoint.is_file()
                and sha256_file(checkpoint) == payload.get("output_checkpoint_sha256")
                and checkpoint_scope(checkpoint)["iteration"] == TARGET_FINAL_ITERATION
            ):
                raise RuntimeError("stored FLR-E100 training result changed")
            remote_gate = self._final_remote_training_gate(checkpoint)
            return checkpoint, remote_gate

        source_before = sha256_file(SOURCE)
        run_id = str(
            self.state.get("wandb_run_id")
            or f"v1122e100{int(time.time())}{os.getpid()}"
        )
        wandb_config = self._write_wandb_config()
        self.update(
            status="running",
            phase="formal_FLR_E100_preparing",
            failure_class="none",
            wandb_run_id=run_id,
            wandb_sync_status="online_launch_pending",
            wandb_live_step_samples=list(self.state.get("wandb_live_step_samples") or []),
            formal_training_started=True,
        )

        attempts = int(self.state.get("formal_training_attempts", 0))
        checkpoint = latest_checkpoint(
            self._formal_run_dirs(), at_most=TARGET_FINAL_ITERATION
        )
        if checkpoint is None:
            checkpoint = REBOUND_SOURCE
        while checkpoint_scope(checkpoint)["iteration"] < TARGET_FINAL_ITERATION:
            if attempts >= MAX_TRAIN_ATTEMPTS:
                raise RuntimeError("FLR-E100 exhausted finite infrastructure retry budget")
            scope = checkpoint_scope(checkpoint)
            remaining = TARGET_FINAL_ITERATION - int(scope["iteration"]) + 1
            attempts += 1
            run_name = (
                f"v1122_flr_E100_attempt{attempts}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            self.update(formal_training_attempts=attempts)
            rc, run_dir = self._launch_training(
                checkpoint=checkpoint,
                iterations=remaining,
                num_envs=FORMAL_NUM_ENVS,
                run_name=run_name,
                phase=f"formal_FLR_E100_attempt{attempts}",
                logger="wandb",
                wandb_run_id=run_id,
                wandb_config=wandb_config,
            )
            run_dirs = self._formal_run_dirs()
            candidate = latest_checkpoint(run_dirs, at_most=TARGET_FINAL_ITERATION)
            if candidate is not None:
                checkpoint = candidate
            if rc != 0:
                self.update(
                    failure_class="training_infrastructure_retryable",
                    last_error=f"formal train attempt {attempts} exited rc={rc}",
                    latest_run_dir=str(run_dir) if run_dir else None,
                )
                continue
            if run_dir is None:
                raise RuntimeError("formal FLR-E100 training produced no isolated run directory")

        output_scope = checkpoint_scope(checkpoint)
        if not (
            output_scope["iteration"] == TARGET_FINAL_ITERATION
            and output_scope["model_tensor_signature_sha256"]
            == checkpoint_scope(SOURCE)["model_tensor_signature_sha256"]
            and output_scope["optimizer_state_entries"]
            == checkpoint_scope(SOURCE)["optimizer_state_entries"]
            and output_scope["optimizer_step_min"]
            > checkpoint_scope(SOURCE)["optimizer_step_min"]
            and sha256_file(SOURCE) == source_before
        ):
            raise RuntimeError("formal FLR-E100 checkpoint/full-resume integrity failed")
        atomic_json(
            marker,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "stage_id": "FLR-E100",
                "source_checkpoint": str(SOURCE),
                "source_checkpoint_sha256_before": source_before,
                "source_checkpoint_sha256_after": sha256_file(SOURCE),
                "additional_updates": FORMAL_UPDATES,
                "target_final_iteration": TARGET_FINAL_ITERATION,
                "output_checkpoint": str(checkpoint),
                "output_checkpoint_sha256": sha256_file(checkpoint),
                "output_scope": output_scope,
                "wandb_run_id": run_id,
                "formal_run_dirs": [str(path) for path in self._formal_run_dirs()],
                "completed_at": now(),
            },
            read_only=True,
        )
        remote_gate = self._final_remote_training_gate(checkpoint)
        return checkpoint, remote_gate

    def _finalize_wandb_summary(
        self,
        checkpoint: Path,
        evaluation_summary: Path,
        decision: str,
    ) -> Path:
        marker = WORK / "wandb/final_remote_summary_verification.json"
        candidate = read_json(evaluation_summary)
        required = {
            "output_checkpoint_sha256": sha256_file(checkpoint),
            "effective_updates": FORMAL_UPDATES,
            "fl_riser_contact_count": candidate["fl_riser_contact_count"],
            "valid_count": candidate["valid_count"],
            "full_climb_count": candidate["full_climb_count"],
            "rear_hold_count": candidate["rear_hold_count"],
            "gate_decision": decision,
            "manifest_path": str(evaluation_summary),
        }
        if marker.exists():
            payload = read_json(marker)
            if payload.get("required_summary") != required:
                raise RuntimeError("stored W&B final summary verification changed")
            return marker

        import wandb

        run_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{self.state['wandb_run_id']}"
        api = wandb.Api(timeout=30)
        run = api.run(run_path)
        run.summary.update(required)
        run.update()
        deadline = time.monotonic() + 120.0
        actual: dict[str, Any] = {}
        while time.monotonic() < deadline:
            refreshed = wandb.Api(timeout=30).run(run_path)
            actual = dict(refreshed.summary)
            if all(actual.get(key) == value for key, value in required.items()):
                break
            time.sleep(5)
        if not all(actual.get(key) == value for key, value in required.items()):
            raise RuntimeError(f"W&B final summary readback mismatch: {actual}")
        atomic_json(
            marker,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "run_id": self.state["wandb_run_id"],
                "run_url": self.state.get("wandb_run_url"),
                "required_summary": required,
                "remote_readback_verified": True,
                "verified_at": now(),
            },
            read_only=True,
        )
        self.update(
            wandb_sync_status="remote_complete_with_evaluation_summary",
            wandb_final_summary_manifest=str(marker),
        )
        return marker

    def visual_evidence(self, checkpoint: Path, candidate_summary: Path) -> Path:
        manifest = WORK / "videos/candidate_visual_manifest.json"
        if manifest.exists():
            return manifest
        training_run_dirs = self._formal_run_dirs()
        if not training_run_dirs:
            raise RuntimeError("candidate video has no formal training run directory")
        video_source_dir = training_run_dirs[-1] / "videos/play"
        records = []
        expected = read_json(candidate_summary)
        for index, seed in enumerate(EVAL_SEEDS, start=1):
            out = WORK / "videos" / f"rollout_{index:02d}_seed{seed}"
            log = out / "play.log"
            target = out / "candidate.mp4"
            if target.exists():
                records.append(
                    {
                        "rollout_index": index,
                        "seed": seed,
                        "video": str(target),
                        "video_sha256": sha256_file(target),
                        "video_bytes": target.stat().st_size,
                    }
                )
                continue
            before = {
                path: path.stat().st_mtime_ns
                for path in video_source_dir.glob("*.mp4")
            } if video_source_dir.exists() else {}
            rc = self._run_eval_child(
                self._play_command(checkpoint, seed, video=True),
                log,
                f"candidate_video_{index:02d}_seed{seed}",
            )
            if rc != 0:
                raise RuntimeError(f"candidate video rollout {index}/15 failed: rc={rc}")
            replay = tagged_eval(log)
            original = expected["rollouts"][index - 1]
            if not (
                replay.get("fl_vertical_riser_or_top_lip_underside_contact")
                is original["fl_forbidden_contact"]
                and (replay.get("full_climb_success") is True) == original["full_climb"]
                and (replay.get("rear_on_platform_hold_success") is True)
                == original["rear_hold"]
            ):
                raise RuntimeError(f"candidate video replay {index}/15 was not deterministic")
            videos = sorted(video_source_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime_ns)
            changed = [
                path for path in videos
                if path not in before or path.stat().st_mtime_ns != before[path]
            ]
            if not changed:
                raise RuntimeError(f"candidate video rollout {index}/15 produced no MP4")
            source_video = changed[-1]
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, target)
            if target.stat().st_size < 100_000:
                raise RuntimeError(f"candidate video rollout {index}/15 is too small")
            records.append(
                {
                    "rollout_index": index,
                    "seed": seed,
                    "video": str(target),
                    "video_sha256": sha256_file(target),
                    "video_bytes": target.stat().st_size,
                    "log": str(log),
                    "log_sha256": sha256_file(log),
                }
            )
        atomic_json(
            manifest,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "camera": "side_top",
                "rollouts": records,
                "complete": len(records) == len(EVAL_SEEDS),
                "created_at": now(),
            },
            read_only=True,
        )
        return manifest

    def run(self) -> None:
        self.update(
            status="running",
            phase="authority_verified",
            failure_class="none",
            requires_user_action=False,
            training_started=False,
            student_training_started=False,
        )
        self.smoke()
        baseline_path = self.evaluate(REBOUND_SOURCE, "baseline_E1000")
        baseline = read_json(baseline_path)
        if baseline["valid_count"] != len(EVAL_SEEDS):
            raise RuntimeError("E1000 baseline was not 15/15 valid")
        checkpoint, remote_gate = self.formal_train()
        candidate_path = self.evaluate(checkpoint, "candidate_FLR_E100")
        candidate = read_json(candidate_path)
        passed = bool(
            candidate["valid_count"] == len(EVAL_SEEDS)
            and candidate["fl_riser_contact_count"] == 0
            and candidate["full_climb_count"] >= baseline["full_climb_count"]
            and candidate["rear_hold_count"] >= baseline["rear_hold_count"]
        )
        decision = (
            "teacher_fl_retraction_candidate_pending_user_visual_review"
            if passed
            else "teacher_fl_retraction_first_round_gate_failed"
        )
        final_wandb = self._finalize_wandb_summary(
            checkpoint, candidate_path, decision
        )
        video_manifest: Path | None = None
        if passed:
            video_manifest = self.visual_evidence(checkpoint, candidate_path)
        handoff = {
            "schema_version": 2,
            "workflow_id": WORKFLOW,
            "authority_version": "v1.12.2",
            "status": decision,
            "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG),
            "preregistration_sha256": PREREG_SHA,
            "source_checkpoint": str(SOURCE),
            "source_checkpoint_sha256": SOURCE_SHA,
            "output_checkpoint": str(checkpoint),
            "output_checkpoint_sha256": sha256_file(checkpoint),
            "effective_updates": FORMAL_UPDATES,
            "baseline_summary": str(baseline_path),
            "baseline_summary_sha256": sha256_file(baseline_path),
            "candidate_summary": str(candidate_path),
            "candidate_summary_sha256": sha256_file(candidate_path),
            "wandb_remote_training_gate": str(remote_gate),
            "wandb_remote_training_gate_sha256": sha256_file(remote_gate),
            "wandb_final_summary": str(final_wandb),
            "wandb_final_summary_sha256": sha256_file(final_wandb),
            "video_manifest": str(video_manifest) if video_manifest else None,
            "video_manifest_sha256": (
                sha256_file(video_manifest) if video_manifest else None
            ),
            "gate": {
                "valid": f"{candidate['valid_count']}/{len(EVAL_SEEDS)}",
                "fl_contacts": f"{candidate['fl_riser_contact_count']}/{len(EVAL_SEEDS)}",
                "full_climb_candidate": candidate["full_climb_count"],
                "full_climb_baseline": baseline["full_climb_count"],
                "rear_hold_candidate": candidate["rear_hold_count"],
                "rear_hold_baseline": baseline["rear_hold_count"],
            },
            "student_training_started": False,
            "automatic_real_robot_deployment": False,
            "next_action": (
                "User reviews the 15-rollout visual evidence; do not start Student or deploy."
                if passed
                else "Stop this single finite stage; do not add updates or change variables without new user authority."
            ),
            "written_at": now(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update(
            status=decision,
            phase=decision,
            active_pid=None,
            active_child_kind=None,
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            effective_updates=FORMAL_UPDATES,
            training_started=True,
            formal_training_started=True,
            evaluation_summary=str(candidate_path),
            evaluation_summary_sha256=sha256_file(candidate_path),
            requires_user_action=True,
            failure_class=("none" if passed else "behavior_gate_failed"),
            last_error=None,
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
    except Exception as error:
        if supervisor is not None:
            safe_checkpoint = latest_checkpoint(
                supervisor._formal_run_dirs(), at_most=TARGET_FINAL_ITERATION
            )
            supervisor.update(
                status="infrastructure_failed_at_safe_checkpoint",
                phase="infrastructure_failed_at_safe_checkpoint",
                failure_class="infrastructure",
                active_pid=None,
                active_child_kind=None,
                checkpoint=str(safe_checkpoint or SOURCE),
                checkpoint_sha256=sha256_file(safe_checkpoint or SOURCE),
                last_error=f"{type(error).__name__}: {error}",
                requires_user_action=True,
            )
            atomic_json(
                supervisor.handoff_path,
                {
                    "schema_version": 2,
                    "workflow_id": WORKFLOW,
                    "status": "infrastructure_failed_at_safe_checkpoint",
                    "error": f"{type(error).__name__}: {error}",
                    "safe_checkpoint": str(safe_checkpoint or SOURCE),
                    "safe_checkpoint_sha256": sha256_file(safe_checkpoint or SOURCE),
                    "behavior_gate_changed": False,
                    "next_action": "Repair only the reported infrastructure fault and resume this same finite stage.",
                    "written_at": now(),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
