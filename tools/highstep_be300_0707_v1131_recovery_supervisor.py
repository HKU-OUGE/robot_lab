#!/usr/bin/env python3
"""Persistent fail-closed supervisor for the v1.13.1 model_1000 recovery."""

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


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORK = ROOT / "tmp/highstep_be300_0707_distill_20260716"
WORKFLOW = "highstep_be300_0707_distill_20260716"
VERSION = "v1.13.1"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "0307b9e5c5c9c81999499ce2c3f05c0beabaa98caedde24aac5ac7843d73d768"
PREREG = WORK / "preregistration_v1131.json"
PREREG_SHA = "2660ad2ba78719018c0d1ba1d926ee6fc6e997aebd8bfabdefcd1e18196b1cc6"
AMENDMENT = Path(os.environ.get("HIGHSTEP_V1131_RECOVERY_AMENDMENT", ""))
AMENDMENT_SHA = os.environ.get("HIGHSTEP_V1131_RECOVERY_AMENDMENT_SHA256", "")
BINDING = Path(os.environ.get("HIGHSTEP_V1131_RECOVERY_BINDING", ""))
BINDING_SHA = os.environ.get("HIGHSTEP_V1131_RECOVERY_BINDING_SHA256", "")
CONTINUITY_PATCH = Path(os.environ.get("HIGHSTEP_V1131_CONTINUITY_PATCH", ""))
CONTINUITY_PATCH_SHA = os.environ.get("HIGHSTEP_V1131_CONTINUITY_PATCH_SHA256", "")
WANDB_CONFIG = WORK / "wandb_config_v1131_recovery_r3.json"
WANDB_CONFIG_SHA = "69169760f9afaebcca3ff35b16f3eb26f86d52ed56ec7e83793dcbd2f7b15207"
HISTORY_EXPORT = WORK / "wandb_source_scalar_history_steps_0_999_v1131_r3.jsonl"
HISTORY_EXPORT_SHA = "3161acef1c859aade699abcd8d7b2ea48624600bf5a822e86cd6b4f0852bb9a7"
HISTORY_MANIFEST = WORK / "wandb_source_scalar_history_steps_0_999_v1131_r3_manifest.json"
HISTORY_MANIFEST_SHA = "fbc1b7499c5eebfc04b9650ce2f119dfdfaa88faef20f550c94392e4ac8d1d05"
MIRROR_GATE = WORK / "wandb_recovery_r2_scalar_mirror_remote_gate.json"
MIRROR_GATE_SHA = "6a535a8b208da0e213e4ef792bf9d9d8e9ce1401b4bb4bb08a05d8626073d966"
CHECKPOINT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_"
    "student_no_prior_Student/2026-07-16_08-35-41_highstep_be300_0707_"
    "distill_long_E4000_20260716/model_1000.pt"
)
CHECKPOINT_SHA = "755c1b16817503b0155e672c68cd44eed66a46d1a5ba73048bc25fca3128f773"
SOURCE_RUN_DIR = CHECKPOINT.parent
RUNTIME_SIDECAR = SOURCE_RUN_DIR / "params/highstep_runtime_state.json"
RUNTIME_SIDECAR_SHA = "e18d3c65430f9a84b972ff3d9bf656fc3e0d4e8b46895744d570d04d3c0e30fc"
SCHEDULE_SIDECAR = SOURCE_RUN_DIR / "params/highstep_schedule_manifest.json"
SCHEDULE_SIDECAR_SHA = "f2beda7d4006f2b3da9a22b9ca9e9d34df074b8f9d6ce111cc75ea1d1a0c7fcd"
PARENT = WORK / "teacher_parent_lineage_v1131.json"
PARENT_SHA = "82dd4934388e5433718394f25d1dcc72c09ae18cddc2a3355a3a75556f4a8f93"
INFRA_DRAFT = WORK / "infrastructure_recovery_model1000_v1131.json"
INFRA_DRAFT_SHA = "9978a530e3eb504106fa33e390f053c1befc5a3cf06e449c062634c4b1d40f3b"
TASK = "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123StudentNoPrior-ArcdogAdjustableLeg-v0"
EXPERIMENT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_"
    "student_no_prior_Student"
)
RUN_NAME = "highstep_be300_0707_distill_long_E4000_recovery_r3_model1000_20260716"
WANDB_ENTITY = "xinqili551-the-university-of-hong-kong"
WANDB_PROJECT = "isaaclab"
WANDB_SOURCE_RUN = "be3000707v1131"
WANDB_RUN = "be3000707v1131r2"
WANDB_GROUP = WORKFLOW
START_COUNT = 1001
TOTAL_COUNT = 4000
REMAINING_UPDATES = 2999
WARMUP = 1400
SAVE_INTERVAL = 100
SERVICE = "highstep-be300-0707-distill-v1131-recovery.service"
STATE = WORK / "state.json"
HEARTBEAT = WORK / "heartbeat.json"
HANDOFF = WORK / "handoff.json"
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
LOCK = WORK / "supervisor.lock"
SUPERVISOR_PID = WORK / "supervisor.pid"
TRAIN_PID = WORK / "train.pid"
CHILD_LOG = WORK / "logs/v1131_recovery_r3_model1000_systemd.log"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
COUNT_RE = re.compile(r"Student_Distill_Update_Count loss:\s*([0-9.]+)")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


class Supervisor:
    def __init__(self) -> None:
        WORK.mkdir(parents=True, exist_ok=True)
        CHILD_LOG.parent.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+", encoding="utf-8")
        fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock_stream.seek(0)
        self.lock_stream.truncate()
        self.lock_stream.write(f"{os.getpid()}\n")
        self.lock_stream.flush()
        os.fsync(self.lock_stream.fileno())
        atomic_text(SUPERVISOR_PID, f"{os.getpid()}\n")
        self.child: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.run_dir: Path | None = None
        self.current_iteration = 1000
        self.effective_count = START_COUNT
        self.remote_latest_step: int | None = None
        self.remote_samples: list[dict[str, Any]] = []
        self.remote_seen = False
        self.wandb_initialized = False
        self.last_remote_ok = time.monotonic()
        self._validate_authority()
        self.state = read_json(STATE)
        signal.signal(signal.SIGTERM, self._handle_stop)
        signal.signal(signal.SIGINT, self._handle_stop)

    def _handle_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()

    def _validate_authority(self) -> None:
        immutable = (
            (SPEC, SPEC_SHA),
            (PREREG, PREREG_SHA),
            (AMENDMENT, AMENDMENT_SHA),
            (BINDING, BINDING_SHA),
            (CONTINUITY_PATCH, CONTINUITY_PATCH_SHA),
            (WANDB_CONFIG, WANDB_CONFIG_SHA),
            (HISTORY_EXPORT, HISTORY_EXPORT_SHA),
            (HISTORY_MANIFEST, HISTORY_MANIFEST_SHA),
            (MIRROR_GATE, MIRROR_GATE_SHA),
            (CHECKPOINT, CHECKPOINT_SHA),
            (RUNTIME_SIDECAR, RUNTIME_SIDECAR_SHA),
            (SCHEDULE_SIDECAR, SCHEDULE_SIDECAR_SHA),
            (PARENT, PARENT_SHA),
            (INFRA_DRAFT, INFRA_DRAFT_SHA),
        )
        for path, expected in immutable:
            if not str(path) or not expected or not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"authority artifact mismatch: {path}")
        for path, _ in immutable[1:]:
            if path.stat().st_mode & 0o222:
                raise RuntimeError(f"immutable recovery artifact is writable: {path}")
        prereg = read_json(PREREG)
        continuity_patch = read_json(CONTINUITY_PATCH)
        if not (
            prereg.get("workflow_id") == WORKFLOW
            and prereg.get("authority", {}).get("version") == VERSION
            and prereg.get("authority", {}).get("spec_sha256") == SPEC_SHA
            and prereg.get("training_budget", {}).get("effective_updates") == TOTAL_COUNT
            and prereg.get("distillation_contract", {}).get("warmup_effective_updates") == WARMUP
            and prereg.get("training_budget", {}).get("save_interval") == SAVE_INTERVAL
        ):
            raise RuntimeError("v1.13.1 preregistration content mismatch")
        for relative, expected in prereg.get("runtime_code", {}).items():
            if relative == "task_registry_init.py":
                path = ROOT / (
                    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
                    "Arcdog_adjustable_leg/__init__.py"
                )
            elif relative == "highstep_schedule.py":
                path = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"
            elif relative == "train.py":
                path = ROOT / "scripts/rsl_rl/base/train.py"
            elif relative == "highstep_env_cfg.py":
                path = ROOT / (
                    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
                    "Arcdog_adjustable_leg/highstep_env_cfg.py"
                )
            else:
                path = ROOT / (
                    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
                    f"Arcdog_adjustable_leg/agents/{relative}"
                )
            actual = sha256_file(path) if path.is_file() else None
            schedule_recovery_patch = (
                relative == "highstep_schedule.py"
                and expected == continuity_patch.get("single_change", {}).get("old_sha256")
                and actual == continuity_patch.get("single_change", {}).get("new_sha256")
                and continuity_patch.get("training_semantics_changed") is False
            )
            if actual != expected and not schedule_recovery_patch:
                raise RuntimeError(f"bound v1.13.1 runtime code changed: {path}")
        amendment = read_json(AMENDMENT)
        binding = read_json(BINDING)
        dashboard = read_json(DASHBOARD)
        state = read_json(STATE)
        if not (
            amendment.get("training_semantics_changed") is False
            and amendment.get("recovery", {}).get("remaining_effective_updates") == REMAINING_UPDATES
            and amendment.get("recovery", {}).get("student_distill_update_count") == START_COUNT
            and binding.get("supervisor_sha256") == sha256_file(Path(__file__))
            and binding.get("amendment_sha256") == AMENDMENT_SHA
            and binding.get("continuity_patch_sha256") == CONTINUITY_PATCH_SHA
            and binding.get("wandb_config_sha256") == WANDB_CONFIG_SHA
            and binding.get("history_export_sha256") == HISTORY_EXPORT_SHA
            and binding.get("mirror_gate_sha256") == MIRROR_GATE_SHA
            and dashboard.get("workflow_id") == WORKFLOW
            and dashboard.get("spec_sha256") == SPEC_SHA
            and dashboard.get("preregistration_sha256") == PREREG_SHA
            and dashboard.get("infrastructure_recovery_binding_sha256") == BINDING_SHA
            and state.get("workflow_id") == WORKFLOW
            and state.get("authority_version") == VERSION
            and state.get("spec_sha256") == SPEC_SHA
            and state.get("preregistration_sha256") == PREREG_SHA
            and state.get("active_pid") is None
        ):
            raise RuntimeError("infrastructure recovery rebinding mismatch")

    def _command(self) -> list[str]:
        return [
            "/home/lxq/miniconda3/envs/env_isaaclab/bin/python", "-u",
            "scripts/rsl_rl/base/train.py",
            "--task", TASK,
            "--num_envs", "4096",
            "--max_iterations", str(REMAINING_UPDATES),
            "--seed", "42",
            "--headless",
            "--logger", "wandb",
            "--log_project_name", WANDB_PROJECT,
            "--resume",
            "--checkpoint", str(CHECKPOINT),
            "--highstep_resume_mode", "refine",
            "--highstep_checkpoint_load_mode", "full",
            "--highstep_schedule_resume_mode", "preserve",
            "--highstep_parent_teacher_manifest", str(PARENT),
            "--run_name", RUN_NAME,
        ]

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH": str(PREREG),
                "HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256": PREREG_SHA,
                "ROBOT_LAB_WANDB_CONFIG_PATH": str(WANDB_CONFIG),
                "ROBOT_LAB_WANDB_CONFIG_SHA256": WANDB_CONFIG_SHA,
                "WANDB_MODE": "online",
                "WANDB_RUN_ID": WANDB_RUN,
                "WANDB_RESUME": "must",
                "WANDB_USERNAME": WANDB_ENTITY,
                "WANDB_ENTITY": WANDB_ENTITY,
                "WANDB_RUN_GROUP": WANDB_GROUP,
                "WANDB_INIT_TIMEOUT": "120",
                "WANDB__SERVICE_WAIT": "300",
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.pop("WANDB_FORK_FROM", None)
        env.pop("WANDB_RESUME_FROM", None)
        return env

    def _remote(self) -> dict[str, Any] | None:
        import wandb

        try:
            run = wandb.Api(timeout=15).run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN}")
        except Exception:
            return None
        if not (
            run.group == WANDB_GROUP
            and run.config.get("source_run_id") == WANDB_SOURCE_RUN
            and run.config.get("recovery_run_id") == WANDB_RUN
            and run.config.get("history_origin") == "verified_source_scalar_mirror"
            and run.config.get("lineage_relation")
            == "independent_recovery_run_due_to_wandb_private_preview_entitlement"
        ):
            raise RuntimeError("W&B recovery group/config binding mismatch")
        step = int(run.lastHistoryStep or 0)
        first_recovery_step = None
        if step >= 1000:
            recovery_rows = list(run.scan_history(min_step=1000, max_step=1001, page_size=10))
            if not recovery_rows or int(recovery_rows[0].get("_step", -1)) != 1000:
                raise RuntimeError("W&B recovery history does not begin at exact safe step 1000")
            first_recovery_step = 1000
        return {
            "state": run.state,
            "url": run.url,
            "step": step,
            "first_recovery_step": first_recovery_step,
        }

    def _assert_remote_precondition(self) -> None:
        import wandb

        api = wandb.Api(timeout=45)
        source = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_SOURCE_RUN}")
        recovery = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN}")
        gate = read_json(MIRROR_GATE)
        if not (
            source.state == "crashed"
            and int(source.lastHistoryStep or -1) == 1078
            and recovery.state == "finished"
            and int(recovery.lastHistoryStep or -1) == 999
            and recovery.group == WANDB_GROUP
            and recovery.config.get("history_origin") == "verified_source_scalar_mirror"
            and gate.get("passed") is True
            and gate.get("run_id") == WANDB_RUN
            and gate.get("remote_last_history_step") == 999
            and gate.get("history_export_sha256") == HISTORY_EXPORT_SHA
        ):
            raise RuntimeError("r3 source/mirrored W&B precondition mismatch")

    def _parse_log(self) -> None:
        if not CHILD_LOG.is_file():
            return
        text = ANSI_RE.sub("", CHILD_LOG.read_text(encoding="utf-8", errors="replace"))
        self.wandb_initialized = (
            f"runs/{WANDB_RUN}" in text
            or "wandb: Tracking run" in text
            or "wandb: View run" in text
        )
        iterations = [int(value[0]) for value in ITER_RE.findall(text)]
        counts = [int(float(value)) for value in COUNT_RE.findall(text)]
        if iterations:
            self.current_iteration = iterations[-1]
        if counts:
            # The metric is captured before VAEPPO increments its persisted
            # counter, while RSL-RL logs only after that update has completed.
            self.effective_count = counts[-1] + 1
        if self.run_dir is None:
            candidates = [path for path in EXPERIMENT.glob(f"*{RUN_NAME}*") if path.is_dir()]
            if candidates:
                self.run_dir = max(candidates, key=lambda path: path.stat().st_mtime)

    def _checkpoint_status(self) -> tuple[str, str | None]:
        if self.run_dir is None:
            return "run_dir_pending", None
        path = self.run_dir / "model_1100.pt"
        if path.is_file():
            return "generated_and_sha_verified", sha256_file(path)
        return "pending", None

    def _update(self, **values: Any) -> None:
        self.state.update(values)
        self.state.update(
            {
                "supervisor_pid": os.getpid(),
                "active_pid": self.child.pid if self.child and self.child.poll() is None else None,
                "service_name": SERVICE,
                "lock_path": str(LOCK),
                "supervisor_pid_path": str(SUPERVISOR_PID),
                "train_pid_path": str(TRAIN_PID),
                "recovery_checkpoint": str(CHECKPOINT),
                "recovery_checkpoint_sha256": CHECKPOINT_SHA,
                "checkpoint_load_mode": "full",
                "schedule_resume_mode": "preserve",
                "restored_student_distill_update_count": START_COUNT,
                "remaining_effective_updates": REMAINING_UPDATES,
                "warmup_reset": False,
                "fresh_environment_state": True,
                "process_exact_resume": False,
                "wandb_source_run_id": WANDB_SOURCE_RUN,
                "wandb_run_id": WANDB_RUN,
                "wandb_resume": "must",
                "wandb_lineage_relation": (
                    "independent_recovery_run_due_to_wandb_private_preview_entitlement"
                ),
                "wandb_history_mirrored_range": [0, 999],
                "wandb_run_url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{WANDB_RUN}",
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "current_iteration": self.current_iteration,
                "effective_updates": self.effective_count,
                "updated_at": now(),
            }
        )
        atomic_json(STATE, self.state)
        model_status, model_sha = self._checkpoint_status()
        atomic_json(
            HEARTBEAT,
            {
                "schema_version": 2,
                "workflow_id": WORKFLOW,
                "authority_version": VERSION,
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "status": self.state.get("status"),
                "phase": self.state.get("phase"),
                "supervisor_pid": os.getpid(),
                "active_pid": self.state.get("active_pid"),
                "lock_path": str(LOCK),
                "current_iteration": self.current_iteration,
                "effective_updates": self.effective_count,
                "warmup_reset": False,
                "warmup_first_eligible_count": WARMUP,
                "wandb_run_id": WANDB_RUN,
                "wandb_run_url": self.state.get("wandb_run_url"),
                "wandb_remote_latest_step": self.remote_latest_step,
                "wandb_sync_status": self.state.get("wandb_sync_status"),
                "model_1100_status": model_status,
                "model_1100_sha256": model_sha,
                "failure_class": self.state.get("failure_class", "none"),
                "written_at": now(),
                "written_epoch": time.time(),
            },
        )

    def _write_handoff(self, last_completed: str, next_action: str) -> None:
        atomic_json(
            HANDOFF,
            {
                "schema_version": 2,
                "workflow_id": WORKFLOW,
                "authority_version": VERSION,
                "status": self.state.get("status"),
                "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "infrastructure_recovery_amendment": str(AMENDMENT),
                "infrastructure_recovery_amendment_sha256": AMENDMENT_SHA,
                "infrastructure_recovery_binding": str(BINDING),
                "infrastructure_recovery_binding_sha256": BINDING_SHA,
                "stale_child_pid": 2898910,
                "stale_child_classification": "stale_session_cleanup",
                "supervisor_pid": os.getpid(),
                "active_pid": self.state.get("active_pid"),
                "recovery_checkpoint": str(CHECKPOINT),
                "recovery_checkpoint_sha256": CHECKPOINT_SHA,
                "restored_student_distill_update_count": START_COUNT,
                "remaining_effective_updates": REMAINING_UPDATES,
                "warmup_reset": False,
                "resume_claim": (
                    "full model, optimizer, runner iteration, cumulative warmup count and schedule "
                    "resume with fresh environment state"
                ),
                "wandb_source_run_id": WANDB_SOURCE_RUN,
                "wandb_run_id": WANDB_RUN,
                "wandb_resume": "must",
                "wandb_history_mirrored_range": [0, 999],
                "wandb_run_url": self.state.get("wandb_run_url"),
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "current_iteration": self.current_iteration,
                "effective_updates": self.effective_count,
                "last_completed": last_completed,
                "next_action": next_action,
                "updated_at": now(),
            },
        )

    def run(self) -> int:
        try:
            self._assert_remote_precondition()
            self._update(
                status="starting",
                phase="v1131_model1000_full_resume_systemd_starting",
                formal_training_started=True,
                failure_class="none",
                wandb_sync_status="online_mirror_verified_resume_must_launch_pending",
                old_child_pid=2898910,
                old_child_status="stale_session_cleanup",
            )
            before = {path.resolve() for path in EXPERIMENT.iterdir() if path.is_dir()}
            log_stream = CHILD_LOG.open("a", encoding="utf-8", buffering=1)
            self.child = subprocess.Popen(
                self._command(),
                cwd=ROOT,
                env=self._environment(),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            atomic_text(TRAIN_PID, f"{self.child.pid}\n")
            self._update(
                status="running",
                phase="formal_E4000_continuous_training_recovery_model1000",
                wandb_sync_status="online_resume_must_initializing_at_step_1000",
            )
            self._write_handoff(
                "Persistent systemd supervisor launched the full model_1000 recovery child.",
                "Continue uninterrupted to cumulative effective_updates=4000; no automatic play/eval.",
            )
            started = time.monotonic()
            last_remote_poll = 0.0
            growth_seen = False
            while self.child.poll() is None:
                self._parse_log()
                if self.run_dir is None:
                    new_dirs = [path for path in EXPERIMENT.iterdir() if path.is_dir() and path.resolve() not in before]
                    if new_dirs:
                        self.run_dir = max(new_dirs, key=lambda path: path.stat().st_mtime)
                now_mono = time.monotonic()
                if self.wandb_initialized and now_mono - last_remote_poll >= 15.0:
                    try:
                        remote = self._remote()
                        if remote is not None:
                            self.remote_seen = True
                            self.last_remote_ok = now_mono
                            step = int(remote["step"])
                            if self.remote_latest_step is None or step > self.remote_latest_step:
                                self.remote_latest_step = step
                                self.remote_samples.append({"step": step, "observed_at": now()})
                                self.remote_samples = self.remote_samples[-20:]
                            growth_seen = growth_seen or step > 1000
                            self.state["wandb_sync_status"] = (
                                "online_remote_growth_verified"
                                if growth_seen
                                else "online_mirror_resume_verified_waiting_growth"
                            )
                            self.state["wandb_remote_state"] = remote["state"]
                            self.state["wandb_remote_latest_step"] = step
                            self.state["wandb_first_recovery_step"] = remote["first_recovery_step"]
                            self.state["wandb_live_step_samples"] = self.remote_samples
                    except Exception as error:
                        self.state["wandb_last_error"] = repr(error)
                    last_remote_poll = now_mono
                if now_mono - started > 300 and not self.remote_seen:
                    raise RuntimeError("W&B recovery run did not become remotely visible within 300 seconds")
                if now_mono - started > 600 and not growth_seen:
                    raise RuntimeError("W&B recovery history did not grow beyond resumed step 1000")
                if now_mono - self.last_remote_ok > 300:
                    raise RuntimeError("unrecoverable W&B remote verification outage exceeded 300 seconds")
                self._update()
                if self.stop_requested:
                    raise RuntimeError("supervisor stop requested")
                time.sleep(5)
            return_code = int(self.child.returncode or 0)
            self._parse_log()
            if return_code != 0:
                raise RuntimeError(f"training child exited with status {return_code}")
            if self.effective_count != TOTAL_COUNT:
                raise RuntimeError(f"training ended at effective count {self.effective_count}, expected {TOTAL_COUNT}")
            final = self.run_dir / "model_3998.pt" if self.run_dir else None
            if final is None or not final.is_file():
                raise RuntimeError("expected final model_3998.pt is missing")
            self._update(
                status="completed",
                phase="formal_E4000_training_complete_pending_manual_checkpoint_selection",
                active_pid=None,
                wandb_sync_status="online_remote_finalization_pending",
                final_checkpoint=str(final),
                final_checkpoint_sha256=sha256_file(final),
            )
            self._write_handoff(
                "Cumulative effective_updates=4000 completed under the persistent v1.13.1 recovery service.",
                "Wait for user-directed manual checkpoint play selection; do not auto-evaluate.",
            )
            return 0
        except Exception as error:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=30)
            self._parse_log()
            self._update(
                status="failed_closed",
                phase="v1131_infrastructure_recovery_failed_closed",
                active_pid=None,
                failure_class="infrastructure_or_training_process_failure",
                failure_detail=repr(error),
            )
            self._write_handoff(
                f"Recovery failed closed: {error!r}",
                "Inspect the single recorded failure before any authorized recovery; do not auto-retry.",
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(Supervisor().run())
