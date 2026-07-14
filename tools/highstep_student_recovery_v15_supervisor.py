#!/usr/bin/env python3
"""Autonomous fail-closed supervisor for the approved highstep v1.5 route."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import highstep_wandb_stage_gate as wandb_gate

PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
WORKFLOW_ID = "highstep_student_recovery_v15_20260713"
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_v15_20260713"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
PREREG = STATE_ROOT / "preregistration_v6.json"
PREREG_SHA = "0be872b32062fe6d162a6b25435ba81c7b45fbc30a703eb0d7597a29a8ff0421"
REFERENCE = STATE_ROOT / "0707_reference/0707_imitation_reference_manifest.json"
REFERENCE_SHA = "f910798c5b8fab23b37c7fc10ccd5ac9a734734f65e4e318d0dbdd3446527e9f"
SOURCE = STATE_ROOT / "source_model_172300_v3/model_172300.pt"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
PARENT_TEACHER_MANIFEST = ROOT / "tmp/highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json"
TRAIN_TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV15-ArcdogAdjustableLeg-v0"
EVAL_TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0"
EXPERIMENT_ROOT = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_v15_Student"
MONITOR = ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh"
CANDIDATE_AUDIT = ROOT / "scripts/rsl_rl/base/highstep_v15_candidate_audit_play.py"
IMITATION_GATE = ROOT / "tools/highstep_v15_imitation_gate.py"
DRIVER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
    "student_no_prior_Student/2026-07-05_00-13-46/model_158797.pt"
)
SAVE_POINTS = (100, 300, 500, 900, 1400, 1800, 2500)
CORE9_POINTS = {500, 900, 1400, 1800, 2500}
TRAIN_RUN_PREFIX = "v15"
PREREG_ENV_PREFIX = "HIGHSTEP_V15"
WARMUP_UPDATES = 1400
ABSOLUTE_CAP = 2500
ENVIRONMENT_PROFILE_MANIFEST = ""
ENVIRONMENT_PROFILE_MANIFEST_SHA256 = ""
SCHEDULE_RESUME_MODE = "preserve"
V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE = ""
V18_STAGE_TRANSITION_MANIFEST = ""
V18_STAGE_TRANSITION_MANIFEST_SHA256 = ""
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: Mapping[str, Any], read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def is_read_only(path: Path) -> bool:
    return not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def finite(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else default


class Supervisor:
    def __init__(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_path = STATE_ROOT / "state.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.lock_path = STATE_ROOT / "supervisor.lock"
        self.lock_stream = self.lock_path.open("a+")
        fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.active: subprocess.Popen[bytes] | None = None
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        existing = {}
        if self.state_path.is_file():
            try:
                candidate = json.loads(self.state_path.read_text())
                if candidate.get("workflow_id") == WORKFLOW_ID:
                    existing = candidate
            except Exception:
                pass
        self.state: dict[str, Any] = {
            **existing,
            "schema_version": 3,
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "reference_sha256": REFERENCE_SHA,
            "supervisor_pid": os.getpid(),
            "status": "starting",
            "phase": "preflight",
            "active_pid": None,
            "requires_user_action": False,
        }
        self.critical_hashes: dict[str, str] = {}
        self.update()

    def _signal(self, *_: Any) -> None:
        self.stop_requested = True
        if self.active is not None:
            self.stop_group(self.active)

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["updated_at"] = now()
        self.state["last_supervisor_check_at"] = time.time()
        atomic_json(self.state_path, self.state)
        atomic_json(self.heartbeat_path, {
            "schema_version": 1, "workflow_id": WORKFLOW_ID,
            "timestamp": self.state["updated_at"], "unix_time": time.time(),
            "supervisor_pid": os.getpid(), "status": self.state.get("status"),
            "phase": self.state.get("phase"), "active_pid": self.state.get("active_pid"),
            "checkpoint": self.state.get("checkpoint"),
            "effective_updates": self.state.get("effective_updates"),
        })

    def heartbeat_sleep(self, seconds: float) -> None:
        """Wait without making a healthy fail-closed supervisor look dead."""
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise RuntimeError("supervisor received stop request")
            self.update()
            time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def stop_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 30), (signal.SIGKILL, 5)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + grace
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(1)
            if process.poll() is not None:
                return

    @staticmethod
    def progress(path: Path) -> tuple[int, int]:
        total = 0
        newest = 0
        if path.exists():
            for item in path.rglob("*"):
                if not item.is_file():
                    continue
                try:
                    info = item.stat()
                except OSError:
                    continue
                total += info.st_size
                newest = max(newest, info.st_mtime_ns)
        return total, newest

    def assert_code(self) -> None:
        current = {name: sha256_file(Path(name)) for name in self.critical_hashes}
        if current != self.critical_hashes:
            changed = [name for name in current if current[name] != self.critical_hashes[name]]
            raise RuntimeError(f"critical v1.5 code changed while supervisor active: {changed}")

    def wait(self, process: subprocess.Popen[bytes], root: Path, phase: str, stall: int) -> None:
        self.active = process
        last = self.progress(root)
        last_progress = time.monotonic()
        last_code = 0.0
        while process.poll() is None:
            if self.stop_requested:
                self.stop_group(process)
                raise RuntimeError("supervisor received stop request")
            moment = time.monotonic()
            if moment - last_code >= 60:
                self.assert_code()
                last_code = moment
            current = self.progress(root)
            if current != last:
                last = current
                last_progress = moment
            elif moment - last_progress > stall:
                self.stop_group(process)
                raise RuntimeError(f"{phase} made no observable progress for {stall}s")
            self.update(status="running", phase=phase, active_pid=process.pid, progress_bytes=current[0])
            time.sleep(10)
        self.active = None
        self.update(active_pid=None)
        if process.returncode != 0:
            raise RuntimeError(f"{phase} exited with return code {process.returncode}")

    def run_command(self, command: Sequence[str], log: Path, root: Path, phase: str, env: Mapping[str, str] | None = None, retries: int = 3, stall: int = 1200) -> Path:
        log.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for attempt in range(1, retries + 1):
            if attempt == 1 and not log.exists():
                attempt_log = log
            else:
                retry = 1
                attempt_log = log.with_name(f"{log.stem}_infra_retry{retry}{log.suffix}")
                while attempt_log.exists():
                    retry += 1
                    attempt_log = log.with_name(f"{log.stem}_infra_retry{retry}{log.suffix}")
            environment = os.environ.copy()
            environment.pop("DISPLAY", None)
            environment.pop("XAUTHORITY", None)
            environment["PYTHONUNBUFFERED"] = "1"
            environment["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libstdc++.so.6"
            environment["TERM"] = "xterm"
            if env:
                environment.update(env)
            with attempt_log.open("wb") as stream:
                process = subprocess.Popen(list(command), cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    self.wait(process, root, f"{phase}_attempt{attempt}", stall)
                    return attempt_log
                except RuntimeError as error:
                    errors.append(str(error))
                    if self.stop_requested:
                        raise
            self.update(status="recovering", phase=phase, recovery_attempt=attempt, last_error=errors[-1])
        raise RuntimeError(f"{phase} failed after {retries} infrastructure attempts: {errors}")

    @staticmethod
    def active_gpu_jobs() -> list[dict[str, Any]]:
        jobs = []
        names = ("train.py", "play.py", "highstep_v15_candidate_audit_play.py", "highstep_centerline_guard_monitor")
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                argv = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
            except OSError:
                continue
            if any(any(Path(token).name == name or name in Path(token).name for name in names) for token in argv if token and not any(c.isspace() for c in token)):
                jobs.append({"pid": int(proc.name), "argv": " ".join(argv)})
        return jobs

    def require_idle(self) -> None:
        jobs = self.active_gpu_jobs()
        if jobs:
            raise RuntimeError(f"another managed train/eval/play process is active: {jobs}")

    def preflight(self) -> None:
        required = {SPEC: SPEC_SHA, PREREG: PREREG_SHA, REFERENCE: REFERENCE_SHA, SOURCE: TEACHER_SHA, TEACHER: TEACHER_SHA}
        for path, expected in required.items():
            if sha256_file(path) != expected:
                raise RuntimeError(f"authority SHA mismatch: {path}")
        self.require_idle()
        critical = [
            Path(__file__), CANDIDATE_AUDIT, IMITATION_GATE,
            ROOT / "tools/highstep_v15_reference_manifest.py",
            ROOT / "tools/highstep_same_state_audit.py",
            ROOT / "tools/highstep_wandb_stage_gate.py",
            ROOT / "scripts/rsl_rl/base/highstep_v15_reference_audit_play.py",
            ROOT / "scripts/rsl_rl/base/highstep_same_state_audit_play.py",
            ROOT / "scripts/rsl_rl/base/train.py", ROOT / "scripts/rsl_rl/base/play.py", MONITOR,
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        ]
        self.critical_hashes = {str(path.resolve()): sha256_file(path.resolve()) for path in critical}
        smoke = EXPERIMENT_ROOT / "2026-07-13_16-42-25_v15_smoke3_20260713_attempt6/model_2.pt"
        payload = torch.load(smoke, map_location="cpu", weights_only=False)
        recovery = payload["infos"][ALGORITHM_KEY]["student_recovery"]
        if payload["infos"][ALGORITHM_KEY]["student_distill_update_count"] != 3 or recovery["effective_update_count"] != 3:
            raise RuntimeError("discarded 3-update smoke checkpoint is invalid")
        audit_runtime = STATE_ROOT / "candidate_audit_runtime_smoke_attempt2/traces/seed11_nominal/run_summary.json"
        audit_payload = json.loads(audit_runtime.read_text())
        if not (
            audit_payload.get("frame_count") == 600
            and audit_payload.get("runner_manual_action_allclose_all_frames") is False
            and audit_payload.get("outcome", {}).get("same_state_runtime_contract_verified") is True
            and audit_payload.get("checkpoint_binding", {}).get("student_checkpoint_sha256") == sha256_file(smoke)
            and audit_payload.get("checkpoint_binding", {}).get("teacher_checkpoint_sha256") == TEACHER_SHA
        ):
            # The runner is intentionally the frozen 0707 driver, so runner/manual
            # parity with the side-effect-free candidate must be false.  All
            # physical-state mutation guards are still mandatory.
            raise RuntimeError("candidate imitation runtime smoke binding is invalid")
        audit = {
            "schema_version": 1, "kind": "highstep_v15_preflight",
            "spec_sha256": SPEC_SHA, "preregistration_sha256": PREREG_SHA,
            "reference_sha256": REFERENCE_SHA, "protected_teacher_sha256": TEACHER_SHA,
            "smoke_checkpoint": str(smoke), "smoke_checkpoint_sha256": sha256_file(smoke),
            "smoke_effective_updates": 3, "smoke_discarded": True,
            "smoke_actor_frozen": recovery["binding_manifest"]["actor_body_frozen"] is True,
            "smoke_ppo_disabled": recovery["binding_manifest"]["student_ppo_permanently_disabled"] is True,
            "candidate_audit_runtime_smoke": str(audit_runtime),
            "candidate_audit_runtime_smoke_sha256": sha256_file(audit_runtime),
            "critical_code_sha256": self.critical_hashes,
            "no_concurrent_gpu_job": True, "completed_at": now(),
        }
        preflight_path = STATE_ROOT / "v15_preflight_manifest_v6.json"
        if preflight_path.exists():
            existing = json.loads(preflight_path.read_text())
            stable_keys = (
                "spec_sha256", "preregistration_sha256", "reference_sha256",
                "protected_teacher_sha256", "smoke_checkpoint_sha256",
                "candidate_audit_runtime_smoke_sha256", "critical_code_sha256",
            )
            if not is_read_only(preflight_path) or any(existing.get(key) != audit.get(key) for key in stable_keys):
                raise RuntimeError("stored v1.5 preflight manifest is stale or writable")
        else:
            atomic_json(preflight_path, audit, read_only=True)
        self.update(status="preflight_passed", phase="ready_for_stage_100", code_sha256=self.critical_hashes)

    @staticmethod
    def checkpoint_count(path: Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload["infos"][ALGORITHM_KEY]["student_distill_update_count"])

    def find_checkpoint(self, run_dir: Path, expected: int) -> Path:
        matches = []
        for path in run_dir.glob("model_*.pt"):
            try:
                if self.checkpoint_count(path) == expected:
                    matches.append(path)
            except Exception:
                continue
        if not matches:
            raise RuntimeError(f"run has no checkpoint at effective update {expected}: {run_dir}")
        return max(matches, key=lambda item: item.stat().st_mtime_ns)

    def checkpoint_scope(self, checkpoint: Path, stage: int, run_dir: Path) -> dict[str, Any]:
        """Audit a completed checkpoint under this supervisor's authority."""
        from tools.highstep_v15_imitation_gate import checkpoint_scope

        return checkpoint_scope(checkpoint, stage)

    def wandb_files(self, stage: int, source: Path, additional: int) -> tuple[Path, Path]:
        config_path = STATE_ROOT / "wandb_stage_configs" / f"E{stage}.json"
        run_name = f"{WORKFLOW_ID}_R_E{stage}_attempt1"
        if not config_path.exists():
            atomic_json(config_path, {
                "workflow_id": WORKFLOW_ID, "route": "R", "stage": f"E{stage}", "attempt": "attempt1",
                "run_name": run_name, "group": WORKFLOW_ID, "spec_sha256": SPEC_SHA,
                "preregistration_sha256": PREREG_SHA,
                "start_checkpoint": str(source.resolve()), "start_checkpoint_sha256": sha256_file(source),
                "teacher_checkpoint": str(TEACHER.resolve()), "teacher_sha256": TEACHER_SHA,
                "frozen_tensors": ["teacher_actor.*", "teacher_priv_encoder.*", "critic.*", "priv_encoder.*", "std", f"actor.* through update {WARMUP_UPDATES}", "actor body and rows 0:12 permanently"],
                "trainable_tensors": ["estimator.*"] if stage <= WARMUP_UPDATES else ["estimator.*", "actor.6.weight[12:16]", "actor.6.bias[12:16]"],
                "optimizer": {"class": "Adam", "estimator_lr": 1e-3, "box_rows_lr": 1e-5, "ppo": "permanently_disabled"},
                "budget": {"additional_updates": additional, "effective_update_target": stage, "absolute_cap": ABSOLUTE_CAP},
                "training_task": TRAIN_TASK, "historical_sync": False,
            }, read_only=True)
        manifest = STATE_ROOT / "wandb_stages" / run_name / "stage_manifest.json"
        if not manifest.exists():
            manifest = wandb_gate.prepare_stage(config_path, STATE_ROOT)
        return config_path, manifest

    def train_stage(self, stage: int, source: Path, previous: int) -> tuple[Path, Path, Path]:
        stage_root = STATE_ROOT / "stages" / f"E{stage}"
        training_result = stage_root / "training_result.json"
        if training_result.is_file():
            record = json.loads(training_result.read_text())
            checkpoint = Path(record["checkpoint"]).resolve(strict=True)
            run_dir = Path(record["run_dir"]).resolve(strict=True)
            if record["checkpoint_sha256"] != sha256_file(checkpoint) or self.checkpoint_count(checkpoint) != stage:
                raise RuntimeError(f"stale training result E{stage}")
            return run_dir, checkpoint, Path(record["wandb_stage_manifest"])
        additional = stage - previous
        _, wandb_manifest = self.wandb_files(stage, source, additional)
        run_name = f"{TRAIN_RUN_PREFIX}_E{stage}_{time.strftime('%Y%m%d_%H%M%S')}"
        log = stage_root / "train.log"
        load_mode = "weights_only" if previous == 0 else "full"
        command = [
            str(PYTHON), "-u", "scripts/rsl_rl/base/train.py", "--task", TRAIN_TASK,
            "--num_envs", "4096", "--max_iterations", str(additional), "--seed", "42", "--headless",
            "--logger", "wandb", "--log_project_name", "isaaclab", "--resume", "--checkpoint", str(source.resolve()),
            "--highstep_resume_mode", "refine", "--highstep_checkpoint_load_mode", load_mode,
            "--highstep_schedule_resume_mode", SCHEDULE_RESUME_MODE, "--highstep_parent_teacher_manifest", str(PARENT_TEACHER_MANIFEST),
            "--run_name", run_name,
        ]
        if ENVIRONMENT_PROFILE_MANIFEST or ENVIRONMENT_PROFILE_MANIFEST_SHA256:
            if not (ENVIRONMENT_PROFILE_MANIFEST and ENVIRONMENT_PROFILE_MANIFEST_SHA256):
                raise RuntimeError("environment profile manifest binding is incomplete")
            command.extend([
                "--highstep_v18_environment_profile_manifest",
                ENVIRONMENT_PROFILE_MANIFEST,
                "--highstep_v18_environment_profile_manifest_sha256",
                ENVIRONMENT_PROFILE_MANIFEST_SHA256,
            ])
        if V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE or V18_STAGE_TRANSITION_MANIFEST:
            if not (
                V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE
                and V18_STAGE_TRANSITION_MANIFEST
                and V18_STAGE_TRANSITION_MANIFEST_SHA256
            ):
                raise RuntimeError("v1.8 schedule transition binding is incomplete")
            command.extend([
                "--highstep_v18_schedule_anchor_effective_update",
                V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE,
                "--highstep_v18_stage_transition_manifest",
                V18_STAGE_TRANSITION_MANIFEST,
                "--highstep_v18_stage_transition_manifest_sha256",
                V18_STAGE_TRANSITION_MANIFEST_SHA256,
            ])
        launch_path = stage_root / "launch.json"
        if launch_path.exists():
            retry = 1
            launch_path = stage_root / f"launch_infra_retry{retry}.json"
            while launch_path.exists():
                retry += 1
                launch_path = stage_root / f"launch_infra_retry{retry}.json"
        atomic_json(launch_path, {
            "command": command, "source": str(source.resolve()), "source_sha256": sha256_file(source),
            "load_mode": load_mode, "additional_updates": additional, "effective_target": stage,
            "wandb_stage_manifest": str(wandb_manifest), "created_at": now(),
        }, read_only=True)
        self.require_idle()
        environment = wandb_gate.process_environment(wandb_manifest)
        environment[f"{PREREG_ENV_PREFIX}_PREREGISTRATION_PATH"] = str(PREREG)
        environment[f"{PREREG_ENV_PREFIX}_PREREGISTRATION_SHA256"] = PREREG_SHA
        # W&B network availability must never hold the GPU before update zero.
        # The complete independent run is written locally, then fail-closed
        # synchronization is required before the next training stage.
        environment["WANDB_MODE"] = "offline"
        environment["WANDB_X_DISABLE_STATS"] = "true"
        environment["WANDB_DISABLE_GIT"] = "true"
        completed_log = self.run_command(
            command, log, stage_root, f"training_{stage}", environment, retries=3, stall=1200
        )
        matches = sorted(EXPERIMENT_ROOT.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime_ns)
        if len(matches) != 1:
            raise RuntimeError(f"expected one completed v1.5 run for {run_name}, got {matches}")
        run_dir = matches[0]
        checkpoint = self.find_checkpoint(run_dir, stage)
        scope_output = stage_root / "checkpoint_scope.json"
        command_scope = [str(PYTHON), str(IMITATION_GATE), "--checkpoint", str(checkpoint), "--audit-root", str(stage_root / "not_yet_collected"), "--effective-updates", str(stage), "--output", str(scope_output)]
        # The default implementation remains the frozen v1.5 helper.  Newer
        # supervisors may narrow this hook to their own authority without
        # widening or modifying the archived helper.
        scope = self.checkpoint_scope(checkpoint, stage, run_dir)
        atomic_json(scope_output, scope, read_only=True)
        atomic_json(training_result, {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "effective_updates": stage, "run_dir": str(run_dir), "train_log": str(completed_log),
            "train_log_sha256": sha256_file(completed_log), "checkpoint_scope": str(scope_output),
            "checkpoint_scope_sha256": sha256_file(scope_output), "wandb_stage_manifest": str(wandb_manifest),
            "completed_at": now(),
        }, read_only=True)
        self.update(checkpoint=str(checkpoint), effective_updates=stage, run_dir=str(run_dir))
        return run_dir, checkpoint, wandb_manifest

    def imitation(self, stage: int, checkpoint: Path) -> dict[str, Any]:
        root = STATE_ROOT / "stages" / f"E{stage}" / "imitation"
        terminal = root / "imitation_gate.json"
        if terminal.is_file():
            payload = json.loads(terminal.read_text())
            if payload["candidate_checkpoint_sha256"] != sha256_file(checkpoint):
                raise RuntimeError("stored imitation gate checkpoint changed")
            return payload
        traces = root / "traces"
        for seed in (11, 22, 33):
            summary = traces / f"seed{seed}_nominal/run_summary.json"
            if summary.is_file():
                continue
            self.require_idle()
            env = {
                "HIGHSTEP_V15_CANDIDATE_CHECKPOINT": str(checkpoint.resolve()),
                "HIGHSTEP_V15_CANDIDATE_SHA256": sha256_file(checkpoint),
                "HIGHSTEP_V15_CANDIDATE_AUDIT_ROOT": str(root.resolve()),
            }
            command = [str(PYTHON), "-u", str(CANDIDATE_AUDIT), "--trajectory", f"seed{seed}_nominal", "--output-root", str(traces), "--task", "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0", "--num_envs", "1", "--headless", "--checkpoint", str(DRIVER)]
            self.run_command(command, root / f"seed{seed}.log", root, f"imitation_E{stage}_seed{seed}", env, retries=3, stall=1200)
        command = [str(PYTHON), str(IMITATION_GATE), "--checkpoint", str(checkpoint), "--audit-root", str(traces), "--effective-updates", str(stage), "--output", str(terminal)]
        self.run_command(command, root / "gate.log", root, f"imitation_gate_E{stage}", retries=1, stall=300)
        return json.loads(terminal.read_text())

    @staticmethod
    def eval_payload(log: Path, checkpoint: Path) -> dict[str, Any]:
        payloads = []
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[HIGHSTEP_EVAL_JSON]" in line:
                payloads.append(json.loads(line.split("[HIGHSTEP_EVAL_JSON]", 1)[1].strip()))
        if len(payloads) != 1:
            raise RuntimeError(f"expected one schema7 evaluation payload: {log}")
        item = payloads[0]
        if not (
            item.get("schema_version") == 7 and item.get("checkpoint_sha256") == sha256_file(checkpoint)
            and item.get("reset_valid") is True and item.get("initial_geometry_valid") is True
            and item.get("target_action_order_valid") is True and item.get("target_limit_contract_valid") is True
            and item.get("eval_action_delay_requested") == 0 and item.get("eval_action_delay_runtime_match") is True
            and item.get("keep_play_randomization") is False and item.get("schedule_valid") is True
            and item.get("schedule_runtime_match") is True and item.get("schedule_clock_runtime_match") is True
        ):
            raise RuntimeError(f"invalid behavior probe runtime contract: {log}")
        return item

    @staticmethod
    def play_command(checkpoint: Path, seed: int, lateral: str, yaw: str) -> list[str]:
        return [
            str(PYTHON), "scripts/rsl_rl/base/play.py", "--task", EVAL_TASK, "--num_envs", "1", "--headless",
            "--seed", str(seed), "--checkpoint", str(checkpoint.resolve()), "--play_terrain_type", "box", "--play_terrain_level", "9",
            "--fixed_velocity_command", "0.45", "0.0", "0.0", "--reset_after_play_terrain_selection", "--front_step_eval_reset",
            "--front_step_eval_side", "x-", "--front_step_eval_edge_gap", "0.55", "--front_step_eval_lateral_offset", lateral,
            "--front_step_eval_yaw_offset_deg", yaw, "--print_rear_width_metrics", "--skip_policy_export",
            "--rear_width_metric_interval", "60", "--play_max_steps", "600", "--eval_action_delay_steps", "0",
        ]

    @staticmethod
    def no_severe(item: Mapping[str, Any]) -> bool:
        return bool(finite(item.get("critical_rear_min_abs_y_q05"), -1) >= 0.04 and finite(item.get("critical_rear_width_q05"), -1) >= 0.18 and finite(item.get("critical_center_violation_rate"), 1) <= 0.10 and finite(item.get("critical_width_violation_rate"), 1) <= 0.10)

    def probe(self, stage: int, checkpoint: Path) -> dict[str, Any]:
        root = STATE_ROOT / "stages" / f"E{stage}" / "probe3"
        terminal = root / "probe_manifest.json"
        if terminal.is_file():
            return json.loads(terminal.read_text())
        scenarios = ((11, "nominal", "0.00", "0.0"), (11, "left_offset", "0.12", "4.0"), (22, "right_offset", "-0.12", "-4.0"))
        rows = []
        for seed, name, lateral, yaw in scenarios:
            log = root / f"seed{seed}_{name}.log"
            if not log.exists():
                self.require_idle()
                self.run_command(self.play_command(checkpoint, seed, lateral, yaw), log, root, f"probe_E{stage}_{name}", retries=3, stall=900)
            item = self.eval_payload(log, checkpoint)
            rows.append({
                "seed": seed, "scenario": name, "valid": True,
                "full_climb": item.get("full_climb_success") is True,
                "rear_hold": item.get("rear_on_platform_hold_success") is True,
                "front_top_support": item.get("front_top_support_reached") is True,
                "no_severe_inward": self.no_severe(item), "log": str(log), "log_sha256": sha256_file(log),
            })
        counts = {key: sum(row[key] is True for row in rows) for key in ("valid", "full_climb", "rear_hold", "front_top_support", "no_severe_inward")}
        payload = {
            "schema_version": 1, "kind": "highstep_v15_behavior_probe3", "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint), "effective_updates": stage, "rows": rows, "counts": counts,
            "behavior_score": counts["full_climb"] + counts["rear_hold"], "support_score": counts["front_top_support"],
            "catastrophic_validity_passed": counts["valid"] == 3 and counts["no_severe_inward"] >= 2,
            "completed_at": now(),
        }
        atomic_json(terminal, payload, read_only=True)
        return payload

    def core9(self, stage: int, checkpoint: Path, run_dir: Path) -> dict[str, Any]:
        root = STATE_ROOT / "stages" / f"E{stage}" / "core9"
        terminal = root / "core9_summary.json"
        if terminal.is_file():
            return json.loads(terminal.read_text())
        self.require_idle()
        environment = {
            "RECOVERY_SPEC_MODE": "1", "WORKFLOW_ID": WORKFLOW_ID,
            "LEDGER_FILE": str(STATE_ROOT / "core9_ledger.jsonl"), "EVAL_CHECKPOINT_COUNT": "1",
            "EVAL_CHECKPOINT_PATH": str(checkpoint.resolve()), "EVAL_CHECKPOINT_STRIDE": "1",
            "EVAL_ALLOW_LEGACY_SCHEDULE_FALLBACK": "0", "POLL_SECONDS": "5",
            "PLAY_STALL_TIMEOUT_SECONDS": "900", "PLAY_WATCHDOG_POLL_SECONDS": "10", "MIN_FREE_DISK_GIB": "15",
        }
        command = ["bash", str(MONITOR), "0", str(run_dir.resolve()), "/dev/null", str(root), EVAL_TASK, "student", str(PARENT_TEACHER_MANIFEST)]
        self.run_command(command, root / "launcher.log", root, f"core9_E{stage}", environment, retries=3, stall=1500)
        manifest = json.loads((root / "evaluation_manifest.json").read_text())
        rows = [json.loads(line) for line in (root / "eval_runs.jsonl").read_text().splitlines() if line.strip()]
        matrix = {(seed, scenario) for seed in (11, 22, 33) for scenario in ("nominal", "left_offset", "right_offset")}
        if not (manifest.get("evaluation_complete") is True and manifest.get("matrix_complete") is True and manifest.get("workflow_id") == WORKFLOW_ID and manifest.get("task") == EVAL_TASK and len(rows) == 9 and {(row.get("seed"), row.get("scenario")) for row in rows} == matrix):
            raise RuntimeError("core9 manifest/matrix is incomplete")
        evaluations = []
        for row in rows:
            if row.get("return_code") != 0 or row.get("randomize") is not False or row.get("action_delay_steps") != 0:
                raise RuntimeError("core9 row runtime contract failed")
            item = row.get("eval")
            if not isinstance(item, dict) or item.get("checkpoint_sha256") != sha256_file(checkpoint) or item.get("schema_version") != 7 or item.get("schedule_valid") is not True or item.get("reset_valid") is not True:
                raise RuntimeError("core9 schema7 payload invalid")
            evaluations.append(item)
        counts = {
            "valid": 9,
            "full_climb": sum(item.get("full_climb_success") is True for item in evaluations),
            "rear_hold": sum(item.get("rear_on_platform_hold_success") is True for item in evaluations),
            "front_top_support": sum(item.get("front_top_support_reached") is True for item in evaluations),
            "no_severe_inward": sum(self.no_severe(item) for item in evaluations),
        }
        payload = {
            "schema_version": 1, "kind": "highstep_v15_real_gain_core9", "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint), "effective_updates": stage, "counts": counts,
            "final_behavior_gate_passed": counts["valid"] == 9 and counts["full_climb"] >= 8 and counts["rear_hold"] >= 8 and counts["front_top_support"] >= 8 and counts["no_severe_inward"] >= 8,
            "evaluation_manifest": str(root / "evaluation_manifest.json"), "evaluation_manifest_sha256": sha256_file(root / "evaluation_manifest.json"),
            "completed_at": now(),
        }
        atomic_json(terminal, payload, read_only=True)
        return payload

    @staticmethod
    def latent_mean(imitation: Mapping[str, Any]) -> float:
        values = [float(row["latent_mae"]) for row in imitation.get("phase_metrics", {}).values() if isinstance(row, Mapping) and finite(row.get("latent_mae"), -1) >= 0]
        return sum(values) / len(values) if values else math.inf

    def decision(self, stage: int, imitation: Mapping[str, Any], probe: Mapping[str, Any], core9: Mapping[str, Any] | None) -> dict[str, Any]:
        history = []
        for point in SAVE_POINTS:
            path = STATE_ROOT / "stages" / f"E{point}" / "stage_result.json"
            if point < stage and path.is_file():
                history.append(json.loads(path.read_text()))
        scores = [float(item["probe"]["behavior_score"]) for item in history] + [float(probe["behavior_score"])]
        supports = [float(item["probe"]["support_score"]) for item in history] + [float(probe["support_score"])]
        best_score = max(scores, default=0.0)
        best_support = max(supports, default=0.0)
        score_ratio = float(probe["behavior_score"]) / best_score if best_score else 1.0
        support_ratio = float(probe["support_score"]) / best_support if best_support else 1.0
        previous = history[-1] if history else None
        latent = self.latent_mean(imitation)
        latent_improved = previous is None or latent < float(previous["decision"]["latent_mean"]) - 1e-6
        imitation_failed_consecutive = bool(previous and not previous["imitation"]["passed"] and not imitation["passed"])
        behavior_degraded = bool(previous and (probe["behavior_score"], probe["support_score"]) < (previous["probe"]["behavior_score"], previous["probe"]["support_score"]))
        prior_behavior_degraded = bool(previous and previous["decision"].get("behavior_degraded"))
        continuous_regression = (imitation_failed_consecutive or (behavior_degraded and prior_behavior_degraded)) and not latent_improved
        late_floor = score_ratio >= 0.90 and support_ratio >= 0.90
        final_floor = score_ratio >= 0.93 and support_ratio >= 0.93
        final_pass = bool(imitation["passed"] and core9 and core9["final_behavior_gate_passed"] and final_floor)
        stop = bool(not probe["catastrophic_validity_passed"] or continuous_regression or not late_floor)
        reason = "continue"
        if final_pass:
            reason = "numeric_candidate_gate_passed"
        elif not probe["catastrophic_validity_passed"]:
            reason = "behavior_probe_catastrophic_failure"
        elif continuous_regression:
            reason = "two_consecutive_regressions_without_latent_improvement"
        elif not late_floor:
            reason = "late_window_below_0.90_best_floor"
        elif stage == 1400 and not imitation["passed"]:
            stop = True; reason = "1400_imitation_gate_failed_box_head_remains_locked"
        elif stage == 2500:
            stop = True; reason = "absolute_2500_cap_without_final_gate"
        return {
            "imitation_passed": imitation["passed"], "latent_mean": latent, "latent_improved": latent_improved,
            "behavior_degraded": behavior_degraded, "continuous_regression": continuous_regression,
            "late_window_score": probe["behavior_score"], "best_score": best_score, "late_to_best_score_ratio": score_ratio,
            "late_window_support": probe["support_score"], "best_support": best_support, "late_to_best_support_ratio": support_ratio,
            "late_minimum_floor_passed": late_floor, "candidate_0_93_floor_passed": final_floor,
            "final_numeric_gate_passed": final_pass, "stop": stop, "reason": reason,
        }

    def finalize_wandb(self, stage: int, checkpoint: Path, stage_result: Path, wandb_manifest: Path, gate_metrics: Mapping[str, Any], conclusion: str) -> None:
        summary = STATE_ROOT / "wandb_stage_summaries" / f"E{stage}.json"
        if not summary.exists():
            atomic_json(summary, {
                "output_checkpoint": str(checkpoint), "output_checkpoint_sha256": sha256_file(checkpoint),
                "effective_updates": stage, "gate_metrics": dict(gate_metrics), "gate_conclusion": conclusion,
                "manifest_path": str(stage_result), "manifest_sha256": sha256_file(stage_result),
            }, read_only=True)
        manifest_payload = json.loads(wandb_manifest.read_text())
        run_id = str(manifest_payload["run_id"])
        offline_marker = wandb_manifest.parent / "offline_training_run_synced.json"
        while True:
            candidates = wandb_gate.offline_run_candidates(ROOT / "wandb", run_id)
            if not candidates:
                self.update(
                    status="pending_sync", phase=f"wandb_sync_E{stage}",
                    last_error=f"offline W&B run directory for {run_id} not found",
                )
                self.heartbeat_sleep(300)
                continue
            pending = wandb_gate.pending_offline_runs(ROOT / "wandb", offline_marker, run_id)
            if not pending:
                break
            offline_run = pending[0]
            sync_log = wandb_manifest.parent / "offline_sync.log"
            command = [
                str(PYTHON.parent / "wandb"), "sync", "--id", run_id,
                "--project", str(manifest_payload["project"]),
                "--entity", str(manifest_payload["entity"]),
                "--include-offline", str(offline_run),
            ]
            if wandb_gate.offline_sync_requires_append(offline_marker, run_id):
                command.insert(-1, "--append")
            environment = os.environ.copy()
            environment["WANDB_ENTITY"] = str(manifest_payload["entity"])
            environment["WANDB_PROJECT"] = str(manifest_payload["project"])
            try:
                with sync_log.open("ab") as stream:
                    process = subprocess.Popen(
                        command, cwd=ROOT, env=environment, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    self.active = process
                    deadline = time.monotonic() + 180
                    while process.poll() is None:
                        if self.stop_requested:
                            self.stop_group(process)
                            raise RuntimeError("supervisor received stop request")
                        if time.monotonic() >= deadline:
                            self.stop_group(process)
                            raise subprocess.TimeoutExpired(command, 180)
                        self.update(
                            status="pending_sync", phase=f"wandb_sync_E{stage}",
                            active_pid=process.pid, last_error=None,
                        )
                        time.sleep(5)
                    self.active = None
                self.update(active_pid=None)
                if process.returncode != 0:
                    raise RuntimeError(f"wandb sync exited {process.returncode}")
                wandb_gate.record_offline_run_synced(
                    offline_marker, run_id=run_id, offline_run=offline_run,
                    sync_log=sync_log, synced_at=now(), expected_offline_runs=candidates,
                )
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
                self.active = None
                self.update(
                    status="pending_sync", phase=f"wandb_sync_E{stage}", active_pid=None,
                    last_error=f"{type(error).__name__}: {error}",
                )
                self.heartbeat_sleep(300)
        while True:
            result = wandb_gate.finalize_stage(wandb_manifest, summary)
            if result.get("sync_status") == "synced":
                return
            self.update(status="pending_sync", phase=f"wandb_sync_E{stage}", last_error=result.get("sync_error"))
            self.heartbeat_sleep(300)

    @staticmethod
    def wandb_offline_coverage_complete(wandb_manifest: Path) -> bool:
        payload = json.loads(wandb_manifest.read_text())
        run_id = str(payload["run_id"])
        marker = wandb_manifest.parent / "offline_training_run_synced.json"
        candidates = wandb_gate.offline_run_candidates(ROOT / "wandb", run_id)
        return bool(candidates) and not wandb_gate.pending_offline_runs(ROOT / "wandb", marker, run_id)

    @staticmethod
    def wandb_gate_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
        imitation = result["imitation"]
        probe = result["probe"]
        core = result.get("core9")
        decision = result["decision"]
        return {
            "imitation_passed": imitation["passed"],
            "imitation_failed_elements": imitation["failed_element_count"],
            "probe": probe["counts"],
            "core9": None if core is None else core["counts"],
            "late_to_best_score_ratio": decision["late_to_best_score_ratio"],
            "late_to_best_support_ratio": decision["late_to_best_support_ratio"],
        }

    def record_terminal(self, status: str, reason: str, best: Mapping[str, Any] | None = None) -> None:
        handoff = {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "status": status, "reason": reason,
            "checkpoint": self.state.get("checkpoint"), "effective_updates": self.state.get("effective_updates"),
            "best_checkpoint": None if best is None else best.get("checkpoint"),
            "best_checkpoint_sha256": None if best is None else best.get("checkpoint_sha256"),
            "imitation_gate": self.state.get("imitation_gate"), "behavior_probe": self.state.get("behavior_probe"),
            "core9": self.state.get("core9"), "requires_user_action": status == "student_candidate_pending_user_visual_review",
            "next_action": "user_visual_review" if status == "student_candidate_pending_user_visual_review" else "none_under_v1.5",
            "written_at": now(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update(status=status, phase=status, stop_reason=reason, requires_user_action=handoff["requires_user_action"], active_pid=None)

    def videos(self, stage: int, checkpoint: Path) -> Path:
        root = STATE_ROOT / "candidate_visual_review" / f"E{stage}"
        manifest_path = root / "paired_video_manifest.json"
        if manifest_path.is_file():
            return manifest_path
        records = []
        scenarios = (("nominal", "0.00", "0.0"), ("left_offset", "0.12", "4.0"), ("right_offset", "-0.12", "-4.0"))
        for name, lateral, yaw in scenarios:
            for role, policy, task in (("student", checkpoint, EVAL_TASK), ("teacher", TEACHER, "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0")):
                label = f"{role}_{name}"
                destination = root / "videos" / f"{label}.mp4"
                if destination.exists():
                    records.append({"label": label, "video": str(destination), "sha256": sha256_file(destination)})
                    continue
                source_dir = policy.parent / "videos/play"
                before = {path.resolve(): path.stat().st_mtime_ns for path in source_dir.glob("*.mp4")} if source_dir.exists() else {}
                command = self.play_command(policy, 11, lateral, yaw)
                command[command.index(EVAL_TASK)] = task
                command.extend(["--enable_cameras", "--video", "--video_length", "600", "--highstep_gap_camera", "side_top"])
                self.run_command(command, root / "logs" / f"{label}.log", root, f"video_{label}", retries=3, stall=1200)
                candidates = [path for path in source_dir.glob("*.mp4") if path.resolve() not in before or path.stat().st_mtime_ns > before[path.resolve()]]
                if not candidates:
                    raise RuntimeError(f"video file missing for {label}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(max(candidates, key=lambda path: path.stat().st_mtime_ns), destination)
                records.append({"label": label, "video": str(destination), "sha256": sha256_file(destination)})
        atomic_json(manifest_path, {
            "schema_version": 1, "kind": "highstep_v15_teacher_student_visual_review_videos",
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "videos": records,
            "robot_visibility_requires_user_review": True, "candidate_status_after_generation": "student_candidate_pending_user_visual_review",
        }, read_only=True)
        return manifest_path

    def run(self) -> None:
        self.preflight()
        source = SOURCE
        previous = 0
        best: dict[str, Any] | None = None
        for stage in SAVE_POINTS:
            self.assert_code()
            result_path = STATE_ROOT / "stages" / f"E{stage}" / "stage_result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                checkpoint = Path(result["checkpoint"]).resolve(strict=True)
                if result["checkpoint_sha256"] != sha256_file(checkpoint):
                    raise RuntimeError(f"stage result E{stage} checkpoint changed")
                wandb_manifest = Path(result["wandb_stage_manifest"]).resolve(strict=True)
                wandb_status = json.loads(wandb_manifest.read_text()).get("sync_status")
                if wandb_status != "synced" or not self.wandb_offline_coverage_complete(wandb_manifest):
                    self.finalize_wandb(
                        stage, checkpoint, result_path, wandb_manifest,
                        self.wandb_gate_metrics(result), result["decision"]["reason"],
                    )
                source, previous = checkpoint, stage
                best = result if best is None or tuple(result["selection_rank"]) > tuple(best["selection_rank"]) else best
                if result["decision"]["stop"]:
                    status = "student_candidate_pending_user_visual_review" if result["decision"]["final_numeric_gate_passed"] else "stopped_by_gate"
                    self.record_terminal(status, result["decision"]["reason"], best)
                    return
                continue
            run_dir, checkpoint, wandb_manifest = self.train_stage(stage, source, previous)
            imitation = self.imitation(stage, checkpoint)
            probe = self.probe(stage, checkpoint)
            core = self.core9(stage, checkpoint, run_dir) if stage in CORE9_POINTS else None
            decision = self.decision(stage, imitation, probe, core)
            selection_rank = [
                int(bool(imitation["passed"])),
                0 if core is None else min(core["counts"]["full_climb"], core["counts"]["rear_hold"]),
                0 if core is None else core["counts"]["full_climb"] + core["counts"]["rear_hold"],
                probe["behavior_score"], probe["support_score"], -stage,
            ]
            result = {
                "schema_version": 1, "kind": "highstep_v15_stage_result", "stage": stage,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "imitation": imitation, "probe": probe, "core9": core, "decision": decision,
                "selection_rank": selection_rank, "wandb_stage_manifest": str(wandb_manifest), "completed_at": now(),
            }
            atomic_json(result_path, result, read_only=True)
            self.finalize_wandb(
                stage, checkpoint, result_path, wandb_manifest,
                self.wandb_gate_metrics(result), decision["reason"],
            )
            best = result if best is None or tuple(selection_rank) > tuple(best["selection_rank"]) else best
            self.update(
                checkpoint=str(checkpoint), effective_updates=stage, imitation_gate={"passed": imitation["passed"], "failed_elements": imitation["failed_element_count"]},
                behavior_probe=probe["counts"], core9=None if core is None else core["counts"], best_checkpoint=best["checkpoint"],
                best_checkpoint_sha256=best["checkpoint_sha256"], decision=decision,
            )
            if decision["final_numeric_gate_passed"]:
                video_manifest = self.videos(stage, checkpoint)
                self.update(video_manifest=str(video_manifest), video_manifest_sha256=sha256_file(video_manifest))
                self.record_terminal("student_candidate_pending_user_visual_review", decision["reason"], best)
                return
            if decision["stop"]:
                self.record_terminal("stopped_by_gate", decision["reason"], best)
                return
            source, previous = checkpoint, stage
        self.record_terminal("stopped_by_gate", "absolute_2500_cap_without_final_gate", best)


def main() -> int:
    supervisor: Supervisor | None = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("v1.5 supervisor lock is already held", file=sys.stderr)
        return 3
    except Exception as error:
        if supervisor is not None:
            payload = {
                "schema_version": 1, "workflow_id": WORKFLOW_ID, "status": "external_fault_needs_user",
                "error": f"{type(error).__name__}: {error}", "checkpoint": supervisor.state.get("checkpoint"),
                "effective_updates": supervisor.state.get("effective_updates"), "written_at": now(),
                "next_action": "systemd_restart_will_retry; user only needed if restart limit is exhausted",
            }
            atomic_json(supervisor.handoff_path, payload)
            supervisor.update(status="external_fault_needs_user", phase="external_fault_needs_user", last_error=payload["error"], active_pid=None)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
