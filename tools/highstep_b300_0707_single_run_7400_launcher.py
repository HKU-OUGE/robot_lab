#!/usr/bin/env python3
"""One-shot launcher for the immutable B300 -> 0707-derived 7400 run.

This process records liveness and the two approved clocks only. It never
restarts, resumes, evaluates, gates, or modifies the training child.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
FLOW = ROOT / "tmp/highstep_b300_0707_derived_single_run_7400_20260718"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_b300_0707_single_run_7400_spec_20260718.md"
SPEC_SHA = "b73470a6b15a9fb5595b7153c7cb155878c2a2c4c63590b178b7b3f640af5fd3"
PREREG = FLOW / "preregistration.json"
PREREG_SHA = "24423ca413b5d924d3388124b4bd84983acc98555a745891aeca52d7dfd11207"
WANDB_CONFIG = FLOW / "wandb_config.json"
WANDB_CONFIG_SHA = "5db44c7de3741ef1bf4b52a81c09876e8a303ac1bb589dab4c768aa98d84d50f"
WANDB_READY = FLOW / "wandb_online_ready.json"
STATE = FLOW / "state.json"
HEARTBEAT = FLOW / "heartbeat.json"
HANDOFF = FLOW / "handoff.json"
LOCK = FLOW / "workflow.lock"
LAUNCHER_PID = FLOW / "launcher.pid"
TRAIN_PID = FLOW / "train.pid"
LOG = FLOW / "train.log"
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/"
    "2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
)
TEACHER_SHA = "d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431"
PARENT = ROOT / "tmp/highstep_be300_0707_distill_20260716/teacher_parent_lineage_v1131.json"
PARENT_SHA = "82dd4934388e5433718394f25d1dcc72c09ae18cddc2a3355a3a75556f4a8f93"
TASK = (
    "RobotLab-Isaac-Velocity-HighstepB3000707DerivedSingleRun7400StudentNoPrior-"
    "ArcdogAdjustableLeg-v0"
)
WORKFLOW = "highstep_b300_0707_derived_single_run_7400_20260718"
RUN_NAME = "highstep_b300_0707_single_run_7400_student_from_173499"
WANDB_ENTITY = "xinqili551-the-university-of-hong-kong"
WANDB_PROJECT = "isaaclab"
WANDB_GROUP = "highstep_b300_0707_single_run_7400_20260718"
WANDB_RUN_ID = "vdqqhj6k"
ABSOLUTE_ORIGIN = 173499
TOTAL_UPDATES = 7400

ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
COUNT_RE = re.compile(r"Student_Distill_Update_Count loss:\s*([0-9.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def verify_authority() -> dict:
    for path, expected in (
        (SPEC, SPEC_SHA),
        (PREREG, PREREG_SHA),
        (WANDB_CONFIG, WANDB_CONFIG_SHA),
        (TEACHER, TEACHER_SHA),
        (PARENT, PARENT_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
    prereg = json.loads(PREREG.read_text())
    dashboard = json.loads(DASHBOARD.read_text())
    if not (
        prereg.get("workflow_id") == WORKFLOW
        and prereg.get("training_budget", {}).get("effective_updates") == TOTAL_UPDATES
        and prereg.get("training_budget", {}).get("continuous_single_process") is True
        and prereg.get("wandb", {}).get("run_id") == WANDB_RUN_ID
        and dashboard.get("workflow_id") == WORKFLOW
        and dashboard.get("spec_sha256") == SPEC_SHA
        and dashboard.get("preregistration_sha256") == PREREG_SHA
        and dashboard.get("state_path") == str(STATE)
    ):
        raise RuntimeError("single-run authority/dashboard binding mismatch")
    return prereg


def parse_clocks() -> tuple[int | None, int]:
    if not LOG.is_file():
        return None, 0
    text = LOG.read_text(encoding="utf-8", errors="replace")
    iterations = [int(item[0]) for item in ITER_RE.findall(text)]
    counts = [int(float(item)) + 1 for item in COUNT_RE.findall(text)]
    return (iterations[-1] if iterations else None), (counts[-1] if counts else 0)


def output_directory() -> str | None:
    candidates = [
        path for path in (ROOT / "logs/rsl_rl").glob(f"*single_run_7400*/*{RUN_NAME}*")
        if path.is_dir()
    ]
    return str(max(candidates, key=lambda item: item.stat().st_mtime)) if candidates else None


def write_runtime(status: str, phase: str, child: subprocess.Popen | None, error: str | None = None) -> None:
    absolute, relative = parse_clocks()
    ready = json.loads(WANDB_READY.read_text()) if WANDB_READY.is_file() else None
    alive = child is not None and child.poll() is None
    payload = {
        "schema_version": 1,
        "workflow_id": WORKFLOW,
        "authority_version": "v1.0",
        "spec_path": str(SPEC),
        "spec_sha256": SPEC_SHA,
        "preregistration_path": str(PREREG),
        "preregistration_sha256": PREREG_SHA,
        "status": status,
        "phase": phase,
        "launcher_pid": os.getpid(),
        "active_pid": child.pid if alive else None,
        "active_pid_alive": alive,
        "lock_path": str(LOCK),
        "heartbeat_path": str(HEARTBEAT),
        "absolute_runner_step": absolute,
        "relative_distill_update": relative,
        "absolute_runner_step_origin": ABSOLUTE_ORIGIN,
        "relative_distill_update_origin": 0,
        "max_effective_updates": TOTAL_UPDATES,
        "output_directory": output_directory(),
        "wandb_run_name": RUN_NAME,
        "wandb_group": WANDB_GROUP,
        "wandb_run_id": WANDB_RUN_ID,
        "wandb_run_url": ready.get("run_url") if ready else None,
        "wandb_online_initialized": ready is not None,
        "automatic_restart": False,
        "evaluation_during_training": False,
        "last_error": error,
        "updated_at": now(),
    }
    atomic_json(STATE, payload)
    atomic_json(HEARTBEAT, payload)
    atomic_json(
        HANDOFF,
        {
            **payload,
            "next_step": (
                "do not interrupt; wait for all 7400 updates"
                if alive else
                "user decision required after unexpected exit"
                if status.startswith("unexpected") or status.startswith("wandb_") else
                "create a separate post-training evaluation specification"
            ),
        },
    )


def child_command() -> list[str]:
    return [
        "/home/lxq/miniconda3/envs/env_isaaclab/bin/python", "-u",
        "scripts/rsl_rl/base/train.py",
        "--task", TASK,
        "--num_envs", "4096",
        "--max_iterations", "7400",
        "--seed", "42",
        "--headless",
        "--logger", "wandb",
        "--log_project_name", WANDB_PROJECT,
        "--resume",
        "--checkpoint", str(TEACHER),
        "--highstep_resume_mode", "refine",
        "--highstep_checkpoint_load_mode", "weights_only",
        "--highstep_schedule_resume_mode", "reset",
        "--highstep_parent_teacher_manifest", str(PARENT),
        "--highstep_absolute_runner_step_origin", str(ABSOLUTE_ORIGIN),
        "--run_name", RUN_NAME,
    ]


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH": str(PREREG),
            "HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256": PREREG_SHA,
            "ROBOT_LAB_WANDB_CONFIG_PATH": str(WANDB_CONFIG),
            "ROBOT_LAB_WANDB_CONFIG_SHA256": WANDB_CONFIG_SHA,
            "ROBOT_LAB_WANDB_PREINITIALIZE": "1",
            "ROBOT_LAB_WANDB_READY_PATH": str(WANDB_READY),
            "ROBOT_LAB_WANDB_RUN_NAME": RUN_NAME,
            "WANDB_MODE": "online",
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_USERNAME": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_RUN_GROUP": WANDB_GROUP,
            "WANDB_RUN_ID": WANDB_RUN_ID,
            "WANDB_RESUME": "never",
            "WANDB_INIT_TIMEOUT": "120",
            "WANDB__SERVICE_WAIT": "300",
            "WANDB_DISABLE_GIT": "true",
            "WANDB_X_DISABLE_STATS": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    environment.pop("WANDB_FORK_FROM", None)
    environment.pop("WANDB_RESUME_FROM", None)
    return environment


def main() -> int:
    FLOW.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as error:
        raise RuntimeError(f"workflow lock already exists: {LOCK}") from error
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.close(descriptor)
    LAUNCHER_PID.write_text(f"{os.getpid()}\n")
    child: subprocess.Popen | None = None
    try:
        verify_authority()
        WANDB_READY.unlink(missing_ok=True)
        write_runtime("starting", "wandb_online_initialization_before_training", None)
        log_stream = LOG.open("ab", buffering=0)
        child = subprocess.Popen(
            child_command(),
            cwd=ROOT,
            env=child_environment(),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        TRAIN_PID.write_text(f"{child.pid}\n")
        write_runtime("running", "training_single_process_7400", child)
        while child.poll() is None:
            write_runtime("running", "training_single_process_7400", child)
            time.sleep(15)
        return_code = int(child.returncode)
        absolute, relative = parse_clocks()
        if return_code == 0 and relative >= TOTAL_UPDATES:
            write_runtime("training_complete", "completed_7400_pending_postrun_evaluation_spec", child)
            return 0
        if not WANDB_READY.is_file() and relative == 0:
            write_runtime(
                "wandb_online_initialization_failed_before_training",
                "stopped_before_first_optimizer_update",
                child,
                error=f"training child exited {return_code}; see {LOG}",
            )
        else:
            write_runtime(
                "unexpected_training_exit_requires_user_decision",
                "safe_evidence_preserved_no_automatic_resume",
                child,
                error=f"training child exited {return_code} at absolute={absolute} relative={relative}",
            )
        return return_code or 1
    finally:
        TRAIN_PID.unlink(missing_ok=True)
        LAUNCHER_PID.unlink(missing_ok=True)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            write_runtime(
                "unexpected_launcher_exit_requires_user_decision",
                "safe_evidence_preserved_no_automatic_resume",
                None,
                error=str(error),
            )
        finally:
            LOCK.unlink(missing_ok=True)
            TRAIN_PID.unlink(missing_ok=True)
            LAUNCHER_PID.unlink(missing_ok=True)
        print(f"FATAL: {error}", file=sys.stderr)
        raise
