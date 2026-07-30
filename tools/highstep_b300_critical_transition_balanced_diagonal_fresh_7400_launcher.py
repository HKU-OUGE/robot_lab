#!/usr/bin/env python3
"""One-shot, no-restart launcher for the frozen A+B 7400 workflow."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORKFLOW = "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_20260719"
FLOW = ROOT / f"tmp/{WORKFLOW}"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_b300_critical_transition_balanced_diagonal_fresh_7400_spec_20260719.md"
SPEC_SHA = "4d23ca975cfe5f91d6536b911af369d0807cee027a8e5f54bfacee83fda68575"
PREREG = FLOW / "preregistration.json"
PREREG_SHA = "f82aa5f5246a32f308a4513b09a504f11bd188988933fc983cb48f48d53e3665"
WANDB_CONFIG = FLOW / "wandb_config_attempt2.json"
WANDB_CONFIG_SHA = "6b3fb52a204004ecfdcb9a3825233349a126311a12c61ac9a305c36ea7677eac"
WANDB_READY = FLOW / "wandb_online_ready_attempt2.json"
RUNTIME_REBINDING = FLOW / "runtime_code_rebinding.json"
RUNTIME_REBINDING_SHA = "0ea328baf30207af81aa0ebc63dfd58ba55ea13f03cb77412634ec3b0c83325c"
CROSS_ROLLOUT_REBINDING = FLOW / "cross_rollout_rebinding_attempt2.json"
CROSS_ROLLOUT_REBINDING_SHA = "f6124053822ec30bce0a61152e6fa58d12035ef82187a406d31a294822698989"
STATE = FLOW / "state.json"
HEARTBEAT = FLOW / "heartbeat.json"
HANDOFF = FLOW / "handoff.json"
LOCK = FLOW / "workflow.lock"
LOG = FLOW / "train_attempt2.log"
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/"
    "2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
)
TEACHER_SHA = "d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431"
PARENT = ROOT / "tmp/highstep_be300_0707_distill_20260716/teacher_parent_lineage_v1131.json"
PARENT_SHA = "82dd4934388e5433718394f25d1dcc72c09ae18cddc2a3355a3a75556f4a8f93"
TASK = (
    "RobotLab-Isaac-Velocity-HighstepB300CriticalTransitionBalancedDiagonalFresh7400"
    "StudentNoPrior-ArcdogAdjustableLeg-v0"
)
RUN_NAME = "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_attempt2_from_173499"
WANDB_ENTITY = "xinqili551-the-university-of-hong-kong"
WANDB_PROJECT = "isaaclab"
WANDB_RUN_ID = "af2d0170"
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


def verify_authority() -> None:
    for path, expected in (
        (SPEC, SPEC_SHA),
        (PREREG, PREREG_SHA),
        (WANDB_CONFIG, WANDB_CONFIG_SHA),
        (RUNTIME_REBINDING, RUNTIME_REBINDING_SHA),
        (CROSS_ROLLOUT_REBINDING, CROSS_ROLLOUT_REBINDING_SHA),
        (TEACHER, TEACHER_SHA),
        (PARENT, PARENT_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
    prereg = json.loads(PREREG.read_text())
    dashboard = json.loads(DASHBOARD.read_text())
    cross_rollout_rebinding = json.loads(CROSS_ROLLOUT_REBINDING.read_text())
    for record in cross_rollout_rebinding.get("changed_runtime_files", []):
        if sha256(Path(record["path"])) != record["new_sha256"]:
            raise RuntimeError(f"cross-rollout rebound runtime code changed: {record['path']}")
    sampling = prereg["critical_transition_sampling"]
    diagonal = prereg["phase_diagonal_loss"]
    if not (
        prereg["workflow_id"] == WORKFLOW
        and sampling["minibatch_source_fractions"]
        == {"front_transition": 0.25, "rear_transition": 0.25, "original_distribution": 0.5}
        and sampling["window_radius_policy_steps"] == 10
        and diagonal["front_indices"] == [1, 5, 9]
        and diagonal["rear_indices"] == [2, 6, 10]
        and diagonal["front_scale"] == diagonal["rear_scale"] == 2.0
        and diagonal["gated_total_per_element_weight_ratio"] == 3.0
        and diagonal["rear_term_sign"] == "positive_addition"
        and prereg["training_budget"]["effective_updates"] == TOTAL_UPDATES
        and prereg["wandb"]["run_id"] == "7058f8e5"
        and cross_rollout_rebinding["restart"]["wandb_run_id"]
        == WANDB_RUN_ID
        and dashboard.get("workflow_id") == WORKFLOW
        and dashboard.get("spec_sha256") == SPEC_SHA
        and dashboard.get("preregistration_sha256") == PREREG_SHA
        and dashboard.get("state_path") == str(STATE)
    ):
        raise RuntimeError("A+B 7400 authority/dashboard binding mismatch")


def parse_clocks() -> tuple[int | None, int]:
    if not LOG.is_file():
        return None, 0
    with LOG.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 2 * 1024 * 1024))
        text = stream.read().decode("utf-8", errors="replace")
    iterations = [int(item[0]) for item in ITER_RE.findall(text)]
    counts = [int(float(item)) + 1 for item in COUNT_RE.findall(text)]
    return (iterations[-1] if iterations else None), (counts[-1] if counts else 0)


def output_directory() -> str | None:
    root = ROOT / "logs/rsl_rl"
    candidates = [p for p in root.glob("*critical_transition_balanced*/*") if p.is_dir() and RUN_NAME in p.name]
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def write_runtime(status: str, phase: str, child: subprocess.Popen | None, error: str | None = None) -> None:
    absolute, relative = parse_clocks()
    ready = json.loads(WANDB_READY.read_text()) if WANDB_READY.is_file() else {}
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
        "supervisor_pid": os.getpid(),
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
        "wandb_group": WORKFLOW,
        "wandb_run_id": WANDB_RUN_ID,
        "wandb_run_url": ready.get("run_url"),
        "wandb_online_initialized": bool(ready),
        "automatic_restart": False,
        "evaluation_during_training": False,
        "last_error": error,
        "updated_at": now(),
    }
    atomic_json(STATE, payload)
    atomic_json(HEARTBEAT, payload)
    atomic_json(HANDOFF, {**payload, "next_step": "do not interrupt; wait for 7400 updates" if alive else "user decision required"})


def child_command() -> list[str]:
    return [
        "/home/lxq/miniconda3/envs/env_isaaclab/bin/python", "-u", "scripts/rsl_rl/base/train.py",
        "--task", TASK, "--num_envs", "4096", "--max_iterations", "7400", "--seed", "42",
        "--headless", "--logger", "wandb", "--log_project_name", WANDB_PROJECT,
        "--resume", "--checkpoint", str(TEACHER),
        "--highstep_resume_mode", "refine",
        "--highstep_checkpoint_load_mode", "weights_only",
        "--highstep_schedule_resume_mode", "reset",
        "--highstep_parent_teacher_manifest", str(PARENT),
        "--highstep_absolute_runner_step_origin", str(ABSOLUTE_ORIGIN),
        "--run_name", RUN_NAME,
    ]


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH": str(PREREG),
        "HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256": PREREG_SHA,
        "ROBOT_LAB_WANDB_CONFIG_PATH": str(WANDB_CONFIG),
        "ROBOT_LAB_WANDB_CONFIG_SHA256": WANDB_CONFIG_SHA,
        "ROBOT_LAB_WANDB_PREINITIALIZE": "1",
        "ROBOT_LAB_WANDB_READY_PATH": str(WANDB_READY),
        "ROBOT_LAB_WANDB_RUN_NAME": RUN_NAME,
        "WANDB_MODE": "online", "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_USERNAME": WANDB_ENTITY, "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_RUN_GROUP": WORKFLOW, "WANDB_RUN_ID": WANDB_RUN_ID,
        "WANDB_RESUME": "never", "WANDB_INIT_TIMEOUT": "120",
        "WANDB__SERVICE_WAIT": "300", "WANDB_DISABLE_GIT": "true",
        "WANDB_X_DISABLE_STATS": "true", "PYTHONUNBUFFERED": "1",
    })
    environment.pop("WANDB_FORK_FROM", None)
    environment.pop("WANDB_RESUME_FROM", None)
    return environment


def main() -> int:
    FLOW.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.close(descriptor)
    child = None
    stream = None
    try:
        verify_authority()
        WANDB_READY.unlink(missing_ok=True)
        write_runtime("starting", "wandb_online_initialization_before_training", None)
        stream = LOG.open("ab", buffering=0)
        child = subprocess.Popen(child_command(), cwd=ROOT, env=child_environment(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (FLOW / "launcher.pid").write_text(f"{os.getpid()}\n")
        (FLOW / "train.pid").write_text(f"{child.pid}\n")
        while child.poll() is None:
            write_runtime("running", "training_single_process_7400", child)
            time.sleep(15)
        return_code = int(child.returncode)
        absolute, relative = parse_clocks()
        if return_code == 0 and relative >= TOTAL_UPDATES:
            write_runtime("training_complete_pending_user_manual_play", "completed_7400_pending_user_manual_play", child)
            return 0
        if not WANDB_READY.is_file() and relative == 0:
            write_runtime("wandb_online_initialization_failed_before_training", "stopped_before_first_optimizer_update", child, f"child exited {return_code}; see {LOG}")
        else:
            write_runtime("unexpected_training_exit_requires_user_decision", "safe_evidence_preserved_no_automatic_resume", child, f"child exited {return_code} at absolute={absolute} relative={relative}")
        return return_code or 1
    finally:
        if stream is not None:
            stream.close()
        (FLOW / "train.pid").unlink(missing_ok=True)
        (FLOW / "launcher.pid").unlink(missing_ok=True)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
