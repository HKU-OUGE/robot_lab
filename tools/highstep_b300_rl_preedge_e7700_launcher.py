#!/usr/bin/env python3
"""No-restart launcher for the exact E5700 -> E7700 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path("/home/lxq/Softwares/robot_lab")
WORKFLOW = "highstep_b300_rl_preedge_continuation_e7700_20260719"
FLOW = ROOT / "tmp" / WORKFLOW
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_b300_rl_preedge_continuation_e7700_spec_20260719.md"
SPEC_SHA = "b1a854ebc0c8ccf674bc25d869b6e088532cf8911e64a1404f0b365fdda59f08"
PREREG = FLOW / "preregistration.json"
PREREG_SHA = "2fe5a5a8b1f3629b6fb7daca023ac0ba7c468f771e758cc2ef94448c1df3e509"
REBIND = FLOW / "runtime_code_rebinding_attempt9.json"
REBIND_SHA = "f439e4207ce5c04f1432d5043b17edf228e53bd766ad6f7074fc8bdaf8345e6a"
WANDB_CONFIG = FLOW / "wandb_config_attempt9.json"
WANDB_CONFIG_SHA = "f249a58192cdf65aa21bcea9811f759386b16e0502869eafa12cee7b42fe1ee6"
WANDB_READY = FLOW / "wandb_online_ready_attempt9.json"
ATTEMPT8_DISPOSITION = FLOW / "failure_recovery/20260720_020843/attempt8_disposition.json"
ATTEMPT8_DISPOSITION_SHA = "0388c31b89977f5948cbb694d8eb36bb8cbb1ee1b0b44f158604b596c9d2d3bb"
PREDICATE_REPAIR = FLOW / "rl_preedge_predicate_implementation_repair.json"
PREDICATE_REPAIR_SHA = "d598277c9517b23a3e7d6ee8a152d256df44a3af80d5a3a990b1c734feb3d67a"
VX_AMENDMENT = FLOW / "user_approved_vx_threshold_amendment.json"
VX_AMENDMENT_SHA = "9a89aa4b32ac325b7a37c59982a80ce18ee65f24cc2b1ad28df8e43741dde115"
CHECKPOINT = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_Student/2026-07-19_06-43-15_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_attempt2_from_173499/model_179198.pt"
CHECKPOINT_SHA = "31fe19c7d1ba9872c0b714d594fe6e66c549a52e88fe856ed62206680d37c498"
PARENT = ROOT / "tmp/highstep_be300_0707_distill_20260716/teacher_parent_lineage_v1131.json"
PARENT_SHA = "82dd4934388e5433718394f25d1dcc72c09ae18cddc2a3355a3a75556f4a8f93"
TASK = "RobotLab-Isaac-Velocity-HighstepB300RLPreEdgeContinuationE7700StudentNoPrior-ArcdogAdjustableLeg-v0"
RUN_NAME = "highstep_b300_rl_preedge_continuation_e7700_attempt9_repaired"
WANDB_GROUP = WORKFLOW
WANDB_RUN_ID = "126d9b3f"
STATE, HEARTBEAT, HANDOFF, LOCK = (FLOW / name for name in ("state.json", "heartbeat.json", "handoff.json", "workflow.lock"))
LOG = FLOW / "train_attempt9.log"
PROGRESS = FLOW / "effective_progress_attempt9.json"
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
COUNT_RE = re.compile(r"Student_Distill_Update_Count loss:\s*([0-9.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def clocks() -> tuple[int | None, int]:
    if PROGRESS.is_file():
        progress = json.loads(PROGRESS.read_text())
        return int(progress["absolute_runner_step"]), int(progress["effective_updates"])
    if not LOG.is_file():
        return None, 5700
    text = LOG.read_text(errors="replace")[-2_000_000:]
    iterations = [int(match[0]) for match in ITER_RE.findall(text)]
    counts = [int(float(value)) for value in COUNT_RE.findall(text)]
    return (iterations[-1] if iterations else None), (counts[-1] if counts else 5700)


def output_dir() -> str | None:
    root = ROOT / "logs/rsl_rl"
    candidates = [path for path in root.glob("*rl_preedge_continuation_e7700*/*") if path.is_dir() and RUN_NAME in path.name]
    return str(max(candidates, key=lambda path: path.stat().st_mtime)) if candidates else None


def runtime(status: str, phase: str, child: subprocess.Popen | None, error: str | None = None) -> None:
    absolute, effective = clocks()
    alive = child is not None and child.poll() is None
    ready = json.loads(WANDB_READY.read_text()) if WANDB_READY.is_file() else {}
    payload = {
        "schema_version": 1, "workflow_id": WORKFLOW, "authority_version": "v1.0",
        "spec_path": str(SPEC), "spec_sha256": SPEC_SHA,
        "preregistration_path": str(PREREG), "preregistration_sha256": PREREG_SHA,
        "status": status, "phase": phase, "supervisor_pid": os.getpid(),
        "active_pid": child.pid if alive else None, "active_pid_alive": alive,
        "lock_path": str(LOCK), "heartbeat_path": str(HEARTBEAT),
        "source_checkpoint": str(CHECKPOINT), "source_checkpoint_sha256": CHECKPOINT_SHA,
        "absolute_runner_step": absolute, "effective_updates": effective,
        "start_effective_updates": 5700, "target_effective_updates": 7700,
        "output_directory": output_dir(), "wandb_run_name": RUN_NAME,
        "wandb_group": WANDB_GROUP, "wandb_run_id": WANDB_RUN_ID,
        "wandb_run_url": ready.get("run_url"), "wandb_online_initialized": bool(ready),
        "automatic_restart": False, "automatic_evaluation": False,
        "last_error": error, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(STATE, payload); atomic_json(HEARTBEAT, payload)
    atomic_json(HANDOFF, {**payload, "next_step": "wait for E7700; then user manual full-push play x6"})


def verify() -> None:
    for path, expected in ((SPEC, SPEC_SHA), (PREREG, PREREG_SHA), (REBIND, REBIND_SHA), (WANDB_CONFIG, WANDB_CONFIG_SHA), (ATTEMPT8_DISPOSITION, ATTEMPT8_DISPOSITION_SHA), (PREDICATE_REPAIR, PREDICATE_REPAIR_SHA), (VX_AMENDMENT, VX_AMENDMENT_SHA), (CHECKPOINT, CHECKPOINT_SHA), (PARENT, PARENT_SHA)):
        if sha256(path) != expected:
            raise RuntimeError(f"SHA mismatch: {path}")
    rebound = json.loads(REBIND.read_text())["files"]
    rebound_paths = {
        "observations.py": ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/observations.py",
        "highstep_schedule.py": ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        "highstep_env_cfg.py": ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
        "rsl_rl_ppo_cfg.py": ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
        "vae_ppo.py": ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
        "task_registration.py": ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__init__.py",
        "train.py": ROOT / "scripts/rsl_rl/base/train.py",
        "effective_update_driver.py": ROOT / "scripts/rsl_rl/base/highstep_effective_update_driver.py",
        "contract_check.py": ROOT / "tools/highstep_b300_rl_preedge_contract_check.py",
        "resume_audit.py": ROOT / "tools/highstep_b300_e5700_resume_audit.py",
    }
    for name, path in rebound_paths.items():
        if sha256(path) != rebound[name]:
            raise RuntimeError(f"runtime code changed after rebinding: {path}")
    subprocess.run(["/home/lxq/miniconda3/envs/env_isaaclab/bin/python", "tools/highstep_b300_e5700_resume_audit.py"], cwd=ROOT, check=True)
    subprocess.run(["/home/lxq/miniconda3/envs/env_isaaclab/bin/python", "tools/highstep_b300_rl_preedge_contract_check.py"], cwd=ROOT, check=True)


def command() -> list[str]:
    return [
        "/home/lxq/miniconda3/envs/env_isaaclab/bin/python", "-u", "scripts/rsl_rl/base/train.py",
        "--task", TASK, "--num_envs", "4096", "--max_iterations", "2000", "--seed", "42",
        "--headless", "--logger", "wandb", "--log_project_name", "isaaclab",
        "--resume", "--checkpoint", str(CHECKPOINT), "--highstep_resume_mode", "refine",
        "--highstep_checkpoint_load_mode", "full", "--highstep_schedule_resume_mode", "preserve",
        "--highstep_parent_teacher_manifest", str(PARENT), "--run_name", RUN_NAME,
    ]


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH": str(PREREG),
        "HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256": PREREG_SHA,
        "ROBOT_LAB_WANDB_CONFIG_PATH": str(WANDB_CONFIG),
        "ROBOT_LAB_WANDB_CONFIG_SHA256": WANDB_CONFIG_SHA,
        "ROBOT_LAB_WANDB_PREINITIALIZE": "1", "ROBOT_LAB_WANDB_READY_PATH": str(WANDB_READY),
        "ROBOT_LAB_WANDB_RUN_NAME": RUN_NAME, "WANDB_MODE": "online",
        "HIGHSTEP_EFFECTIVE_PROGRESS_PATH": str(PROGRESS),
        "WANDB_ENTITY": "xinqili551-the-university-of-hong-kong", "WANDB_PROJECT": "isaaclab",
        "WANDB_RUN_GROUP": WANDB_GROUP, "WANDB_RUN_ID": WANDB_RUN_ID, "WANDB_RESUME": "never",
        "WANDB_INIT_TIMEOUT": "120", "WANDB__SERVICE_WAIT": "300", "WANDB_DISABLE_GIT": "true",
        "WANDB_X_DISABLE_STATS": "true", "PYTHONUNBUFFERED": "1",
    })
    env.pop("WANDB_FORK_FROM", None); env.pop("WANDB_RESUME_FROM", None)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archived one-shot launcher for the completed E5700 -> E7700 continuation."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    FLOW.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(descriptor, f"{os.getpid()}\n".encode()); os.close(descriptor)
    child = None; stream = None
    try:
        verify(); WANDB_READY.unlink(missing_ok=True); PROGRESS.unlink(missing_ok=True)
        runtime("starting", "fail_closed_restore_verified_wandb_initializing", None)
        stream = LOG.open("ab", buffering=0)
        child = subprocess.Popen(command(), cwd=ROOT, env=environment(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (FLOW / "supervisor.pid").write_text(f"{os.getpid()}\n"); (FLOW / "train.pid").write_text(f"{child.pid}\n")
        while child.poll() is None:
            runtime("running", "training_E5700_to_E7700", child); time.sleep(15)
        _, effective = clocks()
        if child.returncode == 0 and effective >= 7700:
            runtime("training_complete_pending_user_manual_play_6", "completed_E7700", child); return 0
        runtime("unexpected_training_exit_requires_user_decision", "safe_evidence_preserved", child, f"exit={child.returncode}, effective={effective}")
        return int(child.returncode or 1)
    finally:
        if stream is not None: stream.close()
        (FLOW / "train.pid").unlink(missing_ok=True); (FLOW / "supervisor.pid").unlink(missing_ok=True); LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
