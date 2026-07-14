#!/usr/bin/env python3
"""Audit and resume the v1.6.1 workflow after the 2026-07-13 planned shutdown."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
STATE_ROOT = ROOT / "tmp/highstep_0707_exact_new_teacher_20260713"
SHUTDOWN = STATE_ROOT / "planned_shutdown_20260713"
MANIFEST = SHUTDOWN / "planned_shutdown_manifest.json"
MANIFEST_SHA = SHUTDOWN / "planned_shutdown_manifest.sha256"
SERVICE_SOURCE = ROOT / "scripts/systemd/highstep-0707-exact-new-teacher.service"
SERVICE_NAME = "highstep-0707-exact-new-teacher.service"
TRAIN_TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior0707Exact-ArcdogAdjustableLeg-v0"
PARENT_TEACHER_MANIFEST = ROOT / "tmp/highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json"
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
TARGET_UPDATES = 300
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict, *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def load_checkpoint(path: Path) -> tuple[dict, dict, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not {"model_state_dict", "optimizer_state_dict", "iter", "infos"} <= set(payload):
        raise RuntimeError("checkpoint is not a full runner checkpoint")
    state = payload["infos"][ALGORITHM_KEY]
    recovery = state["student_recovery"]
    effective = int(recovery["effective_update_count"])
    if int(state["student_distill_update_count"]) != effective:
        raise RuntimeError("effective-update counters disagree")
    if not state["vae_optimizer_state_dict"].get("state"):
        raise RuntimeError("VAE optimizer state is empty")
    if not payload["optimizer_state_dict"].get("param_groups"):
        raise RuntimeError("runner optimizer structure is absent")
    return payload, recovery, effective


def assert_idle() -> None:
    result = subprocess.run(
        ["pgrep", "-af", "scripts/rsl_rl/base/(train|play)\\.py|highstep_0707_exact_new_teacher_supervisor|highstep_centerline_guard_monitor"],
        text=True, capture_output=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        raise RuntimeError(f"highstep process is still active:\n{result.stdout}")


def audit() -> tuple[dict, Path, int]:
    expected_manifest_sha = MANIFEST_SHA.read_text().split()[0]
    if sha256_file(MANIFEST) != expected_manifest_sha:
        raise RuntimeError("planned-shutdown manifest SHA mismatch")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "planned_user_shutdown_pause":
        raise RuntimeError("planned-shutdown status changed")
    for record in manifest["authority_files"] + manifest["checkpoint_bundle"]:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"authority SHA mismatch: {path}")
    checkpoint = Path(manifest["resume_checkpoint"]["path"])
    payload, recovery, effective = load_checkpoint(checkpoint)
    binding = recovery["binding_manifest"]
    if not (
        binding["teacher_sha256"] == manifest["teacher_binding"]["sha256"]
        and binding["student_teacher_storage_independent"] is True
        and binding["student_ppo_permanently_disabled"] is True
        and binding["actor_body_frozen"] is True
        and binding["critic_frozen"] is True
        and binding["student_privileged_encoder_frozen"] is True
        and effective == manifest["resume_checkpoint"]["effective_updates"]
    ):
        raise RuntimeError("checkpoint binding/freeze contract mismatch")
    core = json.loads(Path(manifest["e100_evidence"]["core9_summary"]).read_text())
    if core["counts"]["valid"] != 9 or not core.get("recovered_from_existing_play_logs"):
        raise RuntimeError("E100 read-only core9 aggregation is not valid")
    assert_idle()
    tests = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q",
         "tests/test_highstep_v152_monitor_contract.py", "tests/test_highstep_0707_exact_new_teacher.py"],
        cwd=ROOT, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        text=True, capture_output=True,
    )
    if tests.returncode != 0:
        raise RuntimeError(f"mapping/workflow regression tests failed:\n{tests.stdout}\n{tests.stderr}")
    print(tests.stdout.strip())
    return manifest, checkpoint, effective


def import_wandb_gate():
    path = ROOT / "tools/highstep_wandb_stage_gate.py"
    spec = importlib.util.spec_from_file_location("shutdown_wandb_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import W&B stage gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resume(manifest: dict, checkpoint: Path, effective: int) -> None:
    if effective >= TARGET_UPDATES:
        raise RuntimeError("resume checkpoint is already at or beyond E300")
    remaining = TARGET_UPDATES - effective
    stage_root = STATE_ROOT / "stages/E300"
    training_result = stage_root / "training_result.json"
    if training_result.exists():
        raise RuntimeError("E300 training_result already exists; use the supervisor instead")
    wandb_manifest = Path(manifest["wandb"]["stage_manifest"])
    wandb_gate = import_wandb_gate()
    environment = os.environ.copy()
    environment.update(wandb_gate.process_environment(wandb_manifest))
    environment.update({
        "HIGHSTEP_0707_EXACT_PREREGISTRATION_PATH": str(Path(manifest["preregistration"]["path"])),
        "HIGHSTEP_0707_EXACT_PREREGISTRATION_SHA256": manifest["preregistration"]["sha256"],
        "WANDB_MODE": "offline", "WANDB_X_DISABLE_STATS": "true", "WANDB_DISABLE_GIT": "true",
    })
    run_name = f"0707_exact_E300_shutdown_resume_{time.strftime('%Y%m%d_%H%M%S')}"
    command = [
        str(PYTHON), "-u", "scripts/rsl_rl/base/train.py", "--task", TRAIN_TASK,
        "--num_envs", "4096", "--max_iterations", str(remaining), "--seed", "42", "--headless",
        "--logger", "wandb", "--log_project_name", "isaaclab", "--resume", "--checkpoint", str(checkpoint),
        "--highstep_resume_mode", "refine", "--highstep_checkpoint_load_mode", "full",
        "--highstep_schedule_resume_mode", "preserve", "--highstep_parent_teacher_manifest", str(PARENT_TEACHER_MANIFEST),
        "--run_name", run_name,
    ]
    launch = stage_root / "planned_shutdown_resume_launch.json"
    atomic_json(launch, {
        "command": command, "source": str(checkpoint), "source_sha256": sha256_file(checkpoint),
        "source_effective_updates": effective, "additional_updates": remaining,
        "effective_target": TARGET_UPDATES, "wandb_stage_manifest": str(wandb_manifest),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, read_only=True)
    log = stage_root / "train_after_planned_shutdown.log"
    with log.open("ab") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"planned-shutdown E300 continuation exited {result.returncode}")
    matches = sorted(
        (ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_0707_exact_new_teacher_Student").glob(f"*_{run_name}"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one continuation run, got {matches}")
    run_dir = matches[0]
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        try:
            _, _, count = load_checkpoint(path)
            if count == TARGET_UPDATES:
                candidates.append(path)
        except Exception:
            pass
    if not candidates:
        raise RuntimeError("continuation did not produce an effective-update-300 checkpoint")
    output = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    sys.path.insert(0, str(ROOT))
    from tools.highstep_v15_imitation_gate import checkpoint_scope
    scope_path = stage_root / "checkpoint_scope.json"
    atomic_json(scope_path, checkpoint_scope(output, TARGET_UPDATES), read_only=True)
    atomic_json(training_result, {
        "checkpoint": str(output), "checkpoint_sha256": sha256_file(output),
        "effective_updates": TARGET_UPDATES, "run_dir": str(run_dir),
        "train_log": str(log), "train_log_sha256": sha256_file(log),
        "checkpoint_scope": str(scope_path), "checkpoint_scope_sha256": sha256_file(scope_path),
        "wandb_stage_manifest": str(wandb_manifest),
        "resumed_from_planned_shutdown": True,
        "resume_source": str(checkpoint), "resume_source_effective_updates": effective,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, read_only=True)
    subprocess.run(["systemctl", "--user", "link", str(SERVICE_SOURCE)], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", SERVICE_NAME], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="continue the remaining E300 updates and restart the supervisor")
    args = parser.parse_args()
    manifest, checkpoint, effective = audit()
    print(f"planned-shutdown audit passed: checkpoint={checkpoint} effective_updates={effective}")
    if args.resume:
        resume(manifest, checkpoint, effective)
        print("E300 continuation completed and supervisor enabled/started")
    else:
        print("audit-only: no train/eval/service was started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
