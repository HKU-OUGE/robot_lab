#!/usr/bin/env python3
"""Stop the superseded zero-scale route at its latest complete atomic checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORKFLOW = ROOT / "tmp/highstep_0707_exact_new_teacher_20260713"
RUN_DIR = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_0707_exact_new_teacher_Student/"
    "2026-07-14_03-35-38_0707_exact_E1200_20260714_033533"
)
E900_CHECKPOINT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_0707_exact_new_teacher_Student/"
    "2026-07-14_03-05-39_0707_exact_E900_20260714_030534/model_894.pt"
)
E900_CHECKPOINT_SHA256 = "130512571674170dab7245cd0dfa6f6af7a2938a3b85046d2f68469e03f7b6a7"
TRAIN_PID = 53591
SUPERVISOR_PID = 37117
SERVICE = "highstep-0707-exact-new-teacher.service"
E1400_GUARD_SERVICE = "highstep-e1400-root-cause-guard.service"
CORRECTION = ROOT / (
    "tmp/highstep_historical_0707_exact_20260714/historical_evidence_correction.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict, *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def process_is_expected(pid: int, needle: bytes) -> bool:
    path = Path("/proc") / str(pid) / "cmdline"
    return path.exists() and needle in path.read_bytes()


def latest_complete_checkpoint() -> tuple[Path, str, dict]:
    checkpoint = E900_CHECKPOINT.resolve(strict=True)
    if sha256_file(checkpoint) != E900_CHECKPOINT_SHA256:
        raise RuntimeError("protected E900 checkpoint SHA changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "optimizer_state_dict", "iter", "infos"}
    missing = sorted(required - set(payload))
    infos = payload.get("infos")
    extra = (
        infos.get("robot_lab_algorithm_checkpoint_state")
        if isinstance(infos, dict)
        else None
    )
    if missing or not isinstance(extra, dict):
        raise RuntimeError(
            f"E1200 checkpoint is not full: missing={missing}, extra_state={type(extra).__name__}"
        )
    recovery = extra.get("student_recovery", {})
    effective = recovery.get("effective_update_count")
    if not isinstance(effective, int) or effective < 900:
        raise RuntimeError(f"latest complete zero-scale update count is invalid: {effective!r}")
    return checkpoint, sha256_file(checkpoint), {
        "iter": payload["iter"],
        "effective_update_count": effective,
        "student_recovery_stage": recovery.get("stage"),
        "optimizer_present": True,
        "algorithm_extra_state_present": True,
    }


def main() -> int:
    heartbeat = WORKFLOW / "zero_scale_ablation_stop_guard_heartbeat.json"
    while process_is_expected(TRAIN_PID, b"scripts/rsl_rl/base/train.py"):
        atomic_json(heartbeat, {
            "status": "waiting_for_running_E1200_process_to_stop_at_E900_recovery_boundary",
            "train_pid": TRAIN_PID,
            "supervisor_pid": SUPERVISOR_PID,
            "new_stage_launch_blocked_by_supervisor_sigstop": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        time.sleep(2.0)

    checkpoint, checkpoint_sha, scope = latest_complete_checkpoint()
    if not process_is_expected(SUPERVISOR_PID, b"highstep_0707_exact_new_teacher_supervisor.py"):
        raise RuntimeError("paused zero-scale supervisor identity changed")

    subprocess.run(
        ["systemctl", "--user", "kill", "--signal=SIGTERM", SERVICE], check=True
    )
    os.kill(SUPERVISOR_PID, signal.SIGCONT)
    subprocess.run(["systemctl", "--user", "stop", SERVICE], check=False, timeout=90)
    subprocess.run(["systemctl", "--user", "disable", SERVICE], check=False, timeout=30)
    subprocess.run(["systemctl", "--user", "stop", E1400_GUARD_SERVICE], check=False, timeout=90)
    subprocess.run(["systemctl", "--user", "disable", E1400_GUARD_SERVICE], check=False, timeout=30)

    correction_sha = sha256_file(CORRECTION)
    stop_manifest = WORKFLOW / "zero_scale_ablation_stop_manifest.json"
    atomic_json(stop_manifest, {
        "schema_version": 1,
        "kind": "highstep_zero_scale_ablation_safe_stop",
        "workflow_id": "highstep_0707_exact_new_teacher_20260713",
        "historical_classification": "zero_scale_ablation",
        "status": "superseded_historical_semantics_correction",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_scope": scope,
        "resume_for_delivery_allowed": False,
        "interrupted_E1200_checkpoint_allowed": False,
        "optimizer_hyperparameter_switch_in_place_allowed": False,
        "historical_correction_manifest": str(CORRECTION),
        "historical_correction_manifest_sha256": correction_sha,
        "next_workflow_id": "highstep_historical_0707_exact_new_teacher_20260714",
        "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, read_only=True)

    state_path = WORKFLOW / "state.json"
    state = json.loads(state_path.read_text())
    state.update({
        "status": "superseded_zero_scale_ablation_archived",
        "phase": "safe_stop_after_E1200_training",
        "active_pid": None,
        "supervisor_pid": None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "effective_updates": scope["effective_update_count"],
        "classification": "zero_scale_ablation",
        "stop_reason": "historical_0707_semantics_corrected_to_1400_phase2_rear1_5",
        "historical_correction_manifest": str(CORRECTION),
        "historical_correction_manifest_sha256": correction_sha,
        "zero_scale_ablation_stop_manifest": str(stop_manifest),
        "zero_scale_ablation_stop_manifest_sha256": sha256_file(stop_manifest),
        "resume_allowed": False,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    atomic_json(state_path, state)
    atomic_json(heartbeat, {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "service_stopped": True,
        "e1400_guard_stopped": True,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
