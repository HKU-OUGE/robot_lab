#!/usr/bin/env python3
"""Run the frozen 15-run Isaac gate for the phase-only policy2 adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
FLOW = ROOT / "tmp/highstep_fixed_motion_phase_only_deployment_20260717"
WORKFLOW = "highstep_fixed_motion_phase_only_deployment_20260717"
PREREG = FLOW / "preregistration.json"
EXPECTED_PREREG_SHA = "3a82220b6d4d2bb4ceb3dcafa530478fdb69339f900973473270af8a8eee1b35"
RAW_FLOW = ROOT / "tmp/highstep_fixed_motion_raw_action_20260717"
TRAINING = RAW_FLOW / "training/dagger_round2_attempt1/training_manifest.json"
WANDB = RAW_FLOW / "training/dagger_round2_attempt1/wandb_verification.json"
TEACHER = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
TRACE = ROOT / "logs/play_joint_records/20260717_072307_teacher_b300_box_hard_seed11_pid877984/joint_trace.csv"
SOURCE_MANIFEST = TRACE.parent / "manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def snapshot(phase: str, status: str, *, active_pid=None, completed=0, counts=None, error=None) -> None:
    payload = {
        "schema_version": 1,
        "workflow_id": WORKFLOW,
        "phase": phase,
        "status": status,
        "supervisor_pid": os.getpid(),
        "active_pid": active_pid,
        "completed_runs": completed,
        "required_runs": 15,
        "counts": counts or {},
        "error": error,
        "preregistration_path": str(PREREG),
        "preregistration_sha256": sha(PREREG),
        "automatic_real_robot_deployment_authorized": False,
        "updated_at_epoch": time.time(),
    }
    write(FLOW / "state.json", payload)
    write(FLOW / "heartbeat.json", payload)


def command(output: Path) -> list[str]:
    return [
        "/home/lxq/miniconda3/envs/env_isaaclab/bin/python", "-u", str(ROOT / "scripts/rsl_rl/base/play.py"),
        "--task", "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123-ArcdogAdjustableLeg-v0",
        "--num_envs", "1", "--seed", "11", "--checkpoint", str(TEACHER),
        "--play_terrain_level", "9", "--play_terrain_type", "box_hard",
        "--reset_after_play_terrain_selection", "--front_step_eval_reset", "--front_step_eval_side", "x-",
        "--front_step_eval_edge_gap", "0.55", "--front_step_eval_lateral_offset", "0",
        "--front_step_eval_yaw_offset_deg", "0", "--eval_action_delay_steps", "0",
        "--recorded_command_trace", str(TRACE), "--recorded_command_trace_sha256", sha(TRACE),
        "--recorded_command_episode_id", "4", "--recorded_command_source_manifest", str(SOURCE_MANIFEST),
        "--recorded_command_source_manifest_sha256", sha(SOURCE_MANIFEST),
        "--fixed_motion_raw_action_mode", "student",
        "--fixed_motion_raw_preregistration", str(RAW_FLOW / "preregistration.json"),
        "--fixed_motion_raw_preregistration_sha256", sha(RAW_FLOW / "preregistration.json"),
        "--fixed_motion_raw_output", str(output),
        "--fixed_motion_raw_training_manifest", str(TRAINING),
        "--fixed_motion_raw_training_manifest_sha256", sha(TRAINING),
        "--fixed_motion_raw_wandb_verification", str(WANDB),
        "--fixed_motion_raw_wandb_verification_sha256", sha(WANDB),
        "--fixed_motion_phase_only_deployment_preregistration", str(PREREG),
        "--fixed_motion_phase_only_deployment_preregistration_sha256", EXPECTED_PREREG_SHA,
        "--play_max_steps", "138", "--highstep_gap_camera", "static_side_top",
        "--skip_policy_export", "--headless",
    ]


def summarize() -> tuple[list[dict], dict[str, int]]:
    rows = []
    for index in range(1, 16):
        result = FLOW / f"isaac_gate/run{index:02d}/behavior_result.json"
        if result.exists():
            row = json.loads(result.read_text())
            row["run"] = index
            row["result_path"] = str(result)
            row["result_sha256"] = sha(result)
            rows.append(row)
    counts = {
        "valid": sum(bool(row.get("valid")) for row in rows),
        "full_climb": sum(bool(row.get("full_climb")) for row in rows),
        "rear_hold": sum(bool(row.get("rear_hold")) for row in rows),
        "safe": sum(row.get("illegal_outputs") == 0 and not row.get("terminated") for row in rows),
    }
    return rows, counts


def main() -> int:
    if sha(PREREG) != EXPECTED_PREREG_SHA:
        raise RuntimeError("phase-only preregistration SHA changed")
    FLOW.mkdir(parents=True, exist_ok=True)
    for index in range(1, 16):
        output = FLOW / f"isaac_gate/run{index:02d}"
        result = output / "behavior_result.json"
        if result.exists():
            continue
        rows, counts = summarize()
        log = FLOW / f"isaac_gate/run{index:02d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            child = subprocess.Popen(command(output), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while child.poll() is None:
                snapshot(f"isaac_gate_run{index:02d}", "in_progress", active_pid=child.pid, completed=len(rows), counts=counts)
                time.sleep(3)
        if child.returncode != 0 or not result.exists():
            snapshot(f"isaac_gate_run{index:02d}", "infrastructure_failure", completed=len(rows), counts=counts, error=str(log))
            return 2

    rows, counts = summarize()
    passed = counts == {"valid": 15, "full_climb": 15, "rear_hold": 15, "safe": 15} or (
        counts["valid"] == 15 and counts["full_climb"] >= 12 and counts["rear_hold"] >= 12 and counts["safe"] == 15
    )
    aggregate = {
        "schema_version": 1,
        "kind": "highstep_fixed_motion_phase_only_isaac_gate",
        "workflow_id": WORKFLOW,
        "status": "passed" if passed else "failed_behavior_gate",
        "passed": passed,
        "counts": counts,
        "rows": rows,
        "checkpoint": json.loads(PREREG.read_text())["checkpoint"],
        "checkpoint_sha256": json.loads(PREREG.read_text())["checkpoint_sha256"],
        "preregistration_path": str(PREREG),
        "preregistration_sha256": EXPECTED_PREREG_SHA,
        "mujoco_allowed": passed,
        "automatic_real_robot_deployment_authorized": False,
    }
    write(FLOW / "isaac_gate/aggregate.json", aggregate)
    snapshot("isaac_gate_complete", "isaac_passed_pending_mujoco" if passed else "isaac_failed_requires_contract_retraining", completed=15, counts=counts)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
