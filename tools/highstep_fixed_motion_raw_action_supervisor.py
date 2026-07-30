#!/usr/bin/env python3
"""Independent supervisor for fixed-condition raw-action BC behavior gates."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
FLOW = ROOT / "tmp/highstep_fixed_motion_raw_action_20260717"
WORKFLOW = "highstep_fixed_motion_raw_action_20260717"
PREREG = FLOW / "preregistration.json"
CANONICAL_TRAIN = FLOW / "training/canonical_bc_attempt1/training_manifest.json"
CANONICAL_WANDB = FLOW / "training/canonical_bc_attempt1/wandb_verification.json"
TEACHER = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
TRACE = ROOT / "logs/play_joint_records/20260717_072307_teacher_b300_box_hard_seed11_pid877984/joint_trace.csv"
SOURCE_MANIFEST = TRACE.parent / "manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def snapshot(*, phase: str, status: str, active_pid=None, completed=0, counts=None, error=None) -> dict:
    value = {"schema_version": 1, "workflow_id": WORKFLOW, "phase": phase, "status": status,
             "supervisor_pid": os.getpid(), "active_pid": active_pid, "completed_runs": completed,
             "required_runs": 15, "counts": counts or {}, "error": error,
             "preregistration_path": str(PREREG), "preregistration_sha256": sha(PREREG),
             "updated_at_epoch": time.time()}
    write(FLOW / "state.json", value); write(FLOW / "heartbeat.json", value)
    return value


def command(run: int, output: Path, training: Path, wandb: Path) -> list[str]:
    return [str(Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")), "-u",
        str(ROOT / "scripts/rsl_rl/base/play.py"),
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
        "--fixed_motion_raw_preregistration", str(PREREG),
        "--fixed_motion_raw_preregistration_sha256", sha(PREREG),
        "--fixed_motion_raw_output", str(output),
        "--fixed_motion_raw_training_manifest", str(training),
        "--fixed_motion_raw_training_manifest_sha256", sha(training),
        "--fixed_motion_raw_wandb_verification", str(wandb),
        "--fixed_motion_raw_wandb_verification_sha256", sha(wandb),
        "--play_max_steps", "138", "--highstep_gap_camera", "static_side_top",
        "--skip_policy_export", "--headless"]


def run_child(args: list[str], log: Path, phase: str, completed: int = 0) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        child = subprocess.Popen(args, cwd=ROOT, env={**os.environ, "PYTHONPATH": f"{ROOT / 'source/robot_lab'}:{ROOT / 'tools'}"},
                                 stdout=stream, stderr=subprocess.STDOUT)
        while child.poll() is None:
            snapshot(phase=phase, status="in_progress", active_pid=child.pid, completed=completed)
            time.sleep(3)
    if child.returncode != 0:
        raise RuntimeError(f"{phase} failed with exit={child.returncode}; log={log}")


def advance_dagger(round_index: int, prior_train: Path, prior_wandb: Path) -> None:
    collection = FLOW / f"dagger/round{round_index}_collection_attempt1"
    collection_manifest = collection / "dagger_manifest.json"
    if not collection_manifest.exists():
        dagger_args = command(1, collection, prior_train, prior_wandb)
        mode_index = dagger_args.index("--fixed_motion_raw_action_mode") + 1
        dagger_args[mode_index] = "dagger"
        dagger_args.extend(["--fixed_motion_raw_dagger_round", str(round_index)])
        run_child(dagger_args, FLOW / f"dagger_round{round_index}_collection.log",
                  f"dagger_round{round_index}_collection")
    aggregate = FLOW / f"dagger/round{round_index}_aggregate"
    aggregate_manifest = aggregate / "aggregate_manifest.json"
    if not aggregate_manifest.exists():
        merge = [sys.executable, str(ROOT / "tools/highstep_fixed_motion_raw_action_merge.py"),
                 "--canonical-manifest", str(FLOW / "passthrough_smoke_attempt1/smoke_manifest.json")]
        for index in range(1, round_index + 1):
            merge += ["--dagger-manifest", str(FLOW / f"dagger/round{index}_collection_attempt1/dagger_manifest.json")]
        merge += ["--output-dir", str(aggregate)]
        run_child(merge, FLOW / f"dagger_round{round_index}_merge.log", f"dagger_round{round_index}_merge")
    training = FLOW / f"training/dagger_round{round_index}_attempt1"
    training_manifest = training / "training_manifest.json"
    verification = training / "wandb_verification.json"
    if not training_manifest.exists() or not verification.exists():
        train = [sys.executable, "-u", str(ROOT / "tools/highstep_fixed_motion_raw_action_train.py"),
                 "--preregistration", str(PREREG), "--preregistration-sha256", sha(PREREG),
                 "--smoke-manifest", str(aggregate_manifest), "--smoke-manifest-sha256", sha(aggregate_manifest),
                 "--stage", f"dagger_round{round_index}", "--output-dir", str(training)]
        run_child(train, FLOW / f"dagger_round{round_index}_training.log", f"dagger_round{round_index}_training")
    if json.loads(verification.read_text()).get("remote_verified") is not True:
        raise RuntimeError(f"DAgger round {round_index} W&B is not remotely verified")


def generate_videos(stage: str, train_path: Path, wandb_path: Path) -> Path:
    root = FLOW / f"student_visual_review/{stage}"
    clips: list[Path] = []
    for index in range(1, 16):
        run = root / f"run{index:02d}"
        delivery = run / f"student_{stage}_run{index:02d}.mp4"
        if not delivery.exists():
            args = command(index, run / "evidence", train_path, wandb_path)
            args += ["--video", "--video_length", "138", "--video_output_dir", str(run / "video"),
                     "--record_joint_data", "--joint_record_output", str(run / "trajectory"),
                     "--joint_record_label", f"student_{stage}_run{index:02d}"]
            run_child(args, run / "run.log", f"{stage}_video_run{index:02d}", index - 1)
            videos = sorted((run / "video").rglob("*.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"expected one video for {stage} run {index}, found {len(videos)}")
            delivery.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(videos[0], delivery)
        clips.append(delivery)
    concat = root / "concat.txt"
    concat.write_text("".join(f"file '{clip}'\n" for clip in clips))
    montage = root / f"student_{stage}_batch15_montage.mp4"
    if not montage.exists():
        run_child(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                   "-c", "copy", str(montage)], root / "montage.log", f"{stage}_video_montage", 15)
    return montage


def main() -> int:
    lock_stream = (FLOW / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("raw-action supervisor lock is already held") from error
    lock_stream.seek(0); lock_stream.truncate(); lock_stream.write(str(os.getpid()) + "\n"); lock_stream.flush()
    stages = [("canonical_bc", CANONICAL_TRAIN, CANONICAL_WANDB)]
    for round_index in range(1, 4):
        directory = FLOW / f"training/dagger_round{round_index}_attempt1"
        if (directory / "training_manifest.json").exists():
            stages.append((f"dagger_round{round_index}", directory / "training_manifest.json", directory / "wandb_verification.json"))
    stage, train_path, wandb_path = stages[-1]
    prereg = json.loads(PREREG.read_text()); training = json.loads(train_path.read_text()); wb = json.loads(wandb_path.read_text())
    if prereg.get("workflow_id") != WORKFLOW or training.get("status") != "completed" or wb.get("remote_verified") is not True:
        raise RuntimeError("raw-action authority or W&B binding is incomplete")
    base = FLOW / f"evaluations/{stage}"; base.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, 16):
        output = base / f"run{index:02d}_attempt1"; result = output / "behavior_result.json"
        if not result.exists():
            log = base / f"run{index:02d}_attempt1.log"
            with log.open("w") as stream:
                child = subprocess.Popen(command(index, output, train_path, wandb_path), cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "source/robot_lab")}, stdout=stream, stderr=subprocess.STDOUT)
                snapshot(phase=f"{stage}_behavior_run{index:02d}", status="evaluating", active_pid=child.pid,
                         completed=index - 1)
                while child.poll() is None:
                    snapshot(phase=f"{stage}_behavior_run{index:02d}", status="evaluating", active_pid=child.pid,
                             completed=index - 1); time.sleep(3)
                if child.returncode != 0 or not result.exists():
                    snapshot(phase=f"{stage}_behavior_run{index:02d}", status="infrastructure_failure",
                             completed=index - 1, error=f"eval exit={child.returncode}; log={log}")
                    return 2
        rows.append(json.loads(result.read_text()))
    counts = {key: sum(bool(row.get(key)) for row in rows) for key in ("valid", "full_climb", "rear_hold")}
    counts["safe"] = sum(int(row.get("illegal_outputs", 0)) == 0 for row in rows)
    passed = counts["valid"] == 15 and counts["full_climb"] >= 12 and counts["rear_hold"] >= 12 and counts["safe"] == 15
    aggregate = {"schema_version": 1, "kind": "highstep_fixed_motion_raw_action_behavior_gate",
                 "workflow_id": WORKFLOW, "status": "passed" if passed else "failed_behavior_gate",
                 "passed": passed, "counts": counts, "required": {"valid": 15, "full_climb": 12, "rear_hold": 12, "safe": 15},
                 "stage": stage, "rows": rows, "training_manifest_path": str(train_path), "training_manifest_sha256": sha(train_path)}
    write(base / "aggregate.json", aggregate)
    completed_round = int(stage.removeprefix("dagger_round")) if stage.startswith("dagger_round") else 0
    montage = generate_videos(stage, train_path, wandb_path) if passed else None
    status = "student_fixed_motion_candidate_pending_user_visual_review" if passed else (
        "stopped_after_dagger_round3_below_gate" if completed_round >= 3 else f"dagger_round{completed_round + 1}_required"
    )
    state = snapshot(phase=f"{stage}_behavior_gate_complete", status=status, completed=15, counts=counts)
    write(FLOW / "handoff.json", {**state, "aggregate_path": str(base / "aggregate.json"),
                                  "aggregate_sha256": sha(base / "aggregate.json"),
                                  "student_video_montage": str(montage) if montage else None,
                                  "next_step": "wait_for_user_visual_review" if passed else (
                                      "stop_and_report" if completed_round >= 3 else f"collect_dagger_round{completed_round + 1}"
                                  )})
    if not passed and completed_round < 3:
        advance_dagger(completed_round + 1, train_path, wandb_path)
        os.execv(sys.executable, [sys.executable, "-u", str(Path(__file__).resolve())])
    return 0


if __name__ == "__main__": raise SystemExit(main())
