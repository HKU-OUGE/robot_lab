#!/usr/bin/env python3
"""Run, aggregate, and freeze the 0707 deployed Student current-rule core9 baseline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "tmp/highstep_student_recovery_v15_20260713"
OUTPUT = WORKFLOW / "0707_current_rule_core9_v2"
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
CHECKPOINT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
    "student_no_prior_Student/2026-07-05_00-13-46/model_158797.pt"
)
CHECKPOINT_SHA = "7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284"
TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0"
SCENARIOS = (
    ("nominal", 0.0, 0.0),
    ("left_offset", 0.12, 4.0),
    ("right_offset", -0.12, -4.0),
)
RUNS = tuple(f"seed{seed}_{name}" for seed in (11, 22, 33) for name, _, _ in SCENARIOS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict, *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def update_heartbeat(status: str, phase: str, **extra: object) -> None:
    atomic_json(
        OUTPUT / "heartbeat.json",
        {
            "schema_version": 1,
            "status": status,
            "phase": phase,
            "pid": os.getpid(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **extra,
        },
    )


def no_severe(outcome: dict) -> bool:
    return bool(
        float(outcome.get("critical_rear_min_abs_y_q05", -1)) >= 0.04
        and float(outcome.get("critical_rear_width_q05", -1)) >= 0.18
        and float(outcome.get("critical_center_violation_rate", 1)) <= 0.10
        and float(outcome.get("critical_width_violation_rate", 1)) <= 0.10
    )


def main() -> int:
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("0707 deployed Student checkpoint SHA256 mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stop = False

    def receive_stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, receive_stop)
    signal.signal(signal.SIGINT, receive_stop)
    for run_id in RUNS:
        result_path = OUTPUT / "results" / f"{run_id}.json"
        if result_path.exists():
            continue
        if stop:
            update_heartbeat("stopped", "safe_between_runs", next_run=run_id)
            return 0
        run_log = OUTPUT / "logs" / f"{run_id}.log"
        run_log.parent.mkdir(parents=True, exist_ok=True)
        seed_text, scenario = run_id.split("_", 1)
        seed = int(seed_text.removeprefix("seed"))
        scenario_contract = {name: (lateral, yaw) for name, lateral, yaw in SCENARIOS}
        lateral, yaw = scenario_contract[scenario]
        update_heartbeat("running", f"evaluating_{run_id}", completed=sum((OUTPUT / "results" / f"{item}.json").exists() for item in RUNS), total=9)
        command = [
            str(PYTHON), "-u", str(PLAY), "--task", TASK, "--num_envs", "1",
            "--headless", "--seed", str(seed), "--checkpoint", str(CHECKPOINT),
            "--play_terrain_type", "box", "--play_terrain_level", "9",
            "--fixed_velocity_command", "0.45", "0.0", "0.0",
            "--reset_after_play_terrain_selection", "--front_step_eval_reset",
            "--front_step_eval_side", "x-", "--front_step_eval_edge_gap", "0.55",
            "--front_step_eval_lateral_offset", str(lateral),
            "--front_step_eval_yaw_offset_deg", str(yaw),
            "--print_rear_width_metrics", "--skip_policy_export",
            "--rear_width_metric_interval", "60", "--play_max_steps", "600",
            "--eval_action_delay_steps", "0", "--allow_legacy_highstep_schedule_fallback",
        ]
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libstdc++.so.6"
        with run_log.open("ab") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                if stop:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                    update_heartbeat("stopped", f"interrupted_{run_id}")
                    return 0
                update_heartbeat("running", f"evaluating_{run_id}", active_pid=process.pid, completed=sum((OUTPUT / "results" / f"{item}.json").exists() for item in RUNS), total=9)
                time.sleep(10)
        payloads = []
        for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[HIGHSTEP_EVAL_JSON]" in line:
                payloads.append(json.loads(line.split("[HIGHSTEP_EVAL_JSON]", 1)[1].strip()))
        if process.returncode != 0 or len(payloads) != 1:
            update_heartbeat("failed", f"failed_{run_id}", return_code=process.returncode, log=str(run_log))
            raise RuntimeError(f"0707 core9 run failed: {run_id}")
        outcome = payloads[0]
        if not (
            outcome.get("schema_version") == 7
            and outcome.get("checkpoint_sha256") == CHECKPOINT_SHA
            and outcome.get("reset_valid") is True
            and outcome.get("schedule_valid") is True
            and outcome.get("schedule_runtime_match") is True
            and outcome.get("schedule_clock_runtime_match") is True
            and outcome.get("eval_action_delay_steps_runtime") == 0
            and outcome.get("keep_play_randomization") is False
        ):
            update_heartbeat("failed", f"invalid_contract_{run_id}", log=str(run_log))
            raise RuntimeError(f"0707 core9 runtime contract failed: {run_id}")
        atomic_json(
            result_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "seed": seed,
                "scenario": scenario,
                "lateral_offset_m": lateral,
                "yaw_offset_deg": yaw,
                "eval": outcome,
                "log": str(run_log),
                "log_sha256": sha256_file(run_log),
            },
            read_only=True,
        )

    rows = []
    for run_id in RUNS:
        path = OUTPUT / "results" / f"{run_id}.json"
        payload = json.loads(path.read_text())
        outcome = payload["eval"]
        valid = bool(
            outcome.get("checkpoint_sha256") == CHECKPOINT_SHA
            and outcome.get("schedule_valid") is True
            and outcome.get("schedule_clock_runtime_match") is True
            and outcome.get("reset_valid") is True
            and outcome.get("eval_action_delay_steps_runtime") == 0
        )
        rows.append(
            {
                "run_id": run_id,
                "valid": valid,
                "full_climb": outcome.get("full_climb_success") is True,
                "rear_hold": outcome.get("rear_on_platform_hold_success") is True,
                "front_top_support": outcome.get("front_top_support_reached") is True,
                "no_severe_inward": no_severe(outcome),
                "first_rear_top_step": outcome.get("first_rear_top_step"),
                "second_rear_top_step": outcome.get("second_rear_top_step"),
                "rear_hold_steps": outcome.get("rear_on_platform_hold_samples"),
                "result": str(path),
                "result_sha256": sha256_file(path),
            }
        )
    keys = ("valid", "full_climb", "rear_hold", "front_top_support", "no_severe_inward")
    counts = {key: sum(row[key] is True for row in rows) for key in keys}
    manifest = {
        "schema_version": 1,
        "kind": "highstep_v15_0707_deployed_current_rule_core9_baseline",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "training_allowed": False,
        "evaluation_semantics": "current v1.5 play.py/task rules with explicit verified legacy schedule reconstruction; frozen 0707 saved-config shared-state traces remain the independent imitation authority",
        "matrix": {"seeds": [11, 22, 33], "scenarios": {name: [lateral, yaw] for name, lateral, yaw in SCENARIOS}},
        "counts": counts,
        "behavior_score": counts["full_climb"] + counts["rear_hold"],
        "support_score": counts["front_top_support"],
        "rows": rows,
        "interpretation": "Independent behavior baseline only; the frozen shared-state 0707 imitation envelope remains authoritative for imitation quality.",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(OUTPUT / "core9_baseline_manifest.json", manifest, read_only=True)
    update_heartbeat("complete", "complete", counts=counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
