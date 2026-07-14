#!/usr/bin/env python3
"""Recover a completed v1.5.2 core9 matrix from its immutable play logs.

This does not synthesize behavior evidence.  It rebuilds the JSONL ledger from
the nine existing schema-7 payloads, then executes the aggregate validator
embedded in the current schema-4 monitor source.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess


PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
SCENARIOS = {
    "nominal": (0.00, 0.0),
    "left_offset": (0.12, 4.0),
    "right_offset": (-0.12, -4.0),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def aggregate_source(monitor: Path) -> str:
    lines = monitor.read_text().splitlines()
    start = next(
        index + 1
        for index, line in enumerate(lines)
        if '"$BOOTSTRAP_CHECKPOINT" <<\'PY\'' in line
    )
    end = next(index for index in range(start, len(lines)) if lines[index] == "PY")
    return "\n".join(lines[start:end]) + "\n"


def eval_payload(log: Path) -> dict:
    payloads = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[HIGHSTEP_EVAL_JSON] "):
            payloads.append(json.loads(line.split(" ", 1)[1]))
    if len(payloads) != 1:
        raise RuntimeError(f"expected one schema-7 payload in {log}, got {len(payloads)}")
    return payloads[0]


def completed_launcher_payloads(
    output: Path, explicit_launcher: Path | None = None
) -> tuple[Path, dict[tuple[int, str], dict]]:
    candidates = []
    launch_pattern = re.compile(r"play_eval ckpt=.* seed=(\d+) scenario=([a-z_]+) ")
    launchers = [explicit_launcher.resolve(strict=True)] if explicit_launcher else output.glob("launcher*.log")
    for launcher in launchers:
        current = None
        payloads: dict[tuple[int, str], dict] = {}
        for line in launcher.read_text(encoding="utf-8", errors="replace").splitlines():
            match = launch_pattern.search(line)
            if match:
                current = (int(match.group(1)), match.group(2))
                continue
            marker = "play_eval_rc=0 [HIGHSTEP_EVAL_JSON] "
            if current is not None and marker in line:
                payloads[current] = json.loads(line.split(marker, 1)[1])
                current = None
        if set(payloads) == {(seed, scenario) for seed in (11, 22, 33) for scenario in SCENARIOS}:
            candidates.append((launcher, payloads))
    if not candidates:
        raise RuntimeError("no completed nine-run launcher evidence is available")
    return max(candidates, key=lambda item: item[0].stat().st_mtime_ns)


def no_severe(item: dict) -> bool:
    return bool(
        float(item["critical_rear_min_abs_y_q05"]) >= 0.04
        and float(item["critical_rear_width_q05"]) >= 0.18
        and float(item["critical_center_violation_rate"]) <= 0.10
        and float(item["critical_width_violation_rate"]) <= 0.10
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-teacher-manifest", type=Path, required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--role", default="student")
    parser.add_argument("--launcher-evidence", type=Path)
    parser.add_argument("--effective-updates", type=int, default=500)
    args = parser.parse_args()

    output = args.output_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    monitor = args.monitor.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    parent = args.parent_teacher_manifest.resolve(strict=True)
    checkpoint_sha = sha256_file(checkpoint)
    launcher, launcher_payloads = completed_launcher_payloads(output, args.launcher_evidence)
    rows = []
    log_bindings = []
    for seed in (11, 22, 33):
        for scenario, (lateral, yaw) in SCENARIOS.items():
            log = output / "play_logs" / f"{checkpoint.stem}_seed{seed}_{scenario}.log"
            item = launcher_payloads[(seed, scenario)]
            if not (
                item.get("schema_version") == 7
                and item.get("checkpoint_sha256") == checkpoint_sha
                and item.get("runtime_snapshot_checkpoint_sha256_verified") is True
                and item.get("runtime_snapshot_checkpoint_sha256") == checkpoint_sha
                and item.get("schedule_valid") is True
                and item.get("schedule_runtime_match") is True
                and item.get("schedule_clock_runtime_match") is True
                and item.get("reset_valid") is True
                and item.get("initial_geometry_valid") is True
                and abs(float(item.get("yaw_offset_deg", 999.0)) - yaw) < 1.0e-9
            ):
                raise RuntimeError(f"existing core9 log binding is invalid: {log}")
            rows.append(
                {
                    "checkpoint": str(checkpoint), "seed": seed, "scenario": scenario,
                    "vx": 0.45, "lateral": lateral, "yaw_offset_deg": yaw,
                    "randomize": False, "action_delay_steps": 0, "task": args.task,
                    "role": args.role, "return_code": 0,
                    "log": f"{launcher}#seed{seed}_{scenario}", "eval": item,
                }
            )
            log_bindings.append({
                "launcher_log": str(launcher), "launcher_log_sha256": sha256_file(launcher),
                "record": f"seed{seed}_{scenario}",
                "individual_play_log": str(log),
                "individual_play_log_sha256": sha256_file(log) if log.is_file() else None,
            })

    recovery_root = output / "recovery_evidence" / datetime.now().strftime("%Y%m%d_%H%M%S")
    recovery_root.mkdir(parents=True)
    for name in ("eval_runs.jsonl", "decision.txt", "checkpoint_summary.csv", "evaluation_manifest.json"):
        source = output / name
        if source.exists():
            shutil.copy2(source, recovery_root / f"before_{name}")
    ledger = output / "eval_runs.jsonl"
    atomic_text(ledger, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    decision = output / "decision.txt"
    csv_path = output / "checkpoint_summary.csv"
    manifest = output / "evaluation_manifest.json"
    command = [
        str(PYTHON), "-", str(ledger), str(decision), str(csv_path), str(manifest),
        str(run_dir), args.task, args.role, "1", str(parent), args.workflow_id,
        "core9", "0", "",
    ]
    completed = subprocess.run(
        command, input=aggregate_source(monitor), text=True, capture_output=True, check=False
    )
    atomic_text(recovery_root / "aggregate_stdout.log", completed.stdout)
    atomic_text(recovery_root / "aggregate_stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"schema-4 aggregate validator exited {completed.returncode}")
    aggregate = json.loads(manifest.read_text())
    if not (
        aggregate.get("evaluation_complete") is True
        and aggregate.get("matrix_complete") is True
        and aggregate.get("student_source_lineage_valid") is True
        and aggregate.get("runtime_snapshot_checkpoint_sha256_verified") is True
        and aggregate.get("completed_runs") == 9
    ):
        raise RuntimeError(f"recovered aggregate remains invalid: {aggregate}")

    counts = {
        "valid": 9,
        "full_climb": sum(row["eval"].get("full_climb_success") is True for row in rows),
        "rear_hold": sum(row["eval"].get("rear_on_platform_hold_success") is True for row in rows),
        "front_top_support": sum(row["eval"].get("front_top_support_reached") is True for row in rows),
        "no_severe_inward": sum(no_severe(row["eval"]) for row in rows),
    }
    summary = {
        "schema_version": 1, "kind": "highstep_v152_real_gain_core9",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "effective_updates": args.effective_updates, "counts": counts,
        "final_behavior_gate_passed": bool(
            counts["valid"] == 9 and counts["full_climb"] >= 8 and counts["rear_hold"] >= 8
            and counts["front_top_support"] >= 8 and counts["no_severe_inward"] >= 8
        ),
        "evaluation_manifest": str(manifest),
        "evaluation_manifest_sha256": sha256_file(manifest),
        "recovered_from_existing_play_logs": True,
    }
    atomic_text(output / "core9_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    evidence = {
        "schema_version": 1, "kind": "highstep_v152_core9_recovery",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "monitor": str(monitor), "monitor_sha256": sha256_file(monitor),
        "eval_runs": str(ledger), "eval_runs_sha256": sha256_file(ledger),
        "evaluation_manifest": str(manifest), "evaluation_manifest_sha256": sha256_file(manifest),
        "core9_summary": str(output / "core9_summary.json"),
        "core9_summary_sha256": sha256_file(output / "core9_summary.json"),
        "log_bindings": log_bindings, "counts": counts,
        "behavior_failure_did_not_invalidate_evaluation": True,
    }
    evidence_path = recovery_root / "recovery_manifest.json"
    atomic_text(evidence_path, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    evidence_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({"recovery_manifest": str(evidence_path), **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
