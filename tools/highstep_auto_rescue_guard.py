#!/usr/bin/env python3
"""Run and monitor the highstep support-pose rescue short train.

This intentionally does not edit training code automatically.  It starts one
training process, checks local TensorBoard/W&B event data every 15 minutes, and
stops the run early when the metrics are clearly failed or have reached the
planned play-inspection window.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
TASK = "RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0"
CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-25_05-59-47/model_124598.pt"
)
RUN_ROOT = REPO_ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher"
START_MODEL = 124598


TAGS = {
    "terrain": "Curriculum/terrain_levels",
    "bad_orientation": "Episode_Termination/bad_orientation",
    "rear": "Episode_Reward/rear_feet_highstep_clearance",
    "rear_first": "Episode_Reward/rear_first_foot_highstep_preclearance",
    "rear_second": "Episode_Reward/rear_second_foot_highstep_clearance",
    "lead_drive": "Episode_Reward/lead_rear_support_drive",
    "post_drive": "Episode_Reward/post_lead_body_drive",
    "support_contact": "Episode_Reward/highstep_leg_support_contact",
    "front_reach": "Episode_Reward/front_legs_reach",
    "front_clear": "Episode_Reward/front_feet_highstep_clearance",
    "second_clear_rate": "Curriculum/highstep_rear_branch_metrics/second_clear_rate",
    "one_sided": "Curriculum/highstep_rear_branch_metrics/one_sided_stall_ratio",
    "valid_rate": "Curriculum/highstep_rear_branch_metrics/valid_rate",
}

REAR_BRANCH_PREFIX = "Curriculum/highstep_rear_branch_metrics/"
ACTION_SCORE_PREFIX = "Curriculum/highstep_action_score/"


@dataclass
class Decision:
    state: str
    reason: str
    should_stop: bool = False


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(path: Path, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest_run_dir(start_time: float) -> Path | None:
    candidates: list[Path] = []
    for path in RUN_ROOT.iterdir():
        if not path.is_dir():
            continue
        try:
            if path.stat().st_mtime < start_time - 120:
                continue
        except FileNotFoundError:
            continue
        if any(path.glob("events.out.tfevents.*")) or (path / "params").exists():
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_checkpoint(run_dir: Path) -> int | None:
    best: int | None = None
    for path in run_dir.glob("model_*.pt"):
        match = re.match(r"model_(\d+)\.pt$", path.name)
        if match:
            value = int(match.group(1))
            best = value if best is None else max(best, value)
    return best


def latest_event(run_dir: Path) -> Path | None:
    events = list(run_dir.glob("events.out.tfevents.*"))
    if not events:
        return None
    return max(events, key=lambda p: p.stat().st_mtime)


def candidate_tags(tag: str) -> tuple[str, ...]:
    tags = [tag]
    if tag.startswith(REAR_BRANCH_PREFIX):
        tags.append(ACTION_SCORE_PREFIX + tag[len(REAR_BRANCH_PREFIX) :])
    return tuple(dict.fromkeys(tags))


def scalar_values(acc: EventAccumulator, tag: str) -> list[tuple[int, float]]:
    available = set(acc.Tags().get("scalars", []))
    selected = next((candidate for candidate in candidate_tags(tag) if candidate in available), None)
    if selected is None:
        return []
    return [(event.step, float(event.value)) for event in acc.Scalars(selected)]


def tail_mean(values: Iterable[tuple[int, float]], count: int = 120) -> float | None:
    vals = [v for _, v in values]
    if not vals:
        return None
    vals = vals[-count:]
    return sum(vals) / len(vals)


def last_step(metric_values: dict[str, list[tuple[int, float]]]) -> int | None:
    steps = [vals[-1][0] for vals in metric_values.values() if vals]
    return max(steps) if steps else None


def read_metrics(run_dir: Path) -> tuple[int | None, dict[str, float | None]]:
    event = latest_event(run_dir)
    if event is None:
        return None, {name: None for name in TAGS}
    acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
    acc.Reload()
    raw = {name: scalar_values(acc, tag) for name, tag in TAGS.items()}
    return last_step(raw), {name: tail_mean(vals) for name, vals in raw.items()}


def fmt_value(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def evaluate(
    metrics: dict[str, float | None],
    latest_step: int | None,
    ckpt: int | None,
    play_window_checkpoint: int,
    hard_stop_checkpoint: int,
) -> Decision:
    if latest_step is None:
        return Decision("WAIT", "event file not ready")

    delta = latest_step - START_MODEL
    rear = metrics.get("rear")
    rear_first = metrics.get("rear_first")
    rear_second = metrics.get("rear_second")
    lead = metrics.get("lead_drive")
    second_rate = metrics.get("second_clear_rate")
    one_sided = metrics.get("one_sided")
    bad_orientation = metrics.get("bad_orientation")

    if delta < 350:
        return Decision("WARMUP", f"only {delta} updates after checkpoint; keep collecting")

    missing = [
        name
        for name, value in {
            "rear": rear,
            "rear_first": rear_first,
            "rear_second": rear_second,
            "lead_drive": lead,
            "second_clear_rate": second_rate,
            "one_sided": one_sided,
            "bad_orientation": bad_orientation,
        }.items()
        if value is None
    ]
    if missing:
        return Decision("WAIT", "missing metrics: " + ", ".join(missing))

    assert rear is not None
    assert rear_first is not None
    assert rear_second is not None
    assert lead is not None
    assert second_rate is not None
    assert one_sided is not None
    assert bad_orientation is not None

    if delta >= 500:
        hard_failures: list[str] = []
        if rear < 0.075:
            hard_failures.append(f"rear={rear:.4f}<0.075")
        if rear_first < 0.005:
            hard_failures.append(f"rear_first={rear_first:.4f}<0.005")
        if rear_second < 0.008:
            hard_failures.append(f"rear_second={rear_second:.4f}<0.008")
        if lead < 0.018:
            hard_failures.append(f"lead_drive={lead:.4f}<0.018")
        if second_rate < 0.72:
            hard_failures.append(f"second_clear_rate={second_rate:.4f}<0.72")
        if one_sided > 0.30:
            hard_failures.append(f"one_sided={one_sided:.4f}>0.30")
        if bad_orientation > 0.004:
            hard_failures.append(f"bad_orientation={bad_orientation:.4f}>0.004")
        if hard_failures:
            return Decision("FAIL_STOP", "; ".join(hard_failures), True)

    promising = (
        rear >= 0.09
        and rear_first >= 0.008
        and rear_second >= 0.012
        and lead >= 0.024
        and second_rate >= 0.80
        and one_sided <= 0.24
        and bad_orientation <= 0.0025
    )
    if promising and ckpt is not None and ckpt >= play_window_checkpoint:
        return Decision("PLAY_WINDOW_STOP", f"promising metrics reached and checkpoint model_{ckpt}.pt exists", True)

    if ckpt is not None and ckpt >= hard_stop_checkpoint:
        return Decision("MIXED_STOP", f"checkpoint model_{ckpt}.pt reached hard stop window", True)

    return Decision("CONTINUE", f"delta={delta}, ckpt={ckpt}; not failed, not yet at stop window")


def stop_process(proc: subprocess.Popen[bytes], monitor_log: Path, reason: str) -> None:
    if proc.poll() is not None:
        log_line(monitor_log, f"train process already exited with code {proc.returncode}")
        return
    log_line(monitor_log, f"stopping train process with SIGINT: {reason}")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        log_line(monitor_log, "SIGINT timeout; terminating process")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log_line(monitor_log, "terminate timeout; killing process")
            proc.kill()
            proc.wait(timeout=30)


def write_report(
    report_path: Path,
    run_dir: Path | None,
    ckpt: int | None,
    latest_step: int | None,
    metrics: dict[str, float | None],
    decision: Decision,
    stdout_log: Path,
    monitor_log: Path,
) -> None:
    payload = {
        "time": now(),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "latest_checkpoint": ckpt,
        "latest_step": latest_step,
        "decision": decision.__dict__,
        "stdout_log": str(stdout_log),
        "monitor_log": str(monitor_log),
        "metrics_tail_mean": metrics,
        "auto_code_edits": "disabled",
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--bootstrap-interval-seconds", type=int, default=60)
    parser.add_argument("--play-window-checkpoint", type=int, default=125400)
    parser.add_argument("--hard-stop-checkpoint", type=int, default=125500)
    parser.add_argument("--no-start", action="store_true", help="Only monitor the newest current run.")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stdout_log = RUN_ROOT / f"auto_rescue_support_pose_{stamp}_train_stdout.log"
    monitor_log = RUN_ROOT / f"auto_rescue_support_pose_{stamp}_monitor.log"
    report_path = RUN_ROOT / f"auto_rescue_support_pose_{stamp}_decision.json"

    start_time = time.time()
    train_proc: subprocess.Popen[bytes] | None = None

    if args.no_start:
        log_line(monitor_log, "no-start mode; monitoring newest run only")
    else:
        command = [
            str(PYTHON),
            "scripts/rsl_rl/base/train.py",
            "--task",
            TASK,
            "--logger",
            "wandb",
            "--headless",
            "--resume",
            "--checkpoint",
            str(CHECKPOINT),
            "--max_iterations",
            str(args.max_iterations),
        ]
        log_line(monitor_log, "AUTO_CODE_EDITS_DISABLED")
        log_line(monitor_log, "starting train command: " + " ".join(command))
        stdout_file = stdout_log.open("ab")
        train_proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        log_line(monitor_log, f"train pid: {train_proc.pid}")

    last_decision = Decision("WAIT", "not evaluated yet")
    latest_step: int | None = None
    metrics: dict[str, float | None] = {name: None for name in TAGS}
    run_dir: Path | None = None
    ckpt: int | None = None

    while True:
        if train_proc is not None and train_proc.poll() is not None:
            log_line(monitor_log, f"train process exited with code {train_proc.returncode}")
            run_dir = run_dir or latest_run_dir(start_time)
            if run_dir is not None:
                latest_step, metrics = read_metrics(run_dir)
                ckpt = latest_checkpoint(run_dir)
            last_decision = Decision("TRAIN_EXITED", f"exit code {train_proc.returncode}", True)
            break

        run_dir = latest_run_dir(start_time)
        if run_dir is None:
            last_decision = Decision("WAIT", "run directory not ready")
            log_line(monitor_log, last_decision.reason)
            time.sleep(args.bootstrap_interval_seconds)
            continue

        try:
            latest_step, metrics = read_metrics(run_dir)
            ckpt = latest_checkpoint(run_dir)
            last_decision = evaluate(
                metrics,
                latest_step,
                ckpt,
                args.play_window_checkpoint,
                args.hard_stop_checkpoint,
            )
        except Exception as exc:  # noqa: BLE001 - monitor must not crash the train on parse hiccups
            last_decision = Decision("WAIT", f"metric parse failed: {exc!r}")

        metric_summary = " ".join(
            f"{name}={fmt_value(metrics.get(name))}"
            for name in (
                "terrain",
                "rear",
                "rear_first",
                "rear_second",
                "lead_drive",
                "post_drive",
                "support_contact",
                "front_reach",
                "front_clear",
                "second_clear_rate",
                "one_sided",
                "bad_orientation",
            )
        )
        log_line(
            monitor_log,
            f"run={run_dir.name} step={latest_step} ckpt={ckpt} decision={last_decision.state} "
            f"reason={last_decision.reason} {metric_summary}",
        )
        write_report(report_path, run_dir, ckpt, latest_step, metrics, last_decision, stdout_log, monitor_log)

        if last_decision.should_stop:
            if train_proc is not None:
                stop_process(train_proc, monitor_log, last_decision.reason)
            break

        time.sleep(args.interval_seconds)

    write_report(report_path, run_dir, ckpt, latest_step, metrics, last_decision, stdout_log, monitor_log)
    log_line(monitor_log, f"final decision: {last_decision.state} - {last_decision.reason}")
    log_line(monitor_log, f"decision report: {report_path}")
    log_line(monitor_log, f"train stdout log: {stdout_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
