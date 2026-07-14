#!/usr/bin/env python3
"""Live guard for the current ActionScore highstep run.

This tool is intentionally conservative: it only reads local TensorBoard
scalars, writes a status log, and stops the current training PID on hard
failure.  It never edits source code or starts a second training process.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = {
    "support": "Curriculum/highstep_action_score/support_score",
    "violation": "Curriculum/highstep_action_score/support_floor_violation_rate",
    "total": "Curriculum/highstep_action_score/total",
    "lift_active": "Curriculum/highstep_action_score/lead_support_drive_lift_velocity_active_mean",
    "height_active": "Curriculum/highstep_action_score/lead_support_drive_height_active_mean",
    "body_active": "Curriculum/highstep_action_score/lead_support_drive_body_active_mean",
    "lead_drive": "Episode_Reward/lead_rear_support_drive",
    "post_drive": "Episode_Reward/post_lead_body_drive",
    "bad_orientation": "Episode_Termination/bad_orientation",
    "time_out": "Episode_Termination/time_out",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--tail", type=int, default=120)
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(path: Path, text: str) -> None:
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def latest_event(run_dir: Path) -> Path | None:
    events = list(run_dir.glob("events.out.tfevents.*"))
    if not events:
        return None
    return max(events, key=lambda p: p.stat().st_mtime)


def read_tail(run_dir: Path, tail: int) -> tuple[int | None, dict[str, float | None]]:
    event = latest_event(run_dir)
    if event is None:
        return None, {name: None for name in TAGS}
    acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    latest_step: int | None = None
    values: dict[str, float | None] = {}
    for name, tag in TAGS.items():
        if tag not in available:
            values[name] = None
            continue
        scalars = acc.Scalars(tag)
        if not scalars:
            values[name] = None
            continue
        latest_step = scalars[-1].step if latest_step is None else max(latest_step, scalars[-1].step)
        selected = scalars[-tail:]
        values[name] = sum(float(item.value) for item in selected) / len(selected)
    return latest_step, values


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def stop_reason(delta: int, metrics: dict[str, float | None]) -> str | None:
    bad = metrics.get("bad_orientation")
    timeout = metrics.get("time_out")
    support = metrics.get("support")
    violation = metrics.get("violation")
    lift_active = metrics.get("lift_active")
    post_drive = metrics.get("post_drive")

    if delta < 300:
        return None
    if bad is not None and bad > 0.010:
        return f"bad_orientation too high: {bad:.5f} > 0.010"
    if delta >= 600:
        if (
            timeout is not None
            and bad is not None
            and support is not None
            and violation is not None
            and timeout < 0.80
            and bad > 0.0085
            and support < 0.38
            and violation > 0.65
        ):
            return (
                "combined instability/support failure: "
                f"time_out={timeout:.5f}, bad={bad:.5f}, support={support:.5f}, violation={violation:.5f}"
            )
        if support is not None and violation is not None and support < 0.34 and violation > 0.75:
            return f"support collapsed after 600 updates: support={support:.5f}, violation={violation:.5f}"
        if lift_active is not None and post_drive is not None and lift_active < 0.03 and post_drive < 0.0015:
            return f"support lift signal dead after 600 updates: lift_active={lift_active:.5f}, post_drive={post_drive:.5f}"
    if delta >= 900 and support is not None and violation is not None:
        if support < 0.38 and violation > 0.65:
            return f"late support still failed: support={support:.5f}, violation={violation:.5f}"
    return None


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    log_path = args.log or (run_dir / "actionscore_live_supervisor.log")
    log(log_path, f"supervisor started pid={args.pid} run_dir={run_dir} start_step={args.start_step}")
    while True:
        alive = pid_alive(args.pid)
        step, metrics = read_tail(run_dir, args.tail)
        delta = 0 if step is None else step - args.start_step
        status = " ".join(f"{key}={fmt(metrics.get(key))}" for key in TAGS)
        log(log_path, f"alive={alive} step={step} delta={delta} {status}")
        reason = stop_reason(delta, metrics)
        if reason is not None and alive:
            log(log_path, f"STOP: {reason}")
            os.kill(args.pid, signal.SIGTERM)
            time.sleep(30)
            if pid_alive(args.pid):
                log(log_path, "process still alive after SIGTERM; sending SIGKILL")
                os.kill(args.pid, signal.SIGKILL)
            return 2
        if not alive:
            log(log_path, "training process is no longer alive; supervisor exiting")
            return 0
        time.sleep(max(args.interval_seconds, 30))


if __name__ == "__main__":
    raise SystemExit(main())
