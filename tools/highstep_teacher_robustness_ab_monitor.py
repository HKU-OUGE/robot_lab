#!/usr/bin/env python3
"""Strictly read-only single-screen monitor for the active v1.11 Teacher A/B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/home/lxq/Softwares/robot_lab")
DASH = ROOT / "tmp/highstep_dashboard_active_workflow.json"
GREEN, YELLOW, RED, RESET, BOLD = "\033[32m", "\033[33m", "\033[31m", "\033[0m", "\033[1m"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def cmd(*args: str) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout.strip()


def color(ok: bool, warning: bool = False) -> str:
    return GREEN if ok else (YELLOW if warning else RED)


def snapshot() -> str:
    dash = load(DASH)
    state_path = Path(str(dash.get("state_path", "/nonexistent")))
    state = load(state_path)
    work = state_path.parent
    heartbeat_path = work / "heartbeat.json"
    heartbeat = load(heartbeat_path)
    prereg_path = Path(str(dash.get("preregistration_path", "/nonexistent")))
    spec_path = Path(str(dash.get("spec_path", "/nonexistent")))
    import hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    authority_ok = bool(
        dash.get("workflow_id") == "highstep_teacher_robustness_ab_20260715"
        and state.get("workflow_id") == dash.get("workflow_id")
        and sha(spec_path) == dash.get("spec_sha256") == state.get("spec_sha256")
        and sha(prereg_path) == dash.get("preregistration_sha256") == state.get("preregistration_sha256")
    )
    hb_age = time.time() - float(heartbeat.get("written_epoch", 0)) if heartbeat else float("inf")
    service = cmd("systemctl", "--user", "is-active", "highstep-teacher-robustness-ab.service") or "unknown"
    restarts = cmd("systemctl", "--user", "show", "highstep-teacher-robustness-ab.service", "-p", "NRestarts", "--value") or "?"
    sup = state.get("supervisor_pid")
    child = state.get("active_pid")
    alive = lambda pid: bool(pid and Path(f"/proc/{pid}").exists())
    status = str(state.get("status", "unknown"))
    transition = status in {"preflight", "running", "rollout_completed", "infrastructure_retry"} and not child
    process_ok = alive(sup) and (alive(child) or transition or status.startswith("new_teacher_"))
    gpu = cmd("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader") or "none"
    completed = int(state.get("total_completed", 0))
    durations = []
    family = {"nominal": 0, "inward": 0, "impulse": 0, "combined": 0}
    valid = 0
    for teacher in ("A", "B"):
        base = work / "evaluations" / teacher
        if base.exists():
            for path in base.glob("*/result.json"):
                row = load(path)
                if row.get("valid") is True:
                    valid += 1
                if row.get("family") in family:
                    family[row["family"]] += 1
                if isinstance(row.get("duration_seconds"), (int, float)):
                    durations.append(float(row["duration_seconds"]))
    durations.sort()
    median = durations[len(durations)//2] if durations else None
    eta = (204 - completed) * median if median is not None else None
    row = state.get("current_row") or {}
    latest_log = Path(str(state.get("latest_log", "/nonexistent")))
    tail = " | ".join(latest_log.read_text(errors="replace").splitlines()[-2:])[-240:] if latest_log.exists() else "none"
    lines = ["\033[2J\033[H" + BOLD + "Highstep Teacher Robustness A/B (read-only)" + RESET]
    lines.append(f"{color(authority_ok)}authority={'OK' if authority_ok else 'MISMATCH'}{RESET}  workflow={dash.get('workflow_id')}  spec={str(dash.get('spec_sha256',''))[:12]} prereg={str(dash.get('preregistration_sha256',''))[:12]}")
    service_ok = service == "active" or status.startswith("new_teacher_")
    lines.append(f"{color(service_ok, transition)}service={service}{RESET} restarts={restarts} supervisor={sup}({'alive' if alive(sup) else 'dead'}) eval={child}({'alive' if alive(child) else 'none/dead'})")
    lines.append(f"{color(hb_age <= 10, hb_age <= 30)}heartbeat_age={hb_age:.1f}s{RESET} status={status} phase={state.get('phase')} failure={state.get('failure_class','none')}")
    lines.append(f"Teacher={state.get('current_teacher')} run={state.get('current_run_id')} seed={row.get('seed')} family={row.get('family')}")
    lines.append(f"rear_width={row.get('initial_rear_width_m')} mode={row.get('inward_mode')} impulse={row.get('impulse_phase')}/{row.get('impulse_direction')}/{row.get('impulse_delta_v_mps')}")
    lines.append(f"progress A={state.get('A_completed',0)}/102 B={state.get('B_completed',0)}/102 total={completed}/204  nominal={family['nominal']}/6 inward={family['inward']}/72 impulse={family['impulse']}/108 combined={family['combined']}/18")
    lines.append(f"valid={valid} infra_invalid={state.get('invalid_infrastructure_count',0)} retries={state.get('infrastructure_retry_count',0)} current_elapsed={time.time()-float(state.get('current_started_epoch',time.time())):.0f}s ETA={(eta/60):.1f}min" if eta is not None else f"valid={valid} infra_invalid={state.get('invalid_infrastructure_count',0)} retries={state.get('infrastructure_retry_count',0)} ETA=collecting median")
    lines.append(f"levels A={state.get('teacher_A_aggregate','pending')} B={state.get('teacher_B_aggregate','pending')} gate={state.get('decision','pending')}")
    lines.append(f"GPU: {gpu}")
    lines.append(f"last_error={state.get('last_error','none')}")
    lines.append(f"latest: {tail}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    while True:
        print(snapshot(), flush=True)
        if args.once:
            return 0
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
