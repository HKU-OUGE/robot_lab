#!/usr/bin/env python3
"""Sleep-safe highstep supervisor.

This supervisor may apply exactly one pre-declared fallback edit:

1. First run: current rescue setting, support_pose_scale = 0.0.
2. If the local-event metrics hard-fail, restore support_pose_scale = 0.55
   to reproduce the 2026-06-28_08-23-04 hopeful rear-leg branch.
3. Run the same guarded short train once more.

It must not make any other source-code changes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
RUN_ROOT = REPO_ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher"
ENV_CFG = (
    REPO_ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/highstep_env_cfg.py"
)
BACKUP_ROOT = ENV_CFG.parent / "__backups__"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(path: Path, text: str) -> None:
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest_decision(after: float) -> Path | None:
    candidates = [
        path
        for path in RUN_ROOT.glob("auto_rescue_support_pose_*_decision.json")
        if path.stat().st_mtime >= after - 5
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_decision(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_support_pose_scale(value: float, supervisor_log: Path) -> None:
    text = ENV_CFG.read_text(encoding="utf-8")
    old_zero = 'self.rewards.highstep_leg_support_contact.params["support_pose_scale"] = 0.0'
    old_restore = 'self.rewards.highstep_leg_support_contact.params["support_pose_scale"] = 0.55'
    new = f'self.rewards.highstep_leg_support_contact.params["support_pose_scale"] = {value:.2f}'
    if new in text:
        log(supervisor_log, f"support_pose_scale already {value:.2f}; no edit needed")
        return
    if old_zero not in text and old_restore not in text:
        raise RuntimeError("support_pose_scale override line not found; refusing automatic edit")

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = BACKUP_ROOT / f"{stamp}_auto_sleep_fallback_support_pose_{value:.2f}_from_2026-06-28_08-23-04"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "highstep_env_cfg.py"
    shutil.copy2(ENV_CFG, backup_path)
    log(supervisor_log, f"backup before fallback edit: {backup_path}")

    if old_zero in text:
        text = text.replace(old_zero, new, 1)
    else:
        text = text.replace(old_restore, new, 1)
    ENV_CFG.write_text(text, encoding="utf-8")
    log(supervisor_log, f"applied only allowed fallback edit: support_pose_scale={value:.2f}")

    subprocess.run([str(PYTHON), "-m", "py_compile", str(ENV_CFG)], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "diff", "--check", "--", str(ENV_CFG.relative_to(REPO_ROOT))], cwd=REPO_ROOT, check=True)
    log(supervisor_log, "fallback edit validation passed: py_compile and git diff --check")


def run_guard(label: str, supervisor_log: Path, max_iterations: int, interval_seconds: int) -> dict:
    started = time.time()
    cmd = [
        str(PYTHON),
        "tools/highstep_auto_rescue_guard.py",
        "--max-iterations",
        str(max_iterations),
        "--interval-seconds",
        str(interval_seconds),
        "--bootstrap-interval-seconds",
        "60",
        "--play-window-checkpoint",
        "127000",
        "--hard-stop-checkpoint",
        "128500",
    ]
    log(supervisor_log, f"starting guarded train [{label}]: {' '.join(cmd)}")
    stdout_path = RUN_ROOT / f"sleep_supervisor_{time.strftime('%Y%m%d_%H%M%S')}_{label}.log"
    with stdout_path.open("ab") as stdout:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, stdout=stdout, stderr=subprocess.STDOUT, check=False)
    log(supervisor_log, f"guarded train [{label}] exited code={completed.returncode}, log={stdout_path}")

    decision_path = latest_decision(started)
    if decision_path is None:
        raise RuntimeError(f"no decision report found for guarded train [{label}]")
    decision = load_decision(decision_path)
    log(supervisor_log, f"decision [{label}]: {decision_path}")
    log(supervisor_log, json.dumps(decision.get("decision", {}), ensure_ascii=False))
    return decision


def main() -> int:
    max_iterations = 4000
    interval_seconds = 900
    supervisor_log = RUN_ROOT / f"sleep_supervisor_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log(supervisor_log, "HLC-OK sleep supervisor started")
    log(supervisor_log, "automatic code edits limited to one fallback: support_pose_scale 0.0 -> 0.55")

    first = run_guard("support_pose_0p00", supervisor_log, max_iterations, interval_seconds)
    first_state = first.get("decision", {}).get("state")

    if first_state == "FAIL_STOP":
        log(supervisor_log, "first branch hard-failed; restoring 08:23-style support_pose_scale=0.55")
        set_support_pose_scale(0.55, supervisor_log)
        second = run_guard("fallback_support_pose_0p55", supervisor_log, max_iterations, interval_seconds)
        log(supervisor_log, "fallback branch finished")
        log(supervisor_log, json.dumps(second.get("decision", {}), ensure_ascii=False))
    else:
        log(supervisor_log, f"first branch did not hard-fail ({first_state}); no fallback code edit")

    log(supervisor_log, "sleep supervisor complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
