#!/usr/bin/env python3
"""Pause v1.7.1 immediately after the selected atomic child exits."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
STATE_ROOT = ROOT / "tmp/highstep_historical_0707_exact_new_teacher_20260714"
STATE = STATE_ROOT / "state.json"
SUPERVISOR_PID = int(sys.argv[1])
PHASE_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "training_100_attempt1"


def alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


saw_live_child = False
while alive(SUPERVISOR_PID):
    state = read_json(STATE)
    active_pid = state.get("active_pid")
    if alive(active_pid):
        saw_live_child = True
    if (
        state.get("workflow_id") == "highstep_historical_0707_exact_new_teacher_20260714"
        and str(state.get("phase", "")).startswith(PHASE_PREFIX)
        and saw_live_child
        and active_pid is None
        and not alive(active_pid)
    ):
        os.kill(SUPERVISOR_PID, signal.SIGSTOP)
        print(f"paused supervisor {SUPERVISOR_PID} after {PHASE_PREFIX} child exit", flush=True)
        raise SystemExit(0)
    time.sleep(0.001)

raise SystemExit("supervisor exited before a complete stage boundary")
