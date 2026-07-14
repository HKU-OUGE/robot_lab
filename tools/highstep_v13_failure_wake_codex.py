#!/usr/bin/env python3
"""Wake one bounded Codex repair turn when the v1.3 supervisor fails closed."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_v13_20260713"
CODEX = Path("/home/lxq/.nvm/versions/node/v24.16.0/bin/codex")
THREAD_ID = "019f4c6b-0fbd-73d2-91cc-6c72e8c8c35c"
MODEL = "gpt-5.6-sol"
REASONING = "max"
SERVICE = "highstep-student-recovery-v13.service"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def service_state() -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def another_resume_is_running() -> bool:
    needle = THREAD_ID.encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"codex" in command and b"exec\0" in command and b"resume\0" in command and needle in command:
            return True
    return False


def main() -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with (STATE_ROOT / "v13_failure_repair.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if service_state() == "active":
            return 0
        if (ROOT / "tmp/highstep_manual_play_active").exists() or another_resume_is_running():
            return 75
        state_path = STATE_ROOT / "state.json"
        handoff_path = STATE_ROOT / "handoff.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        handoff = json.loads(handoff_path.read_text()) if handoff_path.is_file() else {}
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = STATE_ROOT / "failure_recovery" / stamp
        output.mkdir(parents=True)
        final_message = output / "codex.final.txt"
        event_log = output / "codex.jsonl"
        prompt = f"""STEER：v1.3 highstep 独立 supervisor 已 fail-closed。严格读取正式 spec：
{ROOT / 'docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md'}
以及 state/handoff/最近阶段子日志。当前 state 摘要：{json.dumps(state, ensure_ascii=False)}
当前 handoff 摘要：{json.dumps(handoff, ensure_ascii=False)}

这是已授权的异常自动接管：定位根因，保留 checkpoint、数据、manifest 和失败日志，只实施 spec 范围内的最小正确性/基础设施修复；不得放宽门禁或修改 Teacher、reward、prior、网络、action_scale、joint_pos.clip、部署输入或训练路线。完成静态和全量测试，重置仅由已修复基础设施缺陷消耗的 retry，重启 {SERVICE}，确认 PID、lock、state、heartbeat 以及越过原失败阶段。若是规范门禁终态则不得绕过。禁止静默停止。

所有 Codex 推理固定 model={MODEL}, reasoning_effort={REASONING}，禁止 ultra。修复结果与验证证据写入 {output / 'repair_result.json'}。"""
        atomic_json(
            output / "request.json",
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "thread_id": THREAD_ID,
                "model": MODEL,
                "reasoning_effort": REASONING,
                "state": state,
                "handoff": handoff,
            },
        )
        command = [
            str(CODEX),
            "-C",
            str(ROOT),
            "exec",
            "-m",
            MODEL,
            "-c",
            f'model_reasoning_effort="{REASONING}"',
            "-c",
            'service_tier="priority"',
            "--strict-config",
            "resume",
            "--dangerously-bypass-approvals-and-sandbox",
            "--output-last-message",
            str(final_message),
            "--json",
            THREAD_ID,
            prompt,
        ]
        environment = os.environ.copy()
        environment.setdefault("ALL_PROXY", "socks5h://127.0.0.1:10808")
        environment.setdefault("all_proxy", environment["ALL_PROXY"])
        with event_log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
        if result.returncode != 0:
            return 75
        current = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if service_state() == "active" or current.get("status") in {
            "candidate_ready",
            "stopped_by_gate",
            "stopped_by_action_regression",
        }:
            return 0
        return 75


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired:
        raise SystemExit(75)
