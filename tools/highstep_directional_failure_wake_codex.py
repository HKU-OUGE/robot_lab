#!/usr/bin/env python3
"""Wake one bounded gpt-5.6-sol+max repair turn after directional fail-closed."""

from __future__ import annotations
import fcntl, json, os
from pathlib import Path
import subprocess, time

ROOT = Path("/home/lxq/Softwares/robot_lab")
STATE_ROOT = ROOT / "tmp/highstep_directional_real_robot_trial_20260713"
CODEX = Path("/home/lxq/.nvm/versions/node/v24.16.0/bin/codex")
THREAD_ID = "019f4c6b-0fbd-73d2-91cc-6c72e8c8c35c"
SERVICE = "highstep-directional-trial.service"

def active() -> bool:
    return subprocess.run(["systemctl","--user","is-active","--quiet",SERVICE], check=False).returncode == 0

def main() -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with (STATE_ROOT/"failure_repair.lock").open("a+") as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return 75
        if active() or (ROOT/"tmp/highstep_manual_play_active").exists(): return 0 if active() else 75
        state = json.loads((STATE_ROOT/"state.json").read_text()) if (STATE_ROOT/"state.json").is_file() else {}
        stamp=time.strftime("%Y%m%d_%H%M%S"); out=STATE_ROOT/"failure_recovery"/stamp; out.mkdir(parents=True)
        prompt=f'''STEER：directional highstep supervisor 已 fail-closed。读取最高 spec、只读 preregistration、state/handoff 与最近日志：{state}。
保留旧 v1.3 stopped_by_gate；仅做规范内最小基础设施修复，不放宽门禁，不改变 Teacher/reward/prior/网络/action_scale/joint_pos.clip/部署输入/gains/action contract，不自动真机部署。全量测试后重启 {SERVICE} 并确认越过原故障。规范门禁终态不得绕过。所有推理固定 gpt-5.6-sol + max，禁止 ultra。'''
        command=[str(CODEX),"-C",str(ROOT),"exec","-m","gpt-5.6-sol","-c",'model_reasoning_effort="max"',"--strict-config","resume","--dangerously-bypass-approvals-and-sandbox","--output-last-message",str(out/"final.txt"),"--json",THREAD_ID,prompt]
        env=os.environ.copy(); env.setdefault("ALL_PROXY","socks5h://127.0.0.1:10808"); env.setdefault("all_proxy",env["ALL_PROXY"])
        with (out/"codex.jsonl").open("w") as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900,check=False)
        return 0 if result.returncode == 0 and active() else 75

if __name__ == "__main__":
    try: raise SystemExit(main())
    except subprocess.TimeoutExpired: raise SystemExit(75)
