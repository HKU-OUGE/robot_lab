#!/usr/bin/env python3
"""Derive the v1.8-only core9 monitor without changing canonical schema4."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat


ROOT = Path("/home/lxq/Softwares/robot_lab")
CANONICAL = ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh"
CANONICAL_SHA = "ccafcd32e803ed6617a5e5bf25429ba56f3128a2d44a43daa1f57383d5a5f6e8"
STAGE_A_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-"
    "ArcdogAdjustableLeg-v0"
)
STAGE_B_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Robust-"
    "ArcdogAdjustableLeg-v0"
)
NEEDLE = (
    "  RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0|\\\n"
    "  RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0)"
)
REPLACEMENT = (
    "  RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0|\\\n"
    f"  {STAGE_A_TASK}|\\\n"
    f"  {STAGE_B_TASK}|\\\n"
    "  RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0)"
)
ROBUST_PARENT_NEEDLE = '''    robust_student = task == (
        "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    )'''
ROBUST_PARENT_REPLACEMENT = f'''    robust_student = task in (
        (
            "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-"
            "ArcdogAdjustableLeg-v0"
        ),
        "{STAGE_A_TASK}",
        "{STAGE_B_TASK}",
    )'''
REUSE_NEEDLE = '''  local log_file="$OUT/play_logs/$(basename "$ckpt" .pt)_seed${seed}_${name}.log"
  local cmd=('''
REUSE_REPLACEMENT = '''  local log_file="$OUT/play_logs/$(basename "$ckpt" .pt)_seed${seed}_${name}.log"
  if [[ -n "${HIGHSTEP_V18_CORE9_REUSE_MANIFEST:-}" ]]; then
    if "$PY" - "$HIGHSTEP_V18_CORE9_REUSE_MANIFEST" "$log_file" "$ckpt" "$seed" "$name" "$vx" "$lateral" "$yaw" "$randomize" "$action_delay" "$TASK" "$ROLE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, log_path, checkpoint, seed, scenario, vx, lateral, yaw, randomize, action_delay, task, role = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
log = Path(log_path).resolve()
checkpoint_path = Path(checkpoint).resolve()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
matches = [
    row for row in manifest.get("rows", [])
    if row == {
        "action_delay_steps": int(action_delay),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "lateral": float(lateral),
        "log": str(log),
        "log_sha256": sha(log),
        "randomize": randomize == "yes",
        "role": role,
        "scenario": scenario,
        "seed": int(seed),
        "task": task,
        "vx": float(vx),
        "yaw_offset_deg": float(yaw),
    }
]
payloads = [
    json.loads(line.split(" ", 1)[1])
    for line in log.read_text(errors="replace").splitlines()
    if line.startswith("[HIGHSTEP_EVAL_JSON] ")
]
valid = bool(
    manifest.get("kind") == "highstep_v18_core9_readonly_reuse_manifest"
    and manifest.get("workflow_id") == "highstep_student_env_curriculum_v18_20260714"
    and manifest.get("checkpoint") == str(checkpoint_path)
    and manifest.get("checkpoint_sha256") == sha(checkpoint_path)
    and manifest.get("matrix_complete") is True
    and len(manifest.get("rows", [])) == 9
    and len(matches) == 1
    and len(payloads) == 1
    and payloads[0].get("schema_version") == 7
    and payloads[0].get("checkpoint_sha256") == sha(checkpoint_path)
    and payloads[0].get("schedule_valid") is True
    and payloads[0].get("reset_valid") is True
)
raise SystemExit(0 if valid else 2)
PY
    then
      append_eval_record "$log_file" "$ckpt" "$seed" "$name" "$vx" "$lateral" "$yaw" "$randomize" 0 "$action_delay"
      log "reused_sha_bound_eval ckpt=$ckpt seed=$seed scenario=$name log=$log_file"
      return 0
    fi
    log "v1.8 reuse manifest/log binding failed ckpt=$ckpt seed=$seed scenario=$name"
    return 1
  fi
  local cmd=('''


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_v18_monitor(source: str) -> str:
    if source.count(NEEDLE) != 1:
        raise RuntimeError("canonical schema4 task allowlist anchor changed")
    if source.count(ROBUST_PARENT_NEEDLE) != 1:
        raise RuntimeError("canonical schema4 robust-parent anchor changed")
    if source.count(REUSE_NEEDLE) != 1:
        raise RuntimeError("canonical schema4 play-log anchor changed")
    rendered = source.replace(NEEDLE, REPLACEMENT).replace(
        ROBUST_PARENT_NEEDLE, ROBUST_PARENT_REPLACEMENT
    ).replace(REUSE_NEEDLE, REUSE_REPLACEMENT)
    if rendered.count(STAGE_A_TASK) != 2 or rendered.count(STAGE_B_TASK) != 2:
        raise RuntimeError("v1.8 core9 task/parent insertion is not exact")
    restored = rendered.replace(REPLACEMENT, NEEDLE).replace(
        ROBUST_PARENT_REPLACEMENT, ROBUST_PARENT_NEEDLE
    ).replace(REUSE_REPLACEMENT, REUSE_NEEDLE)
    if restored != source:
        raise RuntimeError("v1.8 monitor derivation changed content outside the two exact bindings")
    return rendered


def expected_rendered() -> str:
    if sha256_file(CANONICAL) != CANONICAL_SHA:
        raise RuntimeError("canonical schema4 monitor SHA changed")
    return render_v18_monitor(CANONICAL.read_text())


def materialize(output: Path) -> str:
    rendered = expected_rendered()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != rendered:
            raise RuntimeError("stored v1.8 core9 monitor changed")
    else:
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered)
        temporary.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        os.replace(temporary, output)
    if output.stat().st_mode & 0o222:
        raise RuntimeError("v1.8 core9 monitor must be read-only")
    return sha256_file(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
