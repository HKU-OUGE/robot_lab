#!/usr/bin/env python3
"""Finalize immutable v1.5 preregistration after source-sidecar reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "tmp/highstep_student_recovery_v15_20260713"
BASE = WORKFLOW / "preregistration_v5.json"
OUTPUT = WORKFLOW / "preregistration_v6.json"
SOURCE_DIR = WORKFLOW / "source_model_172300_v3"
SOURCE = SOURCE_DIR / "model_172300.pt"
SOURCE_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
CODE = (
    ROOT / "scripts/rsl_rl/base/highstep_same_state_audit_play.py",
    ROOT / "scripts/rsl_rl/base/train.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__init__.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
    ROOT / "tools/highstep_same_state_audit.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable preregistration: {OUTPUT}")
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise RuntimeError("model_172300 source mirror SHA mismatch")
    payload = json.loads(BASE.read_text())
    payload["schema_version"] = 2
    payload["checkpoint_binding"].update({
        "canonical_protected_teacher_checkpoint": (
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
            "2026-07-11_11-22-23/model_172300.pt"
        ),
        "canonical_protected_teacher_sha256": SOURCE_SHA,
        "initial_student_checkpoint": str(SOURCE.resolve()),
        "teacher_checkpoint": str(SOURCE.resolve()),
        "read_only_byte_identical_source_mirror": True,
        "schedule_manifest": str((SOURCE_DIR / "highstep_schedule_manifest.json").resolve()),
        "schedule_manifest_sha256": sha256_file(SOURCE_DIR / "highstep_schedule_manifest.json"),
        "runtime_state": str((SOURCE_DIR / "highstep_runtime_state.json").resolve()),
        "runtime_state_sha256": sha256_file(SOURCE_DIR / "highstep_runtime_state.json"),
        "schedule_reconstruction_evidence": {
            "source_eval_manifest_sha256": "a2916f6ac354711313ebc2b693fbf30b1295bd3d29b8e1449db282ade9c18ba7",
            "source_events_sha256": "b3d9d93d9cfda3fd881e69d25b29bc9b4919efe83835dd0ace14039870a00427",
            "old_teacher_run_modified": False,
        },
    })
    payload["code_sha256"] = {str(path.resolve()): sha256_file(path) for path in CODE}
    payload["supersedes_preregistration"] = {
        "path": str(BASE.resolve()),
        "sha256": sha256_file(BASE),
        "reason": "Bind byte-identical source mirror plus reconstructed schedule/runtime state before any training.",
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, OUTPUT)
    OUTPUT.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({"path": str(OUTPUT.resolve()), "sha256": sha256_file(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
