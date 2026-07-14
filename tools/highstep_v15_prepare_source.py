#!/usr/bin/env python3
"""Create an immutable model_172300 mirror with evidence-backed schedule sidecars."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "tmp/highstep_student_recovery_v15_20260713"
SOURCE = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
SOURCE_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
EVAL_MANIFEST = SOURCE.parent / (
    "eval_schedule_manifests/20260713_145522_360964_highstep_schedule_manifest.json"
)
EVAL_MANIFEST_SHA = "a2916f6ac354711313ebc2b693fbf30b1295bd3d29b8e1449db282ade9c18ba7"
EVENTS = SOURCE.parent / "events.out.tfevents.1783740152.lxq-MS-7E30.2795598.0"
EVENTS_SHA = "b3d9d93d9cfda3fd881e69d25b29bc9b4919efe83835dd0ace14039870a00427"
TARGET_DIR = WORKFLOW / "source_model_172300_v3"
TARGET = TARGET_DIR / "model_172300.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v1.5 source artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main() -> int:
    for path, expected in (
        (SOURCE, SOURCE_SHA), (EVAL_MANIFEST, EVAL_MANIFEST_SHA), (EVENTS, EVENTS_SHA)
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"v1.5 source evidence changed: {path}: {actual} != {expected}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        if sha256_file(TARGET) != SOURCE_SHA:
            raise RuntimeError("existing model_172300 mirror has wrong SHA")
    else:
        shutil.copy2(SOURCE, TARGET)
        TARGET.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    evaluation = json.loads(EVAL_MANIFEST.read_text())
    schedule = copy.deepcopy(evaluation)
    schedule.update({
        "schema_version": max(4, int(evaluation.get("schema_version", 0))),
        "kind": "highstep_v15_reconstructed_train_source_schedule",
        "context": "train",
        "task": "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0",
        "checkpoint_path": str(TARGET.resolve()),
        "checkpoint_sha256": SOURCE_SHA,
        "checkpoint_iteration": 172300,
        "runner_iteration_at_anchor": 172300,
        "schedule_update_at_anchor": 172300.0,
        "runtime_state_required_for_preserve": True,
        "reconstruction": {
            "canonical_teacher_checkpoint": str(SOURCE.resolve()),
            "canonical_teacher_checkpoint_sha256": SOURCE_SHA,
            "source_eval_manifest": str(EVAL_MANIFEST.resolve()),
            "source_eval_manifest_sha256": EVAL_MANIFEST_SHA,
            "source_events": str(EVENTS.resolve()),
            "source_events_sha256": EVENTS_SHA,
            "old_teacher_run_modified": False,
            "reason": "The protected legacy Teacher predates checkpoint-paired train sidecars. Its latest exact-checkpoint v1.5-compatible evaluation manifest supplies the Teacher schedule definition and anchor; the source run's first logged step supplies reset-time adaptive state. The source task remains Teacher so Student lineage is resolved only from the separately pinned parent manifest.",
        },
    })
    # Cross-task initialization is intentionally StudentNoPrior: the frozen
    # Teacher prior remains solely in the post-prior supervision target, while
    # the live Student environment has no action-prior schedule.
    schedule["action_prior"] = {
        "enabled": False,
        "actual_prior_scale": 0.0,
        "start_update": None,
        "full_update": None,
        "num_steps_per_update": 24,
    }
    schedule["schedule_definition"]["action_prior"] = {
        "enabled": False,
        "start_update": None,
        "full_update": None,
        "num_steps_per_update": 24,
    }
    schedule["reconstruction"]["cross_task_schedule_change"] = (
        "Teacher action-prior schedule -> StudentNoPrior disabled; Teacher post-prior remains the 16-D supervision target"
    )

    command = {
        "enabled": True,
        "original_vel_x": [-0.18, 0.72],
        "original_vel_y": [-0.06, 0.06],
        "original_yaw": [-0.55, 0.55],
        "initial_vel_x": [-0.081, 0.324],
        "initial_vel_y": [-0.027, 0.027],
        "initial_yaw": [-0.2475, 0.2475],
        "final_vel_x": [-0.162, 0.648],
        "final_vel_y": [-0.054, 0.054],
        "final_yaw": [-0.495, 0.495],
        "gated_vel_x": [-0.135, 0.54],
        "gated_vel_y": [-0.045, 0.045],
        "gated_yaw": [-0.4125, 0.4125],
        "current_lin_vel_x": [-0.081, 0.324],
        "current_lin_vel_y": [-0.027, 0.027],
        "current_ang_vel_z": [-0.2475, 0.2475],
    }
    runtime = {
        "schema_version": 2,
        "context": "train",
        "snapshots": [{
            "checkpoint_file": TARGET.name,
            "checkpoint_sha256": SOURCE_SHA,
            "runner_iteration": 172300,
            "schedule_update": 172300.0,
            "local_common_step": 0,
            "command_curriculum": command,
            "moving_best": {
                "required": True,
                "observed": True,
                "action_score_best": 0.0,
                "support_score_best": 0.0003878512652590871,
            },
            "reconstruction_evidence": {
                "event_step": 172300,
                "command_levels": 0.3240000009536743,
                "action_score_total": 0.0,
                "support_score": 0.0003878512652590871,
                "events_sha256": EVENTS_SHA,
            },
        }],
    }
    _write(TARGET_DIR / "highstep_schedule_manifest.json", schedule)
    _write(TARGET_DIR / "highstep_runtime_state.json", runtime)
    result = {
        "checkpoint": str(TARGET.resolve()),
        "checkpoint_sha256": sha256_file(TARGET),
        "schedule_manifest": str((TARGET_DIR / "highstep_schedule_manifest.json").resolve()),
        "schedule_manifest_sha256": sha256_file(TARGET_DIR / "highstep_schedule_manifest.json"),
        "runtime_state": str((TARGET_DIR / "highstep_runtime_state.json").resolve()),
        "runtime_state_sha256": sha256_file(TARGET_DIR / "highstep_runtime_state.json"),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
