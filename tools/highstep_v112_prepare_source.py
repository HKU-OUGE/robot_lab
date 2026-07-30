#!/usr/bin/env python3
"""Create the immutable v1.12 model_172300 mirror with exact schedule sidecars."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

from isaaclab.app import AppLauncher


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORK = ROOT / "tmp/highstep_teacher_rear_support_v112_20260715"
CANONICAL = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
CANONICAL_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
PARENT = ROOT / "tmp/highstep_student_recovery_v15_20260713/source_model_172300_v2"
PARENT_SCHEDULE_SHA = "1bff13affba3642d01b15267e4199bfe6ac7d6a3772d17d937e7075cfc776742"
PARENT_RUNTIME_SHA = "b0d413f2edd1d9cc7cec01467112027c0b580614c537b0382770777e260668e4"
TARGET_DIR = WORK / "source_model_172300_v112"
TARGET = TARGET_DIR / "model_172300.pt"
TASK = "RobotLab-Isaac-Velocity-HighstepRearSupportV112-ArcdogAdjustableLeg-v0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_read_only(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v1.12 source artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def reward_stage_definition(env_cfg) -> dict[str, dict[str, int]]:
    definition: dict[str, dict[str, int]] = {}
    for name in dir(env_cfg.rewards):
        if name.startswith("_"):
            continue
        term_cfg = getattr(env_cfg.rewards, name, None)
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict) or "stage_start_update" not in params:
            continue
        definition[name] = {
            "stage_start_update": int(params["stage_start_update"]),
            "stage_ramp_updates": int(params.get("stage_ramp_updates", 1)),
            "num_steps_per_update": int(params.get("num_steps_per_update", 24)),
        }
    return dict(sorted(definition.items()))


def main() -> int:
    for path, expected in (
        (CANONICAL, CANONICAL_SHA),
        (PARENT / "model_172300.pt", CANONICAL_SHA),
        (PARENT / "highstep_schedule_manifest.json", PARENT_SCHEDULE_SHA),
        (PARENT / "highstep_runtime_state.json", PARENT_RUNTIME_SHA),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"v1.12 source evidence changed: {path}: {actual} != {expected}")

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.highstep_env_cfg import (
        ArclabArcdogAdjustableLegHighstepRearSupportV112EnvCfg,
    )
    from robot_lab.tasks.locomotion.velocity.mdp.highstep_schedule import (
        schedule_definition_from_configs,
    )

    env_cfg = ArclabArcdogAdjustableLegHighstepRearSupportV112EnvCfg()
    current_definition = schedule_definition_from_configs(
        action_cfg=env_cfg.actions.joint_pos,
        terrain_term_cfg=env_cfg.curriculum.terrain_levels,
        support_term_cfg=env_cfg.curriculum.highstep_action_score,
        command_term_cfg=env_cfg.curriculum.command_levels,
        command_curriculum_enabled=env_cfg.curriculum.command_levels is not None,
        reward_stage_definition=reward_stage_definition(env_cfg),
    )

    parent_schedule = json.loads((PARENT / "highstep_schedule_manifest.json").read_text())
    parent_definition = parent_schedule["schedule_definition"]
    for section in ("action_prior", "terrain_schedule", "support_bottleneck", "command_curriculum"):
        if current_definition[section] != parent_definition[section]:
            raise RuntimeError(f"v1.12 changed a forbidden schedule section: {section}")
    parent_rewards = dict(parent_definition["reward_stages"])
    current_rewards = dict(current_definition["reward_stages"])
    added = sorted(set(current_rewards) - set(parent_rewards))
    removed = sorted(set(parent_rewards) - set(current_rewards))
    changed_existing = sorted(
        name
        for name in set(parent_rewards) & set(current_rewards)
        if parent_rewards[name] != current_rewards[name]
    )
    if added != ["rear_support_motion_contract"] or removed or changed_existing:
        raise RuntimeError(
            "v1.12 schedule migration is not the one approved reward-stage addition: "
            f"added={added}, removed={removed}, changed={changed_existing}"
        )

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite existing v1.12 mirror: {TARGET}")
    shutil.copy2(CANONICAL, TARGET)
    TARGET.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    schedule = copy.deepcopy(parent_schedule)
    schedule.update(
        {
            "kind": "highstep_v112_rear_support_train_source_schedule_migration",
            "task": TASK,
            "checkpoint_path": str(TARGET.resolve()),
            "checkpoint_sha256": CANONICAL_SHA,
            "schedule_definition": current_definition,
        }
    )
    schedule["reconstruction"]["v112_migration"] = {
        "parent_schedule": str((PARENT / "highstep_schedule_manifest.json").resolve()),
        "parent_schedule_sha256": PARENT_SCHEDULE_SHA,
        "canonical_checkpoint": str(CANONICAL.resolve()),
        "canonical_checkpoint_sha256": CANONICAL_SHA,
        "added_reward_stage": "rear_support_motion_contract",
        "removed_reward_stages": [],
        "changed_existing_reward_stages": [],
        "all_non_reward_schedule_sections_byte_equivalent": True,
        "schedule_reset": False,
    }

    runtime = copy.deepcopy(json.loads((PARENT / "highstep_runtime_state.json").read_text()))
    runtime["v112_migration"] = {
        "parent_runtime_state": str((PARENT / "highstep_runtime_state.json").resolve()),
        "parent_runtime_state_sha256": PARENT_RUNTIME_SHA,
        "command_curriculum_and_moving_best_changed": False,
        "schedule_reset": False,
    }
    atomic_read_only(TARGET_DIR / "highstep_schedule_manifest.json", schedule)
    atomic_read_only(TARGET_DIR / "highstep_runtime_state.json", runtime)
    result = {
        "schema_version": 1,
        "workflow_id": "highstep_teacher_rear_support_v112_20260715",
        "checkpoint": str(TARGET.resolve()),
        "checkpoint_sha256": sha256_file(TARGET),
        "schedule_manifest": str((TARGET_DIR / "highstep_schedule_manifest.json").resolve()),
        "schedule_manifest_sha256": sha256_file(TARGET_DIR / "highstep_schedule_manifest.json"),
        "runtime_state": str((TARGET_DIR / "highstep_runtime_state.json").resolve()),
        "runtime_state_sha256": sha256_file(TARGET_DIR / "highstep_runtime_state.json"),
        "current_schedule_definition": current_definition,
        "only_schedule_definition_change": "add rear_support_motion_contract stage 0/1/24",
    }
    atomic_read_only(TARGET_DIR / "source_migration_manifest.json", result)
    print(json.dumps(result, sort_keys=True))
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
