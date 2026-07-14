#!/usr/bin/env python3
"""Record one SHA-bound 0707 Teacher/Student rollout with saved configs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
PLAY = Path(__file__).with_name("play.py")
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
PROFILES = {
    "teacher": {
        "task": "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0",
        "checkpoint": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
            "2026-07-04_01-45-21/model_151399.pt"
        ),
        "checkpoint_sha256": "d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d",
        "agent_sha256": "a39d614d055d43c53c3f9aead2166e4cee78567c2fec235f721725792bd1fda2",
        "env_sha256": "25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7",
    },
    "early": {
        "task": "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0",
        "checkpoint": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
            "student_no_prior_Student/2026-07-04_06-32-23/model_152100.pt"
        ),
        "checkpoint_sha256": "3c7a335bdc6c1b952e3b0ddbfdd35f0e4311ca1a9fb86cb9ae9a21de2f63a8d6",
        "agent_sha256": "c1941ecca30859aa7ad29fa8e2558bef099234beb83ec8a1e63e72549e1deed4",
        "env_sha256": "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636",
    },
    "deployed": {
        "task": "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0",
        "checkpoint": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
            "student_no_prior_Student/2026-07-05_00-13-46/model_158797.pt"
        ),
        "checkpoint_sha256": "7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284",
        "agent_sha256": "bb01ed1c7a8574363e9cea0d9d1dc4a0828d8453a097715ae2e4b49fa51be9ca",
        "env_sha256": "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _child(target: Any, key: str) -> Any:
    return target.get(key) if isinstance(target, dict) else getattr(target, key, None)


def _assign(target: Any, key: str, value: Any) -> None:
    if isinstance(target, dict):
        if key in target:
            target[key] = value
    elif hasattr(target, key):
        setattr(target, key, value)


def _restore(target: Any, source: Any) -> None:
    if not isinstance(source, dict) or target is None:
        return
    for key, value in source.items():
        if key in {"class_type", "func", "device", "num_envs", "seed"}:
            continue
        current = _child(target, key)
        if isinstance(value, dict) and current is not None:
            _restore(current, value)
        elif isinstance(value, (str, int, float, bool, tuple, list)) or value is None:
            if not callable(current):
                _assign(target, key, value)


def _profile() -> tuple[str, dict[str, Any]]:
    name = os.environ.get("HIGHSTEP_V15_VIDEO_PROFILE", "").strip().lower()
    if name not in PROFILES:
        raise RuntimeError("HIGHSTEP_V15_VIDEO_PROFILE must be teacher, early, or deployed")
    profile = dict(PROFILES[name])
    checkpoint = Path(profile["checkpoint"]).resolve(strict=True)
    profile["checkpoint"] = checkpoint
    profile["agent"] = checkpoint.parent / "params/agent.yaml"
    profile["env"] = checkpoint.parent / "params/env.yaml"
    expected = {
        SPEC: SPEC_SHA, checkpoint: profile["checkpoint_sha256"],
        profile["agent"]: profile["agent_sha256"], profile["env"]: profile["env_sha256"],
    }
    failures = [
        f"{path}: {sha256_file(path)} != {digest}"
        for path, digest in expected.items() if sha256_file(path) != digest
    ]
    if failures:
        raise RuntimeError("historical video authority mismatch: " + "; ".join(failures))
    return name, profile


def _load_yaml(path: Path) -> Mapping[str, Any]:
    import yaml
    return yaml.load(path.read_text(), Loader=yaml.UnsafeLoader)


def main() -> int:
    name, profile = _profile()
    # play.py launches Isaac Sim at import time, so install the fixed CLI first.
    sys.argv = [
        str(PLAY), "--headless", "--video", "--video_length", "600",
        "--num_envs", "1", "--task", profile["task"],
        "--checkpoint", str(profile["checkpoint"]), "--seed", "11",
        "--play_terrain_level", "9", "--play_terrain_type", "box_hard",
        "--reset_after_play_terrain_selection", "--front_step_eval_reset",
        "--front_step_eval_side", "x-", "--front_step_eval_edge_gap", "0.55",
        "--fixed_velocity_command", "0.45", "0.0", "0.0",
        "--highstep_gap_camera", "side_top", "--skip_policy_export",
        "--allow_legacy_highstep_schedule_fallback",
    ]
    spec = importlib.util.spec_from_file_location(f"highstep_v15_video_play_{name}", PLAY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PLAY}")
    play = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = play
    spec.loader.exec_module(play)

    saved_env = _load_yaml(Path(profile["env"]))
    saved_agent = _load_yaml(Path(profile["agent"]))
    original_env = play.parse_env_cfg
    original_agent = play.cli_args.parse_rsl_rl_cfg

    def parse_env_cfg(*args, **kwargs):
        cfg = original_env(*args, **kwargs)
        _restore(cfg.sim, saved_env.get("sim", {}))
        _restore(cfg.scene.robot.init_state, saved_env.get("scene", {}).get("robot", {}).get("init_state", {}))
        _restore(cfg.scene.robot.actuators, saved_env.get("scene", {}).get("robot", {}).get("actuators", {}))
        _restore(cfg.scene.terrain.physics_material, saved_env.get("scene", {}).get("terrain", {}).get("physics_material", {}))
        _restore(cfg.scene.terrain.terrain_generator, saved_env.get("scene", {}).get("terrain", {}).get("terrain_generator", {}))
        _restore(cfg.commands, saved_env.get("commands", {}))
        _restore(cfg.terminations, saved_env.get("terminations", {}))
        return cfg

    def parse_agent_cfg(*args, **kwargs):
        cfg = original_agent(*args, **kwargs)
        _restore(cfg, saved_agent)
        return cfg

    play.parse_env_cfg = parse_env_cfg
    play.cli_args.parse_rsl_rl_cfg = parse_agent_cfg
    try:
        play.main()
    finally:
        play.simulation_app.close()
    print(json.dumps({
        "profile": name, "checkpoint": str(profile["checkpoint"]),
        "checkpoint_sha256": profile["checkpoint_sha256"],
        "saved_agent_sha256": profile["agent_sha256"],
        "saved_env_sha256": profile["env_sha256"],
        "camera": "side_top", "seed": 11, "video_length": 600,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
