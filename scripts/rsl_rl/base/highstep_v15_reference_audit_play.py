#!/usr/bin/env python3
"""Collect one frozen 0707 historical Student/Teacher same-state reference trace."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
BASE_RUNTIME = Path(__file__).with_name("highstep_same_state_audit_play.py")
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
WORKFLOW = ROOT / "tmp/highstep_student_recovery_v15_20260713"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-04_01-45-21/model_151399.pt"
)
TEACHER_SHA = "d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d"
TEACHER_AGENT = TEACHER.parent / "params/agent.yaml"
TEACHER_AGENT_SHA = "a39d614d055d43c53c3f9aead2166e4cee78567c2fec235f721725792bd1fda2"
TEACHER_ENV = TEACHER.parent / "params/env.yaml"
TEACHER_ENV_SHA = "25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7"
SPEC_SHA = "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0"
MAX_STEPS = 600
HISTORICAL_TERRAIN_TYPE = "box_hard"

PROFILES = {
    "early": {
        "checkpoint": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
            "student_no_prior_Student/2026-07-04_06-32-23/model_152100.pt"
        ),
        "checkpoint_sha256": "3c7a335bdc6c1b952e3b0ddbfdd35f0e4311ca1a9fb86cb9ae9a21de2f63a8d6",
        "agent_sha256": "c1941ecca30859aa7ad29fa8e2558bef099234beb83ec8a1e63e72549e1deed4",
        "env_sha256": "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636",
    },
    "deployed": {
        "checkpoint": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
            "student_no_prior_Student/2026-07-05_00-13-46/model_158797.pt"
        ),
        "checkpoint_sha256": "7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284",
        "agent_sha256": "bb01ed1c7a8574363e9cea0d9d1dc4a0828d8453a097715ae2e4b49fa51be9ca",
        "env_sha256": "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636",
    },
}
DRIVER = PROFILES["deployed"]


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_read_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _profile() -> tuple[str, dict[str, Any]]:
    name = os.environ.get("HIGHSTEP_V15_REFERENCE_PROFILE", "").strip().lower()
    if name not in PROFILES:
        raise RuntimeError("HIGHSTEP_V15_REFERENCE_PROFILE must be early or deployed")
    profile = dict(PROFILES[name])
    checkpoint = Path(profile["checkpoint"]).resolve(strict=True)
    profile["checkpoint"] = checkpoint
    profile["agent"] = checkpoint.parent / "params/agent.yaml"
    profile["env"] = checkpoint.parent / "params/env.yaml"
    expected = {
        SPEC: SPEC_SHA,
        TEACHER: TEACHER_SHA,
        TEACHER_AGENT: TEACHER_AGENT_SHA,
        TEACHER_ENV: TEACHER_ENV_SHA,
        checkpoint: profile["checkpoint_sha256"],
        profile["agent"]: profile["agent_sha256"],
        profile["env"]: profile["env_sha256"],
    }
    failures = [
        f"{path}: {sha256_file(path)} != {digest}"
        for path, digest in expected.items() if sha256_file(path) != digest
    ]
    if failures:
        raise RuntimeError("v1.5 historical authority mismatch: " + "; ".join(failures))
    return name, profile


def _apply_saved_runtime_semantics(env_cfg: Any, saved_env: Path) -> None:
    """Restore state-affecting historical values before the deterministic reset."""
    import yaml

    payload = yaml.load(saved_env.read_text(), Loader=yaml.UnsafeLoader)

    def child(target: Any, key: str) -> Any:
        return target.get(key) if isinstance(target, dict) else getattr(target, key, None)

    def assign(target: Any, key: str, value: Any) -> None:
        if isinstance(target, dict):
            if key in target:
                target[key] = value
        elif hasattr(target, key):
            setattr(target, key, value)

    def restore(target: Any, source: Any) -> None:
        if not isinstance(source, dict) or target is None:
            return
        for key, value in source.items():
            if key in {"class_type", "func", "device", "num_envs", "seed"}:
                continue
            current = child(target, key)
            if isinstance(value, dict) and current is not None:
                restore(current, value)
            elif isinstance(value, (str, int, float, bool, tuple, list)) or value is None:
                # Callable/function-valued fields are intentionally SHA-checked,
                # not replaced by their serialized string representation.
                if callable(current):
                    continue
                assign(target, key, value)

    restore(env_cfg.sim, payload.get("sim", {}))
    scene = payload.get("scene", {})
    restore(env_cfg.scene.robot.init_state, scene.get("robot", {}).get("init_state", {}))
    restore(env_cfg.scene.robot.actuators, scene.get("robot", {}).get("actuators", {}))
    terrain_saved = scene.get("terrain", {})
    restore(env_cfg.scene.terrain.physics_material, terrain_saved.get("physics_material", {}))
    restore(env_cfg.scene.terrain.terrain_generator, terrain_saved.get("terrain_generator", {}))
    restore(env_cfg.commands, payload.get("commands", {}))
    restore(env_cfg.terminations, payload.get("terminations", {}))


def _install(base: Any, audit: Any, name: str, profile: Mapping[str, Any]) -> None:
    checkpoint = Path(profile["checkpoint"])
    driver_checkpoint = Path(DRIVER["checkpoint"])
    driver_agent = driver_checkpoint.parent / "params/agent.yaml"
    driver_env = driver_checkpoint.parent / "params/env.yaml"
    base.PLAY_SHA256 = sha256_file(base.PLAY_PATH)
    base.TASK = TASK
    base.TERRAIN_TYPE = HISTORICAL_TERRAIN_TYPE
    base.AUDIT_DRIVER_ACTION_SOURCE = "runner"
    base.B500_CHECKPOINT = driver_checkpoint
    base.B500_SHA256 = str(DRIVER["checkpoint_sha256"])
    base.B500_AGENT_CONFIG = driver_agent
    base.B500_AGENT_CONFIG_SHA256 = str(DRIVER["agent_sha256"])
    base.B500_ENV_CONFIG = driver_env
    base.B500_ENV_CONFIG_SHA256 = str(DRIVER["env_sha256"])
    base.REAL_GAIN_CONTRACT = {
        "legs_hip": (45.0, 1.5),
        "legs_thigh": (50.0, 1.5),
        "legs_calf": (60.0, 2.0),
    }
    for group_name in ("policy", "estimator"):
        contract = dict(base.OBSERVATION_CONTRACT[group_name])
        functions = list(contract["functions"])
        functions[3] = "isaaclab.envs.mdp.observations:joint_pos_rel"
        contract["functions"] = tuple(functions)
        base.OBSERVATION_CONTRACT[group_name] = contract

    rows = tuple(
        {
            "run_id": f"seed{seed}_nominal",
            "seed": seed,
            "scenario": "nominal",
            "lateral_offset_m": 0.0,
            "yaw_offset_deg": 0.0,
            "required_phases": audit.PHASES,
            "forbidden_phases": (),
            "expected_full_climb_success": None,
            "expected_rear_hold_success": None,
        }
        for seed in (11, 22, 33)
    )
    audit.DEFAULT_AUDIT_TRAJECTORIES = rows
    base.TRAJECTORIES = {
        row["run_id"]: base.Trajectory(
            row["run_id"], row["seed"], row["scenario"], 0.0, 0.0
        )
        for row in rows
    }

    plan_payload = {
        "schema_version": 1,
        "kind": "highstep_v15_0707_historical_pair_plan",
        "immutable_after_write": True,
        "workflow_id": "highstep_student_recovery_v15_20260713",
        "profile": name,
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": profile["checkpoint_sha256"],
        "shared_state_driver_checkpoint": str(driver_checkpoint),
        "shared_state_driver_checkpoint_sha256": DRIVER["checkpoint_sha256"],
        "teacher_checkpoint": str(TEACHER.resolve()),
        "teacher_checkpoint_sha256": TEACHER_SHA,
        "saved_student_agent_config": str(Path(profile["agent"]).resolve()),
        "saved_student_agent_config_sha256": profile["agent_sha256"],
        "saved_student_env_config": str(Path(profile["env"]).resolve()),
        "saved_student_env_config_sha256": profile["env_sha256"],
        "saved_teacher_agent_config": str(TEACHER_AGENT.resolve()),
        "saved_teacher_agent_config_sha256": TEACHER_AGENT_SHA,
        "saved_teacher_env_config": str(TEACHER_ENV.resolve()),
        "saved_teacher_env_config_sha256": TEACHER_ENV_SHA,
        "runtime_contract": {
            "task": TASK,
            "num_envs": 1,
            "terrain_type": HISTORICAL_TERRAIN_TYPE,
            "terrain_level": 9,
            "fixed_velocity_command": [0.45, 0.0, 0.0],
            "front_step_eval_side": "x-",
            "front_step_eval_edge_gap": 0.55,
            "play_max_steps": MAX_STEPS,
            "eval_action_delay_steps": 0,
            "observation_corruption_enabled": False,
            "play_random_events_enabled": False,
            "real_gain_center": {
                "hip": [45.0, 1.5], "thigh": [50.0, 1.5], "calf": [60.0, 2.0]
            },
            "student_action_prior_enabled": False,
            "causal_replacement": None,
        },
        "teacher_post_prior_from_saved_config": True,
        "historical_schedule_update": int(driver_checkpoint.stem.split("_")[-1]),
        "trajectories": rows,
        "training_allowed": False,
    }
    audit.base_suite_plan_payload = lambda: copy.deepcopy(plan_payload)

    class PairModels(audit.HistoricalPairedAuditModels):
        def __init__(self, *, device="cpu"):
            super().__init__(
                student_checkpoint=checkpoint,
                student_sha256=str(profile["checkpoint_sha256"]),
                teacher_checkpoint=TEACHER,
                teacher_sha256=TEACHER_SHA,
                student_agent_config=Path(profile["agent"]),
                student_agent_config_sha256=str(profile["agent_sha256"]),
                student_env_config=Path(profile["env"]),
                student_env_config_sha256=str(profile["env_sha256"]),
                teacher_agent_config=TEACHER_AGENT,
                teacher_agent_config_sha256=TEACHER_AGENT_SHA,
                teacher_env_config=TEACHER_ENV,
                teacher_env_config_sha256=TEACHER_ENV_SHA,
                device=device,
            )

    audit.BoundAuditModels = PairModels
    original_contract = audit.PostPriorContract.from_teacher_env_config.__func__

    def historical_contract(cls, path=TEACHER_ENV):
        return original_contract(
            cls,
            TEACHER_ENV,
            expected_sha256=TEACHER_ENV_SHA,
            update_count=151399,
        )

    audit.PostPriorContract.from_teacher_env_config = classmethod(historical_contract)
    original_gain_check = base._assert_real_gain_config

    def restore_then_check(env_cfg):
        _apply_saved_runtime_semantics(env_cfg, driver_env)
        # The historical JointPositionAction has no delay buffer; that is
        # semantically an exact fixed delay of zero.  These audit-only config
        # witnesses let the shared runtime assert the same contract without
        # replacing the historical action class.
        env_cfg.actions.joint_pos.min_action_delay_steps = 0
        env_cfg.actions.joint_pos.max_action_delay_steps = 0
        return original_gain_check(env_cfg)

    base._assert_real_gain_config = restore_then_check

    def runtime_hashes() -> dict[str, str]:
        paths = [
            Path(__file__).resolve(), BASE_RUNTIME.resolve(), SPEC.resolve(),
            ROOT / "tools/highstep_same_state_audit.py", checkpoint,
            Path(profile["agent"]), Path(profile["env"]), driver_checkpoint,
            driver_agent, driver_env,
            TEACHER, TEACHER_AGENT, TEACHER_ENV,
        ]
        return {str(path): sha256_file(path) for path in paths}

    base._runtime_code_hashes = runtime_hashes


def _install_historical_schedule(profile_name: str, profile: Mapping[str, Any]) -> Path:
    from robot_lab.tasks.locomotion.velocity.mdp import highstep_schedule as schedule

    checkpoint = Path(DRIVER["checkpoint"])
    schedule_update = float(int(checkpoint.stem.split("_")[-1]))
    manifest_path = (
        WORKFLOW / "0707_reference" / f"shared_driver_v2_{profile_name}"
        / "historical_schedule_reconstruction.json"
    )
    payload = {
        "schema_version": 1,
        "kind": "highstep_v15_historical_schedule_reconstruction",
        "profile": profile_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": profile["checkpoint_sha256"],
        "schedule_update": schedule_update,
        "shared_state_driver": "deployed_0707_student",
        "shared_state_driver_checkpoint_sha256": DRIVER["checkpoint_sha256"],
        "student_under_audit_checkpoint_sha256": profile["checkpoint_sha256"],
        "student_agent_config_sha256": DRIVER["agent_sha256"],
        "student_env_config_sha256": DRIVER["env_sha256"],
        "teacher_agent_config_sha256": TEACHER_AGENT_SHA,
        "teacher_env_config_sha256": TEACHER_ENV_SHA,
        "reason": "The 0707-era run predates schedule sidecars; its saved configs and checkpoint iteration reconstruct the historical fully-ramped schedule without altering the old run.",
        "old_run_modified": False,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != payload:
            raise RuntimeError("historical schedule reconstruction changed")
    else:
        _atomic_read_only_json(manifest_path, payload)

    def resolve_loaded(*args, **kwargs):
        return schedule_update, {
            "method": "v15_historical_saved_config_iteration_reconstruction",
            "source_manifest": str(manifest_path),
            "checkpoint_iteration": int(schedule_update),
            "legacy_fallback_allowed": False,
            "legacy_fallback_used": False,
        }

    def load_source(*args, **kwargs):
        return manifest_path, payload

    def assert_compatible(source, current):
        driver_env = Path(DRIVER["checkpoint"]).parent / "params/env.yaml"
        if source != payload or sha256_file(driver_env) != DRIVER["env_sha256"]:
            raise RuntimeError("historical schedule/config authority changed")

    schedule.resolve_loaded_schedule_update = resolve_loaded
    schedule.load_source_manifest = load_source
    schedule.assert_schedule_definition_compatible = assert_compatible
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    profile_name, profile = _profile()
    sys.path.insert(0, str(ROOT))
    from tools import highstep_same_state_audit as audit

    base = _load(BASE_RUNTIME, f"highstep_v15_reference_base_{profile_name}")
    _install(base, audit, profile_name, profile)
    from isaaclab.app import AppLauncher
    parser = base._build_parser()
    args = parser.parse_args(argv)
    trajectory = base._validate_fixed_cli(args)
    args.seed = trajectory.seed
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        _install_historical_schedule(profile_name, profile)
        return base._run(args, trajectory, app)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
