from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts/rsl_rl/base/highstep_same_state_audit_play.py"


def _load_runtime():
    spec = importlib.util.spec_from_file_location("highstep_same_state_audit_play", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cpu_import_has_no_isaac_or_torch_side_effects():
    source = RUNTIME_PATH.read_text()
    tree = ast.parse(source)
    top_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_from_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert "torch" not in top_imports
    assert not any(name and name.startswith("isaaclab") for name in top_from_imports)
    runtime = _load_runtime()
    assert runtime.MAX_STEPS == 600


def test_fixed_three_trajectories_match_formal_base_suite():
    runtime = _load_runtime()
    assert {name: value.__dict__ for name, value in runtime.TRAJECTORIES.items()} == {
        "seed11_nominal": {
            "run_id": "seed11_nominal",
            "seed": 11,
            "scenario": "nominal",
            "lateral_offset_m": 0.0,
            "yaw_offset_deg": 0.0,
        },
        "seed11_left_offset": {
            "run_id": "seed11_left_offset",
            "seed": 11,
            "scenario": "left_offset",
            "lateral_offset_m": 0.12,
            "yaw_offset_deg": 4.0,
        },
        "seed22_right_offset": {
            "run_id": "seed22_right_offset",
            "seed": 22,
            "scenario": "right_offset",
            "lateral_offset_m": -0.12,
            "yaw_offset_deg": -4.0,
        },
    }


def test_core9_contract_is_sha_bound_and_extracts_only_required_nodes():
    runtime = _load_runtime()
    module = runtime._play_contract_ast()
    names = [node.name for node in module.body]
    assert set(names) == set(runtime.PLAY_CONTRACT_NODES)
    assert len(names) == len(runtime.PLAY_CONTRACT_NODES)
    assert runtime.sha256_file(runtime.PLAY_PATH) == runtime.PLAY_SHA256


def test_fixed_cli_rejects_task_env_checkpoint_and_gui_drift(monkeypatch):
    runtime = _load_runtime()
    monkeypatch.setattr(runtime, "sha256_file", lambda _: runtime.B500_SHA256)
    base = dict(
        trajectory="seed11_nominal",
        checkpoint=str(runtime.B500_CHECKPOINT),
        task=runtime.TASK,
        num_envs=1,
        headless=True,
    )
    assert runtime._validate_fixed_cli(argparse.Namespace(**base)).run_id == "seed11_nominal"
    for field, bad in (
        ("task", "wrong-task"),
        ("num_envs", 2),
        ("checkpoint", str(runtime.B500_CHECKPOINT.parent / "model_997.pt")),
        ("headless", False),
    ):
        values = {**base, field: bad}
        with pytest.raises((ValueError, FileNotFoundError)):
            runtime._validate_fixed_cli(argparse.Namespace(**values))


def test_suite_plan_is_written_once_read_only_and_fails_on_drift(tmp_path):
    runtime = _load_runtime()

    class Audit:
        @staticmethod
        def base_suite_plan_payload():
            return {"schema_version": 1, "kind": "test", "trajectories": [1, 2, 3]}

    path, digest = runtime._ensure_base_suite_plan(tmp_path, Audit)
    assert path == tmp_path / "base3_suite_plan.json"
    assert len(digest) == 64
    assert path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
    assert json.loads(path.read_text()) == Audit.base_suite_plan_payload()
    assert runtime._ensure_base_suite_plan(tmp_path, Audit) == (path, digest)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(RuntimeError, match="writable"):
        runtime._ensure_base_suite_plan(tmp_path, Audit)


def test_loop_has_single_pre_step_session_action_path_and_no_direct_policy_step():
    source = RUNTIME_PATH.read_text()
    tree = ast.parse(source)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    eval_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate_runtime_frame"
    ]
    step_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "step"
    ]
    assert len(eval_calls) == 1
    assert len(step_calls) == 1
    assert isinstance(step_calls[0].args[0], ast.Name)
    assert step_calls[0].args[0].id == "action_to_apply"
    assert eval_calls[0].lineno < step_calls[0].lineno
    assert "actions = policy(" not in source


def test_runtime_contract_constants_are_exact():
    runtime = _load_runtime()
    assert runtime.TASK.endswith("RobustStudentNoPrior-ArcdogAdjustableLeg-v0")
    assert runtime.NUM_ENVS == 1
    assert runtime.TERRAIN_TYPE == "box"
    assert runtime.TERRAIN_LEVEL == 9
    assert runtime.FIXED_COMMAND == (0.45, 0.0, 0.0)
    assert runtime.FRONT_STEP_SIDE == "x-"
    assert runtime.FRONT_STEP_EDGE_GAP == 0.55
    assert runtime.ACTION_DELAY_STEPS == 0
    assert runtime.MAX_STEPS == 600
    assert runtime.REAL_GAIN_CONTRACT == {
        "legs_hip": (55.0, 1.5),
        "legs_thigh": (65.0, 1.5),
        "legs_calf": (80.0, 2.5),
    }


def test_observation_semantics_bind_570_570_and_critic_3_plus_159():
    runtime = _load_runtime()
    names = {
        group: list(contract["names"])
        for group, contract in runtime.OBSERVATION_CONTRACT.items()
    }
    dims = {
        group: [(value,) for value in contract["dims"]]
        for group, contract in runtime.OBSERVATION_CONTRACT.items()
    }
    cfgs = {}
    for group, contract in runtime.OBSERVATION_CONTRACT.items():
        cfgs[group] = []
        for name, history, scale, function in zip(
            contract["names"],
            contract["history_lengths"],
            contract["scales"],
            contract["functions"],
        ):
            if function == "fixed_velocity_command_override":
                module_name, function_name = "test", "fixed"
            else:
                module_name, function_name = function.split(":")
            func = lambda: None
            func.__module__ = module_name
            func.__name__ = function_name
            params = {}
            if name in {"joint_pos", "joint_vel"}:
                params["asset_cfg"] = SimpleNamespace(joint_names=runtime.ACTION_JOINT_ORDER)
            cfgs[group].append(
                SimpleNamespace(
                    history_length=history,
                    flatten_history_dim=True,
                    scale=(
                        None
                        if name == "velocity_commands"
                        else 0.05000000074505806
                        if name == "joint_vel" and scale == 0.05
                        else scale
                    ),
                    func=func,
                    params=params,
                )
            )
    manager = SimpleNamespace(
        _group_obs_term_names=names,
        _group_obs_term_dim=dims,
        _group_obs_term_cfgs=cfgs,
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(observation_manager=manager))
    observations = {
        "policy": SimpleNamespace(shape=(1, 570)),
        "estimator": SimpleNamespace(shape=(1, 570)),
        "critic": SimpleNamespace(shape=(1, 162)),
    }
    evidence = runtime._assert_observation_semantics(env, observations)
    assert evidence["teacher_critic_privileged_dim"] == 159
    assert evidence["groups"]["critic"]["flattened_term_dims"][0] == 3
    manager._group_obs_term_cfgs["policy"][4].scale = 0.051
    with pytest.raises(RuntimeError, match="scale mismatch"):
        runtime._assert_observation_semantics(env, observations)
    manager._group_obs_term_cfgs["policy"][4].scale = 0.05000000074505806
    manager._group_obs_term_names["critic"][0] = "wrong_velocity_label"
    with pytest.raises(RuntimeError, match="term order"):
        runtime._assert_observation_semantics(env, observations)


def test_runtime_records_aggregate_required_contract_fields():
    source = RUNTIME_PATH.read_text()
    for token in (
        '"same_state_runtime_contract_verified": True',
        '"action_prior_enabled": False',
        '"keep_play_randomization": False',
        '"fixed_velocity_command": list(FIXED_COMMAND)',
        '"loop_steps": timestep',
    ):
        assert token in source


def test_runtime_parser_defines_fabric_switch_used_by_environment_creation():
    source = RUNTIME_PATH.read_text()
    assert '"--disable_fabric"' in source
    assert "use_fabric=not args.disable_fabric" in source


def test_runtime_registers_custom_algorithm_in_runner_eval_namespace():
    source = RUNTIME_PATH.read_text()
    assert "rsl_runner.VAEPPO = VAEPPO" in source
