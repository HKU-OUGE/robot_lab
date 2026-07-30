from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py"
ENV_CFG = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/highstep_env_cfg.py"
)
RUNNER_CFG = ENV_CFG.parent / "agents/rsl_rl_ppo_cfg.py"
REGISTRY = ENV_CFG.parent / "__init__.py"
TRAIN = ROOT / "scripts/rsl_rl/base/train.py"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_block(path: Path, name: str) -> str:
    module = ast.parse(_source(path))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(_source(path), node) or ""
    raise AssertionError(f"function not found: {name}")


def _load_latch_function():
    namespace = {"torch": torch, "ManagerBasedRLEnv": object}
    exec(_function_block(REWARDS, "_highstep_monotonic_reward_release_latch"), namespace)
    return namespace["_highstep_monotonic_reward_release_latch"]


def test_v1123_python_files_parse() -> None:
    for path in (REWARDS, ENV_CFG, RUNNER_CFG, REGISTRY, TRAIN):
        ast.parse(_source(path), filename=str(path))


def test_v1123_geometry_formula_is_exact_and_fl_only() -> None:
    block = _function_block(
        REWARDS, "left_front_highstep_precontact_reward_geometry_contract_bonus"
    )
    for fragment in (
        "low_plane_z = front_terrain_z - detected_step_height",
        "low_plane_clearance = fl_pos_w[:, 2] - low_plane_z",
        "(low_plane_clearance - low_lift_min)",
        "-0.5 - torch.square((fl_rel_b[:, 0] - retraction_target_x)",
        "fl_clearance >= clearance_release",
        "* active_latch",
    ):
        assert fragment in block
    assert "right_front_foot_name" not in block
    assert "observation" not in block
    assert "action_manager" not in block
    assert "checkpoint" not in block


def test_v1123_latch_is_per_env_monotonic_and_resettable() -> None:
    latch = _load_latch_function()
    env = SimpleNamespace(episode_length_buf=torch.tensor([10, 10, 10]))
    assert latch(env, torch.tensor([False, True, False])).tolist() == [1.0, 0.0, 1.0]

    env.episode_length_buf = torch.tensor([11, 11, 11])
    assert latch(env, torch.tensor([False, False, False])).tolist() == [1.0, 0.0, 1.0]
    env.episode_length_buf = torch.tensor([12, 12, 12])
    assert latch(env, torch.tensor([True, False, False])).tolist() == [0.0, 0.0, 1.0]

    # Only env 1 resets; env 0 remains released and env 2 remains active.
    env.episode_length_buf = torch.tensor([13, 0, 13])
    assert latch(env, torch.tensor([False, False, False])).tolist() == [0.0, 1.0, 1.0]


def test_v1123_uncrossed_p11_reference_window_stays_dense() -> None:
    latch = _load_latch_function()
    env = SimpleNamespace(episode_length_buf=torch.arange(4))
    # Four independent envs in the physical reference window, none crossed.
    active = latch(env, torch.zeros(4, dtype=torch.bool))
    lift = torch.tensor([0.35, 0.45, 0.55, 0.65])
    body_x = torch.tensor([0.28, 0.30, 0.32, 0.34])
    retraction = torch.exp(-0.5 - torch.square((body_x - 0.30) / 0.06))
    product = active * lift * retraction
    assert torch.all(product > 1.0e-3)
    assert float(torch.quantile(product, 0.5)) > 1.0e-3


def test_v1123_control_treatment_diff_is_only_reward_weight() -> None:
    source = _source(ENV_CFG)
    treatment = source.split(
        "class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123EnvCfg", 1
    )[1].split("\n\n@configclass", 1)[0]
    control = source.split(
        "class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123ControlEnvCfg", 1
    )[1].split("\n\n@configclass", 1)[0]
    assert "self.rewards.front_legs_reach.weight = 0.45" in treatment
    assert "self.rewards.left_front_precontact_retraction.weight = 0.10" in treatment
    assert "self.rewards.left_front_precontact_retraction.weight = 0.0" in control
    for forbidden in ("FR_", "RL_", "RR_", "action_scale", "joint_pos", "observations"):
        assert forbidden not in treatment
        assert forbidden not in control


def test_v1123_reward_config_preserves_all_non_fl_terms_by_inheritance() -> None:
    source = _source(ENV_CFG)
    block = source.split(
        "class ArcdogAdjustableLegHighstepFrontGeometryV1123RewardsCfg", 1
    )[1].split("\n\n@configclass", 1)[0]
    assert "ArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121RewardsCfg" in block
    assert block.count("RewTerm(") == 1
    assert "weight=0.10" in block
    for term in ("rear_support_motion_contract", "front_legs_reach", "rear_legs_drive_bonus"):
        assert term not in block


def test_v1123_registry_and_runner_are_isolated() -> None:
    registry = _source(REGISTRY)
    runner = _source(RUNNER_CFG)
    for task in (
        "HighstepFrontGeometryV1123-ArcdogAdjustableLeg-v0",
        "HighstepFrontGeometryV1123Control-ArcdogAdjustableLeg-v0",
    ):
        assert task in registry
    assert "highstep_front_geometry_v1123_treatment" in runner
    assert "highstep_front_geometry_v1123_control" in runner
    assert "max_iterations = 300" in runner
    assert "max_iterations = 100" in runner


def test_v1123_latch_not_exposed_or_checkpointed() -> None:
    state_name = "_highstep_fl_precontact_reward_released"
    rewards_source = _source(REWARDS)
    assert state_name in rewards_source
    for path in (
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/observations.py",
        TRAIN,
        ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py",
    ):
        assert state_name not in _source(path)


def test_v1123_fresh_initial_distribution_is_hash_bound() -> None:
    block = _function_block(TRAIN, "_highstep_fresh_initial_terrain_distribution")
    for fragment in (
        "terrain_levels",
        "terrain_types",
        "histogram",
        "joint_sha256",
        "same_fresh_initial_training_distribution_not_process_exact_resume",
    ):
        assert fragment in block
    assert "HighstepFrontGeometryV1123" in _source(TRAIN)


def test_v1123_is_top_authority_and_preserves_rejected_history() -> None:
    spec = _source(SPEC)
    assert spec.startswith("# Highstep Teacher/Student 恢复与自动化规范 v1.12.3")
    top = spec.split("### v1.12.2：", 1)[0]
    for fragment in (
        "left_front_precontact_reward_geometry_contract",
        "model_173299.pt` rejected",
        "behavior_gate_failed",
        "full_climb=6/15",
        "rear_hold=6/15",
        "相同 fresh 初始训练分布",
        "不是 process-exact resume",
    ):
        assert fragment in top
