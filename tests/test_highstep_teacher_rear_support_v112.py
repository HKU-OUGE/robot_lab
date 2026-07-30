from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py"
ENV_CFG = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/highstep_env_cfg.py"
)
RUNNER_CFG = ENV_CFG.parent / "agents/rsl_rl_ppo_cfg.py"
REGISTRY = ENV_CFG.parent / "__init__.py"
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
TRAIN = ROOT / "scripts/rsl_rl/base/train.py"
ALGORITHM_CHECKPOINT = ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py"
SUPERVISOR = ROOT / "tools/highstep_teacher_rear_support_v112_supervisor.py"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_all_v112_python_files_parse() -> None:
    for path in (
        REWARDS, ENV_CFG, RUNNER_CFG, REGISTRY, PLAY, TRAIN, ALGORITHM_CHECKPOINT, SUPERVISOR
    ):
        ast.parse(_source(path), filename=str(path))


def test_v112_has_exact_single_reward_replacement() -> None:
    source = _source(ENV_CFG)
    class_block = source.split(
        "class ArclabArcdogAdjustableLegHighstepRearSupportV112EnvCfg", 1
    )[1].split("\n\n@configclass", 1)[0]
    assert "self.rewards.rear_legs_drive_bonus.weight = 0.0" in class_block
    assert "self.rewards.highstep_rear_push_posture.weight = 0.0" in class_block
    assert "self.rewards.rear_support_motion_contract.weight = 2.30" in class_block
    assert "action_scale" not in class_block
    assert "joint_pos" not in class_block


def test_v112_rear_mirror_contract_excludes_box_and_has_correct_axis_signs() -> None:
    source = _source(ENV_CFG)
    reward_block = source.split(
        "class ArcdogAdjustableLegHighstepRearSupportV112RewardsCfg", 1
    )[1].split("\n\n@configclass", 1)[0]
    assert '["RL_hip_joint", "RR_hip_joint"]' in reward_block
    assert '["RL_thigh_joint", "RR_thigh_joint"]' in reward_block
    assert '["RL_calf_joint", "RR_calf_joint"]' in reward_block
    assert '"rear_mirror_joint_signs": [-1.0, 1.0, 1.0]' in reward_block
    assert "box_joint" not in reward_block


def test_v112_reward_uses_front_only_as_phase_signal() -> None:
    source = _source(REWARDS)
    block = source.split("def highstep_rear_support_motion_contract(", 1)[1].split(
        "\ndef highstep_rear_push_posture_bonus(", 1
    )[0]
    assert "_front_feet_highstep_commit_gate" in block
    assert "front_commit_ready" in block
    assert "double_support_phase" in block
    assert "sequential_phase" in block
    assert "bilateral_contact" in block
    assert "contact_weighted_slip" in block
    assert "width_deficit" in block
    assert "premature_rear_penalty" in block
    assert "front action" not in block.lower()


def test_v112_task_runner_and_long_run_are_narrowly_bound() -> None:
    task = "RobotLab-Isaac-Velocity-HighstepRearSupportV112-ArcdogAdjustableLeg-v0"
    assert task in _source(REGISTRY)
    runner = _source(RUNNER_CFG)
    assert "ArclabArcdogAdjustableLegHighstepRearSupportV112PPORunnerCfg" in runner
    assert 'experiment_name = "arclab_arcdog_adjustable_leg_highstep_rear_support_v112"' in runner
    supervisor = _source(SUPERVISOR)
    assert f'TASK = "{task}"' in supervisor
    assert "ADDITIONAL_UPDATES = 6000" in supervisor
    assert "--highstep_checkpoint_load_mode" in supervisor
    assert '"full"' in supervisor
    assert "--highstep_schedule_resume_mode" in supervisor
    assert '"preserve"' in supervisor
    assert "num_envs=4096" in supervisor
    assert "--verified_legacy_teacher_algorithm_state_manifest" in supervisor
    assert "checkpoint.resolve() == SOURCE.resolve()" in supervisor


def test_v112_legacy_teacher_resume_is_exact_sha_and_role_bound() -> None:
    checkpoint = _source(ALGORITHM_CHECKPOINT)
    train = _source(TRAIN)
    for fragment in (
        '"algorithm_class": "VAEPPO"',
        '"distill_stage": 1',
        '"student_recovery_stage": "NONE"',
        '"stock_optimizer_state_entries": 23',
        '"vae_optimizer_state_entries": 0',
        "Verified legacy Teacher PPO Adam was not fully restored",
        "never-stepped empty state",
    ):
        assert fragment in checkpoint
    assert "Verified legacy Teacher migration is only valid for the exact v1.12 Teacher task" in train
    assert "Verified legacy Teacher checkpoint SHA256 mismatch" in train


def test_v112_evaluation_records_rear_symmetry_observables() -> None:
    play = _source(PLAY)
    for field in (
        "rear_fore_aft_error",
        "rear_lateral_center_error",
        "approach_rear_fore_aft_error_q95",
        "critical_rear_fore_aft_error_q95",
        "critical_rear_lateral_center_error_q95",
    ):
        assert field in play


def test_v112_is_preserved_as_completed_history_under_v1122() -> None:
    spec = _source(SPEC)
    assert spec.startswith("# Highstep Teacher/Student 恢复与自动化规范 v1.12.2")
    v112 = spec.split("### v1.12：", 1)[1].split("### v1.11", 1)[0]
    assert "本节从 v1.12.1 起为 `historical-only`" in v112
    assert "唯一训练语义变量固定为 `rear_support_motion_contract`" in v112
    assert "追加 `6000 effective updates`" in v112
    assert "不得继续调整" in v112


def test_v112_canonical_safety_remains_fail_closed() -> None:
    supervisor = _source(SUPERVISOR)
    block = supervisor.split("def no_severe_inward(", 1)[1].split("\n\n", 1)[0]
    assert 'critical_rear_min_abs_y_q05", -1.0' in block
    assert 'critical_rear_width_q05", -1.0' in block
    assert 'critical_center_violation_rate", 1.0' in block
    assert 'critical_width_violation_rate", 1.0' in block


def test_v112_wandb_remote_scan_does_not_request_system_step_alone() -> None:
    supervisor = _source(SUPERVISOR)
    assert "run.scan_history(page_size=1000)" in supervisor
    assert 'scan_history(keys=["_step"]' not in supervisor
    assert 'remote_unique_steps", 0)) < ADDITIONAL_UPDATES' in supervisor
