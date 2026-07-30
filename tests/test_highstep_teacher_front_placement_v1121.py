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
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
TRAIN = ROOT / "scripts/rsl_rl/base/train.py"
SUPERVISOR = ROOT / "tools/highstep_teacher_front_placement_v1122_supervisor.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_v1121_python_files_parse() -> None:
    for path in (REWARDS, ENV_CFG, RUNNER_CFG, REGISTRY, PLAY, SUPERVISOR):
        ast.parse(_source(path), filename=str(path))


def test_v1121_reward_is_narrowly_fl_precontact_gated() -> None:
    block = _source(REWARDS).split(
        "def left_front_highstep_precontact_retraction_bonus(", 1
    )[1].split("\ndef _highstep_masked_ray_mean", 1)[0]
    for fragment in (
        "left_front_foot_name",
        "_forward_highstep_terrain_gate",
        "_front_feet_highstep_commit_gate",
        "precommit_gate",
        "below_release_gate",
        "lift_score",
        "retraction_score",
    ):
        assert fragment in block
    assert "right_front_foot_name" not in block
    assert "rear_support_motion_contract" not in block
    assert "action_scale" not in block
    assert "joint_pos" not in block


def test_v1121_preserves_front_reward_scale_and_v112_rear_contract() -> None:
    source = _source(ENV_CFG)
    block = source.split(
        "class ArclabArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121EnvCfg",
        1,
    )[1].split("\n\n@configclass", 1)[0]
    assert "ArclabArcdogAdjustableLegHighstepRearSupportV112EnvCfg" in block
    assert "self.rewards.front_legs_reach.weight = 0.45" in block
    assert "self.rewards.left_front_precontact_retraction.weight = 0.10" in block
    assert "rear_support_motion_contract" not in block
    assert "action_scale" not in block
    assert "joint_pos" not in block


def test_v1121_fixed_retraction_contract_is_not_ambiguous() -> None:
    source = _source(ENV_CFG)
    reward_block = source.split(
        "class ArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121RewardsCfg",
        1,
    )[1].split("\n\n@configclass", 1)[0]
    assert '"left_front_foot_name": "FL_foot"' in reward_block
    assert '"retraction_start_x": 0.36' in reward_block
    assert '"retraction_target_x": 0.30' in reward_block
    assert '"clearance_release": 0.04' in reward_block
    assert '"clearance_window": 0.12' in reward_block


def test_v1121_task_and_runner_use_isolated_namespaces() -> None:
    task = (
        "RobotLab-Isaac-Velocity-HighstepRearSupportFrontPlacementV1121-"
        "ArcdogAdjustableLeg-v0"
    )
    assert task in _source(REGISTRY)
    runner = _source(RUNNER_CFG)
    assert "ArclabArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121PPORunnerCfg" in runner
    assert (
        '"arclab_arcdog_adjustable_leg_highstep_rear_support_front_placement_v1121"'
        in runner
    )


def test_v1122_history_is_preserved_under_v1123_authority() -> None:
    spec = _source(SPEC)
    assert spec.startswith("# Highstep Teacher/Student 恢复与自动化规范 v1.12.3")
    top = spec.split("### v1.12：", 1)[0]
    assert "### v1.12.2：6 cm 左前足预接触回收单变量修订" in top
    assert "`left_front_precontact_retraction`" in top
    assert "`front_legs_reach` | `0.45` | `0.45`" in top
    assert "`left_front_precontact_retraction` | 不存在 / `0.00` | `0.10`" in top
    assert "不得把 `0.10` 从对称项中扣除" in top
    assert "Kazam_screencast_00155.mp4" in top
    assert "5350ab46fa2cfe1e966f99e6717b738b60b32b9edcbd1f4a323205a55f2c128e" in top
    assert "`0/15`" in top
    assert "实时在线同步" in top
    assert "`implementation_ready_pending_finite_training_amendment`" in top


def test_v1122_eval_has_monotonic_fl_forbidden_surface_gate() -> None:
    source = _source(PLAY)
    block = source.split("class _HighstepEvalTracker:", 1)[1]
    assert 'for name in ("FL_foot", "FR_foot")' in block
    assert "fl_forbidden_surface_contact = False" in block
    assert "fl_forbidden_surface_contact = True" in block
    assert "fl_forbidden_surface_first_step" in block
    assert '"fl_vertical_riser_or_top_lip_underside_contact"' in block
    assert '"schema_version": 8' in block
    # Once observed, the episode flag is never reset by a later recovery.
    update_block = block.split("def update(self, step: int)", 1)[1].split(
        "@staticmethod\n    def _contact_lag", 1
    )[0]
    assert "self.fl_forbidden_surface_contact = False" not in update_block


def test_v1122_supervisor_is_one_finite_e100_stage() -> None:
    source = _source(SUPERVISOR)
    for fragment in (
        "FORMAL_UPDATES = 100",
        "TARGET_FINAL_ITERATION = SOURCE_ITERATION + FORMAL_UPDATES - 1",
        "FORMAL_NUM_ENVS = 4096",
        "SAVE_INTERVAL = 100",
        "EVAL_SEEDS = tuple(range(1101, 1116))",
        'stage.get("stage_id") == "FLR-E100"',
        '"teacher_fl_retraction_first_round_gate_failed"',
    ):
        assert fragment in source
    assert "6000" not in source


def test_v1122_supervisor_requires_full_resume_and_online_wandb() -> None:
    source = _source(SUPERVISOR)
    for fragment in (
        '"--highstep_checkpoint_load_mode",\n            "full"',
        '"--highstep_schedule_resume_mode",\n            "preserve"',
        '"WANDB_MODE": "online"',
        '"ROBOT_LAB_WANDB_CONFIG_PATH"',
        '"ROBOT_LAB_WANDB_CONFIG_SHA256"',
        '"wandb/flr_E100_config_rebinding4.yaml"',
        '"online_remote_progress_verified"',
        '"W&B real-time step growth was not verified during the active child',
        'remote.get("state") == "finished"',
    ):
        assert fragment in source
    assert '"WANDB_MODE": "offline"' not in source
    assert '"WANDB_CONFIG_PATHS"' not in source


def test_v1122_supervisor_emits_wandb_native_yaml_without_float_coercion() -> None:
    source = _source(SUPERVISOR)
    block = source.split("def _write_wandb_config", 1)[1].split(
        "def _formal_run_dirs", 1
    )[0]
    assert '"wandb/flr_E100_config_rebinding4.yaml"' in block
    assert "yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)" in block
    assert "yaml.safe_load(path.read_text" in block
    assert "atomic_json(path, payload" not in block


def test_v1122_train_loads_wandb_config_as_hash_bound_sequence() -> None:
    source = _source(TRAIN)
    block = source.split("def _setup_hash_bound_wandb_config", 1)[1].split(
        "# add obs&action dict", 1
    )[0]
    for fragment in (
        'os.environ.get("ROBOT_LAB_WANDB_CONFIG_PATH", "")',
        'os.environ.get("ROBOT_LAB_WANDB_CONFIG_SHA256", "")',
        "os.stat(config_path).st_mode & 0o222",
        "hashlib.sha256(stream.read()).hexdigest()",
        "wandb.Settings(config_paths=[config_path])",
    ):
        assert fragment in block
    assert "WANDB_CONFIG_PATHS" not in block
    assert "_setup_hash_bound_wandb_config(agent_cfg.logger)" in source


def test_v1122_supervisor_uses_task_only_schedule_rebinding_without_reset() -> None:
    source = _source(SUPERVISOR)
    for fragment in (
        'REBOUND_SOURCE_DIR = WORK / "source_model_173200_v1122"',
        'rebound.get("schedule_reset") is False',
        'rebound.get("optimizer_reset") is False',
        "shutil.copy2(SOURCE, REBOUND_SOURCE)",
        'checkpoint=REBOUND_SOURCE',
        'self.evaluate(REBOUND_SOURCE, "baseline_E1000")',
    ):
        assert fragment in source
    assert '"--highstep_schedule_resume_mode",\n            "reset"' not in source


def test_v1122_supervisor_binds_baseline_and_hard_0_of_15_gate() -> None:
    source = _source(SUPERVISOR)
    run_block = source.split("def run(self) -> None:", 1)[1].split(
        "def stop(self", 1
    )[0]
    assert 'self.evaluate(REBOUND_SOURCE, "baseline_E1000")' in run_block
    assert 'self.evaluate(checkpoint, "candidate_FLR_E100")' in run_block
    assert 'candidate["fl_riser_contact_count"] == 0' in run_block
    assert 'candidate["full_climb_count"] >= baseline["full_climb_count"]' in run_block
    assert 'candidate["rear_hold_count"] >= baseline["rear_hold_count"]' in run_block
    assert "self.formal_train()" in run_block
    assert run_block.index('self.evaluate(REBOUND_SOURCE, "baseline_E1000")') < run_block.index(
        "self.formal_train()"
    )


def test_v1122_supervisor_never_launches_student_or_deploys() -> None:
    source = _source(SUPERVISOR)
    assert "student_training_started=False" in source
    assert '"student_training_started": False' in source
    assert '"automatic_real_robot_deployment": False' in source
    assert "Distillation" not in source
