from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CFG_ROOT = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped"
    / "Arcdog_adjustable_leg"
)
WORKFLOW = ROOT / "tmp/highstep_be300_0707_distill_20260716"
CHECKPOINT = (
    ROOT
    / "logs/rsl_rl"
    / "arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher"
    / "2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schedule_module():
    path = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"
    spec = importlib.util.spec_from_file_location("be300_highstep_schedule", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_be300_student_task_is_exact_environment_derivation() -> None:
    env_source = _text(CFG_ROOT / "highstep_env_cfg.py")
    registry = _text(CFG_ROOT / "__init__.py")

    assert (
        "class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorEnvCfg(\n"
        "    ArclabArcdogAdjustableLegHighstepFrontGeometryV1123EnvCfg"
    ) in env_source
    student_block = env_source.split(
        "class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorEnvCfg", 1
    )[1].split("@configclass", 1)[0]
    assert "mdp.DelayedJointPositionActionCfg" in student_block
    assert "min_action_delay_steps=0" in student_block
    assert "max_action_delay_steps=1" in student_block
    assert '".*_box_joint": 0.02' in student_block
    assert '".*_(hip_joint|thigh_joint|calf_joint)$": 0.1' in student_block
    assert 'clip={".*": (-60.0, 60.0)}' in student_block
    assert "PhasedHighstepBoxBiasJointPositionAction" not in student_block
    assert (
        "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123StudentNoPrior-"
        in registry
    )


def test_be300_runner_freezes_exact_0707_contract_and_long_budget() -> None:
    source = _text(CFG_ROOT / "agents/rsl_rl_ppo_cfg.py")
    block = source.split(
        "class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorPPORunnerCfg",
        1,
    )[1].split("@configclass", 1)[0]

    for token in (
        'student_recovery_stage = "BE300_0707"',
        "max_iterations = 4000",
        "save_interval = 100",
        "student_actor_warmup_updates = 1400",
        "student_vae_epochs = 4",
        "student_vel_loss_coef = 10.0",
        "student_latent_loss_coef = 50.0",
        "student_teacher_action_loss_coef = 20.0",
        "student_prior_box_loss_coef = 5.0",
        "student_recon_loss_coef = 0.5",
        "student_kl_loss_coef = 0.1",
        "student_highstep_phase_loss_scale = 2.0",
        "student_highstep_rear_box_loss_scale = 1.5",
        "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH",
    ):
        assert token in block
    assert "d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431" in block


def test_be300_route_uses_0707_mixed_prior_and_never_ppo() -> None:
    source = _text(CFG_ROOT / "agents/vae_ppo.py")
    assert '"E1400_CONTINUATION", "BE300_0707"' in source
    assert '"BE300_0707",' in source
    assert "return teacher_pre_prior" in source
    assert "if not actor_adapt_enabled:\n                                prior_box_weight = torch.zeros_like" in source
    assert "self.student_recovery_stage == \"BE300_0707\"" in source
    assert "Student root and independent Teacher must bind the same checkpoint bytes" in source
    assert "student_ppo_permanently_disabled\": True" in source


def test_parent_lineage_accepts_only_the_frozen_be300_bytes(tmp_path: Path) -> None:
    module = _schedule_module()
    parent = WORKFLOW / "teacher_parent_lineage_v1131.json"
    lineage = module._load_parent_teacher_lineage(parent)
    assert lineage["selected_teacher_checkpoint_path"] == str(CHECKPOINT.resolve())
    assert lineage["selected_teacher_checkpoint_sha256"] == _sha(CHECKPOINT)

    tampered = json.loads(parent.read_text(encoding="utf-8"))
    tampered["selected_checkpoint_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered_parent.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(module.ScheduleManifestInvalidError):
        module._load_parent_teacher_lineage(tampered_path)


def test_schedule_preserve_allows_only_exact_teacher_prior_removal() -> None:
    module = _schedule_module()
    source_definition = {
        "action_prior": {
            "enabled": True,
            "num_steps_per_update": 24,
            "start_update": 80,
            "full_update": 520,
        },
        "terrain_schedule": {"frozen": 1},
        "support_bottleneck": {"frozen": 2},
        "command_curriculum": {"frozen": 3},
        "reward_stages": {"frozen": 4},
    }
    student_definition = {
        **source_definition,
        "action_prior": {
            "enabled": False,
            "num_steps_per_update": 24,
            "start_update": None,
            "full_update": None,
        },
    }
    module.assert_schedule_definition_compatible(
        {"schedule_definition": source_definition},
        student_definition,
        allow_be300_teacher_prior_removal=True,
    )
    drifted = dict(student_definition)
    drifted["terrain_schedule"] = {"frozen": 999}
    with pytest.raises(module.ScheduleContinuityError):
        module.assert_schedule_definition_compatible(
            {"schedule_definition": source_definition},
            drifted,
            allow_be300_teacher_prior_removal=True,
        )


def test_schedule_preserve_accepts_exact_student_full_resume_without_remigration() -> None:
    module = _schedule_module()
    student_definition = {
        "action_prior": {
            "enabled": False,
            "num_steps_per_update": 24,
            "start_update": None,
            "full_update": None,
        },
        "terrain_schedule": {"frozen": 1},
        "support_bottleneck": {"frozen": 2},
        "command_curriculum": {"frozen": 3},
        "reward_stages": {"frozen": 4},
    }
    module.assert_schedule_definition_compatible(
        {"schedule_definition": student_definition},
        student_definition,
        allow_be300_teacher_prior_removal=True,
    )

    invalid_reactivation = {
        **student_definition,
        "action_prior": {
            "enabled": True,
            "num_steps_per_update": 24,
            "start_update": 80,
            "full_update": 520,
        },
    }
    with pytest.raises(module.ScheduleContinuityError):
        module.assert_schedule_definition_compatible(
            {"schedule_definition": student_definition},
            invalid_reactivation,
            allow_be300_teacher_prior_removal=True,
        )


def test_preregistration_forbids_behavior_gates_and_formal_start() -> None:
    prereg = json.loads((WORKFLOW / "preregistration_v1131.json").read_text(encoding="utf-8"))
    assert prereg["authority"]["version"] == "v1.13.1"
    assert prereg["teacher_and_student_root"]["checkpoint_sha256"] == _sha(CHECKPOINT)
    assert prereg["distillation_contract"]["main_action_target"] == "teacher_pre_prior_16d"
    assert prereg["distillation_contract"]["warmup_post_prior_box_loss"] == 0.0
    assert prereg["training_budget"]["continuous_single_run"] is True
    assert prereg["training_budget"]["effective_updates"] == 4000
    assert prereg["training_budget"]["automatic_core9"] is False
    assert prereg["formal_training_started"] is False
