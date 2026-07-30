from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
LAUNCHER = ROOT / "tools/highstep_0707_manual_play.sh"


def test_0707_student_profile_is_narrow_and_checkpoint_bound():
    source = PLAY.read_text(encoding="utf-8")
    assert '"--highstep_0707_legacy_student_profile"' in source
    assert "task_name != _HIGHSTEP_0707_STUDENT_TASK" in source
    assert 'policy.student_recovery_stage = "none"' in source
    assert "policy.distill_stage = 2" in source
    assert "policy.student_actor_warmup_updates = 1400" in source
    assert "policy.student_highstep_phase_loss_scale = 2.0" in source
    assert "policy.student_highstep_rear_box_loss_scale = 1.5" in source
    assert "actual_checkpoint_sha256 != _HIGHSTEP_0707_STUDENT_CHECKPOINT_SHA256" in source


def test_0707_launcher_binds_both_historical_checkpoints_and_saved_profiles():
    source = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "model_151399.pt",
        "d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d",
        "model_158797.pt",
        "7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284",
        "25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7",
        "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636",
    ):
        assert required in source


def test_0707_launcher_keeps_manual_comparison_and_recording_contract_fixed():
    source = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "--keyboard",
        "--debug",
        "--record_joint_data",
        "--joint_record_output",
        "--play_terrain_type box_hard",
        "--play_terrain_level 9",
        "--front_step_eval_edge_gap 0.55",
        "--eval_action_delay_steps 0",
        "--seed 11",
    ):
        assert required in source
    assert "--training_distribution" not in source
