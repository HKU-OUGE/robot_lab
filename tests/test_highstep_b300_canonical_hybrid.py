from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import highstep_wandb_stage_gate as wandb_gate


ROOT = Path("/home/lxq/Softwares/robot_lab")
FLOW = ROOT / "tmp/highstep_b300_canonical_hybrid_prior_latent_20260718"
PREREG = FLOW / "preregistration.json"
REPAIRS = tuple(sorted((FLOW / "manifests").glob("*repair_rebinding*.json")))
VAE = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
SUPERVISOR = ROOT / "tools/highstep_b300_canonical_hybrid_supervisor.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_binds_one_verified_canonical_episode() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["canonical_information_count"] == 1
    assert prereg["centerline"]["save_points"] == [100, 300, 500, 700]
    assert prereg["centerline"]["required"] == {
        "valid": 15,
        "full_climb": 12,
        "rear_hold": 12,
        "safe": 15,
    }
    manifest = Path(prereg["canonical_tensor_manifest"])
    dataset = Path(prereg["canonical_tensor_dataset"])
    assert sha(manifest) == prereg["canonical_tensor_manifest_sha256"]
    assert sha(dataset) == prereg["canonical_tensor_dataset_sha256"]
    tensors = torch.load(dataset, map_location="cpu", weights_only=True)
    assert tensors["student_obs_570"].shape == (138, 570)
    assert tensors["teacher_latent_raw_64"].shape == (138, 64)
    assert tensors["teacher_pre_prior_action_16"].shape == (138, 16)
    assert tensors["teacher_post_prior_policy_action_16"].shape == (138, 16)


def test_mixed_target_and_update_zero_box_scope_are_explicit() -> None:
    prereg = json.loads(PREREG.read_text())
    contract = prereg["training_contract"]
    assert contract["frozen_actor_rows"] == list(range(12))
    assert contract["trainable_actor_rows_from_update_zero"] == [12, 13, 14, 15]
    assert contract["student_actor_latent_clamp_backward"] == "straight_through"
    source = VAE.read_text()
    assert 'student_recovery_stage", "NONE") == "B300_CANONICAL_HYBRID"' in source
    assert 'target_action = data["mixed_action_target_16"]' in source
    assert 'self._v15_last_linear.weight[:12].detach()' in source
    assert "B300 estimator/box rows did not both receive gradients" in source


def test_preregistered_training_code_hashes_are_exact() -> None:
    prereg = json.loads(PREREG.read_text())
    repairs = [json.loads(path.read_text()) for path in REPAIRS]
    for name, expected in prereg["critical_training_code"].items():
        resolved = expected
        while True:
            matching = [
                repair for repair in repairs
                if repair["affected_file"] == name and repair["old_sha256"] == resolved
            ]
            if not matching:
                break
            assert len(matching) == 1
            assert matching[0]["training_contract_changed"] is False
            resolved = matching[0]["new_sha256"]
        assert sha(Path(name)) == resolved


def test_b300_preregistration_fields_reach_policy_instance() -> None:
    source = VAE.read_text()
    assert 'kwargs.pop(\n            "student_recovery_b300_hybrid_preregistration_path", ""' in source
    assert 'kwargs.pop(\n            "student_recovery_b300_hybrid_preregistration_sha256", ""' in source


def test_behavior_replay_is_narrowly_bound_to_b300_student_checkpoint() -> None:
    source = PLAY.read_text()
    assert 'args_cli.b300_hybrid_behavior_preregistration_sha256' in source
    assert 'recovery.get("stage") == "B300_CANONICAL_HYBRID"' in source
    assert 'B300 frozen command source is not the preregistered Teacher' in source


def test_supervisor_has_fixed_stage_and_gate_contract() -> None:
    source = SUPERVISOR.read_text()
    assert 'phase="route_correction_preflight_complete"' in source
    assert "self.checkpoint_scope(dagger_source, 300)" in source
    assert 'counts["full_climb"] >= 12' in source
    assert 'counts["rear_hold"] >= 12' in source
    assert '"--highstep_schedule_resume_mode", "reset" if previous == 0 else "preserve"' in source
    assert '"WANDB_MODE": "online"' in source
    assert "collect_dagger_round" in source
    assert "train_dagger_round" in source
    assert 'phase="pending_user_visual_review"' in source


def test_dagger_is_student_driven_and_uses_exact_mixed_target() -> None:
    helper = (ROOT / "scripts/rsl_rl/base/highstep_b300_hybrid_latent.py").read_text()
    assert "class DaggerTensorCollector" in helper
    assert '"student_drives_physics": True' in helper
    assert '"teacher_labels_same_pre_step_state": True' in helper
    assert '"teacher_pre_prior_action_16": teacher_raw.clone()' in helper
    assert '"teacher_post_prior_policy_action_16": teacher_post_policy.clone()' in helper
    play = PLAY.read_text()
    assert "actions = policy(obs)" in play
    assert "b300_hybrid_dagger_collector.prepare(obs, actions" in play


def test_dagger_route_is_frozen_and_bounded() -> None:
    route_path = FLOW / "dagger_route_correction_v11.json"
    route = json.loads(route_path.read_text())
    assert route["status"] == "frozen_before_D1_collection_after_E300_diagnosis"
    assert route["maximum_rounds"] == 3
    assert route["training_updates_per_round"] == 100
    assert route["collection_trajectories_per_round"] == 1
    assert route["single_changed_variable"]["name"] == "supervision_state_distribution"
    assert route["canonical_only_E500_E700_forbidden"] is True
    assert route["interrupted_E500_recovery_forbidden"] is True
    assert route["source_effective_updates"] == 300
    assert route["source_checkpoint_sha256"] == (
        "e7a3eec3cc0848b4af7e9a8608bc281ad94c64ec1e299f69cedcbe34bf04d388"
    )
    assert route["scenarios"] == [{
        "name": "nominal_student_state", "gap_m": 0.55,
        "lateral_m": 0.0, "yaw_deg": 0.0, "joint_delta_sign": 0,
    }]
    merge_source = (ROOT / "tools/highstep_b300_hybrid_dagger_merge.py").read_text()
    assert 'str(scenario_payload.get("name", ""))' in merge_source


def test_user_aborted_e500_is_preserved_but_not_a_d1_sync_dependency(tmp_path: Path) -> None:
    wandb_gate.assert_all_prior_synced(FLOW)
    source = FLOW / (
        "wandb_stages/highstep_b300_canonical_hybrid_prior_latent_20260718_B_E500_attempt1/"
        "stage_manifest.json"
    )
    payload = json.loads(source.read_text())
    payload["aborted_stage_manifest_sha256"] = "0" * 64
    destination = tmp_path / "wandb_stages/E500/stage_manifest.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="aborted-stage binding changed"):
        wandb_gate.assert_all_prior_synced(tmp_path)


def test_wandb_contract_accepts_exact_dagger_route_and_rejects_unknown_route() -> None:
    path = FLOW / "wandb_stage_configs/D1.json"
    config = json.loads(path.read_text())
    assert wandb_gate.validate_contract(config)["route"] == "D"
    config["route"] = "unknown"
    with pytest.raises(RuntimeError, match="unsupported custom route"):
        wandb_gate.validate_contract(config)
