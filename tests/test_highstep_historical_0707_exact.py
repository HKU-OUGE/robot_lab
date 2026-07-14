"""CPU-only truth tests for the evidence-frozen historical 0707 route."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
VAE_PPO = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
SPEC_SHA = "eff246af70dbfca7b3a6661f0a29f71bb1aa3b769a944351431f8b589663437a"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vae = _load("highstep_historical_0707_test_target", VAE_PPO)
v15_fixture = _load(
    "highstep_v15_fixture_for_historical_0707",
    ROOT / "tests/test_highstep_student_recovery_v15.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _algorithm(root: Path):
    source_policy = v15_fixture._Policy(172300)
    source_policy.student_recovery_stage = "HISTORICAL_0707_EXACT"
    source_policy.student_actor_warmup_updates = 1400
    source_policy.student_highstep_phase_loss_scale = 2.0
    source_policy.student_highstep_rear_box_loss_scale = 1.5
    source = root / "model_172300.pt"
    torch.save({"model_state_dict": source_policy.state_dict(), "iter": 172300}, source)
    prereg = root / "preregistration.json"
    prereg.write_text(json.dumps({
        "schema_version": 1,
        "kind": "highstep_historical_0707_exact_preregistration",
        "authority": {"version": "v1.7", "spec_sha256": SPEC_SHA},
    }))
    policy = copy.deepcopy(source_policy)
    policy.student_recovery_historical_0707_exact_preregistration_path = str(prereg.resolve())
    policy.student_recovery_historical_0707_exact_preregistration_sha256 = _sha(prereg)
    policy.student_recovery_source_checkpoint = str(source.resolve())
    policy.student_recovery_source_sha256 = _sha(source)
    policy.student_recovery_teacher_checkpoint = str(source.resolve())
    policy.student_recovery_teacher_sha256 = _sha(source)
    algorithm = object.__new__(vae.VAEPPO)
    algorithm.policy = policy
    algorithm.device = torch.device("cpu")
    algorithm.distill_stage = 2
    algorithm.student_recovery_stage = "HISTORICAL_0707_EXACT"
    algorithm.num_mini_batches = 4
    algorithm.max_grad_norm = 1.0
    algorithm._initialize_student_recovery_v15()
    manifest = algorithm.bind_student_recovery_v15_checkpoints(
        loaded_student_checkpoint=str(source), checkpoint_load_mode="weights_only"
    )
    return algorithm, manifest


def test_historical_contract_and_independent_binding():
    with tempfile.TemporaryDirectory() as directory:
        algorithm, manifest = _algorithm(Path(directory))
        assert manifest["stage"] == "HISTORICAL_0707_EXACT"
        assert manifest["warmup_updates"] == 1400
        assert manifest["warmup_main_action_target"] == "teacher_pre_prior"
        assert manifest["phase_scale"] == 2.0
        assert manifest["rear_box_scale"] == 1.5
        assert manifest["warmup_prior_box_loss_coefficient"] == 0.0
        assert manifest["student_teacher_storage_independent"] is True
        trainable = [name for name, value in algorithm.policy.named_parameters() if value.requires_grad]
        assert trainable and all(name.startswith("estimator.") for name in trainable)


def test_pre_prior_action_loss_reaches_estimator_through_frozen_actor():
    with tempfile.TemporaryDirectory() as directory:
        algorithm, _ = _algorithm(Path(directory))
        batch = 8
        estimator_input = torch.randn(batch, 570)
        policy_obs = torch.randn(batch, 570)
        _, _, mu, _, _ = algorithm.policy.estimator(estimator_input)
        student_action = algorithm.policy.actor(
            torch.cat((policy_obs, torch.clamp(mu, -1.0, 1.0)), dim=-1)
        )
        with torch.no_grad():
            teacher_latent = torch.randn(batch, 64).clamp(-1.0, 1.0)
            teacher_pre_prior = algorithm.teacher_actor(
                torch.cat((policy_obs, teacher_latent), dim=-1)
            )
            teacher_post_prior = teacher_pre_prior + 7.0
        reference = algorithm._student_main_action_reference(
            teacher_pre_prior, teacher_post_prior
        )
        assert reference is teacher_pre_prior

        error = (student_action - reference).square()
        base = 2.0 * error[:, :12].mean(-1) + 0.5 * error[:, 12:].mean(-1)
        phase_gate = torch.linspace(0.0, 1.0, batch)
        rear_box = error[:, 14:16].mean(-1)
        historical_pre_prior_loss = (
            (base + 1.5 * phase_gate * rear_box) * (1.0 + 2.0 * phase_gate)
        ).mean()
        historical_pre_prior_loss.backward()

        estimator_grad = sum(
            float(parameter.grad.norm())
            for parameter in algorithm.policy.estimator.parameters()
            if parameter.grad is not None
        )
        assert estimator_grad > 0.0
        assert all(parameter.grad is None for parameter in algorithm.policy.actor.parameters())
        assert all(
            parameter.grad is None
            for parameter in (algorithm.v15_box_weight, algorithm.v15_box_bias)
        )


def test_checkpoint_binds_historical_optimizer_and_count():
    with tempfile.TemporaryDirectory() as directory:
        algorithm, _ = _algorithm(Path(directory))
        state = algorithm.extra_checkpoint_state_dict()
        assert state["student_recovery"]["stage"] == "HISTORICAL_0707_EXACT"
        assert state["student_recovery"]["effective_update_count"] == 0
        assert state["student_recovery"]["binding_manifest"]["phase_scale"] == 2.0


def test_only_exact_historical_core9_alias_is_allowed():
    compat = _load(
        "highstep_historical_0707_core9_compat",
        ROOT / "tools/highstep_core9_task_compat.py",
    )
    source = (
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorHistorical0707Exact-"
        "ArcdogAdjustableLeg-v0"
    )
    target = (
        "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    )
    assert compat.source_task_is_compatible(
        workflow_id="highstep_historical_0707_exact_new_teacher_20260714",
        role="student",
        source_task=source,
        eval_task=target,
    )
    assert not compat.source_task_is_compatible(
        workflow_id="wrong-workflow", role="student", source_task=source, eval_task=target
    )
    assert not compat.source_task_is_compatible(
        workflow_id="highstep_historical_0707_exact_new_teacher_20260714",
        role="student",
        source_task=source + "-wrong",
        eval_task=target,
    )
