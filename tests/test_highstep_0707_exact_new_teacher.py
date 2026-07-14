"""CPU-only truth tests for the 0707_exact_new_teacher Stage-2 route."""

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
# This route is retained only as the immutable 1200/0/0 zero-scale ablation.
SPEC_SHA = "4baed191f98f9b746eec9181b3f31bcdd16e3cc147726b676d9949d7e1fe4425"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vae = _load("highstep_0707_exact_test_target", VAE_PPO)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy(seed: int):
    # Reuse the established CPU fixture so architecture and state keys stay exact.
    v15 = _load("highstep_v15_fixture_for_0707", ROOT / "tests/test_highstep_student_recovery_v15.py")
    policy = v15._Policy(seed)
    policy.student_recovery_stage = "0707_EXACT"
    policy.student_actor_warmup_updates = 1200
    policy.student_highstep_phase_loss_scale = 0.0
    policy.student_highstep_rear_box_loss_scale = 0.0
    return policy


def _algorithm(root: Path):
    source_policy = _policy(172300)
    source = root / "model_172300.pt"
    torch.save({"model_state_dict": source_policy.state_dict(), "iter": 172300}, source)
    prereg = root / "preregistration.json"
    prereg.write_text(json.dumps({
        "schema_version": 1,
        "kind": "highstep_0707_exact_new_teacher_preregistration",
        "authority": {"version": "v1.6.1", "spec_sha256": SPEC_SHA},
    }))
    policy = copy.deepcopy(source_policy)
    policy.student_recovery_0707_exact_preregistration_path = str(prereg.resolve())
    policy.student_recovery_0707_exact_preregistration_sha256 = _sha(prereg)
    policy.student_recovery_source_checkpoint = str(source.resolve())
    policy.student_recovery_source_sha256 = _sha(source)
    policy.student_recovery_teacher_checkpoint = str(source.resolve())
    policy.student_recovery_teacher_sha256 = _sha(source)
    algorithm = object.__new__(vae.VAEPPO)
    algorithm.policy = policy
    algorithm.device = torch.device("cpu")
    algorithm.distill_stage = 2
    algorithm.student_recovery_stage = "0707_EXACT"
    algorithm.num_mini_batches = 4
    algorithm.max_grad_norm = 1.0
    algorithm._initialize_student_recovery_v15()
    manifest = algorithm.bind_student_recovery_v15_checkpoints(
        loaded_student_checkpoint=str(source), checkpoint_load_mode="weights_only"
    )
    return algorithm, manifest


def test_archived_zero_scale_warmup_contract():
    with tempfile.TemporaryDirectory() as directory:
        algorithm, manifest = _algorithm(Path(directory))
        assert manifest["stage"] == "0707_EXACT"
        assert manifest["warmup_updates"] == 1200
        assert manifest["warmup_main_action_target"] == "teacher_pre_prior"
        assert manifest["phase_scale"] == 0.0
        assert manifest["rear_box_scale"] == 0.0
        assert manifest["student_teacher_storage_independent"] is True
        trainable = [name for name, value in algorithm.policy.named_parameters() if value.requires_grad]
        assert trainable and all(name.startswith("estimator.") for name in trainable)


def test_pre_prior_target_and_warmup_box_isolation():
    with tempfile.TemporaryDirectory() as directory:
        algorithm, _ = _algorithm(Path(directory))
        pre = torch.randn(2, 16)
        post = pre + 10.0
        assert algorithm._student_main_action_reference(pre, post) is pre
        actor_input = torch.randn(2, 634)
        output = algorithm.policy.actor(actor_input)
        assert not output.requires_grad
        estimator_outputs = algorithm.policy.estimator(torch.randn(2, 570))
        sum(value.square().mean() for value in estimator_outputs).backward()
        assert any(parameter.grad is not None for parameter in algorithm.policy.estimator.parameters())
        assert all(parameter.grad is None for parameter in (algorithm.v15_box_weight, algorithm.v15_box_bias))
        assert all(parameter.grad is None for parameter in algorithm.policy.actor.parameters())


def test_full_checkpoint_restores_optimizer_count_and_binding():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        algorithm, _ = _algorithm(root)
        state = algorithm.extra_checkpoint_state_dict()
        assert state["student_recovery"]["stage"] == "0707_EXACT"
        assert state["student_recovery"]["effective_update_count"] == 0
        assert state["student_recovery"]["binding_manifest"]["warmup_main_action_target"] == "teacher_pre_prior"
