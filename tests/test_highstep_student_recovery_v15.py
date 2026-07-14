"""CPU-only fail-closed contracts for the v1.5 Stage-2 recovery route."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
VAE_PPO_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
V15_SPEC_SHA = "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
SPEC_SHA = "4baed191f98f9b746eec9181b3f31bcdd16e3cc147726b676d9949d7e1fe4425"
CURRENT_SPEC_SHA = "e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vae = _load_module("robot_lab_v15_test_target", VAE_PPO_PATH)
VAEPPO = vae.VAEPPO


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Policy(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.actor = nn.Sequential(
            nn.Linear(634, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16),
        )
        self.critic = nn.Sequential(nn.Linear(162, 16), nn.ELU(), nn.Linear(16, 1))
        self.std = nn.Parameter(torch.ones(16))
        self.priv_encoder = vae.PrivilegedEncoder(159, 64, [16, 8])
        self.estimator = vae.ProprioVAE(570, vel_dim=3, latent_dim=64, hidden_dims=[16, 128])
        self.distill_stage = 2
        self.student_recovery_stage = "V15"
        self.student_actor_warmup_updates = 1400
        self.student_vae_epochs = 4
        self.student_low_speed_threshold = 0.10
        self.student_prior_fade_speed = 0.45
        self.student_vel_loss_coef = 10.0
        self.student_latent_loss_coef = 50.0
        self.student_teacher_action_loss_coef = 20.0
        self.student_prior_box_loss_coef = 5.0
        self.student_recon_loss_coef = 0.5
        self.student_kl_loss_coef = 0.1
        self.student_post_prior_mode = "highstep"
        self.student_highstep_phase_loss_scale = 2.0
        self.student_highstep_rear_box_loss_scale = 1.5


class V15Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source_policy = _Policy(172300)
        self.source = root / "model_172300.pt"
        torch.save({"model_state_dict": self.source_policy.state_dict(), "iter": 172300}, self.source)
        self.source_sha = _sha(self.source)
        self.prereg = root / "preregistration.json"
        self.prereg.write_text(json.dumps({
            "schema_version": 1,
            "kind": "highstep_student_recovery_v15_preregistration",
            "authority": {"version": "v1.5", "spec_sha256": V15_SPEC_SHA},
        }))
        self.prereg_sha = _sha(self.prereg)

    def algorithm(self):
        policy = copy.deepcopy(self.source_policy)
        policy.student_recovery_v15_preregistration_path = str(self.prereg.resolve())
        policy.student_recovery_v15_preregistration_sha256 = self.prereg_sha
        policy.student_recovery_source_checkpoint = str(self.source.resolve())
        policy.student_recovery_source_sha256 = self.source_sha
        policy.student_recovery_teacher_checkpoint = str(self.source.resolve())
        policy.student_recovery_teacher_sha256 = self.source_sha
        algorithm = object.__new__(VAEPPO)
        algorithm.policy = policy
        algorithm.device = torch.device("cpu")
        algorithm.distill_stage = 2
        algorithm.student_recovery_stage = "V15"
        algorithm.num_mini_batches = 4
        algorithm.max_grad_norm = 1.0
        algorithm._initialize_student_recovery_v15()
        manifest = algorithm.bind_student_recovery_v15_checkpoints(
            loaded_student_checkpoint=str(self.source), checkpoint_load_mode="weights_only"
        )
        return algorithm, manifest


class HighstepStudentRecoveryV15Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = V15Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_archived_v15_authority_does_not_override_current_spec(self) -> None:
        self.assertEqual(CURRENT_SPEC_SHA, _sha(SPEC))
        self.assertNotEqual(SPEC_SHA, CURRENT_SPEC_SHA)

    def test_optimizer_and_independent_teacher_binding_are_exact(self) -> None:
        algorithm, manifest = self.fixture.algorithm()
        self.assertEqual("V15", manifest["stage"])
        self.assertEqual(0, manifest["effective_update_count"])
        self.assertEqual([1.0e-3, 1.0e-5], manifest["optimizer_group_learning_rates"])
        self.assertEqual([12, 13, 14, 15], manifest["post_warmup_trainable_action_rows"])
        self.assertTrue(manifest["student_teacher_storage_independent"])
        self.assertTrue(all(parameter.requires_grad for parameter in algorithm.policy.estimator.parameters()))
        trainable_non_estimator = [
            name for name, parameter in algorithm.policy.named_parameters()
            if parameter.requires_grad and not name.startswith("estimator.")
        ]
        self.assertEqual([], trainable_non_estimator)
        algorithm._validate_v15_optimizer_scope()

    def test_box_rows_are_frozen_through_1400_and_only_box_rows_open_afterward(self) -> None:
        algorithm, _ = self.fixture.algorithm()
        root_weight = algorithm._v15_last_linear.weight.detach().clone()
        root_bias = algorithm._v15_last_linear.bias.detach().clone()
        with torch.no_grad():
            algorithm.v15_box_weight.add_(0.01)
            algorithm.v15_box_bias.add_(0.01)
        algorithm._sync_v15_box_rows_to_actor()
        algorithm.student_distill_update_count = 1400
        with self.assertRaisesRegex(RuntimeError, "frozen Student tensor changed"):
            algorithm._assert_v15_frozen_unchanged()

        with torch.no_grad():
            algorithm._v15_last_linear.weight.copy_(root_weight)
            algorithm._v15_last_linear.bias.copy_(root_bias)
        algorithm._reset_v15_box_rows_from_actor()
        algorithm.student_distill_update_count = 1401
        with torch.no_grad():
            algorithm.v15_box_weight.add_(0.01)
            algorithm.v15_box_bias.add_(0.01)
        algorithm._sync_v15_box_rows_to_actor()
        algorithm._assert_v15_frozen_unchanged()
        torch.testing.assert_close(
            algorithm._v15_last_linear.weight[:12], root_weight[:12], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            algorithm._v15_last_linear.bias[:12], root_bias[:12], rtol=0.0, atol=0.0
        )

    def test_checkpoint_extra_state_binds_count_optimizer_and_box_leaves(self) -> None:
        algorithm, _ = self.fixture.algorithm()
        state = algorithm.extra_checkpoint_state_dict()
        recovery = state["student_recovery"]
        self.assertEqual("V15", recovery["stage"])
        self.assertEqual(0, recovery["effective_update_count"])
        self.assertEqual(
            list(algorithm._v15_optimizer_parameter_names()), recovery["optimizer_parameter_names"]
        )
        torch.testing.assert_close(recovery["box_weight"], algorithm.v15_box_weight)
        torch.testing.assert_close(recovery["box_bias"], algorithm.v15_box_bias)

    def test_wrong_fresh_checkpoint_fails_closed(self) -> None:
        algorithm, _ = self.fixture.algorithm()
        other = Path(self.temporary.name) / "model_other.pt"
        torch.save({"model_state_dict": self.fixture.source_policy.state_dict()}, other)
        with self.assertRaisesRegex(RuntimeError, "canonical model_172300"):
            algorithm.bind_student_recovery_v15_checkpoints(
                loaded_student_checkpoint=str(other), checkpoint_load_mode="weights_only"
            )

    def test_v152_rebinding_is_hash_bound_and_fail_closed(self) -> None:
        algorithm, _ = self.fixture.algorithm()
        audit = Path(self.temporary.name) / "rebinding_audit.json"
        audit.write_text('{"exact_resume_allowed":true}\n')
        checkpoint = self.fixture.source.resolve()
        old_prereg_sha = self.fixture.prereg_sha
        algorithm._v15_preregistration = {
            "kind": "highstep_student_recovery_v152_preregistration",
            "resume_rebinding": {
                "source_preregistration_sha256": old_prereg_sha,
                "source_effective_updates": 300,
                "preserve_optimizer": True,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": _sha(checkpoint),
                "audit_path": str(audit.resolve()),
                "audit_sha256": _sha(audit),
            },
        }
        result = algorithm._validate_v152_resume_rebinding(
            old_prereg_sha, effective_updates=300, loaded_checkpoint=str(checkpoint)
        )
        self.assertTrue(result["preserve_optimizer"])
        with self.assertRaisesRegex(ValueError, "effective|contract"):
            algorithm._validate_v152_resume_rebinding(
                old_prereg_sha, effective_updates=301, loaded_checkpoint=str(checkpoint)
            )
        algorithm._v15_preregistration["resume_rebinding"]["source_checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
            algorithm._validate_v152_resume_rebinding(
                old_prereg_sha, effective_updates=300, loaded_checkpoint=str(checkpoint)
            )


if __name__ == "__main__":
    unittest.main()
