"""CPU-only contract tests for the approved highstep Student recovery stage B."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
VAE_PPO_PATH = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped"
    / "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
RUNNER_CFG_PATH = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped"
    / "Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py"
)

CANONICAL_STUDENT = (
    "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_04-41-42_robust_student_distill_20260712_044124/model_900.pt"
)
CANONICAL_STUDENT_SHA256 = "9bbd5b597d9c195ecf9afb141152b8f0749dc599a54868674b299107fb40a229"
CANONICAL_TEACHER = (
    "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
CANONICAL_TEACHER_SHA256 = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vae_ppo_module = _load_module("robot_lab_stage_b_test_target", VAE_PPO_PATH)
VAEPPO = vae_ppo_module.VAEPPO


class _TinyEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(5, 2)

    def encode(self, observation: torch.Tensor):
        mu = self.encoder(observation)
        logvar = torch.zeros_like(mu)
        velocity = torch.zeros((*mu.shape[:-1], 3), dtype=mu.dtype, device=mu.device)
        return mu, logvar, velocity


class _TinyPolicy(nn.Module):
    """Small policy with the same component names and 16-action head as production."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        shared_activation = nn.ELU()
        self.actor = nn.Sequential(
            nn.Linear(6, 8),
            shared_activation,
            nn.Linear(8, 8),
            shared_activation,
            nn.Linear(8, 8),
            shared_activation,
            nn.Linear(8, 16),
        )
        self.estimator = _TinyEstimator()
        self.critic = nn.Sequential(nn.Linear(7, 4), nn.ELU(), nn.Linear(4, 1))
        self.priv_encoder = nn.Sequential(nn.Linear(4, 2), nn.Tanh())

        self.distill_stage = 2
        self.student_recovery_stage = "B"
        self.student_recovery_actor_lr = 1.0e-3
        self.student_recovery_epochs = 1
        self.student_recovery_source_checkpoint = ""
        self.student_recovery_source_sha256 = ""
        self.student_recovery_teacher_checkpoint = ""
        self.student_recovery_teacher_sha256 = ""
        self.student_actor_warmup_updates = 0
        self.student_highstep_rear_hip_min_abs = 0.0

        self.policy_obs_dim = 4
        self.vae_latent_dim = 2
        self.policy_keys = ["policy"]
        self.estimator_keys = ["estimator"]
        self.critic_keys = ["critic"]


class _TinyStorage:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(20260712)
        self.observations = {
            "policy": torch.randn(2, 3, 4, generator=generator),
            "estimator": torch.randn(2, 3, 5, generator=generator),
            "critic": torch.randn(2, 3, 7, generator=generator),
        }
        self.step = 2

    def clear(self) -> None:
        self.step = 0


def _make_algorithm(policy: _TinyPolicy | None = None, *, seed: int = 7):
    algorithm = object.__new__(VAEPPO)
    algorithm.policy = policy if policy is not None else _TinyPolicy(seed)
    algorithm.device = torch.device("cpu")
    algorithm.distill_stage = 2
    algorithm.student_recovery_stage = "B"
    algorithm.num_mini_batches = 1
    algorithm.max_grad_norm = 1.0
    algorithm.storage = _TinyStorage()
    algorithm._initialize_student_recovery_stage_b()
    return algorithm


def _write_checkpoint(path: Path, policy: _TinyPolicy, *, drop_prefix: str | None = None) -> str:
    state = {
        key: value.detach().clone()
        for key, value in policy.state_dict().items()
        if drop_prefix is None or not key.startswith(drop_prefix)
    }
    torch.save({"model_state_dict": state, "iter": 0}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_binding(
    algorithm,
    *,
    source_path: Path,
    source_sha256: str,
    teacher_path: Path,
    teacher_sha256: str,
) -> None:
    algorithm.policy.student_recovery_source_checkpoint = str(source_path.resolve())
    algorithm.policy.student_recovery_source_sha256 = source_sha256
    algorithm.policy.student_recovery_teacher_checkpoint = str(teacher_path.resolve())
    algorithm.policy.student_recovery_teacher_sha256 = teacher_sha256


def _make_bound_algorithm(root: Path):
    student_policy = _TinyPolicy(seed=11)
    teacher_policy = _TinyPolicy(seed=29)
    source_path = root / "student_model_900.pt"
    teacher_path = root / "teacher_model_172300.pt"
    source_sha = _write_checkpoint(source_path, student_policy)
    teacher_sha = _write_checkpoint(teacher_path, teacher_policy)
    algorithm = _make_algorithm(student_policy)
    _configure_binding(
        algorithm,
        source_path=source_path,
        source_sha256=source_sha,
        teacher_path=teacher_path,
        teacher_sha256=teacher_sha,
    )
    manifest = algorithm.bind_student_recovery_checkpoints(
        loaded_student_checkpoint=str(source_path),
        checkpoint_load_mode="weights_only",
    )
    return algorithm, manifest, source_path, source_sha, teacher_path, teacher_sha


def _assert_nested_equal(test_case: unittest.TestCase, left, right) -> None:
    test_case.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(test_case, left_item, right_item)
    else:
        test_case.assertEqual(left, right)


def _stage_b_config_assignments() -> dict[str, object]:
    syntax = ast.parse(RUNNER_CFG_PATH.read_text(encoding="utf-8"))
    target_name = "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorPPORunnerCfg"
    target = next(
        node for node in syntax.body if isinstance(node, ast.ClassDef) and node.name == target_name
    )
    post_init = next(
        node for node in target.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    assignments: dict[str, object] = {}
    for node in post_init.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target_node = node.targets[0]
        if not (
            isinstance(target_node, ast.Attribute)
            and isinstance(target_node.value, ast.Attribute)
            and isinstance(target_node.value.value, ast.Name)
            and target_node.value.value.id == "self"
            and target_node.value.attr == "policy"
        ):
            continue
        assignments[target_node.attr] = ast.literal_eval(node.value)
    return assignments


class HighstepStudentRecoveryStageBTests(unittest.TestCase):
    def test_dual_checkpoint_binding_uses_distinct_sources_and_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            algorithm, manifest, source_path, source_sha, teacher_path, teacher_sha = (
                _make_bound_algorithm(Path(temporary_dir))
            )

            self.assertNotEqual(source_path.resolve(), teacher_path.resolve())
            self.assertEqual(str(source_path.resolve()), manifest["initial_student_checkpoint"])
            self.assertEqual(source_sha, manifest["initial_student_sha256"])
            self.assertEqual(str(teacher_path.resolve()), manifest["teacher_checkpoint"])
            self.assertEqual(teacher_sha, manifest["teacher_sha256"])
            self.assertNotEqual(
                manifest["student_actor_component_sha256"],
                manifest["teacher_actor_component_sha256"],
            )
            self.assertTrue(manifest["teacher_independent_storage"])
            self.assertNotEqual(
                next(algorithm.policy.actor.parameters()).data_ptr(),
                next(algorithm.teacher_actor.parameters()).data_ptr(),
            )
            self.assertNotEqual(
                next(algorithm.policy.priv_encoder.parameters()).data_ptr(),
                next(algorithm.teacher_priv_encoder.parameters()).data_ptr(),
            )

    def test_wrong_checkpoint_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            student = _TinyPolicy(seed=11)
            teacher = _TinyPolicy(seed=29)
            source_path = root / "student.pt"
            teacher_path = root / "teacher.pt"
            source_sha = _write_checkpoint(source_path, student)
            teacher_sha = _write_checkpoint(teacher_path, teacher)
            algorithm = _make_algorithm(student)
            _configure_binding(
                algorithm,
                source_path=source_path,
                source_sha256="0" * 64,
                teacher_path=teacher_path,
                teacher_sha256=teacher_sha,
            )

            with self.assertRaisesRegex(RuntimeError, "Checkpoint SHA256 mismatch"):
                algorithm.bind_student_recovery_checkpoints(
                    loaded_student_checkpoint=str(source_path),
                    checkpoint_load_mode="weights_only",
                )
            self.assertFalse(algorithm._recovery_binding_ready)
            self.assertEqual(source_sha, hashlib.sha256(source_path.read_bytes()).hexdigest())

    def test_missing_required_checkpoint_components_fail_closed(self) -> None:
        cases = (("estimator.", "estimator"), ("priv_encoder.", "priv_encoder"))
        for missing_prefix, expected_message in cases:
            with self.subTest(missing_prefix=missing_prefix), tempfile.TemporaryDirectory() as temporary_dir:
                root = Path(temporary_dir)
                student = _TinyPolicy(seed=11)
                teacher = _TinyPolicy(seed=29)
                source_path = root / "student.pt"
                teacher_path = root / "teacher.pt"
                source_sha = _write_checkpoint(
                    source_path,
                    student,
                    drop_prefix=missing_prefix if missing_prefix == "estimator." else None,
                )
                teacher_sha = _write_checkpoint(
                    teacher_path,
                    teacher,
                    drop_prefix=missing_prefix if missing_prefix == "priv_encoder." else None,
                )
                algorithm = _make_algorithm(student)
                _configure_binding(
                    algorithm,
                    source_path=source_path,
                    source_sha256=source_sha,
                    teacher_path=teacher_path,
                    teacher_sha256=teacher_sha,
                )

                with self.assertRaisesRegex(RuntimeError, expected_message):
                    algorithm.bind_student_recovery_checkpoints(
                        loaded_student_checkpoint=str(source_path),
                        checkpoint_load_mode="weights_only",
                    )
                self.assertFalse(algorithm._recovery_binding_ready)

    def test_optimizer_owns_only_two_leaf_rear_hip_row_parameters(self) -> None:
        algorithm = _make_algorithm()
        owned = [parameter for group in algorithm.optimizer.param_groups for parameter in group["params"]]
        policy_parameter_ids = {id(parameter) for parameter in algorithm.policy.parameters()}

        self.assertEqual(
            [algorithm.recovery_rear_hip_weight, algorithm.recovery_rear_hip_bias],
            owned,
        )
        self.assertEqual([(2, 8), (2,)], [tuple(parameter.shape) for parameter in owned])
        self.assertTrue(all(parameter.is_leaf and parameter.requires_grad for parameter in owned))
        self.assertTrue(all(id(parameter) not in policy_parameter_ids for parameter in owned))
        self.assertTrue(all(not parameter.requires_grad for parameter in algorithm.policy.parameters()))
        self.assertIs(algorithm.optimizer, algorithm.vae_optimizer)

    def test_actor_prefix_preserves_reused_activation_positions(self) -> None:
        algorithm = _make_algorithm()
        actor_modules = list(algorithm.policy.actor)
        deduplicated_children = list(algorithm.policy.actor.children())
        self.assertEqual(7, len(actor_modules))
        self.assertEqual(5, len(deduplicated_children))
        self.assertIs(actor_modules[1], actor_modules[3])
        self.assertIs(actor_modules[3], actor_modules[5])

        actor_input = torch.linspace(-0.5, 0.5, steps=24, dtype=torch.float32).reshape(4, 6)
        hidden, max_error = algorithm._stage_b_actor_prefix(actor_input)
        reconstructed = F.linear(
            hidden,
            algorithm._recovery_last_linear.weight[2:4],
            algorithm._recovery_last_linear.bias[2:4],
        )
        reference = algorithm.policy.actor(actor_input)[:, 2:4]
        torch.testing.assert_close(reconstructed, reference, rtol=1.0e-6, atol=1.0e-6)
        self.assertLessEqual(max_error, 1.0e-6)

    def test_one_update_changes_only_final_actor_rows_two_and_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            algorithm, _, *_ = _make_bound_algorithm(Path(temporary_dir))
            before = {
                key: value.detach().clone() for key, value in algorithm.policy.state_dict().items()
            }

            metrics = algorithm._update_student_recovery_stage_b()
            after = algorithm.policy.state_dict()
            final_weight = f"actor.{algorithm._recovery_last_linear_name}.weight"
            final_bias = f"actor.{algorithm._recovery_last_linear_name}.bias"

            self.assertFalse(torch.equal(before[final_weight][2:4], after[final_weight][2:4]))
            self.assertFalse(torch.equal(before[final_bias][2:4], after[final_bias][2:4]))
            for key in before:
                if key == final_weight:
                    frozen_rows = [row for row in range(before[key].shape[0]) if row not in (2, 3)]
                    self.assertTrue(torch.equal(before[key][frozen_rows], after[key][frozen_rows]), key)
                elif key == final_bias:
                    frozen_rows = [row for row in range(before[key].shape[0]) if row not in (2, 3)]
                    self.assertTrue(torch.equal(before[key][frozen_rows], after[key][frozen_rows]), key)
                else:
                    self.assertTrue(torch.equal(before[key], after[key]), key)

            self.assertEqual(1, algorithm.student_distill_update_count)
            self.assertEqual(0, algorithm.storage.step)
            self.assertEqual(2.0, metrics["Debug/StageB_Optimizer_Parameter_Tensors"])
            self.assertEqual(0.0, metrics["Debug/StageB_Frozen_Tensor_Violation_Count"])
            self.assertTrue(all(parameter.grad is None for parameter in algorithm.policy.parameters()))

    def test_extra_state_roundtrip_restores_count_rows_and_adam_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source, _, source_path, _, teacher_path, _ = _make_bound_algorithm(root)
            source._update_student_recovery_stage_b()
            extra_state = source.extra_checkpoint_state_dict()
            restored_policy = copy.deepcopy(source.policy)
            restored = _make_algorithm(restored_policy, seed=99)

            restored.load_extra_checkpoint_state_dict(copy.deepcopy(extra_state))

            self.assertEqual(source.student_distill_update_count, restored.student_distill_update_count)
            torch.testing.assert_close(
                source.recovery_rear_hip_weight,
                restored.recovery_rear_hip_weight,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                source.recovery_rear_hip_bias,
                restored.recovery_rear_hip_bias,
                rtol=0.0,
                atol=0.0,
            )
            _assert_nested_equal(self, source.optimizer.state_dict(), restored.optimizer.state_dict())
            self.assertTrue(restored._recovery_extra_restored)
            self.assertFalse(restored._recovery_binding_ready)

            # A full resume is not ready to train until the canonical Teacher is
            # independently rebound and the frozen snapshot is captured again.
            restored.bind_student_recovery_checkpoints(
                loaded_student_checkpoint=str(root / "resumed_model_1.pt"),
                checkpoint_load_mode="full",
            )
            self.assertTrue(restored._recovery_binding_ready)
            self.assertEqual(str(source_path.resolve()), restored._recovery_binding_manifest["initial_student_checkpoint"])
            self.assertEqual(str(teacher_path.resolve()), restored._recovery_binding_manifest["teacher_checkpoint"])

    def test_production_config_pins_stage_b_contract_and_canonical_checkpoints(self) -> None:
        assignments = _stage_b_config_assignments()

        self.assertEqual("B", assignments["student_recovery_stage"])
        self.assertEqual(0, assignments["student_actor_warmup_updates"])
        self.assertEqual(0.0, assignments["student_highstep_rear_hip_min_abs"])
        self.assertEqual(0.0, assignments["student_highstep_rear_hip_loss_scale"])
        self.assertEqual(CANONICAL_STUDENT, assignments["student_recovery_source_checkpoint"])
        self.assertEqual(CANONICAL_STUDENT_SHA256, assignments["student_recovery_source_sha256"])
        self.assertEqual(CANONICAL_TEACHER, assignments["student_recovery_teacher_checkpoint"])
        self.assertEqual(CANONICAL_TEACHER_SHA256, assignments["student_recovery_teacher_sha256"])
        self.assertTrue(Path(assignments["student_recovery_source_checkpoint"]).is_absolute())
        self.assertTrue(Path(assignments["student_recovery_teacher_checkpoint"]).is_absolute())


if __name__ == "__main__":
    unittest.main(verbosity=2)
