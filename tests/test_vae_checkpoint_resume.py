"""CPU-only tests for VAEPPO optimizer/count checkpoint continuity."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_HELPER_PATH = ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py"
VAE_PPO_PATH = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped"
    / "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
TRAIN_PATH = ROOT / "scripts/rsl_rl/base/train.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint_support = _load_module("robot_lab_algorithm_checkpoint_test_target", CHECKPOINT_HELPER_PATH)
vae_ppo_module = _load_module("robot_lab_vae_ppo_checkpoint_test_target", VAE_PPO_PATH)
VAEPPO = vae_ppo_module.VAEPPO


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.estimator = nn.Linear(3, 2)
        self.actor = nn.Linear(2, 1)


def _make_algorithm(seed: int):
    torch.manual_seed(seed)
    algorithm = object.__new__(VAEPPO)
    algorithm.policy = _TinyPolicy()
    algorithm.optimizer = torch.optim.Adam(algorithm.policy.parameters(), lr=2.0e-4)
    algorithm.distill_stage = 2
    algorithm.vae_optimizer = torch.optim.Adam(
        [
            {"params": algorithm.policy.estimator.parameters(), "lr": 1.0e-3},
            {"params": algorithm.policy.actor.parameters(), "lr": 1.0e-5},
        ]
    )
    algorithm.student_distill_update_count = 0
    algorithm._teacher_actor_synced = False
    return algorithm


def _optimizer_step(algorithm, gradient: float) -> None:
    algorithm.vae_optimizer.zero_grad()
    for parameter in algorithm.policy.parameters():
        parameter.grad = torch.full_like(parameter, gradient)
    algorithm.vae_optimizer.step()


def _make_verified_legacy_teacher(seed: int):
    """Tiny Stage-1 analogue with 23 restored PPO states and an empty VAE Adam."""
    torch.manual_seed(seed)
    teacher_type = type("VAEPPO", (), {})
    algorithm = teacher_type()
    algorithm.policy = nn.ParameterList([nn.Parameter(torch.randn(())) for _ in range(23)])
    algorithm.optimizer = torch.optim.Adam(algorithm.policy.parameters(), lr=1.0e-5)
    algorithm.vae_parameter = nn.Parameter(torch.randn(()))
    algorithm.vae_optimizer = torch.optim.Adam([algorithm.vae_parameter], lr=1.0e-4)
    algorithm.distill_stage = 1
    algorithm.student_recovery_stage = "NONE"
    algorithm.extra_checkpoint_state_dict = lambda: {
        "schema_version": 1,
        "algorithm_class": "VAEPPO",
        "distill_stage": 1,
        "vae_optimizer_state_dict": algorithm.vae_optimizer.state_dict(),
    }
    algorithm.load_extra_checkpoint_state_dict = lambda state: algorithm.vae_optimizer.load_state_dict(
        state["vae_optimizer_state_dict"]
    )
    algorithm.optimizer.zero_grad()
    for parameter in algorithm.policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    algorithm.optimizer.step()
    return algorithm


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _teacher_contract(path: Path, *, iteration: int = 600) -> dict:
    return {
        "kind": "verified_legacy_teacher_algorithm_state_v1",
        "algorithm_class": "VAEPPO",
        "distill_stage": 1,
        "student_recovery_stage": "NONE",
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": _sha256(path),
        "checkpoint_iteration": iteration,
        "stock_optimizer_state_entries": 23,
        "stock_optimizer_param_groups": 1,
        "vae_optimizer_state_entries": 0,
        "vae_optimizer_param_groups": 1,
        "stage1_update_contract": "super_ppo_only_no_vae_optimizer_step",
    }


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


class _StockLikeRunner:
    """Small CPU runner with the same checkpoint contract as OnPolicyRunner."""

    def __init__(self, algorithm) -> None:
        self.alg = algorithm
        self.current_learning_iteration = 0

    def save(self, path: str, infos: dict | None = None) -> None:
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_learning_iteration = int(checkpoint["iter"])
        return checkpoint["infos"]


class VAEPPOCheckpointResumeTests(unittest.TestCase):
    def _save_new_checkpoint(self, path: Path, *, count: int = 317):
        algorithm = _make_algorithm(seed=7)
        _optimizer_step(algorithm, gradient=0.25)
        _optimizer_step(algorithm, gradient=-0.10)
        algorithm.student_distill_update_count = count
        runner = _StockLikeRunner(algorithm)
        runner.current_learning_iteration = count - 1
        checkpoint_support.install_algorithm_checkpoint_state_hook(runner)
        runner.save(str(path), {"caller_metadata": "kept"})
        return algorithm

    def test_new_checkpoint_roundtrip_restores_optimizer_count_and_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model_316.pt"
            source = self._save_new_checkpoint(checkpoint_path)
            raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            infos = raw_checkpoint["infos"]
            self.assertEqual("kept", infos["caller_metadata"])
            extra = infos[checkpoint_support.ALGORITHM_STATE_INFO_KEY]
            self.assertEqual(1, extra["schema_version"])
            self.assertEqual(317, extra["student_distill_update_count"])
            self.assertGreater(len(extra["vae_optimizer_state_dict"]["state"]), 0)

            restored = _make_algorithm(seed=99)
            restored_runner = _StockLikeRunner(restored)
            checkpoint_support.install_algorithm_checkpoint_state_hook(restored_runner)
            restored_infos = restored_runner.load(str(checkpoint_path), map_location="cpu")

            self.assertEqual("kept", restored_infos["caller_metadata"])
            self.assertEqual(316, restored_runner.current_learning_iteration)
            self.assertEqual(317, restored.student_distill_update_count)
            self.assertFalse(restored._teacher_actor_synced)
            _assert_nested_equal(
                self,
                source.vae_optimizer.state_dict(),
                restored.vae_optimizer.state_dict(),
            )
            for source_parameter, restored_parameter in zip(
                source.policy.parameters(), restored.policy.parameters(), strict=True
            ):
                torch.testing.assert_close(source_parameter, restored_parameter, rtol=0.0, atol=0.0)

            # An identical next gradient must produce an identical Adam update.
            _optimizer_step(source, gradient=0.4)
            _optimizer_step(restored, gradient=0.4)
            for source_parameter, restored_parameter in zip(
                source.policy.parameters(), restored.policy.parameters(), strict=True
            ):
                torch.testing.assert_close(source_parameter, restored_parameter, rtol=0.0, atol=0.0)

    def test_weights_only_load_does_not_restore_vae_optimizer_or_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model_316.pt"
            source = self._save_new_checkpoint(checkpoint_path)
            restored = _make_algorithm(seed=99)
            restored_runner = _StockLikeRunner(restored)
            checkpoint_support.install_algorithm_checkpoint_state_hook(restored_runner)

            restored_runner.load(str(checkpoint_path), load_optimizer=False, map_location="cpu")

            self.assertEqual(0, restored.student_distill_update_count)
            self.assertEqual(0, len(restored.vae_optimizer.state))
            for source_parameter, restored_parameter in zip(
                source.policy.parameters(), restored.policy.parameters(), strict=True
            ):
                torch.testing.assert_close(source_parameter, restored_parameter, rtol=0.0, atol=0.0)

    def test_legacy_full_resume_fails_closed_without_explicit_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "legacy_model_600.pt"
            source = _make_algorithm(seed=7)
            _optimizer_step(source, gradient=0.25)
            source.student_distill_update_count = 601
            legacy_runner = _StockLikeRunner(source)
            legacy_runner.current_learning_iteration = 600
            legacy_runner.save(str(checkpoint_path))

            restored = _make_algorithm(seed=99)
            restored_runner = _StockLikeRunner(restored)
            checkpoint_support.install_algorithm_checkpoint_state_hook(restored_runner)
            with self.assertRaisesRegex(
                checkpoint_support.AlgorithmCheckpointStateMissingError,
                "Full resume requires algorithm-owned checkpoint state",
            ):
                restored_runner.load(str(checkpoint_path), map_location="cpu")

    def test_verified_legacy_checkpoint_requires_explicit_count_and_fresh_adam(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "legacy_model_600.pt"
            source = _make_algorithm(seed=7)
            _optimizer_step(source, gradient=0.25)
            legacy_runner = _StockLikeRunner(source)
            legacy_runner.current_learning_iteration = 600
            legacy_runner.save(str(checkpoint_path))

            messages: list[str] = []
            restored = _make_algorithm(seed=99)
            restored_runner = _StockLikeRunner(restored)
            checkpoint_support.install_algorithm_checkpoint_state_hook(
                restored_runner,
                legacy_student_distill_update_count=601,
                log=messages.append,
            )
            restored_runner.load(str(checkpoint_path), map_location="cpu")

            self.assertEqual(601, restored.student_distill_update_count)
            self.assertEqual(0, len(restored.vae_optimizer.state))
            self.assertEqual(1, len(messages))
            self.assertIn("optimizer moments remain freshly initialized", messages[0])

    def test_explicit_legacy_override_is_rejected_for_new_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "model_316.pt"
            self._save_new_checkpoint(checkpoint_path)
            restored_runner = _StockLikeRunner(_make_algorithm(seed=99))
            checkpoint_support.install_algorithm_checkpoint_state_hook(
                restored_runner,
                legacy_student_distill_update_count=317,
            )
            with self.assertRaisesRegex(
                checkpoint_support.AlgorithmCheckpointError,
                "already contains resumable algorithm state",
            ):
                restored_runner.load(str(checkpoint_path), map_location="cpu")

    def test_verified_legacy_teacher_restores_ppo_adam_and_exact_empty_vae_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "legacy_teacher_600.pt"
            source = _make_verified_legacy_teacher(seed=7)
            source_runner = _StockLikeRunner(source)
            source_runner.current_learning_iteration = 600
            source_runner.save(str(checkpoint_path))

            restored = _make_verified_legacy_teacher(seed=99)
            # Reset the newly created PPO moments so runner.load must restore them.
            restored.optimizer.state.clear()
            restored_runner = _StockLikeRunner(restored)
            messages: list[str] = []
            checkpoint_support.install_algorithm_checkpoint_state_hook(
                restored_runner,
                verified_legacy_teacher_checkpoint=_teacher_contract(checkpoint_path),
                log=messages.append,
            )
            restored_runner.load(str(checkpoint_path), map_location="cpu")

            _assert_nested_equal(self, source.optimizer.state_dict(), restored.optimizer.state_dict())
            self.assertEqual(0, len(restored.vae_optimizer.state))
            self.assertEqual(1, len(messages))
            self.assertIn("23-state PPO Adam", messages[0])

    def test_verified_legacy_teacher_wrong_sha_fails_before_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "legacy_teacher_600.pt"
            source_runner = _StockLikeRunner(_make_verified_legacy_teacher(seed=7))
            source_runner.current_learning_iteration = 600
            source_runner.save(str(checkpoint_path))
            contract = _teacher_contract(checkpoint_path)
            contract["checkpoint_sha256"] = "0" * 64
            restored_runner = _StockLikeRunner(_make_verified_legacy_teacher(seed=99))
            checkpoint_support.install_algorithm_checkpoint_state_hook(
                restored_runner, verified_legacy_teacher_checkpoint=contract
            )
            with self.assertRaisesRegex(
                checkpoint_support.AlgorithmCheckpointError, "SHA256 changed before runner.load"
            ):
                restored_runner.load(str(checkpoint_path), map_location="cpu")
            self.assertEqual(0, restored_runner.current_learning_iteration)

    def test_verified_legacy_teacher_rejects_nonempty_vae_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "legacy_teacher_600.pt"
            source_runner = _StockLikeRunner(_make_verified_legacy_teacher(seed=7))
            source_runner.current_learning_iteration = 600
            source_runner.save(str(checkpoint_path))
            restored = _make_verified_legacy_teacher(seed=99)
            restored.vae_optimizer.zero_grad()
            restored.vae_parameter.grad = torch.ones_like(restored.vae_parameter)
            restored.vae_optimizer.step()
            restored_runner = _StockLikeRunner(restored)
            checkpoint_support.install_algorithm_checkpoint_state_hook(
                restored_runner,
                verified_legacy_teacher_checkpoint=_teacher_contract(checkpoint_path),
            )
            with self.assertRaisesRegex(
                checkpoint_support.AlgorithmCheckpointError, "never-stepped empty state"
            ):
                restored_runner.load(str(checkpoint_path), map_location="cpu")

    def test_verified_legacy_teacher_contract_is_not_valid_for_student(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "legacy_teacher_600.pt"
            source_runner = _StockLikeRunner(_make_verified_legacy_teacher(seed=7))
            source_runner.current_learning_iteration = 600
            source_runner.save(str(checkpoint_path))
            restored = _make_verified_legacy_teacher(seed=99)
            restored.distill_stage = 2
            restored_runner = _StockLikeRunner(restored)
            checkpoint_support.install_algorithm_checkpoint_state_hook(
                restored_runner,
                verified_legacy_teacher_checkpoint=_teacher_contract(checkpoint_path),
            )
            with self.assertRaisesRegex(
                checkpoint_support.AlgorithmCheckpointError, "runtime distill stage changed"
            ):
                restored_runner.load(str(checkpoint_path), map_location="cpu")

    def test_train_installs_algorithm_hook_before_load_and_schedule_sha_hook(self) -> None:
        source = TRAIN_PATH.read_text(encoding="utf-8")
        install_call = source.index(
            "    install_algorithm_checkpoint_state_hook(\n",
            source.index("# Install before any runner.load()"),
        )
        first_load = source.index("runner.load(resume_path", install_call)
        schedule_hook = source.index("            _install_highstep_checkpoint_state_hook(", first_load)
        self.assertLess(install_call, first_load)
        self.assertLess(first_load, schedule_hook)


if __name__ == "__main__":
    unittest.main(verbosity=2)
