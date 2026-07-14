"""CPU-only contracts for the immutable highstep Student R2 algorithm."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import torch.nn as nn


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
PREREG_PATH = ROOT / "tmp/highstep_student_recovery_v11_20260712/r2_preregistration.json"
PREREG_SHA256 = "36d39316f8fbba407899a14d1d659873f75b33423464f01ea92000e588c56fc5"
R3_PREREG_PATH = ROOT / "tmp/highstep_student_recovery_v12_20260713/r3_preregistration.json"
R3_PREREG_SHA256 = "13c184e4b6e2c3b514f52466dac1e27ff18256433716fd0c6b5d0890d93945e6"
V12_SPEC_PATH = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
V12_SPEC_SHA256 = "053da1c6d30f9d5ed9aa4c2d5bedbab8d98130edb8a60f8e2657d4a8d0d99352"
ACTIONS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"
ACTIONS_SHA256 = "7f1d340cf53382a7fb3b6281a144e1a75546510aa6fe3078a107d3236ebeb54a"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vae = _load_module("robot_lab_r2_test_target", VAE_PPO_PATH)
VAEPPO = vae.VAEPPO


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def _component_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class _Policy(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        activation = nn.ELU()
        self.actor = nn.Sequential(
            nn.Linear(634, 512),
            activation,
            nn.Linear(512, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 16),
        )
        self.critic = nn.Sequential(nn.Linear(162, 16), nn.ELU(), nn.Linear(16, 1))
        self.std = nn.Parameter(torch.ones(16))
        self.priv_encoder = vae.PrivilegedEncoder(159, 64, [16, 8])
        self.estimator = vae.ProprioVAE(570, vel_dim=3, latent_dim=64, hidden_dims=[16, 128])
        self.distill_stage = 2
        self.student_recovery_stage = "R2"
        self.student_recovery_r2_preregistration_path = ""
        self.student_recovery_r2_preregistration_sha256 = ""
        self.policy_obs_dim = 570
        self.vae_latent_dim = 64
        self.policy_keys = ["policy"]
        self.estimator_keys = ["estimator"]
        self.critic_keys = ["critic"]
        self.teacher_context_keys = ["teacher_context"]
        self.obs_groups = {
            "policy": ["policy"],
            "estimator": ["estimator"],
            "critic": ["critic"],
            "teacher_context": ["teacher_context"],
        }
        self.is_recurrent = False

    def act_inference(self, observations):
        estimator = observations["estimator"]
        mu, _, _ = self.estimator.encode(estimator)
        return self.actor(torch.cat((observations["policy"], torch.clamp(mu, -1.0, 1.0)), dim=-1))


class _Storage:
    def __init__(self, num_envs: int = 2) -> None:
        generator = torch.Generator().manual_seed(20260713)
        self.num_transitions_per_env = 24
        self.step = 24
        self.observations = {
            "policy": torch.randn(24, num_envs, 570, generator=generator) * 0.1,
            "estimator": torch.randn(24, num_envs, 570, generator=generator) * 0.1,
            "critic": torch.randn(24, num_envs, 162, generator=generator) * 0.1,
            "teacher_context": torch.zeros(24, num_envs, 3),
        }
        self.observations["teacher_context"][..., 0] = 0.20
        self.observations["teacher_context"][..., 1] = 0.45
        self.observations["teacher_context"][..., 2] = 0.12

    def clear(self) -> None:
        self.step = 0


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root_policy = _Policy(11)
        self.anchor_policy = copy.deepcopy(self.root_policy)
        with torch.no_grad():
            self.anchor_policy.actor[6].weight[2:4].add_(0.01)
        self.teacher_policy = _Policy(29)
        self.root_checkpoint = root / "model_900.pt"
        self.teacher_checkpoint = root / "model_172300.pt"
        torch.save({"model_state_dict": self.root_policy.state_dict(), "iter": 900}, self.root_checkpoint)
        torch.save({"model_state_dict": self.teacher_policy.state_dict(), "iter": 172300}, self.teacher_checkpoint)
        self.root_sha = _sha(self.root_checkpoint)
        self.teacher_sha = _sha(self.teacher_checkpoint)
        teacher_actor_state = _component_state(self.teacher_policy.state_dict(), "actor.")
        teacher_priv_state = _component_state(self.teacher_policy.state_dict(), "priv_encoder.")

        parent_binding = {
            "initial_student_checkpoint": str(self.root_checkpoint.resolve()),
            "initial_student_sha256": self.root_sha,
            "teacher_checkpoint": str(self.teacher_checkpoint.resolve()),
            "teacher_sha256": self.teacher_sha,
            "teacher_actor_component_sha256": _component_sha(teacher_actor_state),
            "teacher_privileged_encoder_component_sha256": _component_sha(teacher_priv_state),
        }
        self.anchor_checkpoint = root / "model_498.pt"
        torch.save(
            {
                "model_state_dict": self.anchor_policy.state_dict(),
                "iter": 498,
                "infos": {
                    "robot_lab_algorithm_checkpoint_state": {
                        "schema_version": 1,
                        "algorithm_class": "VAEPPO",
                        "distill_stage": 2,
                        "student_distill_update_count": 500,
                        "student_recovery": {
                            "schema_version": 1,
                            "stage": "B",
                            "effective_update_count": 500,
                            "binding_manifest": parent_binding,
                        },
                    }
                },
            },
            self.anchor_checkpoint,
        )
        self.anchor_sha = _sha(self.anchor_checkpoint)

        self.eval_manifest = root / "b500_eval.json"
        self.handoff = root / "handoff.json"
        self.eval_manifest.write_text("{}")
        self.handoff.write_text("{}")
        original = json.loads(PREREG_PATH.read_text())
        # The completed R2 route remains bound to its own immutable v1.1.1
        # authority.  Keep this unit fixture independent of the live formal
        # spec, which may advance after R2 has terminated.
        self.spec = root / "highstep_student_recovery_spec_v111_fixture.md"
        self.spec.write_text("R2 fixture authority: highstep recovery v1.1.1\n")
        original["authority"]["spec_path"] = str(self.spec.resolve())
        original["authority"]["spec_sha256"] = _sha(self.spec)
        action = {
            "class_type": (
                "robot_lab.tasks.locomotion.velocity.mdp.actions:"
                "PhasedHighstepBoxBiasJointPositionAction"
            ),
            "joint_names": original["permanent_contract"]["joint_order"],
            "height_threshold": 0.06,
            "height_gate_width": 0.14,
            "min_forward_command": 0.06,
            "command_gate_width": 0.22,
            "commit_height_delta_min": 0.04,
            "commit_height_delta_target": 0.18,
            "commit_gate_floor": 0.0,
            "front_reach_box_bias": -0.02,
            "rear_approach_box_bias": 0.002,
            "front_support_box_bias": 0.002,
            "rear_push_box_bias": -0.022,
            "min_box_target": 0.0,
            "max_box_target": 0.06,
            "prior_start_update": 80,
            "prior_full_update": 520,
        }
        self.teacher_yaml = root / "teacher_env.yaml"
        self.teacher_yaml.write_text(json.dumps({"actions": {"joint_pos": action}}))

        binding = original["checkpoint_binding"]
        binding["protected_student_root"] = {
            "path": str(self.root_checkpoint.resolve()),
            "sha256": self.root_sha,
        }
        binding["behavior_start_and_anchor"].update(
            {
                "path": str(self.anchor_checkpoint.resolve()),
                "sha256": self.anchor_sha,
                "evaluation_manifest": str(self.eval_manifest.resolve()),
                "evaluation_manifest_sha256": _sha(self.eval_manifest),
                "capped_handoff": str(self.handoff.resolve()),
                "capped_handoff_sha256": _sha(self.handoff),
            }
        )
        binding["frozen_teacher"].update(
            {
                "path": str(self.teacher_checkpoint.resolve()),
                "sha256": self.teacher_sha,
                "env_yaml": str(self.teacher_yaml.resolve()),
                "env_yaml_sha256": _sha(self.teacher_yaml),
            }
        )
        self.prereg = root / "r2_prereg.json"
        self.prereg.write_text(json.dumps(original, indent=2))
        self.prereg_sha = _sha(self.prereg)
        self.runtime_contract = {
            "joint_order": original["permanent_contract"]["joint_order"],
            "action_scale": original["permanent_contract"]["action_scale"],
            "action_offset": [0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4,
            "joint_pos_clip": original["permanent_contract"]["joint_pos_clip"],
            "teacher_context_shape": [3],
            "teacher_context_order": ["height_delta", "command_x", "front_rear_delta"],
        }

    def algorithm(self, policy: _Policy | None = None):
        policy = copy.deepcopy(self.anchor_policy) if policy is None else policy
        policy.student_recovery_r2_preregistration_path = str(self.prereg.resolve())
        policy.student_recovery_r2_preregistration_sha256 = self.prereg_sha
        algorithm = object.__new__(VAEPPO)
        algorithm.policy = policy
        algorithm.device = torch.device("cpu")
        algorithm.distill_stage = 2
        algorithm.student_recovery_stage = "R2"
        algorithm.num_mini_batches = 4
        algorithm.max_grad_norm = 1.0
        algorithm._initialize_student_recovery_r2()
        return algorithm

    def bound_algorithm(self):
        algorithm = self.algorithm()
        manifest = algorithm.bind_student_recovery_r2_checkpoints(
            loaded_student_checkpoint=str(self.anchor_checkpoint),
            checkpoint_load_mode="weights_only",
            schedule_resume_mode="preserve",
            runtime_contract=self.runtime_contract,
        )
        return algorithm, manifest


def _fake_post_prior(**kwargs):
    raw = kwargs["raw_actions"]
    context_push = torch.clamp((kwargs["front_rear_delta"] - 0.04) / 0.14, 0.0, 1.0)
    target = raw.clone()
    target[:, 12:16] = target[:, 12:16] + 0.25 * context_push.unsqueeze(1)
    return SimpleNamespace(raw_equivalent_actions=target, push_gate=context_push)


def _nested_equal(test: unittest.TestCase, left, right) -> None:
    test.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            _nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class HighstepStudentRecoveryR2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = _Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_immutable_prereg_and_shared_helper_are_unchanged(self) -> None:
        self.assertEqual(PREREG_SHA256, _sha(PREREG_PATH))
        self.assertEqual(ACTIONS_SHA256, _sha(ACTIONS_PATH))

    def test_optimizer_owns_exact_eight_tensors_and_freezes_velocity_head(self) -> None:
        algorithm = self.fixture.algorithm()
        owned = [parameter for group in algorithm.optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(8, len(owned))
        self.assertEqual(list(algorithm._r2_exact_optimizer_names()), list(algorithm._r2_optimizer_names))
        self.assertTrue(all(parameter.requires_grad for parameter in owned))
        self.assertTrue(all(not parameter.requires_grad for parameter in algorithm.policy.estimator.vel_estimator.parameters()))
        self.assertIs(algorithm.optimizer, algorithm.vae_optimizer)
        self.assertEqual([1.0e-4, 1.0e-5], [group["lr"] for group in algorithm.optimizer.param_groups])

    def test_weights_only_binding_loads_independent_teacher_and_b500_anchor(self) -> None:
        algorithm, manifest = self.fixture.bound_algorithm()
        self.assertEqual("R2", manifest["stage"])
        self.assertEqual(0, manifest["effective_update_count"])
        self.assertEqual(8, manifest["optimizer_parameter_tensor_count"])
        self.assertTrue(manifest["teacher_independent_storage"])
        self.assertTrue(manifest["anchor_independent_storage"])
        self.assertEqual(0.0, manifest["actor_prefix_equivalence_max_error"])
        self.assertFalse(algorithm.optimizer.state)

    def test_wrong_start_or_schedule_fails_closed(self) -> None:
        algorithm = self.fixture.algorithm()
        with self.assertRaisesRegex(RuntimeError, "schedule preserve"):
            algorithm.bind_student_recovery_r2_checkpoints(
                loaded_student_checkpoint=str(self.fixture.anchor_checkpoint),
                checkpoint_load_mode="weights_only",
                schedule_resume_mode="reset",
                runtime_contract=self.fixture.runtime_contract,
            )
        algorithm = self.fixture.algorithm()
        with self.assertRaisesRegex(RuntimeError, "B500"):
            algorithm.bind_student_recovery_r2_checkpoints(
                loaded_student_checkpoint=str(self.fixture.root_checkpoint),
                checkpoint_load_mode="weights_only",
                schedule_resume_mode="preserve",
                runtime_contract=self.fixture.runtime_contract,
            )

    def test_loss_formula_and_gradient_paths_match_preregistration(self) -> None:
        generator = torch.Generator().manual_seed(7)
        mu = torch.randn(3, 64, generator=generator, requires_grad=True)
        teacher_latent = torch.randn(3, 64, generator=generator)
        velocity = torch.randn(3, 3, generator=generator, requires_grad=True)
        true_velocity = torch.randn(3, 3, generator=generator)
        action = torch.randn(3, 16, generator=generator, requires_grad=True)
        target = torch.randn(3, 16, generator=generator)
        anchor = torch.randn(3, 16, generator=generator)
        push = torch.tensor([0.0, 0.5, 1.0])
        losses = VAEPPO._r2_loss_terms(
            student_mu=mu,
            teacher_latent=teacher_latent,
            velocity_prediction=velocity,
            true_velocity=true_velocity,
            student_action=action,
            teacher_post_prior_action=target,
            b500_action=anchor,
            push_gate=push,
        )
        expected = 50 * losses["latent"] + 10 * losses["velocity"] + 20 * losses["action"] + 5 * losses["anchor"]
        torch.testing.assert_close(losses["total"], expected)
        losses["total"].backward()
        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(velocity.grad)
        self.assertIsNotNone(action.grad)
        self.assertGreater(float(torch.sum(torch.abs(action.grad[:, 12:16]))), 0.0)

    def test_one_update_changes_only_preregistered_scope(self) -> None:
        algorithm, _ = self.fixture.bound_algorithm()
        algorithm.storage = _Storage()
        algorithm._r2_post_prior_function = _fake_post_prior
        before = {key: value.detach().clone() for key, value in algorithm.policy.state_dict().items()}
        metrics = algorithm._update_student_recovery_r2()
        after = algorithm.policy.state_dict()
        allowed = set(algorithm._r2_exact_optimizer_names()[:6])
        for key in before:
            if key in allowed:
                continue
            if key == "actor.6.weight":
                self.assertTrue(torch.equal(before[key][:12], after[key][:12]))
            elif key == "actor.6.bias":
                self.assertTrue(torch.equal(before[key][:12], after[key][:12]))
            else:
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertFalse(torch.equal(before["actor.6.weight"][12:16], after["actor.6.weight"][12:16]))
        self.assertEqual(1, algorithm.student_distill_update_count)
        self.assertEqual(0, algorithm.storage.step)
        self.assertEqual(8.0, metrics["Debug/R2_Optimizer_Parameter_Tensors"])
        self.assertEqual(1.0, metrics["Debug/R2_Deterministic_Mean_Rollout"])
        self.assertTrue(all(parameter.grad is None for parameter in algorithm.policy.estimator.vel_estimator.parameters()))

    def test_full_resume_restores_adam_count_and_box_leaves(self) -> None:
        source, _ = self.fixture.bound_algorithm()
        source.storage = _Storage()
        source._r2_post_prior_function = _fake_post_prior
        source._update_student_recovery_r2()
        extra = copy.deepcopy(source.extra_checkpoint_state_dict())
        restored_policy = copy.deepcopy(source.policy)
        restored = self.fixture.algorithm(restored_policy)
        restored.load_extra_checkpoint_state_dict(copy.deepcopy(extra))
        self.assertEqual(1, restored.student_distill_update_count)
        self.assertTrue(restored._r2_extra_restored)
        _nested_equal(self, source.optimizer.state_dict(), restored.optimizer.state_dict())
        manifest = restored.bind_student_recovery_r2_checkpoints(
            loaded_student_checkpoint=str(self.fixture.root / "r2_model_0.pt"),
            checkpoint_load_mode="full",
            schedule_resume_mode="preserve",
            runtime_contract=self.fixture.runtime_contract,
        )
        self.assertEqual(1, manifest["effective_update_count"])
        self.assertTrue(restored._r2_binding_ready)

    def test_full_resume_rejects_malformed_adam_groups_moments_and_steps(self) -> None:
        source, _ = self.fixture.bound_algorithm()
        source.storage = _Storage()
        source._r2_post_prior_function = _fake_post_prior
        source._update_student_recovery_r2()
        pristine = copy.deepcopy(source.extra_checkpoint_state_dict())

        def bad_group_name(state):
            state["vae_optimizer_state_dict"]["param_groups"][0]["name"] = "wrong"

        def bad_learning_rate(state):
            state["vae_optimizer_state_dict"]["param_groups"][1]["lr"] = 2.0e-5

        def missing_state_entry(state):
            adam = state["vae_optimizer_state_dict"]["state"]
            adam.pop(next(iter(adam)))

        def nonfinite_moment(state):
            slot = next(iter(state["vae_optimizer_state_dict"]["state"].values()))
            slot["exp_avg"].reshape(-1)[0] = float("nan")

        def wrong_moment_shape(state):
            slot = next(iter(state["vae_optimizer_state_dict"]["state"].values()))
            slot["exp_avg_sq"] = torch.zeros(1)

        def wrong_step(state):
            slot = next(iter(state["vae_optimizer_state_dict"]["state"].values()))
            slot["step"] = torch.tensor(5.0)

        cases = (
            ("bad_group_name", bad_group_name, "group structure"),
            ("bad_learning_rate", bad_learning_rate, "hyperparameters"),
            ("missing_state_entry", missing_state_entry, "exactly eight state entries"),
            ("nonfinite_moment", nonfinite_moment, "non-finite"),
            ("wrong_moment_shape", wrong_moment_shape, "shape mismatch"),
            ("wrong_step", wrong_step, "step/count mismatch"),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                corrupted = copy.deepcopy(pristine)
                mutate(corrupted)
                restored = self.fixture.algorithm(copy.deepcopy(source.policy))
                with self.assertRaisesRegex(RuntimeError, message):
                    restored.load_extra_checkpoint_state_dict(corrupted)

    def test_full_resume_rejects_frozen_scope_escape(self) -> None:
        source, _ = self.fixture.bound_algorithm()
        source.storage = _Storage()
        source._r2_post_prior_function = _fake_post_prior
        source._update_student_recovery_r2()
        extra = copy.deepcopy(source.extra_checkpoint_state_dict())
        escaped_policy = copy.deepcopy(source.policy)
        with torch.no_grad():
            escaped_policy.critic[0].weight.add_(0.1)
        restored = self.fixture.algorithm(escaped_policy)
        restored.load_extra_checkpoint_state_dict(extra)
        with self.assertRaisesRegex(RuntimeError, "scope escaped B500"):
            restored.bind_student_recovery_r2_checkpoints(
                loaded_student_checkpoint=str(self.fixture.root / "escaped.pt"),
                checkpoint_load_mode="full",
                schedule_resume_mode="preserve",
                runtime_contract=self.fixture.runtime_contract,
            )

    def test_runner_cfg_declares_dedicated_r2_route(self) -> None:
        source = RUNNER_CFG_PATH.read_text()
        self.assertIn(
            "class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorR2PPORunnerCfg",
            source,
        )
        self.assertIn('self.policy.student_recovery_stage = "R2"', source)
        self.assertIn('"teacher_context": ["teacher_context"]', source)
        self.assertIn(PREREG_SHA256, source)

    def test_deterministic_rollout_and_no_compute_returns_are_explicit(self) -> None:
        source = VAE_PPO_PATH.read_text()
        self.assertIn("actions = self.policy.act_inference(obs).detach()", source)
        self.assertNotIn("self.policy.act(obs)", source[source.index("    def act(self, obs):"):source.index("    def compute_returns", source.index("    def act(self, obs):"))])
        self.assertIn("Recovery DAgger stages have no PPO/reward objective", source)
        self.assertIn("self.estimator.encode(estimator_obs)", source)

    def test_deterministic_rollout_initializes_logging_distribution_without_sampling(self) -> None:
        algorithm, _ = self.fixture.bound_algorithm()
        algorithm.transition = SimpleNamespace()
        observations = {
            "policy": torch.linspace(-0.2, 0.2, steps=2 * 570).reshape(2, 570),
            "estimator": torch.linspace(0.2, -0.2, steps=2 * 570).reshape(2, 570),
            "critic": torch.zeros(2, 162),
            "teacher_context": torch.zeros(2, 3),
        }
        expected = algorithm.policy.act_inference(observations).detach()
        rng_before = torch.random.get_rng_state().clone()
        actions = algorithm.act(observations)
        rng_after = torch.random.get_rng_state()

        self.assertTrue(torch.equal(actions, expected))
        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertTrue(torch.equal(algorithm.policy.distribution.mean, expected))
        self.assertTrue(torch.equal(algorithm.transition.action_mean, expected))
        self.assertTrue(torch.equal(algorithm.transition.action_sigma, torch.ones_like(expected)))
        self.assertTrue(torch.equal(algorithm.policy.distribution.stddev, torch.ones_like(expected)))


class HighstepStudentRecoveryR3Tests(unittest.TestCase):
    def _algorithm(self):
        policy = _Policy(43)
        policy.student_recovery_stage = "R3"
        policy.student_recovery_r3_preregistration_path = str(R3_PREREG_PATH.resolve())
        policy.student_recovery_r3_preregistration_sha256 = R3_PREREG_SHA256
        algorithm = object.__new__(VAEPPO)
        algorithm.policy = policy
        algorithm.device = torch.device("cpu")
        algorithm.distill_stage = 2
        algorithm.student_recovery_stage = "R3"
        algorithm.num_mini_batches = 4
        algorithm.max_grad_norm = 1.0
        algorithm._initialize_student_recovery_r3()
        return algorithm

    def test_v12_authority_and_r3_preregistration_are_immutable(self) -> None:
        # R3 is terminal and the live formal authority has advanced to v1.3;
        # its immutable preregistration remains preserved as historical proof.
        self.assertNotEqual(V12_SPEC_SHA256, _sha(V12_SPEC_PATH))
        self.assertEqual(R3_PREREG_SHA256, _sha(R3_PREREG_PATH))
        self.assertEqual(0, R3_PREREG_PATH.stat().st_mode & 0o222)

    def test_r3_optimizer_owns_only_full_actor_head_and_freezes_estimator(self) -> None:
        algorithm = self._algorithm()
        self.assertEqual(["actor.6.weight", "actor.6.bias"], [
            name for name, parameter in algorithm.policy.named_parameters() if parameter.requires_grad
        ])
        self.assertEqual(2, len(algorithm.optimizer.param_groups[0]["params"]))
        self.assertEqual(5.0e-6, algorithm.optimizer.param_groups[0]["lr"])
        self.assertTrue(all(not parameter.requires_grad for parameter in algorithm.policy.estimator.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in algorithm.policy.actor[:6].parameters()))
        self.assertIs(algorithm.optimizer, algorithm.vae_optimizer)

    def test_r3_loss_is_exact_lateral_teacher_plus_four_center_anchor(self) -> None:
        student = torch.tensor([[1.0] * 16, [2.0] * 16], requires_grad=True)
        teacher = torch.zeros_like(student)
        anchor = torch.ones_like(student)
        gate = torch.tensor([1.0, 0.0])
        losses = VAEPPO._r3_loss_terms(
            student_action=student,
            teacher_post_prior_action=teacher,
            anchor_action=anchor,
            lateral_gate=gate,
        )
        self.assertEqual(1.0, float(losses["teacher_offset"]))
        self.assertEqual(1.0, float(losses["anchor_center"]))
        self.assertEqual(5.0, float(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(student.grad)

    def test_r3_update_changes_only_actor_head_and_preserves_deterministic_rollout(self) -> None:
        algorithm = self._algorithm()
        algorithm.r3_anchor_actor = copy.deepcopy(algorithm.policy.actor).eval()
        algorithm.r3_anchor_estimator = copy.deepcopy(algorithm.policy.estimator).eval()
        algorithm.r3_teacher_actor = copy.deepcopy(algorithm.policy.actor).eval()
        algorithm.r3_teacher_priv_encoder = copy.deepcopy(algorithm.policy.priv_encoder).eval()
        with torch.no_grad():
            algorithm.r3_teacher_actor[6].weight.add_(0.02)
        for module in (
            algorithm.r3_anchor_actor,
            algorithm.r3_anchor_estimator,
            algorithm.r3_teacher_actor,
            algorithm.r3_teacher_priv_encoder,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        algorithm._r2_action_scale = torch.tensor([0.1] * 12 + [0.02] * 4).unsqueeze(0)
        algorithm._r2_action_offset = torch.tensor(
            [0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4
        ).unsqueeze(0)
        algorithm._r2_action_clip = torch.tensor([[[-60.0, 60.0]] * 16])
        algorithm._r2_prior_contract = {
            "box_action_ids": torch.tensor([12, 13, 14, 15], dtype=torch.long)
        }
        algorithm._r2_post_prior_function = _fake_post_prior
        algorithm._r3_binding_manifest = {"effective_update_count": 0}
        algorithm._r3_binding_ready = True
        algorithm._capture_r3_frozen_snapshot()
        storage = _Storage(num_envs=4)
        scan = storage.observations["critic"][..., 60:].reshape(24, 4, 6, 17)
        scan[:, :2, 3:, :] += 0.20
        algorithm.storage = storage
        before = {key: value.detach().clone() for key, value in algorithm.policy.state_dict().items()}
        metrics = algorithm._update_student_recovery_r3()
        after = algorithm.policy.state_dict()
        for key in before:
            if key not in {"actor.6.weight", "actor.6.bias"}:
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertFalse(torch.equal(before["actor.6.weight"], after["actor.6.weight"]))
        self.assertEqual(1, algorithm.student_distill_update_count)
        self.assertEqual(0, algorithm.storage.step)
        self.assertEqual(2.0, metrics["Debug/R3_Optimizer_Parameter_Tensors"])

        algorithm.transition = SimpleNamespace()
        observations = {
            "policy": torch.zeros(2, 570),
            "estimator": torch.zeros(2, 570),
            "critic": torch.zeros(2, 162),
            "teacher_context": torch.zeros(2, 3),
        }
        expected = algorithm.policy.act_inference(observations).detach()
        rng_before = torch.random.get_rng_state().clone()
        actual = algorithm.act(observations)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
