"""CPU-only correctness tests for the v1.1 same-state audit core."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
from typing import NamedTuple
import unittest

import torch
import torch.nn.functional as F

from tools import highstep_same_state_audit as audit


ROOT = Path(__file__).resolve().parents[1]
ACTIONS_PATH = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"
)


def _load_live_post_prior_without_isaac_imports():
    """Compile the exact pure helper nodes while deliberately skipping Isaac imports."""
    tree = ast.parse(ACTIONS_PATH.read_text())
    wanted = {
        "PhasedHighstepPostPriorResult",
        "_phased_highstep_smoothstep",
        "compute_phased_highstep_post_prior",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    if {getattr(node, "name", None) for node in nodes} != wanted:
        raise AssertionError("live actions.py pure post-prior helper nodes are incomplete")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {
        "torch": torch,
        "NamedTuple": NamedTuple,
        "_prior_scale": lambda update, start, full: max(
            0.0,
            min(1.0, (float(update) - float(start)) / max(float(full) - float(start), 1.0)),
        ),
    }
    exec(compile(module, str(ACTIONS_PATH), "exec"), namespace)
    return namespace["compute_phased_highstep_post_prior"]


def _prior_inputs() -> audit.PostPriorInputs:
    scale = torch.tensor([[0.1] * 12 + [0.02] * 4], dtype=torch.float32)
    offset = torch.tensor([[0.0] * 12 + [0.03] * 4], dtype=torch.float32)
    clip = torch.tensor([[[-60.0, 60.0]] * 16], dtype=torch.float32)
    return audit.PostPriorInputs(
        scale=scale,
        offset=offset,
        clip=clip,
        height_delta=torch.tensor([0.20]),
        command_x=torch.tensor([0.40]),
        front_rear_delta=torch.tensor([0.02]),
        terrain_gate_available=True,
    )


def _write_plan(directory: Path) -> tuple[str, str]:
    path = directory / "base3_plan.json"
    audit._atomic_json(path, audit.base_suite_plan_payload(), immutable=True)
    return str(path), audit.sha256_file(path)


class HighstepSameStateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The runtime audit remains SHA-frozen to its historical v1.5
        # authority.  Structural CPU tests explicitly rebind only their
        # in-process test fixture to the advanced, higher-authority spec.
        cls.historical_spec_sha256 = audit.EXPECTED_SPEC_SHA256
        audit.EXPECTED_SPEC_SHA256 = hashlib.sha256(audit.SPEC_PATH.read_bytes()).hexdigest()
        cls.models = audit.BoundAuditModels(device="cpu")
        cls.live_post_prior = staticmethod(_load_live_post_prior_without_isaac_imports())

    @classmethod
    def tearDownClass(cls) -> None:
        audit.EXPECTED_SPEC_SHA256 = cls.historical_spec_sha256

    def test_b500_root_teacher_binding_is_exact_and_frozen(self) -> None:
        binding = self.models.binding_manifest
        self.assertEqual(binding["student_checkpoint_sha256"], audit.CURRENT_STUDENT_SHA256)
        self.assertEqual(binding["root_student_checkpoint_sha256"], audit.ROOT_STUDENT_SHA256)
        self.assertEqual(binding["teacher_checkpoint_sha256"], audit.TEACHER_SHA256)
        self.assertEqual(binding["student_actor_component_sha256"], audit.CURRENT_STUDENT_ACTOR_SHA256)
        self.assertEqual(binding["root_student_actor_component_sha256"], audit.ROOT_STUDENT_ACTOR_SHA256)
        self.assertEqual(binding["student_estimator_component_sha256"], audit.ROOT_STUDENT_ESTIMATOR_SHA256)
        self.assertEqual(binding["teacher_actor_component_sha256"], audit.TEACHER_ACTOR_SHA256)
        self.assertEqual(
            binding["teacher_privileged_encoder_component_sha256"],
            audit.TEACHER_PRIVILEGED_ENCODER_SHA256,
        )
        self.assertTrue(binding["b500_lineage_to_model900_verified"])
        self.assertTrue(binding["student_teacher_storage_independent"])
        self.assertTrue(all(not parameter.requires_grad for parameter in self.models.parameters()))

    def test_forward_shapes_anchor_and_finiteness(self) -> None:
        generator = torch.Generator().manual_seed(17)
        outputs = self.models.forward_same_state(
            torch.randn(1, 570, generator=generator),
            torch.randn(1, 570, generator=generator),
            torch.randn(1, 162, generator=generator),
        )
        for name in (
            "student_action",
            "student_action_teacher_latent",
            "teacher_action_pre_prior",
            "root_student_action",
        ):
            self.assertEqual(tuple(outputs[name].shape), (1, 16))
        for name in ("student_mu", "student_latent", "teacher_latent", "root_student_latent"):
            self.assertEqual(tuple(outputs[name].shape), (1, 64))
        self.assertEqual(tuple(outputs["student_velocity"].shape), (1, 3))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

    def test_student_estimator_reconstruction_includes_final_encoder_elu(self) -> None:
        checkpoint, _ = audit._load_checkpoint(
            audit.CURRENT_STUDENT_CHECKPOINT,
            audit.CURRENT_STUDENT_SHA256,
        )
        state = checkpoint["model_state_dict"]
        generator = torch.Generator().manual_seed(1707)
        observations = torch.randn(1, 570, generator=generator)
        hidden = F.elu(
            F.linear(
                observations,
                state["estimator.encoder.0.weight"],
                state["estimator.encoder.0.bias"],
            )
        )
        hidden = F.elu(
            F.linear(
                hidden,
                state["estimator.encoder.2.weight"],
                state["estimator.encoder.2.bias"],
            )
        )
        expected_mu = F.linear(
            hidden,
            state["estimator.fc_mu.weight"],
            state["estimator.fc_mu.bias"],
        )
        expected_velocity = F.linear(
            hidden,
            state["estimator.vel_estimator.weight"],
            state["estimator.vel_estimator.bias"],
        )
        actual_mu, _, actual_velocity = self.models.student_estimator(observations)
        self.assertTrue(torch.equal(actual_mu, expected_mu))
        self.assertTrue(torch.equal(actual_velocity, expected_velocity))

    def test_exact_live_post_prior_adapter_reports_raw_and_physical_actions(self) -> None:
        raw = torch.zeros(1, 16)
        result = audit._call_shared_post_prior(
            self.live_post_prior,
            raw,
            _prior_inputs(),
            audit.PostPriorContract(),
        )
        self.assertEqual(tuple(result.raw_equivalent_actions.shape), (1, 16))
        self.assertEqual(tuple(result.physical_targets.shape), (1, 16))
        self.assertEqual(tuple(result.box_clip_mask.shape), (1, 16))
        self.assertEqual(result.prior_scale, 1.0)
        self.assertTrue(torch.equal(result.raw_equivalent_actions[:, :12], raw[:, :12]))
        self.assertFalse(torch.equal(result.raw_equivalent_actions[:, 12:], raw[:, 12:]))

    def test_session_enforces_parity_state_digest_decomposition_and_replacement(self) -> None:
        generator = torch.Generator().manual_seed(23)
        policy = torch.randn(1, 570, generator=generator) * 0.1
        estimator = torch.randn(1, 570, generator=generator) * 0.1
        critic = torch.randn(1, 162, generator=generator) * 0.1
        critic[:, :3] = torch.tensor([[0.4, -0.1, 0.2]])
        all_trials = (
            audit.CausalReplacement("rear_hips_probe", "rear_hips", ("approach",)),
        )
        preregistration = audit.AuditPreregistration(
            run_id="unit_seed11_nominal",
            seed=11,
            scenario="nominal",
            suite_plan_path="placeholder",
            suite_plan_sha256="0" * 64,
            suite_kind="causal",
            replacement=all_trials[0],
            all_replacement_trials=all_trials,
            thresholds=audit.AuditThresholds(consecutive_frames=1),
        )
        with tempfile.TemporaryDirectory() as temporary:
            plan_path, plan_sha = _write_plan(Path(temporary))
            preregistration = dataclasses.replace(
                preregistration,
                suite_plan_path=plan_path,
                suite_plan_sha256=plan_sha,
            )
            session = audit.SameStateAuditSession(
                models=self.models,
                output_dir=Path(temporary),
                preregistration=preregistration,
                post_prior_function=self.live_post_prior,
            )

            def runner_action() -> torch.Tensor:
                return self.models.forward_same_state(policy, estimator, critic)["student_action"]

            action = session.evaluate_frame(
                step=0,
                policy_observations=policy,
                estimator_observations=estimator,
                critic_observations=critic,
                true_base_linear_velocity=critic[:, :3] / 2.0,
                prior_inputs=_prior_inputs(),
                phase_signals=audit.PhaseSignals(),
                runner_inference_function=runner_action,
                physical_state_digest_function=lambda: "unchanged",
            )
            record = session.records[0]
            teacher_target = torch.tensor(record["teacher_action_post_prior_raw_equivalent"])
            student = torch.tensor(record["student_action"])
            self.assertTrue(torch.equal(action[0, 2:4], teacher_target[2:4]))
            self.assertTrue(torch.allclose(action[0, :2], student[:2]))
            self.assertTrue(record["runner_manual_action_allclose"])
            self.assertLessEqual(record["runner_manual_action_max_abs_error"], 1.0e-6)
            self.assertLessEqual(record["action_decomposition_residual_max_abs"], 1.0e-6)
            self.assertTrue(record["physical_state_unchanged_by_forward"])
            self.assertEqual(len(record["teacher_prior_box_clip_mask"]), 16)
            self.assertEqual(len(record["velocity_error_estimate_minus_critic_target"]), 3)
            self.assertEqual(len(record["velocity_error_estimate_divide_2_minus_root"]), 3)
            self.assertGreaterEqual(record["student_mu_out_of_bounds_ratio"], 0.0)
            summary = session.finalize(outcome={"full_climb_success": False})
            self.assertTrue(summary["all_forward_state_digests_unchanged"])
            self.assertTrue(summary["runner_manual_action_allclose_all_frames"])
            self.assertLessEqual(summary["action_decomposition_residual_global_max_abs"], 1.0e-6)
            self.assertTrue((Path(temporary) / "preregistration.json").exists())
            self.assertTrue((Path(temporary) / "frames.jsonl").exists())
            self.assertTrue((Path(temporary) / "run_summary.json").exists())

    def test_runner_mismatch_and_physical_mutation_fail_closed(self) -> None:
        zeros_policy = torch.zeros(1, 570)
        zeros_critic = torch.zeros(1, 162)
        with tempfile.TemporaryDirectory() as temporary:
            plan_path, plan_sha = _write_plan(Path(temporary))
            session = audit.SameStateAuditSession(
                models=self.models,
                output_dir=Path(temporary),
                preregistration=audit.AuditPreregistration(
                    "mismatch", 11, "nominal", plan_path, plan_sha, suite_kind="causal"
                ),
                post_prior_function=self.live_post_prior,
            )
            with self.assertRaisesRegex(RuntimeError, "runner inference"):
                session.evaluate_frame(
                    step=0,
                    policy_observations=zeros_policy,
                    estimator_observations=zeros_policy,
                    critic_observations=zeros_critic,
                    true_base_linear_velocity=torch.zeros(1, 3),
                    prior_inputs=_prior_inputs(),
                    phase_signals=audit.PhaseSignals(),
                    runner_inference_function=lambda: torch.full((1, 16), 999.0),
                    physical_state_digest_function=lambda: "same",
                )

        with tempfile.TemporaryDirectory() as temporary:
            plan_path, plan_sha = _write_plan(Path(temporary))
            session = audit.SameStateAuditSession(
                models=self.models,
                output_dir=Path(temporary),
                preregistration=audit.AuditPreregistration(
                    "mutation", 11, "nominal", plan_path, plan_sha, suite_kind="causal"
                ),
                post_prior_function=self.live_post_prior,
            )
            calls = iter(("before", "after"))
            runner = lambda: self.models.forward_same_state(
                zeros_policy, zeros_policy, zeros_critic
            )["student_action"]
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                session.evaluate_frame(
                    step=0,
                    policy_observations=zeros_policy,
                    estimator_observations=zeros_policy,
                    critic_observations=zeros_critic,
                    true_base_linear_velocity=torch.zeros(1, 3),
                    prior_inputs=_prior_inputs(),
                    phase_signals=audit.PhaseSignals(),
                    runner_inference_function=runner,
                    physical_state_digest_function=lambda: next(calls),
                )

    def test_phase_machine_keeps_six_distinct_monotonic_stages(self) -> None:
        tracker = audit.MonotonicPhaseTracker()
        sequence = (
            audit.PhaseSignals(),
            audit.PhaseSignals(front_lifted=True),
            audit.PhaseSignals(front_top_supported=True),
            audit.PhaseSignals(rear_top_count=1),
            audit.PhaseSignals(rear_top_count=2),
            audit.PhaseSignals(rear_top_count=2, rear_hold_confirmed=True),
        )
        self.assertEqual([tracker.update(signals) for signals in sequence], list(audit.PHASES))

    def test_default_plan_preserves_success_and_two_failure_offsets(self) -> None:
        self.assertEqual(len(audit.DEFAULT_AUDIT_TRAJECTORIES), 3)
        self.assertEqual(audit.DEFAULT_AUDIT_TRAJECTORIES[0]["seed"], 11)
        self.assertEqual(audit.DEFAULT_AUDIT_TRAJECTORIES[1]["lateral_offset_m"], 0.12)
        self.assertEqual(audit.DEFAULT_AUDIT_TRAJECTORIES[1]["yaw_offset_deg"], 4.0)
        self.assertEqual(audit.DEFAULT_AUDIT_TRAJECTORIES[2]["seed"], 22)
        self.assertEqual(audit.DEFAULT_AUDIT_TRAJECTORIES[2]["lateral_offset_m"], -0.12)
        self.assertEqual(audit.DEFAULT_AUDIT_TRAJECTORIES[2]["yaw_offset_deg"], -4.0)


if __name__ == "__main__":
    unittest.main()
