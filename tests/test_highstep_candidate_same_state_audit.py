"""CPU-only tests for the read-only R2 candidate same-state adapter."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "scripts/rsl_rl/base/highstep_candidate_same_state_audit_play.py"
SPEC = importlib.util.spec_from_file_location("highstep_candidate_same_state_test_target", ADAPTER_PATH)
assert SPEC is not None and SPEC.loader is not None
candidate_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate_audit
SPEC.loader.exec_module(candidate_audit)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_model_state() -> dict[str, torch.Tensor]:
    return {
        "actor.0.weight": torch.zeros(3, 3),
        "actor.0.bias": torch.zeros(3),
        "actor.6.weight": torch.zeros(16, 3),
        "actor.6.bias": torch.zeros(16),
        "estimator.encoder.0.weight": torch.zeros(3, 3),
        "estimator.encoder.0.bias": torch.zeros(3),
        "estimator.encoder.2.weight": torch.zeros(3, 3),
        "estimator.encoder.2.bias": torch.zeros(3),
        "estimator.fc_mu.weight": torch.zeros(2, 3),
        "estimator.fc_mu.bias": torch.zeros(2),
        "estimator.fc_logvar.weight": torch.zeros(2, 3),
        "estimator.fc_logvar.bias": torch.zeros(2),
        "priv_encoder.net.0.weight": torch.zeros(2, 2),
        "priv_encoder.net.0.bias": torch.zeros(2),
        "std": torch.ones(16),
    }


def _valid_candidate_checkpoint(state: dict[str, torch.Tensor], count: int = 100) -> dict:
    binding = {
        "stage": "R2",
        "preregistration_path": os.path.realpath(candidate_audit.R2_PREREGISTRATION_PATH),
        "preregistration_sha256": candidate_audit.R2_PREREGISTRATION_SHA256,
        "protected_student_root": os.path.realpath(candidate_audit.audit.ROOT_STUDENT_CHECKPOINT),
        "protected_student_root_sha256": candidate_audit.audit.ROOT_STUDENT_SHA256,
        "initial_student_checkpoint": os.path.realpath(candidate_audit.audit.CURRENT_STUDENT_CHECKPOINT),
        "initial_student_sha256": candidate_audit.audit.CURRENT_STUDENT_SHA256,
        "teacher_checkpoint": os.path.realpath(candidate_audit.audit.TEACHER_CHECKPOINT),
        "teacher_sha256": candidate_audit.audit.TEACHER_SHA256,
        "teacher_env_yaml": os.path.realpath(candidate_audit.audit.TEACHER_ENV_CONFIG),
        "teacher_env_yaml_sha256": candidate_audit.audit.TEACHER_ENV_CONFIG_SHA256,
        "effective_update_count": count,
        "optimizer_parameter_names": list(candidate_audit.R2_OPTIMIZER_PARAMETER_NAMES),
        "optimizer_parameter_tensor_count": 8,
        "trainable_live_tensors": list(candidate_audit.R2_OPTIMIZER_PARAMETER_NAMES[:6]),
        "materialized_actor_rows": [12, 13, 14, 15],
        "checkpoint_load_mode": "full",
        "schedule_resume_mode": "preserve",
        "runtime_contract": {"teacher_context_shape": [3]},
    }
    return {
        "model_state_dict": state,
        "iter": count,
        "infos": {
            "robot_lab_algorithm_checkpoint_state": {
                "schema_version": 1,
                "algorithm_class": "VAEPPO",
                "distill_stage": 2,
                "student_distill_update_count": count,
                "vae_optimizer_state_dict": {},
                "student_recovery": {
                    "schema_version": 1,
                    "stage": "R2",
                    "effective_update_count": count,
                    "preregistration_sha256": candidate_audit.R2_PREREGISTRATION_SHA256,
                    "optimizer_parameter_names": list(candidate_audit.R2_OPTIMIZER_PARAMETER_NAMES),
                    "box_weight": state["actor.6.weight"][12:16].clone(),
                    "box_bias": state["actor.6.bias"][12:16].clone(),
                    "binding_manifest": binding,
                },
            }
        },
    }


class HighstepCandidateSameStateAuditTests(unittest.TestCase):
    def test_frozen_base_runtime_and_audit_sources_are_unchanged(self) -> None:
        self.assertEqual(_sha256(candidate_audit.BASE_RUNTIME_PATH), candidate_audit.BASE_RUNTIME_SHA256)
        self.assertEqual(_sha256(candidate_audit.AUDIT_TOOL_PATH), candidate_audit.AUDIT_TOOL_SHA256)
        self.assertEqual(
            _sha256(candidate_audit.R2_PREREGISTRATION_PATH),
            candidate_audit.R2_PREREGISTRATION_SHA256,
        )
        candidate_audit._assert_frozen_sources()

    def test_environment_binding_rejects_wrong_sha_before_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_100.pt"
            checkpoint.write_bytes(b"candidate")
            wrong_sha = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                candidate_audit.resolve_candidate_binding(
                    {
                        candidate_audit.CANDIDATE_CHECKPOINT_ENV: str(checkpoint.resolve()),
                        candidate_audit.CANDIDATE_SHA256_ENV: wrong_sha,
                    }
                )

    def test_only_preregistered_tensor_deltas_are_accepted(self) -> None:
        anchor = _base_model_state()
        candidate = {key: value.clone() for key, value in anchor.items()}
        candidate["estimator.encoder.0.weight"][0, 0] = 0.25
        candidate["estimator.fc_mu.bias"][0] = -0.10
        candidate["actor.6.weight"][12, 0] = 0.05
        candidate["actor.6.bias"][15] = -0.02

        manifest = candidate_audit.validate_candidate_tensor_deltas(candidate, anchor)

        self.assertEqual(
            manifest["changed_full_model_tensors"],
            ["estimator.encoder.0.weight", "estimator.fc_mu.bias"],
        )
        self.assertEqual(
            manifest["changed_actor_box_row_tensors"],
            ["actor.6.bias", "actor.6.weight"],
        )
        self.assertTrue(manifest["all_other_tensors_byte_equal_to_b500"])

    def test_illegal_tensor_or_non_box_actor_row_delta_fails_closed(self) -> None:
        anchor = _base_model_state()

        illegal_tensor = {key: value.clone() for key, value in anchor.items()}
        illegal_tensor["estimator.fc_logvar.bias"][0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "unauthorized tensor delta"):
            candidate_audit.validate_candidate_tensor_deltas(illegal_tensor, anchor)

        illegal_row = {key: value.clone() for key, value in anchor.items()}
        illegal_row["actor.6.weight"][11, 0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "unauthorized non-box actor-row"):
            candidate_audit.validate_candidate_tensor_deltas(illegal_row, anchor)

    def test_r2_extra_stage_prereg_and_all_anchors_are_bound(self) -> None:
        state = _base_model_state()
        checkpoint = _valid_candidate_checkpoint(state, count=300)

        manifest = candidate_audit.validate_candidate_checkpoint_extra(checkpoint, state)

        self.assertEqual(manifest["stage"], "R2")
        self.assertEqual(manifest["effective_update_count"], 300)
        self.assertTrue(manifest["protected_root_bound"])
        self.assertTrue(manifest["b500_anchor_bound"])
        self.assertTrue(manifest["teacher_bound"])

        wrong_stage = copy.deepcopy(checkpoint)
        wrong_stage["infos"]["robot_lab_algorithm_checkpoint_state"]["student_recovery"][
            "stage"
        ] = "B"
        with self.assertRaisesRegex(RuntimeError, "stage is not R2"):
            candidate_audit.validate_candidate_checkpoint_extra(wrong_stage, state)

        wrong_prereg = copy.deepcopy(checkpoint)
        wrong_prereg["infos"]["robot_lab_algorithm_checkpoint_state"]["student_recovery"][
            "preregistration_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "preregistration SHA256 mismatch"):
            candidate_audit.validate_candidate_checkpoint_extra(wrong_prereg, state)

    def test_candidate_plan_binds_identity_without_b500_outcome_expectations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_300.pt"
            checkpoint.write_bytes(b"candidate-plan")
            binding = candidate_audit.CandidateBinding(checkpoint.resolve(), _sha256(checkpoint))

            plan = candidate_audit.candidate_suite_plan_payload(binding)

            self.assertEqual(plan["candidate_checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(plan["candidate_sha256"], binding.sha256)
            self.assertEqual(plan["candidate_required_stage"], "R2")
            self.assertEqual(
                plan["runtime_contract"], candidate_audit.audit.base_suite_plan_payload()["runtime_contract"]
            )
            self.assertEqual(
                [trajectory["run_id"] for trajectory in plan["trajectories"]],
                ["seed11_nominal", "seed11_left_offset", "seed22_right_offset"],
            )
            for trajectory in plan["trajectories"]:
                self.assertNotIn("expected_full_climb_success", trajectory)
                self.assertNotIn("expected_rear_hold_success", trajectory)
                self.assertNotIn("required_phases", trajectory)
                self.assertNotIn("forbidden_phases", trajectory)
                self.assertEqual(trajectory["behavior_outcome"], "observed_not_preregistered")
                self.assertFalse(trajectory["termination_allowed"])

            plan_path, plan_sha = candidate_audit.ensure_candidate_suite_plan(
                root / "evidence", binding, expected_payload=plan
            )
            self.assertEqual(_sha256(plan_path), plan_sha)
            self.assertFalse(
                bool(plan_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            )
            repeated_path, repeated_sha = candidate_audit.ensure_candidate_suite_plan(
                root / "evidence", binding, expected_payload=plan
            )
            self.assertEqual((repeated_path, repeated_sha), (plan_path, plan_sha))

            runtime_binding = candidate_audit.candidate_runtime_binding_payload(
                binding,
                suite_plan_path=plan_path,
                suite_plan_sha256=plan_sha,
                runtime_code_hashes={"adapter": "a" * 64},
                play_contract_nodes=("reset", "tracker"),
                play_sha256="b" * 64,
            )
            self.assertEqual(runtime_binding["candidate_sha256"], binding.sha256)
            self.assertEqual(runtime_binding["suite_plan_sha256"], plan_sha)
            self.assertEqual(
                runtime_binding["r2_preregistration_sha256"],
                candidate_audit.R2_PREREGISTRATION_SHA256,
            )

    def test_completed_trace_requires_600_unterminated_finite_parity_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidate.pt"
            checkpoint.write_bytes(b"candidate")
            binding = candidate_audit.CandidateBinding(checkpoint.resolve(), _sha256(checkpoint))
            trajectory = type("Trajectory", (), {"run_id": "seed11_nominal"})()
            output = root / trajectory.run_id
            output.mkdir()
            frames = output / "frames.jsonl"
            record = {
                "runner_manual_action_allclose": True,
                "physical_state_unchanged_by_forward": True,
                "observation_and_prior_inputs_unchanged_by_forward": True,
                "action_decomposition_residual_max_abs": 0.0,
                "causal_replacement_applied": False,
            }
            frames.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for _ in range(600)),
                encoding="utf-8",
            )
            summary = {
                "frame_count": 600,
                "runner_manual_action_allclose_all_frames": True,
                "all_forward_state_digests_unchanged": True,
                "all_observation_and_prior_inputs_unchanged": True,
                "action_decomposition_residual_global_max_abs": 0.0,
                "frames_path": str(frames),
                "checkpoint_binding": {
                    "student_checkpoint_sha256": binding.sha256,
                    "candidate_r2_extra_binding": {"stage": "R2"},
                },
                "outcome": {
                    "loop_steps": 600,
                    "requested_play_max_steps": 600,
                    "terminated": False,
                    "full_climb_success": True,
                    "rear_on_platform_hold_success": True,
                },
            }
            summary_path = output / "run_summary.json"
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

            result = candidate_audit.validate_completed_candidate_run(root, trajectory, binding)
            self.assertEqual(result["frame_count"], 600)
            self.assertTrue(result["finite_parity_digest_decomposition_verified"])

            summary["outcome"]["terminated"] = True
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "terminated=True"):
                candidate_audit.validate_completed_candidate_run(root, trajectory, binding)

    def test_adapter_is_thin_and_does_not_reimplement_environment_loop(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("def _configure_fixed_environment", source)
        self.assertNotIn("def _run(args", source)
        self.assertNotIn("env.step(", source)
        self.assertIn("base_runtime._run(args, trajectory, simulation_app)", source)
        self.assertIn("audit.BoundAuditModels = CandidateBoundAuditModels", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
