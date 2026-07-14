"""Pure-Python tests for checkpoint-continuous high-step schedules."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"
)
SPEC = importlib.util.spec_from_file_location("robot_lab_highstep_schedule_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schedule)


class FakeEnv:
    def __init__(self, step: int = 0):
        self.common_step_counter = step

    @property
    def unwrapped(self):
        return self


class FakeWrapper:
    def __init__(self, env):
        self.env = env


class HighstepScheduleTest(unittest.TestCase):
    def test_unanchored_fallback_matches_legacy_clock(self):
        self.assertEqual(schedule.global_update(FakeEnv(48), 24), 2.0)

    def test_installed_clock_survives_local_process_origin(self):
        env = FakeEnv(12)
        wrapped = FakeWrapper(FakeWrapper(env))
        schedule.install_global_update(
            wrapped,
            600,
            source="checkpoint_iteration_fallback",
            resume_mode="preserve",
            runner_iteration=600,
        )
        self.assertEqual(schedule.global_update(wrapped, 24), 600.0)
        env.common_step_counter += 48
        self.assertEqual(schedule.global_update(wrapped, 24), 602.0)

    def test_evaluation_clock_is_frozen_at_checkpoint_update(self):
        env = FakeEnv()
        schedule.install_global_update(
            env,
            600,
            source="checkpoint_iteration_fallback",
            resume_mode="preserve",
            runner_iteration=600,
            advance_with_local_steps=False,
        )
        env.common_step_counter = 600
        self.assertEqual(schedule.global_update(env, 24), 600.0)

        state = schedule.schedule_clock_state(env, 24)
        self.assertEqual(state["global_update"], 600.0)
        self.assertFalse(state["advance_with_local_steps"])
        self.assertEqual(state["source"], "checkpoint_iteration_fallback")
        self.assertEqual(state["resume_mode"], "preserve")
        self.assertEqual(state["runner_iteration_at_anchor"], 600)

    def test_schedule_math(self):
        self.assertEqual(schedule.prior_scale(80, 80, 520), 0.0)
        self.assertAlmostEqual(schedule.prior_scale(300, 80, 520), 0.5)
        self.assertEqual(schedule.prior_scale(520, 80, 520), 1.0)
        thresholds = (650, 1300, 2200, 3200)
        levels = (1, 2, 3, 5, 8)
        self.assertEqual(schedule.stage_index(1299, thresholds), 1)
        self.assertEqual(schedule.allowed_max_level(1299, thresholds, levels), 2)
        self.assertEqual(schedule.allowed_max_level(3200, thresholds, levels), 8)

    def test_weights_only_requires_explicit_schedule_choice(self):
        with self.assertRaisesRegex(ValueError, "weights_only requires an explicit"):
            schedule.resolve_schedule_resume_mode(
                checkpoint_load_mode="weights_only",
                requested_mode="auto",
                highstep_resume_kind="refine",
            )
        self.assertEqual(
            schedule.resolve_schedule_resume_mode(
                checkpoint_load_mode="weights_only",
                requested_mode="preserve",
                highstep_resume_kind="refine",
            ),
            "preserve",
        )

    def test_full_resume_auto_distinguishes_refine_and_migration(self):
        self.assertEqual(
            schedule.resolve_schedule_resume_mode(
                checkpoint_load_mode="full", requested_mode="auto", highstep_resume_kind="refine"
            ),
            "preserve",
        )
        self.assertEqual(
            schedule.resolve_schedule_resume_mode(
                checkpoint_load_mode="full", requested_mode="auto", highstep_resume_kind="migration"
            ),
            "reset",
        )

    def test_checkpoint_iteration_mapping(self):
        self.assertEqual(schedule.checkpoint_iteration_from_mapping({"iter": 401}), 401)
        with self.assertRaises(KeyError):
            schedule.checkpoint_iteration_from_mapping({"model_state_dict": {}})

    def test_source_manifest_preserves_separate_runner_and_schedule_origins(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = run_dir / "model_600.pt"
            checkpoint.write_bytes(b"checkpoint-600")
            (params_dir / schedule.SCHEDULE_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "context": "train",
                        "runner_iteration_at_anchor": 0,
                        "schedule_update_at_anchor": 175800,
                        "schedule_definition": {},
                        "runtime_state_required_for_preserve": True,
                    }
                ),
                encoding="utf-8",
            )
            schedule.append_runtime_snapshot(
                params_dir / schedule.RUNTIME_STATE_NAME,
                {
                    "checkpoint_file": checkpoint.name,
                    "checkpoint_sha256": schedule.checkpoint_sha256(checkpoint),
                    "runner_iteration": 600,
                    "schedule_update": 176401,
                    "command_curriculum": {"enabled": True, "current_lin_vel_x": [-0.1, 0.4]},
                    "moving_best": {"observed": True, "action_score_best": 0.6, "support_score_best": 0.4},
                },
            )
            runtime_state = json.loads(
                (params_dir / schedule.RUNTIME_STATE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(runtime_state["schema_version"], 2)
            update, provenance = schedule.schedule_update_for_checkpoint(checkpoint, 600)
            self.assertEqual(update, 176401.0)
            self.assertEqual(provenance["method"], "runtime_snapshot_exact")
            self.assertTrue(provenance["runtime_snapshot_checkpoint_sha256_verified"])
            self.assertEqual(
                provenance["runtime_snapshot_checkpoint_sha256"],
                schedule.checkpoint_sha256(checkpoint),
            )
            self.assertEqual(provenance["anchor_derived_schedule_update"], 176400.0)
            self.assertEqual(
                provenance["runtime_snapshot"]["command_curriculum"]["current_lin_vel_x"],
                [-0.1, 0.4],
            )

    def test_checkpoint_iteration_fallback_requires_explicit_legacy_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "model_600.pt"
            checkpoint.touch()
            with self.assertRaises(schedule.ScheduleManifestMissingError):
                schedule.schedule_update_for_checkpoint(checkpoint, 600)
            update, provenance = schedule.schedule_update_for_checkpoint(
                checkpoint, 600, allow_legacy_checkpoint_fallback=True
            )
            self.assertEqual(update, 600.0)
            self.assertEqual(provenance["method"], "explicit_legacy_checkpoint_iteration_fallback")
            self.assertTrue(provenance["legacy_fallback_used"])

    def test_core9_standard_teacher_can_parent_student(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "model_172300.pt"
            checkpoint.write_bytes(b"core-teacher")
            digest = schedule.checkpoint_sha256(checkpoint)
            manifest = root / "teacher_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "evaluation_payload_schema_version": 6,
                        "workflow_id": "highstep_real_climb_core_20260712",
                        "evaluation_profile": "core9",
                        "evaluation_complete": True,
                        "matrix_complete": True,
                        "role": "teacher",
                        "task": "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0",
                        "decision": "teacher_candidate_for_student",
                        "selected_checkpoint": str(checkpoint),
                        "checkpoint_summaries": [
                            {
                                "checkpoint": str(checkpoint),
                                "checkpoint_sha256": digest,
                                "valid_count": 9,
                                "pass_rate": 2 / 3,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            lineage = schedule._load_parent_teacher_lineage(manifest)
            self.assertEqual(str(checkpoint.resolve()), lineage["selected_teacher_checkpoint_path"])
            self.assertEqual(digest, lineage["selected_teacher_checkpoint_sha256"])

    def test_core9_standard_teacher_below_sixty_percent_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint = root / "model_1.pt"
            checkpoint.write_bytes(b"weak-teacher")
            digest = schedule.checkpoint_sha256(checkpoint)
            manifest = root / "teacher_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "evaluation_payload_schema_version": 6,
                        "workflow_id": "highstep_real_climb_core_20260712",
                        "evaluation_profile": "core9",
                        "evaluation_complete": True,
                        "matrix_complete": True,
                        "role": "teacher",
                        "task": "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0",
                        "decision": "teacher_candidate_for_student",
                        "selected_checkpoint": str(checkpoint),
                        "checkpoint_summaries": [
                            {
                                "checkpoint": str(checkpoint),
                                "checkpoint_sha256": digest,
                                "valid_count": 9,
                                "pass_rate": 5 / 9,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(schedule.ScheduleManifestInvalidError):
                schedule._load_parent_teacher_lineage(manifest)

    def test_checkpoint_cannot_predate_manifest_runner_anchor(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = run_dir / "model_600.pt"
            checkpoint.touch()
            (params_dir / schedule.SCHEDULE_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "context": "train",
                        "runner_iteration_at_anchor": 700,
                        "schedule_update_at_anchor": 0,
                        "schedule_definition": {},
                        "runtime_state_required_for_preserve": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(schedule.ScheduleManifestInvalidError):
                schedule.schedule_update_for_checkpoint(checkpoint, 600)

    def test_schedule_definition_drift_requires_migration(self):
        source = {"schedule_definition": {"action_prior": {"start_update": 80}}}
        schedule.assert_schedule_definition_compatible(
            source, {"action_prior": {"start_update": 80}}
        )
        with self.assertRaises(schedule.ScheduleContinuityError):
            schedule.assert_schedule_definition_compatible(
                source, {"action_prior": {"start_update": 81}}
            )

    def test_all_update_based_curriculum_cadences_are_bound(self):
        action_cfg = SimpleNamespace(
            prior_start_update=80,
            prior_full_update=520,
            num_steps_per_update=24,
        )
        terrain_cfg = SimpleNamespace(
            params={
                "num_steps_per_update": 24,
                "stage_update_thresholds": (650, 1300),
                "stage_max_levels": (1, 2, 3),
                "score_warmup_updates": 450,
                "score_stage_update_thresholds": (800, 1400),
                "action_score_thresholds": (0.14, 0.28, 0.44),
                "support_score_thresholds": (0.005, 0.08, 0.24),
                "support_bottleneck_start_update": 650,
                "support_bottleneck_ramp_updates": 700,
            }
        )
        support_cfg = SimpleNamespace(
            params={
                "num_steps_per_update": 24,
                "support_bottleneck_start_update": 650,
                "support_bottleneck_ramp_updates": 700,
                "support_bottleneck_warmup_min_gate": 0.0,
            }
        )
        reward_stages = {
            "rear_support": {
                "stage_start_update": 120,
                "stage_ramp_updates": 240,
                "num_steps_per_update": 24,
            }
        }
        definition = schedule.schedule_definition_from_configs(
            action_cfg=action_cfg,
            terrain_term_cfg=terrain_cfg,
            support_term_cfg=support_cfg,
            reward_stage_definition=reward_stages,
        )
        self.assertEqual(definition["action_prior"]["num_steps_per_update"], 24)
        self.assertEqual(definition["terrain_schedule"]["num_steps_per_update"], 24)
        self.assertEqual(definition["terrain_schedule"]["score_warmup_updates"], 450)
        self.assertEqual(definition["support_bottleneck"]["num_steps_per_update"], 24)
        self.assertEqual(definition["reward_stages"], reward_stages)

        source = {"schedule_definition": definition}
        schedule.assert_schedule_definition_compatible(source, definition)
        for path in (
            ("action_prior", "num_steps_per_update"),
            ("terrain_schedule", "num_steps_per_update"),
            ("terrain_schedule", "score_warmup_updates"),
            ("terrain_schedule", "support_bottleneck_start_update"),
            ("support_bottleneck", "num_steps_per_update"),
            ("support_bottleneck", "ramp_updates"),
            ("reward_stages", "rear_support", "num_steps_per_update"),
        ):
            drifted = json.loads(json.dumps(definition))
            target = drifted
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] += 1
            with self.subTest(path=path), self.assertRaises(schedule.ScheduleContinuityError):
                schedule.assert_schedule_definition_compatible(source, drifted)

    def test_explicit_migration_reset_allows_legacy_runtime_without_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = run_dir / "model_600.pt"
            checkpoint.write_bytes(b"legacy-patched-checkpoint")
            (params_dir / schedule.SCHEDULE_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "context": "train",
                        "runner_iteration_at_anchor": 0,
                        "schedule_update_at_anchor": 0,
                        "schedule_definition": {"terrain_schedule": {"num_steps_per_update": 24}},
                        "runtime_state_required_for_preserve": True,
                    }
                ),
                encoding="utf-8",
            )
            (params_dir / schedule.RUNTIME_STATE_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "context": "train",
                        "snapshots": [
                            {
                                "checkpoint_file": checkpoint.name,
                                "runner_iteration": 600,
                                "schedule_update": 600,
                                "command_curriculum": {"enabled": False},
                                "moving_best": {"observed": False},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            update, provenance = schedule.resolve_loaded_schedule_update(
                checkpoint,
                600,
                checkpoint_load_mode="weights_only",
                schedule_resume_mode="reset",
            )
            self.assertEqual(update, 0.0)
            self.assertEqual(provenance["method"], "explicit_schedule_reset")
            with self.assertRaisesRegex(schedule.ScheduleManifestInvalidError, "does not bind"):
                schedule.resolve_loaded_schedule_update(
                    checkpoint,
                    600,
                    checkpoint_load_mode="full",
                    schedule_resume_mode="preserve",
                )

    def test_same_name_same_iteration_checkpoint_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = run_dir / "model_600.pt"
            checkpoint.write_bytes(b"original-checkpoint")
            (params_dir / schedule.SCHEDULE_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "context": "train",
                        "runner_iteration_at_anchor": 0,
                        "schedule_update_at_anchor": 0,
                        "schedule_definition": {},
                        "runtime_state_required_for_preserve": True,
                    }
                ),
                encoding="utf-8",
            )
            schedule.append_runtime_snapshot(
                params_dir / schedule.RUNTIME_STATE_NAME,
                {
                    "checkpoint_file": checkpoint.name,
                    "checkpoint_sha256": schedule.checkpoint_sha256(checkpoint),
                    "runner_iteration": 600,
                    "schedule_update": 600,
                    "command_curriculum": {"enabled": False},
                    "moving_best": {"observed": False},
                },
            )
            checkpoint.write_bytes(b"replacement-checkpoint")
            with self.assertRaisesRegex(schedule.ScheduleManifestInvalidError, "does not match"):
                schedule.schedule_update_for_checkpoint(checkpoint, 600)

    def test_new_runtime_snapshot_requires_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_path = Path(temporary_dir) / schedule.RUNTIME_STATE_NAME
            with self.assertRaisesRegex(schedule.ScheduleManifestInvalidError, "checkpoint_sha256 is required"):
                schedule.append_runtime_snapshot(
                    state_path,
                    {
                        "checkpoint_file": "model_1.pt",
                        "runner_iteration": 1,
                        "schedule_update": 1,
                        "command_curriculum": {"enabled": False},
                        "moving_best": {"observed": False},
                    },
                )

    def test_present_malformed_manifest_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            checkpoint = run_dir / "model_600.pt"
            checkpoint.touch()
            (params_dir / schedule.SCHEDULE_MANIFEST_NAME).write_text("{bad json", encoding="utf-8")
            with self.assertRaises(schedule.ScheduleManifestInvalidError):
                schedule.schedule_update_for_checkpoint(
                    checkpoint, 600, allow_legacy_checkpoint_fallback=True
                )

    def test_weights_only_preserve_never_allows_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "model_600.pt"
            checkpoint.touch()
            with self.assertRaises(schedule.ScheduleContinuityError):
                schedule.resolve_loaded_schedule_update(
                    checkpoint,
                    600,
                    checkpoint_load_mode="weights_only",
                    schedule_resume_mode="preserve",
                    allow_legacy_checkpoint_fallback=True,
                )
            with self.assertRaises(schedule.ScheduleManifestMissingError):
                schedule.resolve_loaded_schedule_update(
                    checkpoint,
                    600,
                    checkpoint_load_mode="weights_only",
                    schedule_resume_mode="preserve",
                )

    def test_explicit_reset_allows_absent_but_not_invalid_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            checkpoint = run_dir / "model_600.pt"
            run_dir.mkdir()
            checkpoint.touch()
            update, provenance = schedule.resolve_loaded_schedule_update(
                checkpoint,
                600,
                checkpoint_load_mode="weights_only",
                schedule_resume_mode="reset",
            )
            self.assertEqual(update, 0.0)
            self.assertEqual(provenance["method"], "explicit_schedule_reset")
            params_dir = run_dir / "params"
            params_dir.mkdir()
            (params_dir / schedule.SCHEDULE_MANIFEST_NAME).write_text("[]", encoding="utf-8")
            with self.assertRaises(schedule.ScheduleManifestInvalidError):
                schedule.resolve_loaded_schedule_update(
                    checkpoint,
                    600,
                    checkpoint_load_mode="weights_only",
                    schedule_resume_mode="reset",
                )

    def test_manifest_reports_actual_schedule_values(self):
        env = FakeEnv()
        schedule.install_global_update(
            env,
            1000,
            source="unit_test",
            resume_mode="preserve",
            runner_iteration=1000,
        )
        action_cfg = SimpleNamespace(
            prior_start_update=80,
            prior_full_update=520,
            num_steps_per_update=24,
        )
        terrain_cfg = SimpleNamespace(
            params={
                "stage_update_thresholds": (650, 1300, 2200, 3200),
                "stage_max_levels": (1, 2, 3, 5, 8),
                "score_stage_update_thresholds": (800, 1400, 2200),
            }
        )
        support_cfg = SimpleNamespace(
            params={
                "support_bottleneck_start_update": 650,
                "support_bottleneck_ramp_updates": 700,
                "support_bottleneck_warmup_min_gate": 0.0,
            }
        )
        manifest = schedule.build_schedule_manifest(
            context="test",
            task="Highstep",
            env=env,
            action_cfg=action_cfg,
            terrain_term_cfg=terrain_cfg,
            support_term_cfg=support_cfg,
            terrain_curriculum_enabled=True,
            support_metric_enabled=True,
            checkpoint_path=None,
            checkpoint_iteration=1000,
            checkpoint_load_mode="full",
            requested_resume_mode="preserve",
            resolved_resume_mode="preserve",
            runner_iteration_at_anchor=1000,
            schedule_update_at_anchor=1000,
        )
        self.assertEqual(manifest["action_prior"]["actual_prior_scale"], 1.0)
        self.assertEqual(manifest["terrain_schedule"]["num_steps_per_update"], 24)
        self.assertEqual(manifest["terrain_schedule"]["configured_stage_index"], 1)
        self.assertEqual(manifest["terrain_schedule"]["configured_allowed_max_level"], 2)
        self.assertEqual(manifest["terrain_schedule"]["configured_score_stage_index"], 1)
        self.assertEqual(manifest["support_bottleneck"]["actual_blend"], 0.5)
        self.assertEqual(manifest["support_bottleneck"]["num_steps_per_update"], 24)
        manifest["schedule_source"]["runtime_snapshot_checkpoint_sha256"] = "a" * 64
        manifest["schedule_source"]["runtime_snapshot_checkpoint_sha256_verified"] = True
        eval_fields = schedule.eval_schedule_fields(manifest)
        self.assertTrue(eval_fields["schedule_valid"])
        self.assertEqual(eval_fields["global_update"], 1000)
        self.assertEqual(eval_fields["action_prior_scale"], 1.0)
        self.assertEqual(eval_fields["terrain_stage_index"], 1)
        self.assertEqual(eval_fields["terrain_allowed_max_level"], 2)
        self.assertEqual(eval_fields["support_bottleneck_blend"], 0.5)
        self.assertEqual(eval_fields["runtime_snapshot_checkpoint_sha256"], "a" * 64)
        self.assertTrue(eval_fields["runtime_snapshot_checkpoint_sha256_verified"])
        missing_runtime = schedule.runtime_schedule_match(
            manifest, runtime_prior_scale=None, runtime_support_blend=None
        )
        self.assertFalse(missing_runtime["schedule_runtime_match"])
        self.assertFalse(missing_runtime["action_prior_runtime_observed"])
        self.assertFalse(missing_runtime["support_runtime_observed"])
        matching_runtime = schedule.runtime_schedule_match(
            manifest, runtime_prior_scale=1.0, runtime_support_blend=0.5
        )
        self.assertTrue(matching_runtime["schedule_runtime_match"])
        disabled_manifest = json.loads(json.dumps(manifest))
        disabled_manifest["action_prior"]["enabled"] = False
        disabled_manifest["support_bottleneck"]["metric_enabled"] = False
        disabled_runtime = schedule.runtime_schedule_match(
            disabled_manifest, runtime_prior_scale=None, runtime_support_blend=None
        )
        self.assertTrue(disabled_runtime["schedule_runtime_match"])

        with tempfile.TemporaryDirectory() as temporary_dir:
            output = schedule.write_manifest(Path(temporary_dir) / "manifest.json", manifest)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], 3)

    def test_train_and_play_manifests_pin_source_checkpoint_content(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "model_600.pt"
            checkpoint.write_bytes(b"source-checkpoint")
            original_digest = schedule.checkpoint_sha256(checkpoint)
            for context in ("train", "play"):
                manifest = schedule.build_schedule_manifest(
                    context=context,
                    task="Highstep",
                    env=FakeEnv(),
                    action_cfg=SimpleNamespace(
                        prior_start_update=80,
                        prior_full_update=520,
                        num_steps_per_update=24,
                    ),
                    terrain_term_cfg=SimpleNamespace(params={"num_steps_per_update": 24}),
                    support_term_cfg=SimpleNamespace(params={"num_steps_per_update": 24}),
                    terrain_curriculum_enabled=context == "train",
                    support_metric_enabled=True,
                    checkpoint_path=checkpoint,
                    checkpoint_iteration=600,
                    checkpoint_load_mode="full",
                    requested_resume_mode="preserve",
                    resolved_resume_mode="preserve",
                    runner_iteration_at_anchor=600,
                    schedule_update_at_anchor=600,
                )
                self.assertEqual(manifest["checkpoint_path"], str(checkpoint.resolve()))
                self.assertEqual(manifest["checkpoint_sha256"], original_digest)

            checkpoint.write_bytes(b"replaced-source-checkpoint")
            self.assertNotEqual(manifest["checkpoint_sha256"], schedule.checkpoint_sha256(checkpoint))

    def test_invalid_eval_schedule_fields_are_explicit(self):
        fields = schedule.eval_schedule_fields(None)
        self.assertFalse(fields["schedule_valid"])
        self.assertIsNone(fields["global_update"])
        self.assertIsNone(fields["runtime_snapshot_checkpoint_sha256"])
        self.assertIn("support_bottleneck_blend", fields)

    def test_command_curriculum_does_not_overwrite_restored_state_at_local_zero(self):
        curriculum_path = (
            Path(__file__).resolve().parents[1]
            / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py"
        )
        source = curriculum_path.read_text(encoding="utf-8")
        self.assertIn('if not hasattr(env, "_highstep_original_vel_x"):', source)
        self.assertNotIn(
            'if env.common_step_counter == 0 or not hasattr(env, "_highstep_original_vel_x"):',
            source,
        )


if __name__ == "__main__":
    unittest.main()
