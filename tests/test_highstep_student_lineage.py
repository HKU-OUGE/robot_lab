"""Pure-Python tests for fail-closed Student->Teacher ancestry."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"
)
SPEC = importlib.util.spec_from_file_location("robot_lab_highstep_lineage_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schedule)

STUDENT_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0"
)


class HighstepStudentLineageTest(unittest.TestCase):
    def _parent(self, root: Path, checkpoint_name: str = "teacher.pt") -> tuple[Path, Path, dict[str, str]]:
        checkpoint = root / checkpoint_name
        checkpoint.write_bytes(("weights-" + checkpoint_name).encode("utf-8"))
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        parent = root / (checkpoint_name + ".evaluation_manifest.json")
        parent.write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "evaluation_payload_schema_version": 6,
                    "evaluation_complete": True,
                    "matrix_complete": True,
                    "role": "teacher_robust",
                    "task": "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0",
                    "decision": "robust_teacher_candidate_for_student",
                    "selected_checkpoint": str(checkpoint),
                    "checkpoint_summaries": [
                        {
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_sha,
                            "valid_count": 18,
                            "pass_rate": 0.9,
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        lineage = {
            "parent_teacher_manifest_path": str(parent.resolve()),
            "parent_teacher_manifest_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
            "selected_teacher_checkpoint_path": str(checkpoint.resolve()),
            "selected_teacher_checkpoint_sha256": checkpoint_sha,
        }
        return parent, checkpoint, lineage

    def test_first_student_pins_exact_parent_and_selected_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, checkpoint, expected = self._parent(root)
            lineage = schedule.resolve_student_parent_lineage(
                task=STUDENT_TASK,
                checkpoint_path=checkpoint,
                parent_teacher_manifest_path=parent,
                source_manifest={"task": "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0"},
            )
        self.assertEqual(expected, lineage)

    def test_current_core9_robust_teacher_can_parent_robust_student(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, checkpoint, expected = self._parent(root)
            payload = json.loads(parent.read_text(encoding="utf-8"))
            payload.update(
                {
                    "workflow_id": "highstep_real_climb_core_20260712",
                    "evaluation_profile": "core9",
                }
            )
            payload["checkpoint_summaries"][0]["valid_count"] = 9
            payload["checkpoint_summaries"][0]["pass_rate"] = 2 / 3
            parent.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            expected["parent_teacher_manifest_sha256"] = hashlib.sha256(parent.read_bytes()).hexdigest()
            lineage = schedule.resolve_student_parent_lineage(
                task=STUDENT_TASK,
                checkpoint_path=checkpoint,
                parent_teacher_manifest_path=parent,
                source_manifest={"task": payload["task"]},
            )
        self.assertEqual(expected, lineage)

    def test_robust_student_rejects_standard_teacher_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, checkpoint, _ = self._parent(root)
            payload = json.loads(parent.read_text(encoding="utf-8"))
            payload.update(
                {
                    "workflow_id": "highstep_real_climb_core_20260712",
                    "evaluation_profile": "core9",
                    "role": "teacher",
                    "task": "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0",
                    "decision": "teacher_candidate_requires_real_gain_validation",
                }
            )
            payload["checkpoint_summaries"][0]["valid_count"] = 9
            payload["checkpoint_summaries"][0]["pass_rate"] = 1.0
            parent.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                schedule.ScheduleManifestInvalidError, "Robust Student requires"
            ):
                schedule.resolve_student_parent_lineage(
                    task=STUDENT_TASK,
                    checkpoint_path=checkpoint,
                    parent_teacher_manifest_path=parent,
                    source_manifest={"task": payload["task"]},
                )

    def test_wrong_teacher_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, _, _ = self._parent(root)
            wrong = root / "wrong.pt"
            wrong.write_bytes(b"wrong")
            with self.assertRaisesRegex(schedule.ScheduleContinuityError, "not the parent manifest"):
                schedule.resolve_student_parent_lineage(
                    task=STUDENT_TASK,
                    checkpoint_path=wrong,
                    parent_teacher_manifest_path=parent,
                    source_manifest=None,
                )

    def test_v15_byte_identical_teacher_mirror_is_explicitly_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, checkpoint, expected = self._parent(root)
            mirror = root / "protected_mirror" / checkpoint.name
            mirror.parent.mkdir()
            mirror.write_bytes(checkpoint.read_bytes())
            source_manifest = {
                "kind": "highstep_v15_reconstructed_train_source_schedule",
                "task": "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0",
                "reconstruction": {
                    "canonical_teacher_checkpoint": str(checkpoint.resolve()),
                    "canonical_teacher_checkpoint_sha256": expected["selected_teacher_checkpoint_sha256"],
                    "old_teacher_run_modified": False,
                },
            }
            lineage = schedule.resolve_student_parent_lineage(
                task=STUDENT_TASK,
                checkpoint_path=mirror,
                parent_teacher_manifest_path=parent,
                source_manifest=source_manifest,
            )
        self.assertEqual(expected, lineage)

    def test_unbound_teacher_mirror_is_rejected_even_when_bytes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, checkpoint, expected = self._parent(root)
            mirror = root / "unbound" / checkpoint.name
            mirror.parent.mkdir()
            mirror.write_bytes(checkpoint.read_bytes())
            with self.assertRaisesRegex(schedule.ScheduleContinuityError, "not the parent manifest"):
                schedule.resolve_student_parent_lineage(
                    task=STUDENT_TASK,
                    checkpoint_path=mirror,
                    parent_teacher_manifest_path=parent,
                    source_manifest={
                        "kind": "not_v15",
                        "task": "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0",
                        "reconstruction": {
                            "canonical_teacher_checkpoint": str(checkpoint.resolve()),
                            "canonical_teacher_checkpoint_sha256": expected["selected_teacher_checkpoint_sha256"],
                            "old_teacher_run_modified": False,
                        },
                    },
                )

    def test_student_continuation_propagates_ancestor_across_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _, _, expected = self._parent(root)
            student_checkpoint = root / "student_model_100.pt"
            student_checkpoint.write_bytes(b"student")
            lineage = schedule.resolve_student_parent_lineage(
                task=STUDENT_TASK,
                checkpoint_path=student_checkpoint,
                parent_teacher_manifest_path=None,
                source_manifest={"task": STUDENT_TASK, "student_parent_lineage": expected},
            )
        self.assertEqual(expected, lineage)

    def test_student_continuation_without_ancestor_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "student_model_100.pt"
            checkpoint.write_bytes(b"student")
            with self.assertRaisesRegex(schedule.ScheduleManifestInvalidError, "no student_parent_lineage"):
                schedule.resolve_student_parent_lineage(
                    task=STUDENT_TASK,
                    checkpoint_path=checkpoint,
                    parent_teacher_manifest_path=None,
                    source_manifest={"task": STUDENT_TASK},
                )

    def test_changed_parent_manifest_is_detected_after_ancestry_was_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, _, expected = self._parent(root)
            parent.write_text(parent.read_text(encoding="utf-8") + " ", encoding="utf-8")
            student_checkpoint = root / "student_model_100.pt"
            student_checkpoint.write_bytes(b"student")
            with self.assertRaisesRegex(schedule.ScheduleContinuityError, "lineage changed"):
                schedule.resolve_student_parent_lineage(
                    task=STUDENT_TASK,
                    checkpoint_path=student_checkpoint,
                    parent_teacher_manifest_path=None,
                    source_manifest={"task": STUDENT_TASK, "student_parent_lineage": expected},
                )

    def test_duplicate_selected_teacher_summary_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parent, checkpoint, _ = self._parent(root)
            payload = json.loads(parent.read_text(encoding="utf-8"))
            payload["checkpoint_summaries"].append(dict(payload["checkpoint_summaries"][0]))
            parent.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                schedule.ScheduleManifestInvalidError, "exactly one summary"
            ):
                schedule.resolve_student_parent_lineage(
                    task=STUDENT_TASK,
                    checkpoint_path=checkpoint,
                    parent_teacher_manifest_path=parent,
                    source_manifest=None,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
