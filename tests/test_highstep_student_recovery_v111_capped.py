"""CPU-only contracts for the user-approved v1.1.1 capped Stage-B closeout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path("/home/lxq/Softwares/robot_lab")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import highstep_student_recovery_v111_capped_supervisor as target


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HighstepStudentRecoveryV111CappedTests(unittest.TestCase):
    def test_authority_and_preregistration_hashes_are_exact(self) -> None:
        # The capped route is terminal; advancing the live formal spec to v1.2
        # must make its old authority non-runnable without changing its own
        # immutable preregistration/evidence.
        self.assertNotEqual(target.SPEC_SHA256, sha256(target.SPEC))
        self.assertEqual(target.PREREGISTRATION_SHA256, sha256(target.PREREGISTRATION))
        handoff = json.loads((target.STATE_ROOT / "handoff.json").read_text())
        self.assertEqual("stopped_by_v111_cap", handoff["status"])

    def test_preregistered_budget_and_decision_table_are_capped(self) -> None:
        prereg = json.loads(target.PREREGISTRATION.read_text(encoding="utf-8"))
        self.assertEqual(300, prereg["source_effective_updates"])
        self.assertEqual(200, prereg["b500_additional_effective_updates"])
        self.assertEqual(500, prereg["b1000_additional_effective_updates"])
        self.assertEqual(1000, prereg["absolute_old_route_update_cap"])
        self.assertTrue(prereg["thresholds_locked_after_launch"])
        self.assertFalse(prereg["post_result_threshold_changes_allowed"])
        self.assertEqual(
            ["P < 4", "P == 4", "P >= 5 and all_companion_gates_pass", "any_companion_gate_fails"],
            [item["condition"] for item in prereg["b500_decision_table"]],
        )

    def test_companion_gate_and_p_are_literal_spec_rules(self) -> None:
        passing = {
            "valid_count": 9,
            "front_top_support_count": 7,
            "first_rear_top_count": 4,
            "no_severe_inward_count": 8,
            "full_count": 5,
            "rear_hold_count": 6,
        }
        self.assertEqual(5, target.CappedSupervisor.p_value(passing))
        self.assertTrue(target.CappedSupervisor.companion_gate(passing)[0])
        for key in ("valid_count", "front_top_support_count", "first_rear_top_count", "no_severe_inward_count"):
            failing = dict(passing)
            failing[key] -= 1
            self.assertFalse(target.CappedSupervisor.companion_gate(failing)[0], key)

    def test_completed_route_preserves_the_preregistered_mechanism_snapshot(self) -> None:
        prereg = json.loads(target.PREREGISTRATION.read_text(encoding="utf-8"))
        preflight = json.loads(
            (target.STATE_ROOT / "preflight_manifest.json").read_text(encoding="utf-8")
        )
        historical = preflight["mechanism_file_sha256"]
        for raw_path, expected in prereg["locked_mechanism_file_sha256"].items():
            self.assertEqual(expected, historical.get(raw_path), raw_path)

        # The one-time Stage-B route is closed.  Later, separately preregistered
        # R2 code is allowed to change live source files, but it must never
        # rewrite the immutable bytes that prove which mechanism produced the
        # preserved B500/B1000 result.
        handoff = json.loads((target.STATE_ROOT / "handoff.json").read_text(encoding="utf-8"))
        self.assertEqual("stopped_by_v111_cap", handoff["status"])
        self.assertTrue(handoff["resume_v11_same_state_audit"])

    def test_old_workflow_and_reused_evidence_are_still_the_preregistered_bytes(self) -> None:
        prereg = json.loads(target.PREREGISTRATION.read_text(encoding="utf-8"))
        old = prereg["old_workflow_preserved"]
        self.assertEqual(old["handoff_sha256"], sha256(target.OLD_HANDOFF))
        self.assertEqual(old["state_sha256"], sha256(target.OLD_STATE_ROOT / "state.json"))
        self.assertEqual(old["preflight_sha256"], sha256(target.OLD_STATE_ROOT / "preflight_manifest.json"))
        reused = prereg["reused_evidence"]
        self.assertEqual(reused["baseline"]["manifest_sha256"], sha256(target.BASELINE_EVAL))
        self.assertEqual(reused["corrected_B300"]["manifest_sha256"], sha256(target.B300_EVAL))

    def test_corrected_checkpoint_keeps_positional_shared_elu(self) -> None:
        audit = target.CappedSupervisor._actor_prefix_audit(target.SOURCE_B300)
        self.assertTrue(audit["passed"])
        self.assertEqual(7, audit["positional_module_count"])
        self.assertEqual(5, audit["children_module_count"])
        self.assertEqual(3, audit["shared_elu_identity_count"])
        self.assertTrue(audit["list_actor_allclose_rtol_1e_6_atol_1e_6"])
        self.assertLessEqual(audit["list_actor_max_abs_error"], 3.0e-6)
        self.assertGreater(audit["children_max_abs_error"], 1.0e-3)


if __name__ == "__main__":
    unittest.main()
