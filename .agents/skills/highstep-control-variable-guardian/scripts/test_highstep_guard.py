#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("highstep_guard.py")
SPEC = importlib.util.spec_from_file_location("highstep_guard_under_test", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(guard)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        state_root = self.root / "tmp/current"
        state_root.mkdir(parents=True)
        spec = self.root / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
        spec.parent.mkdir(parents=True)
        spec.write_text("# v-test\n", encoding="utf-8")
        prereg = state_root / "preregistration.json"
        prereg.write_text(
            json.dumps(
                {
                    "workflow_id": "wf",
                    "authority": {"version": "v-test", "spec_path": str(spec), "spec_sha256": sha(spec)},
                }
            ),
            encoding="utf-8",
        )
        state = state_root / "state.json"
        state.write_text(
            json.dumps(
                {
                    "workflow_id": "wf",
                    "phase": "paused_safe_boundary",
                    "status": "paused",
                    "spec_sha256": sha(spec),
                    "preregistration_sha256": sha(prereg),
                    "active_pid": None,
                    "supervisor_pid": None,
                }
            ),
            encoding="utf-8",
        )
        declaration = self.root / "tmp/highstep_dashboard_active_workflow.json"
        declaration.write_text(
            json.dumps(
                {
                    "workflow_id": "wf",
                    "authority_version": "v-test",
                    "state_path": str(state),
                    "spec_sha256": sha(spec),
                    "preregistration_path": str(prereg),
                    "preregistration_sha256": sha(prereg),
                }
            ),
            encoding="utf-8",
        )
        self.spec = spec
        self.prereg = prereg

    def tearDown(self) -> None:
        self.temp.cleanup()

    def payload(self, command: str, tool_name: str = "Bash") -> dict:
        return {"hook_event_name": "PreToolUse", "tool_name": tool_name, "tool_input": {"command": command}}

    def test_authority_audit_passes(self) -> None:
        self.assertTrue(guard.audit(self.root)["ok"])

    def test_authority_audit_fails_closed_on_spec_drift(self) -> None:
        self.spec.write_text("changed\n", encoding="utf-8")
        self.assertFalse(guard.audit(self.root)["ok"])

    def test_read_only_spec_inspection_is_allowed(self) -> None:
        decision = guard.pre_tool_decision(self.payload(f"rg warmup {self.spec}"), self.root)
        self.assertIsNone(decision)

    def test_superseded_service_start_is_denied(self) -> None:
        decision = guard.pre_tool_decision(
            self.payload("systemctl --user start highstep-student-recovery-v15.service"), self.root
        )
        self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_archived_zero_scale_training_is_denied(self) -> None:
        decision = guard.pre_tool_decision(
            self.payload("python train.py --checkpoint logs/highstep_0707_exact_new_teacher_Student/model_894.pt"),
            self.root,
        )
        self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")

    def test_semantic_edit_without_authority_is_denied(self) -> None:
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "apply_patch",
            "tool_input": {
                "patch": "*** Update File: source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py\n+student_actor_warmup_updates"
            },
        }
        decision = guard.pre_tool_decision(payload, self.root)
        self.assertEqual(decision["hookSpecificOutput"]["permissionDecision"], "deny")


if __name__ == "__main__":
    unittest.main()
