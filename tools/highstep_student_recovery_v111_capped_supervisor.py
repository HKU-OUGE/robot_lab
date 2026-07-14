#!/usr/bin/env python3
"""One-time v1.1.1 capped continuation of the corrected Stage-B checkpoint.

This supervisor intentionally reuses the already-audited Stage-B training and
core9 machinery, but owns a separate state root and an immutable decision
table.  It can run only B300->B500 and, conditionally, B500->B1000.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
import torch.nn as nn

import highstep_student_recovery_supervisor as legacy


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORKFLOW_ID = "highstep_student_recovery_v111_capped_20260712"
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_v111_capped_20260712"
V11_AUDIT_ROOT = ROOT / "tmp/highstep_student_recovery_v11_20260712"
OLD_STATE_ROOT = ROOT / "tmp/highstep_student_recovery_20260712"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA256 = "7dbb47d76b5eafab0e425be7ac20b5a0e9487466c6512831600f22fb7509bbe5"
PREREGISTRATION = STATE_ROOT / "preregistration.json"
PREREGISTRATION_SHA256 = "1679928693851ca3da2b7b312b6d491472f9bc4f5c61fc48f4919bea5100a2b4"

SOURCE_B300 = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_22-23-52_student_recovery_B300_20260712_222346/model_299.pt"
)
SOURCE_B300_SHA256 = "5dfb5f36a06e155d97eb4da84def2d91802d6749a5944c72166246257bd147f7"
SOURCE_RUN = SOURCE_B300.parent
SOURCE_RUNTIME_STATE = SOURCE_RUN / "params/highstep_runtime_state.json"
SOURCE_BINDING = SOURCE_RUN / "params/highstep_student_recovery_binding.json"
OLD_HANDOFF = OLD_STATE_ROOT / "handoff.json"
BASELINE_EVAL = OLD_STATE_ROOT / "baseline_core9_retry2/evaluation_manifest.json"
B300_EVAL = OLD_STATE_ROOT / "evaluations/B300_20260712_224547/evaluation_manifest.json"

MECHANISM_FILES = (
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py",
    ROOT / "scripts/rsl_rl/base/algorithm_checkpoint.py",
    ROOT / "scripts/rsl_rl/base/train.py",
    ROOT / "scripts/rsl_rl/base/play.py",
    ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh",
    ROOT / "tests/test_highstep_student_recovery_stage_b.py",
    ROOT / "tests/test_vae_checkpoint_resume.py",
)

V111_CONTROL_FILES = (
    ROOT / "tools/highstep_student_recovery_supervisor.py",
    ROOT / "tests/test_highstep_student_recovery_v111_capped.py",
)

# All inherited methods resolve these names dynamically from the legacy module.
legacy.WORKFLOW_ID = WORKFLOW_ID
legacy.SPEC = SPEC
legacy.SPEC_SHA256 = SPEC_SHA256
legacy.STATE_ROOT = STATE_ROOT
legacy.BASELINE_DIR = OLD_STATE_ROOT / "baseline_core9_retry2"
legacy.CRITICAL_FILES = (
    SPEC,
    PREREGISTRATION,
    Path(__file__).resolve(),
    SOURCE_B300,
    SOURCE_RUNTIME_STATE,
    SOURCE_BINDING,
    OLD_HANDOFF,
    OLD_STATE_ROOT / "state.json",
    OLD_STATE_ROOT / "preflight_manifest.json",
    BASELINE_EVAL,
    B300_EVAL,
    *MECHANISM_FILES,
    *V111_CONTROL_FILES,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _component_state(model_state: dict, prefix: str) -> dict:
    return {key[len(prefix):]: value for key, value in model_state.items() if key.startswith(prefix)}


class CappedSupervisor(legacy.Supervisor):
    def __init__(self) -> None:
        super().__init__((OLD_STATE_ROOT / "baseline_core9_retry2").resolve())
        self.update_state(
            stage="B_v1.1.1_capped",
            status="starting",
            phase="preflight",
            checkpoint=str(SOURCE_B300),
            source_checkpoint=str(SOURCE_B300),
            effective_updates=300,
            v11_audit_root=str(V11_AUDIT_ROOT),
            resume_v11_audit_after_non_candidate=True,
        )
        self.historical: dict[str, dict] = {}

    @staticmethod
    def _actor_prefix_audit(checkpoint: Path) -> dict:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        linear_indices = (0, 2, 4, 6)
        linears: list[nn.Linear] = []
        for index in linear_indices:
            weight = state[f"actor.{index}.weight"]
            bias = state[f"actor.{index}.bias"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
            linears.append(layer)
        shared_elu = nn.ELU()
        actor = nn.Sequential(
            linears[0], shared_elu,
            linears[1], shared_elu,
            linears[2], shared_elu,
            linears[3],
        ).eval()
        torch.manual_seed(20260712)
        actor_input = torch.randn(64, linears[0].in_features)
        with torch.no_grad():
            reference = actor(actor_input)[:, 2:4]
            hidden = actor_input
            exact_modules = list(actor)
            for module in exact_modules[:-1]:
                hidden = module(hidden)
            reconstructed = nn.functional.linear(hidden, linears[-1].weight[2:4], linears[-1].bias[2:4])
            exact_error = float(torch.max(torch.abs(reference - reconstructed)).item())
            exact_equivalent = bool(
                torch.allclose(reference, reconstructed, rtol=1.0e-6, atol=1.0e-6)
            )

            deduplicated = list(actor.children())
            wrong = actor_input
            for module in deduplicated[:-1]:
                wrong = module(wrong)
            wrong_reconstructed = nn.functional.linear(
                wrong, linears[-1].weight[2:4], linears[-1].bias[2:4]
            )
            deduplicated_error = float(torch.max(torch.abs(reference - wrong_reconstructed)).item())
        if len(exact_modules) != 7 or len(deduplicated) != 5:
            raise RuntimeError(
                f"Unexpected shared-ELU module counts: positional={len(exact_modules)} children={len(deduplicated)}"
            )
        if not exact_equivalent or deduplicated_error <= 1.0e-3:
            raise RuntimeError(
                f"Actor-prefix correctness proof failed: exact={exact_error} deduplicated={deduplicated_error}"
            )
        return {
            "checkpoint": str(checkpoint),
            "positional_module_count": len(exact_modules),
            "children_module_count": len(deduplicated),
            "shared_elu_identity_count": sum(module is shared_elu for module in exact_modules),
            "list_actor_max_abs_error": exact_error,
            "list_actor_allclose_rtol_1e_6_atol_1e_6": exact_equivalent,
            "children_max_abs_error": deduplicated_error,
            "passed": True,
        }

    def _historical_summaries(self) -> dict[str, dict]:
        handoff = _json(OLD_HANDOFF)
        records = {record["phase"]: record for record in handoff.get("evaluations", [])}
        if set(records) != {"baseline", "B300"}:
            raise RuntimeError(f"Old handoff has unexpected evaluation phases: {sorted(records)}")
        baseline = records["baseline"]
        b300 = records["B300"]
        required = (
            baseline["checkpoint_sha256"] == legacy.STUDENT_SHA256
            and baseline["summary"]["valid_count"] == 9
            and baseline["summary"]["full_count"] == 4
            and baseline["summary"]["rear_hold_count"] == 4
            and baseline["summary"]["front_top_support_count"] == 7
            and b300["checkpoint_sha256"] == SOURCE_B300_SHA256
            and b300["summary"]["valid_count"] == 9
            and b300["summary"]["full_count"] == 3
            and b300["summary"]["rear_hold_count"] == 3
            and b300["summary"]["front_top_support_count"] == 7
            and b300["summary"]["first_rear_top_count"] == 3
            and b300["summary"]["no_severe_inward_count"] == 8
        )
        if not required:
            raise RuntimeError("Historical baseline/B300 evidence disagrees with v1.1.1")
        return {"baseline": baseline, "B300": b300}

    def preflight(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        if legacy.sha256(SPEC) != SPEC_SHA256:
            raise RuntimeError("v1.1.1 specification SHA256 changed")
        if legacy.sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
            raise RuntimeError("Immutable v1.1.1 preregistration SHA256 changed")
        prereg = _json(PREREGISTRATION)
        if not (
            prereg.get("workflow_id") == WORKFLOW_ID
            and prereg.get("spec_sha256") == SPEC_SHA256
            and prereg.get("source_checkpoint") == str(SOURCE_B300)
            and prereg.get("source_checkpoint_sha256") == SOURCE_B300_SHA256
            and prereg.get("source_effective_updates") == 300
            and prereg.get("b500_additional_effective_updates") == 200
            and prereg.get("b1000_additional_effective_updates") == 500
            and prereg.get("thresholds_locked_after_launch") is True
        ):
            raise RuntimeError("v1.1.1 preregistration semantic binding failed")
        if legacy.sha256(SOURCE_B300) != SOURCE_B300_SHA256:
            raise RuntimeError("Corrected B300 checkpoint SHA256 changed")
        if legacy.sha256(legacy.STUDENT) != legacy.STUDENT_SHA256 or legacy.sha256(legacy.TEACHER) != legacy.TEACHER_SHA256:
            raise RuntimeError("Canonical Student/Teacher checkpoint SHA256 changed")

        old_protected = prereg.get("old_workflow_preserved") or {}
        protected_files = {
            "handoff_sha256": OLD_HANDOFF,
            "state_sha256": OLD_STATE_ROOT / "state.json",
            "preflight_sha256": OLD_STATE_ROOT / "preflight_manifest.json",
        }
        for field, path in protected_files.items():
            expected = old_protected.get(field)
            actual = legacy.sha256(path)
            if expected != actual:
                raise RuntimeError(
                    f"Protected old-workflow artifact changed before launch: {path} expected={expected} actual={actual}"
                )

        reused = prereg.get("reused_evidence") or {}
        evidence_files = {
            "baseline": BASELINE_EVAL,
            "corrected_B300": B300_EVAL,
        }
        for label, path in evidence_files.items():
            expected = (reused.get(label) or {}).get("manifest_sha256")
            actual = legacy.sha256(path)
            if expected != actual:
                raise RuntimeError(
                    f"Reused {label} evidence changed before launch: expected={expected} actual={actual}"
                )

        old_preflight = _json(OLD_STATE_ROOT / "preflight_manifest.json")
        old_hashes = old_preflight.get("critical_file_sha256") or {}
        mechanism_hashes = {}
        for path in MECHANISM_FILES:
            current = legacy.sha256(path)
            expected = old_hashes.get(str(path))
            if current != expected:
                raise RuntimeError(f"Locked Stage-B mechanism file changed since corrected B300: {path}")
            mechanism_hashes[str(path)] = current

        payload = self.checkpoint_payload(SOURCE_B300)
        extra = ((payload.get("infos") or {}).get(legacy.ALGORITHM_STATE_KEY) or {})
        recovery = extra.get("student_recovery") or {}
        binding = recovery.get("binding_manifest") or {}
        if not (
            self.checkpoint_effective_count(SOURCE_B300) == 300
            and recovery.get("stage") == "B"
            and binding.get("initial_student_checkpoint") == str(legacy.STUDENT)
            and binding.get("initial_student_sha256") == legacy.STUDENT_SHA256
            and binding.get("teacher_checkpoint") == str(legacy.TEACHER)
            and binding.get("teacher_sha256") == legacy.TEACHER_SHA256
            and binding.get("trainable_action_rows") == [2, 3]
            and binding.get("trainable_joint_names") == ["RL_hip_joint", "RR_hip_joint"]
            and float(binding.get("actor_learning_rate")) == 1.0e-5
            and binding.get("actor_prefix_equivalence_verified") is True
            and float(binding.get("actor_prefix_equivalence_max_error")) <= 1.0e-6
            and binding.get("estimator_frozen") is True
            and binding.get("actor_body_frozen") is True
            and binding.get("box_rows_frozen") is True
            and binding.get("teacher_independent_storage") is True
        ):
            raise RuntimeError("Corrected B300 recovery binding is incomplete or changed")

        _, adam = self._adam_checkpoint_audit(SOURCE_B300)
        if adam["adam_steps"] != [1200.0, 1200.0]:
            raise RuntimeError(f"Corrected B300 Adam step mismatch: {adam['adam_steps']}")

        runtime = _json(SOURCE_RUNTIME_STATE)
        snapshots = [item for item in runtime.get("snapshots", []) if item.get("checkpoint_file") == SOURCE_B300.name]
        if not (
            len(snapshots) == 1
            and snapshots[0].get("checkpoint_sha256") == SOURCE_B300_SHA256
            and snapshots[0].get("runner_iteration") == 299
            and float(snapshots[0].get("schedule_update")) == 1201.0
        ):
            raise RuntimeError("Corrected B300 checkpoint-paired schedule snapshot is invalid")

        actor_prefix = self._actor_prefix_audit(SOURCE_B300)
        self.historical = self._historical_summaries()
        self.critical_hashes = {str(path): legacy.sha256(path) for path in legacy.CRITICAL_FILES}
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True)
        manifest = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "created_at": legacy.now_text(),
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA256,
            "preregistration": str(PREREGISTRATION),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_checkpoint": str(SOURCE_B300),
            "source_checkpoint_sha256": SOURCE_B300_SHA256,
            "source_effective_updates": 300,
            "source_adam": adam,
            "source_runtime_snapshot": snapshots[0],
            "source_binding": binding,
            "actor_prefix_audit": actor_prefix,
            "mechanism_file_sha256": mechanism_hashes,
            "critical_file_sha256": self.critical_hashes,
            "git_head": head,
            "worktree_dirty": bool(status.strip()),
            "git_status_porcelain": status.splitlines(),
            "old_state_preserved": str(OLD_STATE_ROOT),
            "v11_audit_breakpoint_preserved": str(V11_AUDIT_ROOT),
        }
        legacy.atomic_json(STATE_ROOT / "preflight_manifest.json", manifest)
        self.record_evaluation(
            "reused_baseline", 0, legacy.STUDENT,
            Path(self.historical["baseline"]["out_dir"]), self.historical["baseline"]["summary"],
        )
        self.record_evaluation(
            "reused_corrected_B300", 300, SOURCE_B300,
            Path(self.historical["B300"]["out_dir"]), self.historical["B300"]["summary"],
        )
        self.update_state(
            status="passed",
            phase="preflight",
            checkpoint=str(SOURCE_B300),
            effective_updates=300,
            preflight_manifest=str(STATE_ROOT / "preflight_manifest.json"),
            preregistration=str(PREREGISTRATION),
            preregistration_sha256=PREREGISTRATION_SHA256,
        )

    @staticmethod
    def companion_gate(summary: dict) -> tuple[bool, str]:
        checks = {
            "valid_count": summary["valid_count"] == 9,
            "front_top_support_count": summary["front_top_support_count"] >= 7,
            "first_rear_top_count": summary["first_rear_top_count"] >= 4,
            "no_severe_inward_count": summary["no_severe_inward_count"] >= 8,
        }
        failed = [name for name, passed in checks.items() if not passed]
        return (not failed, "companion gate passed" if not failed else f"companion gate failed: {failed}")

    @staticmethod
    def p_value(summary: dict) -> int:
        return min(int(summary["full_count"]), int(summary["rear_hold_count"]))

    def verify_adam_continuation(self, before: Path, after: Path, expected_delta: float) -> dict:
        _, left = self._adam_checkpoint_audit(before)
        _, right = self._adam_checkpoint_audit(after)
        deltas = [b - a for a, b in zip(left["adam_steps"], right["adam_steps"], strict=True)]
        if deltas != [expected_delta, expected_delta]:
            raise RuntimeError(f"Adam step continuation mismatch: expected {expected_delta}, got {deltas}")
        return {"before": left, "after": right, "step_deltas": deltas, "passed": True}

    def finish(
        self,
        *,
        status: str,
        reason: str,
        best_checkpoint: Path,
        best_summary: dict,
        candidate: bool = False,
        candidate_artifacts: dict | None = None,
    ) -> None:
        self.assert_code_unchanged()
        handoff = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "spec_path": str(SPEC),
            "spec_sha256": SPEC_SHA256,
            "preregistration": str(PREREGISTRATION),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "status": status,
            "stage": "B_v1.1.1_capped",
            "stop_reason": reason,
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_sha256": legacy.sha256(best_checkpoint),
            "best_core9": best_summary,
            "best_P": self.p_value(best_summary),
            "protected_v11_start": str(legacy.STUDENT),
            "protected_v11_start_sha256": legacy.STUDENT_SHA256,
            "teacher_target": str(legacy.TEACHER),
            "teacher_target_sha256": legacy.TEACHER_SHA256,
            "trainable_action_rows": [2, 3],
            "frozen_scope": "all Student tensors except RL/RR hip final output rows",
            "candidate": candidate,
            "candidate_artifacts": candidate_artifacts,
            "resume_v11_same_state_audit": not candidate,
            "v11_audit_root": str(V11_AUDIT_ROOT),
            "evaluations": self.evaluations,
            "completed_at": legacy.now_text(),
        }
        legacy.atomic_json(self.handoff_path, handoff)
        self.update_state(
            status=status,
            phase="candidate_ready" if candidate else "stopped",
            stop_reason=reason,
            checkpoint=str(best_checkpoint),
            core9=best_summary,
            P=self.p_value(best_summary),
            handoff=str(self.handoff_path),
            active_pid=None,
            user_action_required=False,
            next_action=("candidate_review" if candidate else "resume_v11_same_state_audit"),
        )

    def _best(self, results: list[tuple[Path, dict]]) -> tuple[Path, dict]:
        return max(results, key=lambda item: self.rank(item[1]))

    def run(self) -> None:
        self.preflight()
        self.require_no_gpu_job()
        self.update_state(status="running", phase="static_tests")
        test_log = STATE_ROOT / "static_tests.log"
        with test_log.open("wb") as stream:
            completed = subprocess.run(
                [
                    str(legacy.PYTHON), "-m", "pytest", "-q",
                    "tests/test_highstep_student_recovery_stage_b.py",
                    "tests/test_vae_checkpoint_resume.py",
                    "tests/test_highstep_student_recovery_v111_capped.py",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError("v1.1.1 Stage-B static tests failed")

        baseline = self.historical["baseline"]["summary"]
        b300 = self.historical["B300"]["summary"]
        results: list[tuple[Path, dict]] = [(legacy.STUDENT, baseline), (SOURCE_B300, b300)]

        self.update_state(status="running", phase="train_B500", checkpoint=str(SOURCE_B300), effective_updates=300)
        run500, checkpoint500 = self.run_train(
            label="v111_B500", checkpoint=SOURCE_B300, load_mode="full", updates=200, num_envs=4096
        )
        delta500 = self.verify_row_only_delta(SOURCE_B300, checkpoint500, 500)
        adam500 = self.verify_adam_continuation(SOURCE_B300, checkpoint500, expected_delta=800.0)
        legacy.atomic_json(STATE_ROOT / "B500_resume_audit.json", {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "run_dir": str(run500),
            "source_checkpoint": str(SOURCE_B300),
            "result_checkpoint": str(checkpoint500),
            "row_delta": delta500,
            "adam_continuation": adam500,
            "completed_at": legacy.now_text(),
        })
        eval500_dir, summary500 = self.evaluate("v111_B500", checkpoint500)
        results.append((checkpoint500, summary500))
        p500 = self.p_value(summary500)
        companion_ok, companion_reason = self.companion_gate(summary500)
        self.update_state(P=p500, companion_gate=companion_ok, companion_gate_reason=companion_reason)

        if self.behavior_final_gate(summary500):
            self.finish_or_hold_behavior_candidate(
                reason="v1.1.1 B500 reached the final 8/9 behavior gate",
                checkpoint=checkpoint500,
                summary=summary500,
                evaluation_dir=eval500_dir,
            )
            return
        if not companion_ok:
            best_checkpoint, best_summary = self._best(results)
            self.finish(
                status="stopped_by_v111_gate",
                reason=f"B500 {companion_reason}; old route capped; resume v1.1 same-state audit",
                best_checkpoint=best_checkpoint,
                best_summary=best_summary,
            )
            return
        if p500 < 4:
            best_checkpoint, best_summary = self._best(results)
            self.finish(
                status="stopped_by_v111_gate",
                reason="B500 P<4; old route permanently failed; resume v1.1 same-state audit",
                best_checkpoint=best_checkpoint,
                best_summary=best_summary,
            )
            return
        if p500 == 4:
            best_checkpoint, best_summary = self._best(results)
            self.finish(
                status="stopped_by_v111_gate",
                reason="B500 P=4 restored baseline without behavior improvement; B1000 forbidden; resume v1.1 same-state audit",
                best_checkpoint=best_checkpoint,
                best_summary=best_summary,
            )
            return

        self.update_state(status="running", phase="train_B1000", checkpoint=str(checkpoint500), effective_updates=500)
        run1000, checkpoint1000 = self.run_train(
            label="v111_B1000", checkpoint=checkpoint500, load_mode="full", updates=500, num_envs=4096
        )
        delta1000 = self.verify_row_only_delta(checkpoint500, checkpoint1000, 1000)
        adam1000 = self.verify_adam_continuation(checkpoint500, checkpoint1000, expected_delta=2000.0)
        legacy.atomic_json(STATE_ROOT / "B1000_resume_audit.json", {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "run_dir": str(run1000),
            "source_checkpoint": str(checkpoint500),
            "result_checkpoint": str(checkpoint1000),
            "row_delta": delta1000,
            "adam_continuation": adam1000,
            "completed_at": legacy.now_text(),
        })
        eval1000_dir, summary1000 = self.evaluate("v111_B1000", checkpoint1000)
        results.append((checkpoint1000, summary1000))
        p1000 = self.p_value(summary1000)
        companion1000, companion1000_reason = self.companion_gate(summary1000)
        self.update_state(P=p1000, companion_gate=companion1000, companion_gate_reason=companion1000_reason)
        if self.behavior_final_gate(summary1000):
            self.finish_or_hold_behavior_candidate(
                reason="v1.1.1 B1000 reached the final 8/9 behavior gate",
                checkpoint=checkpoint1000,
                summary=summary1000,
                evaluation_dir=eval1000_dir,
            )
            return

        best_checkpoint, best_summary = self._best(results)
        if p1000 < 6:
            reason = "B1000 P<6 minimum gate failed"
        elif p1000 < 8:
            reason = "B1000 P=6--7 improved but did not reach the final 8/9 gate"
        else:
            reason = "B1000 P>=8 but another final candidate gate failed"
        self.finish(
            status="stopped_by_v111_cap",
            reason=f"{reason}; B1000 absolute cap reached; resume v1.1 same-state audit",
            best_checkpoint=best_checkpoint,
            best_summary=best_summary,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = (STATE_ROOT / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("The v1.1.1 capped supervisor lock is already held", file=sys.stderr)
        return 2

    existing = STATE_ROOT / "state.json"
    if existing.exists():
        try:
            old = _json(existing)
        except json.JSONDecodeError:
            print("Existing v1.1.1 state is corrupt; refusing implicit retry", file=sys.stderr)
            return 3
        if old.get("workflow_id") == WORKFLOW_ID:
            print("Existing v1.1.1 workflow state found; refusing implicit retry", file=sys.stderr)
            return 3

    supervisor = CappedSupervisor()

    def handle_signal(signum, _frame):
        supervisor.update_state(status="stopping", stop_reason=f"supervisor_signal_{signum}")
        supervisor.stop_active_process()
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        supervisor.run()
        return 0
    except BaseException as error:
        supervisor.stop_active_process()
        legacy.atomic_json(
            supervisor.handoff_path,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "spec_sha256": SPEC_SHA256,
                "status": "failed_closed",
                "phase": supervisor.state.get("phase"),
                "error": f"{type(error).__name__}: {error}",
                "state": supervisor.state,
                "resume_v11_same_state_audit": True,
                "failed_at": legacy.now_text(),
            },
        )
        supervisor.update_state(
            status="failed_closed",
            stop_reason=f"{type(error).__name__}: {error}",
            handoff=str(supervisor.handoff_path),
            active_pid=None,
            user_action_required=False,
            next_action="resume_v11_same_state_audit",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
