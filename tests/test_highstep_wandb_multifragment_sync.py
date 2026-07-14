"""Regression tests for crash/reboot-split offline W&B stage runs."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from tools import highstep_wandb_stage_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def _load_exact_supervisor():
    path = ROOT / "tools/highstep_0707_exact_new_teacher_supervisor.py"
    spec = importlib.util.spec_from_file_location("highstep_0707_exact_rebinding_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fragment(root: Path, name: str, content: bytes) -> Path:
    path = root / name
    path.mkdir()
    (path / "run-test1234.wandb").write_bytes(content)
    return path.resolve()


def test_legacy_latest_only_marker_leaves_older_fragment_pending(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    wandb_root.mkdir()
    older = _fragment(wandb_root, "offline-run-20260713_223510-test1234", b"older")
    newer = _fragment(wandb_root, "offline-run-20260714_002138-test1234", b"newer")
    marker = tmp_path / "offline_training_run_synced.json"
    marker.write_text(json.dumps({"schema_version": 1, "run_id": "test1234", "offline_run": str(newer)}))

    assert gate.pending_offline_runs(wandb_root, marker, "test1234") == [older]
    assert gate.offline_sync_requires_append(marker, "test1234") is True


def test_first_fragment_does_not_require_append(tmp_path: Path) -> None:
    assert gate.offline_sync_requires_append(tmp_path / "missing.json", "test1234") is False


def test_complete_marker_covers_every_fragment_and_records_hashes(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    wandb_root.mkdir()
    older = _fragment(wandb_root, "offline-run-20260713_223510-test1234", b"older")
    newer = _fragment(wandb_root, "offline-run-20260714_002138-test1234", b"newer")
    marker = tmp_path / "offline_training_run_synced.json"
    log = tmp_path / "offline_sync.log"
    log.write_text("done\n")

    gate.record_offline_run_synced(
        marker, run_id="test1234", offline_run=older, sync_log=log,
        synced_at="one", expected_offline_runs=[older, newer],
    )
    assert json.loads(marker.read_text())["coverage_complete_for_local_fragments"] is False
    gate.record_offline_run_synced(
        marker, run_id="test1234", offline_run=newer, sync_log=log,
        synced_at="two", expected_offline_runs=[older, newer],
    )

    payload = json.loads(marker.read_text())
    assert payload["schema_version"] == 2
    assert payload["coverage_complete_for_local_fragments"] is True
    assert len(payload["offline_runs"]) == 2
    assert all(item["offline_run_wandb_files"][0]["sha256"] for item in payload["offline_runs"])
    assert gate.pending_offline_runs(wandb_root, marker, "test1234") == []


def test_wrong_run_id_marker_fails_closed(tmp_path: Path) -> None:
    wandb_root = tmp_path / "wandb"
    wandb_root.mkdir()
    _fragment(wandb_root, "offline-run-20260713_223510-test1234", b"data")
    marker = tmp_path / "offline_training_run_synced.json"
    marker.write_text(json.dumps({"schema_version": 2, "run_id": "wrong", "offline_runs": []}))

    with pytest.raises(RuntimeError, match="run_id changed"):
        gate.pending_offline_runs(wandb_root, marker, "test1234")


class _SystemFieldQueryRegressionRemote:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def scan_history(self, *, page_size: int, keys=None):
        self.calls.append({"page_size": page_size, "keys": keys})
        # This reproduces the W&B behavior that caused E500 to remain pending:
        # requesting only the system field returns no rows, while a full scan works.
        return [] if keys is not None else list(self.rows)


def test_remote_history_verification_uses_full_scan_when_system_only_query_is_empty() -> None:
    remote = _SystemFieldQueryRegressionRemote([
        {"_step": 297, "loss": 1.0},
        {"_step": 298, "loss": 0.9},
    ])

    assert gate.verified_remote_history_steps(remote, {297, 298}) == {297, 298}
    assert remote.calls == [{"page_size": 1000, "keys": None}]


def test_remote_history_verification_still_fails_closed_for_a_real_missing_step() -> None:
    remote = _SystemFieldQueryRegressionRemote([
        {"_step": 297, "loss": 1.0},
    ])

    with pytest.raises(RuntimeError, match="missing 1 steps"):
        gate.verified_remote_history_steps(remote, {297, 298})


def _rebinding_fixture(module, tmp_path: Path) -> tuple[Path, dict]:
    module.PREREG_SHA = "a" * 64
    exact = str((ROOT / "tools/highstep_0707_exact_new_teacher_supervisor.py").resolve())
    v15 = str((ROOT / "tools/highstep_student_recovery_v15_supervisor.py").resolve())
    wandb_gate = str((ROOT / "tools/highstep_wandb_stage_gate.py").resolve())
    helper = str((ROOT / "tools/highstep_core9_task_compat.py").resolve())
    prereg = {"code_sha256": {exact: "1" * 64, v15: "2" * 64}}
    payload = {
        "schema_version": 1,
        "kind": "highstep_0707_exact_wandb_multifragment_rebinding",
        "workflow_id": module.WORKFLOW_ID,
        "spec_sha256": module.SPEC_SHA,
        "preregistration_sha256": module.PREREG_SHA,
        "scope": {
            "wandb_multifragment_sync_only": True,
            **{name: False for name in (
                "spec_changed", "teacher_changed", "loss_changed", "warmup_changed",
                "freeze_scope_changed", "optimizer_changed", "learning_rate_changed",
                "schedule_changed", "behavior_gate_changed", "training_semantics_changed",
            )},
        },
        "recovery_checkpoint": {
            "path": str((module.EXPERIMENT_ROOT / "2026-07-14_00-21-31_0707_exact_E300_shutdown_resume_20260714_002124/model_297.pt").resolve()),
            "sha256": "ddb068c9be49a618fe3792578b58c8e77d11f14f75d5d0002eb1bed5c650d971",
            "effective_updates": 300,
            "interrupted_e500_checkpoint_allowed": False,
        },
        "preregistered_authority_changes": sorted((exact, v15)),
        "code_changes": [
            {"path": exact, "old_sha256": prereg["code_sha256"][exact], "new_sha256": gate.sha256_file(exact)},
            {"path": v15, "old_sha256": prereg["code_sha256"][v15], "new_sha256": gate.sha256_file(v15)},
            {"path": wandb_gate, "old_sha256": module.PREVIOUS_WANDB_GATE_SHA, "new_sha256": gate.sha256_file(wandb_gate)},
        ],
        "unchanged_authority": {"canonical_task_helper": {"path": helper, "sha256": module.CANONICAL_HELPER_SHA}},
    }
    manifest = tmp_path / "rebinding.json"
    manifest.write_text(json.dumps(payload))
    manifest.chmod(0o444)
    return manifest, prereg


def test_archived_exact_rebinding_fails_closed_after_canonical_helper_moves_on(tmp_path: Path) -> None:
    module = _load_exact_supervisor()
    manifest, prereg = _rebinding_fixture(module, tmp_path)
    with pytest.raises(RuntimeError, match="canonical helper authority changed"):
        module.validate_infrastructure_rebinding(manifest, gate.sha256_file(manifest), prereg)


def test_exact_rebinding_rejects_widened_training_scope(tmp_path: Path) -> None:
    module = _load_exact_supervisor()
    manifest, prereg = _rebinding_fixture(module, tmp_path)
    payload = json.loads(manifest.read_text())
    payload["scope"]["training_semantics_changed"] = True
    manifest.chmod(0o644)
    manifest.write_text(json.dumps(payload))
    manifest.chmod(0o444)
    with pytest.raises(RuntimeError, match="exceeds approved scope"):
        module.validate_infrastructure_rebinding(manifest, gate.sha256_file(manifest), prereg)
