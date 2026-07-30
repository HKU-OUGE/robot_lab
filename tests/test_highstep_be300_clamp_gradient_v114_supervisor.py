from __future__ import annotations

import json
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from tools import highstep_be300_clamp_gradient_v114_supervisor as supervisor


def test_mu_oob_tensorboard_tag_matches_rsl_rl_loss_namespace() -> None:
    assert supervisor.MU_OOB_LOSS_KEY == "Debug/Mu_Out_Of_Bounds_Ratio"
    assert supervisor.MU_OOB_TENSORBOARD_TAG == f"Loss/{supervisor.MU_OOB_LOSS_KEY}"


def test_tensorboard_scalar_reads_the_persisted_mu_oob_tag(tmp_path: Path) -> None:
    with SummaryWriter(log_dir=str(tmp_path)) as writer:
        writer.add_scalar(supervisor.MU_OOB_TENSORBOARD_TAG, 0.125, 99)

    assert supervisor.Supervisor.tensorboard_scalar(
        tmp_path, supervisor.MU_OOB_TENSORBOARD_TAG
    ) == pytest.approx(0.125)
    with pytest.raises(RuntimeError, match="is missing"):
        supervisor.Supervisor.tensorboard_scalar(tmp_path, supervisor.MU_OOB_LOSS_KEY)


def test_mechanism_check_requests_the_persisted_mu_oob_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model_99.pt"
    checkpoint.write_bytes(b"checkpoint")
    instance = object.__new__(supervisor.Supervisor)
    instance.checkpoint_scope = lambda _checkpoint, _stage: {
        "all_checkpoint_tensors_finite": True,
        "student_actor_latent_clamp_backward": "straight_through",
    }
    requested: list[str] = []

    def read_scalar(_run_dir: Path, tag: str) -> float:
        requested.append(tag)
        return 0.125

    monkeypatch.setattr(instance, "tensorboard_scalar", read_scalar)
    result = instance.mechanism_check(100, tmp_path, checkpoint, None)

    assert requested == [supervisor.MU_OOB_TENSORBOARD_TAG]
    assert result["mu_oob_fraction"] == pytest.approx(0.125)
    assert result["passed"] is True


def test_infrastructure_repair_is_hash_bound_and_changes_only_the_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "spec.md"
    prereg = tmp_path / "preregistration.json"
    repaired_supervisor = tmp_path / "supervisor.py"
    dashboard = tmp_path / "dashboard.json"
    manifest = tmp_path / "repair.json"
    artifacts = []
    for label, name in (
        ("E100 training result", "training_result.json"),
        ("E100 checkpoint", "model_99.pt"),
        ("E100 TensorBoard event", "events.out.tfevents.test"),
        ("supervisor regression test", "test_supervisor.py"),
    ):
        path = tmp_path / name
        path.write_text(label, encoding="utf-8")
        artifacts.append(
            {"label": label, "path": str(path), "sha256": supervisor.sha256_file(path)}
        )
    spec.write_text("spec", encoding="utf-8")
    prereg.write_text("{}", encoding="utf-8")
    repaired_supervisor.write_text("repaired", encoding="utf-8")
    original_sha = "1" * 64
    prereg_sha = supervisor.sha256_file(prereg)
    payload = {
        "kind": "highstep_v114_tensorboard_scalar_namespace_repair_rebinding2",
        "workflow_id": supervisor.WORKFLOW_ID,
        "authority_version": "v1.14",
        "spec_path": str(spec),
        "spec_sha256": supervisor.SPEC_SHA,
        "preregistration_path": str(prereg),
        "preregistration_sha256": prereg_sha,
        "fault": {
            "producer_loss_key": supervisor.MU_OOB_LOSS_KEY,
            "persisted_tensorboard_tag": supervisor.MU_OOB_TENSORBOARD_TAG,
            "observed_scalar_count": 100,
            "observed_last_step": 99,
        },
        "preservation": {
            "effective_updates": 100,
            "training_semantics_changed": False,
            "optimizer_reset": False,
            "schedule_reset": False,
            "checkpoint_rewritten": False,
            "tensorboard_event_rewritten": False,
        },
        "code_rebinding": {
            "changed_files": [str(repaired_supervisor)],
            "original_supervisor_sha256": original_sha,
            "repaired_supervisor_sha256": supervisor.sha256_file(repaired_supervisor),
        },
        "artifact_bindings": artifacts,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o444)
    manifest_sha = supervisor.sha256_file(manifest)
    dashboard.write_text(
        json.dumps(
            {
                "infrastructure_repair_manifest": str(manifest),
                "infrastructure_repair_manifest_sha256": manifest_sha,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "SPEC", spec)
    monkeypatch.setattr(supervisor, "PREREG", prereg)
    monkeypatch.setattr(supervisor, "SUPERVISOR_PATH", repaired_supervisor)
    monkeypatch.setattr(supervisor, "INFRA_REPAIR", manifest)
    monkeypatch.setattr(supervisor, "DASHBOARD", dashboard)
    instance = object.__new__(supervisor.Supervisor)
    instance.prereg_sha = prereg_sha
    instance.infrastructure_repair_sha = manifest_sha
    instance.critical_hashes = {str(repaired_supervisor): original_sha}
    instance.runtime_code_overrides = {}

    instance._validate_infrastructure_repair()

    assert instance.runtime_code_overrides == {
        str(repaired_supervisor): supervisor.sha256_file(repaired_supervisor)
    }


def test_service_prevents_restart_for_fail_closed_exit_codes() -> None:
    unit = (
        supervisor.ROOT / "scripts/systemd/highstep-be300-clamp-gradient-v114.service"
    ).read_text(encoding="utf-8")
    assert "RestartPreventExitStatus=3 4" in unit
