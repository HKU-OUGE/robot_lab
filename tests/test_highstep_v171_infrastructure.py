"""Infrastructure-only v1.7.1 authority and delivery checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_classification_is_specific_and_fail_closed():
    source = (ROOT / "tools/highstep_historical_0707_exact_supervisor.py").read_text()
    assert '"wandb_infrastructure_retryable"' in source
    assert '"evaluation_infrastructure_retryable"' in source
    assert '"video_infrastructure_retryable"' in source
    assert '"training_mechanism_error"' in source
    assert '"external_fault_requires_user"' in source


def test_video_visibility_qc_accepts_only_visible_side_top_capture(monkeypatch, tmp_path):
    qc = load("highstep_v171_video_qc", ROOT / "tools/highstep_video_visibility_qc.py")
    video = tmp_path / "visible.mp4"
    video.write_bytes(b"video")
    log = tmp_path / "visible.log"
    log.write_text("[HIGHSTEP_CAMERA] mode=side_top env=0\n")
    payload = {"streams": [{"width": 1280, "height": 720, "nb_frames": "600"}],
               "format": {"duration": "12.0"}}
    monkeypatch.setattr(qc.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps(payload)
    ))
    assert qc.video_delivery_qc(video, log)["passed"] is True
    log.write_text("camera missing\n")
    with pytest.raises(RuntimeError, match="video delivery QC failed"):
        qc.video_delivery_qc(video, log)


def test_completed_e100_is_adopted_not_retrained():
    source = (ROOT / "tools/highstep_historical_0707_exact_supervisor.py").read_text()
    assert "source_v17_training_was_not_restarted" in source
    assert "completed E100 checkpoint SHA mismatch" in source
    assert "return E100_CHECKPOINT.parent, E100_CHECKPOINT, wandb_manifest" in source
    assert "def wait(self, process:" in source
    assert "child_parent_pid" in source
