from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_live_child_stdout_supplies_missing_train_log(monkeypatch, tmp_path):
    module = load("dashboard_state_live_fd_test", ROOT / "tmp/highstep_dashboard_state.py")
    workflow = "highstep_test_live_training"
    state_path = tmp_path / "tmp" / workflow / "state.json"
    state_path.parent.mkdir(parents=True)
    train_log = state_path.parent / "logs" / "training.log"
    train_log.parent.mkdir()
    train_log.write_text("Learning iteration 101/4000\n", encoding="utf-8")

    state_path.write_text(
        json.dumps(
            {
                "workflow_id": workflow,
                "status": "running",
                "phase": "formal_training",
                "active_pid": 4242,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / module.ACTIVE_WORKFLOW_DECLARATION).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow_id": workflow,
                "state_path": str(state_path),
            }
        ),
        encoding="utf-8",
    )

    proc_root = tmp_path / "proc"
    fd_root = proc_root / "4242" / "fd"
    fd_root.mkdir(parents=True)
    (fd_root / "1").symlink_to(train_log)
    monkeypatch.setattr(module, "PROC_ROOT", proc_root)
    monkeypatch.setattr(module, "pid_alive", lambda value: int(value) == 4242)

    _, selected = module.select_workflow_state(tmp_path)

    assert selected["train_log"] == str(train_log.resolve())
    assert selected["_dashboard_train_log_source"] == "live_process_fd"
    assert module.field_value("train-log", tmp_path) == str(train_log.resolve())
