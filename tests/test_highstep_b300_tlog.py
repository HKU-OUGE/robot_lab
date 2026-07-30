from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_b300_tlog_uses_current_contract_and_loss_names(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "tmp"))
    module = load("b300_hybrid_tlog_test", ROOT / "tmp/highstep_train_dashboard.py")
    log_path = tmp_path / "train.log"
    log_path.write_text(
        " Learning iteration 215/299 \n"
        "Mean Loss/VAE_Vel_MSE loss: 0.0008\n"
        "Mean Loss/Distill_Latent_MSE loss: 0.0137\n"
        "Mean Loss/VAE_Recon_MSE loss: 0.1512\n"
        "Mean Loss/Teacher_Action_MSE loss: 0.0483\n"
        "Mean Loss/Prior_Box_Loss loss: 0.0086\n"
        "Mean Loss/B300_Hybrid_Nonbox_PrePrior_MSE loss: 0.0011\n"
        "Mean Loss/B300_Hybrid_Box_PostPrior_Policy_MSE loss: 0.0205\n"
        "Mean Debug/B300_Estimator_Grad_Norm loss: 9.0882\n"
        "Mean Debug/B300_Box_Rows_Grad_Norm loss: 188.5031\n"
        "Mean Debug/B300_Frozen_Nonbox_Grad_Norm loss: 0.0000\n"
        "Mean Debug/B300_Box_Adapt_From_Update_Zero loss: 1.0000\n"
        "Mean Debug/B300_Mixed_Target_Policy_Units loss: 1.0000\n"
        "Mean Debug/B300_Canonical_Unique_Episodes loss: 1.0000\n"
        "Mean Debug/B300_Canonical_Samples loss: 138.0000\n"
        "Mean Debug/Mu_Out_Of_Bounds_Ratio loss: 0.1460\n"
        "Mean Debug/Student_Distill_Update_Count loss: 217.0000\n"
        "Time elapsed: 00:10:00\n"
        "ETA: 00:05:00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "select_workflow_state", lambda: (None, {
        "workflow_id": "highstep_b300_canonical_hybrid_prior_latent_20260718",
        "status": "running",
        "phase": "training_E300",
        "train_pid": 123,
        "train_log": str(log_path),
        "run_dir": str(tmp_path),
        "role": "student",
        "_dashboard_staged_workflow": True,
    }))
    monkeypatch.setattr(module, "pid_alive", lambda _value: True)
    monkeypatch.setattr(
        module.shutil,
        "get_terminal_size",
        lambda _fallback: module.os.terminal_size((120, 40)),
    )

    rendered = module.render(False)

    assert "B300 hybrid distill / freeze contract" in rendered
    assert "Mixed target" in rendered
    assert "Box adapt U0" in rendered
    assert "Frozen nonbox" in rendered
    assert "B300 hybrid losses" in rendered
    assert "Nonbox pre MSE" in rendered
    assert "Box post MSE" in rendered
    assert "Mu out-of-range" in rendered
    assert "Warmup target" not in rendered
    assert "Highstep mode" not in rendered
    assert "Post-prior MSE" not in rendered
    assert "Phase MSE" not in rendered
    assert "Rear-box MSE" not in rendered


def test_b300_tlog_compact_header_does_not_claim_missing_warmup(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "tmp"))
    module = load("b300_hybrid_tlog_compact_test", ROOT / "tmp/highstep_train_dashboard.py")
    log_path = tmp_path / "train.log"
    log_path.write_text(
        " Learning iteration 100/299 \n"
        "Mean Debug/Student_Distill_Update_Count loss: 102.0000\n"
        "Mean Debug/B300_Box_Adapt_From_Update_Zero loss: 1.0000\n"
        "Time elapsed: 00:05:00\n"
        "ETA: 00:10:00\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "select_workflow_state", lambda: (None, {
        "workflow_id": "highstep_b300_canonical_hybrid_prior_latent_20260718",
        "status": "running",
        "train_pid": 123,
        "train_log": str(log_path),
        "run_dir": str(tmp_path),
        "role": "student",
        "_dashboard_staged_workflow": True,
    }))
    monkeypatch.setattr(module, "pid_alive", lambda _value: True)
    monkeypatch.setattr(module.shutil, "get_terminal_size", lambda _fallback: module.os.terminal_size((100, 24)))

    rendered = module.render(False)

    assert "distill 102  box-adapt ON@U0" in rendered
    assert "distill 102/-" not in rendered
