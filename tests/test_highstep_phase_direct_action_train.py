from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
MODULE_PATH = TOOLS / "highstep_phase_direct_action_train.py"


def _load_module():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("phase_direct_action_train_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_model_output_is_always_inside_physical_envelope():
    module = _load_module()
    model = module.PhaseDirectActionMLP(torch.zeros(578), torch.ones(578))
    output = model(torch.randn(128, 578) * 100.0)
    assert torch.all(output >= module.LOWER - 1.0e-7)
    assert torch.all(output <= module.UPPER + 1.0e-7)


def test_small_direct_action_problem_improves_without_ppo():
    module = _load_module()
    rng = np.random.default_rng(20260717)
    episode = np.repeat(np.arange(15), 24)
    step = np.tile(np.arange(24), 15)
    split = np.repeat(np.array([0] * 12 + [1] * 3), 24)
    stage = np.minimum(step // 4, 5).astype(np.int64)
    phase = np.zeros((episode.size, 8), dtype=np.float32)
    phase[:, 0] = step / 23.0
    phase[np.arange(episode.size), 1 + stage] = 1.0
    phase[:, 7] = (step % 4) / 3.0
    obs = rng.normal(0.0, 0.05, (episode.size, 570)).astype(np.float32)
    center = ((module.LOWER + module.UPPER) * 0.5).numpy()
    half = ((module.UPPER - module.LOWER) * 0.5).numpy()
    normalized = np.zeros((episode.size, 16), dtype=np.float32)
    normalized[:, 0] = 0.35 * phase[:, 0]
    normalized[:, 12] = 0.25 * phase[:, 7]
    target = center.reshape(1, 16) + half.reshape(1, 16) * normalized
    arrays = {
        "student_obs_570": obs,
        "phase_features_8": phase,
        "safe_teacher_target_16": target.astype(np.float32),
        "stage": stage,
        "episode_id": episode.astype(np.int64),
        "step": step.astype(np.int64),
        "split": split.astype(np.int64),
    }
    _, result, history = module.train_model(
        arrays, device=torch.device("cpu"), max_epochs=8, patience=8
    )
    assert history[-1]["validation/loss"] < history[0]["validation/loss"]
    assert result["best_epoch"] >= 1
    assert result["validation_metrics"]["physical_mae"] < 0.1


def test_direct_action_play_binding_uses_direct_mode_and_code_sha():
    source = (ROOT / "scripts" / "rsl_rl" / "base" / "play.py").read_text(
        encoding="utf-8"
    )
    assert "--phase_direct_action" in source
    assert "PhaseDirectActionController" in source
    assert '"highstep_phase_direct_action_inference.py"' in source
    assert '"controller_mode"' in source

