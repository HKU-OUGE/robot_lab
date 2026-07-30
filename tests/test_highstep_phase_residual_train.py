from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "highstep_phase_residual_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase_residual_train", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_output_never_exceeds_frozen_bound():
    module = _load_module()
    bound = torch.tensor([0.2] * 12 + [0.006] * 4)
    model = module.BoundedPhaseResidualMLP(bound)
    output = model(torch.randn(64, 578))
    assert torch.all(torch.abs(output) <= bound + 1.0e-7)


def test_residual_bound_uses_train_q99_and_hard_cap():
    module = _load_module()
    target = torch.zeros(20, 16)
    target[:10, :12] = 1.0
    target[:10, 12:] = 0.1
    train = torch.zeros(20, dtype=torch.bool)
    train[:10] = True
    bound = module.compute_residual_bounds(target, train)
    assert torch.allclose(bound[:12], torch.full((12,), 0.2))
    assert torch.allclose(bound[12:], torch.full((4,), 0.006))


def test_reference_binding_is_resolved_from_frozen_source_manifest(tmp_path):
    module = _load_module()
    selected = tmp_path / "selected.json"
    selected.write_text('{"kind":"selected"}\n', encoding="utf-8")
    selected_sha = hashlib.sha256(selected.read_bytes()).hexdigest()
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "selected_reference_manifest_path": str(selected),
                "selected_reference_manifest_sha256": selected_sha,
                "selected_reference_sha256": "reference-sha",
                "metadata": {"source_timing_amendment_sha256": "timing-sha"},
            }
        ),
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    merged = {
        "metadata": {"source_timing_amendment_sha256": "timing-sha"},
        "source_batches": [{"manifest_path": str(source), "manifest_sha256": source_sha}],
    }
    binding = module.load_reference_binding(merged)
    assert binding["selected_reference_sha256"] == "reference-sha"
    assert binding["source_timing_amendment_sha256"] == "timing-sha"


def test_reference_binding_fails_closed_on_source_manifest_sha_change(tmp_path):
    module = _load_module()
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    merged = {
        "metadata": {"source_timing_amendment_sha256": "timing-sha"},
        "source_batches": [{"manifest_path": str(source), "manifest_sha256": "wrong"}],
    }
    try:
        module.load_reference_binding(merged)
    except RuntimeError as error:
        assert "source dataset manifest SHA mismatch" in str(error)
    else:
        raise AssertionError("changed source manifest must fail closed")


def test_temporal_pairs_never_cross_rollouts():
    module = _load_module()
    episode = torch.repeat_interleave(torch.arange(15), 120)
    step = torch.arange(120).repeat(15)
    split = torch.repeat_interleave(torch.tensor([0] * 12 + [1] * 3), 120)
    left, right = module.temporal_pair_indices(episode, step, split, 0)
    assert left.numel() == 12 * 119
    assert torch.all(episode[left] == episode[right])
    assert torch.all(step[right] == step[left] + 1)


def test_temporal_pairs_accept_contiguous_dagger_prefixes():
    module = _load_module()
    episode = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    step = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    split = torch.zeros(8, dtype=torch.long)
    left, right = module.temporal_pair_indices(episode, step, split, 0)
    assert left.numel() == 6
    assert torch.all(episode[left] == episode[right])
    assert torch.all(step[right] == step[left] + 1)


def test_small_supervised_problem_improves_without_ppo():
    module = _load_module()
    rng = np.random.default_rng(7)
    episode = np.repeat(np.arange(15), 120)
    step = np.tile(np.arange(120), 15)
    split = np.repeat(np.array([0] * 12 + [1] * 3), 120)
    phase = np.zeros((1800, 8), np.float32)
    stage = np.minimum(step // 20, 5).astype(np.int64)
    phase[:, 0] = step / 119.0
    phase[np.arange(1800), 1 + stage] = 1.0
    phase[:, 7] = (step % 20) / 19.0
    obs = rng.normal(0, 0.1, (1800, 570)).astype(np.float32)
    target = np.zeros((1800, 16), np.float32)
    target[:, 0] = 0.05 * phase[:, 0]
    arrays = {
        "student_obs_570": obs,
        "phase_features_8": phase,
        "residual_target_16": target,
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
