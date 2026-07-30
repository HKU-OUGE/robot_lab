#!/usr/bin/env python3
"""CPU-only assertions for the frozen E5700 -> E7700 continuation contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
spec = importlib.util.spec_from_file_location("rl_preedge_vae_ppo", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

DRIVER_PATH = ROOT / "scripts/rsl_rl/base/highstep_effective_update_driver.py"
driver_spec = importlib.util.spec_from_file_location("highstep_effective_update_driver", DRIVER_PATH)
driver = importlib.util.module_from_spec(driver_spec)
assert driver_spec and driver_spec.loader
driver_spec.loader.exec_module(driver)


class _FakeAlgorithm:
    def __init__(self, decisions: list[bool], start: int):
        self.decisions = list(decisions)
        self.student_distill_update_count = start
        self.parameter = 0


class _FakeRunner:
    def __init__(self, log_dir: str, decisions: list[bool], start: int):
        self.alg = _FakeAlgorithm(decisions, start)
        self.log_dir = log_dir
        self.current_learning_iteration = 179198
        self.logged_steps: list[int] = []
        self.saved: list[tuple[str, int, int]] = []

    def learn(self, *, num_learning_iterations: int, init_at_random_ep_len: bool) -> None:
        assert num_learning_iterations == 1
        self.logged_steps.append(self.current_learning_iteration)
        updated = self.alg.decisions.pop(0)
        if updated:
            self.alg.parameter += 1
            self.alg.student_distill_update_count += 1
        # Reproduce the stock runner's unconditional final save.  The sparse
        # continuation driver must suppress this call.
        self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos=None) -> None:
        self.saved.append(
            (path, self.current_learning_iteration, self.alg.student_distill_update_count)
        )


def main() -> int:
    short_context = torch.zeros(24, 2, 5)
    short_dones = torch.zeros(24, 2, 1, dtype=torch.uint8)
    assert not module.highstep_rl_preedge_window_mask(
        short_context, short_dones, 30, 10
    ).any()
    context = torch.zeros(64, 3, 5)
    dones = torch.zeros(64, 3, 1, dtype=torch.uint8)
    context[35, 0, 4] = 1.0
    context[35, 1, 4] = 1.0
    dones[20, 1, 0] = 1
    mask = module.highstep_rl_preedge_window_mask(context, dones, 30, 10)
    assert int(mask[:, 0].sum()) == 41
    assert int(mask[:, 1].sum()) == 0
    assert int(mask[:, 2].sum()) == 0

    front = torch.zeros(400, dtype=torch.bool)
    rear = torch.zeros(400, dtype=torch.bool)
    front[:41] = True
    rear[100:141] = True
    _, source = module.highstep_balanced_transition_indices(front, rear, 400)
    assert [int((source == i).sum()) for i in range(3)] == [100, 100, 200]

    error = torch.ones(4, 16, dtype=torch.float64, requires_grad=True)
    base = 2.0 * torch.mean(torch.square(error[:, :12]), dim=-1)
    extra = module.highstep_phase_diagonal_action_loss(
        torch.square(error), torch.ones(4), torch.ones(4), 2.0, 2.0
    )
    torch.mean(base + extra).backward()
    ordinary = error.grad[0, 0]
    for index in range(12):
        ratio = float(error.grad[0, index] / ordinary)
        assert ratio == (3.0 if index in (1, 2, 5, 6, 9, 10) else 1.0)
    assert torch.count_nonzero(error.grad[:, 12:]) == 0

    # More than twenty legal FIFO-prefill rollouts must neither fail nor update
    # parameters/effective count.  The next complete window performs exactly
    # one update, while W&B/log steps remain strictly monotonic.
    with tempfile.TemporaryDirectory() as directory:
        fake = _FakeRunner(directory, [False] * 25 + [True], 5700)
        driver.run_sparse_effective_update_continuation(
            fake,
            target_effective_update=5701,
            checkpoint_interval=1,
            checkpoint_absolute_origin=173499,
            progress_path=str(Path(directory) / "progress.json"),
        )
        assert fake.alg.parameter == 1
        assert fake.alg.student_distill_update_count == 5701
        assert len(fake.logged_steps) == 26
        assert all(b > a for a, b in zip(fake.logged_steps, fake.logged_steps[1:]))
        assert fake.saved == [
            (str(Path(directory) / "model_179199.pt"), 179199, 5701)
        ]
        progress = json.loads((Path(directory) / "progress.json").read_text())
        assert progress["effective_updates"] == 5701
        assert progress["optimizer_update_performed"] is True

        two = _FakeRunner(directory, [True, True], 5700)
        driver.run_sparse_effective_update_continuation(
            two,
            target_effective_update=5702,
            checkpoint_interval=1,
            checkpoint_absolute_origin=173499,
        )
        assert [Path(item[0]).name for item in two.saved] == [
            "model_179199.pt", "model_179200.pt"
        ]
        assert len({item[0] for item in two.saved}) == 2

    print("RL pre-edge [-30,+10], 25/25/50, and unchanged 3x FR/RL weights: OK")
    print(">20 sparse FIFO prefills, monotonic runner steps, and distinct checkpoints: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
