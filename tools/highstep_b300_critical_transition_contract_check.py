#!/usr/bin/env python3
"""Single CPU tensor assertion allowed by the v1.0 preregistration."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).parents[1] / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/vae_ppo.py"
)
SPEC = importlib.util.spec_from_file_location("critical_transition_vae_ppo", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load contract module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HIGHSTEP_FRONT_DIAGONAL_ACTION_INDICES = MODULE.HIGHSTEP_FRONT_DIAGONAL_ACTION_INDICES
HIGHSTEP_REAR_DIAGONAL_ACTION_INDICES = MODULE.HIGHSTEP_REAR_DIAGONAL_ACTION_INDICES
highstep_balanced_transition_indices = MODULE.highstep_balanced_transition_indices
highstep_cross_rollout_transition_windows = MODULE.highstep_cross_rollout_transition_windows
highstep_critical_transition_window_masks = MODULE.highstep_critical_transition_window_masks
highstep_phase_diagonal_action_loss = MODULE.highstep_phase_diagonal_action_loss


def gradient(error: torch.Tensor, front: torch.Tensor, rear: torch.Tensor) -> torch.Tensor:
    error = error.clone().requires_grad_(True)
    base = 2.0 * torch.mean(torch.square(error[:, :12]), dim=-1)
    extra = highstep_phase_diagonal_action_loss(
        torch.square(error), front, rear, 2.0, 2.0
    )
    torch.mean(base + extra).backward()
    return error.grad.detach()


def main() -> int:
    assert HIGHSTEP_FRONT_DIAGONAL_ACTION_INDICES == (1, 5, 9)
    assert HIGHSTEP_REAR_DIAGONAL_ACTION_INDICES == (2, 6, 10)
    error = torch.ones(8, 16, dtype=torch.float64)
    zero = torch.zeros(8, dtype=torch.float64)
    one = torch.ones(8, dtype=torch.float64)
    baseline = gradient(error, zero, zero)
    front = gradient(error, one, zero)
    rear = gradient(error, zero, one)

    # Gate-off gradient is exactly the unchanged base loss.
    assert torch.equal(baseline, gradient(error, zero, zero))
    ordinary = baseline[0, 0].item()
    for index in range(12):
        expected_front = 3.0 if index in (1, 5, 9) else 1.0
        expected_rear = 3.0 if index in (2, 6, 10) else 1.0
        assert front[0, index].item() / ordinary == expected_front
        assert rear[0, index].item() / ordinary == expected_rear
    assert torch.equal(front[:, 12:], baseline[:, 12:])
    assert torch.equal(rear[:, 12:], baseline[:, 12:])

    # One front transition is deliberately followed by a terminal.  Its +10
    # window must not leak into the next episode.
    context = torch.zeros(32, 2, 4)
    dones = torch.zeros(32, 2, 1, dtype=torch.uint8)
    context[10:14, 0, 0] = 1.0
    dones[14, 0, 0] = 1
    context[20:24, 0, 2] = 1.0
    context[8:12, 1, 1] = 1.0
    context[22:26, 1, 3] = 1.0
    front_window, rear_window = highstep_critical_transition_window_masks(
        context, dones, radius=10
    )
    assert front_window[:15, 0].any()
    assert not front_window[15:, 0].any()
    assert rear_window[:, 0].any() and rear_window[:, 1].any()

    # Cross-rollout contract: the bounded carry contains the previous
    # rollout's final 20 steps. An event in its last 10 steps and an event in
    # the live rollout's first 10 steps must each produce a real 21-frame
    # window. A done inside that interval must reject the whole event window.
    carry_context = torch.zeros(20, 3, 4)
    carry_dones = torch.zeros(20, 3, 1, dtype=torch.uint8)
    live_context = torch.zeros(24, 3, 4)
    live_dones = torch.zeros(24, 3, 1, dtype=torch.uint8)
    carry_context[15, 0, 0] = 1.0  # previous rollout, final 10 steps
    live_context[5, 1, 2] = 1.0  # next rollout, first 10 steps
    carry_context[15, 2, 0] = 1.0
    carry_dones[18, 2, 0] = 1  # forbids crossing into the next episode
    cross_front, cross_rear, carry_steps = highstep_cross_rollout_transition_windows(
        live_context,
        live_dones,
        radius=10,
        carry_stage_context=carry_context,
        carry_dones=carry_dones,
    )
    assert carry_steps == 20
    assert int(cross_front[:, 0].sum().item()) == 21
    assert int(cross_rear[:, 1].sum().item()) == 21
    assert int(cross_front[:, 2].sum().item()) == 0
    assert not bool(cross_front[:, 0].any() and cross_front[:, 1].any())
    assert not bool(cross_rear[:, 0].any() and cross_rear[:, 1].any())

    num_envs = live_context.shape[1]
    original_pool = torch.arange(carry_steps * num_envs, cross_front.numel())
    indices, sources = highstep_balanced_transition_indices(
        cross_front.flatten(),
        cross_rear.flatten(),
        batch_size=400,
        original_pool=original_pool,
    )
    counts = [int((sources == value).sum().item()) for value in range(3)]
    assert counts == [100, 100, 200]
    assert bool(cross_front.flatten()[indices[sources == 0]].all().item())
    assert bool(cross_rear.flatten()[indices[sources == 1]].all().item())
    assert bool((indices[sources == 2] >= carry_steps * num_envs).all().item())

    print(json.dumps({
        "ok": True,
        "joint_order_indices": {"FR": [1, 5, 9], "RL": [2, 6, 10]},
        "gated_weight_ratio": 3.0,
        "sampler_counts": counts,
        "episode_boundary_preserved": True,
        "previous_rollout_tail_window_steps": 21,
        "next_rollout_head_window_steps": 21,
        "cross_rollout_fifo_steps": 20,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
