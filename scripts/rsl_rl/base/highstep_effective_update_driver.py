#!/usr/bin/env python3
"""Effective-update continuation driver for sparse highstep sampling.

The stock runner's ``learn(1)`` API leaves ``current_learning_iteration`` at
the iteration it just executed and unconditionally writes a final checkpoint.
Calling it repeatedly therefore reuses both the logging step and checkpoint
path.  This helper advances the rollout/log clock explicitly, suppresses those
per-call final saves, and saves only on the requested effective-update cadence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable


def run_sparse_effective_update_continuation(
    runner,
    *,
    target_effective_update: int,
    checkpoint_interval: int,
    checkpoint_absolute_origin: int,
    progress_path: str | None = None,
    on_rollout: Callable[[int, int, int], None] | None = None,
) -> None:
    """Run until ``target_effective_update`` without charging FIFO prefills.

    ``checkpoint_absolute_origin`` is the frozen Teacher absolute step.  Thus
    effective E5800 is saved as ``model_179298.pt`` for origin 173499.
    """

    if target_effective_update <= 0 or checkpoint_interval <= 0:
        raise ValueError("invalid sparse continuation target/cadence")

    effective = int(runner.alg.student_distill_update_count)
    if effective >= target_effective_update:
        raise ValueError("sparse continuation source is already at/after target")

    fully_wrapped_save = runner.save
    first_rollout = True

    while effective < target_effective_update:
        before = int(runner.alg.student_distill_update_count)

        # The upstream runner stores the iteration it just ran, not the next
        # one.  Advance once per rollout so terminal logs and W&B are strictly
        # monotonic even when the rollout only fills the episode-safe FIFO.
        runner.current_learning_iteration = int(runner.current_learning_iteration) + 1
        rollout_iteration = int(runner.current_learning_iteration)

        def suppress_stock_save(*_args, **_kwargs) -> None:
            # OnPolicyRunner.learn(1) always performs a final save.  Relative
            # E5800/E5900/... saves are emitted explicitly below instead.
            return None

        runner.save = suppress_stock_save
        try:
            runner.learn(
                num_learning_iterations=1,
                init_at_random_ep_len=first_rollout,
            )
        finally:
            runner.save = fully_wrapped_save
        first_rollout = False

        after = int(runner.alg.student_distill_update_count)
        if after not in (before, before + 1):
            raise RuntimeError(
                f"RL pre-edge effective update count jumped unexpectedly: {before}->{after}"
            )

        if on_rollout is not None:
            on_rollout(before, after, rollout_iteration)
        if progress_path:
            payload = {
                "schema_version": 1,
                "effective_updates": after,
                "previous_effective_updates": before,
                "optimizer_update_performed": after == before + 1,
                "absolute_runner_step": rollout_iteration,
            }
            temporary = progress_path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, progress_path)

        if after == before:
            # Legal sparse-event FIFO prefill: no optimizer step, no effective
            # budget consumption and no checkpoint.  There is intentionally no
            # fixed no-update streak abort here.
            continue

        effective = after
        if effective % checkpoint_interval == 0 or effective == target_effective_update:
            checkpoint_iteration = checkpoint_absolute_origin + effective - 1
            checkpoint_path = os.path.join(
                runner.log_dir, f"model_{checkpoint_iteration}.pt"
            )
            previous_iteration = int(runner.current_learning_iteration)
            runner.current_learning_iteration = checkpoint_iteration
            try:
                fully_wrapped_save(checkpoint_path)
            finally:
                runner.current_learning_iteration = previous_iteration

    if int(runner.alg.student_distill_update_count) != target_effective_update:
        raise RuntimeError("sparse continuation did not terminate at the exact target")
