#!/usr/bin/env python3
"""Fail-closed CPU audit of the one approved full E5700 resume source."""

from __future__ import annotations

import hashlib
from pathlib import Path
import torch

CHECKPOINT = Path("/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_Student/2026-07-19_06-43-15_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_attempt2_from_173499/model_179198.pt")
EXPECTED_SHA = "31fe19c7d1ba9872c0b714d594fe6e66c549a52e88fe856ed62206680d37c498"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("E5700 checkpoint SHA mismatch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload["infos"]["robot_lab_algorithm_checkpoint_state"]
    recovery = state["student_recovery"]
    binding = recovery["binding_manifest"]
    optimizer = state["vae_optimizer_state_dict"]
    groups = optimizer["param_groups"]
    if not (
        payload["iter"] == 179198
        and state["student_distill_update_count"] == 5700
        and recovery["effective_update_count"] == 5700
        and binding["effective_update_count"] == 5700
        and binding["warmup_updates"] == 1400
        and binding["warmup_main_action_target"] == "teacher_pre_prior"
        and binding["warmup_prior_box_loss_coefficient"] == 0.0
        and binding["post_warmup_trainable_action_rows"] == [12, 13, 14, 15]
        and binding["optimizer_group_learning_rates"] == [1.0e-3, 1.0e-5]
        and recovery["preregistration_sha256"]
        == "f82aa5f5246a32f308a4513b09a504f11bd188988933fc983cb48f48d53e3665"
        and len(groups) == 2
        and groups[0]["v15_role"] == "estimator"
        and groups[0]["lr"] == 1.0e-3
        and len(groups[0]["params"]) == 16
        and groups[1]["v15_role"] == "box_rows"
        and groups[1]["lr"] == 1.0e-5
        and len(groups[1]["params"]) == 2
        and len(optimizer["state"]) == 18
        and all(param in optimizer["state"] for group in groups for param in group["params"])
    ):
        raise RuntimeError("E5700 count/optimizer/full-checkpoint contract mismatch")
    print(
        "E5700 full resume audit: count=5700, warmup not re-entered, "
        "box adaptation/prior-box active, Adam=18/18, LR=1e-3/1e-5, SHA=OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
