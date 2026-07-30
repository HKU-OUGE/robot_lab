"""Process-independent checkpoint support for algorithm-owned training state.

RSL-RL's stock runner only knows about ``alg.optimizer``.  Algorithms that own
additional optimizers can opt in through the small interface wrapped here:

* ``extra_checkpoint_state_dict()``
* ``load_extra_checkpoint_state_dict(state)``
* ``migrate_legacy_extra_checkpoint_state(...)`` (optional, explicit only)
"""

from __future__ import annotations

import hashlib
import os
from functools import wraps
from typing import Callable, Mapping


ALGORITHM_STATE_INFO_KEY = "robot_lab_algorithm_checkpoint_state"


class AlgorithmCheckpointError(RuntimeError):
    """Base error for RobotLab algorithm checkpoint continuity failures."""


class AlgorithmCheckpointStateMissingError(AlgorithmCheckpointError):
    """Raised when a full resume lacks required algorithm-owned state."""


def install_algorithm_checkpoint_state_hook(
    runner,
    *,
    legacy_student_distill_update_count: int | None = None,
    verified_legacy_teacher_checkpoint: Mapping[str, object] | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Wrap one runner instance so algorithm-owned state follows save/load.

    ``load_optimizer=False`` intentionally remains a clean weights-only load.
    A full load of a legacy checkpoint fails closed unless the caller supplies
    one of two narrow, independently verified migrations:

    * a Student update count (whose missing VAE Adam moments remain fresh); or
    * a Teacher contract proving that Stage-1 never stepped its separate VAE
      Adam, so the checkpoint's absent *empty* state is exactly reconstructible.

    The Teacher exception is deliberately checkpoint-SHA, iteration,
    algorithm-role and optimizer-shape bound.  It is not a generic legacy
    override and it never permits a fresh PPO optimizer.
    """

    if getattr(runner, "_robot_lab_algorithm_checkpoint_hook_installed", False):
        raise AlgorithmCheckpointError("Algorithm checkpoint hook is already installed on this runner")

    algorithm = runner.alg
    exporter = getattr(algorithm, "extra_checkpoint_state_dict", None)
    loader = getattr(algorithm, "load_extra_checkpoint_state_dict", None)
    migrator = getattr(algorithm, "migrate_legacy_extra_checkpoint_state", None)
    supports_extra_state = callable(exporter) and callable(loader)

    if callable(exporter) != callable(loader):
        raise AlgorithmCheckpointError(
            "Algorithm must implement both extra_checkpoint_state_dict() and "
            "load_extra_checkpoint_state_dict()"
        )
    if legacy_student_distill_update_count is not None:
        if isinstance(legacy_student_distill_update_count, bool) or not isinstance(
            legacy_student_distill_update_count, int
        ):
            raise AlgorithmCheckpointError("legacy_student_distill_update_count must be an integer")
        if legacy_student_distill_update_count < 0:
            raise AlgorithmCheckpointError("legacy_student_distill_update_count must be non-negative")
        if not supports_extra_state or not callable(migrator):
            raise AlgorithmCheckpointError(
                "Explicit legacy Student migration was requested, but this algorithm does not support it"
            )

    teacher_contract = None
    if verified_legacy_teacher_checkpoint is not None:
        if legacy_student_distill_update_count is not None:
            raise AlgorithmCheckpointError(
                "Legacy Student and verified legacy Teacher migrations are mutually exclusive"
            )
        if not isinstance(verified_legacy_teacher_checkpoint, Mapping):
            raise AlgorithmCheckpointError("verified_legacy_teacher_checkpoint must be a mapping")
        teacher_contract = dict(verified_legacy_teacher_checkpoint)
        required = {
            "kind": "verified_legacy_teacher_algorithm_state_v1",
            "algorithm_class": "VAEPPO",
            "distill_stage": 1,
            "student_recovery_stage": "NONE",
            "stock_optimizer_state_entries": 23,
            "stock_optimizer_param_groups": 1,
            "vae_optimizer_state_entries": 0,
            "vae_optimizer_param_groups": 1,
            "stage1_update_contract": "super_ppo_only_no_vae_optimizer_step",
        }
        for key, expected in required.items():
            if teacher_contract.get(key) != expected:
                raise AlgorithmCheckpointError(
                    f"Verified legacy Teacher contract changed at {key}: "
                    f"{teacher_contract.get(key)!r} != {expected!r}"
                )
        checkpoint_path = os.path.realpath(str(teacher_contract.get("checkpoint_path", "")))
        checkpoint_sha = str(teacher_contract.get("checkpoint_sha256", ""))
        checkpoint_iteration = teacher_contract.get("checkpoint_iteration")
        if not os.path.isabs(checkpoint_path) or not os.path.isfile(checkpoint_path):
            raise AlgorithmCheckpointError("Verified legacy Teacher checkpoint path is invalid")
        if len(checkpoint_sha) != 64 or any(character not in "0123456789abcdef" for character in checkpoint_sha):
            raise AlgorithmCheckpointError("Verified legacy Teacher checkpoint SHA256 is invalid")
        if isinstance(checkpoint_iteration, bool) or not isinstance(checkpoint_iteration, int):
            raise AlgorithmCheckpointError("Verified legacy Teacher checkpoint iteration is invalid")
        teacher_contract["checkpoint_path"] = checkpoint_path

    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _prevalidate_verified_teacher_path(path: str) -> None:
        if teacher_contract is None:
            return
        resolved = os.path.realpath(path)
        if resolved != teacher_contract["checkpoint_path"]:
            raise AlgorithmCheckpointError(
                "Verified legacy Teacher override was supplied for a different checkpoint path"
            )
        actual_sha = _sha256_file(resolved)
        if actual_sha != teacher_contract["checkpoint_sha256"]:
            raise AlgorithmCheckpointError(
                "Verified legacy Teacher checkpoint SHA256 changed before runner.load"
            )

    def _accept_verified_empty_teacher_extra_state(path: str) -> None:
        if teacher_contract is None:
            raise AlgorithmCheckpointStateMissingError(
                "Full resume requires algorithm-owned checkpoint state, but this checkpoint has none. "
                "For a verified legacy Student checkpoint, explicitly provide its exact completed "
                "distillation-update count; its missing VAE Adam moments will still start fresh. "
                "Use load_optimizer=False for an intentional weights-only restart."
            )
        _prevalidate_verified_teacher_path(path)
        if type(algorithm).__name__ != teacher_contract["algorithm_class"]:
            raise AlgorithmCheckpointError("Verified legacy Teacher runtime algorithm class changed")
        if int(getattr(algorithm, "distill_stage", -1)) != teacher_contract["distill_stage"]:
            raise AlgorithmCheckpointError("Verified legacy Teacher runtime distill stage changed")
        recovery_stage = str(getattr(algorithm, "student_recovery_stage", "NONE")).upper()
        if recovery_stage != teacher_contract["student_recovery_stage"]:
            raise AlgorithmCheckpointError("Verified legacy Teacher runtime recovery stage changed")
        if int(getattr(runner, "current_learning_iteration", -1)) != teacher_contract["checkpoint_iteration"]:
            raise AlgorithmCheckpointError("Verified legacy Teacher checkpoint iteration changed")

        stock_optimizer = getattr(algorithm, "optimizer", None)
        vae_optimizer = getattr(algorithm, "vae_optimizer", None)
        if stock_optimizer is None or vae_optimizer is None or stock_optimizer is vae_optimizer:
            raise AlgorithmCheckpointError("Verified legacy Teacher requires distinct PPO and VAE Adam optimizers")
        stock_state = stock_optimizer.state_dict()
        vae_state = vae_optimizer.state_dict()
        if (
            len(stock_state.get("state", {})) != teacher_contract["stock_optimizer_state_entries"]
            or len(stock_state.get("param_groups", [])) != teacher_contract["stock_optimizer_param_groups"]
        ):
            raise AlgorithmCheckpointError("Verified legacy Teacher PPO Adam was not fully restored")
        if (
            len(vae_state.get("state", {})) != teacher_contract["vae_optimizer_state_entries"]
            or len(vae_state.get("param_groups", [])) != teacher_contract["vae_optimizer_param_groups"]
        ):
            raise AlgorithmCheckpointError(
                "Verified legacy Teacher VAE Adam is not the proven never-stepped empty state"
            )
        log(
            "[INFO] Verified legacy Teacher checkpoint migration: restored the exact 23-state PPO Adam; "
            "the separate Stage-1 VAE Adam is provably never-stepped and remains identically empty."
        )

    original_save = runner.save
    original_load = runner.load

    @wraps(original_save)
    def save_with_algorithm_state(path: str, infos: dict | None = None):
        if not supports_extra_state:
            return original_save(path, infos)
        if infos is None:
            checkpoint_infos = {}
        elif isinstance(infos, dict):
            checkpoint_infos = dict(infos)
        else:
            raise AlgorithmCheckpointError(
                f"Runner checkpoint infos must be a dict or None, got {type(infos).__name__}"
            )
        if ALGORITHM_STATE_INFO_KEY in checkpoint_infos:
            raise AlgorithmCheckpointError(
                f"Checkpoint infos already contains reserved key {ALGORITHM_STATE_INFO_KEY!r}"
            )
        extra_state = exporter()
        if not isinstance(extra_state, dict):
            raise AlgorithmCheckpointError(
                "extra_checkpoint_state_dict() must return a dict, "
                f"got {type(extra_state).__name__}"
            )
        checkpoint_infos[ALGORITHM_STATE_INFO_KEY] = extra_state
        return original_save(path, checkpoint_infos)

    @wraps(original_load)
    def load_with_algorithm_state(
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        if load_optimizer:
            _prevalidate_verified_teacher_path(path)
        infos = original_load(path, load_optimizer=load_optimizer, map_location=map_location)
        if not supports_extra_state or not load_optimizer:
            return infos

        extra_state = infos.get(ALGORITHM_STATE_INFO_KEY) if isinstance(infos, dict) else None
        if extra_state is not None:
            if legacy_student_distill_update_count is not None or teacher_contract is not None:
                raise AlgorithmCheckpointError(
                    "A legacy migration override was supplied for a checkpoint that already contains "
                    "resumable algorithm state"
                )
            loader(extra_state)
            return infos

        if legacy_student_distill_update_count is None:
            _accept_verified_empty_teacher_extra_state(path)
            return infos

        migrator(student_distill_update_count=legacy_student_distill_update_count)
        log(
            "[WARN] Explicit legacy Student checkpoint migration: restored "
            f"student_distill_update_count={legacy_student_distill_update_count}, but the unavailable "
            "VAE optimizer moments remain freshly initialized."
        )
        return infos

    runner.save = save_with_algorithm_state
    runner.load = load_with_algorithm_state
    runner._robot_lab_algorithm_checkpoint_hook_installed = True
