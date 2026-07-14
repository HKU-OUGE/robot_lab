"""Process-independent checkpoint support for algorithm-owned training state.

RSL-RL's stock runner only knows about ``alg.optimizer``.  Algorithms that own
additional optimizers can opt in through the small interface wrapped here:

* ``extra_checkpoint_state_dict()``
* ``load_extra_checkpoint_state_dict(state)``
* ``migrate_legacy_extra_checkpoint_state(...)`` (optional, explicit only)
"""

from __future__ import annotations

from functools import wraps
from typing import Callable


ALGORITHM_STATE_INFO_KEY = "robot_lab_algorithm_checkpoint_state"


class AlgorithmCheckpointError(RuntimeError):
    """Base error for RobotLab algorithm checkpoint continuity failures."""


class AlgorithmCheckpointStateMissingError(AlgorithmCheckpointError):
    """Raised when a full resume lacks required algorithm-owned state."""


def install_algorithm_checkpoint_state_hook(
    runner,
    *,
    legacy_student_distill_update_count: int | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Wrap one runner instance so algorithm-owned state follows save/load.

    ``load_optimizer=False`` intentionally remains a clean weights-only load.
    A full load of a legacy checkpoint fails closed unless the caller supplies
    an explicit, independently verified Student update count.  Even then the
    missing Adam moments cannot be reconstructed and the algorithm migrator
    must retain its freshly initialized optimizer.
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
        infos = original_load(path, load_optimizer=load_optimizer, map_location=map_location)
        if not supports_extra_state or not load_optimizer:
            return infos

        extra_state = infos.get(ALGORITHM_STATE_INFO_KEY) if isinstance(infos, dict) else None
        if extra_state is not None:
            if legacy_student_distill_update_count is not None:
                raise AlgorithmCheckpointError(
                    "Legacy Student migration count was supplied for a checkpoint that already "
                    "contains resumable algorithm state"
                )
            loader(extra_state)
            return infos

        if legacy_student_distill_update_count is None:
            raise AlgorithmCheckpointStateMissingError(
                "Full resume requires algorithm-owned checkpoint state, but this checkpoint has none. "
                "For a verified legacy Student checkpoint, explicitly provide its exact completed "
                "distillation-update count; its missing VAE Adam moments will still start fresh. "
                "Use load_optimizer=False for an intentional weights-only restart."
            )

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
