"""Fixed-condition Teacher dataset collection for the phase-residual branch.

The collector is intentionally policy-agnostic. ``play.py`` supplies the exact
570-D policy observation before a Teacher action and the live post-prior mapped
target after the same environment step.  The collector replaces only the
10-frame action-history slice with the preregistered safe-action semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

from highstep_phase_residual_runtime import (
    EXPECTED_JOINT_ORDER,
    PHYSICAL_LOWER,
    PHYSICAL_UPPER,
    PhaseResidualReference,
    sha256_file,
)


OBSERVATION_DIM = 570
ACTION_HISTORY_OFFSET = 410
ACTION_HISTORY_LENGTH = 10
PHASE_FEATURE_DIM = 8
STAGE_NAMES = (
    "approach",
    "front_lift",
    "front_support",
    "first_rear",
    "second_rear",
    "rear_hold",
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _action_affine_row(value: object, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach()
    if tensor.ndim == 0:
        tensor = tensor.repeat(16)
    elif tensor.ndim > 1:
        tensor = tensor[0]
    tensor = tensor.reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.repeat(16)
    if tensor.numel() != 16 or not bool(torch.all(torch.isfinite(tensor)).item()):
        raise RuntimeError(f"invalid 16-D {name}")
    return tensor


class PhaseFeatureContract:
    """Frozen six-stage, eight-feature monotonic phase contract."""

    def __init__(self, boundaries: dict[str, Sequence[int]]) -> None:
        parsed: list[tuple[int, int]] = []
        expected_start = 0
        for name in STAGE_NAMES:
            raw = boundaries.get(name)
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise RuntimeError(f"missing two-value stage boundary for {name}")
            start, stop = int(raw[0]), int(raw[1])
            if start != expected_start or stop < start:
                raise RuntimeError(
                    f"non-contiguous stage boundary for {name}: start={start}, stop={stop}, "
                    f"expected_start={expected_start}"
                )
            parsed.append((start, stop))
            expected_start = stop + 1
        self.boundaries = tuple(parsed)
        self.nominal_last_step = parsed[-1][1]

    def features(self, step: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
        step = max(0, int(step))
        stage = len(self.boundaries) - 1
        for index, (_, stop) in enumerate(self.boundaries):
            if step <= stop:
                stage = index
                break
        start, stop = self.boundaries[stage]
        denominator = max(stop - start, 1)
        within = min(max((step - start) / denominator, 0.0), 1.0)
        global_phase = min(step / max(self.nominal_last_step, 1), 1.0)
        result = torch.zeros(PHASE_FEATURE_DIM, device=device, dtype=dtype)
        result[0] = global_phase
        result[1 + stage] = 1.0
        result[7] = within
        return result, stage


class PhaseResidualDatasetCollector:
    """Collect one fixed-length rollout from each parallel Teacher environment."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        selected_manifest_path: str | Path,
        selected_manifest_sha256: str,
        reference: PhaseResidualReference,
        action_term: object,
        joint_names: Sequence[str],
        num_envs: int,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite phase-residual dataset: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=False)

        manifest_path = Path(selected_manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"selected-reference manifest is missing: {manifest_path}")
        actual_manifest_sha = sha256_file(manifest_path)
        if actual_manifest_sha != selected_manifest_sha256:
            raise RuntimeError(
                "selected-reference manifest SHA mismatch: "
                f"expected={selected_manifest_sha256} actual={actual_manifest_sha}"
            )
        selected = json.loads(manifest_path.read_text(encoding="utf-8"))
        if selected.get("status") != "frozen_reference_replay_passed_pending_teacher_dataset_collection":
            raise RuntimeError("selected-reference manifest has the wrong frozen status")
        selected_reference = selected.get("selected_reference", {})
        if selected_reference.get("sha256") != reference.sha256:
            raise RuntimeError("selected-reference CSV does not match the frozen manifest")
        if tuple(str(name) for name in joint_names) != EXPECTED_JOINT_ORDER:
            raise RuntimeError("dataset joint order does not match the frozen 16-joint contract")
        self.num_envs = int(num_envs)
        if self.num_envs != 15:
            raise RuntimeError(f"dataset collection requires exactly 15 rollouts; got {self.num_envs}")

        self.selected_manifest_path = manifest_path
        self.selected_manifest_sha256 = actual_manifest_sha
        self.selected_manifest = selected
        self.reference = reference
        self.phase = PhaseFeatureContract(selected["stage_boundaries_control_step"])
        self.scale_cpu = _action_affine_row(getattr(action_term, "_scale", None), name="action scale").cpu()
        self.offset_cpu = _action_affine_row(getattr(action_term, "_offset", None), name="action offset").cpu()
        if bool(torch.any(torch.abs(self.scale_cpu) <= 1.0e-12).item()):
            raise RuntimeError("dataset action mapping contains a zero scale")
        self.action_history: torch.Tensor | None = None
        self._pending_obs: torch.Tensor | None = None
        self._pending_phase: torch.Tensor | None = None
        self._pending_stage: int | None = None
        self._pending_step: int | None = None
        self._records: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "student_obs_570",
                "phase_features_8",
                "teacher_post_prior_mapped_target_16",
                "safe_teacher_target_16",
                "reference_target_16",
                "residual_target_16",
                "joint_pos_16",
                "joint_vel_16",
                "stage",
                "episode_id",
                "step",
                "split",
            )
        }
        self.terminated_envs = torch.zeros(self.num_envs, dtype=torch.bool)

    def _append_safe_action_history(self, safe_target: torch.Tensor) -> None:
        device = safe_target.device
        dtype = safe_target.dtype
        if self.action_history is None:
            self.action_history = torch.zeros(
                self.num_envs,
                ACTION_HISTORY_LENGTH,
                16,
                device=device,
                dtype=dtype,
            )
        scale = self.scale_cpu.to(device=device, dtype=dtype).reshape(1, 16)
        offset = self.offset_cpu.to(device=device, dtype=dtype).reshape(1, 16)
        safe_action = (safe_target - offset) / scale
        self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
        self.action_history[:, -1, :] = safe_action

    def prime_action_history(self, teacher_mapped_target: torch.Tensor) -> None:
        """Advance only the safe action history during the source-equivalent settle."""
        if not isinstance(teacher_mapped_target, torch.Tensor) or tuple(
            teacher_mapped_target.shape
        ) != (self.num_envs, 16):
            raise RuntimeError("settle Teacher target shape changed")
        lower = torch.tensor(
            PHYSICAL_LOWER,
            device=teacher_mapped_target.device,
            dtype=teacher_mapped_target.dtype,
        )
        upper = torch.tensor(
            PHYSICAL_UPPER,
            device=teacher_mapped_target.device,
            dtype=teacher_mapped_target.dtype,
        )
        safe_target = torch.clamp(teacher_mapped_target.detach(), min=lower, max=upper)
        self._append_safe_action_history(safe_target)

    def prepare_observation(self, policy_obs: object, step: int) -> torch.Tensor:
        if not isinstance(policy_obs, torch.Tensor):
            keys = policy_obs.keys() if hasattr(policy_obs, "keys") else ()
            if "policy" in keys:
                policy_obs = policy_obs["policy"]
            elif hasattr(policy_obs, "policy"):
                policy_obs = policy_obs.policy
        if not isinstance(policy_obs, torch.Tensor) or policy_obs.ndim != 2:
            raise RuntimeError("Teacher policy observation must be a rank-2 tensor")
        if tuple(policy_obs.shape) != (self.num_envs, OBSERVATION_DIM):
            raise RuntimeError(
                f"Teacher policy observation shape changed: {tuple(policy_obs.shape)}"
            )
        if self._pending_obs is not None:
            raise RuntimeError("dataset collector prepare_observation called twice without record_after_step")
        if self.action_history is None:
            self.action_history = torch.zeros(
                self.num_envs,
                ACTION_HISTORY_LENGTH,
                16,
                device=policy_obs.device,
                dtype=policy_obs.dtype,
            )
        student_obs = policy_obs.detach().clone()
        student_obs[:, ACTION_HISTORY_OFFSET:] = self.action_history.reshape(self.num_envs, -1)
        phase_row, stage = self.phase.features(
            step, device=policy_obs.device, dtype=policy_obs.dtype
        )
        phase = phase_row.reshape(1, -1).expand(self.num_envs, -1).clone()
        self._pending_obs = student_obs
        self._pending_phase = phase
        self._pending_stage = stage
        self._pending_step = int(step)
        return student_obs

    def record_after_step(
        self,
        *,
        teacher_mapped_target: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if self._pending_obs is None or self._pending_phase is None:
            raise RuntimeError("record_after_step called without a prepared observation")
        assert self._pending_stage is not None and self._pending_step is not None
        tensors = {
            "teacher target": teacher_mapped_target,
            "joint position": joint_pos,
            "joint velocity": joint_vel,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (self.num_envs, 16):
                raise RuntimeError(f"{name} shape changed: {getattr(tensor, 'shape', None)}")
            if not bool(torch.all(torch.isfinite(tensor)).item()):
                raise RuntimeError(f"{name} contains non-finite values")

        device = teacher_mapped_target.device
        dtype = teacher_mapped_target.dtype
        lower = torch.tensor(PHYSICAL_LOWER, device=device, dtype=dtype)
        upper = torch.tensor(PHYSICAL_UPPER, device=device, dtype=dtype)
        safe_target = torch.clamp(teacher_mapped_target.detach(), min=lower, max=upper)
        reference_index = min(self._pending_step, self.reference.frame_count - 1)
        reference_target = self.reference.targets_cpu[reference_index].to(
            device=device, dtype=dtype
        ).reshape(1, 16).expand(self.num_envs, -1)
        residual_target = safe_target - reference_target

        episode_id = torch.arange(self.num_envs, device=device, dtype=torch.int64)
        split = torch.where(episode_id < 12, 0, 1)
        stage = torch.full(
            (self.num_envs,), self._pending_stage, device=device, dtype=torch.int64
        )
        step = torch.full(
            (self.num_envs,), self._pending_step, device=device, dtype=torch.int64
        )
        batch = {
            "student_obs_570": self._pending_obs,
            "phase_features_8": self._pending_phase,
            "teacher_post_prior_mapped_target_16": teacher_mapped_target.detach(),
            "safe_teacher_target_16": safe_target,
            "reference_target_16": reference_target,
            "residual_target_16": residual_target,
            "joint_pos_16": joint_pos.detach(),
            "joint_vel_16": joint_vel.detach(),
            "stage": stage,
            "episode_id": episode_id,
            "step": step,
            "split": split,
        }
        for name, value in batch.items():
            self._records[name].append(value.detach().cpu().numpy())

        self._append_safe_action_history(safe_target)
        self.terminated_envs |= dones.detach().reshape(-1).cpu().bool()
        self._pending_obs = None
        self._pending_phase = None
        self._pending_stage = None
        self._pending_step = None

    def finalize(
        self,
        *,
        metadata: dict[str, Any],
        env_hold_success: Sequence[bool],
        env_full_climb_success: Sequence[bool],
    ) -> dict[str, Any]:
        if self._pending_obs is not None:
            raise RuntimeError("cannot finalize with an unrecorded pending observation")
        if not self._records["step"]:
            raise RuntimeError("cannot finalize an empty phase-residual dataset")
        hold = [bool(value) for value in env_hold_success]
        full = [bool(value) for value in env_full_climb_success]
        if len(hold) != self.num_envs or len(full) != self.num_envs:
            raise RuntimeError("per-environment behavior result length mismatch")

        arrays = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self._records.items()
        }
        dataset_path = self.output_dir / "teacher_dataset.npz"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".teacher_dataset.", suffix=".npz", dir=self.output_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, dataset_path)
        finally:
            temporary.unlink(missing_ok=True)

        stage_counts = {
            STAGE_NAMES[index]: int(np.sum(arrays["stage"] == index))
            for index in range(len(STAGE_NAMES))
        }
        terminated = [
            int(index) for index, value in enumerate(self.terminated_envs.tolist()) if value
        ]
        valid = not terminated and all(hold) and all(full)
        manifest = {
            "schema_version": 1,
            "kind": "highstep_fixed_condition_phase_residual_teacher_dataset",
            "status": "frozen_teacher_dataset_ready" if valid else "invalid_teacher_dataset",
            "workflow_id": "highstep_fixed_condition_phase_residual_20260717",
            "selected_reference_manifest_path": str(self.selected_manifest_path),
            "selected_reference_manifest_sha256": self.selected_manifest_sha256,
            "selected_reference_sha256": self.reference.sha256,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "rollout_count": self.num_envs,
            "train_rollouts": 12,
            "validation_rollouts": 3,
            "samples": int(arrays["step"].shape[0]),
            "steps_per_rollout": int(len(self._records["step"])),
            "stage_counts": stage_counts,
            "terminated_env_ids": terminated,
            "env_hold_success": hold,
            "env_full_climb_success": full,
            "valid": valid,
            "observation_contract": {
                "dim": OBSERVATION_DIM,
                "unchanged_prefix_dim": ACTION_HISTORY_OFFSET,
                "replaced_action_history_dim": OBSERVATION_DIM - ACTION_HISTORY_OFFSET,
                "history_length": ACTION_HISTORY_LENGTH,
                "history_order": "oldest_to_newest",
                "history_value": "inverse_map_of_safe_final_joint_target_using_unchanged_offset_and_scale",
            },
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "metadata": metadata,
        }
        manifest_path = self.output_dir / "dataset_manifest.json"
        _write_atomic(manifest_path, _canonical_json_bytes(manifest))
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        return manifest
