"""Fail-closed same-state DAgger collection for the fixed-condition branch."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

from highstep_phase_residual_dataset import STAGE_NAMES
from highstep_phase_residual_runtime import (
    EXPECTED_JOINT_ORDER,
    PHYSICAL_LOWER,
    PHYSICAL_UPPER,
    sha256_file,
)


WORKFLOW_ID = "highstep_fixed_condition_phase_residual_20260717"
DIRECT_WORKFLOW_ID = "highstep_fixed_condition_phase_direct_action_20260717"

AUTHORITY_PROFILES = {
    "residual": {
        "workflow_id": WORKFLOW_ID,
        "preregistration_kind": "highstep_fixed_condition_phase_residual_dagger_preregistration",
        "training_kind": "highstep_fixed_condition_phase_residual_training",
        "dataset_kind": "highstep_fixed_condition_phase_residual_dagger_dataset",
    },
    "direct_action": {
        "workflow_id": DIRECT_WORKFLOW_ID,
        "preregistration_kind": "highstep_fixed_condition_phase_direct_action_dagger_preregistration",
        "training_kind": "highstep_fixed_condition_phase_direct_action_training",
        "dataset_kind": "highstep_fixed_condition_phase_direct_action_dagger_dataset",
    },
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_bound_json(path: str | Path, expected_sha256: str, name: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is missing: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{name} SHA mismatch: expected={expected_sha256} actual={actual}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} is not a JSON object")
    return resolved, payload


class PhaseResidualDaggerCollector:
    """Record Teacher labels at the exact states driven by the safe candidate."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        preregistration_path: str | Path,
        preregistration_sha256: str,
        training_manifest_path: str | Path,
        training_manifest_sha256: str,
        joint_names: Sequence[str],
        num_envs: int,
        driver_mode: str,
        authority_profile: str = "residual",
    ) -> None:
        try:
            profile = AUTHORITY_PROFILES[authority_profile]
        except KeyError as error:
            raise RuntimeError(f"unknown DAgger authority profile: {authority_profile}") from error
        self.authority_profile = authority_profile
        self.workflow_id = profile["workflow_id"]
        self.dataset_kind = profile["dataset_kind"]
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite DAgger output: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.preregistration_path, prereg = _load_bound_json(
            preregistration_path, preregistration_sha256, "DAgger preregistration"
        )
        round_index = int(prereg.get("round", -1))
        expected_status = f"frozen_before_dagger_round{round_index}_collection"
        if (
            prereg.get("kind") != profile["preregistration_kind"]
            or prereg.get("status") != expected_status
            or prereg.get("workflow_id") != self.workflow_id
            or round_index not in (1, 2)
            or prereg.get("driver_mode") != driver_mode
            or prereg.get("teacher_label") != "same_pre_step_state_post_prior_mapped_target"
            or prereg.get("post_termination_samples") != "recorded_but_excluded_from_training"
        ):
            raise RuntimeError("DAgger preregistration contract changed")
        self.preregistration_sha256 = preregistration_sha256
        self.round = round_index
        self.training_manifest_path, training = _load_bound_json(
            training_manifest_path, training_manifest_sha256, "parent training manifest"
        )
        if (
            training.get("kind") != profile["training_kind"]
            or training.get("workflow_id") != self.workflow_id
            or prereg.get("parent_training_manifest_sha256") != training_manifest_sha256
            or prereg.get("parent_checkpoint_sha256") != training.get("checkpoint_sha256")
        ):
            raise RuntimeError("DAgger parent training binding changed")
        if tuple(str(name) for name in joint_names) != EXPECTED_JOINT_ORDER:
            raise RuntimeError("DAgger joint order changed")
        self.training_manifest_sha256 = training_manifest_sha256
        self.num_envs = int(num_envs)
        if self.num_envs != 15:
            raise RuntimeError("DAgger collection requires exactly 15 fixed-condition environments")
        self.driver_mode = driver_mode
        self.alive = torch.ones(self.num_envs, dtype=torch.bool)
        self.teacher_target_projection_count = 0
        self.teacher_target_projection_max_abs = 0.0
        self._pending: dict[str, torch.Tensor] | None = None
        self._records: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "student_obs_570",
                "phase_features_8",
                "teacher_post_prior_mapped_target_16",
                "safe_teacher_target_16",
                "reference_target_16",
                "residual_target_16",
                "candidate_residual_16",
                "candidate_target_16",
                "joint_pos_16",
                "joint_vel_16",
                "stage",
                "episode_id",
                "step",
                "split",
                "sample_valid",
                "terminated_after_step",
            )
        }

    @staticmethod
    def _check_tensor(name: str, value: object, shape: tuple[int, ...]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(f"DAgger {name} shape changed: {getattr(value, 'shape', None)}")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise RuntimeError(f"DAgger {name} contains non-finite values")
        return value

    def prepare_before_step(
        self,
        *,
        sample: dict[str, torch.Tensor | int],
        teacher_mapped_target: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("DAgger prepare called twice without record_after_step")
        step = int(sample["step"])
        stage = int(sample["stage"])
        if step < 0 or step >= 120 or stage < 0 or stage >= len(STAGE_NAMES):
            raise RuntimeError(f"DAgger phase index changed: step={step} stage={stage}")
        expected = (self.num_envs, 16)
        obs = self._check_tensor(
            "student observation", sample["student_obs_570"], (self.num_envs, 570)
        )
        phase = self._check_tensor(
            "phase features", sample["phase_features_8"], (self.num_envs, 8)
        )
        teacher = self._check_tensor("Teacher target", teacher_mapped_target, expected)
        reference = self._check_tensor("reference target", sample["reference_target_16"], expected)
        candidate_residual = self._check_tensor(
            "candidate residual", sample["candidate_residual_16"], expected
        )
        candidate_target = self._check_tensor(
            "candidate target", sample["candidate_target_16"], expected
        )
        q = self._check_tensor("joint position", joint_pos, expected)
        dq = self._check_tensor("joint velocity", joint_vel, expected)
        lower = torch.tensor(PHYSICAL_LOWER, device=teacher.device, dtype=teacher.dtype)
        upper = torch.tensor(PHYSICAL_UPPER, device=teacher.device, dtype=teacher.dtype)
        safe_teacher = torch.clamp(teacher.detach(), min=lower, max=upper)
        projection = torch.abs(teacher - safe_teacher)
        self.teacher_target_projection_count += int(torch.sum(projection > 0.0).item())
        self.teacher_target_projection_max_abs = max(
            self.teacher_target_projection_max_abs,
            float(torch.max(projection).detach().cpu()),
        )
        episode = torch.arange(self.num_envs, device=teacher.device, dtype=torch.int64)
        split = torch.where(episode < 12, 0, 1)
        self._pending = {
            "student_obs_570": obs.detach().clone(),
            "phase_features_8": phase.detach().clone(),
            "teacher_post_prior_mapped_target_16": teacher.detach().clone(),
            "safe_teacher_target_16": safe_teacher,
            "reference_target_16": reference.detach().clone(),
            "residual_target_16": safe_teacher - reference,
            "candidate_residual_16": candidate_residual.detach().clone(),
            "candidate_target_16": candidate_target.detach().clone(),
            "joint_pos_16": q.detach().clone(),
            "joint_vel_16": dq.detach().clone(),
            "stage": torch.full((self.num_envs,), stage, device=teacher.device, dtype=torch.int64),
            "episode_id": episode,
            "step": torch.full((self.num_envs,), step, device=teacher.device, dtype=torch.int64),
            "split": split,
            "sample_valid": self.alive.to(
                device=teacher.device, dtype=torch.bool
            ).clone(),
        }

    def record_after_step(self, dones: torch.Tensor) -> None:
        if self._pending is None:
            raise RuntimeError("DAgger record_after_step called without a prepared sample")
        if not isinstance(dones, torch.Tensor) or dones.numel() != self.num_envs:
            raise RuntimeError("DAgger dones shape changed")
        done = dones.detach().reshape(-1).cpu().bool()
        terminated_now = self.alive & done
        self._pending["terminated_after_step"] = terminated_now.to(
            device=self._pending["step"].device
        )
        for name, value in self._pending.items():
            self._records[name].append(value.detach().cpu().numpy())
        self.alive &= ~done
        self._pending = None

    def finalize(self, *, metadata: dict[str, Any]) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("cannot finalize DAgger with an unrecorded pending sample")
        if len(self._records["step"]) != 120:
            raise RuntimeError(
                f"DAgger requires exactly 120 active steps; got {len(self._records['step'])}"
            )
        arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in self._records.items()}
        dataset_path = self.output_dir / "dagger_dataset.npz"
        _atomic_npz(dataset_path, arrays)
        valid = arrays["sample_valid"].astype(bool)
        per_env = {
            str(env_id): int(np.sum(valid & (arrays["episode_id"] == env_id)))
            for env_id in range(self.num_envs)
        }
        stage_counts = {
            STAGE_NAMES[index]: int(np.sum(valid & (arrays["stage"] == index)))
            for index in range(len(STAGE_NAMES))
        }
        manifest = {
            "schema_version": 1,
            "kind": self.dataset_kind,
            "status": f"frozen_dagger_round{self.round}_dataset_ready",
            "workflow_id": self.workflow_id,
            "round": self.round,
            "driver_mode": self.driver_mode,
            "teacher_label": "same_pre_step_state_post_prior_mapped_target",
            "preregistration_path": str(self.preregistration_path),
            "preregistration_sha256": self.preregistration_sha256,
            "parent_training_manifest_path": str(self.training_manifest_path),
            "parent_training_manifest_sha256": self.training_manifest_sha256,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "rollout_count": self.num_envs,
            "steps_per_rollout": 120,
            "total_rows": int(arrays["step"].shape[0]),
            "valid_rows": int(np.sum(valid)),
            "invalid_post_termination_rows": int(np.sum(~valid)),
            "teacher_target_projection_count": self.teacher_target_projection_count,
            "teacher_target_projection_max_abs": self.teacher_target_projection_max_abs,
            "valid_rows_by_env": per_env,
            "valid_stage_counts": stage_counts,
            "terminated_env_ids": [
                int(env_id) for env_id, alive in enumerate(self.alive.tolist()) if not alive
            ],
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "metadata": metadata,
        }
        manifest_path = self.output_dir / "dagger_dataset_manifest.json"
        _atomic_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        return manifest
