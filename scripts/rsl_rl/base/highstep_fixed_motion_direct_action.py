"""Runtime contracts for the independent fixed-motion Direct-Action branch.

The network predicts a physical, post-prior 16-D safe target.  The adapter
analytically inverts the live production action mapping (including the four
box-action prior) so the value is never mistaken for a raw policy action.
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
    sha256_file,
)
from highstep_phase_residual_inference import _extract_policy_observation


WORKFLOW_ID = "highstep_fixed_motion_direct_action_20260717"
OBS_DIM = 570
PHASE_DIM = 8
FULL_PUSH_VX = 0.7200000286
FULL_PUSH_STEPS = 59
PHASE_EDGES = (0.0, 0.15, 0.35, 0.55, 0.70, 0.85, 1.0)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_authority(path: str | Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if sha256_file(resolved) != expected_sha256:
        raise RuntimeError("fixed-motion preregistration SHA mismatch")
    payload = json.loads(resolved.read_text())
    if (
        payload.get("kind") != "highstep_fixed_motion_direct_action_preregistration"
        or payload.get("status") != "frozen_before_oracle"
        or payload.get("workflow_id") != WORKFLOW_ID
    ):
        raise RuntimeError("fixed-motion preregistration contract changed")
    spec = Path(payload["spec_path"]).expanduser().resolve()
    teacher = Path(payload["teacher_checkpoint"]).expanduser().resolve()
    reference = Path(payload["approved_reference_manifest_path"]).expanduser().resolve()
    if sha256_file(spec) != payload["spec_sha256"]:
        raise RuntimeError("fixed-motion spec SHA mismatch")
    if sha256_file(teacher) != payload["teacher_sha256"]:
        raise RuntimeError("fixed-motion Teacher SHA mismatch")
    if sha256_file(reference) != payload["approved_reference_manifest_sha256"]:
        raise RuntimeError("fixed-motion approved reference SHA mismatch")
    return resolved, payload


class JoystickPhase:
    """Positive-vx integrated phase shared by collection and deployment."""

    def __init__(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> None:
        self.value = torch.zeros(num_envs, device=device, dtype=dtype)

    def features(self, vx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vx = vx.detach().reshape(-1).to(device=self.value.device, dtype=self.value.dtype)
        increment = torch.clamp(vx, min=0.0) / FULL_PUSH_VX / FULL_PUSH_STEPS
        self.value = torch.clamp(self.value + increment, 0.0, 1.0)
        result = torch.zeros(self.value.shape[0], PHASE_DIM, device=self.value.device, dtype=self.value.dtype)
        result[:, 0] = self.value
        stage = torch.zeros_like(self.value, dtype=torch.long)
        for index in range(6):
            lower, upper = PHASE_EDGES[index], PHASE_EDGES[index + 1]
            in_stage = (self.value >= lower) & ((self.value < upper) if index < 5 else (self.value <= upper))
            stage[in_stage] = index
            result[in_stage, 1 + index] = 1.0
            width = max(upper - lower, 1.0e-9)
            result[in_stage, 7] = torch.clamp((self.value[in_stage] - lower) / width, 0.0, 1.0)
        return result, stage


class SafeMappedTargetAdapter:
    """Execute a physical target through the exact live production action term."""

    def __init__(self, action_term: object, num_envs: int) -> None:
        self.term = action_term
        scale = torch.as_tensor(getattr(action_term, "_scale", None)).detach()
        offset = torch.as_tensor(getattr(action_term, "_offset", None)).detach()
        if scale.ndim > 1:
            scale = scale[0]
        if offset.ndim > 1:
            offset = offset[0]
        self.scale = scale.reshape(1, 16)
        self.offset = offset.reshape(1, 16)
        if bool(torch.any(torch.abs(self.scale) <= 1e-12).item()):
            raise RuntimeError("production action mapping has zero scale")
        self.num_envs = int(num_envs)
        self.last_target: torch.Tensor | None = None
        clip = getattr(action_term, "_clip", None)
        if isinstance(clip, torch.Tensor):
            clip = clip.detach()
            self.lower = clip[0, :, 0].reshape(1, 16)
            self.upper = clip[0, :, 1].reshape(1, 16)
        else:
            self.lower = torch.full((1, 16), -float("inf"))
            self.upper = torch.full((1, 16), float("inf"))

    def raw_for_target(self, target: torch.Tensor) -> torch.Tensor:
        target = target.detach().reshape(self.num_envs, 16)
        lower = self.lower.to(device=target.device, dtype=target.dtype).reshape(16)
        upper = self.upper.to(device=target.device, dtype=target.dtype).reshape(16)
        if not bool(torch.all(torch.isfinite(target)).item()):
            raise RuntimeError("safe mapped target contains non-finite values")
        violations = (target < lower - 1e-7) | (target > upper + 1e-7)
        if bool(torch.any(violations).item()):
            indices = torch.nonzero(violations, as_tuple=False)
            details = [
                {
                    "env": int(row[0]), "joint": EXPECTED_JOINT_ORDER[int(row[1])],
                    "value": float(target[row[0], row[1]].item()),
                    "lower": float(lower[row[1]].item()), "upper": float(upper[row[1]].item()),
                }
                for row in indices[:8]
            ]
            raise RuntimeError("safe mapped target exceeds the physical envelope: " + json.dumps(details, sort_keys=True))
        scale = self.scale.to(device=target.device, dtype=target.dtype)
        offset = self.offset.to(device=target.device, dtype=target.dtype)
        raw = (target - offset) / scale
        if hasattr(self.term, "_last_phased_highstep_box_bias"):
            self.term.process_actions(raw)
            ids = getattr(self.term, "_box_action_ids", None)
            bias = getattr(self.term, "_last_phased_highstep_box_bias", None)
            if not isinstance(ids, torch.Tensor) or not isinstance(bias, torch.Tensor):
                raise RuntimeError("production action prior does not expose exact box bias")
            ids = ids.to(device=target.device, dtype=torch.long).flatten()
            bias = bias.to(device=target.device, dtype=target.dtype)
            raw[:, ids] = (target[:, ids] - offset[:, ids] - bias) / scale[:, ids]
            for _ in range(4):
                self.term.process_actions(raw)
                processed = self.term.processed_actions
                error = target - processed
                if float(torch.max(torch.abs(error)).item()) <= 2.5e-7:
                    break
                raw = raw + error / scale
        self.last_target = target.clone()
        return raw

    def audit(self) -> dict[str, Any]:
        if self.last_target is None:
            raise RuntimeError("target adapter audit called before target")
        processed = getattr(self.term, "processed_actions", None)
        if not isinstance(processed, torch.Tensor):
            raise RuntimeError("production action term exposes no processed target")
        delta = torch.abs(processed - self.last_target)
        maximum = float(torch.max(delta).item())
        return {"max_abs": maximum, "exact": maximum <= 1e-6}


class OracleDatasetCollector:
    """Collect one unique Teacher-driven episode at real production tensor boundaries."""

    def __init__(self, output_dir: str | Path, prereg_path: str | Path, prereg_sha: str, *, action_term: object, device: torch.device, dtype: torch.dtype) -> None:
        self.output = Path(output_dir).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite oracle output: {self.output}")
        self.output.mkdir(parents=True)
        self.prereg_path, self.prereg = load_authority(prereg_path, prereg_sha)
        self.prereg_sha = prereg_sha
        self.phase = JoystickPhase(1, device, dtype)
        self.records: dict[str, list[np.ndarray]] = {name: [] for name in (
            "student_obs_570", "phase_features_8", "teacher_post_prior_mapped_target_16",
            "safe_teacher_target_16", "stage", "episode_id", "step", "split",
        )}
        self.pending_obs: torch.Tensor | None = None
        self.pending_phase: torch.Tensor | None = None
        self.pending_stage: torch.Tensor | None = None
        self.step = 0
        self.max_target_projection = 0.0
        clip = getattr(action_term, "_clip", None)
        if isinstance(clip, torch.Tensor):
            self.lower = clip.detach()[0, :, 0].reshape(1, 16).cpu()
            self.upper = clip.detach()[0, :, 1].reshape(1, 16).cpu()
        else:
            self.lower = torch.full((1, 16), -float("inf"))
            self.upper = torch.full((1, 16), float("inf"))

    def prepare(self, policy_obs: object, vx: torch.Tensor) -> None:
        obs = _extract_policy_observation(policy_obs, 1).detach()
        if tuple(obs.shape) != (1, OBS_DIM):
            raise RuntimeError(f"production Student observation changed: {tuple(obs.shape)}")
        phase, stage = self.phase.features(vx)
        self.pending_obs, self.pending_phase, self.pending_stage = obs.clone(), phase, stage

    def record(self, mapped_target: torch.Tensor) -> None:
        if self.pending_obs is None or self.pending_phase is None or self.pending_stage is None:
            raise RuntimeError("oracle record without pre-step observation")
        target = mapped_target.detach().reshape(1, 16)
        lower = self.lower.to(device=target.device, dtype=target.dtype)
        upper = self.upper.to(device=target.device, dtype=target.dtype)
        safe = torch.clamp(target, min=lower, max=upper)
        projection = float(torch.max(torch.abs(safe - target)).item())
        self.max_target_projection = max(self.max_target_projection, projection)
        values = {
            "student_obs_570": self.pending_obs,
            "phase_features_8": self.pending_phase,
            "teacher_post_prior_mapped_target_16": target,
            "safe_teacher_target_16": safe,
            "stage": self.pending_stage,
            "episode_id": torch.zeros(1, device=target.device, dtype=torch.int64),
            "step": torch.tensor([self.step], device=target.device, dtype=torch.int64),
            "split": torch.zeros(1, device=target.device, dtype=torch.int64),
        }
        for name, value in values.items():
            self.records[name].append(value.detach().cpu().numpy())
        self.pending_obs = self.pending_phase = self.pending_stage = None
        self.step += 1

    def finalize(self, behavior: dict[str, Any]) -> dict[str, Any]:
        arrays = {name: np.concatenate(parts, axis=0) for name, parts in self.records.items()}
        if arrays["student_obs_570"].shape != (138, 570):
            raise RuntimeError(f"oracle episode shape changed: {arrays['student_obs_570'].shape}")
        dataset = self.output / "canonical_oracle_dataset.npz"
        _atomic_npz(dataset, arrays)
        passed = bool(
            behavior.get("full_climb") and behavior.get("rear_hold")
            and self.max_target_projection <= 1e-7
        )
        manifest = {
            "schema_version": 1,
            "kind": "highstep_fixed_motion_safe_oracle_dataset",
            "status": "safe_oracle_passed" if passed else "safe_oracle_failed",
            "workflow_id": WORKFLOW_ID,
            "preregistration_path": str(self.prereg_path),
            "preregistration_sha256": self.prereg_sha,
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "unique_episode_count": 1,
            "sample_count": 138,
            "duplicate_approved_runs_added": 0,
            "max_target_projection": self.max_target_projection,
            "behavior": behavior,
            "passed": passed,
        }
        _atomic_json(self.output / "oracle_manifest.json", manifest)
        return manifest


class FixedMotionStudentController:
    """Hash-bound 578->16 Student with continuous joystick phase."""

    def __init__(self, *, training_manifest_path: str | Path, training_manifest_sha256: str,
                 wandb_verification_path: str | Path, wandb_verification_sha256: str,
                 preregistration_path: str | Path, preregistration_sha256: str,
                 environment_checkpoint_path: str | Path, action_term: object,
                 num_envs: int, device: torch.device, dtype: torch.dtype) -> None:
        self.prereg_path, prereg = load_authority(preregistration_path, preregistration_sha256)
        if sha256_file(environment_checkpoint_path) != prereg["teacher_sha256"]:
            raise RuntimeError("Student environment checkpoint is not the frozen Teacher")
        manifest_path = Path(training_manifest_path).expanduser().resolve()
        if sha256_file(manifest_path) != training_manifest_sha256:
            raise RuntimeError("Student training manifest SHA mismatch")
        training = json.loads(manifest_path.read_text())
        if training.get("workflow_id") != WORKFLOW_ID or training.get("status") != "completed":
            raise RuntimeError("Student training manifest contract changed")
        verification_path = Path(wandb_verification_path).expanduser().resolve()
        if sha256_file(verification_path) != wandb_verification_sha256:
            raise RuntimeError("Student W&B verification SHA mismatch")
        verification = json.loads(verification_path.read_text())
        if verification.get("remote_verified") is not True or verification.get("checkpoint_sha256") != training.get("checkpoint_sha256"):
            raise RuntimeError("Student W&B remote verification incomplete")
        script = Path(training["torchscript_path"]).expanduser().resolve()
        if sha256_file(script) != training["torchscript_sha256"]:
            raise RuntimeError("Student TorchScript SHA mismatch")
        self.model = torch.jit.load(str(script), map_location=device).eval()
        self.phase = JoystickPhase(num_envs, device, dtype)
        self.adapter = SafeMappedTargetAdapter(action_term, num_envs)
        self.num_envs = num_envs
        self.device, self.dtype = device, dtype
        self.last_sample: dict[str, torch.Tensor] | None = None
        self.limit_violations = 0

    def action(self, policy_obs: object, vx: torch.Tensor) -> torch.Tensor:
        obs = _extract_policy_observation(policy_obs, self.num_envs).to(device=self.device, dtype=self.dtype)
        phase, stage = self.phase.features(vx)
        target = self.model(torch.cat((obs, phase), dim=1))
        lower = torch.tensor(PHYSICAL_LOWER, device=self.device, dtype=self.dtype)
        upper = torch.tensor(PHYSICAL_UPPER, device=self.device, dtype=self.dtype)
        violations = (target < lower - 1e-7) | (target > upper + 1e-7) | ~torch.isfinite(target)
        self.limit_violations += int(torch.sum(violations).item())
        if bool(torch.any(violations).item()):
            raise RuntimeError("Student emitted illegal safe mapped target")
        self.last_sample = {"student_obs_570": obs.detach(), "phase_features_8": phase.detach(), "candidate_target_16": target.detach(), "stage": stage.detach()}
        return self.adapter.raw_for_target(target)

    def audit(self) -> dict[str, Any]:
        result = self.adapter.audit()
        result["target_limit_violations"] = self.limit_violations
        return result
