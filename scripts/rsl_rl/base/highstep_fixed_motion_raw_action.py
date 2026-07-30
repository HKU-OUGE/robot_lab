"""Raw-action runtime for the independent user-approved fixed-motion branch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from highstep_phase_residual_runtime import sha256_file
from highstep_phase_residual_inference import _extract_policy_observation


WORKFLOW_ID = "highstep_fixed_motion_raw_action_20260717"
PHASE_ONLY_DEPLOYMENT_WORKFLOW_ID = "highstep_fixed_motion_phase_only_deployment_20260717"
ONE_SHOT_DEPLOYMENT_WORKFLOW_ID = "highstep_fixed_motion_one_shot_deployment_20260718"
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


def load_authority(path: str | Path, expected_sha: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if sha256_file(resolved) != expected_sha:
        raise RuntimeError("raw-action preregistration SHA mismatch")
    payload = json.loads(resolved.read_text())
    if (
        payload.get("kind") != "highstep_fixed_motion_raw_action_preregistration"
        or payload.get("status") != "frozen_before_passthrough_smoke"
        or payload.get("workflow_id") != WORKFLOW_ID
    ):
        raise RuntimeError("raw-action preregistration contract changed")
    bindings = (
        (payload["spec_path"], payload["spec_sha256"], "spec"),
        (payload["teacher_checkpoint"], payload["teacher_sha256"], "Teacher"),
        (payload["approved_reference_manifest_path"], payload["approved_reference_manifest_sha256"], "reference"),
        (payload["raw_target_audit_path"], payload["raw_target_audit_sha256"], "raw audit"),
    )
    for bound_path, bound_sha, name in bindings:
        if sha256_file(bound_path) != bound_sha:
            raise RuntimeError(f"raw-action {name} SHA mismatch")
    return resolved, payload


def load_phase_only_deployment_authority(path: str | Path, expected_sha: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if sha256_file(resolved) != expected_sha:
        raise RuntimeError("phase-only deployment preregistration SHA mismatch")
    payload = json.loads(resolved.read_text())
    if (
        payload.get("kind") != "highstep_fixed_motion_phase_only_deployment_preregistration"
        or payload.get("status") != "frozen_before_isaac_validation"
        or payload.get("workflow_id") != PHASE_ONLY_DEPLOYMENT_WORKFLOW_ID
    ):
        raise RuntimeError("phase-only deployment preregistration contract changed")
    for key, sha_key in (
        ("spec_path", "spec_sha256"),
        ("source_training_preregistration", "source_training_preregistration_sha256"),
        ("user_visual_approval", "user_visual_approval_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("torchscript", "torchscript_sha256"),
        ("training_manifest", "training_manifest_sha256"),
        ("wandb_verification", "wandb_verification_sha256"),
    ):
        if sha256_file(payload[key]) != payload[sha_key]:
            raise RuntimeError(f"phase-only deployment binding changed: {key}")
    if payload.get("automatic_real_robot_deployment_authorized") is not False:
        raise RuntimeError("phase-only deployment authority must forbid automatic real-robot deployment")
    return resolved, payload


def load_one_shot_deployment_authority(path: str | Path, expected_sha: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if sha256_file(resolved) != expected_sha:
        raise RuntimeError("one-shot deployment preregistration SHA mismatch")
    payload = json.loads(resolved.read_text())
    if (
        payload.get("kind") != "highstep_fixed_motion_one_shot_deployment_preregistration"
        or payload.get("status") != "frozen_before_implementation"
        or payload.get("workflow_id") != ONE_SHOT_DEPLOYMENT_WORKFLOW_ID
        or payload.get("training_forbidden") is not True
        or payload.get("automatic_real_robot_deployment_authorized") is not False
    ):
        raise RuntimeError("one-shot deployment authority changed")
    for key, sha_key in (
        ("spec_path", "spec_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("torchscript", "torchscript_sha256"),
    ):
        if sha256_file(payload[key]) != payload[sha_key]:
            raise RuntimeError(f"one-shot deployment binding changed: {key}")
    if payload["sequence"]["segments"] != [[0, 20, 0.0], [21, 79, FULL_PUSH_VX], [80, 137, 0.0]]:
        raise RuntimeError("one-shot deployment sequence changed")
    return resolved, payload


class FixedSequenceClock:
    """The deployment clock: one trigger, then exactly 138 inference samples."""

    def __init__(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> None:
        self.value = torch.zeros(num_envs, device=device, dtype=dtype)
        self.step = 0

    @staticmethod
    def command_for_step(step: int) -> float:
        if step < 0 or step >= 138:
            return 0.0
        return FULL_PUSH_VX if 21 <= step < 80 else 0.0

    def features(self) -> tuple[torch.Tensor, torch.Tensor]:
        vx = self.command_for_step(self.step)
        self.value = torch.clamp(self.value + vx / FULL_PUSH_VX / FULL_PUSH_STEPS, 0.0, 1.0)
        self.value = torch.where(self.value >= 1.0 - 1.0e-6, torch.ones_like(self.value), self.value)
        result = torch.zeros(self.value.shape[0], 8, device=self.value.device, dtype=self.value.dtype)
        result[:, 0] = self.value
        stage = torch.zeros_like(self.value, dtype=torch.long)
        for index in range(6):
            lower, upper = PHASE_EDGES[index], PHASE_EDGES[index + 1]
            mask = (self.value >= lower) & ((self.value < upper) if index < 5 else (self.value <= upper))
            stage[mask] = index
            result[mask, 1 + index] = 1.0
            result[mask, 7] = torch.clamp((self.value[mask] - lower) / (upper - lower), 0.0, 1.0)
        self.step += 1
        return result, stage

    def reset(self) -> None:
        self.value.zero_()
        self.step = 0


class JoystickPhase:
    def __init__(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> None:
        self.value = torch.zeros(num_envs, device=device, dtype=dtype)

    def features(self, vx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vx = vx.detach().reshape(-1).to(self.value)
        self.value = torch.clamp(
            self.value + torch.clamp(vx, min=0.0) / FULL_PUSH_VX / FULL_PUSH_STEPS,
            0.0,
            1.0,
        )
        self.value = torch.where(self.value >= 1.0 - 1.0e-6, torch.ones_like(self.value), self.value)
        result = torch.zeros(self.value.shape[0], 8, device=self.value.device, dtype=self.value.dtype)
        result[:, 0] = self.value
        stage = torch.zeros_like(self.value, dtype=torch.long)
        for index in range(6):
            lower, upper = PHASE_EDGES[index], PHASE_EDGES[index + 1]
            mask = (self.value >= lower) & ((self.value < upper) if index < 5 else (self.value <= upper))
            stage[mask] = index
            result[mask, 1 + index] = 1.0
            result[mask, 7] = torch.clamp((self.value[mask] - lower) / (upper - lower), 0.0, 1.0)
        return result, stage

    def reset(self) -> None:
        self.value.zero_()


class RawPassthroughCollector:
    """Capture one Teacher raw-action episode without changing its action tensor."""

    def __init__(self, output_dir: str | Path, preregistration: str | Path, preregistration_sha: str,
                 *, device: torch.device, dtype: torch.dtype) -> None:
        self.output = Path(output_dir).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite raw smoke output: {self.output}")
        self.output.mkdir(parents=True)
        self.prereg_path, self.prereg = load_authority(preregistration, preregistration_sha)
        self.prereg_sha = preregistration_sha
        self.phase = JoystickPhase(1, device, dtype)
        self.records: dict[str, list[np.ndarray]] = {name: [] for name in (
            "student_obs_570", "phase_features_8", "teacher_policy_raw_16",
            "stage", "episode_id", "step", "split",
        )}
        self.pending_raw: torch.Tensor | None = None
        self.pending: dict[str, torch.Tensor] | None = None
        self.max_entry_error = 0.0
        self.step = 0

    def prepare(self, policy_obs: object, vx: torch.Tensor, teacher_raw: torch.Tensor) -> torch.Tensor:
        obs = _extract_policy_observation(policy_obs, 1).detach()
        raw = teacher_raw.detach().reshape(1, 16)
        if tuple(obs.shape) != (1, 570) or not bool(torch.all(torch.isfinite(raw)).item()):
            raise RuntimeError("raw passthrough input contract changed")
        phase, stage = self.phase.features(vx)
        self.pending_raw = raw.clone()
        self.pending = {"student_obs_570": obs.clone(), "phase_features_8": phase, "stage": stage}
        return teacher_raw

    def record_after_step(self, action_term: object) -> None:
        if self.pending_raw is None or self.pending is None:
            raise RuntimeError("raw passthrough record without pre-step sample")
        live_raw = getattr(action_term, "raw_actions", None)
        if not isinstance(live_raw, torch.Tensor):
            live_raw = getattr(action_term, "_raw_actions", None)
        if not isinstance(live_raw, torch.Tensor):
            raise RuntimeError("production action term exposes no raw action")
        error = float(torch.max(torch.abs(live_raw - self.pending_raw)).item())
        self.max_entry_error = max(self.max_entry_error, error)
        if error != 0.0:
            raise RuntimeError(f"raw Student entrypoint changed Teacher action: max_abs={error}")
        values = {
            **self.pending,
            "teacher_policy_raw_16": self.pending_raw,
            "episode_id": torch.zeros(1, device=self.pending_raw.device, dtype=torch.int64),
            "step": torch.tensor([self.step], device=self.pending_raw.device, dtype=torch.int64),
            "split": torch.zeros(1, device=self.pending_raw.device, dtype=torch.int64),
        }
        for name, value in values.items():
            self.records[name].append(value.detach().cpu().numpy())
        self.pending_raw = None
        self.pending = None
        self.step += 1

    def finalize(self, behavior: dict[str, Any]) -> dict[str, Any]:
        arrays = {name: np.concatenate(parts, axis=0) for name, parts in self.records.items()}
        if arrays["student_obs_570"].shape != (138, 570):
            raise RuntimeError(f"raw smoke episode shape changed: {arrays['student_obs_570'].shape}")
        dataset = self.output / "canonical_raw_bc_dataset.npz"
        _atomic_npz(dataset, arrays)
        passed = bool(
            self.max_entry_error == 0.0
            and behavior.get("full_climb")
            and behavior.get("rear_hold")
        )
        result = {
            "schema_version": 1,
            "kind": "highstep_fixed_motion_raw_passthrough_smoke",
            "status": "passed" if passed else "failed_entrypoint_smoke",
            "workflow_id": WORKFLOW_ID,
            "passed": passed,
            "raw_entry_max_abs_error": self.max_entry_error,
            "normal_production_action_term_used": True,
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "unique_episode_count": 1,
            "sample_count": 138,
            "behavior": behavior,
            "preregistration_path": str(self.prereg_path),
            "preregistration_sha256": self.prereg_sha,
        }
        _atomic_json(self.output / "smoke_manifest.json", result)
        return result


class RawActionStudentController:
    def __init__(self, *, training_manifest_path: str | Path, training_manifest_sha256: str,
                 wandb_verification_path: str | Path, wandb_verification_sha256: str,
                 preregistration_path: str | Path, preregistration_sha256: str,
                 environment_checkpoint_path: str | Path, num_envs: int,
                 device: torch.device, dtype: torch.dtype, phase_only_action_term: object | None = None,
                 one_shot: bool = False) -> None:
        _, prereg = load_authority(preregistration_path, preregistration_sha256)
        if sha256_file(environment_checkpoint_path) != prereg["teacher_sha256"]:
            raise RuntimeError("raw Student environment checkpoint changed")
        manifest_path = Path(training_manifest_path).expanduser().resolve()
        if sha256_file(manifest_path) != training_manifest_sha256:
            raise RuntimeError("raw Student training manifest SHA mismatch")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("workflow_id") != WORKFLOW_ID or manifest.get("status") != "completed":
            raise RuntimeError("raw Student training manifest contract changed")
        verification_path = Path(wandb_verification_path).expanduser().resolve()
        if sha256_file(verification_path) != wandb_verification_sha256:
            raise RuntimeError("raw Student W&B verification SHA mismatch")
        verification = json.loads(verification_path.read_text())
        if verification.get("remote_verified") is not True or verification.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
            raise RuntimeError("raw Student W&B verification incomplete")
        script = Path(manifest["torchscript_path"]).expanduser().resolve()
        if sha256_file(script) != manifest["torchscript_sha256"]:
            raise RuntimeError("raw Student TorchScript SHA mismatch")
        self.model = torch.jit.load(str(script), map_location=device).eval()
        self.phase = JoystickPhase(num_envs, device, dtype)
        self.one_shot_phase = FixedSequenceClock(num_envs, device, dtype) if one_shot else None
        self.num_envs = num_envs
        self.device, self.dtype = device, dtype
        self.illegal_outputs = 0
        self.last_sample: dict[str, torch.Tensor] | None = None
        self.phase_only_action_term = phase_only_action_term
        if self.phase_only_action_term is not None and not hasattr(
            self.phase_only_action_term, "set_phase_only_deployment_phase"
        ):
            raise RuntimeError("production action term lacks the phase-only deployment prior entrypoint")

    def reset(self) -> None:
        self.phase.reset()
        if self.one_shot_phase is not None:
            self.one_shot_phase.reset()

    def one_shot_command(self, step: int) -> float:
        if self.one_shot_phase is None:
            raise RuntimeError("one-shot command requested outside one-shot deployment")
        return self.one_shot_phase.command_for_step(step)

    def action(self, policy_obs: object, vx: torch.Tensor) -> torch.Tensor:
        obs = _extract_policy_observation(policy_obs, self.num_envs).to(device=self.device, dtype=self.dtype)
        if self.one_shot_phase is None:
            phase, stage = self.phase.features(vx)
        else:
            expected = self.one_shot_phase.command_for_step(self.one_shot_phase.step)
            if not bool(torch.all(torch.abs(vx.reshape(-1) - expected) <= 1.0e-7).item()):
                raise RuntimeError("one-shot model command and internal phase clock diverged")
            phase, stage = self.one_shot_phase.features()
        if self.phase_only_action_term is not None:
            self.phase_only_action_term.set_phase_only_deployment_phase(phase[:, 0])
        raw = self.model(torch.cat((obs, phase), dim=1))
        illegal = ~torch.isfinite(raw) | (torch.abs(raw) > 60.0)
        self.illegal_outputs += int(torch.sum(illegal).item())
        if bool(torch.any(illegal).item()):
            raise RuntimeError("raw Student emitted NaN or exceeded the frozen raw clip")
        self.last_sample = {"student_obs_570": obs.detach(), "phase_features_8": phase.detach(), "candidate_raw_16": raw.detach(), "stage": stage.detach()}
        return raw


class RawActionDaggerCollector:
    """Label Student-driven pre-step states with the frozen Teacher raw action."""

    def __init__(self, output_dir: str | Path, preregistration: str | Path,
                 preregistration_sha: str, round_index: int) -> None:
        self.output = Path(output_dir).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.prereg_path, _ = load_authority(preregistration, preregistration_sha)
        self.prereg_sha = preregistration_sha
        self.round = int(round_index)
        if self.round not in (1, 2, 3):
            raise ValueError("raw DAgger round must be 1..3")
        self.records: dict[str, list[np.ndarray]] = {name: [] for name in (
            "student_obs_570", "phase_features_8", "teacher_policy_raw_16",
            "stage", "episode_id", "step", "split",
        )}
        self.pending: dict[str, torch.Tensor] | None = None
        self.step = 0

    def prepare(self, student_sample: dict[str, torch.Tensor], teacher_raw: torch.Tensor) -> None:
        self.pending = {
            "student_obs_570": student_sample["student_obs_570"],
            "phase_features_8": student_sample["phase_features_8"],
            "teacher_policy_raw_16": teacher_raw.detach().reshape(1, 16),
            "stage": student_sample["stage"],
            "episode_id": torch.full((1,), self.round, device=teacher_raw.device, dtype=torch.int64),
            "step": torch.tensor([self.step], device=teacher_raw.device, dtype=torch.int64),
            "split": torch.zeros(1, device=teacher_raw.device, dtype=torch.int64),
        }

    def record_after_step(self) -> None:
        if self.pending is None:
            raise RuntimeError("raw DAgger record without pre-step label")
        for name, value in self.pending.items():
            self.records[name].append(value.detach().cpu().numpy())
        self.pending = None
        self.step += 1

    def finalize(self, behavior: dict[str, Any]) -> dict[str, Any]:
        arrays = {name: np.concatenate(parts, axis=0) for name, parts in self.records.items()}
        if arrays["student_obs_570"].shape != (138, 570):
            raise RuntimeError("raw DAgger episode is incomplete")
        dataset = self.output / f"dagger_round{self.round}_dataset.npz"
        _atomic_npz(dataset, arrays)
        manifest = {
            "schema_version": 1,
            "kind": "highstep_fixed_motion_raw_action_dagger_dataset",
            "status": "completed",
            "workflow_id": WORKFLOW_ID,
            "round": self.round,
            "student_drives_physics": True,
            "teacher_labels_same_pre_step_state_only": True,
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "sample_count": 138,
            "behavior": behavior,
            "preregistration_path": str(self.prereg_path),
            "preregistration_sha256": self.prereg_sha,
        }
        _atomic_json(self.output / "dagger_manifest.json", manifest)
        return manifest
