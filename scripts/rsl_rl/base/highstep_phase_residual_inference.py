"""Hash-bound inference controller for the fixed-condition phase-residual branch."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from highstep_phase_residual_dataset import (
    ACTION_HISTORY_LENGTH,
    ACTION_HISTORY_OFFSET,
    OBSERVATION_DIM,
    PhaseFeatureContract,
)
from highstep_phase_residual_runtime import (
    EXPECTED_JOINT_ORDER,
    PHYSICAL_LOWER,
    PHYSICAL_UPPER,
    PhaseResidualReference,
    PhaseResidualReferenceController,
    sha256_file,
)


WORKFLOW_ID = "highstep_fixed_condition_phase_residual_20260717"


def _load_bound_json(path: str | Path, expected_sha256: str, *, name: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is missing: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{name} SHA mismatch: expected={expected_sha256} actual={actual}")
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _extract_policy_observation(policy_obs: object, num_envs: int) -> torch.Tensor:
    if not isinstance(policy_obs, torch.Tensor):
        keys = policy_obs.keys() if hasattr(policy_obs, "keys") else ()
        if "policy" in keys:
            policy_obs = policy_obs["policy"]
        elif hasattr(policy_obs, "policy"):
            policy_obs = policy_obs.policy
    if not isinstance(policy_obs, torch.Tensor) or tuple(policy_obs.shape) != (
        num_envs,
        OBSERVATION_DIM,
    ):
        raise RuntimeError(
            "phase-residual policy observation shape changed: "
            f"{getattr(policy_obs, 'shape', None)}"
        )
    return policy_obs


class PhaseResidualPreroll:
    """Frozen, physically bounded zero-command target sequence before active phase zero."""

    def __init__(
        self,
        manifest_path: str | Path,
        expected_manifest_sha256: str,
        *,
        expected_reference_sha256: str,
        expected_timing_sha256: str,
    ) -> None:
        self.manifest_path, manifest = _load_bound_json(
            manifest_path,
            expected_manifest_sha256,
            name="phase-residual pre-roll manifest",
        )
        if (
            manifest.get("kind") != "highstep_fixed_condition_teacher_zero_command_preroll"
            or manifest.get("status") != "frozen_preroll_ready_before_behavior_retry"
            or manifest.get("workflow_id") != WORKFLOW_ID
            or manifest.get("selected_reference_sha256") != expected_reference_sha256
            or manifest.get("source_timing_amendment_sha256") != expected_timing_sha256
            or manifest.get("command") != [0.0, 0.0, 0.0]
            or int(manifest.get("frame_count", -1)) != 296
            or int(manifest.get("projection", {}).get("exported_target_limit_violations", -1)) != 0
        ):
            raise RuntimeError("phase-residual pre-roll manifest contract changed")
        self.manifest_sha256 = expected_manifest_sha256
        self.path = Path(manifest["preroll_path"]).expanduser().resolve()
        if sha256_file(self.path) != manifest.get("preroll_sha256"):
            raise RuntimeError("phase-residual pre-roll CSV SHA mismatch")
        columns = [f"reference_target.{name}" for name in EXPECTED_JOINT_ORDER]
        rows: list[list[float]] = []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in ("preroll_step", *columns) if name not in (reader.fieldnames or [])]
            if missing:
                raise RuntimeError(f"phase-residual pre-roll CSV is missing columns: {missing}")
            for expected_step, row in enumerate(reader):
                if int(row["preroll_step"]) != expected_step:
                    raise RuntimeError("phase-residual pre-roll steps are not contiguous")
                values = [float(row[column]) for column in columns]
                if any(not math.isfinite(value) for value in values):
                    raise RuntimeError("phase-residual pre-roll contains non-finite targets")
                rows.append(values)
        if len(rows) != 296:
            raise RuntimeError("phase-residual pre-roll CSV frame count changed")
        targets = torch.tensor(rows, dtype=torch.float64)
        lower = torch.tensor(PHYSICAL_LOWER, dtype=targets.dtype)
        upper = torch.tensor(PHYSICAL_UPPER, dtype=targets.dtype)
        if bool(torch.any((targets < lower - 1.0e-9) | (targets > upper + 1.0e-9)).item()):
            raise RuntimeError("phase-residual pre-roll contains physical-envelope violations")
        self.targets_cpu = targets
        self.sha256 = manifest["preroll_sha256"]

    @property
    def frame_count(self) -> int:
        return int(self.targets_cpu.shape[0])


class PhaseResidualHybridController:
    """Run safe reference + bounded residual from the candidate's own blind state."""

    def __init__(
        self,
        *,
        training_manifest_path: str | Path,
        training_manifest_sha256: str,
        wandb_verification_path: str | Path,
        wandb_verification_sha256: str,
        reference: PhaseResidualReference,
        preroll: PhaseResidualPreroll,
        environment_checkpoint_path: str | Path,
        action_term: object,
        initial_joint_pos: torch.Tensor,
        joint_names: Sequence[str],
        residual_enabled: bool,
        settle_steps: int = 296,
        pre_active_steps: int = 10,
    ) -> None:
        if tuple(str(name) for name in joint_names) != EXPECTED_JOINT_ORDER:
            raise RuntimeError("phase-residual inference joint order changed")
        self.training_manifest_path, training = _load_bound_json(
            training_manifest_path,
            training_manifest_sha256,
            name="phase-residual training manifest",
        )
        if (
            training.get("kind") != "highstep_fixed_condition_phase_residual_training"
            or training.get("workflow_id") != WORKFLOW_ID
        ):
            raise RuntimeError("phase-residual training manifest contract changed")
        self.training_manifest_sha256 = training_manifest_sha256
        checkpoint_path = Path(training["checkpoint_path"]).expanduser().resolve()
        if sha256_file(checkpoint_path) != training.get("checkpoint_sha256"):
            raise RuntimeError("phase-residual checkpoint SHA mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("kind") != "bounded_phase_residual_mlp"
            or checkpoint.get("workflow_id") != WORKFLOW_ID
            or checkpoint.get("dataset_manifest_sha256")
            != training.get("dataset_manifest_sha256")
            or checkpoint.get("selected_reference_sha256") != reference.sha256
            or checkpoint.get("architecture")
            != {"input_dim": 578, "hidden_dims": [256, 256], "output_dim": 16}
        ):
            raise RuntimeError("phase-residual checkpoint binding changed")

        script_path = Path(training["torchscript_path"]).expanduser().resolve()
        if sha256_file(script_path) != training.get("torchscript_sha256"):
            raise RuntimeError("phase-residual TorchScript SHA mismatch")
        _, verification = _load_bound_json(
            wandb_verification_path,
            wandb_verification_sha256,
            name="phase-residual W&B verification",
        )
        if (
            verification.get("status") != "verified"
            or verification.get("remote_verified") is not True
            or verification.get("checkpoint_sha256") != training.get("checkpoint_sha256")
        ):
            raise RuntimeError("phase-residual W&B verification is not complete")

        selected_path, selected = _load_bound_json(
            training["selected_reference_manifest_path"],
            training["selected_reference_manifest_sha256"],
            name="selected reference manifest",
        )
        if selected.get("selected_reference", {}).get("sha256") != reference.sha256:
            raise RuntimeError("selected reference manifest no longer binds the replay CSV")
        preregistration = selected.get("preregistration", {})
        _, prereg = _load_bound_json(
            preregistration.get("path", ""),
            preregistration.get("sha256", ""),
            name="phase-residual preregistration",
        )
        teacher_binding = prereg.get("unchanged_contract", {})
        environment_checkpoint = Path(environment_checkpoint_path).expanduser().resolve()
        if (
            str(environment_checkpoint) != str(Path(teacher_binding.get("teacher_checkpoint", "")).resolve())
            or sha256_file(environment_checkpoint) != teacher_binding.get("teacher_sha256")
        ):
            raise RuntimeError("behavior environment is not bound to the frozen Teacher checkpoint")

        self.reference = reference
        self.preroll = preroll
        self.reference_controller = PhaseResidualReferenceController(
            reference,
            action_term=action_term,
            initial_joint_pos=initial_joint_pos,
            joint_names=joint_names,
            blend_steps=0,
        )
        self.phase = PhaseFeatureContract(selected["stage_boundaries_control_step"])
        self.model = torch.jit.load(str(script_path), map_location=initial_joint_pos.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = initial_joint_pos.device
        self.dtype = initial_joint_pos.dtype
        self.num_envs = int(initial_joint_pos.shape[0])
        self.residual_enabled = bool(residual_enabled)
        self.settle_steps = int(settle_steps)
        self.pre_active_steps = int(pre_active_steps)
        if self.settle_steps < 0 or self.pre_active_steps < 0:
            raise ValueError("phase-residual timing must be non-negative")
        if self.settle_steps != self.preroll.frame_count:
            raise RuntimeError("phase-residual settle length does not match the frozen pre-roll")
        self.action_history = torch.zeros(
            self.num_envs,
            ACTION_HISTORY_LENGTH,
            16,
            device=self.device,
            dtype=self.dtype,
        )
        self.lower = torch.tensor(PHYSICAL_LOWER, device=self.device, dtype=self.dtype)
        self.upper = torch.tensor(PHYSICAL_UPPER, device=self.device, dtype=self.dtype)
        self.residual_bound = torch.tensor(
            checkpoint["residual_bound"], device=self.device, dtype=self.dtype
        )
        if tuple(self.residual_bound.shape) != (16,):
            raise RuntimeError("phase-residual checkpoint bound shape changed")
        self.selected_reference_manifest_path = selected_path
        self.environment_checkpoint_path = environment_checkpoint
        self.environment_checkpoint_sha256 = teacher_binding["teacher_sha256"]
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = training["checkpoint_sha256"]
        self.torchscript_path = script_path
        self.torchscript_sha256 = training["torchscript_sha256"]
        self.max_abs_residual = torch.zeros(16, device=self.device, dtype=self.dtype)
        self.preclamp_projection_count = 0
        self.preclamp_projection_max_abs = 0.0
        self.steps = 0
        # Exact tensors consumed/produced by the most recent active-phase
        # inference call.  DAgger reads clones of these tensors only after the
        # candidate action has been composed, so the recorded Student input is
        # byte-for-byte the input that actually drove the environment.
        self.last_student_obs: torch.Tensor | None = None
        self.last_phase_features: torch.Tensor | None = None
        self.last_reference_target: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None
        self.last_composed_target: torch.Tensor | None = None
        self.last_stage: int | None = None
        self.last_active_step: int | None = None

    def command_is_active(self, control_step: int) -> bool:
        return int(control_step) - self.settle_steps >= self.pre_active_steps

    def _append_action_history(self, safe_target: torch.Tensor) -> None:
        scale = self.reference_controller.scale
        offset = self.reference_controller.offset
        safe_action = (safe_target - offset) / scale
        self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
        self.action_history[:, -1, :] = safe_action

    def action_for_step(self, policy_obs: object, control_step: int) -> torch.Tensor:
        step = int(control_step)
        active_step = step - self.settle_steps
        if active_step < 0:
            preroll_index = step
            if preroll_index < 0 or preroll_index >= self.preroll.frame_count:
                raise RuntimeError("phase-residual pre-roll index is outside the frozen sequence")
            reference_index = 0
            global_phase = 0.0
            residual = torch.zeros(
                self.num_envs, 16, device=self.device, dtype=self.dtype
            )
            stage = None
            student_obs = None
            phase = None
        else:
            obs = _extract_policy_observation(policy_obs, self.num_envs).detach().clone()
            obs[:, ACTION_HISTORY_OFFSET:] = self.action_history.reshape(self.num_envs, -1)
            phase_row, stage = self.phase.features(
                active_step, device=self.device, dtype=self.dtype
            )
            phase = phase_row.reshape(1, -1).expand(self.num_envs, -1)
            student_obs = obs.to(device=self.device, dtype=self.dtype)
            model_input = torch.cat((student_obs, phase), dim=1)
            residual = self.model(model_input)
            if not isinstance(residual, torch.Tensor) or tuple(residual.shape) != (
                self.num_envs,
                16,
            ):
                raise RuntimeError("phase-residual TorchScript output shape changed")
            if not self.residual_enabled:
                residual = torch.zeros_like(residual)
            if bool(torch.any(torch.abs(residual) > self.residual_bound + 1.0e-6).item()):
                raise RuntimeError("phase-residual model exceeded its frozen output bound")
            reference_index = min(active_step, self.reference.frame_count - 1)
            global_phase = float(self.reference.global_phases[reference_index])

        if active_step < 0:
            reference_target = self.preroll.targets_cpu[preroll_index]
        else:
            reference_target = self.reference.targets_cpu[reference_index]
        reference_target = reference_target.to(device=self.device, dtype=self.dtype).reshape(
            1, 16
        ).expand(self.num_envs, -1)
        preclamp = reference_target + residual
        target = torch.clamp(preclamp, min=self.lower, max=self.upper)
        projection = torch.abs(target - preclamp)
        self.preclamp_projection_count += int(torch.sum(projection > 0.0).item())
        self.preclamp_projection_max_abs = max(
            self.preclamp_projection_max_abs,
            float(torch.max(projection).detach().cpu()),
        )
        self.max_abs_residual = torch.maximum(
            self.max_abs_residual, torch.amax(torch.abs(residual), dim=0)
        )
        if active_step >= 0:
            assert student_obs is not None and phase is not None and stage is not None
            self.last_student_obs = student_obs.detach().clone()
            self.last_phase_features = phase.detach().clone()
            self.last_reference_target = reference_target.detach().clone()
            self.last_residual = residual.detach().clone()
            self.last_composed_target = target.detach().clone()
            self.last_stage = int(stage)
            self.last_active_step = int(active_step)
        self._append_action_history(target)
        self.steps += 1
        return self.reference_controller.action_for_target(
            target,
            reference_index=reference_index,
            global_phase=global_phase,
        )

    def dagger_sample(self) -> dict[str, torch.Tensor | int]:
        """Return the exact active-step candidate tensors for read-only DAgger."""
        required = (
            self.last_student_obs,
            self.last_phase_features,
            self.last_reference_target,
            self.last_residual,
            self.last_composed_target,
        )
        if any(value is None for value in required):
            raise RuntimeError("DAgger sample requested before the first active candidate step")
        if self.last_stage is None or self.last_active_step is None:
            raise RuntimeError("DAgger phase metadata is unavailable")
        return {
            "student_obs_570": self.last_student_obs.detach().clone(),
            "phase_features_8": self.last_phase_features.detach().clone(),
            "reference_target_16": self.last_reference_target.detach().clone(),
            "candidate_residual_16": self.last_residual.detach().clone(),
            "candidate_target_16": self.last_composed_target.detach().clone(),
            "stage": self.last_stage,
            "step": self.last_active_step,
        }

    def audit_processed_target(self) -> dict[str, float | int | bool]:
        return self.reference_controller.audit_processed_target()

    def summary(self) -> dict[str, Any]:
        return {
            "workflow_id": WORKFLOW_ID,
            "residual_enabled": self.residual_enabled,
            "steps": self.steps,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "torchscript_path": str(self.torchscript_path),
            "torchscript_sha256": self.torchscript_sha256,
            "training_manifest_path": str(self.training_manifest_path),
            "training_manifest_sha256": self.training_manifest_sha256,
            "selected_reference_manifest_path": str(self.selected_reference_manifest_path),
            "selected_reference_sha256": self.reference.sha256,
            "preroll_manifest_path": str(self.preroll.manifest_path),
            "preroll_manifest_sha256": self.preroll.manifest_sha256,
            "preroll_path": str(self.preroll.path),
            "preroll_sha256": self.preroll.sha256,
            "environment_checkpoint_path": str(self.environment_checkpoint_path),
            "environment_checkpoint_sha256": self.environment_checkpoint_sha256,
            "max_abs_residual_by_joint": [
                float(value) for value in self.max_abs_residual.detach().cpu().tolist()
            ],
            "preclamp_projection_count": self.preclamp_projection_count,
            "preclamp_projection_max_abs": self.preclamp_projection_max_abs,
            "final_command_target_limit_violations": 0,
        }
