"""Hash-bound inference for the fixed-condition phase direct-action branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from highstep_phase_residual_dataset import (
    ACTION_HISTORY_LENGTH,
    ACTION_HISTORY_OFFSET,
    PhaseFeatureContract,
)
from highstep_phase_residual_inference import (
    PhaseResidualPreroll,
    _extract_policy_observation,
    _load_bound_json,
)
from highstep_phase_residual_runtime import (
    EXPECTED_JOINT_ORDER,
    PHYSICAL_LOWER,
    PHYSICAL_UPPER,
    PhaseResidualReference,
    PhaseResidualReferenceController,
    sha256_file,
)


WORKFLOW_ID = "highstep_fixed_condition_phase_direct_action_20260717"


class PhaseDirectActionController:
    """Predict a complete safe target from blind Student observation + phase."""

    controller_mode = "phase_direct_action"
    residual_enabled = True

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
        settle_steps: int = 296,
        pre_active_steps: int = 10,
    ) -> None:
        if tuple(str(name) for name in joint_names) != EXPECTED_JOINT_ORDER:
            raise RuntimeError("direct-action joint order changed")
        self.training_manifest_path, training = _load_bound_json(
            training_manifest_path, training_manifest_sha256, name="direct-action training manifest"
        )
        if (
            training.get("kind") != "highstep_fixed_condition_phase_direct_action_training"
            or training.get("workflow_id") != WORKFLOW_ID
        ):
            raise RuntimeError("direct-action training manifest contract changed")
        self.training_manifest_sha256 = training_manifest_sha256
        checkpoint_path = Path(training["checkpoint_path"]).expanduser().resolve()
        if sha256_file(checkpoint_path) != training.get("checkpoint_sha256"):
            raise RuntimeError("direct-action checkpoint SHA mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("kind") != "bounded_phase_direct_action_mlp"
            or checkpoint.get("workflow_id") != WORKFLOW_ID
            or checkpoint.get("architecture")
            != {"input_dim": 578, "hidden_dims": [512, 512, 256], "output_dim": 16}
            or checkpoint.get("dataset_manifest_sha256") != training.get("dataset_manifest_sha256")
            or checkpoint.get("selected_reference_sha256") != reference.sha256
        ):
            raise RuntimeError("direct-action checkpoint binding changed")
        script_path = Path(training["torchscript_path"]).expanduser().resolve()
        if sha256_file(script_path) != training.get("torchscript_sha256"):
            raise RuntimeError("direct-action TorchScript SHA mismatch")
        _, verification = _load_bound_json(
            wandb_verification_path,
            wandb_verification_sha256,
            name="direct-action W&B verification",
        )
        if (
            verification.get("kind") != "highstep_phase_direct_action_wandb_verification"
            or verification.get("status") != "verified"
            or verification.get("remote_verified") is not True
            or verification.get("checkpoint_sha256") != training.get("checkpoint_sha256")
        ):
            raise RuntimeError("direct-action W&B verification is incomplete")
        prereg_path, prereg = _load_bound_json(
            training["preregistration_path"],
            training["preregistration_sha256"],
            name="direct-action preregistration",
        )
        if (
            prereg.get("kind") != "highstep_fixed_condition_phase_direct_action_preregistration"
            or prereg.get("status") != "frozen_before_training"
            or prereg.get("workflow_id") != WORKFLOW_ID
        ):
            raise RuntimeError("direct-action preregistration contract changed")
        spec_path = Path(prereg["spec_path"]).expanduser().resolve()
        if sha256_file(spec_path) != prereg.get("spec_sha256"):
            raise RuntimeError("direct-action spec SHA mismatch")
        environment_checkpoint = Path(environment_checkpoint_path).expanduser().resolve()
        if (
            str(environment_checkpoint) != str(Path(prereg["teacher_checkpoint"]).resolve())
            or sha256_file(environment_checkpoint) != prereg.get("teacher_sha256")
        ):
            raise RuntimeError("direct-action environment checkpoint is not the frozen Teacher")

        self.reference = reference
        self.preroll = preroll
        self.reference_controller = PhaseResidualReferenceController(
            reference,
            action_term=action_term,
            initial_joint_pos=initial_joint_pos,
            joint_names=joint_names,
            blend_steps=0,
        )
        selected_path, selected = _load_bound_json(
            training["selected_reference_manifest_path"],
            training["selected_reference_manifest_sha256"],
            name="direct-action selected-reference manifest",
        )
        if selected.get("selected_reference", {}).get("sha256") != reference.sha256:
            raise RuntimeError("direct-action selected reference binding changed")
        self.phase = PhaseFeatureContract(selected["stage_boundaries_control_step"])
        self.model = torch.jit.load(str(script_path), map_location=initial_joint_pos.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device, self.dtype = initial_joint_pos.device, initial_joint_pos.dtype
        self.num_envs = int(initial_joint_pos.shape[0])
        self.settle_steps, self.pre_active_steps = int(settle_steps), int(pre_active_steps)
        if self.settle_steps != preroll.frame_count:
            raise RuntimeError("direct-action settle length changed")
        self.lower = torch.tensor(PHYSICAL_LOWER, device=self.device, dtype=self.dtype)
        self.upper = torch.tensor(PHYSICAL_UPPER, device=self.device, dtype=self.dtype)
        self.action_history = torch.zeros(
            self.num_envs, ACTION_HISTORY_LENGTH, 16, device=self.device, dtype=self.dtype
        )
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = training["checkpoint_sha256"]
        self.torchscript_path = script_path
        self.torchscript_sha256 = training["torchscript_sha256"]
        self.selected_reference_manifest_path = selected_path
        self.preregistration_path = prereg_path
        self.preregistration_sha256 = training["preregistration_sha256"]
        self.environment_checkpoint_path = environment_checkpoint
        self.environment_checkpoint_sha256 = prereg["teacher_sha256"]
        self.steps = 0
        self.target_limit_violations = 0
        self.last_student_obs: torch.Tensor | None = None
        self.last_phase_features: torch.Tensor | None = None
        self.last_reference_target: torch.Tensor | None = None
        self.last_target: torch.Tensor | None = None
        self.last_stage: int | None = None
        self.last_active_step: int | None = None

    def command_is_active(self, control_step: int) -> bool:
        return int(control_step) - self.settle_steps >= self.pre_active_steps

    def _append_history(self, target: torch.Tensor) -> None:
        raw = (target - self.reference_controller.offset) / self.reference_controller.scale
        self.action_history = torch.roll(self.action_history, shifts=-1, dims=1)
        self.action_history[:, -1, :] = raw

    def action_for_step(self, policy_obs: object, control_step: int) -> torch.Tensor:
        active_step = int(control_step) - self.settle_steps
        if active_step < 0:
            target = self.preroll.targets_cpu[int(control_step)].to(
                device=self.device, dtype=self.dtype
            ).reshape(1, 16).expand(self.num_envs, -1)
            reference_index, global_phase = 0, 0.0
        else:
            obs = _extract_policy_observation(policy_obs, self.num_envs).detach().clone()
            obs[:, ACTION_HISTORY_OFFSET:] = self.action_history.reshape(self.num_envs, -1)
            phase_row, stage = self.phase.features(
                active_step, device=self.device, dtype=self.dtype
            )
            phase = phase_row.reshape(1, -1).expand(self.num_envs, -1)
            student_obs = obs.to(device=self.device, dtype=self.dtype)
            target = self.model(torch.cat((student_obs, phase), dim=1))
            if not isinstance(target, torch.Tensor) or tuple(target.shape) != (self.num_envs, 16):
                raise RuntimeError("direct-action TorchScript output shape changed")
            violations = (target < self.lower - 1.0e-6) | (target > self.upper + 1.0e-6)
            self.target_limit_violations += int(torch.sum(violations).item())
            if bool(torch.any(violations).item()):
                raise RuntimeError("direct-action model exceeded its physical output contract")
            reference_index = min(active_step, self.reference.frame_count - 1)
            global_phase = float(self.reference.global_phases[reference_index])
            reference_target = self.reference.targets_cpu[reference_index].to(
                device=self.device, dtype=self.dtype
            ).reshape(1, 16).expand(self.num_envs, -1)
            self.last_student_obs = student_obs.detach().clone()
            self.last_phase_features = phase.detach().clone()
            self.last_reference_target = reference_target.detach().clone()
            self.last_target = target.detach().clone()
            self.last_stage = int(stage)
            self.last_active_step = int(active_step)
        self._append_history(target)
        self.steps += 1
        return self.reference_controller.action_for_target(
            target, reference_index=reference_index, global_phase=global_phase
        )

    def audit_processed_target(self) -> dict[str, float | int | bool]:
        return self.reference_controller.audit_processed_target()

    def dagger_sample(self) -> dict[str, torch.Tensor | int]:
        if any(value is None for value in (
            self.last_student_obs, self.last_phase_features, self.last_reference_target,
            self.last_target,
        )) or self.last_stage is None or self.last_active_step is None:
            raise RuntimeError("direct-action DAgger sample requested before active inference")
        assert self.last_target is not None and self.last_reference_target is not None
        return {
            "student_obs_570": self.last_student_obs.detach().clone(),
            "phase_features_8": self.last_phase_features.detach().clone(),
            "reference_target_16": self.last_reference_target.detach().clone(),
            "candidate_residual_16": (self.last_target - self.last_reference_target).detach().clone(),
            "candidate_target_16": self.last_target.detach().clone(),
            "stage": self.last_stage,
            "step": self.last_active_step,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "workflow_id": WORKFLOW_ID,
            "controller_mode": self.controller_mode,
            "steps": self.steps,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "torchscript_path": str(self.torchscript_path),
            "torchscript_sha256": self.torchscript_sha256,
            "training_manifest_path": str(self.training_manifest_path),
            "training_manifest_sha256": self.training_manifest_sha256,
            "preregistration_path": str(self.preregistration_path),
            "preregistration_sha256": self.preregistration_sha256,
            "selected_reference_manifest_path": str(self.selected_reference_manifest_path),
            "selected_reference_sha256": self.reference.sha256,
            "preroll_manifest_path": str(self.preroll.manifest_path),
            "preroll_manifest_sha256": self.preroll.manifest_sha256,
            "environment_checkpoint_path": str(self.environment_checkpoint_path),
            "environment_checkpoint_sha256": self.environment_checkpoint_sha256,
            "final_command_target_limit_violations": self.target_limit_violations,
        }

