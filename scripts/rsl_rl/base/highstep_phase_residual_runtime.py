"""Fail-closed runtime helpers for the fixed-condition phase-residual branch.

This module deliberately has no dependency on Isaac Lab.  ``play.py`` passes
the live joint action term to :class:`PhaseResidualReferenceController`, which
keeps the reference parser and affine inverse mapping unit-testable without
launching the simulator.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Sequence

import torch


EXPECTED_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "FL_box_joint",
    "FR_box_joint",
    "RL_box_joint",
    "RR_box_joint",
)

PHYSICAL_LOWER = (
    -1.2217304764,
    -1.2217304764,
    -1.2217304764,
    -1.2217304764,
    -1.5708,
    -1.5708,
    -1.5708,
    -1.5708,
    -2.7750735107,
    -2.7750735107,
    -2.7750735107,
    -2.7750735107,
    0.0,
    0.0,
    0.0,
    0.0,
)

PHYSICAL_UPPER = (
    1.2217304764,
    1.2217304764,
    1.2217304764,
    1.2217304764,
    3.4907,
    3.4907,
    3.4907,
    3.4907,
    -0.6457718232,
    -0.6457718232,
    -0.6457718232,
    -0.6457718232,
    0.06,
    0.06,
    0.06,
    0.06,
)

FULL_STATE_COLUMNS = (
    "source_root_pos.x",
    "source_root_pos.y",
    "source_root_pos.z",
    "source_root_quat.w",
    "source_root_quat.x",
    "source_root_quat.y",
    "source_root_quat.z",
    "source_root_lin_vel_b.x",
    "source_root_lin_vel_b.y",
    "source_root_lin_vel_b.z",
    "source_root_ang_vel_b.x",
    "source_root_ang_vel_b.y",
    "source_root_ang_vel_b.z",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_env_row(value: object, *, expected: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    tensor = value.detach()
    if tensor.ndim == 0:
        tensor = tensor.repeat(expected).reshape(1, expected)
    elif tensor.ndim == 1:
        if tensor.numel() == 1:
            tensor = tensor.repeat(expected)
        tensor = tensor.reshape(1, -1)
    else:
        tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.shape[-1] != expected:
        raise RuntimeError(f"{name} exposes {tensor.shape[-1]} values; expected {expected}")
    return tensor


class PhaseResidualReference:
    """Hash-bound, physically bounded 16-joint reference loaded from CSV."""

    def __init__(self, path: str | Path, *, expected_sha256: str) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"phase-residual reference is missing: {self.path}")
        actual_sha256 = sha256_file(self.path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "phase-residual reference SHA mismatch: "
                f"expected={expected_sha256} actual={actual_sha256} path={self.path}"
            )
        self.sha256 = actual_sha256

        target_columns = [f"reference_target.{name}" for name in EXPECTED_JOINT_ORDER]
        measured_columns = [f"measured_joint_pos.{name}" for name in EXPECTED_JOINT_ORDER]
        measured_velocity_columns = [
            f"measured_joint_vel.{name}" for name in EXPECTED_JOINT_ORDER
        ]
        rows: list[list[float]] = []
        measured_rows: list[list[float]] = []
        initial_full_state: dict[str, float] | None = None
        initial_joint_velocity: list[float] | None = None
        phases: list[float] = []
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = [
                name
                for name in ("reference_step", "global_phase", *target_columns, *measured_columns)
                if name not in fields
            ]
            if missing:
                raise RuntimeError(f"reference CSV is missing required columns: {missing}")
            full_state_present = [name in fields for name in FULL_STATE_COLUMNS]
            measured_velocity_present = [name in fields for name in measured_velocity_columns]
            if any(full_state_present) and not all(full_state_present):
                missing_state = [
                    name for name, present in zip(FULL_STATE_COLUMNS, full_state_present) if not present
                ]
                raise RuntimeError(f"reference CSV has a partial full-state contract: {missing_state}")
            if any(measured_velocity_present) and not all(measured_velocity_present):
                missing_velocity = [
                    name
                    for name, present in zip(measured_velocity_columns, measured_velocity_present)
                    if not present
                ]
                raise RuntimeError(
                    f"reference CSV has a partial joint-velocity contract: {missing_velocity}"
                )
            has_full_state = all(full_state_present) and all(measured_velocity_present)
            for expected_step, row in enumerate(reader):
                step = int(row["reference_step"])
                if step != expected_step:
                    raise RuntimeError(
                        f"reference_step must be contiguous from zero; row={expected_step} value={step}"
                    )
                phase = float(row["global_phase"])
                values = [float(row[column]) for column in target_columns]
                measured = [float(row[column]) for column in measured_columns]
                if (
                    not math.isfinite(phase)
                    or any(not math.isfinite(value) for value in values)
                    or any(not math.isfinite(value) for value in measured)
                ):
                    raise RuntimeError(f"reference contains non-finite data at step {step}")
                phases.append(phase)
                rows.append(values)
                measured_rows.append(measured)
                if expected_step == 0 and has_full_state:
                    initial_full_state = {name: float(row[name]) for name in FULL_STATE_COLUMNS}
                    initial_joint_velocity = [
                        float(row[name]) for name in measured_velocity_columns
                    ]

        if len(rows) < 2:
            raise RuntimeError("phase-residual reference requires at least two frames")
        if any(phases[index] > phases[index + 1] for index in range(len(phases) - 1)):
            raise RuntimeError("global_phase must be monotonic")
        if abs(phases[0]) > 1.0e-9 or abs(phases[-1] - 1.0) > 1.0e-9:
            raise RuntimeError("global_phase must span exactly 0.0 to 1.0")

        targets = torch.tensor(rows, dtype=torch.float64)
        lower = torch.tensor(PHYSICAL_LOWER, dtype=targets.dtype)
        upper = torch.tensor(PHYSICAL_UPPER, dtype=targets.dtype)
        violations = (targets < lower - 1.0e-9) | (targets > upper + 1.0e-9)
        if bool(torch.any(violations).item()):
            count = int(torch.sum(violations).item())
            raise RuntimeError(f"reference contains {count} physical-envelope violations")
        self.targets_cpu = targets
        measured_targets = torch.tensor(measured_rows, dtype=torch.float64)
        initial_measured = measured_targets[0]
        projected_initial = torch.clamp(initial_measured, min=lower, max=upper)
        initial_projection = torch.abs(projected_initial - initial_measured)
        self.initial_measured_projection_count = int(torch.sum(initial_projection > 0.0).item())
        self.initial_measured_projection_max_abs = float(torch.max(initial_projection).item())
        self.initial_measured_joint_pos_cpu = projected_initial.clone()
        self.global_phases = tuple(phases)
        self.has_initial_full_state = initial_full_state is not None
        self.initial_root_pose_cpu: torch.Tensor | None = None
        self.initial_root_velocity_body_cpu: torch.Tensor | None = None
        self.initial_root_velocity_world_cpu: torch.Tensor | None = None
        self.initial_measured_joint_vel_cpu: torch.Tensor | None = None
        if initial_full_state is not None:
            assert initial_joint_velocity is not None
            state_values = [initial_full_state[name] for name in FULL_STATE_COLUMNS]
            if any(not math.isfinite(value) for value in (*state_values, *initial_joint_velocity)):
                raise RuntimeError("reference initial full state contains non-finite data")
            root_position = torch.tensor(state_values[0:3], dtype=torch.float64)
            root_quaternion = torch.tensor(state_values[3:7], dtype=torch.float64)
            quaternion_norm = float(torch.linalg.vector_norm(root_quaternion).item())
            if quaternion_norm <= 1.0e-9:
                raise RuntimeError("reference initial root quaternion has zero norm")
            root_quaternion = root_quaternion / quaternion_norm
            root_velocity_body = torch.tensor(state_values[7:13], dtype=torch.float64)
            linear_world = self._quat_apply(root_quaternion, root_velocity_body[0:3])
            angular_world = self._quat_apply(root_quaternion, root_velocity_body[3:6])
            self.initial_root_pose_cpu = torch.cat((root_position, root_quaternion))
            self.initial_root_velocity_body_cpu = root_velocity_body
            self.initial_root_velocity_world_cpu = torch.cat((linear_world, angular_world))
            self.initial_measured_joint_vel_cpu = torch.tensor(
                initial_joint_velocity, dtype=torch.float64
            )

    @staticmethod
    def _quat_apply(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Rotate one 3-vector by one normalized wxyz quaternion."""
        scalar = quaternion_wxyz[0]
        imaginary = quaternion_wxyz[1:4]
        twice_cross = 2.0 * torch.linalg.cross(imaginary, vector)
        return vector + scalar * twice_cross + torch.linalg.cross(imaginary, twice_cross)

    @property
    def frame_count(self) -> int:
        return int(self.targets_cpu.shape[0])


class PhaseResidualReferenceController:
    """Replay a safe joint target through a plain affine joint-position term."""

    def __init__(
        self,
        reference: PhaseResidualReference,
        *,
        action_term: object,
        initial_joint_pos: torch.Tensor,
        joint_names: Sequence[str],
        blend_steps: int = 25,
        start_at_next_frame: bool = False,
    ) -> None:
        if tuple(str(name) for name in joint_names) != EXPECTED_JOINT_ORDER:
            raise RuntimeError(
                "phase-residual action order mismatch: "
                f"expected={EXPECTED_JOINT_ORDER} actual={tuple(joint_names)}"
            )
        supported_action_terms = {
            "JointPositionAction",
            "PhasedHighstepBoxBiasJointPositionAction",
        }
        if type(action_term).__name__ not in supported_action_terms:
            raise RuntimeError(
                "phase-residual replay requires the exact plain or production phased "
                f"JointPositionAction; got {type(action_term).__name__}"
            )
        self.has_phased_prior = type(action_term).__name__ == "PhasedHighstepBoxBiasJointPositionAction"
        if self.has_phased_prior and getattr(action_term, "_action_delay_buffer", None) is not None:
            raise RuntimeError(
                "phase-residual prior cancellation requires zero action delay; a delay buffer is active"
            )
        self.reference = reference
        self.action_term = action_term
        self.blend_steps = int(blend_steps)
        self.start_at_next_frame = bool(start_at_next_frame)
        if self.blend_steps < 0:
            raise ValueError("phase-residual blend_steps must be non-negative")
        if self.start_at_next_frame and self.blend_steps != 0:
            raise ValueError("start_at_next_frame requires zero blend steps")

        self.scale = _as_env_row(
            getattr(action_term, "_scale", None), expected=16, name="action-term scale"
        )
        self.offset = _as_env_row(
            getattr(action_term, "_offset", None), expected=16, name="action-term offset"
        )
        if bool(torch.any(torch.abs(self.scale) <= 1.0e-12).item()):
            raise RuntimeError("phase-residual replay cannot invert a zero action scale")
        initial = _as_env_row(initial_joint_pos, expected=16, name="initial joint position")
        self.device = initial.device
        self.dtype = initial.dtype
        self.scale = self.scale.to(device=self.device, dtype=self.dtype)
        self.offset = self.offset.to(device=self.device, dtype=self.dtype)
        if self.scale.shape[0] not in (1, initial.shape[0]):
            raise RuntimeError("action-term scale batch dimension is incompatible with the environment")
        if self.offset.shape[0] not in (1, initial.shape[0]):
            raise RuntimeError("action-term offset batch dimension is incompatible with the environment")
        self.initial_joint_pos = initial.clone()
        self.reference_targets = reference.targets_cpu.to(device=self.device, dtype=self.dtype)
        self.initial_reference_state = reference.initial_measured_joint_pos_cpu.to(
            device=self.device, dtype=self.dtype
        ).reshape(1, -1)
        self.last_target = initial.clone()
        self.last_reference_index = -1
        self.last_global_phase = 0.0
        self.last_prior_probe_iterations = 0
        self.last_prior_probe_max_abs_error = 0.0

    @staticmethod
    def _quintic(value: float) -> float:
        value = min(max(float(value), 0.0), 1.0)
        return value**3 * (10.0 + value * (-15.0 + 6.0 * value))

    def target_for_step(self, control_step: int) -> torch.Tensor:
        step = int(control_step)
        if step < self.blend_steps:
            alpha = self._quintic((step + 1) / self.blend_steps)
            first_state = self.initial_reference_state
            if self.initial_joint_pos.shape[0] > 1:
                first_state = first_state.expand(self.initial_joint_pos.shape[0], -1)
            target = self.initial_joint_pos + alpha * (first_state - self.initial_joint_pos)
            self.last_reference_index = 0
            self.last_global_phase = 0.0
        else:
            index = step - self.blend_steps + int(self.start_at_next_frame)
            index = min(index, self.reference.frame_count - 1)
            target = self.reference_targets[index].reshape(1, -1)
            if self.initial_joint_pos.shape[0] > 1:
                target = target.expand(self.initial_joint_pos.shape[0], -1)
            self.last_reference_index = int(index)
            self.last_global_phase = float(self.reference.global_phases[index])
        lower = torch.tensor(PHYSICAL_LOWER, device=self.device, dtype=self.dtype)
        upper = torch.tensor(PHYSICAL_UPPER, device=self.device, dtype=self.dtype)
        target = torch.clamp(target, min=lower, max=upper)
        self.last_target = target.clone()
        return target

    def action_for_step(self, control_step: int) -> torch.Tensor:
        target = self.target_for_step(control_step)
        return self.action_for_target(target)

    def action_for_target(
        self,
        target: torch.Tensor,
        *,
        reference_index: int | None = None,
        global_phase: float | None = None,
    ) -> torch.Tensor:
        """Map an externally composed safe target through the exact live action term."""
        target = _as_env_row(target, expected=16, name="phase-residual composed target").to(
            device=self.device, dtype=self.dtype
        )
        if target.shape[0] == 1 and self.initial_joint_pos.shape[0] > 1:
            target = target.expand(self.initial_joint_pos.shape[0], -1)
        if target.shape[0] != self.initial_joint_pos.shape[0]:
            raise RuntimeError("phase-residual target batch dimension changed")
        lower = torch.tensor(PHYSICAL_LOWER, device=self.device, dtype=self.dtype)
        upper = torch.tensor(PHYSICAL_UPPER, device=self.device, dtype=self.dtype)
        if bool(torch.any((target < lower - 1.0e-7) | (target > upper + 1.0e-7)).item()):
            raise RuntimeError("phase-residual composed target exceeds the physical envelope")
        self.last_target = target.clone()
        if reference_index is not None:
            self.last_reference_index = int(reference_index)
        if global_phase is not None:
            self.last_global_phase = float(global_phase)
        action = (target - self.offset) / self.scale
        if self.has_phased_prior:
            # The production prior is an additive transform on the four box
            # targets and depends on the current physical state, not on the raw
            # policy action.  Evaluate that exact live transform once without
            # stepping physics, then analytically cancel the observed delta.
            # ``env.step`` processes the corrected action again in the same
            # state; the post-step audit below must still match at 1e-6.
            # First run exposes the exact unclipped additive box bias computed
            # by the production term.  Reading the bias is essential near the
            # 0/0.06 m box limits: an output-only fixed-point iteration cannot
            # see how much bias was hidden by the physical clamp.
            self.action_term.process_actions(action)
            box_ids = getattr(self.action_term, "_box_action_ids", None)
            box_bias = getattr(self.action_term, "_last_phased_highstep_box_bias", None)
            if not isinstance(box_ids, torch.Tensor) or not isinstance(box_bias, torch.Tensor):
                raise RuntimeError("production phased action term does not expose its exact box prior bias")
            box_ids = box_ids.to(device=self.device, dtype=torch.long).flatten()
            box_bias = box_bias.to(device=self.device, dtype=self.dtype)
            if box_ids.numel() != 4 or box_bias.shape[-1] != 4:
                raise RuntimeError("production phased action term exposes an invalid four-box prior contract")
            action[:, box_ids] = (
                target[:, box_ids] - self.offset[:, box_ids] - box_bias
            ) / self.scale[:, box_ids]

            for iteration in range(1, 5):
                self.action_term.process_actions(action)
                processed = _as_env_row(
                    getattr(self.action_term, "processed_actions", None),
                    expected=16,
                    name="prior probe processed target",
                ).to(device=self.device, dtype=self.dtype)
                error = target - processed
                max_abs = float(torch.max(torch.abs(error)).detach().cpu())
                self.last_prior_probe_iterations = iteration
                self.last_prior_probe_max_abs_error = max_abs
                if max_abs <= 2.5e-7:
                    break
                action = action + error / self.scale
        return action

    def audit_processed_target(self) -> dict[str, float | int | bool]:
        processed = _as_env_row(
            getattr(self.action_term, "processed_actions", None),
            expected=16,
            name="processed action target",
        ).to(device=self.device, dtype=self.dtype)
        delta = torch.abs(processed - self.last_target)
        max_abs = float(torch.max(delta).detach().cpu())
        flat_index = int(torch.argmax(delta).detach().cpu())
        joint_index = flat_index % 16
        return {
            "reference_index": self.last_reference_index,
            "global_phase": self.last_global_phase,
            "processed_target_max_abs_error": max_abs,
            "processed_target_max_error_joint": EXPECTED_JOINT_ORDER[joint_index],
            "processed_target_abs_error_by_joint": [
                float(value) for value in delta[0].detach().cpu().tolist()
            ],
            "processed_target_exact": max_abs <= 1.0e-6,
            "prior_probe_iterations": self.last_prior_probe_iterations,
            "prior_probe_max_abs_error": self.last_prior_probe_max_abs_error,
        }
