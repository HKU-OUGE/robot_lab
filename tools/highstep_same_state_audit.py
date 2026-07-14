#!/usr/bin/env python3
"""Read-only, same-state Teacher/Student evidence audit for high-step recovery.

This module deliberately does not launch Isaac Sim and never updates a model.  It
contains the small runtime hook needed by a *single* Student-on-policy environment:

1. load qualified B500 ``model_498`` as the driving Student and separately
   bind protected ``model_900`` as its lineage/behavior anchor;
2. independently load the frozen ``model_172300`` Teacher actor/privileged encoder;
3. evaluate both policies on the exact same observation frame;
4. convert the Teacher output with the live, shared high-step post-prior function;
5. emit frame-level and phase-level evidence, and optionally return one
   preregistered Teacher action-group replacement for a diagnostic rollout.

The simulator launcher remains separate.  A caller creates one Student no-prior
environment, calls :class:`HighstepRuntimeAdapter.sample`, then passes that sample
to :meth:`SameStateAuditSession.evaluate_frame`; only the returned
``action_to_apply`` may be sent to ``env.step``.  This makes it impossible for a
second environment or a Teacher rollout to drift away from the Student state.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
ACTIONS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"
EXPECTED_SPEC_SHA256 = "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
EXPECTED_ACTIONS_SHA256 = "7f1d340cf53382a7fb3b6281a144e1a75546510aa6fe3078a107d3236ebeb54a"
ROOT_STUDENT_CHECKPOINT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_04-41-42_robust_student_distill_20260712_044124/model_900.pt"
)
CURRENT_STUDENT_CHECKPOINT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
    "2026-07-12_23-55-22_student_recovery_v111_B500_20260712_235515/model_498.pt"
)
TEACHER_CHECKPOINT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_ENV_CONFIG = TEACHER_CHECKPOINT.parent / "params/env.yaml"
B500_EVALUATION_MANIFEST = (
    ROOT
    / "tmp/highstep_student_recovery_v111_capped_20260712/evaluations/v111_B500/evaluation_manifest.json"
)
B500_HANDOFF = ROOT / "tmp/highstep_student_recovery_v111_capped_20260712/handoff.json"
ROOT_STUDENT_SHA256 = "9bbd5b597d9c195ecf9afb141152b8f0749dc599a54868674b299107fb40a229"
CURRENT_STUDENT_SHA256 = "f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159"
TEACHER_SHA256 = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"

ROOT_STUDENT_ACTOR_SHA256 = "6042d296cf59b31b87769188f50ec2eacaf743c761bd20be9bb3ef7778cde303"
ROOT_STUDENT_ESTIMATOR_SHA256 = "edee6622d43d9f1211d1a7ac084919ef22dd3c953ced1795fe845d2ed42610fe"
ROOT_STUDENT_PRIVILEGED_ENCODER_SHA256 = "7d6084b771e2004b0d8cee4a4c0af3769773269809323a00115149d4f9f8f57d"
CURRENT_STUDENT_ACTOR_SHA256 = "b4e66d07a0fbb11ebb464c298734153280308cbc2b45af2fbedc2d0fd6e9c2c1"
TEACHER_ACTOR_SHA256 = "6cb9638baf538e529b74ed94db9382f50f9987fb615bcf6b2501746347b01b76"
TEACHER_PRIVILEGED_ENCODER_SHA256 = "7d6084b771e2004b0d8cee4a4c0af3769773269809323a00115149d4f9f8f57d"
TEACHER_ENV_CONFIG_SHA256 = "f8aa66bbc417eb67a7889900e4a1781eb2852cb3d94432a1282963cc0909c009"
B500_EVALUATION_MANIFEST_SHA256 = "e21cfcfe3b9dbb5984015c81e042fb279f3ba4dae7e1ff4e5a8f010a75ba1e73"
B500_HANDOFF_SHA256 = "30904c54bd5b05fc431cc52687e1e52fed97c9bae04cd7f374bed4f67e1b5feb"

JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)
DEFAULT_DOF_POS = (
    0.0, 0.0, 0.0, 0.0,
    0.7, 0.7, 0.7, 0.7,
    -1.3, -1.3, -1.3, -1.3,
    0.03, 0.03, 0.03, 0.03,
)

PHASES = (
    "approach",
    "front_lift",
    "front_top_support",
    "first_rear_top",
    "second_rear_top",
    "rear_hold",
)

ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "front_hips": (0, 1),
    "rear_hips": (2, 3),
    "front_thighs": (4, 5),
    "rear_thighs": (6, 7),
    "front_calves": (8, 9),
    "rear_calves": (10, 11),
    "front_boxes": (12, 13),
    "rear_boxes": (14, 15),
    "rear_revolute": (2, 3, 6, 7, 10, 11),
    "all_boxes": (12, 13, 14, 15),
}

DEFAULT_AUDIT_TRAJECTORIES = (
    {
        "run_id": "seed11_nominal",
        "seed": 11,
        "scenario": "nominal",
        "lateral_offset_m": 0.0,
        "yaw_offset_deg": 0.0,
        "required_phases": PHASES,
        "forbidden_phases": (),
        "expected_full_climb_success": True,
        "expected_rear_hold_success": True,
    },
    {
        "run_id": "seed11_left_offset",
        "seed": 11,
        "scenario": "left_offset",
        "lateral_offset_m": 0.12,
        "yaw_offset_deg": 4.0,
        "required_phases": ("approach", "front_lift"),
        "forbidden_phases": ("front_top_support", "first_rear_top", "second_rear_top", "rear_hold"),
        "expected_full_climb_success": False,
        "expected_rear_hold_success": False,
    },
    {
        "run_id": "seed22_right_offset",
        "seed": 22,
        "scenario": "right_offset",
        "lateral_offset_m": -0.12,
        "yaw_offset_deg": -4.0,
        "required_phases": ("approach", "front_lift", "front_top_support"),
        "forbidden_phases": ("first_rear_top", "second_rear_top", "rear_hold"),
        "expected_full_climb_success": False,
        "expected_rear_hold_success": False,
    },
)


def base_suite_plan_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "highstep_same_state_base3_plan",
        "immutable_after_write": True,
        "student_driver_checkpoint": str(CURRENT_STUDENT_CHECKPOINT.resolve()),
        "student_driver_sha256": CURRENT_STUDENT_SHA256,
        "protected_root_checkpoint": str(ROOT_STUDENT_CHECKPOINT.resolve()),
        "protected_root_sha256": ROOT_STUDENT_SHA256,
        "teacher_checkpoint": str(TEACHER_CHECKPOINT.resolve()),
        "teacher_sha256": TEACHER_SHA256,
        "runtime_contract": {
            "task": "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0",
            "num_envs": 1,
            "terrain_type": "box",
            "terrain_level": 9,
            "fixed_velocity_command": [0.45, 0.0, 0.0],
            "front_step_eval_side": "x-",
            "front_step_eval_edge_gap": 0.55,
            "play_max_steps": 600,
            "eval_action_delay_steps": 0,
            "observation_corruption_enabled": False,
            "play_random_events_enabled": False,
            "real_gain_center": {"hip": [55.0, 1.5], "thigh": [65.0, 1.5], "calf": [80.0, 2.5]},
            "student_action_prior_enabled": False,
            "causal_replacement": None,
        },
        "trajectories": DEFAULT_AUDIT_TRAJECTORIES,
        "training_allowed": False,
    }


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_frozen_authority() -> None:
    checks = (
        ("formal spec", SPEC_PATH, EXPECTED_SPEC_SHA256),
        ("shared action-prior implementation", ACTIONS_PATH, EXPECTED_ACTIONS_SHA256),
    )
    for label, path, expected in checks:
        actual = sha256_file(path.resolve(strict=True))
        if actual != expected:
            raise RuntimeError(f"{label} SHA changed: {actual} != {expected}")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        return data.item() if data.ndim == 0 else data.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    if immutable:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _component_state(model_state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    component = {
        key[len(prefix):]: value.detach().cpu().clone()
        for key, value in model_state.items()
        if key.startswith(prefix)
    }
    if not component:
        raise ValueError(f"checkpoint has no component with prefix {prefix!r}")
    return component


def _component_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Stable hash matching the recovery binding implementation exactly."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_collection_sha256(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_checkpoint(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    real_path = path.expanduser().resolve(strict=True)
    actual_sha256 = sha256_file(real_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA256 mismatch for {real_path}: {actual_sha256} != {expected_sha256}"
        )
    try:
        checkpoint = torch.load(real_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch only
        checkpoint = torch.load(real_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"invalid RSL checkpoint: {real_path}")
    return checkpoint, actual_sha256


def _linear_indices(state: Mapping[str, torch.Tensor]) -> list[int]:
    indices: set[int] = set()
    for key in state:
        pieces = key.split(".")
        if len(pieces) == 2 and pieces[0].isdigit() and pieces[1] in {"weight", "bias"}:
            indices.add(int(pieces[0]))
    result = sorted(indices)
    if not result:
        raise ValueError("component contains no indexed Linear layers")
    for index in result:
        if f"{index}.weight" not in state or f"{index}.bias" not in state:
            raise ValueError(f"incomplete Linear state at index {index}")
    return result


class CheckpointMLP(nn.Module):
    """Exact ELU MLP reconstructed from only checkpoint tensor shapes."""

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        final_tanh: bool = False,
        final_elu: bool = False,
    ):
        super().__init__()
        if final_tanh and final_elu:
            raise ValueError("an MLP cannot request both final_tanh and final_elu")
        linear_indices = _linear_indices(state)
        layers: list[nn.Module] = []
        for position, checkpoint_index in enumerate(linear_indices):
            weight = state[f"{checkpoint_index}.weight"]
            bias = state[f"{checkpoint_index}.bias"]
            if weight.ndim != 2 or bias.shape != (weight.shape[0],):
                raise ValueError(f"invalid Linear tensor shapes at index {checkpoint_index}")
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(bias)
            layers.append(layer)
            if position != len(linear_indices) - 1 or final_elu:
                layers.append(nn.ELU())
        if final_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class StudentEstimator(nn.Module):
    """Inference-only subset of ``ProprioVAE`` used by the deployed Student."""

    def __init__(self, state: Mapping[str, torch.Tensor]):
        super().__init__()
        encoder_state = {
            key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")
        }
        # ProprioVAE.encoder has ELU after every hidden Linear, including its
        # final 128-D layer.  Actor heads intentionally keep a linear output.
        self.encoder = CheckpointMLP(encoder_state, final_elu=True)

        def linear(name: str) -> nn.Linear:
            weight = state[f"{name}.weight"]
            bias = state[f"{name}.bias"]
            module = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                module.weight.copy_(weight)
                module.bias.copy_(bias)
            return module

        self.fc_mu = linear("fc_mu")
        self.fc_logvar = linear("fc_logvar")
        self.vel_estimator = linear("vel_estimator")

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(observations)
        mu = self.fc_mu(hidden)
        logvar = torch.clamp(self.fc_logvar(hidden), min=-4.0, max=4.0)
        velocity = self.vel_estimator(hidden)
        return mu, logvar, velocity


class BoundAuditModels(nn.Module):
    """B500 Student + protected model_900 lineage + independent frozen Teacher."""

    def __init__(
        self,
        *,
        student_checkpoint: Path = CURRENT_STUDENT_CHECKPOINT,
        student_sha256: str = CURRENT_STUDENT_SHA256,
        root_student_checkpoint: Path = ROOT_STUDENT_CHECKPOINT,
        root_student_sha256: str = ROOT_STUDENT_SHA256,
        teacher_checkpoint: Path = TEACHER_CHECKPOINT,
        teacher_sha256: str = TEACHER_SHA256,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        assert_frozen_authority()
        student_checkpoint = Path(student_checkpoint).expanduser().resolve(strict=True)
        root_student_checkpoint = Path(root_student_checkpoint).expanduser().resolve(strict=True)
        teacher_checkpoint = Path(teacher_checkpoint).expanduser().resolve(strict=True)
        if len({student_checkpoint, root_student_checkpoint, teacher_checkpoint}) != 3:
            raise ValueError("current Student, protected root Student, and Teacher paths must be distinct")
        student, student_actual_sha = _load_checkpoint(student_checkpoint, student_sha256)
        root_student, root_student_actual_sha = _load_checkpoint(
            root_student_checkpoint, root_student_sha256
        )
        teacher, teacher_actual_sha = _load_checkpoint(teacher_checkpoint, teacher_sha256)

        student_state = student["model_state_dict"]
        root_student_state = root_student["model_state_dict"]
        teacher_state = teacher["model_state_dict"]
        student_actor_state = _component_state(student_state, "actor.")
        student_estimator_state = _component_state(student_state, "estimator.")
        student_priv_state = _component_state(student_state, "priv_encoder.")
        root_actor_state = _component_state(root_student_state, "actor.")
        root_estimator_state = _component_state(root_student_state, "estimator.")
        root_priv_state = _component_state(root_student_state, "priv_encoder.")
        teacher_actor_state = _component_state(teacher_state, "actor.")
        teacher_priv_state = _component_state(teacher_state, "priv_encoder.")

        expected_hashes = {
            "root Student actor": (_component_sha256(root_actor_state), ROOT_STUDENT_ACTOR_SHA256),
            "root Student estimator": (
                _component_sha256(root_estimator_state),
                ROOT_STUDENT_ESTIMATOR_SHA256,
            ),
            "root Student privileged encoder": (
                _component_sha256(root_priv_state),
                ROOT_STUDENT_PRIVILEGED_ENCODER_SHA256,
            ),
            "B500 Student actor": (
                _component_sha256(student_actor_state),
                CURRENT_STUDENT_ACTOR_SHA256,
            ),
            "B500 Student estimator": (
                _component_sha256(student_estimator_state),
                ROOT_STUDENT_ESTIMATOR_SHA256,
            ),
            "B500 Student privileged encoder": (
                _component_sha256(student_priv_state),
                ROOT_STUDENT_PRIVILEGED_ENCODER_SHA256,
            ),
            "Teacher actor": (_component_sha256(teacher_actor_state), TEACHER_ACTOR_SHA256),
            "Teacher privileged encoder": (
                _component_sha256(teacher_priv_state),
                TEACHER_PRIVILEGED_ENCODER_SHA256,
            ),
        }
        bad_hashes = [
            f"{name}: {actual} != {expected}"
            for name, (actual, expected) in expected_hashes.items()
            if actual != expected
        ]
        if bad_hashes:
            raise RuntimeError("component binding mismatch: " + "; ".join(bad_hashes))

        evidence_files = {
            "teacher_env_config": (TEACHER_ENV_CONFIG, TEACHER_ENV_CONFIG_SHA256),
            "b500_evaluation_manifest": (
                B500_EVALUATION_MANIFEST,
                B500_EVALUATION_MANIFEST_SHA256,
            ),
            "b500_handoff": (B500_HANDOFF, B500_HANDOFF_SHA256),
        }
        for label, (path, expected_sha) in evidence_files.items():
            actual_sha = sha256_file(path.resolve(strict=True))
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"qualified B500 evidence changed at {label}: {actual_sha} != {expected_sha}"
                )

        self._assert_b500_lineage(
            student,
            student_state,
            root_student_state,
            root_student_checkpoint,
            root_student_actual_sha,
        )

        self.student_actor = CheckpointMLP(student_actor_state)
        self.student_estimator = StudentEstimator(student_estimator_state)
        self.student_privileged_encoder = CheckpointMLP(
            _component_state(student_state, "priv_encoder.net."), final_tanh=True
        )
        self.root_student_actor = CheckpointMLP(root_actor_state)
        self.root_student_estimator = StudentEstimator(root_estimator_state)
        self.root_student_privileged_encoder = CheckpointMLP(
            _component_state(root_student_state, "priv_encoder.net."), final_tanh=True
        )
        self.teacher_actor = CheckpointMLP(teacher_actor_state)
        self.teacher_privileged_encoder = CheckpointMLP(
            _component_state(teacher_state, "priv_encoder.net."), final_tanh=True
        )
        self.to(device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self._validate_dimensions()
        student_pointers = self._module_storage_pointers(
            self.student_actor,
            self.student_estimator,
            self.student_privileged_encoder,
            self.root_student_actor,
            self.root_student_estimator,
            self.root_student_privileged_encoder,
        )
        teacher_pointers = self._module_storage_pointers(
            self.teacher_actor, self.teacher_privileged_encoder
        )
        if student_pointers & teacher_pointers:
            raise RuntimeError("Teacher and Student modules unexpectedly share tensor storage")
        if _component_sha256(student_actor_state) == _component_sha256(teacher_actor_state):
            raise RuntimeError("Teacher actor unexpectedly equals protected Student actor")

        self.binding_manifest = {
            "schema_version": 1,
            "student_checkpoint_role": "behavior-best qualified B500 driver",
            "student_checkpoint": str(student_checkpoint),
            "student_checkpoint_sha256": student_actual_sha,
            "student_checkpoint_iteration": int(student.get("iter", -1)),
            "student_effective_update_count": 500,
            "student_actor_component_sha256": CURRENT_STUDENT_ACTOR_SHA256,
            "student_estimator_component_sha256": ROOT_STUDENT_ESTIMATOR_SHA256,
            "student_privileged_encoder_component_sha256": ROOT_STUDENT_PRIVILEGED_ENCODER_SHA256,
            "root_student_checkpoint_role": "protected lineage anchor and optional read-only forward",
            "root_student_checkpoint": str(root_student_checkpoint),
            "root_student_checkpoint_sha256": root_student_actual_sha,
            "root_student_checkpoint_iteration": int(root_student.get("iter", -1)),
            "root_student_actor_component_sha256": ROOT_STUDENT_ACTOR_SHA256,
            "root_student_estimator_component_sha256": ROOT_STUDENT_ESTIMATOR_SHA256,
            "root_student_privileged_encoder_component_sha256": ROOT_STUDENT_PRIVILEGED_ENCODER_SHA256,
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_sha256": teacher_actual_sha,
            "teacher_checkpoint_iteration": int(teacher.get("iter", -1)),
            "teacher_actor_component_sha256": TEACHER_ACTOR_SHA256,
            "teacher_privileged_encoder_component_sha256": TEACHER_PRIVILEGED_ENCODER_SHA256,
            "b500_lineage_to_model900_verified": True,
            "b500_only_actor_rows_changed": [2, 3],
            "teacher_env_config": str(TEACHER_ENV_CONFIG.resolve()),
            "teacher_env_config_sha256": TEACHER_ENV_CONFIG_SHA256,
            "b500_evaluation_manifest": str(B500_EVALUATION_MANIFEST.resolve()),
            "b500_evaluation_manifest_sha256": B500_EVALUATION_MANIFEST_SHA256,
            "b500_handoff": str(B500_HANDOFF.resolve()),
            "b500_handoff_sha256": B500_HANDOFF_SHA256,
            "student_teacher_paths_distinct": True,
            "student_teacher_storage_independent": True,
            "student_teacher_parameter_and_buffer_storage_sets_compared_exhaustively": True,
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in self.parameters()),
            "student_policy_observation_dim": 570,
            "student_estimator_observation_dim": 570,
            "teacher_critic_observation_dim": 162,
            "teacher_privileged_input_dim": 159,
            "latent_dim": 64,
            "action_dim": 16,
            "empirical_normalization": False,
        }

    @staticmethod
    def _module_storage_pointers(*modules: nn.Module) -> set[int]:
        pointers: set[int] = set()
        for module in modules:
            tensors = list(module.parameters()) + list(module.buffers())
            for tensor in tensors:
                if tensor.numel() > 0:
                    pointers.add(tensor.untyped_storage().data_ptr())
        return pointers

    @staticmethod
    def _assert_b500_lineage(
        checkpoint: Mapping[str, Any],
        current_state: Mapping[str, torch.Tensor],
        root_state: Mapping[str, torch.Tensor],
        root_path: Path,
        root_sha256: str,
    ) -> None:
        try:
            recovery = checkpoint["infos"]["robot_lab_algorithm_checkpoint_state"]["student_recovery"]
            binding = recovery["binding_manifest"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("B500 checkpoint lacks versioned recovery lineage state") from error
        if int(recovery.get("effective_update_count", -1)) != 500:
            raise RuntimeError("current Student is not the qualified 500-effective-update checkpoint")
        if (
            Path(binding.get("initial_student_checkpoint", "")).resolve() != root_path
            or binding.get("initial_student_sha256") != root_sha256
        ):
            raise RuntimeError("B500 checkpoint does not bind to the protected model_900 root")
        if (
            Path(binding.get("teacher_checkpoint", "")).resolve() != TEACHER_CHECKPOINT.resolve()
            or binding.get("teacher_sha256") != TEACHER_SHA256
            or binding.get("teacher_actor_component_sha256") != TEACHER_ACTOR_SHA256
            or binding.get("teacher_privileged_encoder_component_sha256")
            != TEACHER_PRIVILEGED_ENCODER_SHA256
        ):
            raise RuntimeError("B500 recovery state does not bind the canonical Teacher components")
        if set(current_state) != set(root_state):
            raise RuntimeError("B500 and model_900 state key sets differ")
        for key in current_state:
            current = current_state[key].detach().cpu()
            root = root_state[key].detach().cpu()
            if key == "actor.6.weight":
                mask = torch.ones(current.shape[0], dtype=torch.bool)
                mask[2:4] = False
                equal = torch.equal(current[mask], root[mask])
            elif key == "actor.6.bias":
                mask = torch.ones(current.shape[0], dtype=torch.bool)
                mask[2:4] = False
                equal = torch.equal(current[mask], root[mask])
            else:
                equal = torch.equal(current, root)
            if not equal:
                raise RuntimeError(f"B500 lineage has an unauthorized tensor delta: {key}")
        if torch.equal(current_state["actor.6.weight"][2:4], root_state["actor.6.weight"][2:4]):
            raise RuntimeError("B500 checkpoint has no trained rear-hip row delta")

    def student_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.student_actor.parameters()
        yield from self.student_estimator.parameters()
        yield from self.student_privileged_encoder.parameters()
        yield from self.root_student_actor.parameters()
        yield from self.root_student_estimator.parameters()
        yield from self.root_student_privileged_encoder.parameters()

    def teacher_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.teacher_actor.parameters()
        yield from self.teacher_privileged_encoder.parameters()

    @staticmethod
    def _first_linear(module: CheckpointMLP) -> nn.Linear:
        return next(layer for layer in module.modules() if isinstance(layer, nn.Linear))

    @staticmethod
    def _last_linear(module: CheckpointMLP) -> nn.Linear:
        return [layer for layer in module.modules() if isinstance(layer, nn.Linear)][-1]

    def _validate_dimensions(self) -> None:
        expected = {
            "student_actor_in": (self._first_linear(self.student_actor).in_features, 634),
            "student_actor_out": (self._last_linear(self.student_actor).out_features, 16),
            "teacher_actor_in": (self._first_linear(self.teacher_actor).in_features, 634),
            "teacher_actor_out": (self._last_linear(self.teacher_actor).out_features, 16),
            "student_estimator_in": (
                self._first_linear(self.student_estimator.encoder).in_features,
                570,
            ),
            "student_latent": (self.student_estimator.fc_mu.out_features, 64),
            "student_velocity": (self.student_estimator.vel_estimator.out_features, 3),
            "teacher_privileged_in": (
                self._first_linear(self.teacher_privileged_encoder).in_features,
                159,
            ),
            "teacher_latent": (self._last_linear(self.teacher_privileged_encoder).out_features, 64),
        }
        mismatches = [f"{name}={actual} (expected {wanted})" for name, (actual, wanted) in expected.items() if actual != wanted]
        if mismatches:
            raise ValueError("checkpoint architecture mismatch: " + ", ".join(mismatches))

    @torch.inference_mode()
    def forward_same_state(
        self,
        policy_observations: torch.Tensor,
        estimator_observations: torch.Tensor,
        critic_observations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tensors = {
            "policy": policy_observations,
            "estimator": estimator_observations,
            "critic": critic_observations,
        }
        expected_dims = {"policy": 570, "estimator": 570, "critic": 162}
        batch = None
        for name, tensor in tensors.items():
            if tensor.ndim != 2 or tensor.shape[1] != expected_dims[name]:
                raise ValueError(f"{name} observations must have shape [B,{expected_dims[name]}], got {tuple(tensor.shape)}")
            batch = tensor.shape[0] if batch is None else batch
            if tensor.shape[0] != batch:
                raise ValueError("all observation groups must describe the same state batch")

        device = next(self.parameters()).device
        policy_observations = policy_observations.to(device)
        estimator_observations = estimator_observations.to(device)
        critic_observations = critic_observations.to(device)
        student_mu, student_logvar, student_velocity = self.student_estimator(estimator_observations)
        student_latent = torch.clamp(student_mu, min=-1.0, max=1.0)
        student_action = self.student_actor(torch.cat((policy_observations, student_latent), dim=-1))

        # The Teacher contract intentionally cuts the true base velocity from
        # critic[:, :3], exactly matching VAEActorCritic._process_obs(stage=1).
        teacher_latent = self.teacher_privileged_encoder(critic_observations[:, 3:])
        student_action_teacher_latent = self.student_actor(
            torch.cat((policy_observations, teacher_latent), dim=-1)
        )
        teacher_action = self.teacher_actor(torch.cat((policy_observations, teacher_latent), dim=-1))
        root_mu, root_logvar, root_velocity = self.root_student_estimator(estimator_observations)
        root_latent = torch.clamp(root_mu, min=-1.0, max=1.0)
        root_action = self.root_student_actor(
            torch.cat((policy_observations, root_latent), dim=-1)
        )
        outputs = {
            "student_action": student_action,
            "student_action_teacher_latent": student_action_teacher_latent,
            "student_mu": student_mu,
            "student_logvar": student_logvar,
            "student_latent": student_latent,
            "student_velocity": student_velocity,
            "teacher_action_pre_prior": teacher_action,
            "teacher_latent": teacher_latent,
            "root_student_action": root_action,
            "root_student_mu": root_mu,
            "root_student_logvar": root_logvar,
            "root_student_latent": root_latent,
            "root_student_velocity": root_velocity,
        }
        if not all(bool(torch.isfinite(value).all().item()) for value in outputs.values()):
            raise RuntimeError("same-state model forward produced a non-finite value")
        return outputs


class HistoricalPairedAuditModels(nn.Module):
    """Independent historical Student/Teacher pair for the frozen 0707 envelope."""

    _module_storage_pointers = staticmethod(BoundAuditModels._module_storage_pointers)
    _first_linear = staticmethod(BoundAuditModels._first_linear)
    _last_linear = staticmethod(BoundAuditModels._last_linear)
    _validate_dimensions = BoundAuditModels._validate_dimensions
    forward_same_state = BoundAuditModels.forward_same_state

    def __init__(
        self,
        *,
        student_checkpoint: Path,
        student_sha256: str,
        teacher_checkpoint: Path,
        teacher_sha256: str,
        student_agent_config: Path,
        student_agent_config_sha256: str,
        student_env_config: Path,
        student_env_config_sha256: str,
        teacher_agent_config: Path,
        teacher_agent_config_sha256: str,
        teacher_env_config: Path,
        teacher_env_config_sha256: str,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        assert_frozen_authority()
        paths = {
            "student_checkpoint": Path(student_checkpoint).resolve(strict=True),
            "teacher_checkpoint": Path(teacher_checkpoint).resolve(strict=True),
            "student_agent_config": Path(student_agent_config).resolve(strict=True),
            "student_env_config": Path(student_env_config).resolve(strict=True),
            "teacher_agent_config": Path(teacher_agent_config).resolve(strict=True),
            "teacher_env_config": Path(teacher_env_config).resolve(strict=True),
        }
        if paths["student_checkpoint"] == paths["teacher_checkpoint"]:
            raise RuntimeError("historical Student and Teacher checkpoints must be distinct")
        expected_files = {
            "student_checkpoint": student_sha256,
            "teacher_checkpoint": teacher_sha256,
            "student_agent_config": student_agent_config_sha256,
            "student_env_config": student_env_config_sha256,
            "teacher_agent_config": teacher_agent_config_sha256,
            "teacher_env_config": teacher_env_config_sha256,
        }
        actual_files = {name: sha256_file(path) for name, path in paths.items()}
        mismatches = [
            f"{name}: {actual_files[name]} != {expected}"
            for name, expected in expected_files.items()
            if actual_files[name] != expected
        ]
        if mismatches:
            raise RuntimeError("historical pair binding mismatch: " + "; ".join(mismatches))

        student, _ = _load_checkpoint(paths["student_checkpoint"], student_sha256)
        teacher, _ = _load_checkpoint(paths["teacher_checkpoint"], teacher_sha256)
        student_state = student["model_state_dict"]
        teacher_state = teacher["model_state_dict"]
        student_actor_state = _component_state(student_state, "actor.")
        student_estimator_state = _component_state(student_state, "estimator.")
        student_priv_state = _component_state(student_state, "priv_encoder.")
        teacher_actor_state = _component_state(teacher_state, "actor.")
        teacher_priv_state = _component_state(teacher_state, "priv_encoder.")

        self.student_actor = CheckpointMLP(student_actor_state)
        self.student_estimator = StudentEstimator(student_estimator_state)
        self.student_privileged_encoder = CheckpointMLP(
            _component_state(student_state, "priv_encoder.net."), final_tanh=True
        )
        # Same-State session retains an optional anchor decomposition.  For a
        # historical pair the anchor is an independent frozen copy of that
        # exact Student, not a second checkpoint or a training reference.
        self.root_student_actor = copy.deepcopy(self.student_actor)
        self.root_student_estimator = copy.deepcopy(self.student_estimator)
        self.root_student_privileged_encoder = copy.deepcopy(self.student_privileged_encoder)
        self.teacher_actor = CheckpointMLP(teacher_actor_state)
        self.teacher_privileged_encoder = CheckpointMLP(
            _component_state(teacher_state, "priv_encoder.net."), final_tanh=True
        )
        self.to(device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._validate_dimensions()
        student_pointers = self._module_storage_pointers(
            self.student_actor,
            self.student_estimator,
            self.student_privileged_encoder,
            self.root_student_actor,
            self.root_student_estimator,
            self.root_student_privileged_encoder,
        )
        teacher_pointers = self._module_storage_pointers(
            self.teacher_actor, self.teacher_privileged_encoder
        )
        if student_pointers & teacher_pointers:
            raise RuntimeError("historical Teacher and Student share tensor storage")
        self.binding_manifest = {
            "schema_version": 1,
            "kind": "highstep_v15_historical_pair_binding",
            "student_checkpoint_role": "0707 historical Student driver",
            "student_checkpoint": str(paths["student_checkpoint"]),
            "student_checkpoint_sha256": actual_files["student_checkpoint"],
            "student_checkpoint_iteration": int(student.get("iter", -1)),
            "student_actor_component_sha256": _component_sha256(student_actor_state),
            "student_estimator_component_sha256": _component_sha256(student_estimator_state),
            "student_privileged_encoder_component_sha256": _component_sha256(student_priv_state),
            "teacher_checkpoint": str(paths["teacher_checkpoint"]),
            "teacher_checkpoint_sha256": actual_files["teacher_checkpoint"],
            "teacher_checkpoint_iteration": int(teacher.get("iter", -1)),
            "teacher_actor_component_sha256": _component_sha256(teacher_actor_state),
            "teacher_privileged_encoder_component_sha256": _component_sha256(teacher_priv_state),
            "saved_configs": {
                name: {"path": str(paths[name]), "sha256": actual_files[name]}
                for name in (
                    "student_agent_config", "student_env_config",
                    "teacher_agent_config", "teacher_env_config",
                )
            },
            "student_teacher_paths_distinct": True,
            "student_teacher_storage_independent": True,
            "historical_anchor_is_independent_student_copy": True,
            "all_parameters_frozen": True,
            "student_policy_observation_dim": 570,
            "student_estimator_observation_dim": 570,
            "teacher_critic_observation_dim": 162,
            "teacher_privileged_input_dim": 159,
            "latent_dim": 64,
            "action_dim": 16,
            "empirical_normalization": False,
        }


@dataclass(frozen=True)
class PostPriorContract:
    """Immutable action-prior values from the validated Teacher env.yaml."""

    box_action_ids: tuple[int, ...] = (12, 13, 14, 15)
    height_sensor_name: str = "height_scanner"
    command_name: str = "base_velocity"
    front_x_min: float = 0.25
    rear_x_max: float = -0.20
    max_abs_y: float = 0.30
    height_threshold: float = 0.060
    height_gate_width: float = 0.14
    min_forward_command: float = 0.06
    command_gate_width: float = 0.22
    commit_height_delta_min: float = 0.04
    commit_height_delta_target: float = 0.18
    commit_gate_floor: float = 0.0
    front_reach_box_bias: float = -0.020
    rear_approach_box_bias: float = 0.002
    front_support_box_bias: float = 0.002
    rear_push_box_bias: float = -0.022
    min_box_target: float = 0.0
    max_box_target: float = 0.06
    update_count: int = 172300
    prior_start_update: int = 80
    prior_full_update: int = 520
    action_scale_revolute: float = 0.1
    action_scale_box: float = 0.02
    action_clip_lower: float = -60.0
    action_clip_upper: float = 60.0
    source_env_config_path: str = str(TEACHER_ENV_CONFIG)
    source_env_config_sha256: str = TEACHER_ENV_CONFIG_SHA256

    @classmethod
    def from_teacher_env_config(
        cls,
        path: Path = TEACHER_ENV_CONFIG,
        *,
        expected_sha256: str | None = None,
        update_count: int | None = None,
    ) -> "PostPriorContract":
        """Parse the frozen Teacher YAML without constructing Python-tagged config objects."""
        import yaml

        path = Path(path).expanduser().resolve(strict=True)
        actual_sha = sha256_file(path)
        expected_sha = TEACHER_ENV_CONFIG_SHA256 if expected_sha256 is None else expected_sha256
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Teacher env config SHA changed: {actual_sha} != {expected_sha}"
            )
        payload = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        action = payload["actions"]["joint_pos"]
        if tuple(action["joint_names"]) != JOINT_NAMES:
            raise RuntimeError("Teacher env YAML action joint order differs from the permanent contract")
        expected_class = "robot_lab.tasks.locomotion.velocity.mdp.actions:PhasedHighstepBoxBiasJointPositionAction"
        if action.get("class_type") != expected_class:
            raise RuntimeError("Teacher env YAML no longer uses the validated phased action prior")

        value_names = (
            "front_x_min",
            "rear_x_max",
            "max_abs_y",
            "height_threshold",
            "height_gate_width",
            "min_forward_command",
            "command_gate_width",
            "commit_height_delta_min",
            "commit_height_delta_target",
            "commit_gate_floor",
            "front_reach_box_bias",
            "rear_approach_box_bias",
            "front_support_box_bias",
            "rear_push_box_bias",
            "min_box_target",
            "max_box_target",
        )
        values = {name: float(action[name]) for name in value_names}
        clip = action["clip"][".*"]
        return cls(
            height_sensor_name=str(action["height_sensor_name"]),
            command_name=str(action["command_name"]),
            prior_start_update=int(action["prior_start_update"]),
            prior_full_update=int(action["prior_full_update"]),
            action_scale_revolute=float(action["scale"][".*_(hip_joint|thigh_joint|calf_joint)$"]),
            action_scale_box=float(action["scale"][".*_box_joint"]),
            action_clip_lower=float(clip[0]),
            action_clip_upper=float(clip[1]),
            source_env_config_path=str(path),
            source_env_config_sha256=actual_sha,
            update_count=(int(update_count) if update_count is not None else cls.update_count),
            **values,
        )


@dataclass
class PostPriorInputs:
    scale: torch.Tensor
    offset: torch.Tensor
    clip: torch.Tensor
    height_delta: torch.Tensor
    command_x: torch.Tensor
    front_rear_delta: torch.Tensor
    terrain_gate_available: bool


class NormalizedPostPriorResult(NamedTuple):
    raw_equivalent_actions: torch.Tensor
    physical_targets: torch.Tensor
    box_bias: torch.Tensor
    gate: torch.Tensor
    reach_gate: torch.Tensor
    push_gate: torch.Tensor
    prior_scale: float
    box_clip_mask: torch.Tensor


def _resolve_shared_post_prior() -> Callable[..., Any]:
    """Resolve lazily so importing this CPU audit never starts Isaac modules."""
    errors: list[str] = []
    for module_name in (
        "robot_lab.tasks.locomotion.velocity.mdp.actions",
        "robot_lab.tasks.locomotion.velocity.mdp.highstep_action_prior",
    ):
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, "compute_phased_highstep_post_prior")
            return function
        except (ImportError, AttributeError) as error:
            errors.append(f"{module_name}: {error}")
    raise RuntimeError(
        "shared compute_phased_highstep_post_prior is unavailable; the runtime must import the live "
        "mdp action implementation or inject the exact shared function. " + " | ".join(errors)
    )


def _call_shared_post_prior(
    function: Callable[..., Any],
    raw_actions: torch.Tensor,
    runtime: PostPriorInputs,
    contract: PostPriorContract,
) -> NormalizedPostPriorResult:
    """Keyword adapter for the shared live helper, with strict output checks."""
    processed_actions = raw_actions * runtime.scale + runtime.offset
    if runtime.clip.ndim == 3:
        processed_actions = torch.clamp(
            processed_actions,
            min=runtime.clip[..., 0],
            max=runtime.clip[..., 1],
        )
    elif runtime.clip.ndim == 2 and runtime.clip.shape[-1] == 2:
        processed_actions = torch.clamp(
            processed_actions,
            min=runtime.clip[:, 0],
            max=runtime.clip[:, 1],
        )
    else:
        raise ValueError(f"invalid action clip shape: {tuple(runtime.clip.shape)}")
    arguments = {
        "raw_actions": raw_actions,
        "actions": raw_actions,
        "processed_actions": processed_actions,
        "scale": runtime.scale,
        "offset": runtime.offset,
        "action_scale": runtime.scale,
        "action_offset": runtime.offset,
        "clip": runtime.clip,
        "box_action_ids": runtime.scale.new_tensor(contract.box_action_ids, dtype=torch.long),
        "height_delta": runtime.height_delta,
        "command_x": runtime.command_x,
        "front_rear_delta": runtime.front_rear_delta,
        "terrain_gate_available": runtime.terrain_gate_available,
        **dataclasses.asdict(contract),
    }
    # Contract fields duplicated above are runtime tensorized where necessary.
    arguments["box_action_ids"] = runtime.scale.new_tensor(contract.box_action_ids, dtype=torch.long)
    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        selected = arguments
    else:
        missing = [
            name for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in arguments
        ]
        if missing:
            raise RuntimeError(f"shared post-prior interface has unsupported required arguments: {missing}")
        selected = {name: value for name, value in arguments.items() if name in signature.parameters}
    result = function(**selected)

    def field_value(*names: str, default: Any = None) -> Any:
        for name in names:
            if hasattr(result, name):
                return getattr(result, name)
            if isinstance(result, Mapping) and name in result:
                return result[name]
        return default

    physical = field_value("physical_targets", "actions")
    raw_equivalent = field_value("raw_equivalent_actions")
    box_bias = field_value("box_bias")
    gate = field_value("gate")
    reach_gate = field_value("reach_gate")
    push_gate = field_value("push_gate")
    prior_scale = field_value("prior_scale")
    box_clip_mask = field_value("box_clip_mask", "clip_mask")
    required = {
        "physical_targets": physical,
        "raw_equivalent_actions": raw_equivalent,
        "box_bias": box_bias,
        "gate": gate,
        "reach_gate": reach_gate,
        "push_gate": push_gate,
        "prior_scale": prior_scale,
        "box_clip_mask": box_clip_mask,
    }
    absent = [name for name, value in required.items() if value is None]
    if absent:
        raise RuntimeError(f"shared post-prior result is missing fields: {absent}")
    box_clip_mask = box_clip_mask.to(dtype=torch.bool)
    if box_clip_mask.shape != raw_actions.shape:
        expected_box_shape = (raw_actions.shape[0], len(contract.box_action_ids))
        if box_clip_mask.shape != expected_box_shape:
            raise RuntimeError(
                "shared post-prior returned invalid box clip mask shape: "
                f"{tuple(box_clip_mask.shape)}"
            )
        expanded_clip_mask = torch.zeros_like(raw_actions, dtype=torch.bool)
        expanded_clip_mask[:, contract.box_action_ids] = box_clip_mask
        box_clip_mask = expanded_clip_mask
    normalized = NormalizedPostPriorResult(
        raw_equivalent_actions=raw_equivalent,
        physical_targets=physical,
        box_bias=box_bias,
        gate=gate,
        reach_gate=reach_gate,
        push_gate=push_gate,
        prior_scale=float(prior_scale),
        box_clip_mask=box_clip_mask,
    )
    expected_action_shape = raw_actions.shape
    if normalized.raw_equivalent_actions.shape != expected_action_shape or normalized.physical_targets.shape != expected_action_shape:
        raise RuntimeError("shared post-prior returned an invalid 16-D action shape")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (
            normalized.raw_equivalent_actions,
            normalized.physical_targets,
            normalized.box_bias,
            normalized.gate,
            normalized.reach_gate,
            normalized.push_gate,
        )
    ):
        raise RuntimeError("shared post-prior returned a non-finite value")
    return normalized


@dataclass(frozen=True)
class PhaseSignals:
    front_lifted: bool = False
    front_top_supported: bool = False
    rear_top_count: int = 0
    rear_hold_confirmed: bool = False

    def __post_init__(self) -> None:
        if self.rear_top_count not in (0, 1, 2):
            raise ValueError("rear_top_count must be 0, 1, or 2")


class MonotonicPhaseTracker:
    """Small explicit phase state machine; first/second rear and hold never collapse."""

    def __init__(self) -> None:
        self.phase_index = 0

    def update(self, signals: PhaseSignals) -> str:
        current = 0
        if signals.front_lifted:
            current = 1
        if signals.front_top_supported:
            current = 2
        if signals.rear_top_count >= 1:
            current = 3
        if signals.rear_top_count >= 2:
            current = 4
        if signals.rear_hold_confirmed:
            current = 5
        self.phase_index = max(self.phase_index, current)
        return PHASES[self.phase_index]


@dataclass(frozen=True)
class CausalReplacement:
    """One preregistered diagnostic intervention; never a deployable policy rule."""

    trial_name: str
    action_group: str
    phases: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action_group not in ACTION_GROUPS:
            raise ValueError(f"unknown action group {self.action_group!r}")
        unknown = set(self.phases) - set(PHASES)
        if unknown:
            raise ValueError(f"unknown intervention phases: {sorted(unknown)}")

    @property
    def action_indices(self) -> tuple[int, ...]:
        return ACTION_GROUPS[self.action_group]


@dataclass(frozen=True)
class AuditThresholds:
    action_group_mae: float = 0.35
    latent_mae: float = 0.15
    velocity_mae: float = 0.20
    sign_reference_epsilon: float = 0.05
    clip_epsilon: float = 1.0e-6
    consecutive_frames: int = 3

    def __post_init__(self) -> None:
        if min(self.action_group_mae, self.latent_mae, self.velocity_mae) <= 0.0:
            raise ValueError("divergence thresholds must be positive")
        if self.consecutive_frames < 1:
            raise ValueError("consecutive_frames must be >= 1")


@dataclass(frozen=True)
class AuditPreregistration:
    run_id: str
    seed: int
    scenario: str
    suite_plan_path: str
    suite_plan_sha256: str
    suite_kind: str = "base3"
    lateral_offset_m: float = 0.0
    yaw_offset_deg: float = 0.0
    thresholds: AuditThresholds = field(default_factory=AuditThresholds)
    replacement: CausalReplacement | None = None
    all_replacement_trials: tuple[CausalReplacement, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.scenario.strip():
            raise ValueError("run_id and scenario are required")
        if self.suite_kind not in {"base3", "causal"}:
            raise ValueError("suite_kind must be 'base3' or 'causal'")
        if not self.suite_plan_path or len(self.suite_plan_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.suite_plan_sha256
        ):
            raise ValueError("suite plan path and lowercase SHA256 are required")
        if abs(self.lateral_offset_m) > 0.5 or abs(self.yaw_offset_deg) > 30.0:
            raise ValueError("audit lateral/yaw offset is outside the preregistration safety envelope")
        if len(self.all_replacement_trials) > 3:
            raise ValueError("the formal spec permits at most three preregistered replacement trials")
        names = [trial.trial_name for trial in self.all_replacement_trials]
        if len(set(names)) != len(names):
            raise ValueError("causal replacement trial names must be unique")
        if self.replacement is not None and self.replacement.trial_name not in names:
            raise ValueError("active replacement must be present in all_replacement_trials")
        if self.suite_kind == "base3":
            if self.replacement is not None or self.all_replacement_trials:
                raise ValueError("the immutable base3 suite forbids causal replacement")
            matches = [
                trajectory
                for trajectory in DEFAULT_AUDIT_TRAJECTORIES
                if trajectory["run_id"] == self.run_id
                and trajectory["seed"] == self.seed
                and trajectory["scenario"] == self.scenario
                and trajectory["lateral_offset_m"] == self.lateral_offset_m
                and trajectory["yaw_offset_deg"] == self.yaw_offset_deg
            ]
            if len(matches) != 1:
                raise ValueError("base3 preregistration is not one of the fixed three trajectories")


class SameStateAuditSession:
    """Collect one single-environment trace and write deterministic audit evidence."""

    def __init__(
        self,
        *,
        models: BoundAuditModels,
        output_dir: Path,
        preregistration: AuditPreregistration,
        prior_contract: PostPriorContract | None = None,
        post_prior_function: Callable[..., Any] | None = None,
        driver_action_source: str = "student",
    ):
        assert_frozen_authority()
        self.models = models
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preregistration = preregistration
        self.prior_contract = prior_contract or PostPriorContract.from_teacher_env_config()
        self.prior_contract_sha256 = _canonical_json_sha256(self.prior_contract)
        suite_plan_path = Path(preregistration.suite_plan_path).expanduser().resolve(strict=True)
        suite_plan_actual_sha = sha256_file(suite_plan_path)
        if suite_plan_actual_sha != preregistration.suite_plan_sha256:
            raise RuntimeError(
                f"suite plan SHA mismatch: {suite_plan_actual_sha} != {preregistration.suite_plan_sha256}"
            )
        if preregistration.suite_kind == "base3":
            if suite_plan_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise RuntimeError("base3 suite plan must be read-only before audit starts")
            if json.loads(suite_plan_path.read_text()) != _jsonable(base_suite_plan_payload()):
                raise RuntimeError("base3 suite plan content differs from the fixed preregistration")
        self.suite_plan_path = suite_plan_path
        self.suite_plan_sha256 = suite_plan_actual_sha
        self.post_prior_function = post_prior_function or _resolve_shared_post_prior()
        if driver_action_source not in {"student", "runner"}:
            raise ValueError("driver_action_source must be 'student' or 'runner'")
        if driver_action_source == "runner" and preregistration.replacement is not None:
            raise RuntimeError("causal replacement is forbidden with an independent state driver")
        self.driver_action_source = driver_action_source
        self.phase_tracker = MonotonicPhaseTracker()
        self.records: list[dict[str, Any]] = []
        self.last_step = -1
        self._divergence_streak = 0
        self._divergence_streak_start: dict[str, Any] | None = None
        self.earliest_divergence: dict[str, Any] | None = None
        self.frames_path = self.output_dir / "frames.jsonl"
        if self.frames_path.exists():
            raise FileExistsError(f"refusing to append to an existing audit trace: {self.frames_path}")

        source_sha = sha256_file(Path(__file__))
        self.audit_code_sha256 = source_sha
        manifest = {
            "schema_version": 1,
            "kind": "highstep_same_state_preregistration",
            "read_only": True,
            "single_student_on_policy_environment_required": driver_action_source == "student",
            "driver_action_source": driver_action_source,
            "teacher_environment_steps": 0,
            "teacher_target": "model_172300 independent actor+privileged encoder, exact post-prior 16-D action",
            "student_source": "qualified B500 model_498 actor+estimator",
            "protected_student_lineage_root": "model_900 (also evaluated as a read-only anchor)",
            "joint_names": JOINT_NAMES,
            "permanent_action_contract": {
                "default_dof_pos": DEFAULT_DOF_POS,
                "action_scale": [0.1] * 12 + [0.02] * 4,
                "joint_pos_clip": [[-60.0, 60.0]] * 16,
            },
            "action_groups": ACTION_GROUPS,
            "phases": PHASES,
            "preregistration": preregistration,
            "suite_plan_path_verified": str(self.suite_plan_path),
            "suite_plan_sha256_verified": self.suite_plan_sha256,
            "post_prior_contract": self.prior_contract,
            "post_prior_contract_sha256": self.prior_contract_sha256,
            "checkpoint_binding": models.binding_manifest,
            "audit_code_sha256": source_sha,
            "shared_actions_path": str(ACTIONS_PATH.resolve()),
            "shared_actions_sha256": EXPECTED_ACTIONS_SHA256,
            "shared_post_prior_function": {
                "name": getattr(self.post_prior_function, "__name__", None),
                "module": getattr(self.post_prior_function, "__module__", None),
            },
            "spec_path": str(SPEC_PATH.resolve()),
            "spec_sha256": EXPECTED_SPEC_SHA256,
        }
        prereg_path = self.output_dir / "preregistration.json"
        if prereg_path.exists():
            raise FileExistsError(f"refusing to overwrite audit preregistration: {prereg_path}")
        _atomic_json(prereg_path, manifest, immutable=True)
        self.preregistration_path = prereg_path
        self.preregistration_sha256 = sha256_file(prereg_path)

    @staticmethod
    def _physical_student_targets(
        raw_actions: torch.Tensor,
        prior_inputs: PostPriorInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unclipped = raw_actions * prior_inputs.scale + prior_inputs.offset
        clip = prior_inputs.clip
        if clip.ndim == 3:
            lower, upper = clip[..., 0], clip[..., 1]
        elif clip.ndim == 2 and clip.shape[-1] == 2:
            lower, upper = clip[:, 0], clip[:, 1]
        else:
            raise ValueError(f"clip must have shape [B,16,2] or [16,2], got {tuple(clip.shape)}")
        clipped = torch.clamp(unclipped, min=lower, max=upper)
        clip_mask = torch.abs(clipped - unclipped) > 1.0e-7
        return clipped, clip_mask

    @torch.inference_mode()
    def evaluate_frame(
        self,
        *,
        step: int,
        policy_observations: torch.Tensor,
        estimator_observations: torch.Tensor,
        critic_observations: torch.Tensor,
        true_base_linear_velocity: torch.Tensor,
        prior_inputs: PostPriorInputs,
        phase_signals: PhaseSignals,
        runner_inference_function: Callable[[], torch.Tensor],
        physical_state_digest_function: Callable[[], str],
    ) -> torch.Tensor:
        if step != self.last_step + 1:
            raise ValueError(f"audit steps must be contiguous: got {step} after {self.last_step}")
        if policy_observations.shape[0] != 1:
            raise ValueError("same-state audit requires exactly one Student on-policy environment")
        if true_base_linear_velocity.shape != (1, 3):
            raise ValueError("true_base_linear_velocity must have shape [1,3]")

        def input_hashes() -> tuple[dict[str, Any], str, str, str]:
            replay = {
                "scale": prior_inputs.scale,
                "offset": prior_inputs.offset,
                "clip": prior_inputs.clip,
                "height_delta": prior_inputs.height_delta,
                "command_x": prior_inputs.command_x,
                "front_rear_delta": prior_inputs.front_rear_delta,
                "terrain_gate_available": prior_inputs.terrain_gate_available,
                "post_prior_contract_sha256": self.prior_contract_sha256,
            }
            replay_sha = _canonical_json_sha256(replay)
            observation_sha = _tensor_collection_sha256(
                (
                    ("policy", policy_observations),
                    ("estimator", estimator_observations),
                    ("critic", critic_observations),
                    ("true_base_linear_velocity", true_base_linear_velocity),
                )
            )
            snapshot_sha = _canonical_json_sha256(
                {
                    "observation_tensor_sha256": observation_sha,
                    "prior_replay_inputs_sha256": replay_sha,
                    "phase_signals": phase_signals,
                }
            )
            return replay, replay_sha, observation_sha, snapshot_sha

        (
            prior_replay_inputs,
            prior_replay_inputs_sha256,
            observation_tensor_sha256,
            same_state_snapshot_sha256,
        ) = input_hashes()

        state_digest_before = str(physical_state_digest_function())
        outputs = self.models.forward_same_state(
            policy_observations,
            estimator_observations,
            critic_observations,
        )
        runner_student_action = runner_inference_function()
        if runner_student_action.shape != outputs["student_action"].shape:
            raise RuntimeError(
                "runner inference action shape differs from manual B500 forward: "
                f"{tuple(runner_student_action.shape)} != {tuple(outputs['student_action'].shape)}"
            )
        runner_student_action = runner_student_action.to(outputs["student_action"].device)
        runner_manual_max_error = float(
            torch.max(torch.abs(runner_student_action - outputs["student_action"])).item()
        )
        runner_manual_match = bool(
            torch.allclose(
                runner_student_action,
                outputs["student_action"],
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        )
        if not runner_manual_match and self.driver_action_source == "student":
            raise RuntimeError(
                "manual B500 Student action differs from runner inference "
                f"(max_abs_error={runner_manual_max_error})"
            )
        teacher_post = _call_shared_post_prior(
            self.post_prior_function,
            outputs["teacher_action_pre_prior"],
            prior_inputs,
            self.prior_contract,
        )
        student_action = outputs["student_action"]
        student_physical, student_clip_mask = self._physical_student_targets(student_action, prior_inputs)
        teacher_target = teacher_post.raw_equivalent_actions
        action_error = student_action - teacher_target
        student_teacher_latent_action = outputs["student_action_teacher_latent"]
        decomposition_latent = student_action - student_teacher_latent_action
        decomposition_actor = student_teacher_latent_action - outputs["teacher_action_pre_prior"]
        decomposition_prior = outputs["teacher_action_pre_prior"] - teacher_target
        decomposition_residual = action_error - (
            decomposition_latent + decomposition_actor + decomposition_prior
        )
        decomposition_residual_max = float(torch.max(torch.abs(decomposition_residual)).item())
        if decomposition_residual_max > 1.0e-6:
            raise RuntimeError(
                "action decomposition identity exceeded tolerance: "
                f"{decomposition_residual_max} > 1e-6"
            )
        latent_raw_error = outputs["student_mu"] - outputs["teacher_latent"]
        latent_error = outputs["student_latent"] - outputs["teacher_latent"]
        student_mu_oob_ratio = float(torch.mean((torch.abs(outputs["student_mu"]) > 1.0).float()).item())
        critic_velocity_target = critic_observations[:, :3].to(outputs["student_velocity"].device)
        velocity_error_critic_scale = outputs["student_velocity"] - critic_velocity_target
        physical_velocity_estimate = outputs["student_velocity"] / 2.0
        velocity_error_physical = physical_velocity_estimate - true_base_linear_velocity.to(
            outputs["student_velocity"].device
        )
        phase = self.phase_tracker.update(phase_signals)

        threshold = self.preregistration.thresholds
        sign_eligible = torch.abs(teacher_target) >= threshold.sign_reference_epsilon
        sign_error = sign_eligible & (torch.sign(student_action) != torch.sign(teacher_target))
        group_mae = {
            name: float(torch.mean(torch.abs(action_error[:, indices])).item())
            for name, indices in ACTION_GROUPS.items()
        }
        teacher_latent_substitution_error = student_teacher_latent_action - teacher_target
        teacher_latent_substitution_group_mae = {
            name: float(torch.mean(torch.abs(teacher_latent_substitution_error[:, indices])).item())
            for name, indices in ACTION_GROUPS.items()
        }
        teacher_latent_substitution_group_reduction = {
            name: {
                "absolute": group_mae[name] - teacher_latent_substitution_group_mae[name],
                "relative": (
                    (group_mae[name] - teacher_latent_substitution_group_mae[name]) / group_mae[name]
                    if group_mae[name] > 1.0e-12
                    else 0.0
                ),
            }
            for name in ACTION_GROUPS
        }
        action_trigger_groups = [
            name for name, mae in group_mae.items() if mae >= threshold.action_group_mae
        ]
        latent_raw_mae = float(torch.mean(torch.abs(latent_raw_error)).item())
        latent_mae = float(torch.mean(torch.abs(latent_error)).item())
        velocity_critic_mae = float(torch.mean(torch.abs(velocity_error_critic_scale)).item())
        velocity_mae = float(torch.mean(torch.abs(velocity_error_physical)).item())
        causes: list[str] = []
        if action_trigger_groups:
            causes.append("action")
        if latent_mae >= threshold.latent_mae:
            causes.append("latent")
        if velocity_mae >= threshold.velocity_mae:
            causes.append("velocity")
        triggered = bool(causes)
        if triggered:
            if self._divergence_streak == 0:
                self._divergence_streak_start = {
                    "step": int(step),
                    "phase": phase,
                    "causes": causes,
                    "action_groups": action_trigger_groups,
                    "group_mae": group_mae,
                    "latent_mae": latent_mae,
                    "latent_raw_mu_mae": latent_raw_mae,
                    "velocity_mae": velocity_mae,
                    "same_state_snapshot_sha256": same_state_snapshot_sha256,
                    "prior_replay_inputs_sha256": prior_replay_inputs_sha256,
                }
            self._divergence_streak += 1
            if self.earliest_divergence is None and self._divergence_streak >= threshold.consecutive_frames:
                self.earliest_divergence = dict(self._divergence_streak_start or {})
                self.earliest_divergence["confirmed_at_step"] = int(step)
                self.earliest_divergence["consecutive_frames"] = int(self._divergence_streak)
        else:
            self._divergence_streak = 0
            self._divergence_streak_start = None

        action_to_apply = (
            student_action.clone()
            if self.driver_action_source == "student"
            else runner_student_action.clone()
        )
        intervention_mask = torch.zeros_like(action_to_apply, dtype=torch.bool)
        replacement = self.preregistration.replacement
        if replacement is not None and phase in replacement.phases:
            indices = replacement.action_indices
            action_to_apply[:, indices] = teacher_target[:, indices]
            intervention_mask[:, indices] = True

        state_digest_after = str(physical_state_digest_function())
        if state_digest_before != state_digest_after:
            raise RuntimeError(
                "same-state Teacher/Student/runner forward mutated the simulator physical state"
            )
        _, replay_sha_after, observation_sha_after, snapshot_sha_after = input_hashes()
        if (
            replay_sha_after != prior_replay_inputs_sha256
            or observation_sha_after != observation_tensor_sha256
            or snapshot_sha_after != same_state_snapshot_sha256
        ):
            raise RuntimeError("same-state model/runner forward mutated observation or prior inputs")

        record = {
            "step": int(step),
            "phase": phase,
            "phase_signals": phase_signals,
            "policy_observations": policy_observations[0],
            "estimator_observations": estimator_observations[0],
            "critic_observations": critic_observations[0],
            "observation_tensor_sha256": observation_tensor_sha256,
            "prior_replay_inputs": prior_replay_inputs,
            "prior_replay_inputs_sha256": prior_replay_inputs_sha256,
            "post_prior_contract_sha256": self.prior_contract_sha256,
            "same_state_snapshot_sha256": same_state_snapshot_sha256,
            "observation_and_prior_inputs_unchanged_by_forward": True,
            "student_action": student_action[0],
            "student_action_with_teacher_latent_SzT": student_teacher_latent_action[0],
            "runner_student_action": runner_student_action[0],
            "runner_manual_action_allclose": runner_manual_match,
            "runner_manual_action_max_abs_error": runner_manual_max_error,
            "root_model900_anchor_action": outputs["root_student_action"][0],
            "b500_minus_root_model900_action": (
                outputs["student_action"] - outputs["root_student_action"]
            )[0],
            "teacher_action_pre_prior": outputs["teacher_action_pre_prior"][0],
            "teacher_action_post_prior_raw_equivalent": teacher_target[0],
            "student_physical_target": student_physical[0],
            "teacher_physical_target_post_prior": teacher_post.physical_targets[0],
            "teacher_prior_box_bias": teacher_post.box_bias[0],
            "teacher_prior_gate": teacher_post.gate[0],
            "teacher_prior_reach_gate": teacher_post.reach_gate[0],
            "teacher_prior_push_gate": teacher_post.push_gate[0],
            "teacher_prior_scale": teacher_post.prior_scale,
            "action_error_student_minus_teacher": action_error[0],
            "action_decomposition_identity": "Smu-Tpost=(Smu-SzT)+(SzT-Tpre)+(Tpre-Tpost)",
            "action_decomposition_student_latent_effect_Smu_minus_SzT": decomposition_latent[0],
            "action_decomposition_actor_gap_SzT_minus_Tpre": decomposition_actor[0],
            "action_decomposition_prior_effect_Tpre_minus_Tpost": decomposition_prior[0],
            "action_decomposition_residual": decomposition_residual[0],
            "action_decomposition_residual_max_abs": decomposition_residual_max,
            "action_abs_error": torch.abs(action_error[0]),
            "action_sign_error": sign_error[0],
            "action_sign_eligible": sign_eligible[0],
            "student_target_clip_mask": student_clip_mask[0],
            "teacher_prior_box_clip_mask": teacher_post.box_clip_mask[0],
            "student_mu": outputs["student_mu"][0],
            "student_latent_clamped": outputs["student_latent"][0],
            "root_model900_mu": outputs["root_student_mu"][0],
            "root_model900_latent_clamped": outputs["root_student_latent"][0],
            "teacher_privileged_latent": outputs["teacher_latent"][0],
            "latent_raw_mu_error_student_minus_teacher": latent_raw_error[0],
            "latent_error_student_minus_teacher": latent_error[0],
            "latent_raw_mu_mae": latent_raw_mae,
            "student_mu_out_of_bounds_ratio": student_mu_oob_ratio,
            "student_velocity_estimate": outputs["student_velocity"][0],
            "critic_velocity_target_scaled": critic_velocity_target[0],
            "velocity_error_estimate_minus_critic_target": velocity_error_critic_scale[0],
            "velocity_critic_target_mae": velocity_critic_mae,
            "student_velocity_estimate_physical_divide_2": physical_velocity_estimate[0],
            "true_base_linear_velocity": true_base_linear_velocity[0],
            "velocity_error_estimate_divide_2_minus_root": velocity_error_physical[0],
            "group_mae": group_mae,
            "teacher_latent_substitution_group_mae": teacher_latent_substitution_group_mae,
            "teacher_latent_substitution_group_reduction": teacher_latent_substitution_group_reduction,
            "latent_mae": latent_mae,
            "velocity_mae": velocity_mae,
            "divergence_triggered": triggered,
            "divergence_causes": causes,
            "action_to_apply": action_to_apply[0],
            "causal_replacement_applied": bool(torch.any(intervention_mask).item()),
            "causal_replacement_mask": intervention_mask[0],
            "physical_state_digest_before_forward": state_digest_before,
            "physical_state_digest_after_forward": state_digest_after,
            "physical_state_unchanged_by_forward": True,
        }
        json_record = _jsonable(record)
        with self.frames_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(json_record)
        self.last_step = int(step)
        return action_to_apply

    def evaluate_runtime_frame(
        self,
        *,
        step: int,
        adapter: "HighstepRuntimeAdapter",
        runner_policy: Callable[[Any], torch.Tensor],
        observations: Any | None = None,
    ) -> torch.Tensor:
        """Narrow play-loop hook: sample once, audit, and return the sole env action."""
        observations = adapter.env.get_observations() if observations is None else observations
        sample = adapter.sample(observations)
        return self.evaluate_frame(
            step=step,
            policy_observations=sample.policy_observations,
            estimator_observations=sample.estimator_observations,
            critic_observations=sample.critic_observations,
            true_base_linear_velocity=sample.true_base_linear_velocity,
            prior_inputs=sample.prior_inputs,
            phase_signals=sample.phase_signals,
            runner_inference_function=lambda: runner_policy(observations),
            physical_state_digest_function=adapter.physical_state_digest,
        )

    @staticmethod
    def _mean_vectors(records: Sequence[Mapping[str, Any]], field: str) -> list[float] | None:
        if not records:
            return None
        tensor = torch.tensor([record[field] for record in records], dtype=torch.float64)
        return tensor.mean(dim=0).tolist()

    def finalize(self, *, outcome: Mapping[str, Any]) -> dict[str, Any]:
        assert_frozen_authority()
        if sha256_file(Path(__file__)) != self.audit_code_sha256:
            raise RuntimeError("same-state audit code changed during the run")
        if not self.records:
            raise RuntimeError("cannot finalize an empty same-state audit trace")
        phase_summaries: dict[str, Any] = {}
        for phase in PHASES:
            records = [record for record in self.records if record["phase"] == phase]
            if not records:
                phase_summaries[phase] = {"samples": 0}
                continue
            errors = torch.tensor(
                [record["action_error_student_minus_teacher"] for record in records],
                dtype=torch.float64,
            )
            sign_errors = torch.tensor(
                [record["action_sign_error"] for record in records], dtype=torch.float64
            )
            sign_eligible = torch.tensor(
                [record["action_sign_eligible"] for record in records], dtype=torch.float64
            )
            student_clip = torch.tensor(
                [record["student_target_clip_mask"] for record in records], dtype=torch.float64
            )
            teacher_clip = torch.tensor(
                [record["teacher_prior_box_clip_mask"] for record in records], dtype=torch.float64
            )
            latent_raw_errors = torch.tensor(
                [record["latent_raw_mu_error_student_minus_teacher"] for record in records],
                dtype=torch.float64,
            )
            latent_clamped_errors = torch.tensor(
                [record["latent_error_student_minus_teacher"] for record in records],
                dtype=torch.float64,
            )
            velocity_scaled_errors = torch.tensor(
                [record["velocity_error_estimate_minus_critic_target"] for record in records],
                dtype=torch.float64,
            )
            velocity_physical_errors = torch.tensor(
                [record["velocity_error_estimate_divide_2_minus_root"] for record in records],
                dtype=torch.float64,
            )
            eligible_count = sign_eligible.sum(dim=0)
            mismatch_count = sign_errors.sum(dim=0)
            sign_rate = [
                (float(mismatch_count[index] / eligible_count[index]) if eligible_count[index] > 0 else None)
                for index in range(len(JOINT_NAMES))
            ]
            phase_summaries[phase] = {
                "samples": len(records),
                "joint_bias_student_minus_teacher": errors.mean(dim=0).tolist(),
                "joint_mae": errors.abs().mean(dim=0).tolist(),
                "joint_sign_eligible_count": eligible_count.to(dtype=torch.int64).tolist(),
                "joint_sign_mismatch_count": mismatch_count.to(dtype=torch.int64).tolist(),
                "joint_sign_error_rate_among_eligible": sign_rate,
                "student_target_clip_rate": student_clip.mean(dim=0).tolist(),
                "teacher_prior_box_clip_rate": teacher_clip.mean(dim=0).tolist(),
                "latent_mae_mean": sum(record["latent_mae"] for record in records) / len(records),
                "latent_raw_mu_mae_mean": sum(
                    record["latent_raw_mu_mae"] for record in records
                ) / len(records),
                "latent_raw_mu_bias_by_dim": latent_raw_errors.mean(dim=0).tolist(),
                "latent_raw_mu_mae_by_dim": latent_raw_errors.abs().mean(dim=0).tolist(),
                "latent_clamped_actor_input_bias_by_dim": latent_clamped_errors.mean(dim=0).tolist(),
                "latent_clamped_actor_input_mae_by_dim": latent_clamped_errors.abs().mean(dim=0).tolist(),
                "student_mu_out_of_bounds_ratio_mean": sum(
                    record["student_mu_out_of_bounds_ratio"] for record in records
                ) / len(records),
                "velocity_mae_mean": sum(record["velocity_mae"] for record in records) / len(records),
                "velocity_critic_target_mae_mean": sum(
                    record["velocity_critic_target_mae"] for record in records
                ) / len(records),
                "velocity_scaled_pred_minus_critic_bias_by_dim": velocity_scaled_errors.mean(dim=0).tolist(),
                "velocity_scaled_pred_minus_critic_mae_by_dim": velocity_scaled_errors.abs().mean(dim=0).tolist(),
                "velocity_physical_pred_div2_minus_root_bias_by_dim": velocity_physical_errors.mean(dim=0).tolist(),
                "velocity_physical_pred_div2_minus_root_mae_by_dim": velocity_physical_errors.abs().mean(dim=0).tolist(),
                "teacher_latent_substitution_group_effect": {
                    group: {
                        "student_mu_to_teacher_post_mae": sum(
                            record["group_mae"][group] for record in records
                        ) / len(records),
                        "teacher_latent_to_teacher_post_mae": sum(
                            record["teacher_latent_substitution_group_mae"][group]
                            for record in records
                        ) / len(records),
                        "absolute_reduction": sum(
                            record["teacher_latent_substitution_group_reduction"][group]["absolute"]
                            for record in records
                        ) / len(records),
                        "relative_reduction": sum(
                            record["teacher_latent_substitution_group_reduction"][group]["relative"]
                            for record in records
                        ) / len(records),
                    }
                    for group in ACTION_GROUPS
                },
                "decomposition_latent_effect_mae_by_joint": torch.tensor(
                    [
                        record["action_decomposition_student_latent_effect_Smu_minus_SzT"]
                        for record in records
                    ],
                    dtype=torch.float64,
                ).abs().mean(dim=0).tolist(),
                "decomposition_actor_gap_mae_by_joint": torch.tensor(
                    [record["action_decomposition_actor_gap_SzT_minus_Tpre"] for record in records],
                    dtype=torch.float64,
                ).abs().mean(dim=0).tolist(),
                "decomposition_prior_effect_mae_by_joint": torch.tensor(
                    [record["action_decomposition_prior_effect_Tpre_minus_Tpost"] for record in records],
                    dtype=torch.float64,
                ).abs().mean(dim=0).tolist(),
                "decomposition_residual_max_abs": max(
                    record["action_decomposition_residual_max_abs"] for record in records
                ),
                "runner_manual_action_max_abs_error": max(
                    record["runner_manual_action_max_abs_error"] for record in records
                ),
                "causal_replacement_frames": sum(
                    bool(record["causal_replacement_applied"]) for record in records
                ),
            }

        reached = [phase for phase in PHASES if phase_summaries[phase]["samples"] > 0]
        summary = {
            "schema_version": 1,
            "kind": "highstep_same_state_run_summary",
            "run_id": self.preregistration.run_id,
            "seed": self.preregistration.seed,
            "scenario": self.preregistration.scenario,
            "lateral_offset_m": self.preregistration.lateral_offset_m,
            "yaw_offset_deg": self.preregistration.yaw_offset_deg,
            "suite_kind": self.preregistration.suite_kind,
            "suite_plan_path": str(self.suite_plan_path),
            "suite_plan_sha256": self.suite_plan_sha256,
            "preregistration_path": str(self.preregistration_path),
            "preregistration_sha256": self.preregistration_sha256,
            "audit_code_sha256": self.audit_code_sha256,
            "shared_actions_sha256": EXPECTED_ACTIONS_SHA256,
            "spec_sha256": EXPECTED_SPEC_SHA256,
            "teacher_env_config_sha256": TEACHER_ENV_CONFIG_SHA256,
            "read_only_model_audit": True,
            "single_student_on_policy_environment": self.driver_action_source == "student",
            "driver_action_source": self.driver_action_source,
            "teacher_environment_steps": 0,
            "frame_count": len(self.records),
            "all_forward_state_digests_unchanged": all(
                record["physical_state_unchanged_by_forward"] for record in self.records
            ),
            "all_observation_and_prior_inputs_unchanged": all(
                record["observation_and_prior_inputs_unchanged_by_forward"]
                for record in self.records
            ),
            "runner_manual_action_allclose_all_frames": all(
                record["runner_manual_action_allclose"] for record in self.records
            ),
            "action_decomposition_residual_global_max_abs": max(
                record["action_decomposition_residual_max_abs"] for record in self.records
            ),
            "joint_names": JOINT_NAMES,
            "phase_coverage": reached,
            "phase_summaries": phase_summaries,
            "earliest_reproducible_divergence": self.earliest_divergence,
            "replacement": self.preregistration.replacement,
            "outcome": dict(outcome),
            "checkpoint_binding": self.models.binding_manifest,
            "frames_path": str(self.frames_path),
            "frames_sha256": sha256_file(self.frames_path),
        }
        _atomic_json(self.output_dir / "run_summary.json", summary)
        return summary


@dataclass
class RuntimeAuditSample:
    policy_observations: torch.Tensor
    estimator_observations: torch.Tensor
    critic_observations: torch.Tensor
    true_base_linear_velocity: torch.Tensor
    prior_inputs: PostPriorInputs
    phase_signals: PhaseSignals


class HighstepRuntimeAdapter:
    """Duck-typed, read-only extractor for one existing Student environment.

    Construction and sampling do not call ``reset`` or ``step``.  The caller is
    responsible for using the same front-step reset and real-gain/no-random-event
    setup as core9.  This adapter fails closed when that reset context or the
    exact 16-joint action contract is unavailable.
    """

    def __init__(self, env: Any, contract: PostPriorContract | None = None):
        self.env = env
        self.unwrapped = env.unwrapped
        self.contract = contract or PostPriorContract.from_teacher_env_config()
        if int(self.unwrapped.num_envs) != 1:
            raise ValueError("same-state audit requires exactly one environment")
        self.asset = self.unwrapped.scene["robot"]
        terms = getattr(self.unwrapped.action_manager, "_terms", {})
        self.action_term = terms.get("joint_pos") if isinstance(terms, dict) else None
        if self.action_term is None:
            raise RuntimeError("Student environment has no joint_pos action term")
        if tuple(getattr(self.action_term, "_joint_names", ())) != JOINT_NAMES:
            raise RuntimeError("runtime action joint order does not match the permanent 16-D contract")
        mapping_reference = self.asset.data.root_pos_w
        runtime_scale = self._vector(
            self.action_term._scale,
            16,
            device=mapping_reference.device,
            dtype=mapping_reference.dtype,
        )
        expected_scale = runtime_scale.new_tensor(
            [[self.contract.action_scale_revolute] * 12 + [self.contract.action_scale_box] * 4]
        )
        if not torch.equal(runtime_scale, expected_scale):
            raise RuntimeError("Student runtime action scale differs from the Teacher action contract")
        runtime_clip = torch.as_tensor(
            self.action_term._clip,
            device=mapping_reference.device,
            dtype=mapping_reference.dtype,
        )
        if runtime_clip.ndim == 2:
            runtime_clip = runtime_clip.unsqueeze(0)
        if runtime_clip.shape != (1, 16, 2) or not bool(
            torch.all(runtime_clip[..., 0] == self.contract.action_clip_lower).item()
            and torch.all(runtime_clip[..., 1] == self.contract.action_clip_upper).item()
        ):
            raise RuntimeError("Student runtime action clip differs from the Teacher action contract")
        runtime_offset = self._vector(
            self.action_term._offset,
            16,
            device=mapping_reference.device,
            dtype=mapping_reference.dtype,
        )
        expected_offset = runtime_offset.new_tensor([DEFAULT_DOF_POS])
        if not torch.allclose(runtime_offset, expected_offset, rtol=0.0, atol=1.0e-7):
            raise RuntimeError("Student runtime default_dof_pos differs from the permanent action contract")
        self.front_foot_ids, _ = self.asset.find_bodies(["FL_foot", "FR_foot"], preserve_order=True)
        self.rear_foot_ids, _ = self.asset.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
        self.context = getattr(self.unwrapped, "_front_step_eval_context", None)
        if not self.context or not bool(self.context.get("reset_valid", False)):
            raise RuntimeError("same-state audit requires a verified front-step reset context")

        self.height_sensor = None
        try:
            self.height_sensor = self.unwrapped.scene[self.contract.height_sensor_name]
        except (KeyError, TypeError):
            sensors = getattr(self.unwrapped.scene, "sensors", {})
            self.height_sensor = (
                sensors.get(self.contract.height_sensor_name) if isinstance(sensors, dict) else None
            )
        if self.height_sensor is None or not hasattr(self.height_sensor, "ray_starts"):
            raise RuntimeError("formal same-state audit requires the Teacher-configured height scanner")
        self.front_ray_mask = None
        self.rear_ray_mask = None
        ray_starts = self.height_sensor.ray_starts[0]
        side = torch.abs(ray_starts[:, 1]) <= self.contract.max_abs_y
        self.front_ray_mask = (ray_starts[:, 0] >= self.contract.front_x_min) & side
        self.rear_ray_mask = (ray_starts[:, 0] <= self.contract.rear_x_max) & side
        if not bool(torch.any(self.front_ray_mask).item()) or not bool(
            torch.any(self.rear_ray_mask).item()
        ):
            raise RuntimeError("Teacher-configured prior ray masks are empty in the Student runtime")

        self.contact_sensor = None
        try:
            self.contact_sensor = self.unwrapped.scene.sensors["contact_forces"]
            self.front_contact_ids, _ = self.contact_sensor.find_bodies(
                ["FL_foot", "FR_foot"], preserve_order=True
            )
            self.rear_contact_ids, _ = self.contact_sensor.find_bodies(
                ["RL_foot", "RR_foot"], preserve_order=True
            )
        except (KeyError, AttributeError, TypeError):
            raise RuntimeError(
                "same-state phase audit requires the four-foot contact_forces sensor"
            )
        if len(self.front_contact_ids) != 2 or len(self.rear_contact_ids) != 2:
            raise RuntimeError("contact_forces sensor did not resolve all four feet")
        self.front_top_streak = 0
        self.front_lift_streak = 0
        self.rear_top_streaks = [0, 0]
        self.rear_hold_streak = 0
        self._last_phase_step: int | None = None
        self._last_phase_signals: PhaseSignals | None = None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
        if mask is None or not bool(torch.any(mask).item()):
            return fallback
        selected = values[:, mask]
        valid = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
        count = valid.float().sum(dim=1)
        total = torch.where(valid, selected, torch.zeros_like(selected)).sum(dim=1)
        mean = total / torch.clamp(count, min=1.0)
        return torch.where(count > 0.0, mean, fallback)

    @staticmethod
    def _vector(value: Any, length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim > 1:
            tensor = tensor[0]
        tensor = tensor.reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(length)
        if tensor.numel() != length:
            raise RuntimeError(f"action mapping vector has {tensor.numel()} entries, expected {length}")
        return tensor.unsqueeze(0)

    def physical_state_digest(self) -> str:
        """Hash physical state and action-term buffers without advancing simulation."""
        data = self.asset.data
        names = (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        tensors: list[tuple[str, torch.Tensor]] = []
        for name in names:
            value = getattr(data, name, None)
            if isinstance(value, torch.Tensor):
                tensors.append((f"robot.{name}", value))
        for name in ("_raw_actions", "_processed_actions"):
            value = getattr(self.action_term, name, None)
            if isinstance(value, torch.Tensor):
                tensors.append((f"action_term.{name}", value))
        if not tensors:
            raise RuntimeError("runtime exposes no physical tensors for same-state digest")
        return _tensor_collection_sha256(tensors)

    def _prior_inputs(self) -> PostPriorInputs:
        reference = self.asset.data.root_pos_w
        device, dtype = reference.device, reference.dtype
        scale = self._vector(self.action_term._scale, 16, device=device, dtype=dtype)
        offset = self._vector(self.action_term._offset, 16, device=device, dtype=dtype)
        clip = torch.as_tensor(self.action_term._clip, device=device, dtype=dtype)
        if clip.ndim == 2:
            clip = clip.unsqueeze(0)
        if clip.shape != (1, 16, 2):
            raise RuntimeError(f"invalid runtime action clip shape: {tuple(clip.shape)}")

        terrain_available = True
        height_delta = torch.zeros(1, device=device, dtype=dtype)
        try:
            ray_hits_z = self.height_sensor.data.ray_hits_w[..., 2]
            sensor_z = self.height_sensor.data.pos_w[:, 2]
            front_z = self._masked_mean(ray_hits_z, self.front_ray_mask, sensor_z)
            rear_z = self._masked_mean(ray_hits_z, self.rear_ray_mask, front_z)
            height_delta = front_z - rear_z
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("height scanner has no valid live ray-hit data") from error
        command = self.unwrapped.command_manager.get_command(self.contract.command_name)
        front_z = self.asset.data.body_pos_w[:, self.front_foot_ids, 2].mean(dim=1)
        rear_z = self.asset.data.body_pos_w[:, self.rear_foot_ids, 2].mean(dim=1)
        return PostPriorInputs(
            scale=scale,
            offset=offset,
            clip=clip,
            height_delta=height_delta,
            command_x=command[:, 0],
            front_rear_delta=front_z - rear_z,
            terrain_gate_available=terrain_available,
        )

    def _phase_signals(self) -> PhaseSignals:
        if not hasattr(self.unwrapped, "common_step_counter"):
            raise RuntimeError("runtime does not expose common_step_counter for phase de-duplication")
        current_step = int(self.unwrapped.common_step_counter)
        if self._last_phase_step == current_step and self._last_phase_signals is not None:
            return self._last_phase_signals
        foot_ids = list(self.front_foot_ids) + list(self.rear_foot_ids)
        foot_pos = self.asset.data.body_pos_w[0, foot_ids, :]
        top_z = self.context["top_z"][0]
        low_z = self.context["low_z"][0]
        approach = self.context["approach"]
        lateral = self.context["lateral"]
        origin_xy = self.context["origin_xy"][0]
        half_width = float(self.context["half_width"])
        relative = foot_pos[:, :2] - origin_xy.unsqueeze(0)
        outward = torch.sum(relative * approach.unsqueeze(0), dim=1)
        lateral_position = torch.sum(relative * lateral.unsqueeze(0), dim=1)
        inside = (torch.abs(outward) <= half_width + 0.04) & (
            torch.abs(lateral_position) <= half_width + 0.04
        )
        top_height = (foot_pos[:, 2] >= top_z - 0.04) & (foot_pos[:, 2] <= top_z + 0.09)

        data = self.contact_sensor.data
        contact_ids = [int(value) for value in self.front_contact_ids] + [
            int(value) for value in self.rear_contact_ids
        ]
        if hasattr(data, "net_forces_w") and data.net_forces_w is not None:
            forces = data.net_forces_w[0, contact_ids, :]
        elif hasattr(data, "net_forces_w_history") and data.net_forces_w_history is not None:
            forces = data.net_forces_w_history[0, 0, contact_ids, :]
        else:
            raise RuntimeError("contact_forces sensor has no current force vectors")
        norm = torch.linalg.norm(forces, dim=-1)
        upward = torch.clamp(forces[:, 2], min=0.0)
        loaded = (upward > 5.0) & (upward / torch.clamp(norm, min=1.0e-6) >= 0.45)
        top_loaded_contact = inside & top_height & loaded
        front_top_now = bool(torch.any(top_loaded_contact[:2]).item())
        self.front_top_streak = self.front_top_streak + 1 if front_top_now else 0
        rear_top_now = top_loaded_contact[2:]
        for index in range(2):
            self.rear_top_streaks[index] = (
                self.rear_top_streaks[index] + 1 if bool(rear_top_now[index].item()) else 0
            )
        rear_count = sum(streak >= 2 for streak in self.rear_top_streaks)
        front_lifted_now = bool(torch.any(foot_pos[:2, 2] > low_z + 0.08).item())
        self.front_lift_streak = self.front_lift_streak + 1 if front_lifted_now else 0

        base_pos = self.asset.data.root_pos_w[0]
        root_relative = base_pos[:2] - origin_xy
        root_outward = torch.sum(root_relative * approach)
        root_edge_margin = half_width - float(root_outward.item())
        rear_edge_margin = half_width - float(torch.max(outward[2:]).item())
        root_h_top = float((base_pos[2] - top_z).item())
        gravity = self.asset.data.projected_gravity_b[0]

        # Heading-frame rear width, matching the core9 no-severe-inward metric.
        quat = self.asset.data.root_quat_w[0]
        w, x, y, z = quat.unbind()
        yaw_sin = 2.0 * (w * z + x * y)
        yaw_cos = 1.0 - 2.0 * (y * y + z * z)
        rear_delta_xy = foot_pos[2:, :2] - base_pos[:2].unsqueeze(0)
        rear_y = -yaw_sin * rear_delta_xy[:, 0] + yaw_cos * rear_delta_xy[:, 1]
        rear_width = float(torch.abs(rear_y[0] - rear_y[1]).item())
        rear_min_abs_y = float(torch.min(torch.abs(rear_y)).item())
        rear_hold_now = bool(
            rear_count == 2
            and bool(torch.all(rear_top_now).item())
            and rear_edge_margin >= 0.04
            and root_edge_margin >= 0.20
            and root_h_top >= 0.28
            and rear_width >= 0.18
            and rear_min_abs_y >= 0.04
            and abs(float(gravity[0].item())) <= 0.42
            and abs(float(gravity[1].item())) <= 0.32
        )
        self.rear_hold_streak = self.rear_hold_streak + 1 if rear_hold_now else 0
        signals = PhaseSignals(
            front_lifted=self.front_lift_streak >= 2,
            front_top_supported=self.front_top_streak >= 2,
            rear_top_count=rear_count,
            rear_hold_confirmed=self.rear_hold_streak >= 25,
        )
        self._last_phase_step = current_step
        self._last_phase_signals = signals
        return signals

    def sample(self, observations: Any | None = None) -> RuntimeAuditSample:
        observations = self.env.get_observations() if observations is None else observations
        try:
            policy = observations["policy"]
            estimator = observations["estimator"]
            critic = observations["critic"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("audit requires policy, estimator, and critic observations from one frame") from error
        return RuntimeAuditSample(
            policy_observations=policy,
            estimator_observations=estimator,
            critic_observations=critic,
            true_base_linear_velocity=self.asset.data.root_lin_vel_b[:, :3],
            prior_inputs=self._prior_inputs(),
            phase_signals=self._phase_signals(),
        )


def aggregate_run_summaries(paths: Sequence[Path]) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError("the immutable base3 audit requires exactly three run summaries")
    runs = [json.loads(Path(path).read_text()) for path in paths]
    run_ids = [run.get("run_id") for run in runs]
    expected_by_id = {trajectory["run_id"]: trajectory for trajectory in DEFAULT_AUDIT_TRAJECTORIES}
    if set(run_ids) != set(expected_by_id) or len(set(run_ids)) != 3:
        raise ValueError("run summaries are not the exact fixed base3 identities")
    suite_shas = {run.get("suite_plan_sha256") for run in runs}
    if len(suite_shas) != 1 or None in suite_shas:
        raise ValueError("base3 run summaries do not share one suite plan SHA")
    evidence_fields = {
        "audit_code_sha256": sha256_file(Path(__file__)),
        "shared_actions_sha256": EXPECTED_ACTIONS_SHA256,
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "teacher_env_config_sha256": TEACHER_ENV_CONFIG_SHA256,
    }
    for field_name, expected_value in evidence_fields.items():
        values = {run.get(field_name) for run in runs}
        if values != {expected_value}:
            raise ValueError(f"base3 {field_name} binding differs across runs or from authority")
    runtime_binding_shas = {
        run.get("outcome", {}).get("same_state_runtime_binding_sha256") for run in runs
    }
    if len(runtime_binding_shas) != 1 or None in runtime_binding_shas:
        raise ValueError("base3 runs do not share one runtime code/config binding SHA")
    for run in runs:
        expected = expected_by_id[run["run_id"]]
        identity = (
            run.get("seed") == expected["seed"]
            and run.get("scenario") == expected["scenario"]
            and run.get("lateral_offset_m") == expected["lateral_offset_m"]
            and run.get("yaw_offset_deg") == expected["yaw_offset_deg"]
            and run.get("suite_kind") == "base3"
            and run.get("replacement") is None
        )
        if not identity:
            raise ValueError(f"base3 run identity/config mismatch: {run['run_id']}")
        plan_path = Path(run["suite_plan_path"]).resolve(strict=True)
        if sha256_file(plan_path) != run["suite_plan_sha256"]:
            raise ValueError(f"suite plan SHA no longer matches for {run['run_id']}")
        if json.loads(plan_path.read_text()) != _jsonable(base_suite_plan_payload()):
            raise ValueError("suite plan content differs from fixed base3")
        covered = set(run.get("phase_coverage", []))
        if not set(expected["required_phases"]) <= covered or set(expected["forbidden_phases"]) & covered:
            raise ValueError(f"base3 phase role was not reproduced: {run['run_id']}")
        outcome = run.get("outcome", {})
        if (
            bool(outcome.get("full_climb_success"))
            != expected["expected_full_climb_success"]
            or bool(outcome.get("rear_on_platform_hold_success"))
            != expected["expected_rear_hold_success"]
        ):
            raise ValueError(f"base3 behavior outcome was not reproduced: {run['run_id']}")
        runtime_contract = base_suite_plan_payload()["runtime_contract"]
        runtime_binding_path = Path(
            outcome.get("same_state_runtime_binding_path", "")
        ).resolve(strict=True)
        if sha256_file(runtime_binding_path) != outcome.get("same_state_runtime_binding_sha256"):
            raise ValueError(f"runtime binding artifact changed: {run['run_id']}")
        if (
            int(run.get("frame_count", -1)) != runtime_contract["play_max_steps"]
            or int(outcome.get("loop_steps", -1)) != runtime_contract["play_max_steps"]
            or outcome.get("reset_valid") is not True
            or outcome.get("terminated_early") is not False
            or outcome.get("fixed_velocity_command") != runtime_contract["fixed_velocity_command"]
            or int(outcome.get("eval_action_delay_steps_runtime", -1))
            != runtime_contract["eval_action_delay_steps"]
            or outcome.get("keep_play_randomization") is not False
            or outcome.get("action_prior_enabled") is not False
            or outcome.get("same_state_runtime_contract_verified") is not True
        ):
            raise ValueError(f"base3 runtime contract was not verified: {run['run_id']}")
        if not run.get("runner_manual_action_allclose_all_frames", False):
            raise ValueError(f"runner/manual parity failed: {run['run_id']}")
        if not run.get("all_forward_state_digests_unchanged", False):
            raise ValueError(f"same-state physical digest failed: {run['run_id']}")
        if not run.get("all_observation_and_prior_inputs_unchanged", False):
            raise ValueError(f"same-state observation/prior digest failed: {run['run_id']}")
        if float(run.get("action_decomposition_residual_global_max_abs", 1.0)) > 1.0e-6:
            raise ValueError(f"action decomposition failed: {run['run_id']}")
    binding_keys = (
        "student_checkpoint_sha256",
        "student_actor_component_sha256",
        "student_estimator_component_sha256",
        "teacher_checkpoint_sha256",
        "teacher_actor_component_sha256",
        "teacher_privileged_encoder_component_sha256",
        "root_student_checkpoint_sha256",
    )
    reference = runs[0]["checkpoint_binding"]
    for run in runs[1:]:
        if any(run["checkpoint_binding"].get(key) != reference.get(key) for key in binding_keys):
            raise ValueError("run summaries do not share the same protected Student/Teacher bindings")

    covered = {phase for run in runs for phase in run.get("phase_coverage", [])}
    earliest = [
        {"run_id": run["run_id"], **run["earliest_reproducible_divergence"]}
        for run in runs
        if run["run_id"] != "seed11_nominal"
        and run.get("earliest_reproducible_divergence") is not None
    ]
    earliest.sort(key=lambda item: (PHASES.index(item["phase"]), item["step"], item["run_id"]))
    required = set(PHASES)
    return {
        "schema_version": 1,
        "kind": "highstep_same_state_aggregate",
        "run_count": len(runs),
        "run_ids": run_ids,
        "suite_plan_sha256": next(iter(suite_shas)),
        "same_state_runtime_binding_sha256": next(iter(runtime_binding_shas)),
        "checkpoint_binding": reference,
        "phase_coverage": [phase for phase in PHASES if phase in covered],
        "missing_required_phases": [phase for phase in PHASES if phase not in covered],
        "formal_phase_coverage_complete": required <= covered,
        "earliest_reproducible_divergence_across_runs": earliest[0] if earliest else None,
        "causal_replacement_run_count": 0,
        "training_route_selected": None,
        "training_allowed": False,
        "note": "This evidence artifact does not start or select R1/R2; route selection requires review of the audit report.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-bindings", help="CPU-only protected checkpoint binding audit")
    verify.add_argument("--student", type=Path, default=CURRENT_STUDENT_CHECKPOINT)
    verify.add_argument("--student-sha256", default=CURRENT_STUDENT_SHA256)
    verify.add_argument("--root-student", type=Path, default=ROOT_STUDENT_CHECKPOINT)
    verify.add_argument("--root-student-sha256", default=ROOT_STUDENT_SHA256)
    verify.add_argument("--teacher", type=Path, default=TEACHER_CHECKPOINT)
    verify.add_argument("--teacher-sha256", default=TEACHER_SHA256)
    verify.add_argument("--output", type=Path)

    aggregate = subparsers.add_parser("aggregate", help="aggregate completed run_summary.json files")
    aggregate.add_argument("run_summaries", type=Path, nargs="+")
    aggregate.add_argument("--output", type=Path, required=True)
    plan = subparsers.add_parser(
        "write-plan", help="write the fixed B500 success/failure three-trajectory audit plan"
    )
    plan.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    assert_frozen_authority()
    if args.command == "verify-bindings":
        models = BoundAuditModels(
            student_checkpoint=args.student,
            student_sha256=args.student_sha256,
            root_student_checkpoint=args.root_student,
            root_student_sha256=args.root_student_sha256,
            teacher_checkpoint=args.teacher,
            teacher_sha256=args.teacher_sha256,
            device="cpu",
        )
        manifest = {
            "binding": models.binding_manifest,
            "audit_code_sha256": sha256_file(Path(__file__)),
            "spec_sha256": sha256_file(SPEC_PATH),
        }
        if args.output:
            _atomic_json(args.output.expanduser().resolve(), manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "aggregate":
        aggregate = aggregate_run_summaries(args.run_summaries)
        _atomic_json(args.output.expanduser().resolve(), aggregate)
        print(json.dumps(aggregate, sort_keys=True))
        return 0
    if args.command == "write-plan":
        payload = base_suite_plan_payload()
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite suite plan: {output}")
        _atomic_json(output, payload, immutable=True)
        print(
            json.dumps(
                {"plan": _jsonable(payload), "path": str(output), "sha256": sha256_file(output)},
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
