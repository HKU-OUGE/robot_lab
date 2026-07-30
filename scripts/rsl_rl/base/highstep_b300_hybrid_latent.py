"""Narrow runtime helpers for B300 canonical hybrid-prior latent distillation."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from highstep_phase_residual_inference import _extract_policy_observation


WORKFLOW_ID = "highstep_b300_canonical_hybrid_prior_latent_20260718"
EXPECTED_JOINTS = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, payload: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_collection_authority(path: str | Path, expected_sha: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if sha256_file(resolved) != expected_sha:
        raise RuntimeError("B300 hybrid collection preregistration SHA mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not (
        payload.get("kind") == "highstep_b300_canonical_hybrid_collection_preregistration"
        and payload.get("workflow_id") == WORKFLOW_ID
        and payload.get("status") == "frozen_before_single_canonical_collection"
        and payload.get("training_forbidden_until_final_preregistration") is True
    ):
        raise RuntimeError("B300 hybrid collection authority changed")
    for path_key, sha_key in (
        ("spec_path", "spec_sha256"),
        ("teacher_checkpoint", "teacher_sha256"),
        ("approved_run01_csv", "approved_run01_csv_sha256"),
        ("approved_run01_manifest", "approved_run01_manifest_sha256"),
        ("user_visual_approval", "user_visual_approval_sha256"),
        ("recorded_command_trace", "recorded_command_trace_sha256"),
        ("recorded_command_manifest", "recorded_command_manifest_sha256"),
    ):
        if sha256_file(payload[path_key]) != payload[sha_key]:
            raise RuntimeError(f"B300 hybrid collection binding changed: {path_key}")
    return resolved, payload


def _obs_tensor(observations: object, key: str) -> torch.Tensor:
    if hasattr(observations, "keys") and key in observations.keys():
        value = observations[key]
    elif isinstance(observations, dict) and key in observations:
        value = observations[key]
    else:
        raise RuntimeError(f"B300 canonical collection lacks observation group {key!r}")
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"B300 observation group {key!r} is not a tensor")
    return value


def _term_vector(term: object, name: str, dim: int, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(getattr(term, name, None), device=reference.device, dtype=reference.dtype)
    if value.ndim > 1:
        value = value[0]
    value = value.reshape(-1)
    if value.numel() == 1:
        value = value.expand(dim)
    if value.numel() != dim or not bool(torch.all(torch.isfinite(value)).item()):
        raise RuntimeError(f"B300 action term exposes invalid {name}")
    return value


class CanonicalTensorCollector:
    """Capture one exact pre-step Teacher trajectory without changing physics."""

    def __init__(
        self,
        output_dir: str | Path,
        preregistration: str | Path,
        preregistration_sha256: str,
        *,
        policy_module: object,
        env: object,
    ) -> None:
        self.output = Path(output_dir).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite canonical tensor collection: {self.output}")
        self.output.mkdir(parents=True)
        self.prereg_path, self.prereg = _load_collection_authority(
            preregistration, preregistration_sha256
        )
        self.prereg_sha = preregistration_sha256
        self.policy = policy_module
        if not hasattr(self.policy, "actor") or not hasattr(self.policy, "priv_encoder"):
            raise RuntimeError("B300 canonical collection requires Teacher actor and priv_encoder")
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        terms = getattr(self.env.action_manager, "_terms", {})
        self.action_term = terms.get("joint_pos") if isinstance(terms, dict) else None
        if self.action_term is None:
            raise RuntimeError("B300 canonical collection requires joint_pos action term")
        self.joint_names = tuple(getattr(self.action_term, "_joint_names", ()))
        if self.joint_names != EXPECTED_JOINTS:
            raise RuntimeError(f"B300 action order changed: {self.joint_names}")
        ids = getattr(self.action_term, "_joint_ids", None)
        if isinstance(ids, slice):
            self.joint_ids = list(range(len(EXPECTED_JOINTS)))[ids]
        elif isinstance(ids, torch.Tensor):
            self.joint_ids = [int(value) for value in ids.detach().cpu().flatten().tolist()]
        else:
            self.joint_ids = [int(value) for value in ids]
        if self.joint_ids != list(range(16)):
            raise RuntimeError(f"B300 joint ids changed: {self.joint_ids}")
        reference = self.robot.data.joint_pos[:, self.joint_ids]
        self.scale = _term_vector(self.action_term, "_scale", 16, reference)
        self.offset = _term_vector(self.action_term, "_offset", 16, reference)
        expected_scale = torch.tensor([0.1] * 12 + [0.02] * 4, device=reference.device, dtype=reference.dtype)
        if not torch.equal(self.scale, expected_scale):
            raise RuntimeError(f"B300 action scale changed: {self.scale.detach().cpu().tolist()}")
        self.records: dict[str, list[torch.Tensor]] = {
            name: [] for name in (
                "student_obs_570", "critic_obs", "teacher_latent_raw_64",
                "teacher_latent_clamped_64", "teacher_pre_prior_action_16",
                "teacher_post_prior_policy_action_16", "mapped_target_16",
                "pre_joint_pos_16", "pre_joint_vel_16", "post_joint_pos_16",
                "post_joint_vel_16", "command_3", "episode_step", "reset_before_step",
            )
        }
        self.pending: dict[str, torch.Tensor] | None = None
        self.max_actor_reproduction_error = 0.0
        self.max_policy_unit_reprojection_error = 0.0
        self.step = 0

    def prepare(self, observations: object, teacher_policy_action: torch.Tensor, command: torch.Tensor) -> None:
        policy_obs = _extract_policy_observation(observations, 1).detach()
        critic_obs = _obs_tensor(observations, "critic").detach()
        if tuple(policy_obs.shape) != (1, 570) or critic_obs.shape[0] != 1:
            raise RuntimeError(
                f"B300 canonical observation contract changed: policy={tuple(policy_obs.shape)} "
                f"critic={tuple(critic_obs.shape)}"
            )
        with torch.no_grad():
            latent = self.policy.priv_encoder(critic_obs[:, 3:])
            clamped = torch.clamp(latent, -1.0, 1.0)
            reproduced = self.policy.actor(torch.cat((policy_obs, clamped), dim=-1))
        raw = teacher_policy_action.detach().reshape(1, 16)
        actor_error = float(torch.max(torch.abs(reproduced - raw)).detach().cpu())
        self.max_actor_reproduction_error = max(self.max_actor_reproduction_error, actor_error)
        if actor_error > 1.0e-6:
            raise RuntimeError(f"Teacher policy/actor reproduction mismatch: {actor_error}")
        self.pending = {
            "student_obs_570": policy_obs.clone(),
            "critic_obs": critic_obs.clone(),
            "teacher_latent_raw_64": latent.detach().clone(),
            "teacher_latent_clamped_64": clamped.detach().clone(),
            "teacher_pre_prior_action_16": raw.clone(),
            "pre_joint_pos_16": self.robot.data.joint_pos[:, self.joint_ids].detach().clone(),
            "pre_joint_vel_16": self.robot.data.joint_vel[:, self.joint_ids].detach().clone(),
            "command_3": command.detach().reshape(1, 3).clone(),
            "episode_step": torch.tensor([[self.step]], device=raw.device, dtype=torch.int64),
            "reset_before_step": torch.tensor([[self.step == 0]], device=raw.device, dtype=torch.bool),
        }

    def record_after_step(self) -> None:
        if self.pending is None:
            raise RuntimeError("B300 canonical record called without a pre-step sample")
        mapped = getattr(self.action_term, "processed_actions", None)
        if not isinstance(mapped, torch.Tensor) or tuple(mapped.shape) != (1, 16):
            raise RuntimeError("B300 action term exposes no 1x16 mapped target")
        mapped = mapped.detach().clone()
        post_policy = (mapped - self.offset.unsqueeze(0)) / self.scale.unsqueeze(0)
        reprojection = post_policy * self.scale.unsqueeze(0) + self.offset.unsqueeze(0)
        error = float(torch.max(torch.abs(reprojection - mapped)).detach().cpu())
        self.max_policy_unit_reprojection_error = max(self.max_policy_unit_reprojection_error, error)
        tolerance = float(self.prereg["collection"]["post_prior_policy_unit_reprojection_tolerance"])
        if error > tolerance:
            raise RuntimeError(f"post-prior policy-unit inversion failed: {error}")
        current = {
            **self.pending,
            "teacher_post_prior_policy_action_16": post_policy,
            "mapped_target_16": mapped,
            "post_joint_pos_16": self.robot.data.joint_pos[:, self.joint_ids].detach().clone(),
            "post_joint_vel_16": self.robot.data.joint_vel[:, self.joint_ids].detach().clone(),
        }
        for name in self.records:
            self.records[name].append(current[name].detach().cpu())
        self.pending = None
        self.step += 1

    @staticmethod
    def _approved_matrix(rows: list[dict[str, str]], prefix: str) -> torch.Tensor:
        return torch.tensor(
            [[float(row[f"{prefix}.{joint}"]) for joint in EXPECTED_JOINTS] for row in rows],
            dtype=torch.float32,
        )

    def finalize(self) -> dict[str, Any]:
        if self.pending is not None or self.step != 138:
            raise RuntimeError(f"B300 canonical collection is incomplete: step={self.step}")
        tensors = {name: torch.cat(values, dim=0).contiguous() for name, values in self.records.items()}
        expected = {
            "student_obs_570": (138, 570),
            "teacher_latent_raw_64": (138, 64),
            "teacher_latent_clamped_64": (138, 64),
            "teacher_pre_prior_action_16": (138, 16),
            "teacher_post_prior_policy_action_16": (138, 16),
            "mapped_target_16": (138, 16),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape or not bool(torch.all(torch.isfinite(tensors[name])).item()):
                raise RuntimeError(f"B300 canonical tensor invalid: {name} {tuple(tensors[name].shape)}")
        commands = tensors["command_3"][:, 0]
        expected_vx = torch.zeros(138)
        expected_vx[21:80] = 0.7200000286
        if not torch.equal(commands, expected_vx):
            raise RuntimeError("B300 canonical command sequence changed")

        with Path(self.prereg["approved_run01_csv"]).open(newline="", encoding="utf-8") as stream:
            approved_rows = list(csv.DictReader(stream))
        if len(approved_rows) != 138:
            raise RuntimeError("approved run01 no longer has 138 rows")
        comparisons = {}
        for name, prefix, tolerance in (
            ("teacher_pre_prior_action_16", "policy_raw", 2.0e-5),
            ("mapped_target_16", "mapped_target", 2.0e-5),
            ("post_joint_pos_16", "joint_pos", 2.0e-4),
            ("post_joint_vel_16", "joint_vel", 2.0e-3),
        ):
            approved = self._approved_matrix(approved_rows, prefix)
            max_error = float(torch.max(torch.abs(tensors[name] - approved)).item())
            comparisons[name] = {"max_abs": max_error, "tolerance": tolerance, "passed": max_error <= tolerance}
            if max_error > tolerance:
                raise RuntimeError(f"canonical centerline mismatch for {name}: {max_error} > {tolerance}")

        dataset_path = self.output / "canonical_tensor_trajectory.pt"
        _atomic_torch(dataset_path, tensors)
        manifest = {
            "schema_version": 1,
            "kind": "highstep_b300_canonical_hybrid_tensor_trajectory",
            "workflow_id": WORKFLOW_ID,
            "status": "completed_and_centerline_verified",
            "sample_count": 138,
            "unique_episode_count": 1,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "preregistration_path": str(self.prereg_path),
            "preregistration_sha256": self.prereg_sha,
            "teacher_checkpoint": self.prereg["teacher_checkpoint"],
            "teacher_sha256": self.prereg["teacher_sha256"],
            "fields": {name: list(value.shape) for name, value in tensors.items()},
            "mixed_target": {
                "nonbox_0_12": "teacher_pre_prior_action_16[0:12]",
                "box_12_16": "teacher_post_prior_policy_action_16[12:16]",
            },
            "post_prior_policy_unit_reprojection_max_abs": self.max_policy_unit_reprojection_error,
            "teacher_actor_reproduction_max_abs": self.max_actor_reproduction_error,
            "approved_centerline_comparison": comparisons,
            "automatic_real_robot_deployment_authorized": False,
        }
        manifest_path = self.output / "manifest.json"
        _atomic_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        manifest["manifest_sha256"] = sha256_file(manifest_path)
        return manifest


class DaggerTensorCollector:
    """Record same-pre-step Teacher labels while the Student drives physics."""

    def __init__(
        self,
        output_dir: str | Path,
        stage_manifest: str | Path,
        stage_manifest_sha256: str,
        *,
        policy_module: object,
        env: object,
    ) -> None:
        self.output = Path(output_dir).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite B300 DAgger collection: {self.output}")
        self.output.mkdir(parents=True)
        self.stage_manifest_path = Path(stage_manifest).expanduser().resolve(strict=True)
        if sha256_file(self.stage_manifest_path) != stage_manifest_sha256:
            raise RuntimeError("B300 DAgger stage manifest SHA mismatch")
        self.stage_manifest = json.loads(self.stage_manifest_path.read_text())
        if not (
            self.stage_manifest.get("kind") == "highstep_b300_hybrid_dagger_collection_stage"
            and self.stage_manifest.get("workflow_id") == WORKFLOW_ID
            and self.stage_manifest.get("student_drives_physics") is True
            and self.stage_manifest.get("teacher_labels_same_pre_step_state") is True
        ):
            raise RuntimeError("B300 DAgger collection authority mismatch")
        route_path = Path(self.stage_manifest["route_amendment"]).resolve(strict=True)
        if sha256_file(route_path) != self.stage_manifest["route_amendment_sha256"]:
            raise RuntimeError("B300 DAgger route amendment SHA mismatch")
        route = json.loads(route_path.read_text())
        if not (
            route.get("kind") == "highstep_b300_hybrid_dagger_route_amendment"
            and route.get("status")
            == "frozen_before_D1_collection_after_E300_diagnosis"
            and route.get("preregistration_sha256")
            == self.stage_manifest.get("preregistration_sha256")
            and self.stage_manifest.get("scenario") in route.get("scenarios", [])
        ):
            raise RuntimeError("B300 DAgger scenario is outside frozen route")
        teacher_path = Path(self.stage_manifest["teacher_checkpoint"]).resolve(strict=True)
        if sha256_file(teacher_path) != self.stage_manifest["teacher_sha256"]:
            raise RuntimeError("B300 DAgger Teacher SHA mismatch")
        teacher_env = Path(self.stage_manifest["teacher_env_config"]).resolve(strict=True)
        if sha256_file(teacher_env) != self.stage_manifest["teacher_env_config_sha256"]:
            raise RuntimeError("B300 DAgger Teacher prior configuration SHA mismatch")
        teacher_state = torch.load(teacher_path, map_location="cpu", weights_only=False)["model_state_dict"]
        self.teacher_actor = copy.deepcopy(policy_module.actor)
        self.teacher_priv_encoder = copy.deepcopy(policy_module.priv_encoder)
        self.teacher_actor.load_state_dict({key[len("actor."):]: value for key, value in teacher_state.items() if key.startswith("actor.")})
        self.teacher_priv_encoder.load_state_dict({key[len("priv_encoder."):]: value for key, value in teacher_state.items() if key.startswith("priv_encoder.")})
        device = next(policy_module.actor.parameters()).device
        self.teacher_actor.to(device).eval()
        self.teacher_priv_encoder.to(device).eval()
        for module in (self.teacher_actor, self.teacher_priv_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        terms = getattr(self.env.action_manager, "_terms", {})
        self.action_term = terms.get("joint_pos") if isinstance(terms, dict) else None
        if self.action_term is None:
            raise RuntimeError("B300 DAgger requires joint_pos action term")
        self.joint_names = tuple(getattr(self.action_term, "_joint_names", ()))
        if self.joint_names != EXPECTED_JOINTS:
            raise RuntimeError("B300 DAgger action order changed")
        self.joint_ids = list(range(16))
        reference = self.robot.data.joint_pos[:, self.joint_ids]
        self.scale = _term_vector(self.action_term, "_scale", 16, reference).unsqueeze(0)
        self.offset = _term_vector(self.action_term, "_offset", 16, reference).unsqueeze(0)
        self.box_ids = torch.tensor([12, 13, 14, 15], device=reference.device, dtype=torch.long)
        self.height_sensor = self.env.scene["height_scanner"]
        ray_starts = self.height_sensor.ray_starts[0]
        self.front_ray_mask = (ray_starts[:, 0] >= 0.25) & (torch.abs(ray_starts[:, 1]) <= 0.30)
        self.rear_ray_mask = (ray_starts[:, 0] <= -0.20) & (torch.abs(ray_starts[:, 1]) <= 0.30)
        self.front_foot_ids, _ = self.robot.find_bodies(["FL_foot", "FR_foot"], preserve_order=True)
        self.rear_foot_ids, _ = self.robot.find_bodies(["RL_foot", "RR_foot"], preserve_order=True)
        self.records: dict[str, list[torch.Tensor]] = {
            name: [] for name in (
                "student_obs_570", "critic_obs", "teacher_latent_raw_64",
                "teacher_latent_clamped_64", "teacher_pre_prior_action_16",
                "teacher_post_prior_policy_action_16", "student_raw_action_16",
                "pre_joint_pos_16", "pre_joint_vel_16", "post_joint_pos_16",
                "post_joint_vel_16", "command_3", "episode_step", "reset_before_step",
            )
        }
        self.pending: dict[str, torch.Tensor] | None = None
        self.step = 0

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        selected = values[:, mask]
        valid = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
        count = valid.float().sum(dim=1)
        mean = torch.where(valid, selected, torch.zeros_like(selected)).sum(dim=1) / torch.clamp(count, min=1.0)
        return torch.where(count > 0.0, mean, fallback)

    def prepare(self, observations: object, student_action: torch.Tensor, command: torch.Tensor) -> None:
        # Delayed until after AppLauncher has initialized Omniverse.
        from robot_lab.tasks.locomotion.velocity.mdp.actions import (
            compute_phased_highstep_post_prior,
        )

        policy_obs = _extract_policy_observation(observations, 1).detach()
        critic_obs = _obs_tensor(observations, "critic").detach()
        with torch.no_grad():
            latent = self.teacher_priv_encoder(critic_obs[:, 3:])
            clamped = torch.clamp(latent, -1.0, 1.0)
            teacher_raw = self.teacher_actor(torch.cat((policy_obs, clamped), dim=-1))
            processed = teacher_raw * self.scale + self.offset
            ray_hits_z = self.height_sensor.data.ray_hits_w[..., 2]
            sensor_z = self.height_sensor.data.pos_w[:, 2]
            front_z = self._masked_mean(ray_hits_z, self.front_ray_mask, sensor_z)
            rear_z = self._masked_mean(ray_hits_z, self.rear_ray_mask, front_z)
            foot_front_z = self.robot.data.body_pos_w[:, self.front_foot_ids, 2].mean(dim=1)
            foot_rear_z = self.robot.data.body_pos_w[:, self.rear_foot_ids, 2].mean(dim=1)
            prior = compute_phased_highstep_post_prior(
                teacher_raw, processed, self.box_ids, self.scale, self.offset,
                front_z - rear_z, command[:, 0], foot_front_z - foot_rear_z,
                terrain_gate_available=True, height_threshold=0.060, height_gate_width=0.14,
                min_forward_command=0.06, command_gate_width=0.22,
                commit_height_delta_min=0.04, commit_height_delta_target=0.18,
                commit_gate_floor=0.0, front_reach_box_bias=-0.020,
                rear_approach_box_bias=0.002, front_support_box_bias=0.002,
                rear_push_box_bias=-0.022, min_box_target=0.0, max_box_target=0.060,
                update_count=520.0, prior_start_update=80.0, prior_full_update=520.0,
            )
            mapped = prior.physical_targets
            teacher_post_policy = prior.raw_equivalent_actions
            if not torch.allclose(
                teacher_post_policy * self.scale + self.offset,
                mapped,
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise RuntimeError("B300 DAgger post-prior policy-unit inversion failed")
        self.pending = {
            "student_obs_570": policy_obs.clone(), "critic_obs": critic_obs.clone(),
            "teacher_latent_raw_64": latent.clone(), "teacher_latent_clamped_64": clamped.clone(),
            "teacher_pre_prior_action_16": teacher_raw.clone(),
            "teacher_post_prior_policy_action_16": teacher_post_policy.clone(),
            "student_raw_action_16": student_action.detach().reshape(1, 16).clone(),
            "pre_joint_pos_16": self.robot.data.joint_pos[:, self.joint_ids].detach().clone(),
            "pre_joint_vel_16": self.robot.data.joint_vel[:, self.joint_ids].detach().clone(),
            "command_3": command.detach().reshape(1, 3).clone(),
            "episode_step": torch.tensor([[self.step]], device=policy_obs.device, dtype=torch.int64),
            "reset_before_step": torch.tensor([[self.step == 0]], device=policy_obs.device, dtype=torch.bool),
        }

    def record_after_step(self) -> None:
        if self.pending is None:
            raise RuntimeError("B300 DAgger record lacks pre-step sample")
        current = {
            **self.pending,
            "post_joint_pos_16": self.robot.data.joint_pos[:, self.joint_ids].detach().clone(),
            "post_joint_vel_16": self.robot.data.joint_vel[:, self.joint_ids].detach().clone(),
        }
        for name in self.records:
            self.records[name].append(current[name].detach().cpu())
        self.pending = None
        self.step += 1

    def finalize(self) -> dict[str, Any]:
        if self.pending is not None or self.step != 138:
            raise RuntimeError(f"B300 DAgger collection incomplete: {self.step}")
        tensors = {name: torch.cat(values, dim=0).contiguous() for name, values in self.records.items()}
        for name, value in tensors.items():
            if not bool(torch.all(torch.isfinite(value)).item()):
                raise RuntimeError(f"B300 DAgger non-finite tensor: {name}")
        dataset = self.output / "dagger_tensor_trajectory.pt"
        _atomic_torch(dataset, tensors)
        manifest = {
            "schema_version": 1, "kind": "highstep_b300_hybrid_dagger_tensor_trajectory",
            "workflow_id": WORKFLOW_ID, "round": self.stage_manifest["round"],
            "scenario": self.stage_manifest["scenario"], "sample_count": 138,
            "student_drives_physics": True, "teacher_labels_same_pre_step_state": True,
            "dataset_path": str(dataset), "dataset_sha256": sha256_file(dataset),
            "stage_manifest": str(self.stage_manifest_path),
            "stage_manifest_sha256": sha256_file(self.stage_manifest_path),
            "fields": {name: list(value.shape) for name, value in tensors.items()},
        }
        manifest_path = self.output / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}
