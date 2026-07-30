"""Crash-tolerant, read-only joint trace recording for interactive play.

The recorder deliberately has no Isaac Lab dependency.  ``play.py`` owns the
runtime tensor extraction and passes plain sequences here, which keeps the
file format testable without starting the simulator.
"""

from __future__ import annotations

import atexit
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import torch


SCHEMA_VERSION = 1

STUDENT_TENSOR_WIDTHS = {
    "student_obs_570": 570,
    "estimator_obs_570": 570,
    "latent_raw_mu_64": 64,
    "latent_clamped_mu_64": 64,
    "actor_input_634": 634,
    "velocity_pred_3": 3,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_list(values: Sequence[Any], *, field: str, expected: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != expected:
        raise ValueError(f"{field} has {len(result)} values; expected {expected}")
    return result


def _float_tensor(values: Sequence[Any] | torch.Tensor, *, field: str, expected: int) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().cpu().reshape(-1).clone()
    if tensor.numel() != expected:
        raise ValueError(f"{field} has {tensor.numel()} values; expected {expected}")
    return tensor


class PlayJointRecorder:
    """Stream one env's applied action and measured joint state to CSV.

    The CSV is line-oriented and flushed periodically, so an interrupted GUI
    session still leaves a readable prefix.  ``manifest.json`` is written
    atomically at start and close and binds the checkpoint and play contract.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        joint_names: Sequence[str],
        step_dt: float,
        metadata: dict[str, Any],
        flush_every: int = 20,
        record_student_tensors: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.joint_names = [str(name) for name in joint_names]
        if len(self.joint_names) != 16:
            raise ValueError(
                "high-step joint recording requires the exact 16-action contract; "
                f"got {len(self.joint_names)} joints"
            )
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint recording joint names must be unique")
        self.step_dt = float(step_dt)
        if self.step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {self.step_dt}")
        self.flush_every = max(1, int(flush_every))
        self.record_student_tensors = bool(record_student_tensors)

        if self.output_dir.exists():
            raise FileExistsError(f"joint recording directory already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.csv_path = self.output_dir / "joint_trace.csv"
        self.student_tensors_path = self.output_dir / "student_tensors.pt"
        self.manifest_path = self.output_dir / "manifest.json"
        self._csv_handle = self.csv_path.open("x", encoding="utf-8", newline="", buffering=1)
        self._fieldnames = self._build_fieldnames()
        self._writer = csv.DictWriter(self._csv_handle, fieldnames=self._fieldnames)
        self._writer.writeheader()

        self._metadata = dict(metadata)
        self._created_at_utc = _utc_now()
        self._sample_count = 0
        self._episode_id = 0
        self._episode_step = 0
        self._pending_reset = True
        self._closed = False
        self._student_tensor_records: dict[str, list[torch.Tensor]] = {
            name: [] for name in STUDENT_TENSOR_WIDTHS
        }
        self._student_tensor_indices: dict[str, list[int]] = {
            "control_step": [],
            "episode_id": [],
            "episode_step": [],
            "reset_before_step": [],
        }
        self._write_manifest(status="recording")
        atexit.register(self.close, "process_exit")

    def _build_fieldnames(self) -> list[str]:
        fields = [
            "schema_version",
            "control_step",
            "episode_id",
            "episode_step",
            "sim_time_s",
            "episode_time_s",
            "wall_time_unix_ns",
            "reset_before_step",
            "terminated_after_step",
            "command.vx",
            "command.vy",
            "command.wz",
            "root_pos.x",
            "root_pos.y",
            "root_pos.z",
            "root_quat.w",
            "root_quat.x",
            "root_quat.y",
            "root_quat.z",
            "root_lin_vel_b.x",
            "root_lin_vel_b.y",
            "root_lin_vel_b.z",
            "root_ang_vel_b.x",
            "root_ang_vel_b.y",
            "root_ang_vel_b.z",
        ]
        for prefix in (
            "policy_raw",
            "action_manager_raw",
            "mapped_target",
            "joint_pos",
            "joint_vel",
        ):
            fields.extend(f"{prefix}.{name}" for name in self.joint_names)
        return fields

    def _manifest(self, *, status: str, close_reason: str | None = None) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "highstep_interactive_play_joint_record",
            "status": status,
            "created_at_utc": self._created_at_utc,
            "updated_at_utc": _utc_now(),
            "sample_count": self._sample_count,
            "episode_count": self._episode_id + 1 if self._sample_count else 0,
            "step_dt_s": self.step_dt,
            "joint_count": len(self.joint_names),
            "joint_names_in_action_order": self.joint_names,
            "csv_path": str(self.csv_path),
            "mapped_target_semantics": (
                "action-term processed_actions for env0 sampled immediately after env.step; "
                "includes the live action mapping, action prior and configured action delay"
            ),
            "joint_state_semantics": "env0 measured joint position/velocity after the same env.step",
            "policy_raw_semantics": "policy output passed to the RSL-RL environment wrapper",
            "action_manager_raw_semantics": "raw action retained by the Isaac Lab action manager",
            "metadata": self._metadata,
            "student_tensor_recording_enabled": self.record_student_tensors,
        }
        if self.record_student_tensors:
            manifest["student_tensor_semantics"] = {
                "student_obs_570": "pre-step policy observation for env0",
                "estimator_obs_570": "pre-step estimator observation actually passed to the Student estimator for env0",
                "latent_raw_mu_64": "pre-step raw estimator mu for env0",
                "latent_clamped_mu_64": "pre-step mu clamped elementwise to [-1, 1], as consumed by the actor",
                "actor_input_634": "pre-step concatenation of student_obs_570 and latent_clamped_mu_64",
                "velocity_pred_3": "pre-step estimator velocity prediction; recorded for diagnosis and not consumed by the actor",
            }
            manifest["student_tensor_sample_count"] = len(
                self._student_tensor_indices["control_step"]
            )
            manifest["student_tensor_expected_widths"] = dict(STUDENT_TENSOR_WIDTHS)
            if self.student_tensors_path.exists():
                manifest["student_tensors_path"] = str(self.student_tensors_path)
                manifest["student_tensors_sha256"] = _sha256_file(self.student_tensors_path)
                manifest["student_tensor_shapes"] = {
                    name: [len(self._student_tensor_records[name]), width]
                    for name, width in STUDENT_TENSOR_WIDTHS.items()
                }
        if close_reason is not None:
            manifest["close_reason"] = close_reason
        if self._closed and self.csv_path.exists():
            manifest["csv_sha256"] = _sha256_file(self.csv_path)
        return manifest

    def _write_manifest(self, *, status: str, close_reason: str | None = None) -> None:
        payload = self._manifest(status=status, close_reason=close_reason)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def mark_reset(self) -> None:
        """Start a new logical attempt on the next recorded control step."""
        self._pending_reset = True

    def record(
        self,
        *,
        control_step: int,
        terminated_after_step: bool,
        student_tensors: dict[str, Sequence[Any] | torch.Tensor] | None = None,
        policy_raw: Sequence[Any],
        action_manager_raw: Sequence[Any],
        mapped_target: Sequence[Any],
        joint_pos: Sequence[Any],
        joint_vel: Sequence[Any],
        command: Sequence[Any],
        root_pos: Sequence[Any],
        root_quat_wxyz: Sequence[Any],
        root_lin_vel_b: Sequence[Any],
        root_ang_vel_b: Sequence[Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed joint recording")
        if self.record_student_tensors:
            if student_tensors is None:
                raise ValueError("student tensor recording is enabled but this sample has no student tensors")
            missing = sorted(set(STUDENT_TENSOR_WIDTHS) - set(student_tensors))
            extra = sorted(set(student_tensors) - set(STUDENT_TENSOR_WIDTHS))
            if missing or extra:
                raise ValueError(
                    f"student tensor fields changed: missing={missing}, extra={extra}"
                )
            parsed_student_tensors = {
                name: _float_tensor(student_tensors[name], field=name, expected=width)
                for name, width in STUDENT_TENSOR_WIDTHS.items()
            }
        else:
            if student_tensors is not None:
                raise ValueError("student tensors were supplied without enabling student tensor recording")
            parsed_student_tensors = None
        if self._pending_reset and self._sample_count > 0:
            self._episode_id += 1
            self._episode_step = 0
        count = len(self.joint_names)
        vectors = {
            "policy_raw": _float_list(policy_raw, field="policy_raw", expected=count),
            "action_manager_raw": _float_list(
                action_manager_raw, field="action_manager_raw", expected=count
            ),
            "mapped_target": _float_list(mapped_target, field="mapped_target", expected=count),
            "joint_pos": _float_list(joint_pos, field="joint_pos", expected=count),
            "joint_vel": _float_list(joint_vel, field="joint_vel", expected=count),
        }
        command_values = _float_list(command, field="command", expected=3)
        root_pos_values = _float_list(root_pos, field="root_pos", expected=3)
        root_quat_values = _float_list(root_quat_wxyz, field="root_quat_wxyz", expected=4)
        root_lin_values = _float_list(root_lin_vel_b, field="root_lin_vel_b", expected=3)
        root_ang_values = _float_list(root_ang_vel_b, field="root_ang_vel_b", expected=3)

        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "control_step": int(control_step),
            "episode_id": self._episode_id,
            "episode_step": self._episode_step,
            "sim_time_s": float(control_step) * self.step_dt,
            "episode_time_s": float(self._episode_step) * self.step_dt,
            "wall_time_unix_ns": time.time_ns(),
            "reset_before_step": int(self._pending_reset),
            "terminated_after_step": int(bool(terminated_after_step)),
            "command.vx": command_values[0],
            "command.vy": command_values[1],
            "command.wz": command_values[2],
            "root_pos.x": root_pos_values[0],
            "root_pos.y": root_pos_values[1],
            "root_pos.z": root_pos_values[2],
            "root_quat.w": root_quat_values[0],
            "root_quat.x": root_quat_values[1],
            "root_quat.y": root_quat_values[2],
            "root_quat.z": root_quat_values[3],
            "root_lin_vel_b.x": root_lin_values[0],
            "root_lin_vel_b.y": root_lin_values[1],
            "root_lin_vel_b.z": root_lin_values[2],
            "root_ang_vel_b.x": root_ang_values[0],
            "root_ang_vel_b.y": root_ang_values[1],
            "root_ang_vel_b.z": root_ang_values[2],
        }
        for prefix, values in vectors.items():
            row.update({f"{prefix}.{name}": value for name, value in zip(self.joint_names, values)})
        self._writer.writerow(row)
        if parsed_student_tensors is not None:
            for name, tensor in parsed_student_tensors.items():
                self._student_tensor_records[name].append(tensor)
            self._student_tensor_indices["control_step"].append(int(control_step))
            self._student_tensor_indices["episode_id"].append(self._episode_id)
            self._student_tensor_indices["episode_step"].append(self._episode_step)
            self._student_tensor_indices["reset_before_step"].append(int(self._pending_reset))
        self._sample_count += 1
        self._episode_step += 1
        self._pending_reset = False
        if self._sample_count % self.flush_every == 0:
            self._csv_handle.flush()
            self._write_manifest(status="recording")

    def _write_student_tensors(self) -> None:
        if not self.record_student_tensors:
            return
        sample_count = len(self._student_tensor_indices["control_step"])
        if sample_count != self._sample_count:
            raise RuntimeError(
                f"student tensor/CSV sample count mismatch: tensors={sample_count}, csv={self._sample_count}"
            )
        payload: dict[str, torch.Tensor | dict[str, Any]] = {
            "schema_version": torch.tensor(SCHEMA_VERSION, dtype=torch.int64),
            "field_semantics": {
                "alignment": "row i is the pre-step neural input for row i of joint_trace.csv",
                "latent": "raw and hard-clamped deterministic estimator mu",
            },
        }
        for name, width in STUDENT_TENSOR_WIDTHS.items():
            records = self._student_tensor_records[name]
            payload[name] = (
                torch.stack(records, dim=0)
                if records
                else torch.empty((0, width), dtype=torch.float32)
            )
        for name, values in self._student_tensor_indices.items():
            payload[name] = torch.tensor(values, dtype=torch.int64)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.student_tensors_path.name}.", dir=self.output_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, self.student_tensors_path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self, reason: str = "normal") -> None:
        if self._closed:
            return
        self._csv_handle.flush()
        os.fsync(self._csv_handle.fileno())
        self._csv_handle.close()
        self._write_student_tensors()
        self._closed = True
        status = "completed" if reason == "normal" else "closed_usable_partial"
        self._write_manifest(status=status, close_reason=reason)


__all__ = ["PlayJointRecorder", "SCHEMA_VERSION", "STUDENT_TENSOR_WIDTHS"]
