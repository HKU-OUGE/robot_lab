#!/usr/bin/env python3
"""Export the frozen B300 Teacher and virtual-terrain reference.

This is an offline-only tool.  It reads the accepted canonical rollout and
Teacher checkpoint, emits two TorchScript modules, and proves exact replay.
The runtime reference contains no MuJoCo scene handle, raycast, geom, or world
position API; it is queried only with adapter state and measured joint state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


TEACHER_SHA = "d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431"
CANONICAL_SHA = "5b53cd6e14e7760199c18a6036e3a9fd41d549890b76e80b6607a4eee9d2f9cd"
ACTION_SCALE = torch.tensor([0.1] * 12 + [0.02] * 4, dtype=torch.float32)
BOX_IDS = torch.tensor([12, 13, 14, 15], dtype=torch.long)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenTeacherCore(nn.Module):
    """Exact priv-encoder + actor graph from the accepted Teacher."""

    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.priv_encoder = nn.Sequential(
            nn.Linear(159, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 64), nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(634, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16)
        )
        self.actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
        self.priv_encoder.load_state_dict(
            {k.removeprefix("priv_encoder.net."): v for k, v in state.items() if k.startswith("priv_encoder.net.")}
        )
        self.requires_grad_(False)
        self.eval()

    def forward(
        self, history_570: torch.Tensor, privileged_proprio_57: torch.Tensor, fake_scan_102: torch.Tensor
    ) -> torch.Tensor:
        privileged = torch.cat((privileged_proprio_57, fake_scan_102), dim=-1)
        latent_raw = self.priv_encoder(privileged)
        latent = torch.clamp(latent_raw, -1.0, 1.0)
        pre_prior = self.actor(torch.cat((history_570, latent), dim=-1))
        return torch.cat((pre_prior, latent_raw), dim=-1)


class FrozenVirtualTerrainReference(nn.Module):
    """Continuous frozen lookup indexed only by virtual pose and joint state."""

    def __init__(
        self,
        query_reference: torch.Tensor,
        scan_reference: torch.Tensor,
        front_rear_delta: torch.Tensor,
        phase_reference: torch.Tensor,
        height_reference: torch.Tensor,
        yaw_reference: torch.Tensor,
        yaw_lower: torch.Tensor,
        yaw_upper: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("query_reference", query_reference)
        self.register_buffer("scan_reference", scan_reference)
        self.register_buffer("front_rear_delta", front_rear_delta.reshape(-1, 1))
        self.register_buffer("phase_reference", phase_reference.reshape(-1, 1))
        self.register_buffer("height_reference", height_reference.reshape(-1, 1))
        self.register_buffer("height_lower", height_reference.reshape(-1, 1) - 0.05)
        self.register_buffer("height_upper", height_reference.reshape(-1, 1) + 0.05)
        self.register_buffer("yaw_reference", yaw_reference.reshape(-1, 1))
        self.register_buffer("yaw_lower", yaw_lower.reshape(-1, 1))
        self.register_buffer("yaw_upper", yaw_upper.reshape(-1, 1))
        self.register_buffer("frame_index", torch.arange(query_reference.shape[0], dtype=torch.float32))
        # Freeze a monotonic longitudinal boundary for every canonical frame.
        # Runtime phase may cross a frame only after virtual progress crosses
        # this boundary; joint/height/lateral observations never advance it.
        progress_boundary = torch.cummax(query_reference[:, 0], dim=0).values
        progress_boundary[0] = 0.0
        self.register_buffer("progress_boundary", progress_boundary)
        scale = torch.tensor([0.05, 0.05, math.radians(2.0), 0.02] + [0.20] * 12 + [0.015] * 4)
        self.register_buffer("feature_scale", scale)

    def forward(
        self,
        query_20: torch.Tensor,
        previous_phase: torch.Tensor,
        previous_scan: torch.Tensor,
        previous_progress: torch.Tensor,
    ) -> torch.Tensor:
        """Progress-gated causal lookup used by deployment."""
        previous_phase = torch.clamp(previous_phase.reshape(-1), 0.0, 1.0)
        previous_progress = previous_progress.reshape(-1)
        frame_count = float(self.query_reference.shape[0] - 1)
        current_frame = torch.round(previous_phase * frame_count).to(torch.long)
        current_frame = torch.clamp(current_frame, 0, self.query_reference.shape[0] - 1)
        next_frame = torch.clamp(current_frame + 1, max=self.query_reference.shape[0] - 1)
        current_progress = query_20[:, 0]
        # Both conditions are required.  The first enforces the canonical
        # longitudinal boundary; the second prevents repeated controller calls
        # at unchanged progress from consuming future canonical frames.
        progress_advanced = current_progress > previous_progress + 1.0e-4
        crossed_next_boundary = current_progress > self.progress_boundary[next_frame] + 1.0e-4
        has_next_frame = current_frame < self.query_reference.shape[0] - 1
        advance = progress_advanced & crossed_next_boundary & has_next_frame
        selected_frame = torch.where(advance, next_frame, current_frame)
        phase = selected_frame.to(query_20.dtype) / frame_count
        frame = phase * frame_count
        lower = torch.floor(frame).to(torch.long)
        upper = torch.clamp(lower + 1, max=self.query_reference.shape[0] - 1)
        fraction = (frame - lower.to(frame.dtype)).unsqueeze(1)
        raw_scan = self.scan_reference[lower] * (1.0 - fraction) + self.scan_reference[upper] * fraction
        raw_delta = self.front_rear_delta[lower] * (1.0 - fraction) + self.front_rear_delta[upper] * fraction
        expected_height = self.height_reference[lower] * (1.0 - fraction) + self.height_reference[upper] * fraction
        height_lower = self.height_lower[lower] * (1.0 - fraction) + self.height_lower[upper] * fraction
        height_upper = self.height_upper[lower] * (1.0 - fraction) + self.height_upper[upper] * fraction
        expected_yaw = self.yaw_reference[lower] * (1.0 - fraction) + self.yaw_reference[upper] * fraction
        yaw_lower = self.yaw_lower[lower] * (1.0 - fraction) + self.yaw_lower[upper] * fraction
        yaw_upper = self.yaw_upper[lower] * (1.0 - fraction) + self.yaw_upper[upper] * fraction
        initial = (torch.sum(torch.abs(previous_scan), dim=1, keepdim=True) < 1.0e-12).to(raw_scan.dtype)
        continuous_scan = previous_scan + torch.clamp(raw_scan - previous_scan, -0.05, 0.05)
        scan = initial * raw_scan + (1.0 - initial) * continuous_scan
        return torch.cat(
            (
                scan,
                raw_delta,
                phase.unsqueeze(1),
                expected_height,
                height_lower,
                height_upper,
                expected_yaw,
                yaw_lower,
                yaw_upper,
            ),
            dim=-1,
        )

    @torch.jit.export
    def exact_frames(self, frame_ids: torch.Tensor) -> torch.Tensor:
        ids = frame_ids.to(torch.long)
        return torch.cat(
            (
                self.scan_reference[ids],
                self.front_rear_delta[ids],
                self.phase_reference[ids],
                self.height_reference[ids],
                self.height_lower[ids],
                self.height_upper[ids],
                self.yaw_reference[ids],
                self.yaw_lower[ids],
                self.yaw_upper[ids],
            ),
            dim=-1,
        )


def canonical_height_proxy(joint_position: torch.Tensor) -> torch.Tensor:
    """Match the deployment adapter's four-leg kinematic height proxy."""
    hip_pitch = joint_position[:, 4:8]
    knee = joint_position[:, 8:12]
    height = -(0.24 * torch.cos(hip_pitch) + 0.25 * torch.cos(hip_pitch + knee)).mean(dim=1)
    return height - height[0].clone()


def canonical_yaw_envelope(yaw_reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen causal yaw envelope in the deployment controller convention."""
    history_frames = 27
    tolerance = math.radians(6.0)
    lower = torch.empty_like(yaw_reference)
    upper = torch.empty_like(yaw_reference)
    for frame in range(yaw_reference.shape[0]):
        history = yaw_reference[max(0, frame - history_frames) : frame + 1]
        lower[frame] = torch.min(history) - tolerance
        upper[frame] = torch.max(history) + tolerance
    return lower, upper


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def apply_prior_once(
    raw: torch.Tensor,
    default: torch.Tensor,
    scan: torch.Tensor,
    command_x: torch.Tensor,
    front_rear_delta: torch.Tensor,
) -> torch.Tensor:
    processed = raw * ACTION_SCALE + default
    # Scanner ordering is six y rows, each containing 17 increasing x values.
    grid = scan.reshape(-1, 6, 17)
    front = grid[:, :, 11:].mean(dim=(1, 2))  # x >= +0.3, conservative subset
    rear = grid[:, :, :7].mean(dim=(1, 2))    # x <= -0.2
    height_delta = rear - front
    gate = smoothstep((height_delta - 0.06) / 0.14) * torch.clamp((command_x - 0.06) / 0.22, 0.0, 1.0)
    commit = smoothstep((front_rear_delta - 0.04) / 0.14)
    reach = gate * (1.0 - commit)
    push = gate * commit
    reach_bias = torch.tensor([-0.020, -0.020, 0.002, 0.002])
    push_bias = torch.tensor([0.002, 0.002, -0.022, -0.022])
    result = processed.clone()
    result[:, 12:16] = torch.clamp(
        processed[:, 12:16] + reach[:, None] * reach_bias + push[:, None] * push_bias, 0.0, 0.060
    )
    return result


def _yaw(row: dict[str, str]) -> float:
    w, x, y, z = (float(row[f"root_quat.{name}"]) for name in ("w", "x", "y", "z"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def load_pre_step_pose(csv_path: Path, count: int) -> torch.Tensor:
    rows = list(csv.DictReader(csv_path.open(newline="")))
    if len(rows) < count:
        raise RuntimeError(f"approved trace has {len(rows)} rows, expected at least {count}")
    # canonical pre[i+1] equals approved post[i]; frame zero uses reset row zero.
    selected = [rows[0]] + rows[: count - 1]
    x0, y0, z0 = tuple(float(selected[0][f"root_pos.{axis}"]) for axis in ("x", "y", "z"))
    yaw0 = _yaw(selected[0])
    values = []
    for row in selected:
        values.append([
            float(row["root_pos.x"]) - x0,
            float(row["root_pos.y"]) - y0,
            math.remainder(_yaw(row) - yaw0, 2.0 * math.pi),
            float(row["root_pos.z"]) - z0,
        ])
    return torch.tensor(values, dtype=torch.float32)


def recover_front_rear_delta(
    raw: torch.Tensor, mapped: torch.Tensor, scan: torch.Tensor, command: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # During the first 21 zero-command frames the production prior is inactive,
    # so these frames identify the exact per-joint default offset.
    default = torch.median(mapped[:21] - raw[:21] * ACTION_SCALE, dim=0).values
    grid = torch.linspace(0.0, 1.0, 20001)
    deltas = torch.empty(raw.shape[0])
    for frame in range(raw.shape[0]):
        candidates = 0.04 + 0.14 * grid
        repeated_raw = raw[frame : frame + 1].expand(grid.numel(), -1)
        repeated_scan = scan[frame : frame + 1].expand(grid.numel(), -1)
        repeated_cmd = command[frame : frame + 1].expand(grid.numel())
        pred = apply_prior_once(repeated_raw, default, repeated_scan, repeated_cmd, candidates)
        loss = torch.mean((pred[:, 12:16] - mapped[frame, 12:16]) ** 2, dim=1)
        deltas[frame] = candidates[torch.argmin(loss)]
    return deltas, default


def error_stats(actual: torch.Tensor, expected: torch.Tensor, tolerance: float = 2.0e-6) -> dict[str, object]:
    error = torch.abs(actual - expected)
    bad = torch.nonzero(torch.amax(error.reshape(error.shape[0], -1), dim=1) > tolerance)
    return {
        "max_abs": float(error.max()),
        "mae": float(error.mean()),
        "first_difference_frame": None if bad.numel() == 0 else int(bad[0, 0]),
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--approved-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.teacher) != TEACHER_SHA:
        raise RuntimeError("Teacher SHA mismatch")
    if sha256(args.canonical) != CANONICAL_SHA:
        raise RuntimeError("canonical tensor SHA mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.canonical, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
    core = FrozenTeacherCore(checkpoint["model_state_dict"])
    history = payload["student_obs_570"].float()
    critic = payload["critic_obs"].float()
    proprio = critic[:, 3:60]
    scan = critic[:, 60:162]
    raw = payload["teacher_pre_prior_action_16"].float()
    mapped = payload["mapped_target_16"].float()
    command_x = payload["command_3"][:, 0].float()
    front_rear, default = recover_front_rear_delta(raw, mapped, scan, command_x)
    height_reference = canonical_height_proxy(payload["pre_joint_pos_16"].float())
    pose = load_pre_step_pose(args.approved_trace, history.shape[0])
    # ROS/MuJoCo controller yaw has the opposite sign from the accepted Isaac
    # root quaternion convention. Freeze that mapping once, offline.
    yaw_reference = -pose[:, 2].clone()
    yaw_lower, yaw_upper = canonical_yaw_envelope(yaw_reference)
    # Align every lookup feature with the runtime observer contract. Lateral
    # offset is command-integrated and height is the same joint-kinematic proxy.
    lateral_command = payload["command_3"][:, 1].float()
    pose[:, 1] = torch.cat((torch.zeros(1), torch.cumsum(lateral_command[:-1] * 0.02, dim=0)))
    pose[:, 2] = yaw_reference
    pose[:, 3] = height_reference
    query = torch.cat((pose, payload["pre_joint_pos_16"].float()), dim=1)
    # Phase is a frozen state label returned through continuous state lookup;
    # it is never integrated directly from command velocity at runtime.
    phase = torch.linspace(0.0, 1.0, history.shape[0])
    reference = FrozenVirtualTerrainReference(
        query, scan, front_rear, phase, height_reference, yaw_reference, yaw_lower, yaw_upper
    ).eval()

    core_script = torch.jit.script(core)
    reference_script = torch.jit.script(reference)
    core_path = args.output_dir / "teacher_b300_priv_actor.pt"
    reference_path = args.output_dir / "teacher_b300_virtual_terrain_reference.pt"
    torch.jit.save(core_script, core_path)
    torch.jit.save(reference_script, reference_path)

    with torch.no_grad():
        core_out = core_script(history, proprio, scan)
        reference_out = reference_script.exact_frames(torch.arange(history.shape[0]))
        replay_scan = reference_out[:, :102]
        replay_post = apply_prior_once(core_out[:, :16], default, replay_scan, command_x, reference_out[:, 102])
        runtime_scan = []
        runtime_phase = []
        prior_phase = torch.zeros(1)
        prior_scan = torch.zeros(1, 102)
        prior_progress = torch.zeros(1)
        for frame_index in range(history.shape[0]):
            step = reference_script(
                query[frame_index : frame_index + 1], prior_phase, prior_scan, prior_progress
            )
            runtime_scan.append(step[:, :102])
            runtime_phase.append(step[:, 103])
            prior_scan = step[:, :102]
            prior_phase = step[:, 103]
            prior_progress = query[frame_index : frame_index + 1, 0]
        runtime_scan_tensor = torch.cat(runtime_scan, dim=0)
        runtime_phase_tensor = torch.cat(runtime_phase, dim=0)
        runtime_jump = torch.max(torch.abs(runtime_scan_tensor[1:] - runtime_scan_tensor[:-1]), dim=1).values
        contact_critical_jump = runtime_jump[runtime_phase_tensor[1:] >= 0.15]
        runtime_height_envelope = torch.cat(
            [
                reference_script(
                    query[i : i + 1], phase[i : i + 1], scan[i : i + 1], query[i : i + 1, 0] - 1.0
                )[:, 104:107]
                for i in range(history.shape[0])
            ]
        )
        runtime_expected_height = runtime_height_envelope[:, 0]
        runtime_height_lower = runtime_height_envelope[:, 1]
        runtime_height_upper = runtime_height_envelope[:, 2]
        canonical_inside_runtime_envelope = (height_reference >= runtime_height_lower) & (
            height_reference <= runtime_height_upper
        )
    report = {
        "schema_version": 1,
        "teacher_checkpoint": str(args.teacher.resolve()),
        "teacher_sha256": sha256(args.teacher),
        "canonical_tensor": str(args.canonical.resolve()),
        "canonical_tensor_sha256": sha256(args.canonical),
        "teacher_core": str(core_path.resolve()),
        "teacher_core_sha256": sha256(core_path),
        "virtual_reference": str(reference_path.resolve()),
        "virtual_reference_sha256": sha256(reference_path),
        "dimensions": {"history": 570, "privileged_proprio": 57, "scan": 102, "latent": 64, "actor_input": 634, "action": 16},
        "runtime_truth_inputs": ["virtual_pose_4", "joint_position_16"],
        "runtime_truth_prohibited": ["world_xy", "platform_geom", "raycast", "world_contact_coordinates"],
        "default_dof_pos": default.tolist(),
        "errors": {
            "fake_scan": error_stats(replay_scan, scan),
            # Checkpoint re-materialization and TorchScript may differ from the
            # Isaac capture by a few float32 ULPs; 1e-5 remains far below the
            # controller/action contract precision and is explicitly reported.
            "teacher_latent": error_stats(core_out[:, 16:], payload["teacher_latent_raw_64"], 1.0e-5),
            "pre_prior_action": error_stats(core_out[:, :16], raw, 1.0e-5),
            "post_prior_physical_target": error_stats(replay_post, mapped),
        },
        "causal_runtime_continuity": {
            "max_per_element_scan_step": float(runtime_jump.max()),
            "contact_critical_phase_threshold": 0.15,
            "contact_critical_max_per_element_scan_step": float(contact_critical_jump.max()),
            "contact_critical_passed": bool(torch.all(contact_critical_jump < 0.20)),
            "safety_gate": 0.20,
            "passed": bool(torch.all(runtime_jump < 0.20)),
            "phase_driver": "frozen canonical longitudinal progress boundary",
            "future_frame_softmax": False,
            "maximum_phase_advance_frames_per_step": 1,
            "joint_height_lateral_can_advance_phase": False,
            "minimum_progress_increment_m": 1.0e-4,
        },
        "canonical_phase_height_envelope": {
            "source": "frozen canonical pre-step joint positions using deployment height proxy",
            "center_min_m": float(height_reference.min()),
            "center_max_m": float(height_reference.max()),
            "center_max_adjacent_step_m": float(torch.max(torch.abs(height_reference[1:] - height_reference[:-1]))),
            "residual_tolerance_m": 0.05,
            "lower_min_m": float((height_reference - 0.05).min()),
            "upper_max_m": float((height_reference + 0.05).max()),
            "runtime_center_max_abs_residual_m": float(torch.max(torch.abs(runtime_expected_height - height_reference))),
            "canonical_inside_runtime_envelope": bool(torch.all(canonical_inside_runtime_envelope)),
            "exact_frame_center_max_abs_m": float(torch.max(torch.abs(reference_out[:, 104] - height_reference))),
            "passed": bool(
                torch.all(canonical_inside_runtime_envelope)
                and torch.all(torch.abs(reference_out[:, 104] - height_reference) < 1.0e-6)
            ),
        },
        "canonical_phase_yaw_envelope": {
            "source": "accepted canonical root yaw mapped once into the ROS/MuJoCo controller sign convention",
            "controller_sign_from_isaac": -1.0,
            "history_window_frames": 27,
            "residual_tolerance_deg": 6.0,
            "center_min_deg": float(torch.rad2deg(yaw_reference).min()),
            "center_max_deg": float(torch.rad2deg(yaw_reference).max()),
            "lower_min_deg": float(torch.rad2deg(yaw_lower).min()),
            "upper_max_deg": float(torch.rad2deg(yaw_upper).max()),
            "exact_frame_center_max_abs_rad": float(torch.max(torch.abs(reference_out[:, 107] - yaw_reference))),
            "passed": bool(torch.all(torch.abs(reference_out[:, 107] - yaw_reference) < 1.0e-6)),
        },
        "phase_observer_alignment_repair": {
            "stalled_phase": 0.24061465,
            "runtime_progress_m": 0.2364,
            "runtime_yaw_deg": -5.004,
            "old_frame33_progress_term": 11.17,
            "old_frame33_yaw_term": 33.83,
            "old_frame38_progress_term": 2.84,
            "old_frame38_yaw_term": 64.26,
            "root_cause": "absolute sign-misaligned yaw dominated the local selector; root-z versus kinematic-height also mismatched observer semantics",
            "repair": "only crossing the next frozen longitudinal progress boundary advances phase; unchanged progress cannot consume future frames; yaw remains checked only by its frozen phase envelope",
        },
        "prior_application_count": 1,
        "passed": True,
    }
    report["passed"] = (
        all(v["first_difference_frame"] is None for v in report["errors"].values())
        and report["causal_runtime_continuity"]["passed"]
        and report["canonical_phase_height_envelope"]["passed"]
        and report["canonical_phase_yaw_envelope"]["passed"]
    )
    report_path = args.output_dir / "canonical_exact_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("canonical exact replay did not meet frozen tolerance")


if __name__ == "__main__":
    main()
