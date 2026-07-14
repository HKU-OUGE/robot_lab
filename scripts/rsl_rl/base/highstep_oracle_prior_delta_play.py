#!/usr/bin/env python3
"""Run canonical play with a read-only same-state Teacher prior-delta oracle.

All ordinary CLI arguments are parsed by ``play.py``.  The diagnostic mode and
evidence directory are supplied through the immutable v1.9 supervisor only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PLAY_PATH = Path(__file__).with_name("play.py")
TEACHER = ROOT / "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt"
TEACHER_SHA256 = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_play() -> Any:
    spec = importlib.util.spec_from_file_location("highstep_v19_bound_play", PLAY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical play module: {PLAY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _OracleController:
    def __init__(self, runner: Any, play: Any, device: str | None):
        import torch

        sys.path.insert(0, str(ROOT))
        from tools import highstep_oracle_prior_delta as contract
        from tools import highstep_same_state_audit as audit

        self.torch = torch
        self.contract = contract
        self.audit = audit
        self.mode = os.environ.get("HIGHSTEP_ORACLE_PRIOR_DELTA_MODE", "").strip()
        if self.mode not in contract.MODES:
            raise RuntimeError(f"invalid HIGHSTEP_ORACLE_PRIOR_DELTA_MODE={self.mode!r}")
        evidence_root = Path(os.environ["HIGHSTEP_ORACLE_PRIOR_DELTA_EVIDENCE_ROOT"]).resolve()
        prereg = Path(os.environ["HIGHSTEP_ORACLE_PRIOR_DELTA_PREREGISTRATION"]).resolve(strict=True)
        expected_prereg_sha = os.environ["HIGHSTEP_ORACLE_PRIOR_DELTA_PREREGISTRATION_SHA256"]
        if _sha256(prereg) != expected_prereg_sha:
            raise RuntimeError("oracle prior-delta preregistration SHA mismatch")
        prereg_payload = json.loads(prereg.read_text())
        spec_path = Path(prereg_payload["spec_path"]).resolve(strict=True)
        if _sha256(spec_path) != prereg_payload["spec_sha256"]:
            raise RuntimeError("oracle prior-delta spec SHA mismatch")

        checkpoint = Path(play.args_cli.checkpoint).expanduser().resolve(strict=True)
        if str(checkpoint) != prereg_payload["student_checkpoint"] or _sha256(checkpoint) != prereg_payload["student_checkpoint_sha256"]:
            raise RuntimeError("oracle prior-delta Student checkpoint binding mismatch")
        if _sha256(TEACHER) != TEACHER_SHA256 or prereg_payload["teacher_checkpoint_sha256"] != TEACHER_SHA256:
            raise RuntimeError("oracle prior-delta Teacher checkpoint binding mismatch")

        self.checkpoint = checkpoint
        self.checkpoint_sha_before = _sha256(checkpoint)
        self.teacher_checkpoint_sha_before = _sha256(TEACHER)
        teacher = torch.load(TEACHER, map_location="cpu", weights_only=False)
        state = teacher.get("model_state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("Teacher checkpoint lacks model_state_dict")
        actor_state = audit._component_state(state, "actor.")
        priv_state = audit._component_state(state, "priv_encoder.net.")
        self.teacher_actor = audit.CheckpointMLP(actor_state)
        self.teacher_priv = audit.CheckpointMLP(priv_state, final_tanh=True)
        target_device = torch.device(device or runner.env.unwrapped.device)
        self.teacher_actor.to(target_device).eval()
        self.teacher_priv.to(target_device).eval()
        for parameter in list(self.teacher_actor.parameters()) + list(self.teacher_priv.parameters()):
            parameter.requires_grad_(False)
        self.teacher_actor_checkpoint_component_sha256 = audit._component_sha256(actor_state)
        self.teacher_priv_checkpoint_component_sha256 = audit._component_sha256(priv_state)
        # Compare loaded modules with themselves using identical state-dict key
        # semantics.  The checkpoint component keys omit the CheckpointMLP
        # wrapper's ``net.`` prefix and are therefore a separate lineage hash,
        # not a valid before/after mutation digest.
        self.teacher_actor_sha_before = audit._component_sha256(
            {name: value.detach().cpu() for name, value in self.teacher_actor.state_dict().items()}
        )
        self.teacher_priv_sha_before = audit._component_sha256(
            {name: value.detach().cpu() for name, value in self.teacher_priv.state_dict().items()}
        )
        self.post_contract = audit.PostPriorContract.from_teacher_env_config()
        self.post_function = audit._resolve_shared_post_prior()
        self.adapter = audit.HighstepRuntimeAdapter(runner.env, contract=self.post_contract)

        seed = int(play.args_cli.seed)
        scenario = contract.scenario_from_offsets(
            float(play.args_cli.front_step_eval_lateral_offset),
            float(play.args_cli.front_step_eval_yaw_offset_deg),
        )
        self.identity = {"mode": self.mode, "seed": seed, "scenario": scenario}
        self.output_dir = evidence_root / self.mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed{seed}_{scenario}"
        self.frames_path = self.output_dir / f"{stem}.frames.jsonl"
        self.summary_path = self.output_dir / f"{stem}.summary.json"
        if self.frames_path.exists() or self.summary_path.exists():
            raise FileExistsError(f"refusing to overwrite oracle evidence for {stem}")
        self.frames = 0
        self.nonbox_max_abs = 0.0
        self.delta_abs_max = 0.0
        self.delta_abs_sum = 0.0
        self.delta_values = 0
        self.preregistration = str(prereg)
        self.preregistration_sha256 = expected_prereg_sha

    def apply(self, observations: Any, student_action: Any) -> Any:
        torch = self.torch
        state_before = self.adapter.physical_state_digest()
        sample = self.adapter.sample(observations)
        critic = sample.critic_observations.to(next(self.teacher_priv.parameters()).device)
        policy = sample.policy_observations.to(next(self.teacher_actor.parameters()).device)
        teacher_latent = self.teacher_priv(critic[:, 3:])
        teacher_pre = self.teacher_actor(torch.cat((policy, teacher_latent), dim=-1))
        teacher_post = self.audit._call_shared_post_prior(
            self.post_function, teacher_pre, sample.prior_inputs, self.post_contract
        ).raw_equivalent_actions
        applied, delta4 = self.contract.apply_oracle_prior_delta(
            student_action, teacher_pre, teacher_post, mode=self.mode
        )
        state_after = self.adapter.physical_state_digest()
        if state_before != state_after:
            raise RuntimeError("oracle Teacher forward mutated simulator state")
        nonbox_max = float(torch.max(torch.abs(applied[:, :12] - student_action[:, :12])).item())
        self.nonbox_max_abs = max(self.nonbox_max_abs, nonbox_max)
        self.delta_abs_max = max(self.delta_abs_max, float(torch.max(torch.abs(delta4)).item()))
        self.delta_abs_sum += float(torch.sum(torch.abs(delta4)).item())
        self.delta_values += int(delta4.numel())
        record = {
            **self.identity,
            "step": self.frames,
            "student_action_pre_injection": student_action[0].detach().cpu().tolist(),
            "teacher_action_pre_prior": teacher_pre[0].detach().cpu().tolist(),
            "teacher_action_post_prior": teacher_post[0].detach().cpu().tolist(),
            "teacher_prior_delta_box4": delta4[0].detach().cpu().tolist(),
            "action_applied": applied[0].detach().cpu().tolist(),
            "protected_nonbox_max_abs_change": nonbox_max,
            "physical_state_digest_before": state_before,
            "physical_state_digest_after": state_after,
            "physical_state_unchanged": True,
        }
        with self.frames_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.frames += 1
        return applied

    def finalize(self) -> dict[str, Any]:
        actor_state = {name: value.detach().cpu() for name, value in self.teacher_actor.state_dict().items()}
        priv_state = {name: value.detach().cpu() for name, value in self.teacher_priv.state_dict().items()}
        payload = {
            "schema_version": 1,
            "kind": "highstep_v19_oracle_prior_delta_runtime_evidence",
            **self.identity,
            "student_checkpoint": str(self.checkpoint),
            "student_checkpoint_sha256_before": self.checkpoint_sha_before,
            "student_checkpoint_sha256_after": _sha256(self.checkpoint),
            "teacher_checkpoint": str(TEACHER),
            "teacher_checkpoint_sha256_before": self.teacher_checkpoint_sha_before,
            "teacher_checkpoint_sha256_after": _sha256(TEACHER),
            "teacher_actor_checkpoint_component_sha256": self.teacher_actor_checkpoint_component_sha256,
            "teacher_privileged_encoder_checkpoint_component_sha256": self.teacher_priv_checkpoint_component_sha256,
            "teacher_actor_tensor_sha256_before": self.teacher_actor_sha_before,
            "teacher_actor_tensor_sha256_after": self.audit._component_sha256(actor_state),
            "teacher_privileged_encoder_tensor_sha256_before": self.teacher_priv_sha_before,
            "teacher_privileged_encoder_tensor_sha256_after": self.audit._component_sha256(priv_state),
            "frame_count": self.frames,
            "protected_nonbox_max_abs_change": self.nonbox_max_abs,
            "teacher_prior_delta_box4_abs_max": self.delta_abs_max,
            "teacher_prior_delta_box4_abs_mean": self.delta_abs_sum / max(self.delta_values, 1),
            "optimizer_step_calls": 0,
            "parameter_write_calls": 0,
            "preregistration": self.preregistration,
            "preregistration_sha256": self.preregistration_sha256,
            "frames": str(self.frames_path),
            "frames_sha256": _sha256(self.frames_path),
        }
        checks = (
            payload["student_checkpoint_sha256_before"] == payload["student_checkpoint_sha256_after"],
            payload["teacher_checkpoint_sha256_before"] == payload["teacher_checkpoint_sha256_after"],
            payload["teacher_actor_tensor_sha256_before"] == payload["teacher_actor_tensor_sha256_after"],
            payload["teacher_privileged_encoder_tensor_sha256_before"] == payload["teacher_privileged_encoder_tensor_sha256_after"],
            payload["protected_nonbox_max_abs_change"] == 0.0,
            payload["frame_count"] > 0,
        )
        payload["all_read_only_and_action_contract_checks_passed"] = all(checks)
        if not payload["all_read_only_and_action_contract_checks_passed"]:
            raise RuntimeError(f"oracle prior-delta runtime evidence failed: {payload}")
        self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload


def main() -> int:
    play = _load_play()
    original = play.OnPolicyRunner.get_inference_policy
    controllers: list[_OracleController] = []

    def patched(runner: Any, device: str | None = None):
        base_policy = original(runner, device=device)
        controller = _OracleController(runner, play, device)
        controllers.append(controller)

        def inference(observations: Any):
            action = base_policy(observations)
            return controller.apply(observations, action)

        return inference

    play.OnPolicyRunner.get_inference_policy = patched
    try:
        play.main()
        if len(controllers) != 1:
            raise RuntimeError(f"expected exactly one oracle controller, got {len(controllers)}")
        summary = controllers[0].finalize()
        print("[HIGHSTEP_ORACLE_PRIOR_DELTA_JSON] " + json.dumps(summary, sort_keys=True), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        play.simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
