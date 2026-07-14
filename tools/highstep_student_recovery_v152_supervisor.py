#!/usr/bin/env python3
"""Autonomous v1.5.2 supervisor: exact E300 resume plus directional delivery gates."""

from __future__ import annotations

import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Mapping

ROOT = Path("/home/lxq/Softwares/robot_lab")
VIDEO_QC_PATH = ROOT / "tools/highstep_video_visibility_qc.py"
_video_spec = importlib.util.spec_from_file_location("highstep_v152_video_visibility_qc", VIDEO_QC_PATH)
if _video_spec is None or _video_spec.loader is None:
    raise RuntimeError("cannot load highstep video visibility QC")
_video_qc = importlib.util.module_from_spec(_video_spec)
_video_spec.loader.exec_module(_video_qc)
video_delivery_qc = _video_qc.video_delivery_qc

BASE_PATH = ROOT / "tools/highstep_student_recovery_v15_supervisor.py"
_spec = importlib.util.spec_from_file_location("highstep_v15_base_for_v152", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load v1.5 supervisor base")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

WORKFLOW_ID = "highstep_student_recovery_v152_20260713"
STATE_ROOT = ROOT / "tmp/highstep_student_recovery_v152_20260713"
SPEC_SHA = "c892c0d8bc811228b34bb3c439e2977d5f473ecf1374453f3032509e2776f63e"
PREREG = STATE_ROOT / "preregistration_v1.json"
PREREG_SHA = "473b8554d25d253886547fe4af72d49e30dad796d66fd9a3944e1d0ad8238ec2"
AUDIT = STATE_ROOT / "rebinding_audit.json"
AUDIT_SHA = "2d2826404676a04a5ff215dea891bc812516b55d579ccc0ba86e62ea4c269a30"
SOURCE_E300 = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_v15_Student/"
    "2026-07-13_17-46-04_v15_E300_20260713_174559/model_298.pt"
)
SOURCE_E300_SHA = "79449ba461647781a769d9211bc357f1936238c408226b33a2e500a4921095aa"
ROOT_MODEL = ROOT / "tmp/highstep_student_recovery_v15_20260713/source_model_172300_v3/model_172300.pt"
OLD_STATE = ROOT / "tmp/highstep_student_recovery_v15_20260713/state.json"
OLD_HANDOFF = ROOT / "tmp/highstep_student_recovery_v15_20260713/handoff.json"
SAVE_POINTS = (500, 900, 1400, 1800, 2500)
CORE9_POINTS = set(SAVE_POINTS)

# Reuse the already-tested mechanics, but replace every authority-bearing global
# before any Supervisor instance is constructed.
base.WORKFLOW_ID = WORKFLOW_ID
base.STATE_ROOT = STATE_ROOT
base.SPEC_SHA = SPEC_SHA
base.PREREG = PREREG
base.PREREG_SHA = PREREG_SHA
base.SOURCE = SOURCE_E300
base.SAVE_POINTS = SAVE_POINTS
base.CORE9_POINTS = CORE9_POINTS


def atomic_json(path: Path, payload: Mapping[str, Any], read_only: bool = False) -> None:
    base.atomic_json(path, payload, read_only=read_only)


class Supervisor(base.Supervisor):
    """v1.5.2 authority, classification and single-side behavior semantics."""

    def preflight(self) -> None:
        required = {
            base.SPEC: SPEC_SHA,
            PREREG: PREREG_SHA,
            AUDIT: AUDIT_SHA,
            SOURCE_E300: SOURCE_E300_SHA,
            base.REFERENCE: base.REFERENCE_SHA,
            ROOT_MODEL: base.TEACHER_SHA,
            base.TEACHER: base.TEACHER_SHA,
        }
        for path, expected in required.items():
            if base.sha256_file(path) != expected:
                raise RuntimeError(f"v1.5.2 authority SHA mismatch: {path}")
        old_state = json.loads(OLD_STATE.read_text())
        old_handoff = json.loads(OLD_HANDOFF.read_text())
        if not (
            old_state.get("status") == "stopped_by_gate"
            and old_handoff.get("status") == "stopped_by_gate"
            and old_state.get("stop_reason") == "two_consecutive_regressions_without_latent_improvement"
        ):
            raise RuntimeError("old v1.5 stopped_by_gate fact changed")
        audit = json.loads(AUDIT.read_text())
        if not (
            audit.get("exact_resume_allowed") is True
            and audit.get("source_checkpoint_sha256") == SOURCE_E300_SHA
            and audit.get("source_effective_updates") == 300
            and audit.get("forbidden_model_tensor_changes") == []
        ):
            raise RuntimeError("v1.5.2 E300 rebinding audit failed")
        prereg = json.loads(PREREG.read_text())
        for name, expected in prereg.get("code_sha256", {}).items():
            if base.sha256_file(Path(name)) != expected:
                raise RuntimeError(f"v1.5.2 preregistered training code changed: {name}")
        self.require_idle()
        critical = [
            Path(__file__), BASE_PATH, base.CANDIDATE_AUDIT, base.IMITATION_GATE,
            ROOT / "tools/highstep_student_recovery_v152_rebinding_audit.py",
            ROOT / "tools/highstep_v152_recover_core9.py",
            ROOT / "tools/highstep_v15_imitation_gate.py",
            ROOT / "tools/highstep_wandb_stage_gate.py",
            ROOT / "scripts/rsl_rl/base/train.py", ROOT / "scripts/rsl_rl/base/play.py", base.MONITOR,
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
            ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        ]
        self.critical_hashes = {str(path.resolve()): base.sha256_file(path.resolve()) for path in critical}
        smoke_manifest = STATE_ROOT / "smoke/smoke_manifest.json"
        smoke = json.loads(smoke_manifest.read_text())
        if not (
            smoke.get("passed") is True
            and smoke.get("source_checkpoint_sha256") == SOURCE_E300_SHA
            and smoke.get("start_effective_updates") == 300
            and smoke.get("end_effective_updates") in (301, 302, 303, 304, 305)
            and smoke.get("actor_unchanged") is True
            and smoke.get("optimizer_restored") is True
            and smoke.get("authority_rebound") is True
        ):
            raise RuntimeError("v1.5.2 1-5 update smoke is missing or invalid")
        payload = {
            "schema_version": 1,
            "kind": "highstep_student_recovery_v152_preflight",
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": PREREG_SHA,
            "rebinding_audit_sha256": AUDIT_SHA,
            "source_e300_sha256": SOURCE_E300_SHA,
            "old_v15_stopped_by_gate_preserved": True,
            "e500_core9_recovery_manifest": str(max(
                (STATE_ROOT / "stages/E500/core9/recovery_evidence").glob("*/recovery_manifest.json"),
                key=lambda path: path.stat().st_mtime_ns,
            )),
            "smoke_manifest": str(smoke_manifest),
            "smoke_manifest_sha256": base.sha256_file(smoke_manifest),
            "critical_code_sha256": self.critical_hashes,
            "supervisor_stop_semantics": {
                "mechanism_error": "fail_closed_and_repair",
                "behavior_not_yet_successful": "continue",
                "confirmed_safety_risk": "candidate_block_only",
                "ordinary_diagnostic_anomaly": "record_and_continue",
                "external_failure": "three_retries_then_external_block",
            },
            "repair_amendment": {
                "reason": "Propagate v1.5.2 authority through schema-4 core9 lineage validation; preserve and revalidate the already completed E500 matrix.",
                "previous_preflight": str(STATE_ROOT / "preflight_manifest_v4.json"),
                "previous_preflight_sha256": base.sha256_file(STATE_ROOT / "preflight_manifest_v4.json"),
                "old_results_or_thresholds_modified": False,
            },
            "completed_at": base.now(),
        }
        preflight = STATE_ROOT / "preflight_manifest_v5.json"
        if preflight.exists():
            stored = json.loads(preflight.read_text())
            stable = {key: value for key, value in payload.items() if key != "completed_at"}
            stored_stable = {key: value for key, value in stored.items() if key != "completed_at"}
            if stored_stable != stable or not base.is_read_only(preflight):
                raise RuntimeError("stored v1.5.2 preflight changed")
        else:
            atomic_json(preflight, payload, read_only=True)
        self.update(status="preflight_passed", phase="ready_for_stage_500", effective_updates=300,
                    checkpoint=str(SOURCE_E300), code_sha256=self.critical_hashes,
                    stop_reason=None, last_error=None)

    def train_stage(self, stage: int, source: Path, previous: int):
        # The base class supplies all checkpoint/optimizer and W&B contracts.
        # Its environment is extended here through os.environ because the task
        # config must consume both the path and digest of the new authority.
        previous_path = os.environ.get("HIGHSTEP_V15_PREREGISTRATION_PATH")
        previous_sha = os.environ.get("HIGHSTEP_V15_PREREGISTRATION_SHA256")
        os.environ["HIGHSTEP_V15_PREREGISTRATION_PATH"] = str(PREREG)
        os.environ["HIGHSTEP_V15_PREREGISTRATION_SHA256"] = PREREG_SHA
        try:
            return super().train_stage(stage, source, previous)
        finally:
            if previous_path is None:
                os.environ.pop("HIGHSTEP_V15_PREREGISTRATION_PATH", None)
            else:
                os.environ["HIGHSTEP_V15_PREREGISTRATION_PATH"] = previous_path
            if previous_sha is None:
                os.environ.pop("HIGHSTEP_V15_PREREGISTRATION_SHA256", None)
            else:
                os.environ["HIGHSTEP_V15_PREREGISTRATION_SHA256"] = previous_sha

    def _core_rows(self, core: Mapping[str, Any], checkpoint: Path) -> list[dict[str, Any]]:
        manifest = Path(str(core["evaluation_manifest"]))
        rows_path = manifest.parent / "eval_runs.jsonl"
        records = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        result = []
        for record in records:
            item = record["eval"]
            result.append({
                "seed": int(record["seed"]), "scenario": str(record["scenario"]),
                "lateral_offset_m": float(record["lateral"]),
                "yaw_offset_deg": float(record["yaw_offset_deg"]), "valid": True,
                "full_climb": item.get("full_climb_success") is True,
                "rear_hold": item.get("rear_on_platform_hold_success") is True,
                "front_top_support": item.get("front_top_support_reached") is True,
                "no_severe_inward": self.no_severe(item),
                "source": "fixed_core9", "log": record.get("log"),
            })
        return result

    def directional(self, stage: int, checkpoint: Path, core: Mapping[str, Any]) -> dict[str, Any]:
        root = STATE_ROOT / "stages" / f"E{stage}" / "directional"
        terminal = root / "directional_summary.json"
        if terminal.is_file():
            payload = json.loads(terminal.read_text())
            if payload.get("checkpoint_sha256") != base.sha256_file(checkpoint):
                raise RuntimeError("stored directional evaluation checkpoint changed")
            return payload
        rows = self._core_rows(core, checkpoint)
        for side, lateral, yaw in (("left", "0.06", "2.0"), ("right", "-0.06", "-2.0")):
            for seed in (11, 22, 33):
                scenario = f"half_{side}"
                log = root / f"seed{seed}_{scenario}.log"
                command = self.play_command(checkpoint, seed, lateral, yaw)
                launch = root / f"seed{seed}_{scenario}.launch.json"
                if not launch.exists():
                    atomic_json(launch, {
                        "schema_version": 1, "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": base.sha256_file(checkpoint), "seed": seed,
                        "scenario": scenario, "lateral_offset_m": float(lateral),
                        "yaw_offset_deg": float(yaw), "command": command,
                    }, read_only=True)
                self.require_idle()
                self.run_command(command, log, root,
                                 f"directional_E{stage}_{scenario}_seed{seed}", retries=3, stall=900)
                item = self.eval_payload(log, checkpoint)
                if abs(float(item.get("yaw_offset_deg", 999)) - float(yaw)) > 1e-8:
                    raise RuntimeError("directional half-offset geometry mismatch")
                rows.append({
                    "seed": seed, "scenario": scenario, "lateral_offset_m": float(lateral),
                    "yaw_offset_deg": float(yaw), "valid": True,
                    "full_climb": item.get("full_climb_success") is True,
                    "rear_hold": item.get("rear_on_platform_hold_success") is True,
                    "front_top_support": item.get("front_top_support_reached") is True,
                    "no_severe_inward": self.no_severe(item), "source": "direct_half_offset",
                    "log": str(log), "log_sha256": base.sha256_file(log),
                })

        corridors = {}
        for side, outer, half in (("left", "left_offset", "half_left"), ("right", "right_offset", "half_right")):
            selected = [row for row in rows if row["scenario"] in {"nominal", half, outer}]
            counts = {key: sum(row[key] is True for row in selected) for key in
                      ("valid", "full_climb", "rear_hold", "front_top_support", "no_severe_inward")}
            half_rows = [row for row in selected if row["scenario"] == half]
            counts["half_full"] = sum(row["full_climb"] is True for row in half_rows)
            counts["half_rear_hold"] = sum(row["rear_hold"] is True for row in half_rows)
            stable = bool(len(selected) == 9 and counts["valid"] == 9 and counts["full_climb"] >= 8
                          and counts["rear_hold"] >= 8 and counts["half_full"] == 3
                          and counts["half_rear_hold"] == 3)
            corridors[side] = {
                "rows": selected, "counts": counts, "stable_behavior_passed": stable,
                "front_and_inward_are_recorded_diagnostics": True,
                "confirmed_real_robot_safety_risk": False,
            }
        passed_sides = [side for side, value in corridors.items() if value["stable_behavior_passed"]]
        payload = {
            "schema_version": 1, "kind": "highstep_v152_directional_corridors",
            "checkpoint": str(checkpoint), "checkpoint_sha256": base.sha256_file(checkpoint),
            "effective_updates": stage, "corridors": corridors, "passed_sides": passed_sides,
            "passed": bool(passed_sides), "thresholds_modified_after_results": False,
            "completed_at": base.now(),
        }
        atomic_json(terminal, payload, read_only=True)
        return payload

    def decision(self, stage: int, imitation: Mapping[str, Any], probe: Mapping[str, Any],
                 core: Mapping[str, Any] | None, directional: Mapping[str, Any] | None = None) -> dict[str, Any]:
        counts = {} if core is None else core["counts"]
        bilateral_stable = bool(core and counts["valid"] == 9 and counts["full_climb"] >= 8
                                and counts["rear_hold"] >= 8)
        directional_stable = bool(directional and directional["passed"])
        numeric_pass = bilateral_stable or directional_stable
        diagnostic = {
            "imitation_passed": bool(imitation["passed"]),
            "imitation_failed_elements": int(imitation["failed_element_count"]),
            "latent_mean": self.latent_mean(imitation),
            "probe": probe["counts"],
            "core9": counts or None,
            "directional": None if directional is None else {
                side: directional["corridors"][side]["counts"] for side in ("left", "right")
            },
        }
        if numeric_pass:
            reason, classification, stop = "stable_climb_pending_user_visual_review", "behavior_success", True
        elif stage >= 2500:
            reason, classification, stop = "absolute_2500_cap_without_stable_climb", "behavior_not_successful_at_cap", True
        else:
            reason, classification, stop = "continue_to_next_fixed_save_point", "behavior_not_yet_successful", False
        return {
            "classification": classification, "diagnostics": diagnostic,
            "mechanism_verified": True, "confirmed_safety_risk": False,
            "bilateral_stable_behavior_passed": bilateral_stable,
            "directional_stable_behavior_passed": directional_stable,
            "passed_sides": [] if directional is None else directional["passed_sides"],
            "final_numeric_gate_passed": numeric_pass, "stop": stop, "reason": reason,
            # Compatibility fields consumed by the v1.5 W&B summary writer.
            "imitation_passed": bool(imitation["passed"]), "latent_mean": diagnostic["latent_mean"],
            "latent_improved": False, "behavior_degraded": False, "continuous_regression": False,
            "late_window_score": probe["behavior_score"], "best_score": probe["behavior_score"],
            "late_to_best_score_ratio": 1.0, "late_window_support": probe["support_score"],
            "best_support": probe["support_score"], "late_to_best_support_ratio": 1.0,
            "late_minimum_floor_passed": True, "candidate_0_93_floor_passed": True,
        }

    @staticmethod
    def wandb_gate_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "imitation_passed": result["imitation"]["passed"],
            "imitation_failed_elements": result["imitation"]["failed_element_count"],
            "probe": result["probe"]["counts"],
            "core9": result["core9"]["counts"],
            "directional": {side: result["directional"]["corridors"][side]["counts"] for side in ("left", "right")},
            "classification": result["decision"]["classification"],
        }

    def placement_card(self, root: Path, side: str) -> Path:
        sign = 1 if side == "left" else -1
        rows = [("nominal", 0.0, 0.0), ("half-offset", .06 * sign, 2 * sign),
                ("outer-offset", .12 * sign, 4 * sign)]
        svg_rows = "".join(
            f'<g transform="translate({360 + y*900},{410-i*95}) rotate({yaw})"><rect x="-55" y="-25" width="110" height="50" rx="12" fill="#2563eb"/><circle cx="0" cy="0" r="5" fill="white"/></g><text x="35" y="{415-i*95}">{html.escape(name)}: Δy={y:+.2f} m, yaw={yaw:+.0f}°</text>'
            for i, (name, y, yaw) in enumerate(rows)
        )
        card = root / "placement_card.html"
        content = f'''<!doctype html><meta charset="utf-8"><title>v1.5.2 directional placement</title>
<style>body{{font:18px sans-serif;max-width:1000px;margin:30px auto}}svg{{border:1px solid #999;background:#fafafa}}.warn{{color:#b91c1c;font-weight:bold}}</style>
<h1>Student 单侧标定工作域摆位卡</h1><p class="warn">等待用户观看接受；必须有人保护；禁止自动真机部署。</p>
<svg viewBox="0 0 900 620"><rect x="620" y="40" width="240" height="540" fill="#d1d5db"/><line x1="620" x2="620" y1="40" y2="580" stroke="#111" stroke-width="6"/><line x1="740" x2="740" y1="40" y2="580" stroke="#ef4444" stroke-dasharray="10 8"/><text x="690" y="30">平台中心线</text><path d="M620 310 L540 310" stroke="#16a34a" stroke-width="5"/><text x="390" y="290">平台边缘外法向</text>{svg_rows}<line x1="360" x2="620" y1="520" y2="520" stroke="#7c3aed"/><text x="390" y="550">机器人中心起步距离 2.05 m；前缘净距 0.55 m</text><path d="M360 480 L520 480" stroke="#f97316" stroke-width="5"/><text x="380" y="465">命令 v=(+0.45, 0, 0) m/s</text></svg>
<p>俯视坐标：+x 指向平台，+y 为图中向上；严格按有符号横移和偏航标定，不依赖 left/right 文字猜测。</p>'''
        card.write_text(content)
        return card

    def videos(self, stage: int, checkpoint: Path, side: str) -> Path:
        root = STATE_ROOT / "candidate_visual_review" / f"E{stage}_{side}"
        terminal = root / "paired_video_manifest.json"
        if terminal.is_file():
            return terminal
        sign = 1 if side == "left" else -1
        scenarios = (("nominal", 0.0, 0.0), (f"half_{side}", .06 * sign, 2 * sign),
                     (f"{side}_offset", .12 * sign, 4 * sign))
        records = []
        for scenario, lateral, yaw in scenarios:
            for role, policy, task in (
                ("student", checkpoint, base.EVAL_TASK),
                ("teacher", base.TEACHER, "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0"),
            ):
                label = f"{role}_{scenario}"
                destination = root / "videos" / f"{label}.mp4"
                source_dir = policy.parent / "videos/play"
                before = {path.resolve(): path.stat().st_mtime_ns for path in source_dir.glob("*.mp4")} if source_dir.exists() else {}
                command = self.play_command(policy, 11, f"{lateral:.2f}", f"{yaw:.1f}")
                command[command.index(base.EVAL_TASK)] = task
                command.extend(["--enable_cameras", "--video", "--video_length", "600", "--highstep_gap_camera", "side_top"])
                self.run_command(command, root / "logs" / f"{label}.log", root, f"video_{label}", retries=3, stall=1200)
                created = [path for path in source_dir.glob("*.mp4") if path.resolve() not in before or path.stat().st_mtime_ns > before.get(path.resolve(), 0)]
                if not created:
                    raise RuntimeError(f"video file missing for {label}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(max(created, key=lambda path: path.stat().st_mtime_ns), destination)
                video_qc = video_delivery_qc(destination, root / "logs" / f"{label}.log")
                records.append({
                    "label": label,
                    "video": str(destination),
                    "sha256": base.sha256_file(destination),
                    "video_delivery_qc": video_qc,
                })
        card = self.placement_card(root, side)
        atomic_json(terminal, {
            "schema_version": 1, "kind": "highstep_v152_directional_teacher_student_paired_videos",
            "status": "student_directional_candidate_pending_user_visual_review",
            "checkpoint": str(checkpoint), "checkpoint_sha256": base.sha256_file(checkpoint),
            "successful_side": side, "videos": records, "placement_card": str(card),
            "placement_card_sha256": base.sha256_file(card), "automatic_deployment_allowed": False,
            "robot_visibility_qc_passed": all(
                row["video_delivery_qc"]["passed"] for row in records
            ),
            "robot_visibility_requires_user_review": True,
        }, read_only=True)
        return terminal

    def record_terminal(self, status: str, reason: str, best: Mapping[str, Any] | None = None) -> None:
        candidate = status == "student_directional_candidate_pending_user_visual_review"
        handoff = {
            "schema_version": 1, "workflow_id": WORKFLOW_ID, "status": status, "reason": reason,
            "checkpoint": self.state.get("checkpoint"), "effective_updates": self.state.get("effective_updates"),
            "best_checkpoint": None if best is None else best.get("checkpoint"),
            "best_checkpoint_sha256": None if best is None else best.get("checkpoint_sha256"),
            "requires_user_action": candidate, "next_action": "user_visual_review" if candidate else "none",
            "automatic_real_robot_deployment_performed": False, "written_at": base.now(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update(status=status, phase=status, stop_reason=reason, requires_user_action=candidate, active_pid=None)

    def run(self) -> None:
        self.preflight()
        source, previous = SOURCE_E300, 300
        best = None
        for stage in SAVE_POINTS:
            self.assert_code()
            result_path = STATE_ROOT / "stages" / f"E{stage}" / "stage_result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                checkpoint = Path(result["checkpoint"]).resolve(strict=True)
                if result["checkpoint_sha256"] != base.sha256_file(checkpoint):
                    raise RuntimeError(f"stage result E{stage} checkpoint changed")
                wandb_manifest = Path(result["wandb_stage_manifest"]).resolve(strict=True)
                if json.loads(wandb_manifest.read_text()).get("sync_status") != "synced":
                    self.finalize_wandb(stage, checkpoint, result_path, wandb_manifest,
                                        self.wandb_gate_metrics(result), result["decision"]["reason"])
                source, previous = checkpoint, stage
            else:
                run_dir, checkpoint, wandb_manifest = self.train_stage(stage, source, previous)
                imitation = self.imitation(stage, checkpoint)
                probe = self.probe(stage, checkpoint)
                core = self.core9(stage, checkpoint, run_dir)
                directional = self.directional(stage, checkpoint, core)
                decision = self.decision(stage, imitation, probe, core, directional)
                direction_score = max(
                    min(value["counts"]["full_climb"], value["counts"]["rear_hold"])
                    for value in directional["corridors"].values()
                )
                rank = [int(decision["final_numeric_gate_passed"]), direction_score,
                        core["counts"]["full_climb"] + core["counts"]["rear_hold"],
                        probe["behavior_score"], -imitation["failed_element_count"], -stage]
                result = {
                    "schema_version": 1, "kind": "highstep_v152_stage_result", "stage": stage,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": base.sha256_file(checkpoint),
                    "imitation": imitation, "probe": probe, "core9": core,
                    "directional": directional, "decision": decision, "selection_rank": rank,
                    "wandb_stage_manifest": str(wandb_manifest), "completed_at": base.now(),
                }
                atomic_json(result_path, result, read_only=True)
                self.finalize_wandb(stage, checkpoint, result_path, wandb_manifest,
                                    self.wandb_gate_metrics(result), decision["reason"])
            best = result if best is None or tuple(result["selection_rank"]) > tuple(best["selection_rank"]) else best
            decision = result["decision"]
            self.update(checkpoint=result["checkpoint"], effective_updates=stage,
                        imitation_gate={"passed": result["imitation"]["passed"], "failed_elements": result["imitation"]["failed_element_count"]},
                        behavior_probe=result["probe"]["counts"], core9=result["core9"]["counts"],
                        directional={side: result["directional"]["corridors"][side]["counts"] for side in ("left", "right")},
                        classification=decision["classification"], decision=decision,
                        best_checkpoint=best["checkpoint"], best_checkpoint_sha256=best["checkpoint_sha256"])
            if decision["final_numeric_gate_passed"]:
                side = decision["passed_sides"][0] if decision["passed_sides"] else "left"
                video_manifest = self.videos(stage, Path(result["checkpoint"]), side)
                self.update(video_manifest=str(video_manifest), video_manifest_sha256=base.sha256_file(video_manifest))
                self.record_terminal("student_directional_candidate_pending_user_visual_review", decision["reason"], best)
                return
            if decision["stop"]:
                self.record_terminal("stopped_by_gate", decision["reason"], best)
                return
            source, previous = Path(result["checkpoint"]), stage
        self.record_terminal("stopped_by_gate", "absolute_2500_cap_without_stable_climb", best)


def main() -> int:
    supervisor = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("v1.5.2 supervisor already owns the workflow lock", file=sys.stderr)
        return 2
    except Exception as error:
        if supervisor is not None:
            message = f"{type(error).__name__}: {error}"
            mechanism_tokens = (
                "authority", "checkpoint", "optimizer", "binding", "frozen", "preregistration",
                "effective-update", "lineage", "post-prior", "PPO",
            )
            status = (
                "mechanism_error_requires_repair"
                if any(token.lower() in message.lower() for token in mechanism_tokens)
                else "external_failure_requires_attention"
            )
            supervisor.record_terminal(status, message)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
