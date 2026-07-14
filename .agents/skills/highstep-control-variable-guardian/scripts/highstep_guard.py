#!/usr/bin/env python3
"""Read-only highstep authority audit and Codex PreToolUse guard."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
ACTIVE_DECLARATION = REPO / "tmp/highstep_dashboard_active_workflow.json"
CHANGE_AUTHORITY = REPO / "tmp/highstep_codex_change_authority.json"

PROTECTED_PATH_FRAGMENTS = (
    "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md",
    "scripts/rsl_rl/base/train.py",
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py",
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py",
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
    "tools/highstep_historical_0707_exact_supervisor.py",
)

SEMANTIC_TOKENS = (
    "student_actor_warmup_updates",
    "student_teacher_action_loss_coef",
    "student_prior_box_loss_coef",
    "student_highstep_phase_loss_scale",
    "student_highstep_rear_box_loss_scale",
    "student_vae_epochs",
    "learning_rate",
    "action_scale",
    "joint_pos.clip",
    "default_dof_pos",
    "teacher_pre_prior",
    "teacher_post_prior",
)

SUPERSEDED_SERVICE_RE = re.compile(
    r"highstep-(?:student-recovery-(?:v15|v13|r2|r3)|directional-trial|"
    r"e1400-root-cause-guard|zero-scale-ablation-stop-guard)\.service"
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def pid_alive(value: Any) -> bool:
    try:
        pid = int(value)
        os.kill(pid, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return pid > 1


def current_context(root: Path = REPO) -> dict[str, Any]:
    declaration_path = root / ACTIVE_DECLARATION.relative_to(REPO)
    declaration = read_json(declaration_path)
    state_path_raw = declaration.get("state_path")
    prereg_path_raw = declaration.get("preregistration_path")
    state_path = Path(state_path_raw) if isinstance(state_path_raw, str) else None
    prereg_path = Path(prereg_path_raw) if isinstance(prereg_path_raw, str) else None
    state = read_json(state_path) if state_path else {}
    prereg = read_json(prereg_path) if prereg_path else {}
    authority = prereg.get("authority") if isinstance(prereg.get("authority"), dict) else {}
    spec_raw = authority.get("spec_path")
    spec_path = Path(spec_raw) if isinstance(spec_raw, str) else None
    return {
        "declaration_path": declaration_path,
        "declaration": declaration,
        "state_path": state_path,
        "state": state,
        "preregistration_path": prereg_path,
        "preregistration": prereg,
        "spec_path": spec_path,
        "spec_sha256": sha256_file(spec_path) if spec_path else None,
    }


def audit(root: Path = REPO) -> dict[str, Any]:
    ctx = current_context(root)
    decl = ctx["declaration"]
    state = ctx["state"]
    prereg = ctx["preregistration"]
    authority = prereg.get("authority") if isinstance(prereg.get("authority"), dict) else {}
    issues: list[str] = []

    if not decl:
        issues.append("missing active workflow declaration")
    if not state:
        issues.append("missing declared workflow state")
    if not prereg:
        issues.append("missing declared preregistration")

    workflow_ids = {
        value
        for value in (decl.get("workflow_id"), state.get("workflow_id"), prereg.get("workflow_id"))
        if value
    }
    if len(workflow_ids) != 1:
        issues.append(f"workflow mismatch: {sorted(workflow_ids)}")

    actual_spec_sha = ctx["spec_sha256"]
    expected_spec_shas = {
        value
        for value in (decl.get("spec_sha256"), state.get("spec_sha256"), authority.get("spec_sha256"))
        if value
    }
    if len(expected_spec_shas) != 1 or actual_spec_sha not in expected_spec_shas:
        issues.append(
            f"spec SHA mismatch: actual={actual_spec_sha} expected={sorted(expected_spec_shas)}"
        )

    prereg_path = ctx["preregistration_path"]
    actual_prereg_sha = sha256_file(prereg_path) if prereg_path else None
    expected_prereg_shas = {
        value
        for value in (decl.get("preregistration_sha256"), state.get("preregistration_sha256"))
        if value
    }
    if expected_prereg_shas and (
        len(expected_prereg_shas) != 1 or actual_prereg_sha not in expected_prereg_shas
    ):
        issues.append(
            f"preregistration SHA mismatch: actual={actual_prereg_sha} expected={sorted(expected_prereg_shas)}"
        )

    active_pid = state.get("active_pid")
    supervisor_pid = state.get("supervisor_pid")
    return {
        "ok": not issues,
        "issues": issues,
        "workflow_id": state.get("workflow_id") or decl.get("workflow_id"),
        "authority_version": decl.get("authority_version") or authority.get("version"),
        "phase": state.get("phase"),
        "status": state.get("status"),
        "spec_path": str(ctx["spec_path"] or ""),
        "spec_sha256": actual_spec_sha,
        "preregistration_path": str(prereg_path or ""),
        "preregistration_sha256": actual_prereg_sha,
        "active_pid": active_pid,
        "active_pid_alive": pid_alive(active_pid),
        "supervisor_pid": supervisor_pid,
        "supervisor_pid_alive": pid_alive(supervisor_pid),
        "requires_user_action": state.get("requires_user_action"),
    }


def _path_mentions(text: str, fragments: tuple[str, ...] = PROTECTED_PATH_FRAGMENTS) -> list[str]:
    normalized = text.replace("\\", "/")
    return [fragment for fragment in fragments if fragment in normalized]


def _is_mutating(tool_name: str, text: str) -> bool:
    if re.search(r"apply_patch|Edit|Write", tool_name, re.IGNORECASE):
        return True
    return bool(
        re.search(
            r"(?:sed\s+-i|perl\s+-pi|\b(?:rm|mv|cp|truncate|tee)\b|>>?|python\d*\s+-c)",
            text,
        )
    )


def _authority_allows(
    path_fragments: list[str],
    semantic_tokens: list[str],
    ctx: dict[str, Any],
    root: Path = REPO,
) -> bool:
    authority_path = root / CHANGE_AUTHORITY.relative_to(REPO)
    authority = read_json(authority_path)
    if authority.get("schema_version") != 1 or authority.get("user_approved") is not True:
        return False
    if authority.get("workflow_id") != ctx["workflow_id"]:
        return False
    if authority.get("spec_sha256") != ctx["spec_sha256"]:
        return False
    expires = authority.get("expires_at_unix")
    if not isinstance(expires, (int, float)) or expires < dt.datetime.now().timestamp():
        return False
    variable = authority.get("single_variable")
    if not isinstance(variable, str) or not variable.strip():
        return False
    allowed_paths = authority.get("allowed_path_fragments")
    allowed_tokens = authority.get("allowed_semantic_tokens")
    if not isinstance(allowed_paths, list) or not isinstance(allowed_tokens, list):
        return False
    if any(fragment not in allowed_paths for fragment in path_fragments):
        return False
    if any(token not in allowed_tokens for token in semantic_tokens):
        return False
    prereg_raw = authority.get("preregistration_path")
    prereg_sha = authority.get("preregistration_sha256")
    if not isinstance(prereg_raw, str) or sha256_file(Path(prereg_raw)) != prereg_sha:
        return False
    return True


def deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def pre_tool_decision(payload: dict[str, Any], root: Path = REPO) -> dict[str, Any] | None:
    tool_name = str(payload.get("tool_name") or payload.get("toolName") or "")
    tool_input = payload.get("tool_input") or payload.get("toolInput") or {}
    text = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
    ctx = audit(root)

    if not ctx["ok"] and _is_mutating(tool_name, text):
        return deny("Highstep authority audit failed; mutation is blocked until declaration/state/spec/preregistration SHAs agree.")

    if SUPERSEDED_SERVICE_RE.search(text) and re.search(
        r"\b(?:start|enable|restart|reenable)\b", text
    ):
        return deny("Starting or enabling a superseded highstep service is forbidden by the current authority.")

    active_pids = [str(value) for value in (ctx.get("active_pid"), ctx.get("supervisor_pid")) if value]
    if re.search(r"\bkill\s+-9\b|\bpkill\s+-9\b", text) and any(pid in text for pid in active_pids):
        return deny("SIGKILL against the active highstep workflow is forbidden; stop only at a verified safe boundary.")

    if "train.py" in text and "highstep_0707_exact_new_teacher_Student" in text:
        return deny("Training from the archived zero-scale Student/optimizer lineage is forbidden.")

    mentioned_paths = _path_mentions(text)
    semantic_tokens = [token for token in SEMANTIC_TOKENS if token in text]
    mutating = _is_mutating(tool_name, text)
    if not mutating or (not mentioned_paths and not semantic_tokens):
        return None

    if ctx.get("active_pid_alive"):
        return deny("Protected highstep authority/training files cannot be changed while an active child exists.")

    if not _authority_allows(mentioned_paths, semantic_tokens, ctx, root):
        return deny(
            "Protected highstep changes require an unexpired, user-approved, single-variable "
            "tmp/highstep_codex_change_authority.json bound to the current spec and preregistration."
        )
    return None


def hook_main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps(deny("Malformed hook payload; protected mutation denied.")))
        return 0

    event = payload.get("hook_event_name") or payload.get("hookEventName")
    if event == "SessionStart":
        message = (
            "For every RobotLab highstep task, use $highstep-control-variable-guardian, "
            "run its authority audit before acting, and never infer current parameters from historical-only spec sections."
        )
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": message,
                    }
                }
            )
        )
        return 0
    if event == "PreToolUse":
        decision = pre_tool_decision(payload)
        if decision:
            print(json.dumps(decision, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="audit current highstep authority")
    audit_parser.add_argument("--json", action="store_true")
    subparsers.add_parser("hook", help="serve one Codex hook event from stdin")
    args = parser.parse_args()

    if args.command == "hook":
        return hook_main()

    result = audit()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"authority={'OK' if result['ok'] else 'FAIL'} "
            f"workflow={result['workflow_id']} version={result['authority_version']} "
            f"phase={result['phase']} status={result['status']} "
            f"child={result['active_pid']} alive={result['active_pid_alive']}"
        )
        for issue in result["issues"]:
            print(f"- {issue}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
