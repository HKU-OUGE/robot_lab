#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/lxq/Softwares/robot_lab
PYTHON=/home/lxq/miniconda3/envs/env_isaaclab/bin/python
STATE_ROOT="$ROOT/tmp/highstep_student_recovery_20260712"
BASELINE="$STATE_ROOT/baseline_core9_retry2"
LOG="$STATE_ROOT/quota_bridge_launcher.log"
SUPERVISOR="$ROOT/tools/highstep_student_recovery_supervisor.py"

mkdir -p "$STATE_ROOT"
exec >>"$LOG" 2>&1

on_error() {
    rc=$?
    printf '%s quota bridge failed_closed rc=%s line=%s\n' "$(date --iso-8601=seconds)" "$rc" "${BASH_LINENO[0]:-unknown}"
    exit "$rc"
}
trap on_error ERR

printf '%s quota bridge started pid=%s\n' "$(date --iso-8601=seconds)" "$$"

# The already-running baseline is useful work and is the mandatory no-update
# gate.  Wait for its own manifest; never interrupt it to start training early.
while [[ ! -f "$BASELINE/evaluation_manifest.json" ]]; do
    printf '%s waiting for baseline, rows=%s\n' \
        "$(date --iso-8601=seconds)" \
        "$(wc -l <"$BASELINE/eval_runs.jsonl" 2>/dev/null || printf '0')"
    sleep 15
done

# Do not race the final fail-closed correctness edits.  Require the exact files
# that define Stage B to remain byte-identical for two consecutive minutes.
critical_hashes() {
    sha256sum \
        "$SUPERVISOR" \
        "$ROOT/scripts/rsl_rl/base/train.py" \
        "$ROOT/scripts/rsl_rl/base/play.py" \
        "$ROOT/scripts/rsl_rl/base/algorithm_checkpoint.py" \
        "$ROOT/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py" \
        "$ROOT/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py" \
        "$ROOT/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py" \
        "$ROOT/tmp/highstep_centerline_guard_monitor_20260709.sh"
}

while true; do
    before="$(critical_hashes)"
    sleep 120
    after="$(critical_hashes)"
    if [[ "$before" == "$after" ]]; then
        break
    fi
    printf '%s correctness files still changing; restarting stability window\n' "$(date --iso-8601=seconds)"
done

if [[ -e "$STATE_ROOT/state.json" ]]; then
    printf '%s existing state.json found; refusing implicit second workflow\n' "$(date --iso-8601=seconds)"
    exit 31
fi

if pgrep -af 'scripts/rsl_rl/base/(train|play)\.py' >/dev/null; then
    printf '%s another train/play is still active; waiting for exclusive GPU\n' "$(date --iso-8601=seconds)"
    while pgrep -af 'scripts/rsl_rl/base/(train|play)\.py' >/dev/null; do
        sleep 15
    done
fi

"$PYTHON" -m py_compile \
    "$SUPERVISOR" \
    "$ROOT/scripts/rsl_rl/base/train.py" \
    "$ROOT/scripts/rsl_rl/base/play.py" \
    "$ROOT/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/vae_ppo.py"

printf '%s launching spec-locked supervisor\n' "$(date --iso-8601=seconds)"
cd "$ROOT"
exec "$PYTHON" -u "$SUPERVISOR" --baseline-dir "$BASELINE"
