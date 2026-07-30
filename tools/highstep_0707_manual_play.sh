#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lxq/Softwares/robot_lab"
PY="/home/lxq/miniconda3/envs/env_isaaclab/bin/python"
PLAY="$ROOT/scripts/rsl_rl/base/play.py"
GUARD="$ROOT/.agents/skills/highstep-control-variable-guardian/scripts/highstep_guard.py"
LOCK="$ROOT/tmp/highstep_manual_play.lock"
MARKER="$ROOT/tmp/highstep_manual_play_active"
LAUNCH_LOG_DIR="$ROOT/logs/play_joint_records/launch_logs"

TEACHER_TASK="RobotLab-Isaac-Velocity-HighstepActionScoreTeacherV18Bootstrap-ArcdogAdjustableLeg-v0"
TEACHER_CHECKPOINT="$ROOT/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-04_01-45-21/model_151399.pt"
TEACHER_CHECKPOINT_SHA="d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d"
TEACHER_ENV="$ROOT/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-04_01-45-21/params/env.yaml"
TEACHER_ENV_SHA="25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7"
TEACHER_AGENT="$ROOT/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-04_01-45-21/params/agent.yaml"
TEACHER_AGENT_SHA="a39d614d055d43c53c3f9aead2166e4cee78567c2fec235f721725792bd1fda2"

STUDENT_TASK="RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-ArcdogAdjustableLeg-v0"
STUDENT_CHECKPOINT="$ROOT/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-05_00-13-46/model_158797.pt"
STUDENT_CHECKPOINT_SHA="7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284"
STUDENT_ENV="$ROOT/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-05_00-13-46/params/env.yaml"
STUDENT_ENV_SHA="f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636"
STUDENT_AGENT="$ROOT/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-05_00-13-46/params/agent.yaml"
STUDENT_AGENT_SHA="bb01ed1c7a8574363e9cea0d9d1dc4a0828d8453a097715ae2e4b49fa51be9ca"

usage() {
  cat <<'EOF'
Internal launcher. Use one of these short commands instead:
  bplay0707   # 0707 Teacher model_151399
  hsplay0707  # 0707 deployed Student model_158797
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

ROLE="$1"
case "$ROLE" in
  teacher)
    TASK="$TEACHER_TASK"
    CHECKPOINT="$TEACHER_CHECKPOINT"
    CHECKPOINT_SHA="$TEACHER_CHECKPOINT_SHA"
    ENV_SNAPSHOT="$TEACHER_ENV"
    ENV_SHA="$TEACHER_ENV_SHA"
    AGENT_SNAPSHOT="$TEACHER_AGENT"
    AGENT_SHA="$TEACHER_AGENT_SHA"
    PROFILE_ARGS=(--highstep_v18_teacher_bootstrap_profile)
    DISPLAY_NAME="0707 Teacher model_151399"
    ;;
  student)
    TASK="$STUDENT_TASK"
    CHECKPOINT="$STUDENT_CHECKPOINT"
    CHECKPOINT_SHA="$STUDENT_CHECKPOINT_SHA"
    ENV_SNAPSHOT="$STUDENT_ENV"
    ENV_SHA="$STUDENT_ENV_SHA"
    AGENT_SNAPSHOT="$STUDENT_AGENT"
    AGENT_SHA="$STUDENT_AGENT_SHA"
    PROFILE_ARGS=(--highstep_0707_legacy_student_profile --allow_legacy_highstep_schedule_fallback)
    DISPLAY_NAME="0707 deployed Student model_158797"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

for required in "$PY" "$PLAY" "$GUARD" "$CHECKPOINT" "$ENV_SNAPSHOT" "$AGENT_SNAPSHOT"; do
  if [[ ! -f "$required" ]]; then
    echo "缺少文件，拒绝启动：$required" >&2
    exit 3
  fi
done

verify_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA256不匹配，拒绝启动。" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 4
  fi
}

verify_sha "$CHECKPOINT" "$CHECKPOINT_SHA" checkpoint
verify_sha "$ENV_SNAPSHOT" "$ENV_SHA" env.yaml
verify_sha "$AGENT_SNAPSHOT" "$AGENT_SHA" agent.yaml

# Read-only authority audit plus one shared manual-play lock prevents this
# historical comparison from colliding with a live train/eval/play process.
"$PY" "$GUARD" audit --json >/dev/null
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "已有手动play正在运行；请先退出它。" >&2
  exit 5
fi
if pgrep -af '[s]cripts/rsl_rl/base/(train|play)\.py|[h]ighstep_.*supervisor\.py' >/dev/null 2>&1; then
  echo "检测到训练、评估、play或supervisor进程；为避免冲突，本次拒绝启动。" >&2
  exit 6
fi

mkdir -p "$LAUNCH_LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RECORD_DIR="$ROOT/logs/play_joint_records/${STAMP}_0707_${ROLE}_pid$$"
LAUNCH_LOG="$LAUNCH_LOG_DIR/${STAMP}_0707_${ROLE}_pid$$.log"

cleanup() {
  rm -f "$MARKER"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

marker_tmp="${MARKER}.tmp.$$"
printf 'pid=%s\nstarted_at=%s\nmode=0707_historical_manual_recording\nrole=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nenv_snapshot=%s\nenv_sha256=%s\nrecord_dir=%s\n' \
  "$$" "$(date -Ins)" "$ROLE" "$CHECKPOINT" "$CHECKPOINT_SHA" \
  "$ENV_SNAPSHOT" "$ENV_SHA" "$RECORD_DIR" >"$marker_tmp"
mv "$marker_tmp" "$MARKER"

echo "启动：$DISPLAY_NAME"
echo "场景：box_hard最高等级（level 9），seed=11，距高台边缘0.55 m。"
echo "操作：方向键移动，Z/X转向，L归零，R在同一高台摆位重生，Ctrl+C退出。"
echo "joint log：$RECORD_DIR"
echo "终端日志：$LAUNCH_LOG"
echo "请把录屏文件名标为 0707_${ROLE}，之后把这两个joint log目录和视频一起发我分析。"

cd "$ROOT"
set +e
PYTHONPATH="$ROOT/source/robot_lab${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -u "$PLAY" \
  --task "$TASK" \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --debug \
  --seed 11 \
  --play_terrain_level 9 \
  --play_terrain_type box_hard \
  --reset_after_play_terrain_selection \
  --front_step_eval_reset \
  --front_step_eval_side x- \
  --front_step_eval_edge_gap 0.55 \
  --front_step_eval_lateral_offset 0 \
  --front_step_eval_yaw_offset_deg 0 \
  --eval_action_delay_steps 0 \
  --record_joint_data \
  --joint_record_output "$RECORD_DIR" \
  --joint_record_label "0707_${ROLE}_seed11_box_hard_level9" \
  "${PROFILE_ARGS[@]}" \
  --checkpoint "$CHECKPOINT" \
  --skip_policy_export \
  2>&1 | tee "$LAUNCH_LOG"
status="${PIPESTATUS[0]}"
set -e
if rg -q '^Traceback \(most recent call last\):' "$LAUNCH_LOG"; then
  status=1
fi
printf '\n[0707_PLAY_EXIT] role=%s status=%s finished_at=%s\n' \
  "$ROLE" "$status" "$(date -Ins)" | tee -a "$LAUNCH_LOG"
exit "$status"
