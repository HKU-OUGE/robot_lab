#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lxq/Softwares/robot_lab"
PY="/home/lxq/miniconda3/envs/env_isaaclab/bin/python"
PLAY="$ROOT/scripts/rsl_rl/base/play.py"
GUARD="$ROOT/.agents/skills/highstep-control-variable-guardian/scripts/highstep_guard.py"
TASK="RobotLab-Isaac-Velocity-HighstepB300DiagonalImitationFresh7400StudentNoPrior-ArcdogAdjustableLeg-v0"
FLOW="$ROOT/tmp/highstep_b300_diagonal_imitation_fresh_7400_20260719"
PREREG="$FLOW/preregistration.json"
PREREG_SHA="51d78fa0ef0b1ab90a6c05704e7380ebc3703d7cfa36b129d2f47b0548ec26bb"
RUN="$ROOT/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_diagonal_imitation_fresh_7400_Student/2026-07-19_02-26-51_highstep_b300_diagonal_imitation_fresh_7400_student_from_173499"
LOCK="$ROOT/tmp/highstep_manual_play.lock"
MARKER="$ROOT/tmp/highstep_manual_play_active"
LOG_DIR="$ROOT/logs/play_joint_records/diagonal_manual_launch_logs"

usage() {
  cat <<'EOF'
用法：
  diagplay latest [--debug] [--hard] [--seed N]
  diagplay e1400  [--debug] [--hard] [--seed N]
  diagplay previous [--debug] [--hard] [--seed N]

固定语义：键盘手动控制、自由视角、普通box level 9（约35 cm）附近出生；R只重生，不退出。
兼容参数--hard只表示普通box的最高level，不会切换到可达38 cm的box_hard。
启动后和每次按R后，必须等待终端出现：
  RESET READY / 现在可以推动方向键
再按方向键。恢复期间方向键输入会被忽略。
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
WHICH="$1"
shift
case "$WHICH" in
  latest)
    CHECKPOINT="$RUN/model_175098.pt"
    EXPECTED_SHA="f2855f996aa43af710253b79405503e5800db068a4a09336c17f32adbcbcced7"
    LABEL="latest_E1600"
    ;;
  e1400)
    CHECKPOINT="$RUN/model_174898.pt"
    EXPECTED_SHA="34b5e69085df58a6e5062bad51a0830c2bfcd307304a954ac96d4c6862192a4b"
    LABEL="E1400"
    ;;
  previous)
    CHECKPOINT="$RUN/model_174998.pt"
    EXPECTED_SHA="4ece7f907de6126beb23ef138c294bc2c16f9aba1acffa8d08480f2675d20920"
    LABEL="previous_E1500"
    ;;
  -h|--help) usage; exit 0 ;;
  *) echo "未知checkpoint：$WHICH" >&2; usage >&2; exit 2 ;;
esac

SEED=11
TERRAIN="box"
DEBUG_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) DEBUG_ARGS=(--debug); shift ;;
    --hard) TERRAIN="box"; shift ;;
    --seed)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "--seed后需要整数" >&2; exit 2; }
      SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in "$PY" "$PLAY" "$GUARD" "$PREREG" "$CHECKPOINT"; do
  [[ -f "$required" ]] || { echo "缺少文件：$required" >&2; exit 3; }
done
actual_prereg="$(sha256sum "$PREREG" | awk '{print $1}')"
actual_checkpoint="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$actual_prereg" == "$PREREG_SHA" ]] || { echo "preregistration SHA不匹配，拒绝启动" >&2; exit 4; }
[[ "$actual_checkpoint" == "$EXPECTED_SHA" ]] || { echo "checkpoint SHA不匹配，拒绝启动" >&2; exit 4; }

"$PY" "$GUARD" audit --json >/dev/null
exec 9>"$LOCK"
flock -n 9 || { echo "已有手动play正在运行；请先退出它。" >&2; exit 5; }
if pgrep -af '[s]cripts/rsl_rl/base/(train|play)\.py|[h]ighstep_.*(supervisor|launcher)\.py' >/dev/null 2>&1; then
  echo "检测到训练、评估、play、supervisor或launcher；本次拒绝启动。" >&2
  exit 6
fi

mkdir -p "$LOG_DIR"
stamp="$(date +%Y%m%d_%H%M%S)"
record_dir="$ROOT/logs/play_joint_records/${stamp}_diag_${LABEL}_seed${SEED}_pid$$"
launch_log="$LOG_DIR/${stamp}_${LABEL}_seed${SEED}_pid$$.log"
cleanup() { rm -f "$MARKER"; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

marker_tmp="${MARKER}.tmp.$$"
printf 'pid=%s\nstarted_at=%s\nmode=diagonal_student_manual_review\ncheckpoint=%s\ncheckpoint_sha256=%s\nrecord_dir=%s\n' \
  "$$" "$(date -Ins)" "$CHECKPOINT" "$EXPECTED_SHA" "$record_dir" >"$marker_tmp"
mv "$marker_tmp" "$MARKER"

echo "启动：$LABEL，seed=$SEED，terrain=$TERRAIN level 9（约35 cm），正常一致reset（不使用00167意外斜姿态）。"
echo "R只重生。启动后或按R后：保持不操作至少100控制步/2.00秒，并继续等待稳定性检查。"
echo "必须看到 RESET READY / 现在可以推动方向键 后，才能按方向键。"
echo "日志：$launch_log"
echo "关节记录：$record_dir"

cd "$ROOT"
set +e
HIGHSTEP_BE300_0707_PREREGISTRATION_PATH="$PREREG" \
HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256="$PREREG_SHA" \
PYTHONPATH="$ROOT/source/robot_lab${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -u "$PLAY" \
  --task "$TASK" \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --seed "$SEED" \
  --play_terrain_level 9 \
  --play_terrain_type "$TERRAIN" \
  --reset_after_play_terrain_selection \
  --front_step_eval_reset \
  --front_step_eval_side x- \
  --front_step_eval_edge_gap 0.55 \
  --front_step_eval_lateral_offset 0 \
  --front_step_eval_yaw_offset_deg 0 \
  --eval_action_delay_steps 0 \
  --record_joint_data \
  --joint_record_output "$record_dir" \
  --joint_record_label "diag_${LABEL}_seed${SEED}_box_level9_35cm" \
  "${DEBUG_ARGS[@]}" \
  --checkpoint "$CHECKPOINT" \
  --skip_policy_export \
  2>&1 | tee "$launch_log"
status="${PIPESTATUS[0]}"
set -e
if rg -q '^Traceback \(most recent call last\):' "$launch_log"; then status=1; fi
printf '\n[DIAGPLAY_EXIT] status=%s finished_at=%s\n' "$status" "$(date -Ins)" | tee -a "$launch_log"
exit "$status"
