#!/bin/bash

# ---------------- default values ----------------
TASK_NAME="RobotLab-Isaac-Velocity-Stand-CUHKLRL-SiriusW-v0"
NUM_ENVS=50
HEADLESS=false
VIDEO=false
VIDEO_LENGTH=200
VIDEO_INTERVAL=2000
SEED=""
MAX_ITERATIONS=""
RECOVERY_MODE=false
DEBUG=false

# ---------------- parse CLI args ----------------
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task) TASK_NAME="$2"; shift; shift ;;
        --num_envs) NUM_ENVS="$2"; shift; shift ;;
        --headless) HEADLESS=true; shift ;;
        --video) VIDEO=true; shift ;;
        --video_length) VIDEO_LENGTH="$2"; shift; shift ;;
        --video_interval) VIDEO_INTERVAL="$2"; shift; shift ;;
        --seed) SEED="$2"; shift; shift ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift; shift ;;
        --recovery_mode) RECOVERY_MODE=true; shift ;;
        --debug) DEBUG=true; shift ;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
done

# ---------------- logger ----------------
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TASK_NAME}_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

# ---------------- command composition ----------------
CMD="python scripts/rsl_rl/base/train.py"
CMD+=" --task \"${TASK_NAME}\""
CMD+=" --num_envs ${NUM_ENVS}"

[ "$HEADLESS" = true ] && CMD+=" --headless"
[ "$VIDEO" = true ] && CMD+=" --video"
if [ "$VIDEO" = true ]; then
    [ "$VIDEO_LENGTH" != "200" ] && CMD+=" --video_length ${VIDEO_LENGTH}"
    [ "$VIDEO_INTERVAL" != "2000" ] && CMD+=" --video_interval ${VIDEO_INTERVAL}"
fi
[ -n "$SEED" ] && CMD+=" --seed ${SEED}"
[ -n "$MAX_ITERATIONS" ] && CMD+=" --max_iterations ${MAX_ITERATIONS}"
[ "$RECOVERY_MODE" = true ] && CMD+=" --recovery_mode"
[ "$DEBUG" = true ] && CMD+=" --debug"

# ---------------- execute ----------------
eval "${CMD} 2>&1 | tee \"${LOG_FILE}\""


