#!/bin/bash

# ---------------- default ----------------
TASK_NAME="RobotLab-Isaac-Velocity-Stand-CUHKLRL-SiriusW-v0"
NUM_ENVS=50

HEADLESS=false
VIDEO=false
VIDEO_LENGTH=200
DISABLE_FABRIC=false
USE_PRETRAINED_CHECKPOINT=false
REAL_TIME=false
KEYBOARD=false
DEBUG=false

# ---------------- parse CLI ----------------
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task) TASK_NAME="$2"; shift; shift ;;
        --num_envs) NUM_ENVS="$2"; shift; shift ;;
        --headless) HEADLESS=true; shift ;;
        --video) VIDEO=true; shift ;;
        --video_length) VIDEO_LENGTH="$2"; shift; shift ;;
        --disable_fabric) DISABLE_FABRIC=true; shift ;;
        --use_pretrained_checkpoint) USE_PRETRAINED_CHECKPOINT=true; shift ;;
        --real-time) REAL_TIME=true; shift ;;
        --keyboard) KEYBOARD=true; shift ;;
        --debug) DEBUG=true; shift ;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
done

# ---------------- logger ----------------
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/play_${TASK_NAME}_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

# ---------------- build command ----------------
CMD="python scripts/rsl_rl/base/play.py \
    --task \"${TASK_NAME}\" \
    --num_envs ${NUM_ENVS}"

[ "$HEADLESS" = true ] && CMD+=" --headless"
if [ "$VIDEO" = true ]; then
    CMD+=" --video"
    [ "$VIDEO_LENGTH" != "200" ] && CMD+=" --video_length ${VIDEO_LENGTH}"
fi
[ "$DISABLE_FABRIC" = true ] && CMD+=" --disable_fabric"
[ "$USE_PRETRAINED_CHECKPOINT" = true ] && CMD+=" --use_pretrained_checkpoint"
[ "$REAL_TIME" = true ] && CMD+=" --real-time"
[ "$KEYBOARD" = true ] && CMD+=" --keyboard"
[ "$DEBUG" = true ] && CMD+=" --debug"

# ---------------- execute ----------------
eval "${CMD} 2>&1 | tee \"${LOG_FILE}\""

