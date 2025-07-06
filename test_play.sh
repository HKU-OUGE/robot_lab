#!/bin/bash

TASK_NAME="RobotLab-Isaac-Velocity-Flat-Sirius-v0"
NUM_ENVS=50
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/play_${TASK_NAME}_${TIMESTAMP}.log"

# 创建日志目录（如果不存在）
mkdir -p "${LOG_DIR}"

# 执行 play.py 并保存 log
python scripts/rsl_rl/base/play.py \
    --task "${TASK_NAME}" \
    --num_envs ${NUM_ENVS} \
    2>&1 | tee "${LOG_FILE}"

