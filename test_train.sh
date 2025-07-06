#!/bin/bash

# ---------------- 配置部分（可自定义） ----------------
TASK_NAME="RobotLab-Isaac-Velocity-Flat-Sirius-v0"
NUM_ENVS=50
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ---------------- 执行部分 ----------------
# 创建日志目录（如果不存在）
mkdir -p "${LOG_DIR}"

# 执行训练脚本并记录 stdout + stderr
python scripts/rsl_rl/base/train.py \
    --task "${TASK_NAME}" \
    --num_envs ${NUM_ENVS} \
    2>&1 | tee "${LOG_FILE}"

