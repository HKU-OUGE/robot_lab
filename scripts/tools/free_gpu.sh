#!/usr/bin/env bash
set -euo pipefail

echo "[1] FIND & KILL train.py / torch / Isaac / ray "
PIDS=$(ps -ef | egrep 'python|torch|train\.py|kit|Isaac|nccl|ray' | grep -v egrep | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "FOUND PID: $PIDS"
    kill $PIDS 2>/dev/null || true
    sleep 2
    kill -9 $PIDS 2>/dev/null || true
else
    echo "NO PID FOUND"
fi

echo "[2] KILL Ray（sudo needed）"
sudo pkill -f "ray|gcs_server|raylet|dashboard|plasma_store|runtime_env" 2>/dev/null || true
sudo -E ray stop --force 2>/dev/null || true
sudo rm -rf /tmp/ray 2>/dev/null || true

echo "[3] CLEAN NCCL / Torch TEMP / MEM"
rm -f /dev/shm/nccl_* /dev/shm/torch_* 2>/dev/null || true
rm -rf /tmp/nccl_* /tmp/torch_* 2>/dev/null || true

echo "[4] COMPLETE，CHECK NVIDIA GPU:"
nvidia-smi
