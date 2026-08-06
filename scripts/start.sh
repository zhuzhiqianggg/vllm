#!/usr/bin/env bash
# 一键启动 vLLM
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "[INFO] .env 不存在，从 .env.example 复制"
  cp .env.example .env
  echo "[WARN] 请检查 .env 后再启动，尤其是 MODEL_NAME、TENSOR_PARALLEL_SIZE、MAX_MODEL_LEN"
  exit 1
fi

# docker compose v2
docker compose up -d

echo "---"
echo "查看日志:    ./scripts/logs.sh"
echo "查看状态:    docker compose ps"
echo "测试接口:    curl http://localhost:8000/v1/models"
