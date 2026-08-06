#!/usr/bin/env bash
# 一键停止 vLLM
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose down
echo "[OK] vLLM 已停止"
