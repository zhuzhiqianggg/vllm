#!/usr/bin/env bash
# 基础性能基准测试
# 用法: ./scripts/benchmark.sh [concurrency] [num_prompts]
set -euo pipefail
HOST="${HOST:-http://localhost:8000}"
MODEL_NAME="${SERVED_MODEL_NAME:-$(grep '^MODEL_NAME=' .env | cut -d= -f2 | tr -d '"')}"
CONCURRENCY="${1:-32}"
NUM_PROMPTS="${2:-200}"

if ! command -v vllm >/dev/null 2>&1 && ! python3 -c "import vllm" >/dev/null 2>&1; then
  echo "[INFO] vLLM bench 工具未找到，使用原生 python 压测"
  exec python3 "$(dirname "$0")/bench_http.py" "$@"
fi

vllm bench serve \
  --host "${HOST}" \
  --model "${MODEL_NAME}" \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sonnet \
  --dataset-path benchmarks/sonnet.txt \
  --sonnet-input-len 1024 \
  --sonnet-output-len 256 \
  --max-concurrency "${CONCURRENCY}" \
  --num-prompts "${NUM_PROMPTS}" \
  --request-rate inf \
  --save-result \
  --result-dir benchmarks
