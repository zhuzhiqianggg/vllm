#!/usr/bin/env bash
# ============================================================
# 快速切换 vLLM 模型
# 用法:
#   ./switch_model.sh list                   列出所有预设配置
#   ./switch_model.sh <profile>              切换到指定配置
#   ./switch_model.sh current                查看当前配置
#   ./switch_model.sh add <name> <key>=<val> ...  添加自定义配置
# 示例:
#   ./switch_model.sh qwen3.5-35b-fp8        切回 MoE 模型
#   ./switch_model.sh qwen3.6-27b            切到 27B 密集模型
#   ./switch_model.sh my-custom MODEL_NAME=/path SERVED_MODEL_NAME=test
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
BACKUP_FILE="${SCRIPT_DIR}/.env.bak"

# -------- 预设配置 --------
# 每个 profile 是一段完整的 .env 内容
declare -A PROFILES

# Profile: qwen3.5-35b-fp8 (当前 MoE 多模态模型)
PROFILES["qwen3.5-35b-fp8"]='# ============================================================
# vLLM 环境变量 - Qwen3.6-35B-A3B-FP8 (多模态 MoE)
# 适配: 2 x RTX 4090 (48GB)
# ============================================================
MODEL_NAME=/root/.cache/huggingface/Qwen__Qwen3.6-35B-A3B-FP8
SERVED_MODEL_NAME=qwen3.6-35b-a3b
HOST=0.0.0.0
PORT=8000
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=0
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=128000
DTYPE=auto
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=16384
BLOCK_SIZE=16
SWAP_SPACE=8
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER=qwen3_coder
REASONING_PARSER=qwen3
CHAT_TEMPLATE_KWARGS={"enable_thinking":false}
ENABLE_LOG_REQUESTS=true
VLLM_API_KEY=
NCCL_IB_HCA_LIST=
NCCL_DEBUG=WARN'

# Profile: qwen3.6-27b (密集模型，推理快，原生 256K ctx)
PROFILES["qwen3.6-27b"]='# ============================================================
# vLLM 环境变量 - Qwen3.6-27B (密集模型)
# 适配: 2 x RTX 4090 (48GB)
# 特点: 27B dense, 64层, hidden_size=5120, 原生256K ctx, 支持思考
# ============================================================
MODEL_NAME=/root/.cache/huggingface/Qwen__Qwen3.6-27B
SERVED_MODEL_NAME=qwen3.6-27b
HOST=0.0.0.0
PORT=8000
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=0
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=128000
DTYPE=auto
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=8192
BLOCK_SIZE=16
SWAP_SPACE=8
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER=qwen3_coder
REASONING_PARSER=qwen3
CHAT_TEMPLATE_KWARGS={"enable_thinking":false}
ENABLE_LOG_REQUESTS=true
VLLM_API_KEY=
NCCL_IB_HCA_LIST=
NCCL_DEBUG=WARN'

# Profile: qwen3.6-27b-fp8-e4m3 (测试 fp8_e4m3 KV)
PROFILES["qwen3.6-27b-fp8-e4m3"]='# ============================================================
# vLLM 环境变量 - Qwen3.6-27B + fp8_e4m3 KV cache (测试)
# 适配: 2 x RTX 4090 (48GB)
# 特点: 使用 fp8_e4m3 格式存储 KV cache，比 fp8 再省 25%
# ============================================================
MODEL_NAME=/root/.cache/huggingface/Qwen__Qwen3.6-27B
SERVED_MODEL_NAME=qwen3.6-27b
HOST=0.0.0.0
PORT=8000
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=0
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=128000
DTYPE=auto
KV_CACHE_DTYPE=fp8_e4m3
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=8192
BLOCK_SIZE=16
SWAP_SPACE=8
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER=qwen3_coder
REASONING_PARSER=qwen3
CHAT_TEMPLATE_KWARGS={"enable_thinking":false}
ENABLE_LOG_REQUESTS=true
VLLM_API_KEY=
NCCL_IB_HCA_LIST=
NCCL_DEBUG=WARN'

# Profile: qwen3.6-27b-fp8 (FP8 量化版，低显存占用)
PROFILES["qwen3.6-27b-fp8"]='# ============================================================
# vLLM 环境变量 - Qwen3.6-27B-FP8 (FP8 量化模型)
# 适配: 2 x RTX 4090 (48GB)
# 特点: FP8 量化权重，体积小约 50%，推理速度快
# ============================================================
MODEL_NAME=/root/.cache/huggingface/Qwen__Qwen3.6-27B-FP8
SERVED_MODEL_NAME=qwen3.6-27b-fp8
HOST=0.0.0.0
PORT=8000
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=0
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=128000
DTYPE=auto
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=8192
BLOCK_SIZE=16
SWAP_SPACE=8
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER=qwen3_coder
REASONING_PARSER=qwen3
CHAT_TEMPLATE_KWARGS={"enable_thinking":false}
ENABLE_LOG_REQUESTS=true
VLLM_API_KEY=
NCCL_IB_HCA_LIST=
NCCL_DEBUG=WARN'

# Profile: qwen3.6-27b-thinking (开启思考模式)
PROFILES["qwen3.6-27b-thinking"]='# ============================================================
# vLLM 环境变量 - Qwen3.6-27B + 思考模式开启
# 适配: 2 x RTX 4090 (48GB)
# 特点: 输出包含 <think> 思考过程，用于思维链推理场景
# ============================================================
MODEL_NAME=/root/.cache/huggingface/Qwen__Qwen3.6-27B
SERVED_MODEL_NAME=qwen3.6-27b
HOST=0.0.0.0
PORT=8000
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=0
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=128000
DTYPE=auto
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=8192
BLOCK_SIZE=16
SWAP_SPACE=8
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER=qwen3_coder
REASONING_PARSER=qwen3
CHAT_TEMPLATE_KWARGS={"enable_thinking":true}
ENABLE_LOG_REQUESTS=true
VLLM_API_KEY=
NCCL_IB_HCA_LIST=
NCCL_DEBUG=WARN'

# -------- 命令实现 --------
cmd="${1:-help}"
shift || true

show_profiles() {
  echo -e "${GREEN}=== 可用模型配置 ===${NC}"
  echo
  for name in $(echo "${!PROFILES[@]}" | tr ' ' '\n' | sort); do
    local model
    model=$(echo "${PROFILES[$name]}" | grep '^MODEL_NAME=' | cut -d= -f2)
    local served
    served=$(echo "${PROFILES[$name]}" | grep '^SERVED_MODEL_NAME=' | cut -d= -f2)
    local kv
    kv=$(echo "${PROFILES[$name]}" | grep '^KV_CACHE_DTYPE=' | cut -d= -f2)
    local maxlen
    maxlen=$(echo "${PROFILES[$name]}" | grep '^MAX_MODEL_LEN=' | cut -d= -f2)
    local seqs
    seqs=$(echo "${PROFILES[$name]}" | grep '^MAX_NUM_SEQS=' | cut -d= -f2)
    local thinking
    thinking=$(echo "${PROFILES[$name]}" | grep 'enable_thinking' | grep -o 'enable_thinking":[a-z]*' | cut -d: -f2)
    printf "  %-28s model=%s  served=%s  kv=%s  max_len=%s  seqs=%s  thinking=%s\n" \
      "$name" "$model" "$served" "$kv" "$maxlen" "$seqs" "$thinking"
  done
  echo
  echo "用法: $0 <profile>     应用预设配置"
  echo "      $0 current       查看当前配置"
  echo "      $0 add <name> k=v ...   添加自定义配置"
}

show_current() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    err ".env 文件不存在"
    exit 1
  fi
  local model served kv maxlen seqs
  model=$(grep '^MODEL_NAME=' "${ENV_FILE}" | cut -d= -f2)
  served=$(grep '^SERVED_MODEL_NAME=' "${ENV_FILE}" | cut -d= -f2)
  kv=$(grep '^KV_CACHE_DTYPE=' "${ENV_FILE}" | cut -d= -f2)
  maxlen=$(grep '^MAX_MODEL_LEN=' "${ENV_FILE}" | cut -d= -f2)
  seqs=$(grep '^MAX_NUM_SEQS=' "${ENV_FILE}" | cut -d= -f2)
  echo -e "${GREEN}=== 当前配置 ===${NC}"
  echo "  MODEL_NAME      = $model"
  echo "  SERVED_MODEL_NAME = $served"
  echo "  KV_CACHE_DTYPE  = $kv"
  echo "  MAX_MODEL_LEN   = $maxlen"
  echo "  MAX_NUM_SEQS    = $seqs"
  echo "  完整配置: cat ${ENV_FILE}"
}

apply_profile() {
  local name="$1"
  if [[ -z "${PROFILES[$name]:-}" ]]; then
    err "未知配置: $name"
    echo "可用配置:"
    show_profiles
    exit 1
  fi

  info "备份当前 .env -> .env.bak"
  cp "${ENV_FILE}" "${BACKUP_FILE}"

  info "应用配置: $name"
  echo "${PROFILES[$name]}" > "${ENV_FILE}"

  ok "配置已更新。正在重启服务..."
  cd "${SCRIPT_DIR}"
  docker compose up -d --force-recreate 2>&1 | tail -3

  info "等待服务就绪..."
  local waited=0
  for i in {1..180}; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q "200"; then
      ok "服务就绪 (等待 $((i*2))s)"
      echo
      show_current
      echo
      ok "切换完成！测试接口:"
      echo "  curl -s http://192.168.2.11:8000/v1/models"
      echo "  curl -X POST http://192.168.2.11:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"<served_name>\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
      return
    fi
    sleep 2
  done
  warn "服务启动超时，请查看日志: docker compose logs vllm --tail=50"
}

case "${cmd}" in
  list|ls|"")
    show_profiles
    ;;
  current)
    show_current
    ;;
  add)
    name="${1:?用法: $0 add <name> <key>=<val> ...}"
    shift
    if [[ $# -lt 1 ]]; then
      err "请至少提供一个 key=val 参数"
      exit 1
    fi
    # 基于当前 .env 修改
    local new_profile="# ============================================================
# 自定义配置 - ${name}
# ============================================================
"
    if [[ -f "${ENV_FILE}" ]]; then
      while IFS= read -r line; do
        if [[ "$line" =~ ^[A-Z_] ]]; then
          key="${line%%=*}"
          val="${line#*=}"
          # 检查是否在命令行指定了新值
          found=0
          for arg in "$@"; do
            if [[ "${arg%%=*}" == "$key" ]]; then
              new_profile+="${key}=${arg#*=}"$'\n'
              found=1
              break
            fi
          done
          if [[ $found -eq 0 ]]; then
            new_profile+="${key}=${val}"$'\n'
          fi
        fi
      done < "${ENV_FILE}"
    fi
    PROFILES["$name"]="$new_profile"
    # 写入脚本自身
    info "保存配置 '$name' 到 switch_model.sh ..."
    # 追加到 PROFILES 数组
    local escaped_profile
    escaped_profile=$(printf '%s' "$new_profile" | sed "s/\\\\/\\\\\\\\/g")
    # 用 bash 追加
    sed -i "/^# --- 自定义配置 ---/a\\
PROFILES[\"${name}\"]='${new_profile}'" "${BASH_SOURCE[0]}"
    ok "配置 '$name' 已添加。使用: $0 $name 切换"
    ;;
  *)
    apply_profile "$cmd"
    ;;
esac
