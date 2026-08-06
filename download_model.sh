#!/usr/bin/env bash
# ============================================================
# 一键下载 Hugging Face / ModelScope 模型
# 用法:
#   ./download_model.sh qwen2.5-7b
#   ./download_model.sh llama3-8b
#   ./download_model.sh custom <repo_id>
#   ./download_model.sh list
#   ./download_model.sh ms <repo_id>      # ModelScope 渠道
# ============================================================

set -euo pipefail

# -------- 颜色 --------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*"; }

# -------- 路径 --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODELS_DIR}"

# -------- 模型清单 --------
# 命名: short_name|repo_id|precision|size_gb|tag
declare -A MODELS=(
  ["qwen2.5-7b"]="Qwen/Qwen2.5-7B-Instruct|bf16|15|通用对话"
  ["qwen2.5-14b"]="Qwen/Qwen2.5-14B-Instruct|bf16|30|通用对话"
  ["qwen2.5-32b"]="Qwen/Qwen2.5-32B-Instruct|bf16|65|通用对话(需2卡+)"
  ["qwen2.5-72b"]="Qwen/Qwen2.5-72B-Instruct|bf16|145|通用对话(需4卡+)"
  ["qwen2.5-coder-7b"]="Qwen/Qwen2.5-Coder-7B-Instruct|bf16|15|代码"
  ["qwen2.5-coder-14b"]="Qwen/Qwen2.5-Coder-14B-Instruct|bf16|30|代码"
  ["qwen2.5-coder-32b"]="Qwen/Qwen2.5-Coder-32B-Instruct|bf16|65|代码(需2卡+)"

  ["llama3-8b"]="meta-llama/Meta-Llama-3-8B-Instruct|bf16|16|通用对话(需HF同意)"
  ["llama3-70b"]="meta-llama/Meta-Llama-3-70B-Instruct|bf16|140|通用对话(需4卡+)"

  ["deepseek-v2.5"]="deepseek-ai/DeepSeek-V2.5|bf16|240|MoE 236B(需多卡)"
  ["deepseek-coder-v2"]="deepseek-ai/DeepSeek-Coder-V2-Instruct|bf16|240|MoE 236B(需多卡)"

  ["yi-1.5-9b"]="01-ai/Yi-1.5-9B-Chat|bf16|18|中英双语"
  ["gemma2-9b"]="google/gemma-2-9b-it|bf16|18|通用对话"

  ["qwen2.5-7b-awq"]="Qwen/Qwen2.5-7B-Instruct-AWQ|awq|8|低显存"
  ["qwen2.5-14b-awq"]="Qwen/Qwen2.5-14B-Instruct-AWQ|awq|13|低显存"
  ["llama3-8b-awq"]="lmstudio-community/Meta-Llama-3-8B-Instruct-AWQ|awq|8|低显存"

  # Qwen3.5 多模态 MoE（35B 总参 / 3B 激活，FP8 量化，支持图片理解）
  ["qwen3.5-35b-a3b-fp8"]="Qwen/Qwen3.6-35B-A3B-FP8|fp8|35|多模态MoE/图文"
  ["qwen3.5-35b-a3b"]="Qwen/Qwen3.6-35B-A3B|bf16|70|多模态MoE/图文(需2卡+)"

  # Qwen3.6 系列（ dense 模型，推理速度快）
  ["qwen3.6-27b"]="Qwen/Qwen3.6-27B|bf16|54|通用对话/思考模型"
  ["qwen3.6-27b-awq"]="Qwen/Qwen3.6-27B-Instruct-AWQ|awq|14|低显存/通用对话"
)

# 引入下载函数实现
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=download_model_lib.sh
source "${SCRIPT_DIR}/download_model_lib.sh"

# -------- 命令分发 --------
cmd="${1:-help}"
shift || true

case "${cmd}" in
  list|ls)
    echo -e "${GREEN}=== 可用模型快捷名 ===${NC}"
    printf "%-22s %-55s %-10s %-8s %s\n" "short_name" "repo_id" "precision" "size(GB)" "tag"
    echo "--------------------------------------------------------------------------------------------------------"
    for k in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
      IFS='|' read -r repo prec size tag <<< "${MODELS[$k]}"
      printf "%-22s %-55s %-10s %-8s %s\n" "$k" "$repo" "$prec" "$size" "$tag"
    done
    echo
    echo "用法: $0 <short_name>     # 例如: $0 qwen2.5-7b"
    echo "      $0 custom <repo_id> # 任意 HF repo_id"
    echo "      $0 ms <repo_id>     # ModelScope 渠道"
    echo "      $0 sources          # 查看可用下载源"
    echo "      $0 list             # 列出可用模型"
    ;;

  sources|src)
    list_sources
    ;;

  custom)
    repo_id="${1:?用法: $0 custom <repo_id>}"
    download_hf "${repo_id}"
    ;;

  download)
    # download.sh <source> <repo_id>
    source_name="${1:?用法: $0 download <source> <repo_id>}"
    repo_id="${2:?用法: $0 download <source> <repo_id>}"
    case "$source_name" in
      hf|huggingface)
        download_hf "${repo_id}"
        ;;
      ms|modelscope)
        download_ms "${repo_id}"
        ;;
      *)
        DOWNLOAD_SOURCE="$source_name" download_hf "${repo_id}"
        ;;
    esac
    ;;

  ms)
    repo_id="${1:?用法: $0 ms <repo_id>}"
    download_ms "${repo_id}"
    ;;

  help|-h|--help|"")
    cat <<EOF
${GREEN}vLLM 一键模型下载 - 多途径 + 断点续传${NC}

用法:
  $0 list                           列出全部预设模型
  $0 sources                        查看可用下载源
  $0 <short_name>                   下载预设模型 (e.g. qwen3.6-27b)
  $0 custom <repo_id>               下载任意 HF repo (自动多源+续传)
  $0 download <source> <repo_id>    指定源下载
  $0 ms <repo_id>                   通过 ModelScope 下载

下载源 (可通过 DOWNLOAD_SOURCE 环境变量切换):
  hf-mirror     国内镜像站 (默认, 推荐)
  hf-official   HuggingFace 官方
  hf-mirror-2   备用镜像
  modelscope    ModelScope 平台

示例:
  $0 qwen3.6-27b
  $0 custom Qwen/Qwen2.5-7B-Instruct
  $0 download hf-official Qwen/Qwen3.6-27B    # 指定官方源
  DOWNLOAD_SOURCE=hf-official $0 qwen3.6-27b  # 切换默认源
  $0 ms Qwen/Qwen2.5-7B-Instruct

断点续传:
  脚本自动检测已存在的文件，中断后重新执行即可继续下载。
  所有下载均支持 curl -C - 续传 和 huggingface_hub resume_download。

下载完成后, 在 .env 中设置:
  MODEL_NAME=<repo_id>
然后执行:  docker compose up -d
EOF
    ;;

  *)
    if [[ -n "${MODELS[$cmd]+x}" ]]; then
      IFS='|' read -r repo prec size tag <<< "${MODELS[$cmd]}"
      info "选中预设: ${cmd}  ->  ${repo}  (${prec}, ~${size}GB, ${tag})"
      download_hf "${repo}"
    else
      err "未知命令或模型: ${cmd}"
      echo "执行 '$0 list' 查看可用模型"
      exit 1
    fi
    ;;
esac
