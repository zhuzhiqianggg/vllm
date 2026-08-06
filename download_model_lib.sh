#!/usr/bin/env bash
# ============================================================
# vLLM 模型下载库 - 多途径 + 断点续传
# 支持: HuggingFace 官方 / HF 镜像站 / ModelScope / 直接URL
# 所有下载均支持断点续传
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
mkdir -p "${MODELS_DIR}"

need() {
  command -v "$1" >/dev/null 2>&1 || { err "缺少依赖: $1，请先安装"; exit 1; }
}

# -------- 下载源配置 --------
# 按优先级排列的 HF 下载源
declare -A HF_SOURCES
HF_SOURCES["hf-official"]="https://huggingface.co"
HF_SOURCES["hf-mirror"]="https://hf-mirror.com"
HF_SOURCES["hf-mirror-2"]="https://hf.co"

# 下载源顺序（默认用镜像站）
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-hf-mirror}"

# -------- 通用函数 --------

# 选择可用的 Python
select_python() {
  if python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "python3"
  elif python -c "import huggingface_hub" 2>/dev/null; then
    echo "python"
  else
    info "安装 huggingface_hub..."
    python3 -m pip install --user --quiet huggingface_hub 2>/dev/null
    echo "python3"
  fi
}

# 清理不完整的下载文件
clean_incomplete() {
  local dir="$1"
  find "$dir" -name "*.incomplete" -delete 2>/dev/null
  find "$dir" -name "*.lock" -delete 2>/dev/null
  find "$dir/.cache" -name "*.lock" -delete 2>/dev/null
}

# 检查文件是否完整（对比远程大小）
check_file_complete() {
  local file="$1"
  local min_size="${2:-100000000}"  # 默认 100MB 以上才算完整
  if [ ! -f "$file" ]; then
    return 1
  fi
  local size
  size=$(stat -c%s "$file" 2>/dev/null || echo 0)
  if [ "$size" -lt "$min_size" ]; then
    return 1
  fi
  return 0
}

# -------- 方式1: huggingface_hub SDK 下载（推荐） --------
# 支持断点续传，自动重试
download_hf_sdk() {
  local repo_id="$1"
  local target="$2"
  local PY="$3"

  info "使用 huggingface_hub SDK 下载..."
  info "仓库: $repo_id"
  info "目标: $target"

  # 检查是否已完整下载
  if [ -f "$target" ] && check_file_complete "$target/config.json" 10000; then
    local safetensors_count
    safetensors_count=$(ls "$target"/model-*.safetensors 2>/dev/null | wc -l)
    if [ "$safetensors_count" -ge 10 ]; then
      ok "模型已存在且完整 ($safetensors_count 个 safetensors 文件)"
      return 0
    fi
    info "模型部分存在，将断点续传..."
  fi

  local max_retries=3
  local retry=0

  while [ $retry -lt $max_retries ]; do
    retry=$((retry + 1))
    info "尝试 $retry/$max_retries ..."

    # 清理锁文件
    clean_incomplete "$target"

    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_ENDPOINT="${HF_SOURCES[$DOWNLOAD_SOURCE]}" \
    "$PY" - <<PY 2>&1
import os
import sys
from huggingface_hub import snapshot_download

target = "${target}"
repo_id = "${repo_id}"

try:
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        max_workers=4,
        allow_patterns=None,
        ignore_patterns=["*.gguf", "*.msgpack", "*.h5", "*.onnx", "*.ot"],
        resume_download=True,
    )
    print(f"SUCCESS:{path}")
    sys.exit(0)
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
PY

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
      ok "下载完成"
      return 0
    fi

    warn "下载失败 (exit=$exit_code)，将重试..."
    sleep 5
  done

  return 1
}

# -------- 方式2: 直接 HTTP 下载（备用，支持断点续传）--------
download_hf_http() {
  local repo_id="$1"
  local target="$2"
  local source="${3:-hf-mirror}"

  local base_url="${HF_SOURCES[$source]}/${repo_id}/resolve/main"
  info "使用 HTTP 直接下载 (源: $source)..."
  info "基础 URL: $base_url"

  # 获取文件列表
  local file_list
  file_list=$(curl -s "${HF_SOURCES[$source]}/api/models/${repo_id}" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
files = [s['rfilename'] for s in d.get('siblings',[]) if s['rfilename'].endswith('.safetensors') or s['rfilename'] in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','generation_config.json','chat_template.jinja','merges.txt','vocab.json']]
print('\n'.join(files))
" 2>/dev/null)

  if [ -z "$file_list" ]; then
    err "无法获取文件列表"
    return 1
  fi

  local total_files
  total_files=$(echo "$file_list" | wc -l)
  info "共 $total_files 个文件需要下载"

  local success=0
  local failed=0

  while IFS= read -r fname; do
    [ -z "$fname" ] && continue
    local dest="${target}/${fname}"

    # 检查是否已存在且完整
    if [ -f "$dest" ] && check_file_complete "$dest" 100000000; then
      success=$((success + 1))
      continue
    fi

    # 检查部分下载，计算续传起点
    local resume_byte=0
    if [ -f "$dest" ]; then
      resume_byte=$(stat -c%s "$dest" 2>/dev/null || echo 0)
      if [ "$resume_byte" -gt 0 ]; then
        info "续传 $fname (已有 ${resume_byte} bytes)..."
      fi
    else
      info "下载 $fname ..."
    fi

    # 支持续传下载
    local curl_args=""
    if [ "$resume_byte" -gt 0 ]; then
      curl_args="-C -"
    fi

    if curl -L --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 600 \
      $curl_args \
      -o "$dest" \
      "${base_url}/${fname}" 2>/dev/null; then
      success=$((success + 1))
      local size
      size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
      if [ "$size" -gt 1000000 ]; then
        local size_mb=$((size / 1024 / 1024))
        echo "  ✅ $fname: ${size_mb}MB"
      fi
    else
      failed=$((failed + 1))
      warn "❌ $fname 下载失败"
      # 删除不完整文件
      rm -f "$dest"
    fi
  done <<< "$file_list"

  echo ""
  info "下载结果: 成功 $success/$total_files, 失败 $failed"

  if [ "$failed" -eq 0 ]; then
    ok "全部下载完成"
    return 0
  else
    warn "部分文件下载失败，可重试"
    return 1
  fi
}

# -------- 方式3: ModelScope 下载 --------
download_modelscope() {
  local repo_id="$1"
  local target="$2"

  info "使用 ModelScope 下载..."
  info "仓库: $repo_id"

  if ! python3 -c "import modelscope" 2>/dev/null; then
    info "安装 modelscope..."
    python3 -m pip install --user --quiet modelscope 2>/dev/null
  fi

  python3 << PY 2>&1
from modelscope import snapshot_download
import os

# 构造 ModelScope 兼容的路径
# HF 路径: Qwen/Qwen3.6-27B -> ModelScope 路径: Qwen/Qwen3.6-27B
ms_repo = "${repo_id}"

try:
    path = snapshot_download(
        ms_repo,
        cache_dir="${MODELS_DIR}",
    )
    print(f"SUCCESS:{path}")
except Exception as e:
    print(f"ERROR:{e}")
    # 尝试不同的命名格式
    try:
        # 尝试把 / 替换为 -
        alt_name = ms_repo.replace('/', '-')
        path = snapshot_download(
            alt_name,
            cache_dir="${MODELS_DIR}",
        )
        print(f"SUCCESS(alt):{path}")
    except Exception as e2:
        print(f"ERROR2:{e2}")
        import sys
        sys.exit(1)
PY

  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    ok "ModelScope 下载完成"
    return 0
  fi

  err "ModelScope 下载失败"
  return 1
}

# -------- 主下载函数（自动选择最优方式）--------
download_hf() {
  local repo_id="$1"
  local force_method="${2:-}"

  need python3

  local PY
  PY=$(select_python)

  local target="${MODELS_DIR}/${repo_id//\//__}"
  mkdir -p "$target"

  info "============================================"
  info "下载模型: $repo_id"
  info "目标目录: $target"
  info "下载源:   ${DOWNLOAD_SOURCE} (可通过 DOWNLOAD_SOURCE 环境变量切换)"
  info "============================================"

  # 检查已存在的模型
  local safetensors_count
  safetensors_count=$(find "$target" -name "*.safetensors" 2>/dev/null | wc -l)
  if [ "$safetensors_count" -ge 10 ]; then
    info "检测到已有 $safetensors_count 个 safetensors 文件，将断点续传..."
  fi

  local success=0

  # 方式选择
  if [ "$force_method" = "sdk" ] || [ "$force_method" = "" ]; then
    # 优先用 SDK（支持完善的断点续传和缓存）
    if download_hf_sdk "$repo_id" "$target" "$PY"; then
      success=1
    else
      warn "SDK 下载失败，尝试 HTTP 方式..."
    fi
  fi

  if [ "$success" -eq 0 ] && [ "$force_method" != "sdk" ]; then
    # 备用：直接 HTTP 下载（支持续传）
    if download_hf_http "$repo_id" "$target" "$DOWNLOAD_SOURCE"; then
      success=1
    else
      warn "HTTP 下载也失败..."
    fi
  fi

  if [ "$success" -eq 0 ] && [ "$force_method" = "" ]; then
    # 最后尝试 ModelScope
    warn "尝试 ModelScope 作为备用源..."
    download_modelscope "$repo_id" "$target" || true
  fi

  # 最终检查
  echo ""
  safetensors_count=$(find "$target" -name "*.safetensors" 2>/dev/null | wc -l)
  if [ "$safetensors_count" -ge 10 ]; then
    local total_size
    total_size=$(du -sb "$target" 2>/dev/null | cut -f1)
    local size_gb=$((total_size / 1024 / 1024 / 1024))
    ok "============================================"
    ok "✅ 模型下载完成！"
    ok "   路径: $target"
    ok "   文件: $safetensors_count 个 safetensors"
    ok "   大小: ${size_gb}GB"
    ok "============================================"
    echo ""
    ok "接下来在 .env 中设置:"
    echo "   MODEL_NAME=${repo_id}"
    echo "或:"
    echo "   HF_MODEL_PATH=${target}"
    return 0
  else
    err "模型下载不完整 (只有 $safetensors_count 个文件)，请重试"
    return 1
  fi
}

# ModelScope 专用下载
download_ms() {
  local repo_id="$1"
  local target="${MODELS_DIR}/ms_${repo_id//\//__}"
  mkdir -p "$target"
  download_modelscope "$repo_id" "$target"
}

# 列出可用下载源
list_sources() {
  echo "可用下载源:"
  echo "  hf-official:   https://huggingface.co (官方，国内较慢)"
  echo "  hf-mirror:     https://hf-mirror.com (国内镜像，推荐)"
  echo "  hf-mirror-2:   https://hf.co (备用镜像)"
  echo "  modelscope:    ModelScope (国内平台)"
  echo ""
  echo "当前下载源: ${DOWNLOAD_SOURCE}"
  echo ""
  echo "切换方式:"
  echo "  DOWNLOAD_SOURCE=hf-official ./download_model.sh <model>"
  echo "  DOWNLOAD_SOURCE=hf-mirror ./download_model.sh <model>"
}
