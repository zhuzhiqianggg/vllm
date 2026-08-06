# ============================================================
# 常用 vLLM 启动配置预设（命令行 / compose 改写参考）
# ============================================================

# -------- 7B~14B 单卡 (Qwen2.5 / Llama3) --------
# GPU 需求: 1 x 24GB+
# 上下文: 8K
PARAMS_SMALL="
  --model Qwen/Qwen2.5-7B-Instruct
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.90
  --max-model-len 8192
  --dtype bfloat16
  --max-num-seqs 256
  --max-num-batched-tokens 8192
  --enable-prefix-caching
  --enable-chunked-prefill
  --kv-cache-dtype auto
"

# -------- 32B 单卡 AWQ 量化 --------
# GPU 需求: 1 x 24GB+ (AWQ 4bit 约 18GB)
PARAMS_32B_AWQ="
  --model Qwen/Qwen2.5-32B-Instruct-AWQ
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.92
  --max-model-len 8192
  --dtype auto
  --quantization awq
  --max-num-seqs 128
  --max-num-batched-tokens 4096
  --enable-prefix-caching
"

# -------- 70B 双卡 tensor parallel --------
# GPU 需求: 2 x 48GB (4090/A6000)
PARAMS_70B_TP2="
  --model meta-llama/Meta-Llama-3-70B-Instruct
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.90
  --max-model-len 8192
  --dtype bfloat16
  --max-num-seqs 64
  --max-num-batched-tokens 4096
  --enable-prefix-caching
"

# -------- 高并发吞吐优化配置 (online serving) --------
PARAMS_HIGH_THROUGHPUT="
  --max-num-seqs 512
  --max-num-batched-tokens 16384
  --block-size 16
  --enable-prefix-caching
  --enable-chunked-prefill
  --num-speculative-tokens 5
  --swap-space 4
"

# -------- 长上下文 128K 配置 --------
PARAMS_LONG_CTX="
  --max-model-len 131072
  --max-num-seqs 16
  --max-num-batched-tokens 2048
  --block-size 32
  --gpu-memory-utilization 0.95
"
