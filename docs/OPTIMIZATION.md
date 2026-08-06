# 参数优化指南

> 目标：在你的 2 × RTX 4090 上，把 vLLM 调到 **低延迟 + 高吞吐** 的平衡点。

---

## 一、调参总览

| 维度 | 关键参数 | 倾向 |
| --- | --- | --- |
| 显存 | `GPU_MEMORY_UTILIZATION`、`MAX_MODEL_LEN`、`KV_CACHE_DTYPE` | 显存不足 → 减小/降精度 |
| 吞吐 | `MAX_NUM_SEQS`、`MAX_NUM_BATCHED_TOKENS` | 在线服务 → 调大 |
| 延迟 | `BLOCK_SIZE`、`enable_prefix_caching` | 长 prompt 复用 → 开 |
| 并行 | `TENSOR_PARALLEL_SIZE` | 单卡够 → TP=1，通信更少 |
| 预填充 | `ENABLE_CHUNKED_PREFILL` | 几乎总是 true |

---

## 二、显存管理

### 1. `GPU_MEMORY_UTILIZATION`（默认 0.90）
- 表示 vLLM 可用显存占总显存的 **比例**，不是绝对值。
- 长上下文 / 加大 `MAX_NUM_SEQS` → 需更低（如 0.85）。
- 已 OOM → 降到 0.80 试试。

### 2. `MAX_MODEL_LEN`
- 单请求最大 token 数（输入+输出）。
- 设得越大，KV 缓存占比越多 → 并发能力下降。
- **建议**：在满足业务前提下尽量小（对话 4K~8K，长文 32K~128K）。

### 3. `KV_CACHE_DTYPE`
- `auto`（默认）：按模型权重类型。
- `fp8`：KV 用 FP8，节省 30~50% 显存，几乎无损。
- 适合大模型或长上下文。

### 4. 量化
- 通过 `--quantization awq` 加载 4bit 量化模型，**显存接近 1/4**。
- 也可以不下载量化版，启动时加 `--quantization bitsandbytes` 即时 NF4 量化。

---

## 三、吞吐调优

### 1. `MAX_NUM_SEQS`（同时处理的请求数）
- 越大 → 吞吐越高，但单请求延迟略升。
- 经验：
  - 7B 单卡：128~512
  - 14B 单卡：64~256
  - 32B 双卡：32~128
  - 70B 双卡：16~64

### 2. `MAX_NUM_BATCHED_TOKENS`（单批最大 token）
- 调度器在每步最多预填充+解码的 token 总数。
- 经验：4K~16K。
- 太大会让单步 GPU 计算过满，**对小请求反而不公平**。

### 3. `BLOCK_SIZE`（PagedAttention block 大小）
- 16 是默认，**长上下文 + 大模型可调到 32**。
- 太小 → block 表大；太大 → 碎片。

### 4. `ENABLE_PREFIX_CACHING=true`
- **强烈建议开启**。相同前缀（系统提示词、few-shot、工具定义）只算一次。
- RAG 场景效果显著。

### 5. `ENABLE_CHUNKED_PREFILL=true`
- **强烈建议开启**。把长 prompt 切成块，与解码请求混合调度，提升利用率。

### 6. `SWAP_SPACE`（CPU 交换，GB）
- 当 KV 满了，会把部分请求 swap 到 CPU RAM。
- 默认 4 GB；机器内存大可调高到 8~16，**降低 OOM/拒绝率**。

### 7. 投机解码（推测执行）
- 需要搭配小 draft 模型：传 `--speculative-model`、`--num-speculative-tokens 5`。
- 适合批量同质化请求，能提速 1.5~2.5x。

---

## 四、并行策略

| 场景 | 推荐 |
| --- | --- |
| 7B/14B | `TENSOR_PARALLEL_SIZE=1`（单卡） |
| 32B 单卡放不下 | `TENSOR_PARALLEL_SIZE=2`（双卡 TP=2） |
| 70B | TP=2（4090），但 KV 较紧张 |
| 多请求真正独立 | 考虑 **PIPELINE_PARALLEL_SIZE** 或多实例（见下） |

> **多实例 vs TP**：当一张卡能装下模型时，**开多个实例 + 负载均衡** 往往比 TP 吞吐更高（无通信开销）。

启动两个 vLLM 实例示例（用不同端口 + nginx/upstream 负载均衡）：

```bash
# 终端 1
docker compose -p vllm-1 up -d

# 终端 2
VLLM_GPU=0 PORT=8001 docker compose -p vllm-2 -f docker-compose.yml up -d
```

---

## 五、网络与解码参数

请求端可控（不需重启 vLLM）：

- `temperature`：0 表示贪心，0.7~1.0 平衡，>1.5 发散。
- `top_p`：通常 0.9~0.95。
- `top_k`：40~100 常用。
- `repetition_penalty`：1.0~1.2。
- `max_tokens`：单请求输出上限。
- `stop`：自定义停止符。
- `stream`：true/false，开启后拿到 SSE 流。
- `seed`：固定随机种子，结果可复现。

> **Tip**：在 batch 场景把 `temperature=0` 能显著提升吞吐（vLLM 走 greedy path）。

---

## 六、监控与瓶颈定位

### 1. 关键指标
- `vllm:gpu_cache_usage` < 1，缓存没打满
- `vllm:num_requests_running` ≈ `MAX_NUM_SEQS` 时已饱和
- `vllm:request_success_total` / `request_failure_total`
- `vllm:e2e_request_latency_seconds` p50/p95/p99

开启 Prometheus：

```yaml
# docker-compose.yml 中 vllm 服务增加
command:
  - --model=...
  - --enable-metrics
  - --metrics-port=8001
ports:
  - "8001:8001"
```

### 2. 看 GPU
```bash
nvidia-smi -l 1        # 1s 刷新
nvidia-smi pmon -s u   # 进程粒度
```

### 3. 压测
```bash
./scripts/benchmark.sh 32 200
# 或 vllm bench
vllm bench serve --model <repo> --backend openai-chat \
  --dataset-name sonnet --max-concurrency 32 --num-prompts 200
```

---

## 七、调参速查表（推荐起点）

| 模型 | TP | GPU_UTIL | MAX_LEN | MAX_NUM_SEQS | BATCH_TOK |
| --- | --- | --- | --- | --- | --- |
| 7B BF16 | 1 | 0.90 | 8192 | 256 | 8192 |
| 14B BF16 | 1 | 0.90 | 8192 | 128 | 8192 |
| 32B AWQ | 1 | 0.92 | 8192 | 96 | 4096 |
| 32B BF16 | 2 | 0.90 | 8192 | 64 | 4096 |
| 70B AWQ | 2 | 0.92 | 8192 | 32 | 2048 |
| 128K 长文 | 1/2 | 0.95 | 131072 | 16 | 2048 |

---

## 八、常见调优误区

1. **TP 越大越好？** — 单卡放得下就别 TP，通信开销会让延迟上升。
2. **`MAX_NUM_SEQS` 越大越好？** — 超过 GPU 并行能力后会被调度器排队，**延迟上升**但吞吐未必涨。
3. **`MAX_MODEL_LEN` 拉到最大？** — 会挤占 KV 缓存，**直接降低并发能力**。
4. **不开 prefix caching？** — 同一 system prompt 反复算，浪费 30%+ 算力。
5. **不开 chunked prefill？** — 长 prompt 会让 prefill 阶段独占 GPU，**其他请求卡死**。
