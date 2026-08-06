# 一键部署 vLLM + Qwen3.5-35B-A3B-FP8

> 完整执行记录：从环境准备到模型部署到 API 验证的端到端流程。

---

## 1. 当前部署状态

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| Docker 29.6.2 + Compose v5.3.1 | ✅ | 系统已就绪 |
| NVIDIA Driver 570.172.08 / CUDA 12.8 | ✅ | 双 RTX 4090 |
| nvidia-container-toolkit 1.19.1 | ✅ 已装 | 容器可识别 GPU |
| Docker 镜像加速 | ✅ 已配置 | daemon.json 多镜像源 |
| vLLM 镜像 `vllm/vllm-openai:latest` | 🔄 拉取中 | ~10GB |
| 模型 `Qwen/Qwen3.6-35B-A3B-FP8` | 🔄 下载中 | ~35GB |

---

## 2. 已配置 `.env`

```env
MODEL_NAME=Qwen/Qwen3.6-35B-A3B-FP8
SERVED_MODEL_NAME=qwen3.5-35b-a3b
HF_ENDPOINT=https://hf-mirror.com
TENSOR_PARALLEL_SIZE=2          # 35B MoE + 视觉编码器，TP=2 留 KV 空间
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=32768
DTYPE=auto
KV_CACHE_DTYPE=auto
QUANTIZATION=fp8                # 模型已 FP8 量化
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=8192
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
```

---

## 3. 一键部署脚本（已就绪）

```bash
cd /opt/vllm

# 1) 下载模型（已执行，等完成即可）
./download_model.sh qwen3.5-35b-a3b-fp8

# 2) 启动
./scripts/start.sh

# 3) 跟日志
./scripts/logs.sh

# 4) 健康检查
curl http://localhost:8000/v1/models

# 5) 端到端测试（文本/流式/多模态/并发）
python3 /opt/vllm/scripts/test_api.py

# 6) 压测
./scripts/benchmark.sh 32 200

# 7) 停止
./scripts/stop.sh
```

---

## 4. 模型选型

`Qwen/Qwen3.6-35B-A3B-FP8` 是 Qwen3.5 系列多模态 MoE 模型：
- **35B 总参 / 3B 激活**：推理速度接近 3B 模型
- **FP8 量化**：单卡 18GB 即可装下（TP=2 后单卡 ~18GB 权重 + 大量 KV 空间）
- **多模态**：原生支持图文理解（image-text-to-text）
- **架构**：`Qwen3_5MoeForConditionalGeneration`
- **许可**：Apache 2.0，可商用

---

## 5. 硬件部署

| GPU | TP | 显存分配 |
| --- | --- | --- |
| RTX 4090 #0 | rank 0 | 18GB 模型 + KV |
| RTX 4090 #1 | rank 1 | 18GB 模型 + KV |

`GPU_MEMORY_UTILIZATION=0.90` 下，每张卡可用约 44GB，远大于权重 18GB，预留 26GB 供 KV 缓存（MAX_LEN=32K × 64 并发）。

---

## 6. 访问入口

部署成功后，OpenAI 兼容 API：

| 项目 | 值 |
| --- | --- |
| Base URL | `http://<本机IP>:8000/v1` |
| Chat 端点 | `POST /v1/chat/completions` |
| 模型名 | `qwen3.5-35b-a3b` |
| API Key | (空) |
| 鉴权 | 按需设置 `.env` 中 `VLLM_API_KEY` |

---

## 7. 快速调用

**cURL**：
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b-a3b",
    "messages": [{"role":"user","content":"用一句话介绍 MoE 架构"}],
    "max_tokens": 128
  }'
```

**Python**：
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
print(client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{"role":"user","content":"Hello"}],
    max_tokens=128,
).choices[0].message.content)
```

**多模态**（传入图片）：
```python
import base64
with open("test.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
resp = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```
