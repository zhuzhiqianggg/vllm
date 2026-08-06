# Qwen3.5-35B-A3B-FP8 部署完成 ✅

> 完整端到端部署已通过测试，所有 API 可用。

---

## 一、当前服务状态

| 项目 | 值 |
| --- | --- |
| **模型** | Qwen3.5-35B-A3B-FP8 (Qwen3.5 MoE 多模态, 35B 总参/3B 激活, FP8) |
| **vLLM 版本** | 0.26.0 |
| **服务地址** | `http://localhost:8000`（容器内 `0.0.0.0:8000`） |
| **OpenAI 兼容 base_url** | `http://<本机IP>:8000/v1` |
| **客户端模型名** | `qwen3.5-35b-a3b` |
| **API Key** | （未设） |
| **容器状态** | ✅ Up 22 minutes (healthy) |
| **GPU 占用** | 43.7 GB / 49 GB × 2 |
| **GPU 驱动** | 580.173.02 (CUDA 13.0) |

---

## 二、性能实测

| 场景 | 指标 |
| --- | --- |
| 冷启动到就绪 | ~4 分钟（含 torch.compile） |
| 单请求解码速度 | **134 tokens/s**（流式） |
| 并发 16，64 请求 | **8.67 req/s**, **1109 output tokens/s** |
| p50 延迟 | 5.22 s |
| p95 延迟 | 7.38 s |
| 视觉问答 | 5.93 s（252+781 tokens） |
| KV 缓存容量 | 2,140,218 tokens（MAX_LEN=32K 时最大并发 65×） |

---

## 三、API 访问方式

### 1. cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b-a3b",
    "messages": [{"role":"user","content":"用一句话介绍 MoE 架构"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### 2. Python OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{"role":"user","content":"Hello"}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

### 3. 流式

```python
stream = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[{"role":"user","content":"写一首五言绝句"}],
    max_tokens=1024, stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 4. 多模态（图文）

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
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

### 5. 网络访问

容器监听 `0.0.0.0:8000`，外部访问：
```bash
# 同网段
curl http://<主机IP>:8000/v1/models
```

---

## 四、运维命令

| 操作 | 命令 |
| --- | --- |
| 启动 | `./scripts/start.sh` |
| 停止 | `./scripts/stop.sh` |
| 实时日志 | `./scripts/logs.sh` |
| 健康检查 | `curl http://localhost:8000/v1/models` |
| 完整测试 | `python3 scripts/test_api.py` |
| 压测 | `./scripts/benchmark.sh 32 200` |
| GPU 监控 | `nvidia-smi -l 1` |

---

## 五、本次部署关键步骤回顾

1. **环境补齐**：安装 `nvidia-container-toolkit`，配置 Docker 镜像加速
2. **驱动升级**：从 570.172.08 升级到 580.173.02（支持 CUDA 13.0，匹配 vLLM 0.26.0）
3. **模型下载**：`Qwen/Qwen3.6-35B-A3B-FP8` 35GB，通过 `hf-mirror.com`（禁用 xet）下载 ~50 分钟
4. **配置适配**：`docker-compose.yml` 改用 `runtime: nvidia`（CDI 在驱动升级后未自动重建）；`.env` 设为 `TP=2 / FP8 / 32K 上下文`
5. **启动与测试**：vLLM 加载 17.5s，torch.compile 50s，CUDA graph 7s，最终 ~4 分钟就绪
6. **API 验证**：5 类测试（健康/文本/流式/视觉/并发 64）全部通过

---

## 六、文件交付清单

```
/opt/vllm/
├── docker-compose.yml          # vLLM 编排（已修复 runtime 路径）
├── .env                         # 运行时配置（已设好 35B-A3B-FP8）
├── download_model.sh            # 一键下载（含 qwen3.5-35b-a3b-fp8 短名）
├── download_model_lib.sh
├── README.md                    # 总览
├── PLAN.md                      # 规划文档
├── DEPLOY_NOTES.md              # 本次部署笔记
├── FINAL_STATUS.md              # 本文档
├── docs/
│   ├── MODELS.md
│   ├── OPTIMIZATION.md
│   ├── EXAMPLES.md
│   └── TROUBLESHOOTING.md
├── configs/presets.sh
├── scripts/
│   ├── start.sh / stop.sh / logs.sh
│   ├── benchmark.sh / bench_http.py
│   └── test_api.py              # 端到端测试（已验证通过）
├── models/                      # 35GB 模型已就位
│   └── Qwen__Qwen3.6-35B-A3B-FP8/
└── logs/                        # 日志目录
```

---

## 七、热切换其他模型

```bash
# 1) 下载（preset 已有 17 个，或任意 HF repo）
./download_model.sh qwen2.5-7b
# 或 ./download_model.sh custom <repo_id>

# 2) 改 .env
#    MODEL_NAME=/root/.cache/huggingface/<新模型目录>
#    SERVED_MODEL_NAME=xxx
#    TENSOR_PARALLEL_SIZE 按需调整

# 3) 重启
./scripts/stop.sh && ./scripts/start.sh
```

---

## 八、注意事项

1. **思考模式**：Qwen3.5 默认会进行 Chain-of-Thought，输出包含 `<think>...</think>` 块。生产中可：
   - 加 `max_tokens` 限制
   - 用 system prompt 强制简洁
   - 提取 `</think>` 后的内容
2. **GPU 显存**：当前使用 ~88%，后续如需扩展 `MAX_NUM_SEQS` 要注意
3. **生产建议**：在 `.env` 中设置 `VLLM_API_KEY` 启用鉴权
