# vLLM 部署方案规划

> 面向 **单机 2 × RTX 4090 (48 GB × 2)** 环境的完整 vLLM 部署计划与文档导航。

---

## 1. 项目目标

| 目标 | 衡量 |
| --- | --- |
| 一键拉起 vLLM | `./scripts/start.sh` 后 5 分钟内可调用 |
| 一键下载模型 | `./download_model.sh <short_name>` 即可 |
| 模型可热切换 | 改 `.env` → `docker compose up -d` |
| OpenAI 兼容 | 现有 OpenAI 客户端零代码改动接入 |
| 性能可调 | 文档 + 预设参数覆盖 90% 场景 |

---

## 2. 已交付文件清单

```
/opt/vllm/
├── docker-compose.yml         # vLLM 容器编排（含 2 卡 GPU）
├── .env.example               # 环境变量模板
├── download_model.sh          # 一键下载入口
├── download_model_lib.sh      # 下载实现
├── README.md                  # 总览与快速开始
├── docs/
│   ├── MODELS.md              # 模型选型指南
│   ├── OPTIMIZATION.md        # 参数调优指南
│   ├── EXAMPLES.md            # API 调用示例
│   └── TROUBLESHOOTING.md     # 故障排查
├── configs/presets.sh         # 启动参数预设
├── scripts/
│   ├── start.sh / stop.sh
│   ├── logs.sh
│   ├── benchmark.sh
│   └── bench_http.py
├── models/                    # 模型缓存（挂载到容器）
└── logs/                      # 容器日志
```

---

## 3. 部署流程（4 步走）

### 步骤 A：环境准备（一次）
1. 安装 NVIDIA Container Toolkit（README 第三章有脚本）
2. 验证：`docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi`
3. `cd /opt/vllm && cp .env.example .env`

### 步骤 B：选型 + 下载
1. 阅读 `docs/MODELS.md` 选定模型
2. 执行 `./download_model.sh list` 看短名
3. `./download_model.sh qwen2.5-7b`（按需替换）

### 步骤 C：启动 + 调参
1. 改 `.env` 中的 `MODEL_NAME` / `TENSOR_PARALLEL_SIZE` / `MAX_MODEL_LEN` / `MAX_NUM_SEQS`
2. `./scripts/start.sh`
3. `./scripts/logs.sh` 看日志，看到 `Application startup complete` 即成功
4. `curl http://localhost:8000/v1/models` 验证

### 步骤 D：接入应用
- 现有 OpenAI 客户端：把 `base_url` 改成 `http://<host>:8000/v1`，`api_key` 填 `EMPTY`（或你的 VLLM_API_KEY）
- 参考 `docs/EXAMPLES.md`

---

## 4. 推荐起点（4090 ×2 场景）

| 用途 | 推荐模型 | 关键参数 |
| --- | --- | --- |
| 通用中文对话 | Qwen2.5-14B-Instruct | TP=1, MAX_LEN=8192, MAX_NUM_SEQS=128 |
| 极致中文质量 | Qwen2.5-32B-Instruct | TP=2, MAX_LEN=8192, MAX_NUM_SEQS=64 |
| 代码生成 | Qwen2.5-Coder-14B-Instruct | TP=1, MAX_LEN=16384, MAX_NUM_SEQS=64 |
| 低显存/高并发 | Qwen2.5-7B-Instruct-AWQ | TP=1, MAX_LEN=8192, MAX_NUM_SEQS=512 |
| 128K 长文 | Qwen2.5-14B-Instruct | MAX_LEN=131072, MAX_NUM_SEQS=16, KV=fp8 |

---

## 5. 性能目标（7B BF16 单卡参考）

| 指标 | 期望 |
| --- | --- |
| Prefill 吞吐 | 8K~15K tokens/s |
| Decode 吞吐 | 1.5K~3K tokens/s |
| 单请求 TTFT | < 200 ms（短 prompt） |
| 并发 32 p99 | < 1.5 s |
| 冷启动 | < 60 s（已下载模型） |

实际数值会随 prompt 长度 / 模型大小波动，**基准测试**：`./scripts/benchmark.sh 32 200`。

---

## 6. 升级与扩展

- **升级镜像**：`docker pull vllm/vllm-openai:latest && docker compose up -d`
- **多实例**：复制 compose 文件改 `PORT`，用 nginx upstream 负载均衡。
- **API 网关**：在前面加 nginx/openresty 做限流、鉴权、路由。
- **监控**：compose 加 `--enable-metrics --metrics-port=8001`，接入 Prometheus + Grafana。
- **多机**：把 vLLM 改成 ray 部署，或用 k8s + vllm-operator。

---

## 7. 安全建议

- 生产环境务必设置 `VLLM_API_KEY`
- 容器内不要 root（vLLM 镜像默认是 root；如需加固，自定义 Dockerfile 加 `USER`）
- 网络层：仅暴露给内部网关，对外用 nginx 反代并加 rate limit
- 模型许可：商用前确认每个模型 License
