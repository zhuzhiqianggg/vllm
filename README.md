# vLLM 一键部署与运维（2 x RTX 4090）

> 面向单机双 RTX 4090（48 GB × 2）环境，提供 **docker-compose 编排 + 一键模型下载 + OpenAI 兼容 API** 的开箱即用方案。

---

## 目录

- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [模型选择](#模型选择)
- [参数优化](#参数优化)
- [API 调用](#api-调用示例)
- [运维命令](#运维命令)
- [常见问题](#常见问题)

---

## 一、环境前置

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| Docker | 24+（已验证 29.x） | `docker --version` |
| Docker Compose | v2（已验证 v5.x 兼容语法） | `docker compose version` |
| NVIDIA Driver | >= 535（已验证 570） | `nvidia-smi` |
| NVIDIA Container Toolkit | 最新 | 让容器能识别 GPU |
| 磁盘 | 模型按需，7B≈15 GB，14B≈30 GB，32B≈65 GB，70B≈140 GB | 建议 SSD |
| 网络 | 访问 huggingface.co（或用镜像） | 首次拉镜像与模型 |

> 当前机器：**2 × RTX 4090 (48 GB)**，已就绪。

如果 `nvidia-smi` 正常但 `docker run --gpus all nvidia/cuda:12.8.0-base nvidia-smi` 失败，需要安装 NVIDIA Container Toolkit：

```bash
# Debian/Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 二、目录结构

```
/opt/vllm/
├── docker-compose.yml        # vLLM 容器编排
├── .env.example              # 环境变量模板
├── download_model.sh         # 一键模型下载（入口）
├── download_model_lib.sh     # 下载实现
├── README.md                 # 本文档
├── docs/
│   ├── MODELS.md             # 模型选择指南
│   ├── OPTIMIZATION.md       # 参数优化指南
│   ├── EXAMPLES.md           # API 调用示例
│   └── TROUBLESHOOTING.md    # 故障排查
├── configs/presets.sh        # 启动参数预设
├── scripts/
│   ├── start.sh / stop.sh    # 启停
│   ├── logs.sh               # 跟踪日志
│   ├── benchmark.sh          # 压测
│   └── bench_http.py         # 纯 HTTP 压测（无需 vllm bench）
├── models/                   # 挂载到容器 /root/.cache/huggingface
└── logs/                     # 容器日志
```

---

## 三、快速开始

### 1. 拉取编排文件（已完成）

```bash
cd /opt/vllm
cp .env.example .env
```

按需编辑 `.env`，核心字段：

```env
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=256
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
```

### 2. 一键下载模型

```bash
# 列出全部预设
./download_model.sh list

# 下载预设（Qwen2.5-7B-Instruct）
./download_model.sh qwen2.5-7b

# 任意 HF repo
./download_model.sh custom Qwen/Qwen2.5-Coder-32B-Instruct-AWQ

# ModelScope 渠道（国内更稳）
./download_model.sh ms Qwen/Qwen2.5-7B-Instruct
```

> 模型会下载到 `/opt/vllm/models/`，与容器 `/root/.cache/huggingface` 共享。

### 3. 启动 vLLM

```bash
# 首次会拉 vllm/vllm-openai:latest 镜像（约 8~10 GB）
./scripts/start.sh

# 跟踪日志
./scripts/logs.sh

# 健康检查
curl http://localhost:8000/v1/models
```

启动成功的标志（`logs.sh` 输出）：

```
INFO 11-22 10:00:00 server.py:NNN Starting vLLM API server on http://0.0.0.0:8000
INFO 11-22 10:00:10 launcher.py:NNN Started server process
INFO 11-22 10:00:12 server.py:NNN Application startup complete.
INFO 11-22 10:00:12 server.py:NNN Uvicorn running on http://0.0.0.0:8000
```

### 4. 调用测试

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role":"user","content":"用一句话介绍 vLLM"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

更多调用方式见 [docs/EXAMPLES.md](docs/EXAMPLES.md)。

---

## 四、运维命令

| 操作 | 命令 |
| --- | --- |
| 启动 | `./scripts/start.sh` |
| 停止 | `./scripts/stop.sh` |
| 重启 | `docker compose restart vllm` |
| 实时日志 | `./scripts/logs.sh` |
| 进入容器 | `docker compose exec vllm bash` |
| 资源监控 | `docker stats vllm-server` / `nvidia-smi` |
| 压测 | `./scripts/benchmark.sh 32 200` |

---

## 五、常见问题（速查）

- **拉镜像慢**：配置 Docker 镜像加速（`/etc/docker/daemon.json`）。
- **拉模型慢**：用 `./download_model.sh ms <repo_id>` 走 ModelScope，或设置 `HF_ENDPOINT=https://hf-mirror.com`。
- **OOM（显存不足）**：降低 `GPU_MEMORY_UTILIZATION`（如 0.85）或 `MAX_MODEL_LEN`、或用 AWQ 量化模型。
- **`max model length` 报错**：在 `.env` 调小 `MAX_MODEL_LEN`，或确认模型支持更长（如 Qwen2.5-7B 支持 131K）。
- **HTTP 502 / 健康检查失败**：查看 `./scripts/logs.sh` 顶部堆栈，参考 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

---

## 六、文档导航

- 模型选型与硬件匹配：[docs/MODELS.md](docs/MODELS.md)
- 参数调优与吞吐优化：[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)
- 完整 API 示例（含流式、函数调用、tools）：[docs/EXAMPLES.md](docs/EXAMPLES.md)
- 故障排查：[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
