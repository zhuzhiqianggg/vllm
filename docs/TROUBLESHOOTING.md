# 故障排查与运维手册

> 整合 **本次部署中遇到的所有问题 + 通用 vLLM 故障排查 + 性能优化方案**。
> 建议先看「一、问题排查总流程」确定方向，再按编号查找具体问题。

---

## 目录

- [一、问题排查总流程](#一问题排查总流程)
- [二、部署阶段问题](#二部署阶段问题)
- [三、启动阶段问题](#三启动阶段问题)
- [四、加载与显存问题](#四加载与显存问题)
- [五、推理阶段问题](#五推理阶段问题)
- [六、网络与下载问题](#六网络与下载问题)
- [七、性能优化方案](#七性能优化方案)
- [八、运维监控告警](#八运维监控告警)
- [九、彻底重置与恢复](#九彻底重置与恢复)
- [十、命令速查](#十命令速查)

---

## 一、问题排查总流程

遇到任何问题时，按这个流程定位：

```
1. 收集现象
   ├── 错误日志: docker compose logs vllm --tail=200
   ├── 容器状态: docker compose ps
   ├── GPU 状态: nvidia-smi
   └── 资源状态: df -h / free -h

2. 分类
   ├── 启动前（下载/镜像/驱动）  → 看 二 / 六
   ├── 启动中（参数/校验）       → 看 三
   ├── 加载中（OOM/IO）          → 看 四
   ├── 推理中（报错/慢）         → 看 五
   └── 性能（吞吐/延迟不达标）   → 看 七

3. 对照具体编号问题
   └── 找到根因 → 应用方案 → 验证
```

**速查表**：

| 现象关键词 | 跳到 |
| --- | --- |
| `forward compatibility` | §3.1 |
| `could not select device driver` | §2.1 |
| `libcuda.so` not found | §2.2 |
| `vulkan` | §2.3 |
| `xet` / `401 Unauthorized` | §6.3 |
| `unrecognized arguments` | §3.2 |
| `OutOfMemoryError` | §4.1 |
| `max sequence length` | §4.2 |
| `model_not_found` | §5.1 |
| `prefix cache` 命中率低 | §5.2 |
| 性能差/TTFT 高 | §7 |

---

## 二、部署阶段问题

### 2.1 容器无法识别 GPU（NVIDIA Container Toolkit 缺失）

**症状**：
```
could not select device driver "" with capabilities: [[gpu]]
# 或
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

**根因**：
Docker daemon 默认**不识别 GPU 设备**。`nvidia-container-toolkit` 是桥接层：
- 把主机的 `/dev/nvidia*` 设备文件挂进容器
- 把主机的 `libcuda.so` / `libnvidia-*.so` 注入容器
- 通过 OCI runtime hook 实现

**修复（Ubuntu 22.04）**：
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
# 验证
docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi
```

### 2.2 升级驱动后容器报 `libcuda.so.<旧版本>: cannot open`

**症状**：
升级 NVIDIA 驱动（如 570→580）后，启动容器报：
```
libcuda.so.570.172.08: cannot open shared object file
```

**根因**：
`nvidia-container-cli` 缓存了升级前的驱动库路径。驱动库已变成 580，但 CLI/hook 仍引用 570。

**修复**：
```bash
systemctl restart docker   # 重启 dockerd 重置 hook 缓存
docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi
```

**预防**：每次升级驱动后，**必须**重启 dockerd。

### 2.3 vulkan ICD 文件路径错误

**症状**：
```
Error: open /etc/vulkan/icd.d/nvidia_icd.json: no such file or directory
```

**根因**：
不同发行版 vulkan ICD 路径不同：
- 部分包装到 `/etc/vulkan/icd.d/`
- 部分包到 `/usr/share/vulkan/icd.d/`
- 容器内只挂一个路径

**修复**：
```bash
ln -sf /usr/share/vulkan /etc/vulkan
```

### 2.4 compose 中 GPU 配置走错路径

**症状**：
docker-compose 启动后容器**仍然**报 libcuda 错误（即使驱动已升级、runtime 已装）。

**根因**：
compose 中**同时**写了 `runtime: nvidia` 和 `deploy.resources.reservations.devices`，新版 Compose 优先走 CDI 路径，CDI spec 在驱动升级后**未自动重建**。

**修复**：
```yaml
# docker-compose.yml - 只保留 legacy runtime
services:
  vllm:
    runtime: nvidia            # ← 保留
    # 删除下面这段！
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
```

---

## 三、启动阶段问题

### 3.1 vLLM 启动报 `Error 804: forward compatibility was attempted on non supported HW`

**症状**：
```
Failed to get device capability: Unexpected error from cudaGetDeviceCount()
Error 804: forward compatibility was attempted on non supported HW
```

**根因**（**最关键**）：
- vLLM 新版镜像（≥0.26）打包了 **CUDA 13.0**（PyTorch 2.11+cu130）
- 主机驱动是旧版（如 570），**最高支持 CUDA 12.8**
- **驱动向上不兼容 CUDA**：驱动不能跑比它声明支持的更新 CUDA

**诊断**：
```bash
# 主机驱动最高支持的 CUDA
nvidia-smi | head -3 | tail -1   # 显示 "CUDA Version: X.Y"

# vLLM 镜像实际使用的 CUDA
docker run --rm --entrypoint python3 vllm/vllm-openai:latest -c "import torch; print(torch.version.cuda)"
```

**修复**：
```bash
# 升级驱动到能支持镜像 CUDA 的版本
apt-get install -y nvidia-driver-580   # 580+ 支持 CUDA 13.0
modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia nvidia_uvm nvidia_drm nvidia_modeset
nvidia-smi   # 应显示新驱动
systemctl restart docker
```

**版本兼容矩阵**：

| 驱动 | 最高 CUDA | 对应 PyTorch / vLLM |
| --- | --- | --- |
| 470 | 11.4 | 1.x |
| 525 | 12.0 | 2.0-2.2 |
| 535 | 12.2 | 2.3 |
| 550 | 12.4 | 2.4-2.5 |
| 570 | 12.8 | 2.6-2.10, vLLM ≤0.25 |
| **580+** | **13.0** | **2.11+, vLLM ≥0.26** |

### 3.2 vLLM 启动报 `unrecognized arguments`

**症状**：
```
vllm serve: error: unrecognized arguments: --swap-space=4 --disable-log-requests
```

**根因**：
- vLLM 镜像 ENTRYPOINT 已经是 `vllm serve`
- compose 中 `command:` 是 vLLM 子命令的参数
- 传入了已废弃/不存在的参数

**修复**：
```bash
# 1) 看 vLLM 实际支持什么
docker run --rm vllm/vllm-openai:latest serve --help 2>&1 | head -30

# 2) 移除已废弃参数（swap-space / disable-log-requests）
# 3) 让 vLLM 用默认值
```

**常见废弃参数**：
- `--swap-space`：改名/移走，用 `--swap-space` 实际名称随版本变
- `--disable-log-requests`：新版用 `--disable-log-stats`

### 3.3 镜像拉取失败

**症状**：
```
ERROR: failed to resolve reference ...: failed to do request:
Head ...: dial tcp: i/o timeout
# 或
ERROR: pull access denied for vllm/vllm-openai
```

**根因**：
- 网络不通 Docker Hub
- daemon.json 残留占位符镜像
- 私有仓库未登录

**修复**：
```bash
# 配置 Docker 镜像加速
cat > /etc/docker/daemon.json <<'EOF'
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://hub-mirror.c.163.com",
        "https://mirror.baidubce.com",
        "https://docker.mirrors.ustc.edu.cn"
    ]
}
EOF
systemctl restart docker

# 验证
docker info | grep -A 5 "Registry Mirrors"
```

### 3.4 启动后长时间 `Loading model`

**症状**：
- 日志停在 `Loading safetensors ...%` 几分钟
- 不是真的死锁

**根因**：
大模型加载 = 磁盘读取 + 反序列化 + GPU 拷贝 + cuBLAS 初始化：
- 35GB FP8 模型：~17.5s 加载 + 50s torch.compile + 7s CUDA graph
- **完全就绪约 4 分钟**

**判断是否真的卡住**：
```bash
# 磁盘 IO
iostat -x 1
# GPU 加载进度
nvidia-smi   # 看 memory.used 是否在涨
# 进程
docker compose logs vllm --tail=10
```

---

## 四、加载与显存问题

### 4.1 CUDA Out of Memory

**症状**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X GiB...
```

**根因**：
显存需求 > 实际可用：
- 模型权重 + KV 缓存 + 激活 + CUDA 上下文 > `GPU_MEMORY_UTILIZATION` 配的额度

**优化方案（按优先级）**：

| # | 方案 | 节省 | 性能影响 |
| --- | --- | --- | --- |
| 1 | 换 AWQ/GPTQ/FP8 量化 | 50-75% | 略降 |
| 2 | `--gpu-memory-utilization 0.85` | ~5% | 略降 |
| 3 | `--max-model-len` ↓ | 按 ctx 减少 | 长请求受限 |
| 4 | `--kv-cache-dtype fp8` | 30-50% KV | 几乎无 |
| 5 | `--max-num-seqs` ↓ | 减少并发 | 吞吐降 |
| 6 | TP=2 跨卡 | 权重减半 | 通信开销 |
| 7 | `--enforce-eager` | ~2GB | 5% 降 |

**推荐组合**：
```bash
# 7B/14B：不用动
GPU_MEMORY_UTILIZATION=0.90
# 32B 量化：调到 0.92
GPU_MEMORY_UTILIZATION=0.92
# 70B 双卡：调到 0.90
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=2048
```

### 4.2 `max sequence length` 超出模型支持

**症状**：
```
The model's max sequence length is X, but you requested Y
```

**修复**：
```bash
# .env 中调整
MAX_MODEL_LEN=32768  # ≤ 模型实际支持
```

**参考各模型最大长度**：

| 模型 | 最大 ctx |
| --- | --- |
| Qwen2.5 系列 | 131072 |
| Qwen2.5 AWQ 版 | 32768-131072 |
| Llama 3 8B | 8192 |
| Llama 3 70B | 8192 |
| Qwen3 / Qwen3.5 | 32768-131072 |

### 4.3 启动时卡在 `Loading model weights`

**检查**：
```bash
# 模型完整性
ls -la /opt/vllm/models/<repo>/
du -sh /opt/vllm/models/<repo>/
# 磁盘
df -h /opt/vllm/models
# 残留 .incomplete
find /opt/vllm/models -name "*.incomplete" | head -5
```

**修复**：
```bash
# 清理后重下
rm -rf /opt/vllm/models/<repo>
./download_model.sh <short_name>
```

---

## 五、推理阶段问题

### 5.1 404 `model_not_found`

**症状**：
```json
{"error":"The model `xxx` does not exist."}
```

**根因**：
客户端请求的 `model` 字段与 `SERVED_MODEL_NAME` 不一致。

**修复**：
```bash
# 查服务端实际名称
curl http://localhost:8000/v1/models

# 客户端用 .env 中 SERVED_MODEL_NAME 的值
# 或者请求中写 "model": "qwen3.5-35b-a3b"（与 SERVED_MODEL_NAME 一致）
```

### 5.2 prefix caching 命中率低

**症状**：相同 system prompt 反复算，TTFT 没改善。

**根因**：
- 请求前缀**未完全一致**（含空格、换行、特殊字符）
- `--enable-prefix-caching` 未开启
- `MAX_NUM_BATCHED_TOKENS` 太小

**优化**：
```bash
# .env
ENABLE_PREFIX_CACHING=true
MAX_NUM_BATCHED_TOKENS=8192  # 调到 4K-16K
```

**验证命中**：
```bash
# 启动时加 --enable-metrics --metrics-port 8001
curl http://localhost:8001/metrics | grep prefix
# vllm:prefix_cache_hits_total
# vllm:prefix_cache_queries_total
# 命中率 = hits / queries
```

### 5.3 Tool use / function calling 报错

**症状**：
- Qwen 系列一般原生支持 tool，但旧 vLLM 镜像可能不识别
- 报 `tool_choice` 解析失败

**修复**：
```bash
# 升级 vLLM 镜像
docker pull vllm/vllm-openai:latest

# 或显式加参数
# 在 compose command 里加：
- --enable-auto-tool-choice
- --tool-call-parser=hermes
```

### 5.4 响应卡顿 / 延迟高

**判断瓶颈**（参见 §7 性能优化）：

| 现象 | 瓶颈 | 优化方向 |
| --- | --- | --- |
| GPU util < 50% | 调度/排队 | 加大 MAX_NUM_SEQS |
| GPU util > 90% | 算力 | 减少 prefill 量、投机解码 |
| 显存占满 | KV 满 | 量化、减并发 |
| TTFT 高（>1s） | prefill | 减少 ctx、切分请求 |
| TPOT 高（>100ms） | decode | 量化、减 ctx |

### 5.5 思考模型（CoT）输出过长

**症状**：Qwen3.5 等 thinking 模型输出 `<think>...</think>` 占大量 token。

**优化**：
```python
# 客户端：限制 max_tokens
resp = client.chat.completions.create(
    model="qwen3.5-35b-a3b",
    messages=[...],
    max_tokens=512,  # 强制上限
    # system prompt 引导
    # system: "请直接给出答案，不要输出思考过程"
)
```

```bash
# 服务端：extra-body 控制
# 部分版本支持 chat_template_kwargs: {"enable_thinking": false}
```

### 5.6 503 Service Unavailable

**症状**：模型还没加载完。

**修复**：
```bash
# 等 1-3 分钟
./scripts/logs.sh | grep "Application startup complete"
# 看到后再试
```

---

## 六、网络与下载问题

### 6.1 Docker 镜像拉取慢/失败

参见 §3.3 镜像加速配置。

### 6.2 HuggingFace 直连慢/超时

**修复**：
```bash
# 方案 1：国内镜像
export HF_ENDPOINT=https://hf-mirror.com
# 写入 .env
echo "HF_ENDPOINT=https://hf-mirror.com" >> /opt/vllm/.env

# 方案 2：ModelScope（独立体系）
./download_model.sh ms <repo_id>

# 方案 3：代理
export HTTPS_PROXY=http://127.0.0.1:7890
```

### 6.3 HF xet 后端在镜像站 401

**症状**：
```
RuntimeError: Task error: File reconstruction error: CAS Client Error:
Request error: HTTP status client error (401 Unauthorized),
domain: https://cas-server.xethub.hf.co/...
```

**根因**：
- `huggingface_hub ≥ 0.25` 默认启用 **xet (CAS)** 协议
- 镜像站 `hf-mirror.com` **不支持** xet
- xet 鉴权 token 镜像站没有

**修复**：
```bash
# 禁用 xet
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0  # 加速与 xet 冲突，也关

# 重新下载
```

**永久解决**：
- ModelScope：完全独立，**最稳**
- 自建代理：完整 mirror 镜像

### 6.4 模型下载中断/不完整

**症状**：
- 下载到一半失败
- 模型文件 `.incomplete` 残留
- 启动时模型校验失败

**修复**：
```bash
# 清理残留
rm -rf /opt/vllm/models/<repo>
find /opt/vllm/models -name "*.incomplete" -delete

# 重新下载
./download_model.sh <short_name>

# 如果一直失败：换 ModelScope
./download_model.sh ms <repo_id>
```

### 6.5 私有模型无访问权限

**症状**：`401 Unauthorized` 或 `403 Forbidden`。

**修复**：
```bash
# 1) 登录 HF
huggingface-cli login
# 输入 token（https://huggingface.co/settings/tokens）

# 2) 同意 License（Llama 3 等）
# 在 HF 模型页面点 "Agree and access repository"

# 3) 重启 vLLM
./scripts/stop.sh && ./scripts/start.sh
```

### 6.6 容器内访问外网失败

**修复**：
```bash
# 方案 A：宿主机代理
# .env 加
HTTPS_PROXY=http://host.docker.internal:7890

# 方案 B：compose 网络模式
network_mode: host  # 用主机网络
```

---

## 七、性能优化方案

> 本节是 **系统性优化方案**，不只是单个故障。涉及调优时直接看这里。

### 7.1 性能指标体系

| 指标 | 含义 | 理想值（在线） |
| --- | --- | --- |
| **TTFT** | 首个 token 延迟 | < 300ms |
| **TPOT** | 每 token 延迟 | < 50ms |
| **Throughput (req/s)** | 每秒请求数 | 10-50 |
| **Throughput (out tokens/s)** | 每秒输出 token | 越高越好 |
| **GPU util** | SM 占用 | > 70% |
| **显存占用** | 显存 | < 90% |
| **KV cache usage** | 缓存命中 | 70%+ |

### 7.2 显存优化（决定能跑多大）

| 方法 | 参数 | 节省 | 推荐度 |
| --- | --- | --- | --- |
| 量化 | `--quantization awq/fp8` | 50-75% | ⭐⭐⭐⭐⭐ |
| 减小上下文 | `--max-model-len` | 显著 | ⭐⭐⭐⭐ |
| KV 量化 | `--kv-cache-dtype fp8` | 30-50% KV | ⭐⭐⭐⭐ |
| 减并发 | `--max-num-seqs` | 减少 KV | ⭐⭐⭐ |
| 关 CUDA graph | `--enforce-eager` | ~2GB | ⭐⭐ |
| 降利用率 | `--gpu-memory-utilization 0.85` | 5% | ⭐⭐ |

### 7.3 吞吐优化（决定服务多少用户）

| 方法 | 参数 | 效果 | 推荐度 |
| --- | --- | --- | --- |
| 启用 prefix caching | `--enable-prefix-caching` | 节省重复 prefill | ⭐⭐⭐⭐⭐ |
| 启用 chunked prefill | `--enable-chunked-prefill` | 减少长 prompt 阻塞 | ⭐⭐⭐⭐⭐ |
| 加大并发 | `--max-num-seqs 256-512` | 提升吞吐 | ⭐⭐⭐⭐ |
| 加大 batch | `--max-num-batched-tokens 8192-16384` | 提升吞吐 | ⭐⭐⭐⭐ |
| 加大 swap | `--swap-space 8-16` | 减少拒绝 | ⭐⭐⭐ |
| 多实例 + LB | 跑多个 vLLM + nginx | 容量翻倍 | ⭐⭐⭐⭐ |
| 投机解码 | `--speculative-model` | 1.5-2.5x | ⭐⭐⭐ |

### 7.4 延迟优化（决定用户体验）

| 方法 | 参数 | 效果 |
| --- | --- | --- |
| temperature=0 | 客户端参数 | 走 greedy，TPOT 减少 10-20% |
| 投机解码 | `--speculative-model` | 1.5-2.5x |
| 合理 batch | `--max-num-batched-tokens` 不超 16K | 短请求公平调度 |
| 流式输出 | 客户端 `stream=True` | 改善感知延迟 |
| 限 max_tokens | 客户端 | 防止超长 |

### 7.5 调参方法论（**必须遵守**）

```
1) 跑 baseline：./scripts/benchmark.sh 32 200，记录 throughput / p50 / p95
2) 改一个参数（不要同时改多个！）
3) 再跑一次，对比
4) 看 nvidia-smi：判断瓶颈
5) 重复 2-4 找到最优
```

### 7.6 推荐起点配置

| 场景 | TP | GPU_UTIL | MAX_LEN | MAX_SEQS | BATCH_TOK | 量化 |
| --- | --- | --- | --- | --- | --- | --- |
| 7B 单卡 | 1 | 0.90 | 8K | 256 | 8K | - |
| 14B 单卡 | 1 | 0.90 | 8K | 128 | 8K | - |
| 32B AWQ | 1 | 0.92 | 8K | 64 | 4K | awq |
| 32B BF16 | 2 | 0.90 | 8K | 64 | 4K | - |
| 70B AWQ | 2 | 0.92 | 8K | 32 | 2K | awq |
| 35B-A3B FP8 | 2 | 0.90 | 32K | 64 | 8K | fp8 |
| 128K 长文 | 1/2 | 0.95 | 128K | 16 | 2K | kv=fp8 |

### 7.7 本次 35B-A3B 调优实战

**环境**：2×RTX 4090 (48GB，无 NVLink) + Qwen3.5-35B-A3B-FP8

**基础配置**：
```env
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=8192
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
```

**实测基线**（16 并发 64 请求）：
```
throughput_req_s:       8.67
throughput_out_tok_s:   1109
p50_s:                 5.22
p95_s:                 7.38
```

**思考模型影响**：
Qwen3.5 默认 CoT，输出含 `<think>...</think>`。生产中：
- 客户端 `max_tokens=512` 限长
- system prompt 加 `"请直接回答，不要输出思考过程"`
- 客户端解析时过滤 `<think>` 块

**进一步优化方向**：
- 关闭 CoT（如果支持）：`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- 加大 `MAX_NUM_SEQS`（显存还有余量）
- 多实例 + 负载均衡（替代 TP=2，避免 PCIe 通信开销）

### 7.8 性能瓶颈速查

| 现象 | 根因 | 解法 |
| --- | --- | --- |
| GPU util < 50% | 调度/排队 | MAX_NUM_SEQS ↑ |
| GPU util > 95% | 算力饱和 | 减 MAX_NUM_BATCHED_TOKENS 或投机解码 |
| 显存满 / OOM | KV cache 满 | 量化 / 减并发 / 减 ctx |
| TTFT > 1s | prefill 慢 | 减 ctx / chunked prefill / 投机 |
| TPOT > 100ms | decode 慢 | 量化 / 减 ctx / 温度 0 |
| 拒绝率上升 | KV 满 | 加大 swap / 减 ctx |

---

## 八、运维监控告警

### 8.1 关键指标监控

**GPU**：
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 1
```

**vLLM**（需启用 `--enable-metrics --metrics-port 8001`）：
```bash
# 关键指标
curl -s http://localhost:8001/metrics | grep -E "^vllm:(num_requests|gpu_cache_usage|prefix_cache|request_success|request_failure|e2e_request_latency)"
```

**Prometheus + Grafana**：
- 拉取 `vllm-server:8001/metrics`
- 仪表盘参考：vLLM 官方 GitHub examples

### 8.2 健康检查

```bash
# 简单
curl -f http://localhost:8000/v1/models

# compose healthcheck（已配）
docker compose ps   # 状态字段
```

### 8.3 日志

```bash
./scripts/logs.sh                     # 实时跟踪
docker compose logs vllm --tail=200    # 最近 200 行
docker compose logs vllm 2>&1 | grep -E "ERROR|WARN"  # 错误
```

### 8.4 告警建议

| 指标 | 警告阈值 | 告警阈值 |
| --- | --- | --- |
| GPU 温度 | > 80°C | > 85°C |
| 显存占用 | > 90% | > 95% |
| 请求失败率 | > 1% | > 5% |
| p95 延迟 | > 基线 2x | > 基线 5x |
| 容器 OOM 重启 | 1 次/小时 | 3 次/小时 |

### 8.5 容量规划

```
单卡吞吐 = 实际压测（不要信理论值）
所需卡数 = ceil(目标 QPS × 平均 token / 单卡吞吐 × 1.5)
```

---

## 九、彻底重置与恢复

### 9.1 仅重启服务

```bash
cd /opt/vllm
./scripts/stop.sh && ./scripts/start.sh
```

### 9.2 重置容器（保留模型）

```bash
cd /opt/vllm
docker compose down
docker compose rm -f
docker image rm vllm/vllm-openai:latest   # 删镜像
docker compose up -d                       # 拉新镜像
```

### 9.3 完全清理（删模型）

```bash
cd /opt/vllm
docker compose down
docker compose rm -f
docker image rm vllm/vllm-openai:latest
rm -rf /opt/vllm/models/<repo>__*    # 删模型
./scripts/start.sh                    # 启动（按需重下模型）
```

### 9.4 驱动回滚

```bash
# 装回旧驱动
apt-get install -y nvidia-driver-570
modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia nvidia_uvm nvidia_drm nvidia_modeset
systemctl restart docker
```

### 9.5 灾备脚本

```bash
# 备份
tar czf vllm-backup-$(date +%Y%m%d).tar.gz \
  /opt/vllm/.env /opt/vllm/docker-compose.yml /opt/vllm/scripts

# 恢复
tar xzf vllm-backup-*.tar.gz -C /
cd /opt/vllm && ./scripts/start.sh
```

---

## 十、命令速查

### 10.1 诊断

```bash
# GPU
nvidia-smi
nvidia-smi -l 1
nvidia-smi pmon -s u -c 1
nvidia-smi --query-gpu=... --format=csv -l 1

# 驱动
cat /proc/driver/nvidia/version
lsmod | grep nvidia
nvcc --version

# 容器
docker ps -a
docker stats vllm-server
docker compose logs vllm
docker exec vllm-server nvidia-smi

# 资源
df -h /opt/vllm
free -h
iostat -x 1

# 网络
ss -tlnp | grep 8000
curl -v http://localhost:8000/v1/models
```

### 10.2 修复

```bash
# 驱动升级
apt-get install -y nvidia-driver-580
modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia nvidia_uvm nvidia_drm nvidia_modeset
systemctl restart docker

# 重装 toolkit
apt-get install --reinstall -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Docker 加速
cat > /etc/docker/daemon.json <<'EOF'
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://hub-mirror.c.163.com",
        "https://mirror.baidubce.com",
        "https://docker.mirrors.ustc.edu.cn"
    ]
}
EOF
systemctl restart docker
```

### 10.3 运维

```bash
cd /opt/vllm
./scripts/start.sh        # 启动
./scripts/stop.sh         # 停止
./scripts/logs.sh         # 跟日志
./scripts/benchmark.sh 32 200  # 压测
python3 scripts/test_api.py    # 端到端测试
docker compose restart vllm    # 重启
docker compose pull            # 拉新镜像
```

---

> 文档结束。遇到新问题请反馈，将持续更新。
