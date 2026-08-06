# 大模型部署与运维：知识体系手册

> 本文档按 **部署 / 运行 / 优化 / 配置 / 运维** 五个维度系统化整理 LLM 部署与运维知识。
>
> 适用对象：基于 vLLM + Docker 在 NVIDIA GPU 上部署开源 LLM/VLM 的工程师。
>
> **注意**：本次部署中遇到的所有问题与解决方案已整合到 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)，请先看它做问题定位，本文档侧重"知其所以然"的原理与体系。

---

## 文档结构

```
第一篇：部署 (Deployment)        ← 硬件 + 系统 + 容器
  1. GPU 硬件深度
  2. NVIDIA 驱动 / CUDA / Toolkit 体系
  3. Docker + GPU 容器化
  4. 模型下载与管理

第二篇：运行 (Operation)         ← vLLM 工作机制
  5. vLLM 内部原理
  6. vLLM 启动与生命周期
  7. OpenAI 兼容 API 行为

第三篇：优化 (Optimization)      ← 性能与精度
  8. LLM 推理性能原理
  9. 量化技术详解
  10. MoE 架构与优化
  11. 推理性能调优实战

第四篇：配置 (Configuration)     ← 模型与参数
  12. 模型选型方法论
  13. vLLM 参数配置详解
  14. 多实例与负载均衡

第五篇：运维 (Operations & Maintenance)  ← 上线后管理
  15. 监控与可观测性
  16. 性能压测
  17. 高可用与灾备
  18. 安全与合规
  19. 容量规划
```

---

# 第一篇：部署（Deployment）

> **部署** = 把 LLM 跑起来的所有前置准备：硬件选型、驱动安装、容器化、模型获取。

## 第 1 章：GPU 硬件深度

### 1.1 GPU 架构基础

```
GPU
└── 多个 GPC (Graphics Processing Cluster)
    └── 多个 TPC (Texture Processing Cluster)
        └── 多个 SM (Streaming Multiprocessor)  ← 计算核心
            ├── CUDA Core (FP32/INT 单元)
            ├── Tensor Core (矩阵乘单元, FP16/BF16/FP8/INT8)  ← LLM 关键
            ├── SFU (Special Function Unit)
            ├── LD/ST Unit (内存访问)
            ├── Register File
            └── Shared Memory (~100KB/SM)
```

| 概念 | 作用 | LLM 部署相关性 |
| --- | --- | --- |
| SM | 计算核心，~100 个/卡 | 决定并行度上限 |
| Tensor Core | 矩阵乘累加 (MMA) | **LLM 推理最关键**，FP16/BF16 加速 4-8x |
| 显存 (VRAM) | GPU 专用内存 | **决定能装多大模型** |
| 显存带宽 | 显存读写速度 | **决定 LLM 推理速度上限**（不是算力） |
| NVLink/NVSwitch | 卡间互联 | 多卡推理性能关键 |

### 1.2 LLM 推理瓶颈 = 显存带宽

LLM 推理是 **memory-bound**：
- 每个 token 生成要**读一遍**模型权重
- 4090 显存带宽 1.0 TB/s → 理论上限 1TB / 模型大小
- 7B FP16 (14GB) → 理论 **71 tokens/s**（单卡单请求）
- 35B FP8 (35GB) → 理论 **28 tokens/s**

**算力与带宽对比**：

| GPU | FP16 TFLOPS | 带宽 GB/s | LLM 适用度 |
| --- | --- | --- | --- |
| RTX 4090 | 82.6 | 1008 | 消费级之王 |
| RTX 3090 | 35.6 | 936 | 还行 |
| A100 80G | 312 | 2039 | 训练+推理顶级 |
| H100 SXM | 989 | 3350 | 训练最优 |
| H200 | 989 | 4800 | 推理最优 |
| B200 | 2250 | 8000 | 下一代 |

### 1.3 显存需求估算

```
总显存 = 模型权重 + KV 缓存 + 激活 + CUDA 上下文开销（~1.5GB）
```

**模型权重**：
| dtype | bytes/param | 7B | 14B | 32B | 70B |
| --- | --- | --- | --- | --- | --- |
| FP32 | 4 | 28GB | 56GB | 128GB | 280GB |
| BF16/FP16 | 2 | 14GB | 28GB | 64GB | 140GB |
| FP8 | 1 | 7GB | 14GB | 32GB | 70GB |
| INT8 | 1.25 | 9GB | 18GB | 40GB | 88GB |
| INT4 (AWQ) | 0.6 | 4.2GB | 8.4GB | 19GB | 42GB |

**KV 缓存**（**最常被低估**）：
```
KV = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes
```

例：Llama-7B 32 上下文 32K batch 64 → KV ~5GB
例：Qwen-32B 32K ctx batch 64 → KV ~85GB

**经验公式**：
```
可并发请求数 ≈ 显存预算 / (模型权重大小 × 0.2)
```

### 1.4 RTX 4090 深度（本次部署环境）

| 指标 | 值 |
| --- | --- |
| 架构 | Ada Lovelace（TSMC 5nm） |
| SM 数 | 128 |
| CUDA Cores | 16384 |
| Tensor Cores | 512 (4 代) |
| 显存 | 24 GB GDDR6X |
| 带宽 | 1008 GB/s |
| FP16 算力 (Tensor) | 82.6 TFLOPS |
| FP8 算力 (Tensor) | 165.2 TFLOPS |
| TDP | 450W |
| NVLink | **不支持** ⚠️ |
| Compute Capability | 8.9 |

**4090 多卡限制**（重要）：
- 2 张 4090 之间**无 NVLink**，只能走 PCIe (32 GB/s)
- TP=2 时 all-reduce 走 PCIe，**比 NVLink 慢 20 倍**
- 小模型（≤14B）TP=2 性能可能**比 TP=1 慢**
- **建议**：小模型用多实例 + 负载均衡，代替 TP

### 1.5 GPU 选型决策树

```
预算 < 2 万 / 单卡 ─→ RTX 4090 (24GB)
预算 5-10 万 ────┬─ 单卡 A6000 (48GB)        ← 本次环境
                ├─ 双卡 4090 (48GB×2)
                └─ L40 / L40S (48GB)
预算 20+ 万 ────┬─ A100 80G (单/双卡)
                ├─ H100 80G (单/双卡)
                └─ H200 (141GB)
巨资 ─────────→ B200 / GB200
```

### 1.6 选型 Checklist

- [ ] 显存 ≥ 模型权重 × 1.5
- [ ] 带宽决定 decode 速度
- [ ] 算力决定 prefill 速度
- [ ] NVLink 多卡性能关键（4090 没有）
- [ ] TDP 决定机房散热
- [ ] 驱动版本决定可用 CUDA 工具包
- [ ] Compute Capability ≥ 8.0（BF16 Tensor Core）

---

## 第 2 章：NVIDIA 驱动 / CUDA / Toolkit 体系

### 2.1 三者关系

```
应用代码 (vLLM / PyTorch)
    ↓ 调
PyTorch CUDA 扩展 (libtorch_cuda.so)
    ↓ 调
CUDA Toolkit (libcudart.so, libcublas.so, ...)   ← 容器内自带
    ↓ 调
NVIDIA Driver (libcuda.so)                       ← 主机唯一
    ↓ 调
/dev/nvidia* + GPU 硬件
```

| 组件 | 位置 | 谁装 | 作用 |
| --- | --- | --- | --- |
| **NVIDIA 驱动** | 主机 | OS 包 | 唯一与硬件对话的层 |
| **CUDA Toolkit** | 容器内 | 容器自带 | 运行时 + 编译器 + 库 |
| **cuDNN / NCCL** | 容器内 | 容器自带 | 深度学习原语、集合通信 |

**核心认知**：
- **驱动 = 在主机**（唯一一个，必须与 GPU 匹配）
- **CUDA Toolkit = 容器内可独立**（每容器可不同版本）
- **驱动版本 ≥ CUDA Toolkit 要求的最低版本**（向下兼容）

### 2.2 版本兼容矩阵

| 驱动 | 最高 CUDA | 对应 PyTorch / vLLM |
| --- | --- | --- |
| 470 | 11.4 | 1.x |
| 525 | 12.0 | 2.0-2.2 |
| 535 | 12.2 | 2.3 |
| 550 | 12.4 | 2.4-2.5 |
| 570 | 12.8 | 2.6-2.10, vLLM ≤0.25 |
| **580+** | **13.0** | **2.11+, vLLM ≥0.26** |

**检查当前支持的最高 CUDA**：
```bash
nvidia-smi  # 顶部右上角: CUDA Version: X.Y
```

**驱动/CUDA 关系**：
- 驱动**向下兼容 CUDA**：可跑 ≤ 它声明支持的版本
- 驱动**不向上兼容 CUDA**：不能跑 > 它声明支持的版本
- 错误："Error 804: forward compatibility was attempted on non supported HW"

### 2.3 升级驱动标准流程

```bash
# 1) 停所有使用 GPU 的进程
docker ps -q | xargs -r docker stop

# 2) 装新驱动
apt-get install -y nvidia-driver-580

# 3) 重启或重载
# 方案 A（最稳）
reboot
# 方案 B（无显卡用户）
modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia nvidia_uvm nvidia_drm nvidia_modeset

# 4) 重启 docker
systemctl restart docker

# 5) 验证
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.0.0-base nvidia-smi
```

### 2.4 诊断命令

```bash
nvidia-smi                          # 卡 + 驱动 + 进程
nvidia-smi -q                       # 详细
nvidia-smi pmon -s u -c 1           # 进程级每秒
nvcc --version                      # 主机 CUDA Toolkit 版本
cat /proc/driver/nvidia/version     # 内核模块版本
lsmod | grep nvidia                 # 已加载模块
```

---

## 第 3 章：Docker + GPU 容器化

### 3.1 为什么需要特殊处理

普通 Docker 容器只做 CPU/内存隔离，**不识别 GPU**。要让容器用 GPU：
1. 挂载主机 GPU 设备文件（`/dev/nvidia0`, `/dev/nvidia-uvm*`）
2. 挂载主机驱动库（`libcuda.so`, `libnvidia-*.so`）
3. 配置 `LD_LIBRARY_PATH` 让应用能找到

### 3.2 nvidia-container-toolkit 组件

```
dockerd
  ↓ 启动容器
nvidia-container-runtime (替代 runc)
  ↓ 调用
nvidia-container-cli
  ↓ 查询
libnvidia-container
```

| 二进制 | 作用 |
| --- | --- |
| `nvidia-container-runtime` | 替代 `runc`，处理 GPU-aware 容器 |
| `nvidia-container-cli` | CLI 工具，列出 GPU、生成 OCI spec |
| `nvidia-ctk` | 配置 + 诊断工具 |
| `nvidia-container-runtime-hook` | 早期 OCI pre-start hook |

### 3.3 两种 GPU 接入模式

**Legacy nvidia runtime**：
```bash
docker run --runtime=nvidia --gpus all ...
```
- 通过 OCI hook 注入设备
- 成熟稳定，驱动升级后基本自动恢复
- **生产推荐**

**CDI（Container Device Interface）**：
```bash
docker run --device nvidia.com/gpu=all ...
```
- 通过 CDI spec 文件（YAML）描述 GPU
- 未来方向，但驱动升级后 spec 不会自动更新
- 升级驱动后必须 `nvidia-ctk cdi generate`

### 3.4 最佳实践（compose）

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia                # legacy 路径
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
    # 不要写 deploy.resources.reservations.devices（走 CDI）
```

### 3.5 容器 GPU 调试

```bash
docker exec <ctr> nvidia-smi
docker exec <ctr> env | grep -E "NVIDIA|CUDA"
docker exec <ctr> ldconfig -p | grep cuda
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 \
  /usr/local/cuda/extras/demo_suite/deviceQuery
```

### 3.6 安装 nvidia-container-toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

---

## 第 4 章：模型下载与管理

### 4.1 模型权重格式

| 格式 | 特点 |
| --- | --- |
| **safetensors** | HuggingFace 默认，**安全 + 加载快**（推荐） |
| PyTorch (.bin/.pt) | 旧格式，可能含恶意代码（不推荐） |
| GGUF | llama.cpp 用，CPU/GPU 通用 |
| AWQ/GPTQ | 4-bit 量化，显存省 75% |
| FP8 | 8-bit 浮点，原生硬件加速 |

### 4.2 下载渠道

| 渠道 | 适用 | 镜像 |
| --- | --- | --- |
| **HuggingFace** | 官方 | 需科学上网或镜像 |
| **hf-mirror.com** | 国内中科大镜像 | 标准 HTTPS |
| **ModelScope** | 阿里，国内最快 | 独立协议 |
| 自建代理 | 企业内网 | 完全控制 |

### 4.3 关键技术点

**huggingface_hub xet 协议**（v0.25+ 默认启用）：
- 自研去重存储协议
- 镜像站可能**不支持** → 401 错误
- 解决：`HF_HUB_DISABLE_XET=1`

**hf_transfer 加速**：
- 多线程分块下载
- 与 xet 冲突时需关闭

**ModelScope snapshot_download**：
- 国内独立体系，**最稳**
- API 与 HF 类似

### 4.4 模型目录规范

```
models/
└── <org>__<model>__<quantization>/      # __ 是 / 的转义
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── chat_template.jinja
    ├── model-00001-of-00042.safetensors
    ├── model-00002-of-00042.safetensors
    ├── ...
    └── mtp.safetensors (可选，multi-token prediction)
```

容器内挂载：`./models` → `/root/.cache/huggingface`

---

# 第二篇：运行（Operation）

> **运行** = 理解 vLLM 的工作机制，从启动到响应请求的完整流程。

## 第 5 章：vLLM 内部原理

### 5.1 定位

vLLM = **高吞吐 LLM 推理和服务引擎**，UC Berkeley 2018 起开发，2023 开源。

**核心创新**：
- **PagedAttention**（分页注意力）—— 解决 KV 缓存碎片
- **Continuous Batching**（连续批处理）—— 提升吞吐
- 与 HuggingFace 模型无缝兼容
- OpenAI 兼容 API

### 5.2 架构总览

```
┌─────────────────────────────────────────────────────┐
│ API Server (FastAPI + uvicorn)                      │
│      ↓ 接收 HTTP 请求                                │
│ AsyncLLMEngine                                      │
│      ↓ 调度                                         │
│ Scheduler (连续批处理调度器)                          │
│      ↓ 调度好的请求                                  │
│ Worker (多进程/GPU，ZMQ 通信)                        │
│      ↓ 准备 batch                                    │
│ ModelRunner                                         │
│      ↓ 模型 + KV cache                              │
│ GPU (forward pass)                                   │
└─────────────────────────────────────────────────────┘
```

- **AsyncLLMEngine** 在主进程，负责调度 + IO
- **Worker** 在子进程，跑实际 GPU 计算（避免 GIL）
- **每张卡**一个 worker

### 5.3 PagedAttention 核心

**问题**：传统 KV cache 每个请求**连续**占一段显存，释放后留下空洞，浪费 60-80%。

**解决**：把 KV cache **分页**（类比 OS 虚拟内存）：
- 显存按 **block**（默认 16 tokens）分块
- 每个 block 包含固定数量 token 的 K/V
- 逻辑序列 → block 表 → 物理 block
- 类似**页表**的二级映射

```
请求的逻辑序列: [t1, t2, ..., t32768]
block 表: [P1, P2, ..., P2048]   # 每 block 16 token
物理显存: P1, P2, ... 散布在显存池里
```

**好处**：
- 零碎片、零浪费
- **block sharing**：同一前缀的不同请求共享 block（prefix caching）

### 5.4 Continuous Batching（连续批处理）

**传统静态批处理**：
```
batch = [req1(100), req2(200), req3(150)]
每轮都要等所有请求完成 → 短请求被长请求拖累 → GPU 利用率 30-50%
```

**连续批处理**：
```
时刻 t:   跑 req1[1..10], req2[1..20], req3[1..15]
时刻 t+1: req1 完成 → 新 req4 加入；跑 req2[21..40], req3[16..35], req4[1..5]
```

每步重新组装 batch，GPU 始终**接近满载**。

### 5.5 Chunked Prefill

**问题**：长 prompt (32K) 一次 prefill 占满 GPU 算力，其他请求卡住。

**解决**：把 prefill **切成 chunk**（如 4K 一块），与 decode 请求混合调度。

**vLLM 配置**：
```bash
--enable-chunked-prefill         # 默认 true
--max-num-batched-tokens 8192    # 每步总 token 上限
```

### 5.6 Prefix Caching

**原理**：两个请求有相同前缀（如 system prompt）→ **只算一次 prefill**。

**适用场景**：
- RAG（同一 system prompt）
- 多轮对话（历史共享）
- Tool use（工具定义共享）

**关键**：前缀必须**完全一致**（含空格、换行、tokenizer 处理后的序列）。

```bash
--enable-prefix-caching  # 默认开启
```

**监控**：
```bash
vllm:prefix_cache_hits_total
vllm:prefix_cache_queries_total
# 命中率 = hits / queries
```

### 5.7 张量并行（TP）

**原理**：把模型切分到多张卡：
- 行切（按 hidden dim）
- 列切（按 FFN）
- 头切（按 attention head）

**all-reduce**：每层输出时多卡通信合并。

**性能影响**：
- 有 NVLink（H100/A100）→ 开销小
- 无 NVLink（4090）→ 走 PCIe，慢 20 倍，**小模型变慢**
- **建议**：小模型（≤14B）跑多实例 + LB

### 5.8 投机解码（Speculative Decoding）

**原理**：用小 draft 模型猜 N 个 token，主模型一次验证：
```
draft:    t1, t2, t3, t4, t5
main:     verify t1~t5 (1 次 forward)
accept:   实际接受 3-4 个 → 节省 3-4 次 forward
```

**加速比**：1.5-2.5x（批量同质化请求最有效）

```bash
--speculative-model <draft_model>
--num-speculative-tokens 5
```

---

## 第 6 章：vLLM 启动与生命周期

### 6.1 启动流程

```
1. 解析命令行参数
2. 加载配置 → EngineArgs
3. 初始化 Worker + ModelRunner
4. 加载模型权重 → GPU
5. torch.compile（JIT 编译关键 kernel）
6. CUDA graph capture（减少 kernel launch 开销）
7. 启动 FastAPI/uvicorn
8. 注册路由 /v1/chat/completions 等
9. "Application startup complete"
```

**典型耗时**（35B FP8, 2×4090）：
- 加载模型：17.5s
- torch.compile：50s
- CUDA graph：7s
- **总启动：~4 分钟**

### 6.2 请求处理流程

```
HTTP → 解析 → AsyncLLMEngine.add_request() → EngineCore 队列
        ↓
     Scheduler 调度（本步要跑的请求）
        ↓
     Worker 准备 batch
        ↓
     Model forward（prefill + decode）
        ↓
     Sampling（top_p, temperature）
        ↓
     Streaming/SSE 推回客户端
```

### 6.3 关键参数全景

**模型加载**：
| 参数 | 含义 | 默认 | 建议 |
| --- | --- | --- | --- |
| `--model` | 模型路径或 HF repo_id | 必填 | 本地路径或 repo_id |
| `--dtype` | 权重 dtype | auto | auto / float16 / bfloat16 |
| `--quantization` | 量化方法 | None | awq / gptq / bitsandbytes / fp8 |
| `--max-model-len` | 最大上下文 | 模型默认 | 按业务设 |

**显存**：
| 参数 | 含义 | 默认 | 建议 |
| --- | --- | --- | --- |
| `--gpu-memory-utilization` | 显存利用率 | 0.90 | OOM 降到 0.85 |
| `--kv-cache-dtype` | KV 缓存类型 | auto | auto / fp8 |
| `--swap-space` | CPU swap (GB) | 4 | 内存大调到 8-16 |
| `--block-size` | PagedAttention block | 16 | 长 ctx 改 32 |

**调度**：
| 参数 | 含义 | 默认 | 建议 |
| --- | --- | --- | --- |
| `--max-num-seqs` | 同时处理的请求数 | 256 | 256(小) / 64-128(中) / 16(长 ctx) |
| `--max-num-batched-tokens` | 单步最大 token | 2048 | 4096-16384 |
| `--enable-chunked-prefill` | 启用 chunked prefill | True | **几乎总是 true** |

**服务**：
| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--host` | 监听地址 | 0.0.0.0 |
| `--port` | 端口 | 8000 |
| `--served-model-name` | 客户端用的模型名 | 模型 id |
| `--api-key` | 鉴权 | None |
| `--enable-metrics` | Prometheus 指标 | False |
| `--metrics-port` | 指标端口 | 8001 |

**高级**：
| 参数 | 含义 |
| --- | --- |
| `--enable-prefix-caching` | 启用前缀缓存 |
| `--enable-lora` | 启用 LoRA 适配器 |
| `--speculative-model` | 投机解码 draft 模型 |
| `--num-speculative-tokens` | draft token 数 |
| `--enforce-eager` | 禁用 CUDA graph（省显存，-5% 性能） |
| `--trust-remote-code` | 信任自定义模型代码 |

---

## 第 7 章：OpenAI 兼容 API 行为

### 7.1 端点

| 路径 | 方法 | 用途 |
| --- | --- | --- |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | 对话（推荐） |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 嵌入向量（需 Embedding 模型） |
| `/v1/messages` | POST | Anthropic 兼容 |
| `/metrics` | GET | Prometheus（需启用） |
| `/health` | GET | 健康检查 |

### 7.2 请求字段

```json
{
  "model": "qwen3.5-35b-a3b",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["\n\n"],
  "tools": [...],
  "tool_choice": "auto"
}
```

### 7.3 流式输出（SSE）

`stream: true` 返回 Server-Sent Events：
```
data: {"choices":[{"delta":{"content":"你"}}]}
data: {"choices":[{"delta":{"content":"好"}}]}
...
data: [DONE]
```

### 7.4 错误码

| HTTP | 含义 | 处理 |
| --- | --- | --- |
| 400 | 请求体错误 | 检查 `model`、`messages` 字段 |
| 401 | 未授权 | 检查 API Key |
| 404 | 模型不存在 | 确认 `SERVED_MODEL_NAME` |
| 422 | Pydantic 校验失败 | 看 `detail` |
| 503 | 服务尚未就绪 | 等启动完成 |
| 504 | 生成超时 | 调大 `timeout` 或减小 `max_tokens` |

---

# 第三篇：优化（Optimization）

> **优化** = 在固定硬件下，让推理更快、更省显存、更高质量。

## 第 8 章：LLM 推理性能原理

### 8.1 两个阶段

```
Pre-fill 阶段：处理整个 prompt 一次 forward
    ↓
Decode 阶段：逐 token 生成
    ↓
   每个 decode step：1 token 的 forward（auto-regressive）
```

**Pre-fill** 是 **compute-bound**（算力受限）：
- 大量 token 并行计算
- 吃满 Tensor Core
- TTFT 取决于此

**Decode** 是 **memory-bound**（带宽受限）：
- 每个 token 都要读一遍 KV cache 和权重
- 带宽决定速度
- TPOT 取决于此

### 8.2 性能指标

| 指标 | 含义 | 理想值（在线） |
| --- | --- | --- |
| **TTFT** (Time To First Token) | 首个 token 延迟 | < 300ms |
| **TPOT** (Time Per Output Token) | 每 token 延迟 | < 50ms |
| **E2E Latency** | 端到端 | TTFT + token×TPOT |
| **Throughput (req/s)** | 每秒请求 | 10-50 |
| **Throughput (out tok/s)** | 每秒输出 token | 越高越好 |
| **Throughput (in tok/s)** | 每秒输入 token | prefill 速度 |

### 8.3 三个阶段的瓶颈

```
请求生命周期:
1. 接收 + 排队    → 调度器
2. Pre-fill       → GPU 算力
3. Decode（循环）  → 显存带宽
4. 调度下一轮      → 调度器
```

| 阶段 | 瓶颈 | 现象 | 解法 |
| --- | --- | --- | --- |
| 排队 | 调度器满 | `num_requests_waiting` 高 | 加大 `MAX_NUM_SEQS` |
| Pre-fill | 算力 | GPU util 高但生成慢 | 减少 `MAX_NUM_BATCHED_TOKENS` |
| Decode | 显存带宽 | GPU util 低但 TPOT 高 | 量化（FP8/AWQ） |

### 8.4 算力 vs 带宽

```
LLM 推理的"算力 / 带宽"比 = 性能上限
4090: 82.6 TFLOPS / 1008 GB/s = 0.082
H100: 989 TFLOPS / 3350 GB/s = 0.295
H200: 989 TFLOPS / 4800 GB/s = 0.206  ← 推理最优（更高带宽）
B200:  2250 TFLOPS / 8000 GB/s = 0.281
```

**结论**：LLM 推理**带宽更重要**，所以 H200 推理比 H100 更优。

---

## 第 9 章：量化技术详解

### 9.1 目的

- 显存：FP16 14GB → INT4 4GB（节省 71%）
- 速度：内存带宽压力小 → 推理可能**快 1.5-2x**

### 9.2 量化粒度

| 粒度 | 精度 | 代表 |
| --- | --- | --- |
| Per-tensor | 最低 | 不推荐 |
| Per-channel | 中 | bitsandbytes |
| Per-group (32-128) | 高 | **AWQ, GPTQ** |
| Per-token | 高 | KV cache |

### 9.3 方案对比

| 方案 | 位宽 | 算法 | 质量 | 速度 | 适用 |
| --- | --- | --- | --- | --- | --- |
| **FP8** | 8 | E4M3/E5M2 | 几乎无损失 | 1.5x | **H100/4090 原生支持** |
| INT8 | 8 | PTQ/QAT | 几乎无损失 | 1.3x | 通用 |
| **AWQ** | 4 | 激活感知 | 优秀 | 2-3x | **推荐** |
| GPTQ | 4 | OBQ | 优秀 | 2-3x | 备选 |
| bitsandbytes NF4 | 4 | 分位数 | 良好 | 1.5-2x | 免下载量化版 |
| GGUF | 4-8 | llama.cpp | 良好 | 2x | CPU/GPU 通用 |

### 9.4 AWQ 原理

观察：**并非所有权重都重要**，少量"显著权重"（1%）贡献大部分精度
- 显著权重 → 保留 FP16
- 其余权重 → INT4
- **对激活值**分析（不是权重），找显著通道

vLLM 加载：
```bash
./download_model.sh qwen2.5-7b-awq   # 下载 AWQ 版
# 启动时 vLLM 自动识别，或显式：--quantization awq
```

### 9.5 FP8 详解

**两种格式**：
- **E4M3**（4+3）：精度高，范围 ±448，用于权重和激活
- **E5M2**（5+2）：范围 ±57344，精度低，用于梯度

**硬件支持**：
- **H100** (Hopper)：原生 FP8 Tensor Core，3000 TFLOPS
- **RTX 4090** (Ada)：第四代 Tensor Core **支持 FP8**，165 TFLOPS
- **A100** (Ampere)：**不支持** FP8

vLLM 加载：
- 已有 FP8 checkpoint（如 Qwen3.5-35B-A3B-FP8）→ 直接加载
- FP16 → FP8：`--quantization fp8`

### 9.6 量化决策

```
模型 ≥ 30B + 显存 < 32GB ─→ AWQ 4bit 或 GPTQ 4bit
模型 < 14B + 显存紧 ──────→ AWQ 4bit
模型 任意 + 显存够 ────────→ FP8 (H100/4090) 或 BF16
延迟极敏感 ────────────────→ AWQ 4bit (Decode 最快)
```

---

## 第 10 章：MoE 架构与优化

### 10.1 MoE 原理

**传统 Transformer**：每层 FFN，**所有 token** 都跑完整 FFN。

**MoE 改进**：把 FFN 拆成 N 个"专家"（小 FFN）+ 一个路由器：
```
x → Attention → Add & Norm → Router → 选 Top-K 专家 → 加权求和
```

**数学表达**：
```
router_logits = x · W_router            # (batch, seq, n_experts)
top_k_logits, top_k_idx = topk(router_logits, k=K)
weights = softmax(top_k_logits)
output = sum(weight_i · expert_i(x) for i in top_k_idx)
```

### 10.2 Qwen3.5-35B-A3B 解读

- **35B 总参数**：所有专家 + 路由器 + attention
- **3B 激活**：每个 token 只跑 3B 参数
- 推理速度 ≈ 3B 模型
- 知识容量 ≈ 35B 模型

**优点**：容量大、速度快、训练 FLOPs 低
**缺点**：
- 总显存**仍要装下 35B**
- 路由器增加一点计算
- expert 负载不均衡（需要 auxiliary loss）

### 10.3 MoE 部署注意事项

- **TP 切分**：专家按 round-robin 分配到卡
- **负载不均**：高并发时部分专家可能过载
- **专家并行 (EP)**：vLLM 实验性支持，可减少负载不均

---

## 第 11 章：推理性能调优实战

### 11.1 调参方法论（**必须遵守**）

```
1) 跑 baseline：./scripts/benchmark.sh 32 200，记录 throughput / p50 / p95
2) 改一个参数（不要同时改多个！）
3) 再跑一次，对比
4) 看 nvidia-smi：判断瓶颈
5) 重复 2-4 找到最优
```

### 11.2 显存优化（决定能跑多大）

| 方法 | 参数 | 节省 | 推荐度 |
| --- | --- | --- | --- |
| 量化 | `--quantization awq/fp8` | 50-75% | ⭐⭐⭐⭐⭐ |
| 减小上下文 | `--max-model-len` | 显著 | ⭐⭐⭐⭐ |
| KV 量化 | `--kv-cache-dtype fp8` | 30-50% KV | ⭐⭐⭐⭐ |
| 减并发 | `--max-num-seqs` | 减少 KV | ⭐⭐⭐ |
| 关 CUDA graph | `--enforce-eager` | ~2GB | ⭐⭐ |
| 降利用率 | `--gpu-memory-utilization 0.85` | 5% | ⭐⭐ |

### 11.3 吞吐优化（决定服务多少用户）

| 方法 | 参数 | 效果 | 推荐度 |
| --- | --- | --- | --- |
| 启用 prefix caching | `--enable-prefix-caching` | 节省重复 prefill | ⭐⭐⭐⭐⭐ |
| 启用 chunked prefill | `--enable-chunked-prefill` | 减少长 prompt 阻塞 | ⭐⭐⭐⭐⭐ |
| 加大并发 | `--max-num-seqs 256-512` | 提升吞吐 | ⭐⭐⭐⭐ |
| 加大 batch | `--max-num-batched-tokens 8192-16384` | 提升吞吐 | ⭐⭐⭐⭐ |
| 加大 swap | `--swap-space 8-16` | 减少拒绝 | ⭐⭐⭐ |
| 多实例 + LB | 跑多个 vLLM + nginx | 容量翻倍 | ⭐⭐⭐⭐ |
| 投机解码 | `--speculative-model` | 1.5-2.5x | ⭐⭐⭐ |

### 11.4 延迟优化（决定用户体验）

| 方法 | 参数 | 效果 |
| --- | --- | --- |
| `temperature=0` | 客户端 | 走 greedy，TPOT 减少 10-20% |
| 投机解码 | `--speculative-model` | 1.5-2.5x |
| 合理 batch | `--max-num-batched-tokens` ≤ 16K | 短请求公平调度 |
| 流式输出 | 客户端 `stream=True` | 改善感知延迟 |
| 限 `max_tokens` | 客户端 | 防止超长 |

### 11.5 瓶颈定位速查

| 现象 | 根因 | 解法 |
| --- | --- | --- |
| GPU util < 50% | 调度/排队 | MAX_NUM_SEQS ↑ |
| GPU util > 95% | 算力饱和 | 减 MAX_NUM_BATCHED_TOKENS 或投机 |
| 显存满 / OOM | KV cache 满 | 量化 / 减并发 / 减 ctx |
| TTFT > 1s | prefill 慢 | 减 ctx / chunked prefill |
| TPOT > 100ms | decode 慢 | 量化 / 减 ctx / 温度 0 |
| 拒绝率上升 | KV 满 | 加大 swap / 减 ctx |

### 11.6 本次 35B-A3B 调优案例

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

**CoT 影响**：Qwen3.5 默认 CoT，输出含 `<think>...</think>`。生产中：
- 客户端 `max_tokens=512` 限长
- system prompt 加 `"请直接回答，不要输出思考过程"`
- 客户端过滤 `<think>` 块

**进一步优化**：
- 关闭 CoT（如果支持）：`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- 加大 `MAX_NUM_SEQS`（显存还有余量）
- 多实例 + 负载均衡（替代 TP=2，避免 PCIe 通信开销）

---

# 第四篇：配置（Configuration）

> **配置** = 选什么模型、配什么参数、用什么部署拓扑。

## 第 12 章：模型选型方法论

### 12.1 选型三问

1. **任务场景**：通用对话 / 代码 / 长文档 / 推理 / 多语言？
2. **并发量级**：单用户/小团队（<5 并发）vs online serving（≥32 并发）？
3. **可接受显存**：单卡 24GB / 48GB / 双卡 48×2？

### 12.2 决策矩阵（2×4090 环境）

| 任务 | 推荐 | 显存(BF16) | 部署 |
| --- | --- | --- | --- |
| 通用对话 | Qwen2.5-14B | 30GB | 单卡 |
| 中文 SOTA | Qwen2.5-32B | 65GB | 双卡 TP=2 |
| 代码生成 | Qwen2.5-Coder-14B | 30GB | 单卡 |
| 极小显存 | Qwen2.5-7B-AWQ | 8GB | 单卡 |
| 128K 长文 | Qwen2.5-14B | 30GB | MAX_LEN=128K |
| 多模态 VL | Qwen3.5-35B-A3B-FP8 | 35GB | 双卡 TP=2 |

### 12.3 模型许可

| 模型 | License | 商用 |
| --- | --- | --- |
| Qwen 系列 | Apache 2.0 | ✅ |
| Llama 3 | Meta License（需同意） | ✅ |
| DeepSeek | 自研（基本允许） | ✅ |
| Gemma | Google（需同意） | 受限 |

### 12.4 选型清单

- [ ] 任务匹配（通用/代码/长文/多模态）
- [ ] 显存匹配（模型 ≤ 显存 × 0.6）
- [ ] 上下文长度匹配（业务需求 ≤ 模型 max_len）
- [ ] 量化版可用（AWQ/FP8）
- [ ] License 允许商用
- [ ] 生态成熟（vLLM 支持、HF 下载量）

---

## 第 13 章：vLLM 参数配置详解

### 13.1 配置层次

```
环境变量 (.env)        ← 推荐：环境相关
  ↓
docker-compose.yml    ← 推荐：编排
  ↓
vllm CLI 参数         ← 直接传
```

### 13.2 配置预设

**7B 单卡**（Qwen2.5-7B / Llama3-8B）：
```env
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=8192
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
```

**14B 单卡**（Qwen2.5-14B / Coder-14B）：
```env
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=8192
```

**32B AWQ 单卡**：
```env
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=4096
QUANTIZATION=awq
```

**32B BF16 双卡 TP=2**：
```env
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=4096
```

**70B AWQ 双卡**：
```env
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=2048
QUANTIZATION=awq
```

**128K 长文**：
```env
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.95
KV_CACHE_DTYPE=fp8
```

**35B-A3B FP8 双卡**（本次部署）：
```env
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=8192
QUANTIZATION=fp8
```

**在线高吞吐**：
```env
MAX_NUM_SEQS=512
MAX_NUM_BATCHED_TOKENS=16384
SWAP_SPACE=8
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
```

### 13.3 客户端配置（不需重启）

| 参数 | 推荐 | 说明 |
| --- | --- | --- |
| `temperature` | 0.7 | 0=greedy，0.7-1 平衡，>1.5 发散 |
| `top_p` | 0.9-0.95 | 核采样 |
| `top_k` | 40-100 | top-k 采样 |
| `max_tokens` | 256-2048 | 业务控制 |
| `stream` | true | 改善感知延迟 |
| `stop` | 自定义 | 停止符 |
| `seed` | 固定值 | 结果可复现 |

---

## 第 14 章：多实例与负载均衡

### 14.1 为什么多实例

- 故障隔离
- 容量扩展
- 避免 TP 通信开销
- 蓝绿/灰度发布

### 14.2 部署模式

**单实例**（最简）：
```
[vLLM] → [客户端]
```

**多实例 + Nginx**：
```
[vLLM-1, GPU0]  ┐
[vLLM-2, GPU1]  ┴→ [Nginx LB] → [客户端]
```

**多机集群**（K8s）：
```
[vLLM-Pod1]  ┐
[vLLM-Pod2]  ┴→ [K8s Service] → [客户端]
[vLLM-Pod3]  │
```

### 14.3 Nginx 配置示例

```nginx
upstream vllm_backends {
    least_conn;
    server 10.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 10.0.0.2:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    location /v1/ {
        proxy_pass http://vllm_backends;
        proxy_set_header Host $host;
        proxy_read_timeout 600;
        proxy_send_timeout 600;
    }
}
```

### 14.4 多实例启动示例

```bash
# 终端 1
NVIDIA_VISIBLE_DEVICES=0 PORT=8000 \
  docker compose -p vllm-1 up -d

# 终端 2
NVIDIA_VISIBLE_DEVICES=1 PORT=8001 \
  docker compose -p vllm-2 -f docker-compose.yml up -d
```

---

# 第五篇：运维（Operations & Maintenance）

> **运维** = 服务上线后，监控、调优、扩缩容、安全、灾备。

## 第 15 章：监控与可观测性

### 15.1 GPU 监控

```bash
# 一次性
nvidia-smi

# 每秒刷新
nvidia-smi -l 1

# 进程级每秒
nvidia-smi pmon -s u -c 1

# 关键字段
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 1
```

**关键指标**：

| 指标 | 健康值 | 异常处理 |
| --- | --- | --- |
| `utilization.gpu` | > 70% | 调度瓶颈或小模型 |
| `memory.used/total` | < 90% | 减并发或量化 |
| `temperature.gpu` | < 85°C | 散热 |
| `utilization.memory` | > 60% | decode 正常 |

### 15.2 vLLM 监控

启用 Prometheus：
```bash
# docker-compose.yml
- --enable-metrics
- --metrics-port=8001
ports:
  - "8001:8001"
```

**关键 metrics**：

| 指标 | 含义 |
| --- | --- |
| `vllm:gpu_cache_usage_perc` | GPU KV 缓存占用率 |
| `vllm:cpu_cache_usage_perc` | CPU swap 缓存占用率 |
| `vllm:num_requests_running` | 正在 decode |
| `vllm:num_requests_waiting` | 排队 |
| `vllm:num_requests_swapped` | 被换出到 CPU |
| `vllm:e2e_request_latency_seconds_bucket` | 端到端延迟分布 |
| `vllm:time_to_first_token_seconds_bucket` | TTFT 分布 |
| `vllm:prompt_tokens_total` | 输入 token 累计 |
| `vllm:generation_tokens_total` | 输出 token 累计 |
| `vllm:request_success_total` | 成功请求数 |
| `vllm:request_failure_total` | 失败请求数 |
| `vllm:prefix_cache_hits_total` | 前缀缓存命中 |
| `vllm:prefix_cache_queries_total` | 前缀缓存查询 |

### 15.3 Grafana 仪表盘

社区有现成 vLLM 仪表盘 JSON：https://github.com/vllm-project/vllm/tree/main/examples

**推荐面板**：
- 实时 QPS、p50/p95/p99 延迟
- GPU util、显存、温度
- KV cache usage
- Prefix cache hit rate
- 请求成功率

### 15.4 日志

```bash
./scripts/logs.sh                      # 实时跟踪
docker compose logs vllm --tail=200     # 最近 200 行
docker compose logs vllm 2>&1 | grep -E "ERROR|WARN"  # 错误
```

### 15.5 告警建议

| 指标 | 警告阈值 | 告警阈值 | 通知 |
| --- | --- | --- | --- |
| GPU 温度 | > 80°C | > 85°C | 5min |
| 显存占用 | > 90% | > 95% | 5min |
| 请求失败率 | > 1% | > 5% | 1min |
| p95 延迟 | > 基线 2x | > 基线 5x | 5min |
| 容器 OOM 重启 | 1 次/小时 | 3 次/小时 | 立即 |
| 磁盘剩余 | < 20% | < 10% | 1h |

---

## 第 16 章：性能压测

### 16.1 压测目标

- 找并发数拐点
- 量化参数效果
- 回归测试
- 容量规划数据

### 16.2 压测工具

**vllm bench**（推荐）：
```bash
vllm bench serve \
  --host http://localhost:8000 \
  --model <model_name> \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sonnet \
  --max-concurrency 32 \
  --num-prompts 200 \
  --save-result
```

支持的 dataset：
- `sonnet`：英文长文本
- `sharegpt`：多轮对话
- `random`：随机长度
- `hf`：自定义 HF dataset

**自定义压测**（`scripts/benchmark.sh` / `bench_http.py`）：
- 并发控制
- 记录 TTFT、TPOT、p50/p95/p99
- 输出 JSON 结果

### 16.3 压测指标解读

```json
{
  "throughput_req_s": 8.67,         // 每秒请求
  "throughput_out_tok_s": 1109,     // 每秒输出 token
  "p50_s": 5.22,                     // 中位延迟
  "p95_s": 7.38                      // 95% 延迟
}
```

**判断**：
- 改并发数 → 找 `throughput_out_tok_s` 拐点 = 最佳并发
- 看 GPU util / 显存 → 判断是否到达瓶颈
- TTFT 看 prefill，TPOT 看 decode

### 16.4 容量规划公式

```
单卡吞吐 = 实际压测（不要信理论值）
所需卡数 = ceil(目标 QPS × 平均 token / 单卡吞吐 × 1.5)
```

---

## 第 17 章：高可用与灾备

### 17.1 健康检查

docker-compose 已配：
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -fsS http://localhost:8000/v1/models || exit 1"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 180s
```

### 17.2 自动重启

```yaml
restart: unless-stopped
```

### 17.3 多实例 + LB

参见 [第 14 章](#第-14-章多实例与负载均衡)。

### 17.4 优雅退出

vLLM 收到 SIGTERM 时排空正在处理的请求。

### 17.5 灾备脚本

```bash
# 备份
tar czf vllm-backup-$(date +%Y%m%d).tar.gz \
  /opt/vllm/.env /opt/vllm/docker-compose.yml /opt/vllm/scripts

# 模型备份（按需）
tar czf models-backup-$(date +%Y%m%d).tar.gz \
  /opt/vllm/models/<repo>

# 恢复
tar xzf vllm-backup-*.tar.gz -C /
tar xzf models-backup-*.tar.gz -C /opt/vllm/models/
cd /opt/vllm && ./scripts/start.sh
```

### 17.6 升级与变更

```bash
# 1) 备份 .env
cp .env .env.bak

# 2) 拉新镜像
docker pull vllm/vllm-openai:latest

# 3) 滚动重启（多实例）
for i in 1 2; do
  docker compose -p vllm-$i restart
  sleep 30
done

# 4) 验证
python3 scripts/test_api.py
```

---

## 第 18 章：安全与合规

### 18.1 鉴权

```bash
# .env
VLLM_API_KEY=sk-your-strong-secret

# 客户端
client = OpenAI(api_key="sk-your-strong-secret", ...)
```

### 18.2 网络隔离

```bash
# vLLM 仅监听内网
HOST=127.0.0.1

# 前置 Nginx 暴露 443
server {
  listen 443 ssl;
  location /v1/ {
    proxy_pass http://127.0.0.1:8000/v1/;
    # 限流 / 鉴权 / 审计
  }
}
```

### 18.3 容器最小权限

- 自定义 Dockerfile 加 `USER`
- 限制 cap
- read-only root fs
- 关闭特权模式

### 18.4 数据保护

- 提示词和输出可能含敏感信息
- 加密传输（TLS）
- 日志脱敏
- 定期清理缓存

### 18.5 合规

- 各模型 License 商用前确认
- 用户数据保护（GDPR/个保法）
- 内容审核（可选前置 vLLM 实例专门做审核）

---

## 第 19 章：容量规划

### 19.1 估算流程

```
1) 业务目标：峰值 QPS、并发数、响应时间
2) 模型选择：根据质量需求
3) 压测：单卡吞吐
4) 算卡数：ceil(目标 QPS × 平均 token / 单卡吞吐 × 1.5)
5) 留 buffer：考虑 1.5x 突发流量
```

### 19.2 成本优化

| 优化 | 措施 | 节省 |
| --- | --- | --- |
| 模型大小 | AWQ/FP8 量化 | 显存 50%+ |
| 上下文 | 业务允许内尽量小 | 显存 30%+ |
| 批处理 | 加大 `MAX_NUM_SEQS` | 卡数 -30% |
| 闲时关停 | 云上低峰期关实例 | 50% |
| 模型选择 | 7B 够就别 70B | 卡数 1/10 |

### 19.3 扩缩容策略

- **手动扩缩容**：监控 + 人工
- **半自动**：HPA based on GPU util / QPS
- **全自动**：K8s HPA + 自定义指标

---

# 附录

## A. 命令速查

```bash
# GPU
nvidia-smi
nvidia-smi -l 1
nvidia-smi pmon -s u

# Docker
docker ps
docker stats vllm-server
docker exec -it vllm-server bash
docker compose logs -f vllm

# vLLM 运维
cd /opt/vllm
./scripts/start.sh
./scripts/stop.sh
./scripts/logs.sh
python3 scripts/test_api.py
./scripts/benchmark.sh 32 200
curl http://localhost:8000/v1/models
curl http://localhost:8001/metrics  # 启用 --enable-metrics 后

# 模型
./download_model.sh list
./download_model.sh <short_name>
```

## B. 参考资源

- vLLM 官方：https://docs.vllm.ai
- vLLM GitHub：https://github.com/vllm-project/vllm
- HuggingFace：https://huggingface.co
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- NVIDIA Container Toolkit：https://github.com/NVIDIA/nvidia-container-toolkit
- AWQ 论文：https://arxiv.org/abs/2306.00978
- CUDA Toolkit 文档：https://docs.nvidia.com/cuda/

## C. 术语表

| 术语 | 含义 |
| --- | --- |
| LLM | Large Language Model |
| VLM | Vision-Language Model |
| MoE | Mixture of Experts |
| TP | Tensor Parallel |
| PP | Pipeline Parallel |
| KV Cache | Key-Value 缓存 |
| Prefill | 第一个 token 之前的全量计算 |
| Decode | 逐 token 生成 |
| AWQ | Activation-aware Weight Quantization |
| GPTQ | GPT Quantization |
| FP8 | 8 位浮点（E4M3 / E5M2） |
| CoT | Chain of Thought，思维链 |
| PagedAttention | vLLM 核心，分页注意力 |
| CDI | Container Device Interface |
| OCI | Open Container Initiative |
| TTFT | Time To First Token |
| TPOT | Time Per Output Token |

---

> 文档结束。配合 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 使用效果最佳。
