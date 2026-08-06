# 知识体系快速索引

> 文档入口与速查。所有文件位于 `/opt/vllm/`。

## 主文档

| 文档 | 内容 | 推荐阅读顺序 |
| --- | --- | --- |
| [README.md](../README.md) | 部署总览 + 快速开始 | 1 |
| [FINAL_STATUS.md](../FINAL_STATUS.md) | 本次部署最终状态 | 2 |
| [DEPLOY_NOTES.md](../DEPLOY_NOTES.md) | 部署笔记 | 3 |
| [**docs/DEEP_DIVE.md**](DEEP_DIVE.md) | **深度知识手册（5 篇 19 章）** | **深度学习必读** |
| [**docs/TROUBLESHOOTING.md**](TROUBLESHOOTING.md) | **故障排查 + 优化方案** | **实战速查** |
| [docs/MODELS.md](MODELS.md) | 模型选型指南 | 4 |
| [docs/EXAMPLES.md](EXAMPLES.md) | API 快速示例 | 5 |
| [**docs/API.md**](API.md) | **API 完整对接文档（研发用）** | **对接必读** |
| [PLAN.md](../PLAN.md) | 整体规划 | 备查 |

## DEEP_DIVE.md 五维知识体系

按 **部署 → 运行 → 优化 → 配置 → 运维** 五个角度系统化整理：

### 第一篇：部署（Deployment）
- 第 1 章 GPU 硬件深度（架构、显存、4090、选型决策树）
- 第 2 章 NVIDIA 驱动 / CUDA / Toolkit 体系
- 第 3 章 Docker + GPU 容器化（legacy runtime vs CDI）
- 第 4 章 模型下载与管理

### 第二篇：运行（Operation）
- 第 5 章 vLLM 内部原理（PagedAttention、连续批处理、Chunked Prefill、Prefix Caching、TP、投机解码）
- 第 6 章 vLLM 启动与生命周期（参数全景表）
- 第 7 章 OpenAI 兼容 API 行为

### 第三篇：优化（Optimization）
- 第 8 章 LLM 推理性能原理（prefill vs decode、算力 vs 带宽）
- 第 9 章 量化技术详解（FP8 / AWQ / GPTQ）
- 第 10 章 MoE 架构与优化
- 第 11 章 推理性能调优实战（含 35B-A3B 调优案例）

### 第四篇：配置（Configuration）
- 第 12 章 模型选型方法论
- 第 13 章 vLLM 参数配置详解（含 8 个场景的预设配置）
- 第 14 章 多实例与负载均衡（Nginx）

### 第五篇：运维（Operations & Maintenance）
- 第 15 章 监控与可观测性（GPU + vLLM metrics）
- 第 16 章 性能压测（vllm bench）
- 第 17 章 高可用与灾备
- 第 18 章 安全与合规
- 第 19 章 容量规划

## TROUBLESHOOTING.md 实战手册

按 **问题类别 → 优化方案 → 运维监控** 组织：

| 章节 | 内容 |
| --- | --- |
| 一 | 问题排查总流程 + 速查表 |
| 二 | 部署阶段问题（toolkit 缺失、驱动缓存、vulkan 路径、compose 路径） |
| 三 | 启动阶段问题（forward compat、参数废弃、镜像拉取、Loading 卡顿） |
| 四 | 加载与显存问题（OOM、max sequence length、模型不完整） |
| 五 | 推理阶段问题（404、prefix cache、tool use、延迟、CoT、503） |
| 六 | 网络与下载问题（Docker、HF 镜像、xet 401、私有模型） |
| 七 | **性能优化方案**（指标、显存、吞吐、延迟、调参方法论、推荐配置、35B-A3B 案例） |
| 八 | 运维监控告警（GPU/vLLM 指标、Grafana、告警阈值） |
| 九 | 彻底重置与恢复 |
| 十 | 命令速查 |

## 学习路径建议

### 路径 1：我是初学者
1. README → DEPLOY_NOTES（看一遍实战）
2. TROUBLESHOOTING.md 速查表（看会遇到什么）
3. DEEP_DIVE 第一篇（GPU 基础 + 容器化）
4. DEEP_DIVE 第二篇前半（vLLM 是什么）

### 路径 2：我要上线服务
1. DEEP_DIVE 第一篇 1-3 章（驱动/CUDA/容器）
2. DEEP_DIVE 第二篇 5-6 章（vLLM 机制+参数）
3. DEEP_DIVE 第三篇 11 章（调优实战）
4. DEEP_DIVE 第四篇 13-14 章（配置+LB）
5. DEEP_DIVE 第五篇 15-17 章（监控+HA）
6. TROUBLESHOOTING.md 第七章（优化方案）

### 路径 3：我要选 GPU 采购
1. DEEP_DIVE 第一篇 第 1 章（GPU 硬件深度）
2. MODELS.md（模型规模与显存）
3. DEEP_DIVE 第三篇 9-10 章（量化 + MoE）

### 路径 4：我要解决具体问题
1. TROUBLESHOOTING.md 速查表（看症状关键词）
2. 跳到对应章节
3. 实在找不到 → DEEP_DIVE 第一篇/第二篇原理部分

### 路径 5：我要对接 API
1. [docs/API.md](API.md) — 完整接口规范（cURL + Python + Node + Go + Java）
2. [docs/EXAMPLES.md](EXAMPLES.md) — 常用场景代码片段
3. DEEP_DIVE 第二篇 第 7 章（OpenAI API 行为细节）

## 关键知识卡片

> 每日记忆，1 周即可上手

### 🔥 必须记住的 5 件事

1. **驱动 ≥ CUDA Toolkit 要求的最低版本**（向上不兼容 → "Error 804"）
2. **PagedAttention = 显存分页 + Prefix Caching 共享**（vLLM 核心）
3. **LLM 推理是 memory-bound**（带宽决定速度，不是算力）
4. **Decode 阶段读模型权重 = 显存带宽瓶颈**（量化可提速）
5. **4090 无 NVLink**（TP=2 走 PCIe 比 NVLink 慢 20 倍）

### 🔥 遇到问题先看

| 现象 | 看哪 |
| --- | --- |
| 容器内 `nvidia-smi` 失败 | TROUBLESHOOTING §2.1 / DEEP_DIVE §3.6 |
| `Error 804 forward compatibility` | TROUBLESHOOTING §3.1 / DEEP_DIVE §2.2 |
| OOM | TROUBLESHOOTING §4.1 / DEEP_DIVE §11.2 |
| 模型下载失败 | TROUBLESHOOTING §6 / DEEP_DIVE §4 |
| vLLM 启动失败 | TROUBLESHOOTING §3 / DEEP_DIVE §6 |
| 推理慢/延迟高 | TROUBLESHOOTING §7 / DEEP_DIVE §11 |
| 选什么卡 | DEEP_DIVE §1.5 |
| 参数怎么配 | DEEP_DIVE §13.2 |
