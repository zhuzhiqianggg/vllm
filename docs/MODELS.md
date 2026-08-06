# 模型部署与切换手册

> 面向研发和运维：如何快速下载、切换、测试不同大模型

---

## 一、当前已部署模型

### 1. Qwen3.6-35B-A3B-FP8（当前生产模型）

| 属性 | 值 |
| --- | --- |
| 架构 | Qwen3_5MoeForConditionalGeneration |
| 总参数量 | 35B（激活 3B，MoE） |
| 精度 | FP8（量化） |
| 上下文 | 原生 262144，当前设 128000 |
| 显存占用 | ~44GB / 2卡 |
| 特点 | 多模态（图文）、思考模型、工具调用 |
| 服务名 | `qwen3.6-35b-a3b` |

### 2. Qwen3.6-27B（新增待切换）

| 属性 | 值 |
| --- | --- |
| 架构 | Qwen3_5ForConditionalGeneration（dense） |
| 参数量 | 27B（全密集） |
| 精度 | BF16 |
| 上下文 | 原生 262144，当前设 128000 |
| 显存占用 | ~54GB / 2卡 |
| 特点 | 文本模型、支持思考、支持工具调用 |
| 服务名 | `qwen3.6-27b` |

### 性能对比

| 指标 | 35B-A3B-FP8（MoE） | 27B（Dense） |
| --- | --- | --- |
| 单 token 延迟 | ~50-80ms | ~30-50ms |
| 并发能力 | 64 seq | 32 seq |
| 多模态 | ✅ 图文 | ❌ 纯文本 |
| 思考模式 | ✅ | ✅ |
| 推理速度 | 中等（MoE 路由开销） | 快（dense 并行） |

---

## 二、模型下载

### 2.1 使用脚本下载

```bash
cd /opt/vllm

# 列出所有预设模型
./download_model.sh list

# 下载预设模型
./download_model.sh qwen3.6-27b

# 下载任意 HF 模型
./download_model.sh custom Qwen/Qwen2.5-72B-Instruct

# 通过 ModelScope 下载
./download_model.sh ms Qwen/Qwen2.5-7B-Instruct
```

### 2.2 目录结构

```
/opt/vllm/models/
├── Qwen__Qwen3.6-35B-A3B-FP8/   # MoE 多模态模型
│   ├── config.json
│   ├── model-00001-of-0015.safetensors
│   └── ...
└── Qwen__Qwen3.6-27B/           # 27B 密集模型
    ├── config.json
    ├── model-00001-of-0015.safetensors
    └── ...
```

---

## 三、模型切换

### 3.1 使用切换脚本（推荐）

```bash
cd /opt/vllm

# 查看所有预设配置
./switch_model.sh list

# 查看当前配置
./switch_model.sh current

# 切换到 27B 模型
./switch_model.sh qwen3.6-27b

# 切换回 MoE 多模态模型
./switch_model.sh qwen3.5-35b-fp8

# 测试 fp8_e4m3 KV cache
./switch_model.sh qwen3.6-27b-fp8-e4m3

# 开启思考模式
./switch_model.sh qwen3.6-27b-thinking
```

脚本会自动：
1. 备份当前 `.env` → `.env.bak`
2. 写入新配置到 `.env`
3. 强制重建容器并重启服务
4. 等待健康检查通过
5. 显示新配置摘要

### 3.2 手动切换（可选）

```bash
# 1. 编辑 .env
vim /opt/vllm/.env

# 2. 修改关键参数
MODEL_NAME=/root/.cache/huggingface/Qwen__Qwen3.6-27B
SERVED_MODEL_NAME=qwen3.6-27b
# 其他参数按需调整

# 3. 重启服务
cd /opt/vllm && docker compose up -d --force-recreate

# 4. 验证
curl -s http://localhost:8000/v1/models
```

---

## 四、预设配置一览

| 配置名 | 模型 | KV类型 | 上下文 | 并发 | 思考 |
| --- | --- | --- | --- | --- | --- |
| `qwen3.5-35b-fp8` | 35B MoE | fp8 | 128K | 64 | 关 |
| `qwen3.6-27b` | 27B dense | fp8 | 128K | 32 | 关 |
| `qwen3.6-27b-fp8-e4m3` | 27B dense | fp8_e4m3 | 128K | 32 | 关 |
| `qwen3.6-27b-thinking` | 27B dense | fp8 | 128K | 32 | 开 |

---

## 五、API 调用

### 5.1 获取当前模型

```bash
curl http://192.168.2.11:8000/v1/models
```

返回：
```json
{
  "data": [{
    "id": "qwen3.6-27b",
    "max_model_len": 128000
  }]
}
```

### 5.2 对话请求

```bash
curl -X POST http://192.168.2.11:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'
```

### 5.3 流式输出

```bash
curl -N -X POST http://192.168.2.11:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","stream":true,"messages":[{"role":"user","content":"hello"}]}'
```

### 5.4 工具调用

```json
POST /v1/chat/completions
{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "北京天气"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }
  }]
}
```

---

## 六、常用运维命令

```bash
# 查看服务状态
cd /opt/vllm && docker compose ps

# 查看日志
docker compose logs vllm --tail=100

# 实时日志
docker compose logs -f vllm

# 查看 GPU 状态
nvidia-smi

# 重启服务
docker compose restart vllm

# 完全重建
docker compose up -d --force-recreate

# 停止服务
docker compose down

# 性能测试
curl -s http://localhost:8000/metrics | grep vllm:

# 健康检查
curl http://localhost:8000/health
```

---

## 七、故障排查速查

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 启动失败 OOM | 显存不足 | 降 `gpu-memory-utilization` 或换更小模型 |
| 模型不存在 | `SERVED_MODEL_NAME` 不匹配 | 查 `/v1/models` 获取正确 ID |
| 加载超时 | 磁盘 IO 慢或模型大 | 等待几分钟，查看 `nvidia-smi` |
| 推理慢 | 上下文过长 | 降 `MAX_MODEL_LEN` 或换 `fp8_e4m3` |
| 429 错误 | 并发超限 | 客户端限流，服务端降 `MAX_NUM_SEQS` |
| 思考输出 | `enable_thinking` 未关闭 | 检查 `CHAT_TEMPLATE_KWARGS` |

---

## 八、KV Cache 类型选择指南

| 类型 | 大小/元素 | 精度 | 适用场景 |
| --- | --- | --- | --- |
| `fp16` | 4 bytes | 无损 | 最高精度要求 |
| `fp8` | 2 bytes | 极小损失 | **推荐（95% 场景）** |
| `fp8_e4m3` | 1.5 bytes | 小损失 | 显存紧张/高并发 |

修改方法：
```bash
# 方法1：切换脚本
./switch_model.sh qwen3.6-27b-fp8-e4m3

# 方法2：改 .env
KV_CACHE_DTYPE=fp8_e4m3
docker compose up -d --force-recreate
```

详细参数说明见 [docs/API.md#11.2](../docs/API.md)
