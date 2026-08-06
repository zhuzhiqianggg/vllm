# vLLM 大模型 API 接口文档

> **服务地址**：`http://192.168.2.11:8000`
> **协议**：OpenAI 兼容（REST + JSON）
> **鉴权**：未启用（按需在 `.env` 设 `VLLM_API_KEY`）
> **当前模型**：`qwen3.6-35b-a3b`（Qwen3.6-35B-A3B-FP8）
> **最大上下文**：32768 tokens（输入+输出）

---

## 一、健康检查

```bash
curl http://192.168.2.11:8000/health
# 200 → 就绪
# 503 → 还在加载模型
```

---

## 二、接口列表

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET`  | `/v1/models` | 查看可用模型 |
| `POST` | `/v1/chat/completions` | **对话（主要用这个）** |
| `POST` | `/v1/completions` | 文本续写 |
| `POST` | `/v1/embeddings` | 文本向量化（需 Embedding 模型） |

---

## 三、GET /v1/models

**请求**

```bash
curl http://192.168.2.11:8000/v1/models
```

**响应**

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.6-35b-a3b",
      "object": "model",
      "created": 1785412340,
      "owned_by": "vllm",
      "max_model_len": 32768
    }
  ]
}
```

返回字段：
- `id`：客户端调用时填到 `model` 字段的值
- `max_model_len`：单次请求的输入+输出 token 上限

---

## 四、POST /v1/chat/completions（核心接口）

### 4.1 请求参数

| 参数 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `model` | string | ✅ | — | 固定 `qwen3.6-35b-a3b` |
| `messages` | array | ✅ | — | 对话历史，详见 4.2 |
| `temperature` | float | ❌ | 1.0 | 0=贪心，越大越发散，范围 0~2 |
| `top_p` | float | ❌ | 1.0 | nucleus 采样，0~1 |
| `max_tokens` | int | ❌ | 不限 | 输出 token 上限 |
| `stream` | bool | ❌ | false | true=流式（SSE） |
| `stop` | string\|array | ❌ | null | 遇到则停止生成 |
| `n` | int | ❌ | 1 | 一次返回几个候选 |
| `presence_penalty` | float | ❌ | 0 | -2~2，主题重复惩罚 |
| `frequency_penalty` | float | ❌ | 0 | -2~2，高频词惩罚 |
| `user` | string | ❌ | null | 业务用户 ID（审计用） |
| `seed` | int | ❌ | null | 随机种子（仅采样相关） |
| `tools` | array | ❌ | null | 工具定义（见 4.5） |
| `tool_choice` | string\|object | ❌ | "auto" | 工具选择：`auto`/`none`/`required` |
| `response_format` | object | ❌ | null | 强制 JSON 输出：`{"type":"json_object"}` |

### 4.2 messages 结构

```json
{
  "messages": [
    {"role": "system",    "content": "你是助手"},
    {"role": "user",      "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user",      "content": "介绍一下你自己"}
  ]
}
```

`role` 取值：

| 值 | 含义 |
| --- | --- |
| `system` | 系统提示词（建议作为第一条） |
| `user` | 用户输入 |
| `assistant` | 模型上轮回复（多轮对话时塞回去） |
| `tool` | 工具执行结果（配合 `tools` 使用） |

`content` 支持纯字符串或多模态数组（图文混排）：

```json
{
  "role": "user",
  "content": [
    {"type": "text",      "text": "描述这张图"},
    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}
  ]
}
```

### 4.3 响应（非流式）

```json
{
  "id": "chatcmpl-bc8d3e6c106699c9",
  "object": "chat.completion",
  "created": 1785412344,
  "model": "qwen3.6-35b-a3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "我是 Qwen3.6-35B-A3B，一个大语言模型。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 12,
    "total_tokens": 26
  },
  "system_fingerprint": "vllm-0.26.0-tp2-c51cd7ad"
}
```

**返回字段**：

| 字段 | 说明 |
| --- | --- |
| `id` | 请求唯一 ID（排错用） |
| `choices[].message.content` | 生成的文本内容 |
| `choices[].finish_reason` | `stop` 正常结束 / `length` 触达 max_tokens / `tool_calls` 调用工具 / `error` 出错 |
| `usage.prompt_tokens` | 输入 token 数 |
| `usage.completion_tokens` | 输出 token 数 |
| `usage.total_tokens` | 上面两者之和（计费口径） |
| `system_fingerprint` | 后端实例指纹，便于排查 |

### 4.4 调用示例

**cURL**

```bash
curl -X POST http://192.168.2.11:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [
      {"role": "system", "content": "你是一个简洁的助手"},
      {"role": "user", "content": "用 50 字介绍 vLLM"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

**Python（OpenAI 官方 SDK）**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.2.11:8000/v1",   # 注意带 /v1
    api_key="EMPTY",                     # 未开鉴权时任意非空字符串
)

resp = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[
        {"role": "system", "content": "你是资深 SRE"},
        {"role": "user",   "content": "K8s pod OOM 怎么排查？"},
    ],
    temperature=0.6,
    max_tokens=512,
)

print(resp.choices[0].message.content)
print("tokens:", resp.usage.total_tokens)
```

**Node.js**

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://192.168.2.11:8000/v1",
  apiKey: "EMPTY",
});

const resp = await client.chat.completions.create({
  model: "qwen3.6-35b-a3b",
  messages: [
    { role: "system", content: "你是资深 SRE" },
    { role: "user",   content: "K8s pod OOM 怎么排查？" },
  ],
  temperature: 0.6,
  max_tokens: 512,
});

console.log(resp.choices[0].message.content);
```

**Go**

```go
import (
    openai "github.com/openai/openai-go"
)

client := openai.NewClient(
    openai.WithBaseURL("http://192.168.2.11:8000/v1"),
    openai.WithAPIKey("EMPTY"),
)

resp, _ := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
    Model: "qwen3.6-35b-a3b",
    Messages: []openai.ChatCompletionMessageParamUnion{
        openai.SystemMessage("你是资深 SRE"),
        openai.UserMessage("K8s pod OOM 怎么排查？"),
    },
    Temperature: openai.Float(0.6),
    MaxTokens:   openai.Int(512),
})
fmt.Println(resp.Choices[0].Message.Content)
```

**Java**

```java
OpenAIClient client = OpenAIOkHttpClient.builder()
    .baseUrl("http://192.168.2.11:8000/v1")
    .apiKey("EMPTY")
    .build();

ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
    .model("qwen3.6-35b-a3b")
    .addSystemMessage("你是资深 SRE")
    .addUserMessage("K8s pod OOM 怎么排查？")
    .maxTokens(512)
    .build();

ChatCompletion resp = client.chat().completions().create(params);
System.out.println(resp.choices().get(0).message().content().orElse(""));
```

### 4.5 工具调用（Function Calling）

**第一步：声明工具 + 用户提问**

```json
{
  "model": "qwen3.6-35b-a3b",
  "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "获取指定城市的实时天气",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "城市名"}
        },
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}
```

**第二步：模型返回 `finish_reason=tool_calls`**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\":\"北京\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**第三步：把工具执行结果回喂**

```json
{
  "model": "qwen3.6-35b-a3b",
  "messages": [
    {"role": "user", "content": "北京今天天气怎么样？"},
    {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\":\"北京\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "{\"temp\":18,\"weather\":\"晴\"}"
    }
  ]
}
```

**完整 Python 示例**

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://192.168.2.11:8000/v1", api_key="EMPTY")

def get_weather(city): return json.dumps({"city": city, "temp": 18, "weather": "晴"})

messages = [{"role": "user", "content": "北京天气怎么样？"}]
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取城市天气",
        "parameters": {"type": "object", "properties": {"city": {"type":"string"}}, "required": ["city"]},
    },
}]

# 1) 模型决定调工具
r1 = client.chat.completions.create(model="qwen3.6-35b-a3b", messages=messages, tools=tools)
msg = r1.choices[0].message
messages.append(msg)

# 2) 执行工具
for call in (msg.tool_calls or []):
    if call.function.name == "get_weather":
        args = json.loads(call.function.arguments)
        result = get_weather(**args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

# 3) 模型基于结果生成自然语言回复
r2 = client.chat.completions.create(model="qwen3.6-35b-a3b", messages=messages)
print(r2.choices[0].message.content)
```

### 4.6 流式输出（SSE）

请求加 `"stream": true`，响应是 `text/event-stream`，逐 chunk 推送：

```
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}

data: {"choices":[{"delta":{"content":"你"}}]}

data: {"choices":[{"delta":{"content":"好"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Python 流式**

```python
stream = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "写一首关于秋天的诗"}],
    stream=True,
    max_tokens=256,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

**Node.js 流式**

```javascript
const stream = await client.chat.completions.create({
  model: "qwen3.6-35b-a3b",
  messages: [{ role: "user", content: "写一首关于秋天的诗" }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

**cURL 流式**

```bash
curl -N -X POST http://192.168.2.11:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","stream":true,"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

`-N` 关闭 curl 缓冲。

### 4.7 多模态（图文）

`content` 用数组同时传文字和图片：

```json
{
  "model": "qwen3.6-35b-a3b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text",      "text": "用一句话描述这张图"},
      {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
    ]
  }],
  "max_tokens": 200
}
```

图片支持三种来源：
- 公网 URL：`"url": "https://..."`
- Base64：`"url": "data:image/jpeg;base64,xxx..."`（推荐内网）
- 本地路径（vLLM 0.26+）：`"url": "file:///data/abc.png"`

**Python 多模态示例**

```python
import base64
from openai import OpenAI

with open("cat.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

client = OpenAI(base_url="http://192.168.2.11:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text",      "text": "描述这张图"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
    max_tokens=200,
)
print(resp.choices[0].message.content)
```

---

## 五、POST /v1/completions（Legacy）

纯文本续写，没有 chat 模板。一般用于 prompt 续写场景。

**请求**

```json
{
  "model": "qwen3.6-35b-a3b",
  "prompt": "Once upon a time,",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**响应**

```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1785412344,
  "model": "qwen3.6-35b-a3b",
  "choices": [{
    "index": 0,
    "text": " there was a ...",
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 96,
    "total_tokens": 100
  }
}
```

> 日常对话请用 `/v1/chat/completions`（§四）。

---

## 六、POST /v1/embeddings

**当前服务加载的是对话模型，不支持此端点。** 若需启用，把 `.env` 中 `MODEL_NAME` 改为 Embedding 模型（如 `BAAI/bge-m3`）后重启即可。

**请求（参考格式）**

```json
{
  "model": "BAAI/bge-m3",
  "input": ["hello world", "你好世界"]
}
```

**响应**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.012, -0.045, ...], "index": 0},
    {"object": "embedding", "embedding": [0.078, 0.023,  ...], "index": 1}
  ],
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

---

## 七、错误码

| HTTP | 含义 | 常见原因 | 处理 |
| --- | --- | --- | --- |
| 200 | 成功 | — | 正常解析 |
| 400 | 请求体错误 | JSON 解析失败、参数类型错 | 检查请求体 |
| 401 | 未授权 | API Key 缺失/错 | 修正 `Authorization: Bearer sk-xxx` |
| 404 | 模型不存在 | `model` 字段写错 | 调 `/v1/models` 查正确 ID |
| 422 | 字段校验失败 | 必填字段缺失 | 看响应体 `detail` |
| 429 | 请求过载 | 并发超限 | 客户端限流 + 退避重试 |
| 500 | 服务内部错误 | OOM、推理异常 | 看 logs |
| 503 | 服务未就绪 | 模型还在加载 | 等 1-3 分钟 |
| 504 | 推理超时 | prompt/output 过长 | 减小 `max_tokens` |

**错误响应体**

```json
{
  "object": "error",
  "message": "The model `xxx` does not exist.",
  "type": "InvalidRequestError",
  "code": 404
}
```

---

## 八、鉴权（启用 API Key 时）

在宿主机 `/opt/vllm/.env` 设置：

```bash
VLLM_API_KEY=sk-your-secret-key
```

重启服务：`docker compose restart vllm` 或 `cd /opt/vllm && ./scripts/stop.sh && ./scripts/start.sh`

调用时携带：

```bash
# cURL
curl -H "Authorization: Bearer sk-your-secret-key" ...

# Python
client = OpenAI(api_key="sk-your-secret-key", base_url="http://192.168.2.11:8000/v1")
```

未启用时，客户端 `api_key` 字段填任意非空字符串（如 `EMPTY`）。

---

## 九、限流与并发建议

- **每客户端并发 ≤ 8**（服务端 `MAX_NUM_SEQS=64`）
- **超时**：流式 60s，非流式 120s
- **重试**：429/503 用指数退避（1s → 2s → 4s，最多 3 次）
- **首字延迟优化**：用流式 + 精简 system prompt

---

## 十、FAQ

**Q1. 报 `The model 'xxx' does not exist`？**
调 `GET /v1/models` 拿真实 ID，客户端 `model` 字段必须严格一致。

**Q2. 报 `max sequence length is 32768, but you requested YYY`？**
说明 `prompt_tokens + max_tokens > MAX_MODEL_LEN`，请减少 prompt 或降低 `max_tokens`。（当前部署 `MAX_MODEL_LEN=256000`，按需调整即可。）

**Q3. 输出被 `finish_reason: length` 截断？**
说明达到 `max_tokens` 上限，增大 `max_tokens` 重试。

**Q4. 模型输出里有 `<think>...</think>` 占 token？**
当前部署默认已关闭思考模式（`enable_thinking=false`），不会输出思考内容。如需开启：在请求中传 `extra_body={"chat_template_kwargs": {"enable_thinking": true}}`，或服务启动时去掉 `--default-chat-template-kwargs '{"enable_thinking":false}'`。

**Q5. 怎么切换其他模型？**
改 `/opt/vllm/.env` 中的 `MODEL_NAME` 和 `SERVED_MODEL_NAME`，然后 `docker compose restart vllm`。

---

## 十一、服务端启动参数详解与调优

> 本节面向**运维 / 平台工程师**。客户端研发可跳过。
> vLLM 启动参数在 `/opt/vllm/.env` 中配置，重启服务生效：`cd /opt/vllm && docker compose restart vllm`

### 11.1 当前部署参数总表

| 参数 | 当前值 | 含义 | 调优建议 |
| --- | --- | --- | --- |
| `--model` | `MODEL_NAME` | 模型路径（本地或 HF repo ID） | 切换模型时改 `.env` 的 `MODEL_NAME` |
| `--served-model-name` | `qwen3.6-35b-a3b` | 客户端调用时填到 `model` 字段的名字 | 一般与实际模型同名，可自定义 |
| `--tensor-parallel-size` | `2` | 张量并行 GPU 数 | 等于物理 GPU 数量；不要超过 |
| `--gpu-memory-utilization` | `0.90` | 显存占用上限比例 | 0.85~0.95；OOM 时下调 |
| `--max-model-len` | `256000` | 最大上下文（输入+输出） | 不要超过模型原生支持（262144） |
| `--max-num-seqs` | `64` | 最大并发请求数 | 显存紧时下调；吞吐紧时上调 |
| `--max-num-batched-tokens` | `8192` | 单批 token 上限 | 短请求多→小；长请求多→大（16384） |
| `--block-size` | `16` | KV cache 分块大小 | 一般不动 |
| `--enable-prefix-caching` | `true` | 启用 prompt 前缀缓存 | 相同 system prompt 场景必开 |
| `--enable-chunked-prefill` | `true` | 启用分块 prefill | 长 prompt 场景必开 |
| `--enable-auto-tool-choice` | `true` | 自动选择/调用工具 | 用 function calling 时必开 |
| `--tool-call-parser` | `qwen3_coder` | 工具调用解析器 | Qwen 系列填 `qwen3_coder`；Llama 填 `llama3_json`；Hermes 填 `hermes` |
| `--reasoning-parser` | `qwen3` | 思考内容解析器（剥离 `<think>`） | 思考模型必开；非思考模型可关 |
| `--default-chat-template-kwargs` | `{"enable_thinking":false}` | 默认 chat 模板参数 | false=直接给答案；true=带思考过程 |
| `--enable-log-requests` | `true` | 记录每个请求的摘要日志 | 调试期开启；高 QPS 时可关 |

### 11.2 关键参数详细说明

#### `--max-model-len`（上下文长度）
- **作用**：限制单次请求输入+输出的 token 总数
- **当前**：256000（模型原生支持 262144，保留余量）
- **调优**：
  - 长文档 RAG 场景 → 调到模型原生上限（如 262144）
  - 短对话场景 → 调小到 32768，省显存给并发
  - 调小后 KV cache 占用下降，吞吐提升
- **报错**：`max sequence length is YYY, but you requested ZZZ` → 调大此值或减少 prompt

#### `--gpu-memory-utilization`（显存利用率）
- **作用**：vLLM 启动时按此比例预留显存给 KV cache
- **当前**：0.90
- **调优**：
  - **0.85**：保守，多卡通信压力大时用
  - **0.90**：推荐，省出 10% 给系统/驱动
  - **0.95**：激进，可能 OOM
- **告警**：vLLM 0.21+ 启用了 CUDA graph 内存预分配，0.90 实际相当于旧版的 0.883，需要 0.917 维持等效 KV 容量
- **OOM 时**：先降此值，再降 `max-num-seqs`

#### `--tensor-parallel-size`（张量并行度）
- **作用**：把模型权重切到 N 张 GPU 上推理
- **当前**：2（2×RTX 4090）
- **调优**：
  - 70B+ 模型必须 ≥ 2
  - 7B~14B 模型 1 即可（多卡反而降低效率，通信开销）
  - 不要超过物理 GPU 数量
- **影响**：TP 越大，通信开销越大，延迟略升

#### `--max-num-seqs`（最大并发）
- **作用**：同时处理的请求数上限
- **当前**：64
- **调优**：
  - 显存紧 → 降到 32
  - 客户端并发高 → 调到 128
  - 不建议超过 256（边际收益递减）
- **影响**：值越大，单请求平均延迟略升，但总吞吐提升

#### `--max-num-batched-tokens`（批处理 token 上限）
- **作用**：一次 decode 步处理的总 token 数
- **当前**：8192
- **调优**：
  - 短请求为主 → 4096（避免长 prompt 占用）
  - 长 prompt 为主 → 16384（提升 prefill 吞吐）
  - 极端场景 → 32768

#### `--enable-prefix-caching`（前缀缓存）
- **作用**：相同 system prompt / 上下文前缀只算一次
- **当前**：true
- **场景**：所有 system prompt 一致的 ChatBot、Docs QA
- **验证**：开启 `--enable-metrics --metrics-port 8001`，看 `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`

#### `--enable-chunked-prefill`（分块 prefill）
- **作用**：把长 prompt 拆成小块，与 decode 混合调度
- **当前**：true
- **场景**：长文档 RAG（>4K 上下文）
- **收益**：避免长 prompt 阻塞短请求

#### `--enable-auto-tool-choice` + `--tool-call-parser`（工具调用）
- **作用**：让 vLLM 自动识别模型输出的 `<tool_call>...</tool_call>` 块并解析为标准 JSON
- **当前**：true + `qwen3_coder`
- **调优**：
  - Qwen3 / Qwen2.5 → `qwen3_coder`
  - Qwen2 旧版 → `hermes`
  - Llama 3 → `llama3_json` 或 `llama4_pythonic`
  - 不使用 function calling → 关闭，节省解析开销
- **客户端配合**：请求体传 `tools` 数组 + `tool_choice: "auto"`

#### `--reasoning-parser`（思考内容解析）
- **作用**：把模型输出的 `<think>...</think>` 内容单独抽取到 `reasoning_content` 字段
- **当前**：`qwen3`
- **效果**：返回的 `choices[0].message` 会出现：
  ```json
  {
    "role": "assistant",
    "content": "正式答案",
    "reasoning_content": "思考过程...",
    "tool_calls": null
  }
  ```
- **适用**：思考模型（Qwen3 reasoning / DeepSeek-R1 / o1）
- **关闭**：非思考模型不指定即可

#### `--default-chat-template-kwargs`（chat 模板默认参数）
- **作用**：注入到 chat 模板的默认变量
- **当前**：`{"enable_thinking": false}`（默认关闭思考）
- **覆盖方式**：客户端单次请求用 `extra_body` 覆盖：
  ```json
  "extra_body": {"chat_template_kwargs": {"enable_thinking": true}}
  ```
- **关闭**：去掉此参数，模型走默认（通常 `enable_thinking=true`）

#### `--enable-log-requests`（请求日志）
- **作用**：每个请求的输入/输出/延迟/状态打印到日志
- **当前**：true
- **注意**：
  - 默认 INFO 级别只打印摘要（method、path、status、duration）
  - 想看完整 prompt/response：`VLLM_LOGGING_LEVEL=DEBUG`
  - 高 QPS 场景 → 关闭，避免日志爆炸
- **客户端排查**：拿到请求 ID `req-xxx` 后 `docker compose logs vllm | grep req-xxx`

#### `--kv-cache-dtype`（KV Cache 数据类型）⭐ 关键优化参数

- **作用**：控制 KV cache 中每个 token 的 key/value 用什么数据类型存储
- **显存占比**：KV cache 通常占 vLLM 总显存的 **40-60%**，改这个参数是**最有效的显存优化手段**
- **可选值**：

| 值 | 每个 token 大小 | 相对 fp16 | 精度损失 | 适用场景 |
| --- | --- | --- | --- | --- |
| `auto` | 跟随模型 dtype | — | 无 | 默认，一般不建议 |
| `fp16` | 4 bytes | 1.0x（基准） | 无 | 精度要求最高的场景 |
| `bf16` | 4 bytes | 1.0x | 极小 | 部分模型默认 |
| `fp8` | 2 bytes | **0.5x** | 极小（推荐） | **95% 场景首选** |
| `fp8_e4m3` | 1.5 bytes | **0.375x** | 小 | 显存紧张 / 高并发 |
| `fp8_e5m2` | 2 bytes | **0.5x** | 极小 | 与 fp8 近似，精度稍高 |

- **显存节省公式**：
  ```
  KV显存 = 2(每层) × num_layers × hidden_dim × bytes_per_element × max_tokens × batch_size
  
  以 27B 模型为例:
    40 层 × 2048 hidden × 2(K+V) × dtype × tokens
    fp16:     40 × 2048 × 2 × 2  bytes = 320 KB / 1K tokens / batch
    fp8:      40 × 2048 × 2 × 1  bytes = 160 KB / 1K tokens / batch  (省 50%)
    fp8_e4m3: 40 × 2048 × 2 × 0.75 bytes = 120 KB / 1K tokens / batch (省 62.5%)
  ```

- **选择建议**：

| 场景 | 推荐值 | 原因 |
| --- | --- | --- |
| 通用对话（精度优先） | `fp8` | 无损，省 50% KV 显存 |
| 长文档 RAG（>32K ctx） | `fp8_e4m3` | 极限压缩，省 62.5% KV |
| 高并发（>64 seq） | `fp8_e4m3` | 更多 KV 容量给并发 |
| 代码生成（精度敏感） | `fp8` 或 `fp16` | 代码对精度更敏感 |
| 医疗/法律（严谨场景） | `fp16` | 零精度损失 |
| 多模态（图文） | `fp8` | 视觉编码器占显存，KV 省出空间 |

- **精度影响实测**：
  - `fp8`：几乎感知不到差异，主流生产方案
  - `fp8_e4m3`：短回答无差异，长文本生成末尾可能出现微小偏差
  - `fp16` vs `bf16`：完全可互换

- **修改方法**：
  ```bash
  # 方案 A：改 .env
  KV_CACHE_DTYPE=fp8_e4m3
  docker compose up -d --force-recreate

  # 方案 B：用切换脚本（已预设）
  ./switch_model.sh qwen3.6-27b-fp8-e4m3
  ```

- **验证生效**：
  ```bash
  docker compose logs vllm 2>&1 | grep kv_cache_dtype
  # 输出: 'kv_cache_dtype': 'fp8_e4m3'
  ```

- **与 `--gpu-memory-utilization` 配合**：
  ```
  KV_CACHE_DTYPE=fp8_e4m3 + GPU_MEMORY_UTILIZATION=0.95
  → 比 fp16+0.90 多出约 20% KV 容量
  → 可支持 2x 并发 或 2x 上下文长度
  ```

### 11.3 调优决策树

```
性能不达标
├─ 显存 OOM？
│   ├─ 是 → gpu-memory-utilization 0.85 / 降 max-num-seqs / 开 kv-cache-dtype fp8
│   └─ 否 ↓
├─ TTFT 高（首字 > 500ms）？
│   ├─ 是 → 减 max-model-len / 开 prefix caching / 开 chunked prefill
│   └─ 否 ↓
├─ TPOT 高（每 token > 80ms）？
│   ├─ 是 → 降 max-num-seqs / 用量化 / 升 GPU
│   └─ 否 ↓
├─ 总吞吐低？
│   ├─ 是 → 加 max-num-seqs / 加 max-num-batched-tokens / 多实例 LB
│   └─ 否 ↓
└─ 客户端报错率高？
    ├─ 429 → 客户端限流 + 退避重试
    ├─ 500 → 看 logs 排查 OOM / 解析错
    └─ 503 → 等模型加载完
```

### 11.4 关键环境变量速查

| 变量 | 对应参数 | 默认 | 建议值 |
| --- | --- | --- | --- |
| `MODEL_NAME` | `--model` | （无） | 本地路径或 HF repo |
| `SERVED_MODEL_NAME` | `--served-model-name` | `default` | 与模型同名 |
| `TENSOR_PARALLEL_SIZE` | `--tensor-parallel-size` | `1` | 2 卡=2 / 1 卡=1 |
| `GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | `0.90` | 0.85~0.95 |
| `MAX_MODEL_LEN` | `--max-model-len` | `8192` | 模型原生上限 |
| `MAX_NUM_SEQS` | `--max-num-seqs` | `256` | 32~128 |
| `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | `8192` | 4096~16384 |
| `ENABLE_AUTO_TOOL_CHOICE` | `--enable-auto-tool-choice` | `true` | 调工具时 true |
| `TOOL_CALL_PARSER` | `--tool-call-parser` | `qwen3_coder` | 按模型系列选 |
| `REASONING_PARSER` | `--reasoning-parser` | `qwen3` | 思考模型才用 |
| `CHAT_TEMPLATE_KWARGS` | `--default-chat-template-kwargs` | `{"enable_thinking":false}` | 业务定 |
| `ENABLE_LOG_REQUESTS` | `--enable-log-requests` | `true` | 调试 true / 生产可选 false |
