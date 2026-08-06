# API 调用示例

vLLM 启动后默认监听 `:8000`，对外暴露 **OpenAI 兼容 API**，路径：
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`（仅 Embedding 模型）
- `GET  /v1/models`
- `POST /v1/responses`（部分新版本支持）

> 下面的示例默认 `SERVED_MODEL_NAME=qwen2.5-7b`，可在 `.env` 中改。

---

## 1. cURL 基础

```bash
# 列出可用模型
curl http://localhost:8000/v1/models

# 一次性聊天
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [
      {"role":"system","content":"你是一个简洁的助手"},
      {"role":"user","content":"用 50 字介绍 vLLM"}
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

---

## 2. Python（OpenAI 官方 SDK）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # 未设 VLLM_API_KEY 时填任意
)

resp = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[
        {"role": "system", "content": "你是一个资深 SRE"},
        {"role": "user", "content": "K8s pod OOM 怎么排查？"},
    ],
    temperature=0.6,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

---

## 3. 流式输出（SSE）

```python
stream = client.chat.completions.create(
    model="qwen2.5-7b",
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

cURL 流式：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","stream":true,"messages":[{"role":"user","content":"hello"}]}' \
  --no-buffer
```

---

## 4. 多轮对话

```python
messages = [
    {"role": "system", "content": "你是 Linux 助手"},
    {"role": "user", "content": "如何查看 CPU 信息？"},
]
r1 = client.chat.completions.create(model="qwen2.5-7b", messages=messages)
messages.append({"role": "assistant", "content": r1.choices[0].message.content})
messages.append({"role": "user", "content": "那内存呢？"})
r2 = client.chat.completions.create(model="qwen2.5-7b", messages=messages)
print(r2.choices[0].message.content)
```

> vLLM 启用 prefix caching 后，**相同 system prompt 只算一次**。

---

## 5. 函数调用 / Tool Use

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }
}]

resp = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto",
)
print(resp.choices[0].message.tool_calls)
# [ChatCompletionMessageToolCall(id='...', function=Function(name='get_weather', arguments='{"city":"北京"}'), ...)]
```

---

## 6. 文本补全（Completions 端点）

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

---

## 7. Embeddings（仅 Embedding 模型）

```python
resp = client.embeddings.create(
    model="BAAI/bge-m3",   # 加载 Embedding 模型
    input=["hello world", "你好世界"]
)
print(len(resp.data[0].embedding), resp.data[0].embedding[:5])
```

---

## 8. 并发 / 批量

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def ask(q):
    r = await client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": q}],
        max_tokens=128,
    )
    return r.choices[0].message.content

async def main():
    res = await asyncio.gather(*[ask(f"一句话回答：什么是 {t}？")
                                  for t in ["TCP", "UDP", "HTTP", "QUIC"]])
    for r in res: print(r)

asyncio.run(main())
```

---

## 9. LangChain / LlamaIndex

```python
# LangChain
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="qwen2.5-7b",
    temperature=0.7,
)
print(llm.invoke("用 20 字介绍 RAG").content)
```

```python
# LlamaIndex
from llama_index.llms.openai import OpenAI
llm = OpenAI(
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    model="qwen2.5-7b",
)
print(llm.complete("用 20 字介绍 RAG").text)
```

---

## 10. 鉴权（开启 API Key）

`.env` 中设 `VLLM_API_KEY=sk-your-secret`，重启后：

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-secret",
)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-secret" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## 11. 常见错误码

| HTTP | 含义 | 处理 |
| --- | --- | --- |
| 400 | 请求体错误（参数非法） | 检查 `model`、`messages` 字段 |
| 401 | 未授权 | 检查 API Key |
| 404 | 模型不存在 | 确认 `SERVED_MODEL_NAME` |
| 422 | Pydantic 校验失败 | 看返回 JSON 中 `detail` |
| 503 | 服务尚未就绪 | 等启动完成，或查 logs |
| 504 | 生成超时 | 调大 `timeout` 或减小 `max_tokens` |
