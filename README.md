# MLX LLM Inference Server

M4 Pro 本地部署 Qwen 32B，OpenAI 兼容 API 服务。

## 功能

| Feature | 说明 |
|---------|------|
| **Streaming SSE** | `stream=true` 逐 token 流式输出，完全兼容 OpenAI SSE 格式 |
| **并发控制** | `asyncio.Semaphore` 限制同时推理数，超时自动 503 |
| **KV Cache** | 按 `conversation_id` 缓存 KV 状态，多轮对话跳过重复计算 |
| **OpenAI SDK 直连** | 可直接用 `openai` Python SDK 连接 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install openai requests  # 测试用

# 2. 启动服务
python server.py
# 或
uvicorn server:app --host 0.0.0.0 --port 8000

# 3. 运行测试
python test_client.py
```

## 使用 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# 非流式
response = client.chat.completions.create(
    model="qwen32b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="qwen32b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## KV Cache 使用

传入 `conversation_id` 即可自动缓存和复用 KV 状态：

```python
import requests

# 第 1 轮
resp = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "qwen32b",
    "messages": [{"role": "user", "content": "什么是 Python?"}],
    "conversation_id": "my-conv-001",
})

# 第 2 轮（自动复用上一轮的 KV cache，prompt 处理速度大幅提升）
resp = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "qwen32b",
    "messages": [
        {"role": "user", "content": "什么是 Python?"},
        {"role": "assistant", "content": "...上一轮的回复..."},
        {"role": "user", "content": "能详细说说吗?"},
    ],
    "conversation_id": "my-conv-001",
})
```

## 配置项

在 `server.py` 顶部修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `./models/qwen32b` | 模型路径 |
| `MAX_CONCURRENT_INFERENCE` | `1` | 最大并发推理数 |
| `QUEUE_TIMEOUT` | `120.0` | 请求排队超时(秒) |
| `MAX_CACHE_ENTRIES` | `8` | KV Cache 最大缓存对话数 |
| `MAX_CACHE_TOKENS` | `4096` | 单条缓存最大 token 数 |

## API 端点

| Method | Path | 说明 |
|--------|------|------|
| POST | `/v1/chat/completions` | Chat Completions（兼容 OpenAI） |
| GET | `/v1/models` | 模型列表 |
| GET | `/health` | 健康检查 |
