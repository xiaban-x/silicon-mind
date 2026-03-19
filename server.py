"""
MLX LLM Inference Server — OpenAI-compatible API
=================================================
Features:
  - Phase 1: Streaming SSE (Server-Sent Events)
  - Phase 2: Concurrency control (asyncio.Semaphore + queue)
  - Phase 3: KV Cache with LRU eviction (conversation-level)
  - Phase 4: Full OpenAI SDK compatibility

Usage:
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "../llm/models/qwen32b"

# 并发控制：同时推理请求数上限（M4 Pro 单 GPU，设为 1 最稳）
MAX_CONCURRENT_INFERENCE = 1

# 请求排队超时（秒）
QUEUE_TIMEOUT = 120.0

# KV Cache：最大缓存对话数
MAX_CACHE_ENTRIES = 8

# KV Cache：单条缓存的最大 token 数（防止超长对话撑爆内存）
MAX_CACHE_TOKENS = 8192

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

model: Optional["nn.Module"] = None
tokenizer: Optional["TokenizerWrapper"] = None
inference_semaphore: "asyncio.Semaphore | None" = None

# LRU KV Cache: conversation_id -> CacheEntry
kv_cache_store: "LRUKVCache | None" = None


# ---------------------------------------------------------------------------
# KV Cache Manager (LRU)
# ---------------------------------------------------------------------------


class CacheEntry:
    """单条 KV cache 记录"""

    prompt_tokens: list[int]
    cache: list[Any]
    last_used: float

    def __init__(self, prompt_tokens: list[int], cache: list[Any]):
        self.prompt_tokens = prompt_tokens
        self.cache = cache
        self.last_used = time.time()


class LRUKVCache:
    """基于 LRU 策略的会话级 KV Cache 管理器"""

    max_entries: int
    _store: OrderedDict[str, CacheEntry]

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES):
        self.max_entries = max_entries
        self._store = OrderedDict()

    def get(self, conversation_id: str) -> CacheEntry | None:
        """获取缓存并移到末尾（最近使用）"""
        if conversation_id in self._store:
            self._store.move_to_end(conversation_id)
            entry = self._store[conversation_id]
            entry.last_used = time.time()
            return entry
        return None

    def put(self, conversation_id: str, prompt_tokens: list[int], cache: list[Any]):
        """存入缓存，超出容量时淘汰最久未使用的"""
        if conversation_id in self._store:
            self._store.move_to_end(conversation_id)
        self._store[conversation_id] = CacheEntry(prompt_tokens, cache)
        while len(self._store) > self.max_entries:
            _, evicted = self._store.popitem(last=False)
            # 释放 MLX 内存
            del evicted.cache

    def delete(self, conversation_id: str):
        """主动删除某条缓存"""
        if conversation_id in self._store:
            entry = self._store.pop(conversation_id)
            del entry.cache

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Pydantic Models — OpenAI-compatible schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen32b"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    # 扩展字段：传入 conversation_id 可复用 KV cache
    conversation_id: str | None = None
    # 控制是否使用 KV cache（默认开启）
    use_cache: bool = True


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionUsage | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Lifespan: load model on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, tokenizer, inference_semaphore, kv_cache_store

    print(f"🚀 Loading model from {MODEL_PATH} ...")
    loaded = load(MODEL_PATH)
    model, tokenizer = loaded[0], loaded[1]
    print("✅ Model loaded.")

    inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
    kv_cache_store = LRUKVCache(max_entries=MAX_CACHE_ENTRIES)

    yield

    # cleanup
    del model, tokenizer
    print("🛑 Server shutdown, model unloaded.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="MLX LLM Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_prompt(messages: list[ChatMessage]) -> str:
    """
    利用 tokenizer 的 chat_template 构建 prompt。
    如果 tokenizer 没有 chat_template，则手动拼接。
    """
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

    if tokenizer is not None and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(  # type: ignore[reportCallIssue, reportOptionalMemberAccess]
            msg_dicts, add_generation_prompt=True, tokenize=False)

    # fallback: 手动拼接（Qwen 格式）
    prompt = ""
    for m in msg_dicts:
        role = m["role"]
        content = m["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def find_common_prefix_length(a: list[int], b: list[int]) -> int:
    """找到两个 token 序列的公共前缀长度"""
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


# ---------------------------------------------------------------------------
# Phase 3: KV Cache — 准备 prompt_cache
# ---------------------------------------------------------------------------


class CachePrepareResult:
    """prepare_cache_for_request 的返回结果"""

    prompt_cache: list[Any] | None
    prompt_for_generate: Any  # str 或 mx.array，传给 stream_generate 的 prompt
    full_prompt_tokens: list[int]  # 完整 prompt 的 token ids（用于存回）
    cache_hit: bool

    def __init__(
        self,
        prompt_cache: list[Any] | None,
        prompt_for_generate: Any,
        full_prompt_tokens: list[int],
        cache_hit: bool,
    ):
        self.prompt_cache = prompt_cache
        self.prompt_for_generate = prompt_for_generate
        self.full_prompt_tokens = full_prompt_tokens
        self.cache_hit = cache_hit


def prepare_cache_for_request(
    prompt_text: str,
    conversation_id: str | None,
    use_cache: bool,
) -> CachePrepareResult:
    """
    根据 conversation_id 查找可复用的 KV cache。

    核心逻辑：
    - generate_step 不会自动跳过已缓存的 token
    - 它假设传入的 prompt 是 cache 之后的「增量」
    - 所以我们必须只传增量部分给 stream_generate

    返回 CachePrepareResult:
        - prompt_cache: cache 对象（可能为 None）
        - prompt_for_generate: 要传给 stream_generate 的 prompt（可能是截断后的 token 数组）
        - full_prompt_tokens: 完整 prompt 的 token ids（存回用）
        - cache_hit: 是否命中了已有缓存
    """
    # tokenizer 在 lifespan 阶段已初始化，请求处理时必定非 None
    assert tokenizer is not None
    # model 和 kv_cache_store 同理
    assert model is not None
    assert kv_cache_store is not None

    full_prompt_tokens = tokenizer.encode(prompt_text)  # type: ignore[reportCallIssue]

    if not use_cache or not conversation_id:
        return CachePrepareResult(
            prompt_cache=None,
            prompt_for_generate=prompt_text,
            full_prompt_tokens=full_prompt_tokens,
            cache_hit=False,
        )

    entry = kv_cache_store.get(conversation_id)

    if entry is None:
        # 首次请求：预创建空 cache 对象，stream_generate 会原地填充它
        cache = make_prompt_cache(model)
        print("  [KV Cache] MISS (first request) conversation={}, prompt_tokens={}".format(
            conversation_id, len(full_prompt_tokens)
        ))
        return CachePrepareResult(
            prompt_cache=cache,
            prompt_for_generate=prompt_text,
            full_prompt_tokens=full_prompt_tokens,
            cache_hit=False,
        )

    # 有已有缓存 → 尝试复用
    # entry.prompt_tokens 是上一轮的 prompt token 序列（不含生成内容）
    # 但 cache 对象包含了 prompt + 生成内容的完整 KV
    prefix_len = find_common_prefix_length(entry.prompt_tokens, full_prompt_tokens)

    if prefix_len == 0:
        # 没有公共前缀，缓存没用，创建新的
        kv_cache_store.delete(conversation_id)
        cache = make_prompt_cache(model)
        return CachePrepareResult(
            prompt_cache=cache,
            prompt_for_generate=prompt_text,
            full_prompt_tokens=full_prompt_tokens,
            cache_hit=False,
        )

    # 有公共前缀 → 复用 cache
    cache = entry.cache
    try:
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        # cache 对象包含了上一轮 prompt + 生成内容的完整 KV 状态
        # 我们需要 trim 到 prefix_len（只保留当前轮与上一轮共享的前缀 KV）
        # 这样 generate_step 只需处理增量 tokens
        if can_trim_prompt_cache(cache):
            trim_prompt_cache(cache, prefix_len)
    except (ImportError, AttributeError):
        pass

    # 只取增量部分的 token 传给 generate
    # generate_step 会将这些 token 作为 cache 之后的新输入处理
    incremental_tokens = full_prompt_tokens[prefix_len:]

    print("  [KV Cache] HIT conversation={}, reusing {}/{} tokens, incremental={} tokens".format(
        conversation_id, prefix_len, len(full_prompt_tokens), len(incremental_tokens)
    ))

    return CachePrepareResult(
        prompt_cache=cache,
        prompt_for_generate=mx.array(incremental_tokens),
        full_prompt_tokens=full_prompt_tokens,
        cache_hit=True,
    )


# ---------------------------------------------------------------------------
# Phase 1: Streaming SSE Generator
# ---------------------------------------------------------------------------


async def stream_chat_sse(
    req: ChatRequest,
    prompt_text: str,
    completion_id: str,
) -> AsyncGenerator[str, None]:
    """
    SSE 流式生成器。
    每产出一个 token 就发送一个 SSE event，格式完全兼容 OpenAI。
    """
    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)

    # Phase 3: 准备 KV cache
    cache_result = prepare_cache_for_request(
        prompt_text, req.conversation_id, req.use_cache
    )

    prompt_tokens_count = 0
    completion_tokens_count = 0
    full_response = ""

    # 首个 chunk: role
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    # 用 queue 桥接同步 stream_generate 和异步 SSE 生成器
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    generation_done = asyncio.Event()

    def _run_generate():
        nonlocal prompt_tokens_count
        try:
            kwargs: dict[str, Any] = {
                "max_tokens": req.max_tokens,
                "sampler": sampler,
            }
            if cache_result.prompt_cache is not None:
                kwargs["prompt_cache"] = cache_result.prompt_cache

            for response in stream_generate(
                model,  # type: ignore[reportArgumentType]
                tokenizer,  # type: ignore[reportArgumentType]
                prompt=cache_result.prompt_for_generate,
                **kwargs,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, response)
                prompt_tokens_count = response.prompt_tokens
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(generation_done.set)  # type: ignore[operator]

    # 启动生成线程
    loop.run_in_executor(None, _run_generate)  # type: ignore[func-returns-none]

    while True:
        # 等待下一个 token 或生成结束
        try:
            response: Any = await asyncio.wait_for(queue.get(), timeout=QUEUE_TIMEOUT)
        except asyncio.TimeoutError:
            break

        if isinstance(response, Exception):
            error_chunk: dict[str, Any] = {
                "error": {
                    "message": str(response),
                    "type": "server_error",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            break

        # 正常 token
        completion_tokens_count: int = response.generation_tokens  # type: ignore[reportUnknownMemberType]
        full_response += response.text  # type: ignore[reportUnknownMemberType]

        chunk: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": response.text},  # type: ignore[reportUnknownMemberType]
                    "finish_reason": response.finish_reason,  # type: ignore[reportUnknownMemberType]
                }
            ],
        }

        # 最后一个 chunk 带 usage
        if response.finish_reason is not None:  # type: ignore[reportUnknownMemberType]
            # 使用完整 prompt tokens 数（stream_generate 报告的可能只是增量部分）
            actual_prompt_tokens = len(cache_result.full_prompt_tokens)
            chunk["usage"] = {
                "prompt_tokens": actual_prompt_tokens,
                "completion_tokens": completion_tokens_count,
                "total_tokens": actual_prompt_tokens + completion_tokens_count,
            }

        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        if response.finish_reason is not None:
            break

    # 等待生成线程彻底结束
    await generation_done.wait()  # type: ignore[func-returns-none]

    # Phase 3: 存回 KV cache
    if req.use_cache and req.conversation_id and cache_result.prompt_cache is not None:
        # 注意：cache 对象包含了 prompt + 生成内容的完整 KV 状态
        # 但我们只存 prompt 部分的 token ids 用于下轮匹配
        # 因为下轮对话的 prompt 会通过 chat_template 重新构建，
        # 其中上一轮的 assistant 回复会带有 <|im_start|>assistant 等标记，
        # 和裸生成的 text tokens 不一致，所以只匹配 prompt 前缀
        if len(cache_result.full_prompt_tokens) <= MAX_CACHE_TOKENS:
            kv_cache_store.put(  # type: ignore[reportOptionalMemberAccess]
                req.conversation_id,
                cache_result.full_prompt_tokens,
                cache_result.prompt_cache,
            )

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Phase 1 + 4: Non-streaming response
# ---------------------------------------------------------------------------


async def generate_chat_response(
    req: ChatRequest,
    prompt_text: str,
    completion_id: str,
) -> ChatCompletionResponse:
    """非流式：收集所有 token 后一次性返回"""
    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)

    # Phase 3: 准备 KV cache
    cache_result = prepare_cache_for_request(
        prompt_text, req.conversation_id, req.use_cache
    )

    loop = asyncio.get_event_loop()

    def _sync_generate():
        kwargs: dict[str, Any] = {
            "max_tokens": req.max_tokens,
            "sampler": sampler,
        }
        if cache_result.prompt_cache is not None:
            kwargs["prompt_cache"] = cache_result.prompt_cache

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for response in stream_generate(
            model,  # type: ignore[reportArgumentType]
            tokenizer,  # type: ignore[reportArgumentType]
            prompt=cache_result.prompt_for_generate,
            **kwargs,
        ):
            full_text += response.text  # type: ignore[reportUnknownMemberType]
            prompt_tokens = response.prompt_tokens  # type: ignore[reportUnknownMemberType]
            completion_tokens = response.generation_tokens  # type: ignore[reportUnknownMemberType]

        return full_text, prompt_tokens, completion_tokens

    full_text, _reported_prompt_tokens, completion_tokens = await loop.run_in_executor(
        None, _sync_generate
    )

    # Phase 3: 存回 KV cache
    if req.use_cache and req.conversation_id and cache_result.prompt_cache is not None:
        # 只存 prompt tokens，不含生成内容（理由见 streaming 部分注释）
        if len(cache_result.full_prompt_tokens) <= MAX_CACHE_TOKENS:
            kv_cache_store.put(  # type: ignore[reportOptionalMemberAccess]
                req.conversation_id,
                cache_result.full_prompt_tokens,
                cache_result.prompt_cache,
            )

    # 当 cache 命中时，stream_generate 报告的 prompt_tokens 只是增量部分
    # 需要修正为完整 prompt 的 token 数
    actual_prompt_tokens = len(cache_result.full_prompt_tokens)

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=full_text),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=actual_prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=actual_prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Phase 2: 通过 semaphore 控制并发，超出则排队等待。
    """
    completion_id = make_completion_id()
    prompt_text = build_prompt(req.messages)

    # inference_semaphore 在 lifespan 阶段已初始化，必定非 None
    assert inference_semaphore is not None

    # Phase 2: 并发控制 — 获取推理锁（带超时）
    try:
        await asyncio.wait_for(
            inference_semaphore.acquire(),  # type: ignore[func-returns-none]
            timeout=QUEUE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "Server is busy, please try again later.",
                    "type": "server_overloaded",
                    "code": "server_overloaded",
                }
            },
        )

    try:
        if req.stream:
            # Phase 1: Streaming SSE
            async def _stream_with_release():
                try:
                    async for chunk in stream_chat_sse(
                        req, prompt_text, completion_id
                    ):
                        yield chunk
                finally:
                    inference_semaphore.release()  # type: ignore[func-returns-none, reportOptionalMemberAccess]

            return StreamingResponse(
                _stream_with_release(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # nginx 兼容
                },
            )
        else:
            # 非流式
            try:
                result = await generate_chat_response(
                    req, prompt_text, completion_id
                )
                return result
            finally:
                inference_semaphore.release()  # type: ignore[func-returns-none, reportOptionalMemberAccess]
    except Exception:
        inference_semaphore.release()  # type: ignore[func-returns-none, reportOptionalMemberAccess]
        raise


# Phase 4: /v1/models endpoint
@app.get("/v1/models")
async def list_models():
    """List available models — OpenAI SDK 需要这个端点"""
    return ModelListResponse(
        data=[
            ModelInfo(
                id="qwen32b",
                created=int(time.time()),
            )
        ]
    )


# Health check
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "cache_entries": kv_cache_store.size if kv_cache_store else 0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
