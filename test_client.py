"""
测试客户端 — 验证所有 Phase 功能
用法: python test_client.py
前置条件: 服务端已启动 (uvicorn server:app --host 0.0.0.0 --port 8000)
"""

import json
import time

# ======================================================================
# Test 1: 使用 OpenAI SDK 直连（Phase 4）
# ======================================================================


def test_openai_sdk():
    """Phase 4: 验证 OpenAI SDK 可以直连本地服务"""
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  跳过 OpenAI SDK 测试 (pip install openai)")
        return

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # 本地不需要 key，但 SDK 要求传
    )

    print("=" * 60)
    print("🧪 Test 1: OpenAI SDK — 非流式请求")
    print("=" * 60)

    start = time.time()
    response = client.chat.completions.create(
        model="qwen32b",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "用一句话解释什么是 KV Cache"},
        ],
        max_tokens=256,
        temperature=0.7,
    )
    elapsed = time.time() - start

    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")
    print(f"耗时: {elapsed:.2f}s")
    print()


# ======================================================================
# Test 2: Streaming SSE + OpenAI SDK（Phase 1 + 4）
# ======================================================================


def test_streaming():
    """Phase 1: 验证 SSE 流式输出"""
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  跳过流式测试 (pip install openai)")
        return

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    print("=" * 60)
    print("🧪 Test 2: OpenAI SDK — 流式请求 (SSE)")
    print("=" * 60)

    start = time.time()
    stream = client.chat.completions.create(
        model="qwen32b",
        messages=[
            {"role": "user", "content": "写一首关于 MacBook 的五言绝句"},
        ],
        max_tokens=256,
        stream=True,
    )

    full_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_text += text
            print(text, end="", flush=True)

    elapsed = time.time() - start
    print(f"\n\n完整输出: {full_text}")
    print(f"耗时: {elapsed:.2f}s")
    print()


# ======================================================================
# Test 3: KV Cache 加速（Phase 3）
# ======================================================================


def test_kv_cache():
    """Phase 3: 验证 KV cache 多轮对话加速

    测试策略：用一段很长的 system prompt 来让 prefill 时间占主导，
    这样 cache 命中时跳过的 prefill 量足够大，加速效果才明显。
    """
    try:
        import requests
    except ImportError:
        print("⚠️  跳过 KV Cache 测试 (pip install requests)")
        return

    print("=" * 60)
    print("🧪 Test 3: KV Cache — 多轮对话加速")
    print("=" * 60)

    conversation_id = "test-conv-001"
    url = "http://localhost:8000/v1/chat/completions"

    # 用一段长 system prompt，让 prefill 时间成为主要开销
    long_system = (
        "你是一个资深 Python 专家和技术导师。你精通 Python 的所有核心概念，"
        "包括但不限于：装饰器、生成器、协程、元类、描述符协议、上下文管理器、"
        "类型注解、异步编程、GIL、内存管理、垃圾回收、C 扩展开发。"
        "你的回答风格是：先简要概括核心概念，再给出可运行的代码示例，"
        "最后总结最佳实践和常见陷阱。你特别擅长用类比来解释复杂概念。\n\n"
        "以下是你需要遵循的回答规范：\n"
        "1. 代码示例必须包含完整的导入语句\n"
        "2. 用中文回答，代码注释也用中文\n"
        "3. 如果概念有多种实现方式，至少展示两种\n"
        "4. 对于进阶话题，要提及 CPython 实现细节\n"
        "5. 始终提供性能考量和适用场景\n"
        "6. 使用 Python 3.10+ 的现代语法\n"
        "7. 提及相关的 PEP 编号\n"
        "8. 避免过度简化，但也要让初学者能理解\n"
        "9. 如果问题涉及标准库，要提及替代的第三方库\n"
        "10. 在适当的时候引用 Python 之禅的相关原则\n\n"
        "以下是你的知识库摘要，用于提供更准确的回答：\n"
        "- Python 装饰器本质是高阶函数，接受函数并返回函数\n"
        "- functools.wraps 保留被装饰函数的元信息\n"
        "- 类装饰器通过 __call__ 方法实现\n"
        "- 装饰器可以带参数，需要三层嵌套\n"
        "- Python 3.9+ 支持 @cache 装饰器简化缓存\n"
        "- 装饰器执行顺序是从下到上（最靠近函数的先执行）\n"
        "- property、staticmethod、classmethod 是内置装饰器\n"
        "- dataclasses.dataclass 是 Python 3.7+ 的重要装饰器\n"
        "- typing.overload 用于类型注解的函数重载\n"
        "- contextlib.contextmanager 将生成器转为上下文管理器\n"
    )

    # 第 1 轮
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": "什么是装饰器?"},
    ]

    print("\n--- 第 1 轮（冷启动，无 cache）---")
    start = time.time()
    resp = requests.post(
        url,
        json={
            "model": "qwen32b",
            "messages": messages,
            "max_tokens": 64,
            "conversation_id": conversation_id,
            "use_cache": True,
        },
    )
    elapsed1 = time.time() - start
    data = resp.json()
    assistant_reply = data["choices"][0]["message"]["content"]
    usage1 = data.get("usage", {})
    print(f"回复: {assistant_reply[:100]}...")
    print(f"Usage: prompt_tokens={usage1.get('prompt_tokens')}, completion_tokens={usage1.get('completion_tokens')}")
    print(f"耗时: {elapsed1:.2f}s")

    # 第 2 轮（追加上一轮回复 + 新问题）
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "能给个具体例子吗?"})

    print("\n--- 第 2 轮（有 cache，应更快）---")
    start = time.time()
    resp = requests.post(
        url,
        json={
            "model": "qwen32b",
            "messages": messages,
            "max_tokens": 64,
            "conversation_id": conversation_id,
            "use_cache": True,
        },
    )
    elapsed2 = time.time() - start
    data = resp.json()
    usage2 = data.get("usage", {})
    print(f"回复: {data['choices'][0]['message']['content'][:100]}...")
    print(f"Usage: prompt_tokens={usage2.get('prompt_tokens')}, completion_tokens={usage2.get('completion_tokens')}")
    print(f"耗时: {elapsed2:.2f}s")

    # 第 3 轮（继续追加，cache 前缀应该更长）
    messages.append({"role": "assistant", "content": data['choices'][0]['message']['content']})
    messages.append({"role": "user", "content": "那类装饰器呢？和函数装饰器有什么区别?"})

    print("\n--- 第 3 轮（cache 应覆盖更多前缀）---")
    start = time.time()
    resp = requests.post(
        url,
        json={
            "model": "qwen32b",
            "messages": messages,
            "max_tokens": 64,
            "conversation_id": conversation_id,
            "use_cache": True,
        },
    )
    elapsed3 = time.time() - start
    data = resp.json()
    usage3 = data.get("usage", {})
    print(f"回复: {data['choices'][0]['message']['content'][:100]}...")
    print(f"Usage: prompt_tokens={usage3.get('prompt_tokens')}, completion_tokens={usage3.get('completion_tokens')}")
    print(f"耗时: {elapsed3:.2f}s")

    print(f"\n--- 耗时对比 ---")
    print(f"第 1 轮: {elapsed1:.2f}s (冷启动)")
    print(f"第 2 轮: {elapsed2:.2f}s")
    print(f"第 3 轮: {elapsed3:.2f}s")

    if elapsed2 < elapsed1:
        print(f"🚀 第2轮 vs 第1轮 加速: {elapsed1 / elapsed2:.1f}x")
    else:
        print("⚠️  第 2 轮未加速")

    if elapsed3 < elapsed1:
        print(f"🚀 第3轮 vs 第1轮 加速: {elapsed1 / elapsed3:.1f}x")

    print("（请同时查看服务端日志中的 [KV Cache] HIT 信息确认 cache 命中）")
    print()


# ======================================================================
# Test 4: 并发控制（Phase 2）
# ======================================================================


def test_concurrency():
    """Phase 2: 验证并发请求排队"""
    import concurrent.futures

    try:
        import requests
    except ImportError:
        print("⚠️  跳过并发测试 (pip install requests)")
        return

    print("=" * 60)
    print("🧪 Test 4: 并发控制 — 2 个请求同时发送")
    print("=" * 60)

    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "qwen32b",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 32,
    }

    def send_request(idx):
        start = time.time()
        resp = requests.post(url, json=payload)
        elapsed = time.time() - start
        return idx, elapsed, resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(send_request, i) for i in range(2)]
        for f in concurrent.futures.as_completed(futures):
            idx, elapsed, status = f.result()
            print(f"  请求 #{idx}: status={status}, 耗时={elapsed:.2f}s")

    print("（第 2 个请求应排队等待第 1 个完成后才开始推理）")
    print()


# ======================================================================
# Test 5: Health check + Model list
# ======================================================================


def test_endpoints():
    """基础端点测试"""
    try:
        import requests
    except ImportError:
        print("⚠️  跳过端点测试 (pip install requests)")
        return

    print("=" * 60)
    print("🧪 Test 5: 基础端点")
    print("=" * 60)

    resp = requests.get("http://localhost:8000/health")
    print(f"  /health: {resp.json()}")

    resp = requests.get("http://localhost:8000/v1/models")
    print(f"  /v1/models: {resp.json()}")
    print()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("\n🔧 MLX LLM Server — 功能测试\n")

    test_endpoints()
    test_openai_sdk()
    test_streaming()
    test_kv_cache()
    test_concurrency()

    print("✅ 全部测试完成！")
