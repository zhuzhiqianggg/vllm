#!/usr/bin/env python3
"""
vLLM API 端到端测试 - Qwen3.5-35B-A3B-FP8
覆盖：纯文本/多模态/流式/并发
"""
import sys
import time
import json
import base64
import asyncio
import aiohttp
from urllib.request import urlopen
from urllib.error import URLError

HOST = "http://localhost:8000"
MODEL = "qwen3.5-35b-a3b"  # 与 .env 中 SERVED_MODEL_NAME 一致
API_KEY = ""  # 如设置了 VLLM_API_KEY 则填入

def check_health():
    print("=" * 60)
    print(f"[1] 健康检查: GET {HOST}/v1/models")
    print("=" * 60)
    try:
        with urlopen(f"{HOST}/v1/models", timeout=10) as r:
            data = json.load(r)
            print(f"  Status: OK")
            for m in data.get("data", []):
                print(f"  Model: {m.get('id')}")
            return True
    except URLError as e:
        print(f"  Status: FAIL - {e}")
        return False

def test_chat():
    print("=" * 60)
    print("[2] 纯文本对话测试")
    print("=" * 60)
    import urllib.request
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个简洁的技术助手，回答控制在 50 字以内。"},
            {"role": "user", "content": "用一句话介绍 vLLM 的核心优势。"}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.time()
    try:
        with urlopen(req, timeout=120) as r:
            data = json.load(r)
        dt = time.time() - t0
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})
        print(f"  Latency: {dt:.2f}s")
        print(f"  Tokens:  in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} total={usage.get('total_tokens')}")
        print(f"  Reply:   {content}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if hasattr(e, "read"):
            print("  body:", e.read().decode("utf-8", errors="ignore"))
        return False

def test_stream():
    print("=" * 60)
    print("[3] 流式输出测试")
    print("=" * 60)
    import urllib.request
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "用中文写一首五言绝句，主题是秋月。"}],
        "max_tokens": 1024,
        "temperature": 0.8,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.time()
    tokens = 0
    try:
        with urlopen(req, timeout=120) as r:
            full = []
            for line in r:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        full.append(delta)
                        print(delta, end="", flush=True)
                        tokens += 1
                except Exception:
                    pass
        dt = time.time() - t0
        print()
        print(f"  Latency: {dt:.2f}s, chunks: {tokens}, tps: {tokens/dt:.1f}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

def test_vision_local_image():
    """用本地生成的测试图片（红底白色 'Q' 字符）做视觉理解测试"""
    print("=" * 60)
    print("[4] 多模态视觉理解测试（本地图片）")
    print("=" * 60)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  Pillow 未安装，跳过视觉测试")
        return True

    import io
    import urllib.request
    img = Image.new("RGB", (640, 360), (220, 38, 38))  # 红色背景
    d = ImageDraw.Draw(img)
    d.rectangle([(40, 40), (600, 320)], outline=(255, 255, 255), width=8)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 220)
    except Exception:
        font = ImageFont.load_default()
    d.text((180, 60), "Qwen3.5", fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用 30 字以内简要描述这张图：背景颜色是什么？图中有哪些文字？"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.time()
    try:
        with urlopen(req, timeout=180) as r:
            data = json.load(r)
        dt = time.time() - t0
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(f"  Latency: {dt:.2f}s")
        print(f"  Tokens:  in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}")
        print(f"  Reply:   {content}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if hasattr(e, "read"):
            print("  body:", e.read().decode("utf-8", errors="ignore"))
        return False

async def bench_session(conc, n):
    """简单并发吞吐测试"""
    sem = asyncio.Semaphore(conc)
    stats = {"ok": 0, "fail": 0, "in_tok": 0, "out_tok": 0, "lat": []}
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    prompt = "请用 50 字以内解释 GPU 张量并行的原理。" * 4
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128, "temperature": 0.6}
    async with aiohttp.ClientSession() as s:
        async def one():
            t0 = time.perf_counter()
            async with sem:
                try:
                    async with s.post(f"{HOST}/v1/chat/completions",
                                      json=body, headers=headers, timeout=180) as r:
                        data = await r.json()
                        t1 = time.perf_counter()
                        stats["lat"].append(t1 - t0)
                        if r.status == 200:
                            stats["ok"] += 1
                            u = data.get("usage", {})
                            stats["in_tok"] += u.get("prompt_tokens", 0)
                            stats["out_tok"] += u.get("completion_tokens", 0)
                        else:
                            stats["fail"] += 1
                except Exception:
                    stats["fail"] += 1
        await asyncio.gather(*(one() for _ in range(n)))
    wall = max(stats["lat"]) if stats["lat"] else 0
    p50 = sorted(stats["lat"])[len(stats["lat"]) // 2] if stats["lat"] else 0
    p95 = sorted(stats["lat"])[int(len(stats["lat"]) * 0.95)] if stats["lat"] else 0
    return {
        "concurrency": conc, "ok": stats["ok"], "fail": stats["fail"],
        "in_tok": stats["in_tok"], "out_tok": stats["out_tok"],
        "wall_s": round(wall, 2),
        "throughput_req_s": round(stats["ok"] / wall, 2) if wall else 0,
        "throughput_out_tok_s": round(stats["out_tok"] / wall, 2) if wall else 0,
        "p50_s": round(p50, 3), "p95_s": round(p95, 3),
    }

async def test_concurrency():
    print("=" * 60)
    print("[5] 并发压测 (16 并发, 64 请求)")
    print("=" * 60)
    r = await bench_session(16, 64)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return r.get("fail", 0) == 0

def main():
    print(f"Target: {HOST}  Model: {MODEL}")
    print()
    if not check_health():
        print("服务未就绪，退出。")
        sys.exit(1)
    print()
    test_chat()
    print()
    test_stream()
    print()
    test_vision_local_image()
    print()
    ok = asyncio.run(test_concurrency())
    print()
    print("=" * 60)
    print("DONE" if ok else "PARTIAL")
    print("=" * 60)

if __name__ == "__main__":
    main()
