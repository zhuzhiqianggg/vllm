#!/usr/bin/env python3
"""
简单 HTTP 压测工具（无需 vllm bench）
用法:
  python3 bench_http.py [concurrency=32] [num_prompts=200] [input_tokens=512] [output_tokens=256]
"""
import os
import sys
import time
import json
import asyncio
import aiohttp

HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
MODEL = os.getenv("SERVED_MODEL_NAME", "default")
API_KEY = os.getenv("VLLM_API_KEY", "")

PROMPT = "请用中文写一段关于人工智能对软件开发影响的论述。" * 16  # 大约 512 tokens

async def send_one(session, sem, stats, idx):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": int(os.getenv("OUT_TOK", sys.argv[4] if len(sys.argv) > 4 else 256)),
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with sem:
        try:
            async with session.post(f"{HOST}/v1/chat/completions",
                                    json=body, headers=headers, timeout=600) as r:
                data = await r.json()
                t1 = time.perf_counter()
                if r.status != 200:
                    stats["fail"] += 1
                    return
                usage = data.get("usage", {})
                stats["ok"] += 1
                stats["in_tok"]  += usage.get("prompt_tokens", 0)
                stats["out_tok"] += usage.get("completion_tokens", 0)
                stats["lat"].append(t1 - t0)
        except Exception as e:
            stats["fail"] += 1

async def main():
    conc = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    stats = {"ok":0,"fail":0,"in_tok":0,"out_tok":0,"lat":[]}
    sem = asyncio.Semaphore(conc)
    t_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(send_one(session, sem, stats, i) for i in range(n)))
    t_end = time.perf_counter()
    wall = t_end - t_start
    lat = sorted(stats["lat"])
    p50 = lat[len(lat)//2] if lat else 0
    p95 = lat[int(len(lat)*0.95)] if lat else 0
    p99 = lat[int(len(lat)*0.99)] if lat else 0
    print(json.dumps({
        "concurrency": conc,
        "requests": n,
        "ok": stats["ok"],
        "fail": stats["fail"],
        "wall_time_s": round(wall, 2),
        "throughput_req_s": round(stats["ok"]/wall, 2),
        "throughput_in_tok_s":  round(stats["in_tok"]/wall, 2),
        "throughput_out_tok_s": round(stats["out_tok"]/wall, 2),
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "latency_p99_s": round(p99, 3),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
