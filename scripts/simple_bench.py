"""
Benchmark N concurrent Ollama chat requests: latency/throughput + memory.

Edit CONFIG, then run: python benchmark.py
"""

import asyncio
import time
import uuid
import ollama

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HOST = "http://localhost:11441"
MODEL = "mistral-small3.2:24b-32k"
N = 2                  # concurrent requests
NUM_CTX = 8192
TIMEOUT = 240
PROMPT = "Explain KV caching in 3 sentences."
VERBOSE = True          # print streamed tokens as they arrive

client = ollama.AsyncClient(host=HOST)


# ---------------------------------------------------------------------------
# MEMORY: point-in-time snapshot from `ollama ps` (separate from latency,
# since it reflects total server state, not a single request)
# ---------------------------------------------------------------------------
async def get_memory():
    """Return RAM/VRAM split for MODEL, or None if it isn't loaded."""
    for m in (await client.ps()).models:
        if m.model == MODEL:
            total = (getattr(m, "size", 0) or 0) / 1024**2
            vram = (getattr(m, "size_vram", 0) or 0) / 1024**2
            gpu_pct = 100 * vram / total if total else 0
            return {"total_mb": total, "vram_mb": vram, "gpu_pct": gpu_pct, "cpu_pct": 100 - gpu_pct}
    return None


async def unload_model():
    """Force-unload MODEL so the 'before' snapshot is a clean baseline."""
    await client.chat(model=MODEL, messages=[], keep_alive=0)
    await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# REQUESTS: stream one chat completion, collect timing + token metrics
# ---------------------------------------------------------------------------
async def run_request(i: int):
    start = time.perf_counter()
    ttft, tokens, stats = None, 0, {}

    async for chunk in await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": f"[request_id={uuid.uuid4().hex}]\n{PROMPT}"}],
        stream=True,
        options={"temperature": 0.0, "num_ctx": NUM_CTX},
        keep_alive="10m",
    ):
        msg = chunk.get("message", {})
        for kind, text in (("think", msg.get("thinking", "")), ("text", msg.get("content", ""))):
            if text:
                tokens += 1
                ttft = ttft or (time.perf_counter() - start)
                if VERBOSE:
                    print(f"[R{i} T{tokens} {kind}] {text}", flush=True)
        if chunk.get("done"):
            stats = chunk

    wall = time.perf_counter() - start
    eval_n, eval_ns = stats.get("eval_count", 0) or 0, stats.get("eval_duration", 0) or 0
    prompt_n, prompt_ns = stats.get("prompt_eval_count", 0) or 0, stats.get("prompt_eval_duration", 0) or 0

    return {
        "request": i,
        "wall_time_sec": wall,
        "ttft_sec": ttft,
        "output_tokens": eval_n,
        "output_tok_per_sec": eval_n / (eval_ns / 1e9) if eval_ns else 0,
        "prompt_tokens": prompt_n,
        "prompt_tok_per_sec": prompt_n / (prompt_ns / 1e9) if prompt_ns else 0,
    }


# ---------------------------------------------------------------------------
# MAIN: baseline memory -> fire N concurrent requests -> combined report
# ---------------------------------------------------------------------------
async def main():
    print(f"--- experiment: {MODEL} @ {HOST} | N={N} | num_ctx={NUM_CTX} ---")

    await unload_model()
    mem_before = await get_memory()

    batch_start = time.perf_counter()
    results = await asyncio.gather(
        *[asyncio.wait_for(run_request(i), timeout=TIMEOUT) for i in range(1, N + 1)],
        return_exceptions=True,
    )
    batch_elapsed = time.perf_counter() - batch_start
    mem_after = await get_memory()

    print("\n--- results ---")
    total_tokens = 0
    for i, r in enumerate(results, start=1):
        if isinstance(r, Exception):
            print(f"R{i}: FAILED -> {type(r).__name__}: {r}")
            continue
        total_tokens += r["output_tokens"]
        print(
            f"R{i}: wall={r['wall_time_sec']:.2f}s ttft={r['ttft_sec']:.2f}s "
            f"out_tok={r['output_tokens']} out_tok/s={r['output_tok_per_sec']:.2f} "
            f"prompt_tok={r['prompt_tokens']} prompt_tok/s={r['prompt_tok_per_sec']:.2f}"
        )

    batch_tok_per_sec = total_tokens / batch_elapsed if batch_elapsed else 0
    print(f"\nbatch: elapsed={batch_elapsed:.2f}s tokens={total_tokens} tok/s={batch_tok_per_sec:.2f}")

    for label, mem in (("before", mem_before), ("after", mem_after)):
        if mem is None:
            print(f"memory ({label}): {MODEL} not loaded")
        else:
            print(
                f"memory ({label}): total={mem['total_mb']:.0f}MB vram={mem['vram_mb']:.0f}MB "
                f"gpu={mem['gpu_pct']:.1f}% cpu={mem['cpu_pct']:.1f}%"
            )


if __name__ == "__main__":
    asyncio.run(main())
