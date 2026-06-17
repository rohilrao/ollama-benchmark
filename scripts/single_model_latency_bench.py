"""
benchmark_latency.py
=====================
Measures latency for a single Ollama model under increasing concurrency (N
parallel requests). Tracks, per N:
  - wall_time:       total time from request sent to response fully received
  - ttft:             time-to-first-token (time until the first content chunk arrives)
  - tokens_per_sec:   per-request generation speed (output_tokens / eval_duration)
  - batch_tokens_per_sec: aggregate throughput across all N requests in the batch
                      (sum of output tokens / batch wall-clock time)
Plots all four metrics against N and saves a PNG.
"""

import asyncio
import statistics
import time
import uuid

import matplotlib.pyplot as plt
import ollama

# ── Configuration ────────────────────────────────────────────────────────
HOST   = "http://localhost:11436"
MODEL  = "mistral-small3.2:24b"
N_LIST = [1, 2, 4, 6, 8, 10]   # concurrency levels to sweep
M      = 2                     # repeats per N, averaged for stability
PROMPT_BASE = "Write a sentence with each letter of the english alphabet used EXACTLY once:"

client = ollama.AsyncClient(host=HOST)


def generate_prompt() -> str:
    # unique id per request avoids any prompt-level caching skewing latency
    return f"[BypassCacheID: {uuid.uuid4().hex}]\n{PROMPT_BASE}"


# ── Warmup ───────────────────────────────────────────────────────────────
async def warmup(n: int):
    """Send n throwaway requests so the model is loaded and GPU is primed."""
    print(f"Warming up '{MODEL}' with {n} requests...")
    await asyncio.gather(*[
        client.chat(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
        for _ in range(n)
    ])
    print("Warmup complete.\n")


# ── Single request ───────────────────────────────────────────────────────
async def run_request(i: int) -> dict:
    """Run one streaming request, capturing wall time, TTFT, and tok/s.

    Prints i once per token received, so you can see which of the N
    concurrent requests is actively streaming at any moment.
    """
    output_tokens = 0
    eval_duration_ns = 0
    ttft = None

    start = time.perf_counter()
    async for chunk in await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": generate_prompt()}],
        stream=True,
        options={"temperature": 0.0},
    ):
        if chunk.get("message", {}).get("content"):
            print(i, end=" ", flush=True)
        if ttft is None and chunk.get("message", {}).get("content"):
            ttft = time.perf_counter() - start
        if chunk.get("done"):
            output_tokens = chunk.get("eval_count", 0)
            eval_duration_ns = chunk.get("eval_duration", 0)

    wall_time = time.perf_counter() - start
    tps = output_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0

    return {
        "wall_time_sec": wall_time,
        "ttft_sec": ttft if ttft is not None else wall_time,
        "tokens_per_sec": tps,
        "output_tokens": output_tokens,
    }


# ── Batch of N concurrent requests ─────────────────────────────────────────
async def run_batch(n: int) -> dict:
    """Fire n concurrent requests, return per-request results + batch throughput."""
    batch_start = time.perf_counter()
    results = await asyncio.gather(*[run_request(i) for i in range(1, n + 1)])
    print()  # newline after the interleaved token-index output
    batch_time = time.perf_counter() - batch_start

    total_tokens = sum(r["output_tokens"] for r in results)
    return {
        "n": n,
        "wall_time_sec": statistics.mean(r["wall_time_sec"] for r in results),
        "ttft_sec": statistics.mean(r["ttft_sec"] for r in results),
        "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in results),
        "batch_tokens_per_sec": total_tokens / batch_time,
    }


# ── Sweep over concurrency levels ───────────────────────────────────────
async def run_sweep() -> list:
    summary = []
    for n in N_LIST:
        repeats = [await run_batch(n) for _ in range(M)]
        avg = {
            "n": n,
            "wall_time_sec": statistics.mean(r["wall_time_sec"] for r in repeats),
            "ttft_sec": statistics.mean(r["ttft_sec"] for r in repeats),
            "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in repeats),
            "batch_tokens_per_sec": statistics.mean(r["batch_tokens_per_sec"] for r in repeats),
        }
        summary.append(avg)
        print(f"N={n:2d}  wall={avg['wall_time_sec']:.2f}s  "
              f"ttft={avg['ttft_sec']:.2f}s  "
              f"tok/s/req={avg['tokens_per_sec']:.1f}  "
              f"batch tok/s={avg['batch_tokens_per_sec']:.1f}")
    return summary


# ── Plotting ────────────────────────────────────────────────────────────
def plot_results(summary: list):
    n_vals = [s["n"] for s in summary]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Latency vs. Concurrency — {MODEL}")

    axes[0, 0].plot(n_vals, [s["wall_time_sec"] for s in summary], marker="o")
    axes[0, 0].set_title("Wall time per request (s)")
    axes[0, 0].set_xlabel("Concurrent requests (N)")

    axes[0, 1].plot(n_vals, [s["ttft_sec"] for s in summary], marker="o", color="orange")
    axes[0, 1].set_title("Time to first token (s)")
    axes[0, 1].set_xlabel("Concurrent requests (N)")

    axes[1, 0].plot(n_vals, [s["tokens_per_sec"] for s in summary], marker="o", color="green")
    axes[1, 0].set_title("Tokens/sec per request")
    axes[1, 0].set_xlabel("Concurrent requests (N)")

    axes[1, 1].plot(n_vals, [s["batch_tokens_per_sec"] for s in summary], marker="o", color="red")
    axes[1, 1].set_title("Batch tokens/sec (aggregate)")
    axes[1, 1].set_xlabel("Concurrent requests (N)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = "latency_benchmark.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot → {out_path}")


# ── Main ────────────────────────────────────────────────────────────────
async def main():
    await warmup(max(N_LIST))
    summary = await run_sweep()
    plot_results(summary)


if __name__ == "__main__":
    asyncio.run(main())
