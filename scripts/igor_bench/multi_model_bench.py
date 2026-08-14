"""
benchmark_multi_model.py
=========================
Benchmarks up to 3 Ollama models running CONCURRENTLY on the same Ollama
instance. For every (context_size, concurrency N) combination, it:

  1. Unloads every model and verifies via `ollama ps` that nothing is
     resident (a clean slate) BEFORE firing any requests.
  2. Fires N concurrent requests to EACH model simultaneously (so total
     in-flight requests = N * len(MODELS)), each with a unique prompt to
     avoid prompt-cache skew.
  3. Captures per-request wall time, time-to-first-token (TTFT), cold
     load time, and tokens/sec.
  4. Immediately after the batch, snapshots `ollama ps` to record each
     model's resident VRAM AND the combined VRAM across all models
     (the number that actually matters when serving several models at once).
  5. Repeats each (context, N) combo RUNS times for stability, then moves on.

Because N requests to the same model are fired at once right after an
unload, only ONE of them actually pays the "cold load" cost -- Ollama loads
the model once and queues the rest. All N load_time values are recorded in
the CSV, but the per-round "effective load time" reported/plotted is the
MAX across that model's N requests in the round.

NOTE: Running 3 models concurrently means their VRAM footprints stack.
Watch the "total_vram_all_models_gb" column -- if it's near/over your GPU's
capacity, later requests may fail or partially offload to CPU, which is
useful signal but will also show up as errors/rows with status=error.

WARM MODE (BENCHMARK_MODE = "warm"):
Instead of unloading before every round, each model is loaded once per
context size with a long keep_alive (WARM_KEEP_ALIVE, default "24h"), and
every request in the sweep also carries that keep_alive so the model never
falls back to Ollama's short default and unloads mid-sweep. No unloading
happens between N levels or RUNS repeats -- you're measuring steady-state
latency/throughput/VRAM with the load cost already paid, which is the
realistic picture for a server that keeps models hot. All models are still
fully unloaded at the very end of the script regardless of mode.
"""

import asyncio
import csv
import statistics
import time
import uuid
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import ollama

# ── Configuration ────────────────────────────────────────────────────────
HOST = "http://localhost:11436"

# Up to 3 models to benchmark concurrently. Order doesn't matter.
MODELS = [
    "qwen3:32b-q8",
    "mistral-small3.2:24b",
    "qwen3-235b-a22b:q4_k_m",
]

CONTEXT_SIZES = [8192, 32768]      # num_ctx values to sweep
N_LIST = [1, 3, 5]                 # concurrent requests PER MODEL, per round
RUNS = 5                           # repeats of each (context, N) combo

# "cold"  -> unload + verify-clean before every round, so every round pays
#            a fresh load cost (measures cold load time).
# "warm"  -> load each model ONCE per context size with a long keep_alive,
#            then run all N/RUNS repeats against the already-resident models
#            with no unload in between (measures steady-state warm latency/
#            throughput/VRAM). All models are still unloaded at the very end.
# NOTE: changing num_ctx always forces Ollama to reload a model (context
# size is fixed at load time), so a reload naturally happens once per new
# context size even in warm mode -- that's real Ollama behavior, not
# something this script forces.
BENCHMARK_MODE = "cold"
WARM_KEEP_ALIVE = "24h"            # Ollama duration string (or an int seconds value)

UNLOAD_TIMEOUT_S = 180             # generous: 3 large models to flush
UNLOAD_POLL_INTERVAL_S = 1.0

REQUEST_RETRY_ATTEMPTS = 3
REQUEST_RETRY_BACKOFF_S = 3        # linear backoff: wait = attempt * backoff

PROMPT_BASE = "Write a sentence with each letter of the english alphabet used EXACTLY once:"

CSV_FIELDS = [
    "timestamp_utc", "context_size", "concurrency_n", "run", "model",
    "status", "wall_time_s", "ttft_s", "load_time_s",
    "tokens_per_sec", "output_tokens", "prompt_tokens", "prompt_eval_time_s",
    "batch_wall_time_s", "vram_gb", "total_size_gb",
    "total_vram_all_models_gb", "error_message",
]

client = ollama.AsyncClient(host=HOST)


# ── Small helpers ────────────────────────────────────────────────────────
def normalize(name: str) -> str:
    """Ollama sometimes reports 'model:latest' vs 'model' inconsistently."""
    return name if ":" in name else f"{name}:latest"


def make_prompt() -> str:
    # unique id per request avoids any prompt-level caching skewing latency
    return f"[BypassCacheID: {uuid.uuid4().hex}]\n{PROMPT_BASE}"


async def with_retries_async(coro_func, *args, attempts=REQUEST_RETRY_ATTEMPTS,
                              backoff=REQUEST_RETRY_BACKOFF_S, **kwargs):
    """Retry an async call with linear backoff. Re-raises the last exception."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                wait = attempt * backoff
                print(f"    [warn] attempt {attempt}/{attempts} failed ({e}); "
                      f"retrying in {wait}s...")
                await asyncio.sleep(wait)
    raise last_exc


# ── Ollama helpers ───────────────────────────────────────────────────────
async def get_ps_snapshot() -> dict:
    """One `ollama ps` call -> {normalized_model_name: {vram_mb, total_mb}}."""
    resp = await client.ps()
    snap = {}
    for m in resp.models:
        snap[normalize(m.model)] = {
            "vram_mb": m.size_vram / (1024 ** 2),
            "total_mb": m.size / (1024 ** 2),
        }
    return snap


async def unload_model(model: str):
    try:
        await client.chat(model=model, messages=[], keep_alive=0)
    except Exception as e:
        print(f"    [warn] unload call failed for {model}: {e}")


async def unload_all_and_verify(models: list[str],
                                 timeout: int = UNLOAD_TIMEOUT_S,
                                 poll: float = UNLOAD_POLL_INTERVAL_S):
    """Unload the target models (and anything else resident), then poll
    `ollama ps` until NOTHING is loaded. Raises TimeoutError if it never clears."""
    targets = {normalize(m) for m in models}

    for m in models:
        await unload_model(m)
    snap = await get_ps_snapshot()
    for other in snap:
        if other not in targets:
            await unload_model(other)

    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        snap = await get_ps_snapshot()
        if not snap:
            return
        await asyncio.sleep(poll)

    raise TimeoutError(
        f"Models still resident after {timeout}s: {list((await get_ps_snapshot()).keys())}"
    )


async def warm_up_models(models: list[str], num_ctx: int, keep_alive: str):
    """Load every model once (concurrently) with a long keep_alive so it's
    resident and ready before the timed sweep starts. Used in warm mode."""
    print(f"\n  Warm-up: loading {len(models)} models "
          f"(num_ctx={num_ctx}, keep_alive={keep_alive})...")
    tasks = [with_retries_async(run_single_request, m, num_ctx, keep_alive) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for m, r in zip(models, results):
        if isinstance(r, Exception):
            print(f"    [error] warm-up failed for {m}: {r}")
        else:
            print(f"    {m:<32} loaded in {r['load_time_s']:.2f}s")
    snap = await get_ps_snapshot()
    total_gb = sum(v["vram_mb"] for v in snap.values()) / 1024
    print(f"  Combined VRAM after warm-up: {total_gb:.2f} GB")


# ── Single request ───────────────────────────────────────────────────────
async def run_single_request(model: str, num_ctx: int, keep_alive=None) -> dict:
    """keep_alive: pass a duration (e.g. "24h") to keep the model resident
    after this request; omit for cold-mode requests to use Ollama's default."""
    output_tokens = 0
    eval_duration_ns = 0
    load_duration_ns = 0
    prompt_eval_count = 0
    prompt_eval_duration_ns = 0
    ttft = None

    chat_kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": make_prompt()}],
        stream=True,
        options={"temperature": 0.0, "num_ctx": num_ctx},
    )
    if keep_alive is not None:
        chat_kwargs["keep_alive"] = keep_alive

    start = time.perf_counter()
    async for chunk in await client.chat(**chat_kwargs):
        content = chunk.get("message", {}).get("content")
        if content and ttft is None:
            ttft = time.perf_counter() - start
        if chunk.get("done"):
            output_tokens = chunk.get("eval_count", 0)
            eval_duration_ns = chunk.get("eval_duration", 0)
            load_duration_ns = chunk.get("load_duration", 0)
            prompt_eval_count = chunk.get("prompt_eval_count", 0)
            prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
    wall_time = time.perf_counter() - start
    tps = output_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0

    return {
        "model": model,
        "wall_time_s": wall_time,
        "ttft_s": ttft if ttft is not None else wall_time,
        "load_time_s": load_duration_ns / 1e9,
        "tokens_per_sec": tps,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_eval_count,
        "prompt_eval_time_s": prompt_eval_duration_ns / 1e9,
    }


# ── CSV I/O ──────────────────────────────────────────────────────────────
def init_csv() -> str:
    path = (f"multi_model_benchmark_{BENCHMARK_MODE}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    return path


def append_rows(path: str, rows: list[dict]):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            writer.writerow(row)


# ── One (context, N, run) round across all models ──────────────────────
async def run_round(models: list[str], num_ctx: int, n: int, run_idx: int,
                     csv_path: str, warm: bool = False,
                     keep_alive: str | None = None) -> dict:
    mode_label = "warm" if warm else "cold"
    print(f"\n=== [{mode_label}] ctx={num_ctx} | N={n}/model | run {run_idx}/{RUNS} ===")

    if warm:
        # Models are expected to already be resident from warm_up_models()
        # (or a previous round at this same context size) -- no unload here,
        # that's the whole point of warm mode.
        pass
    else:
        print("  Unloading all models & verifying clean state...")
        await unload_all_and_verify(models)

    tasks = []
    for model in models:
        for _ in range(n):
            tasks.append(with_retries_async(run_single_request, model, num_ctx, keep_alive))

    batch_start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    batch_wall = time.perf_counter() - batch_start

    vram_snap = await get_ps_snapshot()
    total_vram_gb = sum(v["vram_mb"] for v in vram_snap.values()) / 1024

    per_model_results = {m: [] for m in models}
    errors = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
            continue
        per_model_results[r["model"]].append(r)

    ts = datetime.now(timezone.utc).isoformat()
    rows = []
    round_summary = {}  # model -> aggregated stats for this round

    for model in models:
        reqs = per_model_results[model]
        vram_info = vram_snap.get(normalize(model), {"vram_mb": 0.0, "total_mb": 0.0})
        for r in reqs:
            rows.append({
                "timestamp_utc": ts, "context_size": num_ctx, "concurrency_n": n,
                "run": run_idx, "model": model, "status": "ok",
                "wall_time_s": round(r["wall_time_s"], 2),
                "ttft_s": round(r["ttft_s"], 2),
                "load_time_s": round(r["load_time_s"], 2),
                "tokens_per_sec": round(r["tokens_per_sec"], 2),
                "output_tokens": r["output_tokens"], "prompt_tokens": r["prompt_tokens"],
                "prompt_eval_time_s": round(r["prompt_eval_time_s"], 2),
                "batch_wall_time_s": round(batch_wall, 2),
                "vram_gb": round(vram_info["vram_mb"] / 1024, 3),
                "total_size_gb": round(vram_info["total_mb"] / 1024, 3),
                "total_vram_all_models_gb": round(total_vram_gb, 3),
                "error_message": "",
            })

        n_failed = n - len(reqs)
        for _ in range(n_failed):
            rows.append({
                "timestamp_utc": ts, "context_size": num_ctx, "concurrency_n": n,
                "run": run_idx, "model": model, "status": "error",
                "wall_time_s": "", "ttft_s": "", "load_time_s": "",
                "tokens_per_sec": "", "output_tokens": "", "prompt_tokens": "",
                "prompt_eval_time_s": "", "batch_wall_time_s": round(batch_wall, 2),
                "vram_gb": "", "total_size_gb": "",
                "total_vram_all_models_gb": round(total_vram_gb, 3),
                "error_message": "; ".join(errors)[:300],
            })

        if reqs:
            round_summary[model] = {
                "wall_time_s": statistics.mean(r["wall_time_s"] for r in reqs),
                "ttft_s": statistics.mean(r["ttft_s"] for r in reqs),
                "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in reqs),
                "load_time_s": max(r["load_time_s"] for r in reqs),  # see module docstring
                "vram_gb": vram_info["vram_mb"] / 1024,
            }
            s = round_summary[model]
            print(f"  {model:<32} ok={len(reqs)}/{n}  "
                  f"wall={s['wall_time_s']:.2f}s  ttft={s['ttft_s']:.2f}s  "
                  f"tok/s={s['tokens_per_sec']:.1f}  "
                  f"load={s['load_time_s']:.2f}s  vram={s['vram_gb']:.2f}GB")
        else:
            print(f"  {model:<32} ALL {n} REQUESTS FAILED")

    print(f"  Combined VRAM (all models resident): {total_vram_gb:.2f} GB")
    append_rows(csv_path, rows)
    round_summary["_total_vram_gb"] = total_vram_gb
    return round_summary


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_results(agg: dict, models: list[str]):
    """agg[num_ctx][n][model] = {wall_time_s, ttft_s, tokens_per_sec, load_time_s, vram_gb}
       agg[num_ctx][n]['_total_vram_gb'] = combined vram across models."""
    for num_ctx, by_n in agg.items():
        n_vals = sorted(by_n.keys())
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f"Multi-model concurrent benchmark [{BENCHMARK_MODE}] — num_ctx={num_ctx}")

        metrics = [
            ("wall_time_s", "Wall time per request (s)", axes[0, 0]),
            ("ttft_s", "Time to first token (s)", axes[0, 1]),
            ("tokens_per_sec", "Tokens/sec per request", axes[1, 0]),
            ("vram_gb", "VRAM per model (GB)", axes[1, 1]),
        ]
        for key, title, ax in metrics:
            for model in models:
                ys = [by_n[n].get(model, {}).get(key) for n in n_vals]
                if any(y is not None for y in ys):
                    ax.plot(n_vals, ys, marker="o", label=model)
            ax.set_title(title)
            ax.set_xlabel("Concurrent requests per model (N)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # overlay combined VRAM on the VRAM subplot
        total_ys = [by_n[n].get("_total_vram_gb") for n in n_vals]
        axes[1, 1].plot(n_vals, total_ys, marker="s", linestyle="--",
                         color="black", label="TOTAL (all models)")
        axes[1, 1].legend(fontsize=8)

        fig.tight_layout()
        out_path = f"multi_model_benchmark_{BENCHMARK_MODE}_ctx{num_ctx}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot -> {out_path}")


# ── Main ────────────────────────────────────────────────────────────────
async def main():
    assert 1 <= len(MODELS) <= 3, "This script is designed for up to 3 concurrent models."

    csv_path = init_csv()
    agg = {}  # agg[num_ctx][n][model] = averaged stats across RUNS
    warm = BENCHMARK_MODE == "warm"

    if warm:
        print(f"Running in WARM mode (keep_alive={WARM_KEEP_ALIVE}). "
              "Models stay resident between rounds; only a context-size "
              "change triggers a reload.")
    else:
        print("Running in COLD mode. Every round unloads and reloads from scratch.")

    for num_ctx in CONTEXT_SIZES:
        agg[num_ctx] = {}
        if warm:
            # Load once for this context size; subsequent rounds at the
            # same num_ctx reuse the already-resident models.
            await warm_up_models(MODELS, num_ctx, WARM_KEEP_ALIVE)

        for n in N_LIST:
            round_runs = []
            for run_idx in range(1, RUNS + 1):
                summary = await run_round(MODELS, num_ctx, n, run_idx, csv_path,
                                           warm=warm, keep_alive=WARM_KEEP_ALIVE if warm else None)
                round_runs.append(summary)

            # average the RUNS repeats for this (num_ctx, n)
            merged = {}
            for model in MODELS:
                samples = [r[model] for r in round_runs if model in r]
                if samples:
                    merged[model] = {
                        k: statistics.mean(s[k] for s in samples)
                        for k in ("wall_time_s", "ttft_s", "tokens_per_sec",
                                  "load_time_s", "vram_gb")
                    }
            merged["_total_vram_gb"] = statistics.mean(
                r["_total_vram_gb"] for r in round_runs if "_total_vram_gb" in r
            )
            agg[num_ctx][n] = merged

    print("\nFinal cleanup: unloading all models...")
    try:
        await unload_all_and_verify(MODELS, timeout=60)
    except TimeoutError as e:
        print(f"  [warn] cleanup unload incomplete: {e}")

    plot_results(agg, MODELS)
    print(f"\nPer-request results written to: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
