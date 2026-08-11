import time
import csv
import statistics
import requests
from datetime import datetime, timezone

OLLAMA_URL = "http://localhost:11436"
CSV_PATH = "ollama_load_benchmark_results.csv"
CSV_FIELDS = [
    "model", "context_size", "run", "timestamp_utc", "status",
    "load_time_s", "total_time_s", "wall_time_s",
    "prompt_eval_time_s", "eval_time_s",
    "prompt_tokens", "generated_tokens",
    "vram_gb", "total_size_gb", "vram_pct",
    "error_message",
]

# Context sizes (num_ctx) to benchmark, in tokens.
CONTEXT_SIZES = [8192, 32768, 65536, 65536*2]

# List every model you want to benchmark, in order.
MODELS = [
    "qwen3:32b-q8",
    "mistral-small3.2:24b",
    "qwen3-235b-a22b:q4_k_m",
    "qwen3-next-80b-a3b-thinking:latest",
    "deepseek-r1-0528-q4_k_m:latest"
]

RUNS = 5
UNLOAD_TIMEOUT_S = 120       # generous timeout for large models
UNLOAD_POLL_INTERVAL_S = 1.0
UNLOAD_RETRY_ATTEMPTS = 3    # re-send the unload call if the model is still resident

REQUEST_RETRY_ATTEMPTS = 5   # retries for transient HTTP/connection errors (e.g. 500s, timeouts)
REQUEST_RETRY_BACKOFF_S = 3  # linear backoff: wait = attempt * REQUEST_RETRY_BACKOFF_S


def with_retries(func, *args, attempts: int = REQUEST_RETRY_ATTEMPTS, **kwargs):
    """
    Call func(*args, **kwargs), retrying on transient network/HTTP errors
    (connection errors, timeouts, 5xx responses, etc.) with linear backoff.
    Re-raises the last exception if every attempt fails.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < attempts:
                wait = attempt * REQUEST_RETRY_BACKOFF_S
                print(f"    [warn] attempt {attempt}/{attempts} failed ({e}); "
                      f"retrying in {wait}s...")
                time.sleep(wait)
    raise last_exc


def normalize(name: str) -> str:
    """Ollama sometimes reports 'model:latest' vs 'model' inconsistently. Normalize for comparison."""
    return name if ":" in name else f"{name}:latest"


def get_loaded_models() -> list[str]:
    """Return the list of model names currently resident in memory, per /api/ps."""
    response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10)
    response.raise_for_status()
    models = response.json().get("models", [])
    # Different Ollama versions expose the identifier under "name" and/or "model" - check both.
    names = []
    for m in models:
        names.append(normalize(m.get("name") or m.get("model", "")))
    return names


def get_model_memory_info(model: str) -> dict:
    """
    Query /api/ps and return the memory breakdown for `model`, if currently loaded:
    total size, how much of that sits in VRAM, and the VRAM percentage
    (useful for spotting partial CPU/GPU offload, e.g. on a context size
    that no longer fits entirely on the GPU).
    """
    target = normalize(model)
    response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10)
    response.raise_for_status()
    models = response.json().get("models", [])
    for m in models:
        name = normalize(m.get("name") or m.get("model", ""))
        if name == target:
            size_bytes = m.get("size", 0)
            vram_bytes = m.get("size_vram", 0)
            return {
                "vram_gb": vram_bytes / 1e9,
                "total_size_gb": size_bytes / 1e9,
                "vram_pct": (vram_bytes / size_bytes * 100) if size_bytes else 0.0,
            }
    # Model wasn't found loaded (shouldn't normally happen right after a successful load).
    return {"vram_gb": 0.0, "total_size_gb": 0.0, "vram_pct": 0.0}


def request_unload(model: str):
    """Send a single unload request for `model`. No-op if it isn't loaded."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "keep_alive": 0},
        timeout=60,
    )
    response.raise_for_status()


def unload_and_verify(model: str, timeout: int = UNLOAD_TIMEOUT_S):
    """
    Unload `model` and don't return until /api/ps confirms it is actually gone.
    Also sweeps up any *other* resident models, so a later model's benchmark
    never starts with leftover VRAM/RAM pressure from a previous one.
    """
    target = normalize(model)

    for attempt in range(1, UNLOAD_RETRY_ATTEMPTS + 1):
        # Unload the target model explicitly.
        with_retries(request_unload, model)

        # Also unload anything else currently resident (e.g. a different model
        # left loaded from a previous benchmark run or another process).
        for other in with_retries(get_loaded_models):
            if other != target:
                with_retries(request_unload, other)

        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            loaded = with_retries(get_loaded_models)
            if not loaded:
                return  # nothing resident at all - confirmed clean
            time.sleep(UNLOAD_POLL_INTERVAL_S)

        if attempt < UNLOAD_RETRY_ATTEMPTS:
            print(f"  [warn] {model} still resident after {timeout}s, retrying unload "
                  f"(attempt {attempt + 1}/{UNLOAD_RETRY_ATTEMPTS})...")
        else:
            raise TimeoutError(
                f"{model} still shows as loaded after {UNLOAD_RETRY_ATTEMPTS} unload "
                f"attempts and {timeout}s each. Currently loaded: {with_retries(get_loaded_models)}"
            )


def cold_load(model: str, num_ctx: int) -> dict:
    """Send a request that forces `model` to load with the given context size. Returns Ollama's timing stats."""
    wall_start = time.perf_counter()
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": "Say OK.",
            "stream": False,
            "keep_alive": -1,
            "options": {
                "num_predict": 1,
                "num_ctx": num_ctx,
            },
        },
        timeout=1800,  # important for very large models
    )
    wall_time = time.perf_counter() - wall_start
    response.raise_for_status()
    data = response.json()
    return {
        "load_time_s": data.get("load_duration", 0) / 1e9,
        "total_time_s": data.get("total_duration", 0) / 1e9,
        "prompt_eval_time_s": data.get("prompt_eval_duration", 0) / 1e9,
        "eval_time_s": data.get("eval_duration", 0) / 1e9,
        "wall_time_s": wall_time,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "generated_tokens": data.get("eval_count", 0),
    }


def init_csv(path: str = CSV_PATH):
    """Create the CSV with a header row (overwrites any previous file)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv_row(model: str, num_ctx: int, run: int, result: dict, path: str = CSV_PATH):
    """Append a single successful run's result immediately, so data survives even if a later run crashes."""
    row = {
        "model": model,
        "context_size": num_ctx,
        "run": run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "error_message": "",
        **{k: round(result[k], 2) for k in (
            "load_time_s", "total_time_s", "wall_time_s",
            "prompt_eval_time_s", "eval_time_s",
            "vram_gb", "total_size_gb", "vram_pct",
        )},
        "prompt_tokens": result["prompt_tokens"],
        "generated_tokens": result["generated_tokens"],
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def append_error_row(model: str, num_ctx: int, run: int, error_message: str, path: str = CSV_PATH):
    """Record a run that failed after exhausting all retries, then move on."""
    row = {
        "model": model,
        "context_size": num_ctx,
        "run": run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "error_message": str(error_message)[:500],  # keep the CSV readable
        "load_time_s": "", "total_time_s": "", "wall_time_s": "",
        "prompt_eval_time_s": "", "eval_time_s": "",
        "prompt_tokens": "", "generated_tokens": "",
        "vram_gb": "", "total_size_gb": "", "vram_pct": "",
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def print_summary(name: str, values: list[float]):
    print(
        f"{name:<22}"
        f"mean={statistics.mean(values):8.2f}s  "
        f"std={statistics.stdev(values) if len(values) > 1 else 0:7.2f}s  "
        f"min={min(values):8.2f}s  "
        f"max={max(values):8.2f}s"
    )


def benchmark_model_context(model: str, num_ctx: int) -> list[dict]:
    results = []
    for run in range(1, RUNS + 1):
        print(f"\n--- {model} | ctx={num_ctx} | Run {run}/{RUNS} ---")

        try:
            print("Unloading (and verifying)...")
            unload_and_verify(model)
        except (TimeoutError, requests.exceptions.RequestException) as e:
            print(f"  [error] unload failed after retries: {e}")
            append_error_row(model, num_ctx, run, f"unload failed: {e}")
            continue  # skip this run, move to the next one

        try:
            print("Loading...")
            result = with_retries(cold_load, model, num_ctx)
        except requests.exceptions.RequestException as e:
            print(f"  [error] load failed after retries: {e}")
            append_error_row(model, num_ctx, run, f"load failed: {e}")
            continue

        # Fetch VRAM/size breakdown while the model is still resident.
        # Non-fatal: if this fails, keep the timing data and just zero out the memory fields.
        try:
            mem_info = with_retries(get_model_memory_info, model)
        except requests.exceptions.RequestException as e:
            print(f"  [warn] could not fetch VRAM info: {e}")
            mem_info = {"vram_gb": 0.0, "total_size_gb": 0.0, "vram_pct": 0.0}
        result.update(mem_info)

        results.append(result)
        append_csv_row(model, num_ctx, run, result)
        print(f"Load duration:       {result['load_time_s']:.2f} s")
        print(f"Total duration:      {result['total_time_s']:.2f} s")
        print(f"HTTP wall time:      {result['wall_time_s']:.2f} s")
        print(f"Prompt eval:         {result['prompt_eval_time_s']:.2f} s")
        print(f"Generation:          {result['eval_time_s']:.2f} s")
        print(f"VRAM used:           {result['vram_gb']:.2f} GB / "
              f"{result['total_size_gb']:.2f} GB total ({result['vram_pct']:.2f}% on GPU)")
    return results


def main():
    init_csv()
    all_results = {}  # (model, num_ctx) -> list[dict]
    for model in MODELS:
        for num_ctx in CONTEXT_SIZES:
            all_results[(model, num_ctx)] = benchmark_model_context(model, num_ctx)

    # Final unload so nothing is left resident after the benchmark.
    print("\nCleaning up - unloading all models...")
    for model in MODELS:
        try:
            unload_and_verify(model, timeout=60)
        except (TimeoutError, requests.exceptions.RequestException) as e:
            print(f"  [warn] cleanup unload failed for {model}: {e}")

    print("\n" + "=" * 70)
    print(f"RUNS PER (MODEL, CONTEXT SIZE): {RUNS}")
    print("=" * 70)
    for (model, num_ctx), results in all_results.items():
        print(f"\nMODEL: {model}  |  CONTEXT: {num_ctx}")
        if not results:
            print("  No successful runs (all attempts errored - see CSV for details).")
            continue
        print_summary("Model load", [r["load_time_s"] for r in results])
        print_summary("Total Ollama time", [r["total_time_s"] for r in results])
        print_summary("Wall-clock time", [r["wall_time_s"] for r in results])
        print_summary("VRAM used (GB)", [r["vram_gb"] for r in results])

    print(f"\nPer-run results written to: {CSV_PATH}")


if __name__ == "__main__":
    main()
