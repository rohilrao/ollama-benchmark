import time
import csv
import statistics
import requests
from datetime import datetime, timezone

OLLAMA_URL = "http://localhost:11434"
CSV_PATH = "ollama_load_benchmark_results.csv"
CSV_FIELDS = [
    "model", "run", "timestamp_utc",
    "load_time_s", "total_time_s", "wall_time_s",
    "prompt_eval_time_s", "eval_time_s",
    "prompt_tokens", "generated_tokens",
]

# List every model you want to benchmark, in order.
MODELS = [
    "qwen3:235b",
    "llama3.1:70b",
    "mistral-large:latest",
]

RUNS = 5
UNLOAD_TIMEOUT_S = 120       # generous timeout for large models
UNLOAD_POLL_INTERVAL_S = 1.0
UNLOAD_RETRY_ATTEMPTS = 3    # re-send the unload call if the model is still resident


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
        request_unload(model)

        # Also unload anything else currently resident (e.g. a different model
        # left loaded from a previous benchmark run or another process).
        for other in get_loaded_models():
            if other != target:
                request_unload(other)

        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            loaded = get_loaded_models()
            if not loaded:
                return  # nothing resident at all - confirmed clean
            time.sleep(UNLOAD_POLL_INTERVAL_S)

        if attempt < UNLOAD_RETRY_ATTEMPTS:
            print(f"  [warn] {model} still resident after {timeout}s, retrying unload "
                  f"(attempt {attempt + 1}/{UNLOAD_RETRY_ATTEMPTS})...")
        else:
            raise TimeoutError(
                f"{model} still shows as loaded after {UNLOAD_RETRY_ATTEMPTS} unload "
                f"attempts and {timeout}s each. Currently loaded: {get_loaded_models()}"
            )


def cold_load(model: str) -> dict:
    """Send a request that forces `model` to load. Returns Ollama's timing statistics."""
    wall_start = time.perf_counter()
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": "Say OK.",
            "stream": False,
            "keep_alive": -1,
            "options": {"num_predict": 1},
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


def append_csv_row(model: str, run: int, result: dict, path: str = CSV_PATH):
    """Append a single run's result immediately, so data survives even if a later run crashes."""
    row = {
        "model": model,
        "run": run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **{k: result[k] for k in (
            "load_time_s", "total_time_s", "wall_time_s",
            "prompt_eval_time_s", "eval_time_s",
            "prompt_tokens", "generated_tokens",
        )},
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def print_summary(name: str, values: list[float]):
    print(
        f"{name:<22}"
        f"mean={statistics.mean(values):8.3f}s  "
        f"std={statistics.stdev(values) if len(values) > 1 else 0:7.3f}s  "
        f"min={min(values):8.3f}s  "
        f"max={max(values):8.3f}s"
    )


def benchmark_model(model: str) -> list[dict]:
    results = []
    for run in range(1, RUNS + 1):
        print(f"\n--- {model} | Run {run}/{RUNS} ---")
        print("Unloading (and verifying)...")
        unload_and_verify(model)
        print("Loading...")
        result = cold_load(model)
        results.append(result)
        append_csv_row(model, run, result)
        print(f"Load duration:       {result['load_time_s']:.3f} s")
        print(f"Total duration:      {result['total_time_s']:.3f} s")
        print(f"HTTP wall time:      {result['wall_time_s']:.3f} s")
        print(f"Prompt eval:         {result['prompt_eval_time_s']:.3f} s")
        print(f"Generation:          {result['eval_time_s']:.3f} s")
    return results


def main():
    init_csv()
    all_results = {}
    for model in MODELS:
        all_results[model] = benchmark_model(model)

    # Final unload so nothing is left resident after the benchmark.
    print("\nCleaning up - unloading all models...")
    for model in MODELS:
        try:
            unload_and_verify(model, timeout=60)
        except TimeoutError as e:
            print(f"  [warn] cleanup unload failed for {model}: {e}")

    print("\n" + "=" * 70)
    print(f"RUNS PER MODEL: {RUNS}")
    print("=" * 70)
    for model, results in all_results.items():
        print(f"\nMODEL: {model}")
        print_summary("Model load", [r["load_time_s"] for r in results])
        print_summary("Total Ollama time", [r["total_time_s"] for r in results])
        print_summary("Wall-clock time", [r["wall_time_s"] for r in results])

    print(f"\nPer-run results written to: {CSV_PATH}")


if __name__ == "__main__":
    main()
