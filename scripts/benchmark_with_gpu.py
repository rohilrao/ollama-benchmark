"""
benchmark_with_gpu.py
=====================
Runs N concurrent Ollama requests across multiple models while tracking GPU metrics via NVML.
"""


import asyncio
import json
import statistics
import time
import uuid
from datetime import datetime


import ollama
import pynvml


# ── Configuration ──────────────────────────────────────────────────────────────


HOST   = "http://localhost:11436"
MODELS = [
    "mistral-small3.2:24b",
    "mistral-small3.2:24b-32k",
    "mistral-v0.3:latest",
]
N_MAX  = 10  # number of parallel requests per model
M      = 2  # number of repeats for averaging
PROMPT_BASE = "Write a sentence with each letter of the english alphabet used EXACTLY once:"
GPU_POLL_INTERVAL = 0.5  # seconds between GPU telemetry snapshots


client = ollama.AsyncClient(host=HOST)




def generate_prompt() -> str:
    return f"[BypassCacheID: {uuid.uuid4().hex}]\n{PROMPT_BASE}"




# ── GPU Monitor ────────────────────────────────────────────────────────────────


class GPUMonitor:
    """Asynchronously polls NVML for GPU stats."""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.telemetry = []


        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]


        print(f"GPU Monitor Initialized. Found {self.device_count} GPUs:")
        for i, h in enumerate(self.handles):
            name = pynvml.nvmlDeviceGetName(h)
            print(f"  GPU {i}: {name}")


    async def start(self):
        self.running = True
        self.telemetry = []
        self._task = asyncio.create_task(self._poll_loop())


    async def _poll_loop(self):
        while self.running:
            snapshot = {"timestamp": time.time(), "gpus": []}
            for i, handle in enumerate(self.handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)


                snapshot["gpus"].append({
                    "id": i,
                    "gpu_util_percent": util.gpu,
                    "mem_util_percent": util.memory,
                    "vram_used_mb": round(mem.used / (1024**2), 2),
                    "power_w": round(power_mw / 1000.0, 2)
                })
            self.telemetry.append(snapshot)
            await asyncio.sleep(self.interval)


    async def stop(self) -> list:
        self.running = False
        await self._task
        return self.telemetry


    def shutdown(self):
        pynvml.nvmlShutdown()




# ── Core Functions ─────────────────────────────────────────────────────────────


async def warmup_model():
    print(f"\nWarming up {len(MODELS)} models...")
    async def ping(model):
        await client.chat(model=model, messages=[{"role": "user", "content": "Hi"}])
    await asyncio.gather(*[ping(m) for m in MODELS for _ in range(N_MAX)])
    print("Warmup complete.")




async def run_request(i: int, model: str) -> dict:
    text = ""
    output_tokens = 0
    eval_duration_ns = 0
    start = time.perf_counter()


    async for chunk in await client.chat(
        model=model,
        messages=[{"role": "user", "content": generate_prompt()}],
        stream=True,
        options={"temperature": 0.0},
    ):
        if "message" in chunk and "content" in chunk["message"]:
            text += chunk["message"]["content"]
            print(f"{i}", end=" ", flush=True)


        if chunk.get("done"):
            output_tokens    = chunk.get("eval_count", 0)
            eval_duration_ns = chunk.get("eval_duration", 0)


    wall_time = time.perf_counter() - start
    tps = output_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0


    return {
        "model": model,
        "request_id": i,
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tps, 2),
        "wall_time_sec": round(wall_time, 3),
    }




async def run_batch(n: int, monitor: GPUMonitor) -> dict:
    await monitor.start()


    batch_start = time.perf_counter()
    results = await asyncio.gather(*[
        run_request(i, model)
        for model in MODELS
        for i in range(1, n + 1)
    ])
    batch_time = time.perf_counter() - batch_start


    gpu_telemetry = await monitor.stop()


    per_model = {}
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        tokens = [r["output_tokens"]  for r in model_results]
        tps    = [r["tokens_per_sec"] for r in model_results]
        times  = [r["wall_time_sec"]  for r in model_results]
        per_model[model] = {
            "avg_tokens_per_sec_per_request": round(statistics.mean(tps), 2),
            "batch_tokens_per_sec":           round(sum(tokens) / batch_time, 2),
            "avg_time_per_request":           round(statistics.mean(times), 3),
            "avg_tokens_per_response":        round(statistics.mean(tokens), 2),
        }


    return {
        "per_model":           per_model,
        "batch_wall_time_sec": round(batch_time, 3),
        "requests":            results,
        "gpu_telemetry":       gpu_telemetry,
    }




# ── Main ───────────────────────────────────────────────────────────────────────


async def main():
    monitor = GPUMonitor(interval=GPU_POLL_INTERVAL)
    await warmup_model()


    all_runs = []
    summary  = []


    try:
        for n in range(N_MAX + 1, 0, -5):
            repeat_results = []
            for m in range(1, M + 1):
                print(f"\n[N={n}, repeat={m}/{M}] ", end="")
                batch = await run_batch(n, monitor)
                batch["n"] = n
                batch["repeat"] = m
                all_runs.append(batch)
                repeat_results.append(batch)


            model_summary = {}
            for model in MODELS:
                model_summary[model] = {
                    key: round(statistics.mean(
                        r["per_model"][model][key] for r in repeat_results
                    ), 3)
                    for key in [
                        "batch_tokens_per_sec",
                        "avg_tokens_per_sec_per_request",
                        "avg_time_per_request",
                        "avg_tokens_per_response",
                    ]
                }
            summary.append({"n": n, "per_model": model_summary})
            print(f"\n  → {model_summary}")


    finally:
        monitor.shutdown()


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "models": MODELS,
                "N_MAX": N_MAX,
                "M": M,
                "gpu_poll_interval_sec": GPU_POLL_INTERVAL
            },
            "summary": summary,
            "all_runs": all_runs,
        }, f, indent=2)


    print(f"\nSaved → {json_path}")




if __name__ == "__main__":
    asyncio.run(main())



