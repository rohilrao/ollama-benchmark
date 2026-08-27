"""Single-model vLLM benchmark: deepseek-r1 tensor-parallel across GPUs 0-3.

Use this when one vLLM podman container is sharding a single model across
several GPUs (e.g. `--tensor-parallel-size 4`), rather than several
independent containers each on their own GPU. There's only one base_url to
hit either way -- the only difference from the multi-endpoint case is that
gpu_index is a *list*, so VRAM is summed/reported across all of them instead
of looked up per model.

Run it:  uv run run_vllm_bench_single.py
"""

import asyncio

from multi_endpoint_vllm_bench import MultiEndpointVLLMBenchmark


async def main(
    model: str = "/models/deepseek-r1",     # id exactly as the vLLM server serves it
    base_url: str = "http://localhost:8555/v1",
    gpu_indices: list = (0, 1, 2, 3),       # nvidia-smi indices this container's TP ranks sit on
    n_list: list = (1, 2, 4, 8, 16),        # concurrency sweep
    pad_list: list = (0,),                  # input-length sweep (filler words prepended)
    max_tokens_list: list = (256,),         # output-length sweep
    output_dir: str = ".",                  # results.csv and plots land directly here
    reps: int = 1,                          # repeats per grid point; each rep is its own CSV row
    request_timeout: float = 240.0,         # seconds allowed for one generation request
    log_tokens: bool = False,               # echo every streamed token (unreadable above n≈4)
    vram_sample_interval: float = 2.0,      # how often to poll nvidia-smi while generating
):
    bench = MultiEndpointVLLMBenchmark(
        model_specs=[{
            "model": model,
            "base_url": base_url,
            "gpu_index": list(gpu_indices),   # list -> summed VRAM across all TP ranks
            "n_list": list(n_list),
            "pad_list": list(pad_list),
            "max_tokens_list": list(max_tokens_list),
        }],
        output_dir=output_dir,
        reps=reps,
        request_timeout=request_timeout,
        log_tokens=log_tokens,
        vram_sample_interval=vram_sample_interval,
        verbose=True,
    )
    try:
        rows = await bench.run_grid_search()
        bench.save_results(rows)   # -> <output_dir>/results.csv
        bench.plot_all(rows)       # -> <output_dir>/vram.png, latency_<metric>.png
        return rows
    finally:
        await bench.cleanup()


if __name__ == "__main__":
    asyncio.run(main(
        model="/models/deepseek-r1",
        base_url="http://localhost:8555/v1",
        gpu_indices=[0, 1, 2, 3],
        n_list=[1, 2, 4, 8, 16],
        pad_list=[0],
        max_tokens_list=[256],
        output_dir=".",
        reps=1,
    ))
