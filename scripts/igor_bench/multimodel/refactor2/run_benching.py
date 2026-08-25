"""Runner for the Ollama co-residency benchmark.

One entry point, any number of models. One model in `model_specs` gives the
single-model VRAM + latency run; two or more are loaded and driven at once.
Edit the call at the bottom of this file and run it:  uv run run_benching.py
"""

import asyncio

from multi_model_bench import MultiModelBenchmark


async def main(
    host: str = "http://localhost:11434",   # Ollama server URL
    model_specs: list = None,               # one dict per model; see the call below
    output_dir: str = ".",                  # results land in <output_dir>/results/<models>/
    reps: int = 1,                          # repeats per grid point; each rep is its own CSV row
    request_timeout: float = 240.0,         # seconds allowed for one generation request
    load_timeout: float = 900.0,            # seconds allowed to cold-load one model
    log_tokens: bool = False,               # echo every streamed token (unreadable above n≈4)
    vram_sample_interval: float = 2.0,      # how often to poll `ollama ps` while generating
):
    bench = MultiModelBenchmark(
        host=host,
        model_specs=model_specs,
        output_dir=output_dir,
        reps=reps,
        request_timeout=request_timeout,
        load_timeout=load_timeout,
        log_tokens=log_tokens,
        vram_sample_interval=vram_sample_interval,
        verbose=True,
    )
    try:
        rows = await bench.run_grid_search()
        bench.save_results(rows)   # -> <output_dir>/results/<models>/results.csv
        bench.plot_all(rows)       # -> vram.png, latency_<metric>.png
        return rows
    finally:
        # Runs on success, on error, and on Ctrl-C — never leave models pinned
        # in VRAM for the rest of keep_alive.
        await bench.cleanup()


if __name__ == "__main__":
    asyncio.run(main(
        host="http://localhost:11443",
        model_specs=[
            # model:    tag exactly as `ollama list` prints it
            # n_list:   concurrent requests sent to THIS model at each grid point
            # ctx_list: num_ctx values to sweep (None = whatever the server defaults to)
            # pad_list: filler words prepended to the prompt, i.e. an input-length sweep
            {"model": "qwen3-235b-a22b:q4_k_m", "n_list": [10], "ctx_list": [32768, 65536], "pad_list": [0]},
            {"model": "qwen3:32b-q8",           "n_list": [10], "ctx_list": [32768, 65536], "pad_list": [0]},
        ],
        output_dir=".",
        reps=1,
    ))
