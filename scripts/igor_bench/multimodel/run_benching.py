import asyncio
import csv
import itertools
import json
import os
import statistics
import time
import uuid

import matplotlib.pyplot as plt
import ollama
from ollama_bench import OllamaBenchmarkBase
from vram_bench import VRAMBenchmark
from latency_bench import LatencyBenchmark
from multi_model_bench import MultiModelBenchmark

# ── Example usage ──────────────────────────────────────────────────────────
async def _main(output_dir: str = "."):
    # VRAM benchmark: full grid search over n x num_ctx x pad_words.
    # NOTE: total measurements = len(n_list) * len(ctx_list) * len(pad_list),
    # each doing a full model unload + generation — keep lists small for a
    # quick smoke test, then widen them for the real run. Failed configs
    # (OOM, runner crash, timeout) are recorded, not fatal.
    #
    # `output_dir` is the BASE directory (e.g. "." or wherever the user
    # points it, via --output-dir below); vram/ and latency/ subfolders are
    # created under it per run.

    HOST="http://localhost:11458"
    MODEL="qwen3-235b-a22b:q4_k_m" #"mistral-small3.2:24b"
    N_LIST=[40]
    CTX_LIST=[32768, 65536, 131072]
    PAD_LIST=[0]

    run_dir = os.path.join(output_dir, "qwen3_235b_np40_mlm3")
    OUTPUT_DIR_VRAM = os.path.join(run_dir, "vram")
    OUTPUT_DIR_LATENCY = os.path.join(run_dir, "latency")
    REQUEST_TIMEOUT=240.0

    NUM_REPS = 2 #m

    vram_bench = VRAMBenchmark(
        host=HOST,
        model=MODEL,
        output_dir=OUTPUT_DIR_VRAM,
        verbose=True,
        n_list=N_LIST,
        ctx_list=CTX_LIST,
        pad_list=PAD_LIST,
        request_timeout=REQUEST_TIMEOUT,
    )
    vram_rows = await vram_bench.run_grid_search()
    vram_bench.save_results(vram_rows)
    vram_bench.plot_results(vram_rows)

    # Latency benchmark: concurrency sweep, with thinking+content token tracking
    # for reasoning models (e.g. anything that streams msg["thinking"]).
    latency_bench = LatencyBenchmark(
        host=HOST,
        model=MODEL,
        output_dir=OUTPUT_DIR_LATENCY,
        verbose=True,
        n_list=N_LIST,
        m=NUM_REPS,
        request_timeout=REQUEST_TIMEOUT,
    )
    latency_summary = await latency_bench.run_all()
    latency_bench.save_results(latency_summary)
    latency_bench.plot_results(latency_summary)


async def _main_multi_model(output_dir: str = "."):
    # Combined VRAM + latency benchmark: both models loaded and generating
    # simultaneously, full independent (n, num_ctx) grid per model, cross-
    # producted together. Generalizes to 3+ models by adding more specs —
    # just watch the combination count (logged up front) since it multiplies
    # across every model's grid.
    #
    # `output_dir` is the BASE directory (e.g. "." or wherever the user
    # points it, via --output-dir below). Actual results land in
    # <output_dir>/results/<model1>_<model2>_.../ — that subfolder is
    # derived from the model names automatically, so different model
    # combinations never collide or overwrite each other's CSV/plots.

    HOST = "http://localhost:11443"
    REQUEST_TIMEOUT = 240.0

    MODEL_SPECS = [
        {
            "model": "qwen3-235b-a22b:q4_k_m",
            "n_list": [10],
            "ctx_list": [32768, 65536],
            "pad_list": [0],
        },
        {
            "model": "qwen3:32b-q8",                
            "n_list": [10],
            "ctx_list": [32768, 65536],
            "pad_list": [0],
        },
    ]

    bench = MultiModelBenchmark(
        host=HOST,
        model_specs=MODEL_SPECS,
        output_dir=output_dir,
        verbose=True,
        request_timeout=REQUEST_TIMEOUT,
    )
    rows = await bench.run_grid_search()
    bench.save_results(rows)          # -> <output_dir>/results/<models>/results.csv
    bench.plot_results(rows)          # -> .../vram.png
    bench.plot_latency(rows, metric="ttft_sec")            # -> .../latency_ttft_sec.png
    bench.plot_latency(rows, metric="tokens_per_sec")       # -> .../latency_tokens_per_sec.png


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ollama VRAM/latency benchmarks")
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                         help="'single' runs the single-model VRAM+latency benchmark; "
                              "'multi' runs the combined multi-model benchmark.")
    parser.add_argument("--output-dir", default=".",
                         help="Base directory for results (default: current directory). "
                              "For --mode multi, results are written under "
                              "<output-dir>/results/<model1>_<model2>_.../")
    args = parser.parse_args()

    if args.mode == "multi":
        asyncio.run(_main_multi_model(output_dir=args.output_dir))
    else:
        asyncio.run(_main(output_dir=args.output_dir))

# uv run ./scripts/ollama/benchmarking/multimodel/run_benching.py --mode "multi" --output-dir "/home/rrao/projectcode/inference-bench/inference-bench/scripts/ollama/benchmarking/multi_model"