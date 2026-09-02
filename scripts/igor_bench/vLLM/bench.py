"""
Grid sweep entrypoint. For every (input_tokens_target, concurrency) pair in
config.py, runs config.REPS repetitions and writes one aggregated row to
config.OUT_FILE - so you can see how TTFT/throughput/KV-cache pressure move
as prompts get longer and load gets heavier, not just at one fixed shape.

Run with: python bench.py
"""

import asyncio
import csv
import itertools

import httpx
from openai import AsyncOpenAI

import config
import prompts
import runner
import aggregate
import server_metrics


async def run_grid_point(client, http, ctx_reported, input_tokens_target, concurrency):
    prompt = prompts.build_prompt(input_tokens_target, config.WORDS_PER_TOKEN)
    actual_tokens = await prompts.measure_input_tokens(
        http, config.TOKENIZE_URL, config.MODEL_NAME, prompt
    )
    prompt_batch = [prompt] * concurrency

    rep_rows = []
    for rep in range(config.REPS):
        print(f"--- input~{input_tokens_target}tok (actual={actual_tokens}) "
              f"concurrency={concurrency} rep {rep + 1}/{config.REPS} ---")
        results, b_start, b_end, vram_gb, m_before, m_after = await runner.run_batch(
            client, http, prompt_batch, input_tokens_target, config.MAX_TOKENS,
            config.MODEL_NAME, config.TEMPERATURE, config.ENABLE_THINKING,
            config.METRICS_URL,
        )
        row = aggregate.summarize_rep(
            results, b_start, b_end, vram_gb, m_before, m_after,
            ctx_reported, input_tokens_target, concurrency,
        )
        row["input_tokens_actual"] = actual_tokens
        print({k: row[k] for k in (
            "wall_time_sec", "ttft_content_sec", "batch_tokens_per_sec",
            "kv_cache_usage_pct", "requests_failed",
        )})
        rep_rows.append(row)

    agg = aggregate.aggregate_reps(rep_rows)
    agg["input_tokens_actual"] = actual_tokens
    return agg


async def main():
    client = AsyncOpenAI(base_url=config.BASE_URL, api_key="EMPTY")

    async with httpx.AsyncClient() as http:
        ctx_reported = await server_metrics.get_ctx_reported(http, config.MODELS_URL)

        grid = list(itertools.product(config.INPUT_TOKEN_TARGETS, config.CONCURRENCY_LEVELS))
        all_rows = []
        for input_tokens_target, concurrency in grid:
            agg = await run_grid_point(client, http, ctx_reported, input_tokens_target, concurrency)
            all_rows.append(agg)

    fieldnames = sorted({k for row in all_rows for k in row})
    with open(config.OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} grid points to {config.OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
