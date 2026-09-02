"""
Fires N concurrent streamed chat completions and times each one.

ttft_thinking / ttft_content are measured client-side by watching for the
first chunk with delta.reasoning_content vs delta.content, since vLLM's
/metrics only exposes an aggregate TTFT histogram with no thinking/content
split. This requires the server to have reasoning parsing enabled, and for
Qwen3 specifically requires enable_thinking (see config.ENABLE_THINKING) -
without it, reasoning_content never appears and ttft_thinking is just NaN.
"""

import asyncio
import time
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

import server_metrics


@dataclass
class RequestResult:
    req_id: int
    start: float
    input_tokens_target: int
    ok: bool = False
    error: str = ""
    first_thinking: float = None
    first_content: float = None
    end: float = 0.0
    thinking_chunks: int = 0
    content_chunks: int = 0
    output_tokens: int = 0


async def stream_request(client: AsyncOpenAI, req_id: int, prompt: str,
                          input_tokens_target: int, max_tokens: int,
                          model: str, temperature: float,
                          enable_thinking: bool) -> RequestResult:
    res = RequestResult(req_id=req_id, start=time.perf_counter(),
                         input_tokens_target=input_tokens_target)
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_options={"include_usage": True},
        )
        if enable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        response = await client.chat.completions.create(**kwargs)

        async for chunk in response:
            now = time.perf_counter()

            usage = getattr(chunk, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None):
                res.output_tokens = usage.completion_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            content = getattr(delta, "content", None)

            if reasoning:
                if res.first_thinking is None:
                    res.first_thinking = now
                res.thinking_chunks += 1
            if content:
                if res.first_content is None:
                    res.first_content = now
                res.content_chunks += 1

        res.end = time.perf_counter()
        if not res.output_tokens:
            res.output_tokens = res.thinking_chunks + res.content_chunks
        res.ok = True

    except Exception as e:
        res.end = time.perf_counter()
        res.error = str(e)

    return res


async def run_batch(client: AsyncOpenAI, http: httpx.AsyncClient,
                     prompts: list, input_tokens_target: int, max_tokens: int,
                     model: str, temperature: float, enable_thinking: bool,
                     metrics_url: str):
    """Runs one batch of len(prompts) concurrent requests and snapshots
    NVML VRAM + vLLM /metrics immediately before and after."""
    vram_before = await server_metrics.read_vram_gb()
    metrics_before = await server_metrics.fetch_metrics(http, metrics_url)

    batch_start = time.perf_counter()
    tasks = [
        stream_request(client, i + 1, p, input_tokens_target, max_tokens,
                        model, temperature, enable_thinking)
        for i, p in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
    batch_end = time.perf_counter()

    vram_after = await server_metrics.read_vram_gb()
    metrics_after = await server_metrics.fetch_metrics(http, metrics_url)

    vram_gb = _max_ignoring_nan(vram_before, vram_after)
    return results, batch_start, batch_end, vram_gb, metrics_before, metrics_after


def _max_ignoring_nan(*vals):
    good = [v for v in vals if v == v]
    return max(good) if good else float("nan")
