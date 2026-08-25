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

# ── Latency benchmark ─────────────────────────────────────────────────────
class LatencyBenchmark(OllamaBenchmarkBase):
    """
    Measures full request latency for a single model under increasing concurrency.
    Warms up the model once before sweeping; never unloads it.

    Reasoning models can stream both "thinking" and "content" tokens in the same
    response (msg.get("thinking") / msg.get("content")). Both are streamed and
    counted separately, and included in the latency metrics.

    Failed batches (OOM, runner crash, timeout, connection error) are recorded
    as rows with status != "ok" rather than aborting the sweep.
    """

    def __init__(self, host: str, model: str, output_dir: str = ".", verbose: bool = True,
                 n_list: list = None, m: int = 2,
                 request_timeout: float = 120.0, capture_ps_snapshots: bool = True):
        super().__init__(host, model, output_dir, verbose, request_timeout, capture_ps_snapshots)
        self.n_list = n_list or [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.m = m  # repeats per N, averaged for stability

    async def warmup(self, n: int):
        """Send n throwaway requests so the model is loaded and GPU is primed."""
        self._log(f"Warming up '{self.model}' with {n} requests...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*[
                    self.client.chat(model=self.model, messages=[{"role": "user", "content": "Hi"}])
                    for _ in range(n)
                ]),
                timeout=self.request_timeout,
            )
            self._log("Warmup complete.\n")
        except Exception as e:
            status, err_msg = self._classify_error(e)
            self._log(f"Warmup failed ({status}: {err_msg}); attempting recovery and continuing anyway.\n")
            await self._attempt_recovery()

    async def run_request(self, i: int) -> dict:
        """Run one streaming request, capturing wall time, TTFT, tok/s, and
        thinking/content token counts (reasoning models stream both)."""
        output_tokens = 0
        eval_duration_ns = 0
        ttft = None
        token_index = 0
        thinking_tokens = 0
        content_tokens = 0

        start = time.perf_counter()
        async for chunk in await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": f"[BypassCacheID: {uuid.uuid4().hex}]\n{self.PROMPT_BASE}"}],
            stream=True,
            options={"temperature": 0.0},
        ):
            token_index, got_thinking, got_content = self._stream_tokens(chunk, i, token_index, self._log)
            if got_thinking:
                thinking_tokens += 1
            if got_content:
                content_tokens += 1

            if ttft is None and (got_thinking or got_content):
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
            "thinking_tokens": thinking_tokens,
            "content_tokens": content_tokens,
        }

    async def run_batch(self, n: int) -> dict:
        """
        Fire n concurrent requests. Returns a dict with status="ok" and averaged
        metrics on success, or status != "ok" with error_message on failure.
        Never raises.
        """
        batch_start = time.perf_counter()
        raw_results = await asyncio.gather(
            *[asyncio.wait_for(self.run_request(i), timeout=self.request_timeout) for i in range(1, n + 1)],
            return_exceptions=True,
        )
        self._log("")
        batch_time = time.perf_counter() - batch_start

        errors = [r for r in raw_results if isinstance(r, Exception)]
        if errors:
            status, err_msg = self._classify_error(errors[0])
            self._log(f"  ✗ FAILED batch n={n}: {status} — {err_msg}")
            row = {"n": n, "status": status, "error_message": err_msg, "elapsed_time_sec": batch_time,
                   "wall_time_sec": None, "ttft_sec": None, "tokens_per_sec": None,
                   "batch_tokens_per_sec": None, "thinking_tokens": None, "content_tokens": None}
            if self.capture_ps_snapshots:
                row["ps_after"] = await self._ps_snapshot()
            await self._attempt_recovery()
            return row

        results = raw_results
        total_tokens = sum(r["output_tokens"] for r in results)
        return {
            "n": n, "status": "ok", "error_message": None, "elapsed_time_sec": batch_time,
            "wall_time_sec": statistics.mean(r["wall_time_sec"] for r in results),
            "ttft_sec": statistics.mean(r["ttft_sec"] for r in results),
            "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in results),
            "batch_tokens_per_sec": total_tokens / batch_time,
            "thinking_tokens": statistics.mean(r["thinking_tokens"] for r in results),
            "content_tokens": statistics.mean(r["content_tokens"] for r in results),
        }

    async def run_sweep(self) -> list:
        """
        Sweeps n_list, repeating each N self.m times and averaging over the
        successful repeats. If every repeat at a given N fails, records a
        single failed row for that N and moves on — never raises.
        """
        summary = []
        for n in self.n_list:
            repeats = [await self.run_batch(n) for _ in range(self.m)]
            successes = [r for r in repeats if r["status"] == "ok"]
            failures = [r for r in repeats if r["status"] != "ok"]

            if not successes:
                last_failure = failures[-1]
                summary.append({
                    "model": self.model, "n": n,
                    "status": last_failure["status"], "error_message": last_failure["error_message"],
                    "elapsed_time_sec": last_failure["elapsed_time_sec"],
                    "wall_time_sec": None, "ttft_sec": None, "tokens_per_sec": None,
                    "batch_tokens_per_sec": None, "thinking_tokens": None, "content_tokens": None,
                    "n_successful_repeats": 0, "n_failed_repeats": len(failures),
                })
                self._log(f"N={n:2d}  ALL {len(failures)} REPEAT(S) FAILED  status={last_failure['status']}")
                continue

            avg = {
                "model": self.model, "n": n,
                "status": "ok" if not failures else "partial_failure",
                "error_message": None if not failures else failures[-1]["error_message"],
                "elapsed_time_sec": statistics.mean(r["elapsed_time_sec"] for r in successes),
                "wall_time_sec": statistics.mean(r["wall_time_sec"] for r in successes),
                "ttft_sec": statistics.mean(r["ttft_sec"] for r in successes),
                "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in successes),
                "batch_tokens_per_sec": statistics.mean(r["batch_tokens_per_sec"] for r in successes),
                "thinking_tokens": statistics.mean(r["thinking_tokens"] for r in successes),
                "content_tokens": statistics.mean(r["content_tokens"] for r in successes),
                "n_successful_repeats": len(successes),
                "n_failed_repeats": len(failures),
            }
            summary.append(avg)
            self._log(f"N={n:2d}  wall={avg['wall_time_sec']:.2f}s  "
                       f"ttft={avg['ttft_sec']:.2f}s  "
                       f"tok/s/req={avg['tokens_per_sec']:.1f}  "
                       f"batch tok/s={avg['batch_tokens_per_sec']:.1f}  "
                       f"thinking_tok={avg['thinking_tokens']:.1f}  "
                       f"content_tok={avg['content_tokens']:.1f}  "
                       f"({avg['n_successful_repeats']}/{self.m} repeats ok)")
        return summary

    async def run_all(self) -> list:
        """Warms up once, then runs the concurrency sweep. Returns the summary list."""
        await self.warmup(max(self.n_list))
        return await self.run_sweep()

    def save_results(self, summary: list, filename: str = "latency_results.csv"):
        self.save_csv(summary, filename)

    def plot_results(self, summary: list, filename: str = "latency_benchmark.png"):
        """Plots successful (or partially successful) rows only; rows where
        every repeat failed (status not in {"ok", "partial_failure"}) are skipped."""
        ok_rows = [s for s in summary if s.get("status") in ("ok", "partial_failure")]
        n_skipped = len(summary) - len(ok_rows)
        if n_skipped:
            self._log(f"Skipping {n_skipped} fully-failed N value(s) when plotting.")
        if not ok_rows:
            self._log("No successful measurements to plot.")
            return

        n_vals = [s["n"] for s in ok_rows]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(f"Latency vs. Concurrency — {self.model}")

        axes[0, 0].plot(n_vals, [s["wall_time_sec"] for s in ok_rows], marker="o")
        axes[0, 0].set_title("Wall time per request (s)")
        axes[0, 0].set_xlabel("Concurrent requests (N)")

        axes[0, 1].plot(n_vals, [s["ttft_sec"] for s in ok_rows], marker="o", color="orange")
        axes[0, 1].set_title("Time to first token (s)")
        axes[0, 1].set_xlabel("Concurrent requests (N)")

        axes[1, 0].plot(n_vals, [s["tokens_per_sec"] for s in ok_rows], marker="o", color="green")
        axes[1, 0].set_title("Tokens/sec per request")
        axes[1, 0].set_xlabel("Concurrent requests (N)")

        axes[1, 1].plot(n_vals, [s["batch_tokens_per_sec"] for s in ok_rows], marker="o", color="red")
        axes[1, 1].set_title("Batch tokens/sec (aggregate)")
        axes[1, 1].set_xlabel("Concurrent requests (N)")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        self._save_plot(fig, filename)

