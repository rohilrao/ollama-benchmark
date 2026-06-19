"""
ollama_bench.py
================
Class-based refactor of vram_bench.py and latency_bench.py.

- OllamaBenchmarkBase: shared client setup, prompt building, CSV saving, plot saving.
- VRAMBenchmark: measures VRAM usage across concurrency / num_ctx / prompt-padding sweeps.
  Unloads the model before every measurement for a clean read. No warmup (intentionally).
- LatencyBenchmark: measures wall time, TTFT, and tok/s across a concurrency sweep.
  Warms up the model once before sweeping. No unloading (intentionally).

Usage (see bottom of file for a runnable example):

    vram_bench = VRAMBenchmark(host=..., model=..., output_dir="./results/vram")
    rows = await vram_bench.run_all()
    vram_bench.save_results(rows)
    vram_bench.plot_results(rows)

    latency_bench = LatencyBenchmark(host=..., model=..., output_dir="./results/latency")
    summary = await latency_bench.run_all()
    latency_bench.save_results(summary)
    latency_bench.plot_results(summary)
"""

import asyncio
import csv
import os
import statistics
import time
import uuid

import matplotlib.pyplot as plt
import ollama


# ── Base class ────────────────────────────────────────────────────────────
class OllamaBenchmarkBase:
    """Shared plumbing: client, prompt building, CSV/plot saving, logging."""

    PROMPT_BASE = "Write a sentence with each letter of the english alphabet used EXACTLY once:"
    LOREM = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
             "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim ") * 400

    def __init__(self, host: str, model: str, output_dir: str = ".", verbose: bool = True):
        self.host = host
        self.model = model
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = ollama.AsyncClient(host=host)

    def _log(self, msg="", end="\n", flush=False):
        if self.verbose:
            print(msg, end=end, flush=flush)

    def make_prompt(self, pad_words: int = 0) -> str:
        """Unique (cache-busting) prompt, optionally padded with extra words."""
        unique = f"[{uuid.uuid4().hex}]\n"
        padding = " ".join(self.LOREM.split()[:pad_words]) + "\n" if pad_words else ""
        return unique + padding + self.PROMPT_BASE

    def save_csv(self, rows: list, filename: str):
        """Write a list of dicts to <output_dir>/<filename>. Fieldnames = union of keys."""
        if not rows:
            self._log(f"No rows to save for {filename}; skipping.")
            return
        path = os.path.join(self.output_dir, filename)
        fieldnames = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self._log(f"Saved CSV → {path}")

    def _save_plot(self, fig, filename: str):
        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        self._log(f"Saved plot → {path}")


# ── VRAM benchmark ────────────────────────────────────────────────────────
class VRAMBenchmark(OllamaBenchmarkBase):
    """
    Measures VRAM usage for a single model across concurrency, num_ctx, and
    prompt-padding sweeps. Model is unloaded before each measurement; no warmup.
    """

    def __init__(self, host: str, model: str, output_dir: str = ".", verbose: bool = True,
                 n_list: list = None, ctx_list: list = None, pad_list: list = None):
        super().__init__(host, model, output_dir, verbose)
        self.n_list = n_list or [1, 2, 4, 5, 10, 15, 20]
        self.ctx_list = ctx_list or [8192, 16384, 32768, 32768 * 2]
        self.pad_list = pad_list or [0, 2000, 6000, 12000]

    async def ollama_vram_mb(self) -> float:
        """VRAM (MB) Ollama reports for self.model via `ollama ps`; 0 if not loaded."""
        resp = await self.client.ps()
        for m in resp.models:
            if m.model == self.model:
                return m.size_vram / (1024 ** 2)
        return 0.0

    async def unload_model(self):
        await self.client.chat(model=self.model, messages=[], keep_alive=0)
        await asyncio.sleep(2)  # Give the server a moment to release VRAM

    async def run_request(self, i: int, num_ctx: int = None, pad_words: int = 0):
        options = {"temperature": 0.0}
        if num_ctx:
            options["num_ctx"] = num_ctx

        token_index = 0
        async for chunk in await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": self.make_prompt(pad_words)}],
            stream=True,
            options=options,
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                token_index += 1
                self._log(f"[R{i},T{token_index}]: {content.strip()}", end=" | ", flush=True)

    async def measure_batch(self, n: int, num_ctx: int = None, pad_words: int = 0) -> float:
        await self.unload_model()
        await asyncio.gather(*[self.run_request(i, num_ctx, pad_words) for i in range(1, n + 1)])
        self._log("")
        return await self.ollama_vram_mb()

    async def sweep_concurrency(self, num_ctx: int) -> list:
        self._log(f"Concurrency sweep (num_ctx={num_ctx})...")
        rows = []
        for n in self.n_list:
            v = await self.measure_batch(n, num_ctx=num_ctx)
            rows.append({"sweep_type": "concurrency", "n": n, "num_ctx": num_ctx,
                         "pad_words": 0, "vram_mb": v})
            self._log(f"  N={n:2d}  vram={v:.0f}MB")
        return rows

    async def sweep_num_ctx(self, n: int = 1) -> list:
        self._log(f"num_ctx sweep (N={n})...")
        rows = []
        for ctx in self.ctx_list:
            v = await self.measure_batch(n, num_ctx=ctx)
            rows.append({"sweep_type": "num_ctx", "n": n, "num_ctx": ctx,
                         "pad_words": 0, "vram_mb": v})
            self._log(f"  num_ctx={ctx:6d}  vram={v:.0f}MB")
        return rows

    async def sweep_prompt_padding(self, num_ctx: int) -> list:
        self._log(f"Prompt padding sweep (num_ctx={num_ctx})...")
        rows = []
        for pad in self.pad_list:
            v = await self.measure_batch(1, num_ctx=num_ctx, pad_words=pad)
            rows.append({"sweep_type": "padding", "n": 1, "num_ctx": num_ctx,
                         "pad_words": pad, "vram_mb": v})
            self._log(f"  pad_words={pad:6d}  vram={v:.0f}MB")
        return rows

    async def run_all(self) -> list:
        """Runs all six sweeps (matches original main()) and returns one combined list of rows."""
        ctx_min, ctx_max = min(self.ctx_list), max(self.ctx_list)
        rows = []
        rows += await self.sweep_concurrency(ctx_min)
        rows += await self.sweep_concurrency(ctx_max)
        rows += await self.sweep_num_ctx(n=1)
        rows += await self.sweep_num_ctx(n=max(self.n_list))
        rows += await self.sweep_prompt_padding(ctx_min)
        rows += await self.sweep_prompt_padding(ctx_max)
        return rows

    def save_results(self, rows: list, filename: str = "vram_results.csv"):
        self.save_csv(rows, filename)

    def plot_results(self, rows: list, filename: str = "vram_benchmark.png"):
        ctx_min, ctx_max = min(self.ctx_list), max(self.ctx_list)
        n_max = max(self.n_list)

        def filt(sweep_type, **fixed):
            return [r for r in rows if r["sweep_type"] == sweep_type
                     and all(r[k] == v for k, v in fixed.items())]

        conc_min = filt("concurrency", num_ctx=ctx_min)
        conc_max = filt("concurrency", num_ctx=ctx_max)
        ctx_n1 = filt("num_ctx", n=1)
        ctx_nmax = filt("num_ctx", n=n_max)
        pad_min = filt("padding", num_ctx=ctx_min)
        pad_max = filt("padding", num_ctx=ctx_max)

        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        fig.suptitle(f"VRAM Usage — {self.model}")
        axes = axes.flatten()

        axes[0].plot([r["n"] for r in conc_min], [r["vram_mb"] for r in conc_min], marker="o")
        axes[0].set_title(f"VRAM vs. Concurrency (num_ctx={ctx_min})")
        axes[0].set_xlabel("Concurrent requests (N)")

        axes[1].plot([r["n"] for r in conc_max], [r["vram_mb"] for r in conc_max], marker="o")
        axes[1].set_title(f"VRAM vs. Concurrency (num_ctx={ctx_max})")
        axes[1].set_xlabel("Concurrent requests (N)")

        axes[2].plot([r["num_ctx"] for r in ctx_n1], [r["vram_mb"] for r in ctx_n1], marker="o")
        axes[2].set_title("VRAM vs. num_ctx (N=1)")
        axes[2].set_xlabel("num_ctx (tokens)")

        axes[3].plot([r["num_ctx"] for r in ctx_nmax], [r["vram_mb"] for r in ctx_nmax], marker="o")
        axes[3].set_title(f"VRAM vs. num_ctx (N={n_max})")
        axes[3].set_xlabel("num_ctx (tokens)")

        axes[4].plot([r["pad_words"] for r in pad_min], [r["vram_mb"] for r in pad_min], marker="o")
        axes[4].set_title(f"VRAM vs. Prompt Length (num_ctx={ctx_min})")
        axes[4].set_xlabel("Extra input words")

        axes[5].plot([r["pad_words"] for r in pad_max], [r["vram_mb"] for r in pad_max], marker="o")
        axes[5].set_title(f"VRAM vs. Prompt Length (num_ctx={ctx_max})")
        axes[5].set_xlabel("Extra input words")

        for ax in axes:
            ax.set_ylabel("VRAM used (MB)")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

        self._save_plot(fig, filename)


# ── Latency benchmark ─────────────────────────────────────────────────────
class LatencyBenchmark(OllamaBenchmarkBase):
    """
    Measures full request latency for a single model under increasing concurrency.
    Warms up the model once before sweeping; never unloads it.
    """

    def __init__(self, host: str, model: str, output_dir: str = ".", verbose: bool = True,
                 n_list: list = None, m: int = 2):
        super().__init__(host, model, output_dir, verbose)
        self.n_list = n_list or [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.m = m  # repeats per N, averaged for stability

    async def warmup(self, n: int):
        """Send n throwaway requests so the model is loaded and GPU is primed."""
        self._log(f"Warming up '{self.model}' with {n} requests...")
        await asyncio.gather(*[
            self.client.chat(model=self.model, messages=[{"role": "user", "content": "Hi"}])
            for _ in range(n)
        ])
        self._log("Warmup complete.\n")

    async def run_request(self, i: int) -> dict:
        """Run one streaming request, capturing wall time, TTFT, and tok/s."""
        output_tokens = 0
        eval_duration_ns = 0
        ttft = None
        token_index = 0

        start = time.perf_counter()
        async for chunk in await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": f"[BypassCacheID: {uuid.uuid4().hex}]\n{self.PROMPT_BASE}"}],
            stream=True,
            options={"temperature": 0.0},
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                token_index += 1
                self._log(f"[R{i},T{token_index}]: {content.strip()}", end=" | ", flush=True)

            if ttft is None and content:
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
        }

    async def run_batch(self, n: int) -> dict:
        """Fire n concurrent requests, return per-request averages + batch throughput."""
        batch_start = time.perf_counter()
        results = await asyncio.gather(*[self.run_request(i) for i in range(1, n + 1)])
        self._log("")
        batch_time = time.perf_counter() - batch_start

        total_tokens = sum(r["output_tokens"] for r in results)
        return {
            "n": n,
            "wall_time_sec": statistics.mean(r["wall_time_sec"] for r in results),
            "ttft_sec": statistics.mean(r["ttft_sec"] for r in results),
            "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in results),
            "batch_tokens_per_sec": total_tokens / batch_time,
        }

    async def run_sweep(self) -> list:
        summary = []
        for n in self.n_list:
            repeats = [await self.run_batch(n) for _ in range(self.m)]
            avg = {
                "n": n,
                "wall_time_sec": statistics.mean(r["wall_time_sec"] for r in repeats),
                "ttft_sec": statistics.mean(r["ttft_sec"] for r in repeats),
                "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in repeats),
                "batch_tokens_per_sec": statistics.mean(r["batch_tokens_per_sec"] for r in repeats),
            }
            summary.append(avg)
            self._log(f"N={n:2d}  wall={avg['wall_time_sec']:.2f}s  "
                       f"ttft={avg['ttft_sec']:.2f}s  "
                       f"tok/s/req={avg['tokens_per_sec']:.1f}  "
                       f"batch tok/s={avg['batch_tokens_per_sec']:.1f}")
        return summary

    async def run_all(self) -> list:
        """Warms up once, then runs the concurrency sweep. Returns the summary list."""
        await self.warmup(max(self.n_list))
        return await self.run_sweep()

    def save_results(self, summary: list, filename: str = "latency_results.csv"):
        self.save_csv(summary, filename)

    def plot_results(self, summary: list, filename: str = "latency_benchmark.png"):
        n_vals = [s["n"] for s in summary]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(f"Latency vs. Concurrency — {self.model}")

        axes[0, 0].plot(n_vals, [s["wall_time_sec"] for s in summary], marker="o")
        axes[0, 0].set_title("Wall time per request (s)")
        axes[0, 0].set_xlabel("Concurrent requests (N)")

        axes[0, 1].plot(n_vals, [s["ttft_sec"] for s in summary], marker="o", color="orange")
        axes[0, 1].set_title("Time to first token (s)")
        axes[0, 1].set_xlabel("Concurrent requests (N)")

        axes[1, 0].plot(n_vals, [s["tokens_per_sec"] for s in summary], marker="o", color="green")
        axes[1, 0].set_title("Tokens/sec per request")
        axes[1, 0].set_xlabel("Concurrent requests (N)")

        axes[1, 1].plot(n_vals, [s["batch_tokens_per_sec"] for s in summary], marker="o", color="red")
        axes[1, 1].set_title("Batch tokens/sec (aggregate)")
        axes[1, 1].set_xlabel("Concurrent requests (N)")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        self._save_plot(fig, filename)


# ── Example usage ──────────────────────────────────────────────────────────
async def _main():
    vram_bench = VRAMBenchmark(
        host="http://localhost:11441",
        model="mistral-small3.2:24b-32k",
        output_dir="./results/vram",
        verbose=True,
    )
    vram_rows = await vram_bench.run_all()
    vram_bench.save_results(vram_rows)
    vram_bench.plot_results(vram_rows)

    latency_bench = LatencyBenchmark(
        host="http://localhost:11436",
        model="mistral-small3.2:24b-32k",
        output_dir="./results/latency",
        verbose=True,
    )
    latency_summary = await latency_bench.run_all()
    latency_bench.save_results(latency_summary)
    latency_bench.plot_results(latency_summary)


if __name__ == "__main__":
    asyncio.run(_main())
