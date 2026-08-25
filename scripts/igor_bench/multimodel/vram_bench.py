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

class VRAMBenchmark(OllamaBenchmarkBase):
    """
    Measures VRAM usage for a single model across a full grid search of
    (n, num_ctx, pad_words). Model is unloaded before each measurement; no warmup.

    Failed configurations (OOM, runner crash, timeout, connection error) are
    recorded as rows with status != "ok" rather than aborting the grid search.
    """

    def __init__(self, host: str, model: str, output_dir: str = ".", verbose: bool = True,
                 n_list: list = None, ctx_list: list = None, pad_list: list = None,
                 request_timeout: float = 240.0, capture_ps_snapshots: bool = True):
        super().__init__(host, model, output_dir, verbose, request_timeout, capture_ps_snapshots)
        self.n_list = n_list or [1, 2, 4, 5, 10, 15, 20]
        self.ctx_list = ctx_list or [8192, 16384, 32768, 32768 * 2]
        self.pad_list = pad_list or [0, 2000, 6000, 12000]

    async def ollama_vram_mb(self) -> float:
        """VRAM (MB) Ollama reports for self.model via `ollama ps`; 0 if not loaded."""
        breakdown = await self.ollama_memory_breakdown()
        return breakdown["vram_mb"]

    async def ollama_memory_breakdown(self) -> dict:
        """
        Reads `ollama ps` and reports not just VRAM used, but how it splits against
        the model's total memory footprint — i.e. whether the model is fully on GPU
        or partially offloaded to CPU (the same `size` vs `size_vram` fields the
        `ollama ps` CLI uses to print "100% GPU" / "43%/57% CPU/GPU").

        Returns a dict: {vram_mb, total_mb, gpu_pct, cpu_pct, fully_on_gpu}.
        If the model isn't currently loaded, all fields are 0/None as appropriate.
        """
        resp = await self.client.ps()
        for m in resp.models:
            if m.model == self.model:
                size_vram = getattr(m, "size_vram", 0) or 0
                size_total = getattr(m, "size", 0) or 0
                vram_mb = size_vram / (1024 ** 2)
                total_mb = size_total / (1024 ** 2)
                if size_total > 0:
                    gpu_pct = round(100 * size_vram / size_total, 1)
                    cpu_pct = round(100 - gpu_pct, 1)
                    fully_on_gpu = gpu_pct >= 99.95  # allow tiny rounding slack
                else:
                    gpu_pct = cpu_pct = fully_on_gpu = None
                return {"vram_mb": vram_mb, "total_mb": total_mb, "gpu_pct": gpu_pct,
                        "cpu_pct": cpu_pct, "fully_on_gpu": fully_on_gpu}
        return {"vram_mb": 0.0, "total_mb": 0.0, "gpu_pct": None, "cpu_pct": None, "fully_on_gpu": None}

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
            token_index, _, _ = self._stream_tokens(chunk, i, token_index, self._log)

    async def measure_batch(self, n: int, num_ctx: int = None, pad_words: int = 0) -> dict:
        """
        Returns a dict: {"status", "error_message", "elapsed_time_sec", "vram_mb",
        "total_mb", "gpu_pct", "cpu_pct", "fully_on_gpu", and optionally
        "ps_before"/"ps_after" on failure}. Never raises.
        """
        batch_start = time.perf_counter()

        try:
            await asyncio.wait_for(self.unload_model(), timeout=self.request_timeout)
        except Exception as e:
            status, err_msg = self._classify_error(e)
            elapsed = time.perf_counter() - batch_start
            self._log(f"  ✗ Unload failed before n={n} num_ctx={num_ctx} pad_words={pad_words}: "
                       f"{status} — {err_msg}")
            await self._attempt_recovery()
            return {"status": status, "error_message": f"unload_failed: {err_msg}",
                    "vram_mb": None, "elapsed_time_sec": elapsed}

        results = await asyncio.gather(
            *[asyncio.wait_for(self.run_request(i, num_ctx, pad_words), timeout=self.request_timeout)
              for i in range(1, n + 1)],
            return_exceptions=True,
        )
        self._log("")
        elapsed = time.perf_counter() - batch_start

        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            status, err_msg = self._classify_error(errors[0])
            self._log(f"  ✗ FAILED n={n} num_ctx={num_ctx} pad_words={pad_words}: {status} — {err_msg}")
            row = {"status": status, "error_message": err_msg, "vram_mb": None, "elapsed_time_sec": elapsed}
            if self.capture_ps_snapshots:
                row["ps_after"] = await self._ps_snapshot()
            await self._attempt_recovery()
            return row

        try:
            breakdown = await asyncio.wait_for(self.ollama_memory_breakdown(), timeout=30)
            if breakdown["fully_on_gpu"] is False:
                self._log(f"  ⚠ Partial CPU offload detected: gpu={breakdown['gpu_pct']}% "
                           f"cpu={breakdown['cpu_pct']}% (n={n} num_ctx={num_ctx} pad_words={pad_words})")
            return {"status": "ok", "error_message": None, "elapsed_time_sec": elapsed, **breakdown}
        except Exception as e:
            status, err_msg = self._classify_error(e)
            self._log(f"  ✗ VRAM read failed n={n} num_ctx={num_ctx} pad_words={pad_words}: "
                       f"{status} — {err_msg}")
            return {"status": "ps_read_error", "error_message": err_msg,
                    "vram_mb": None, "total_mb": None, "gpu_pct": None, "cpu_pct": None,
                    "fully_on_gpu": None, "elapsed_time_sec": elapsed}

    async def run_grid_search(self) -> list:
        """
        Full grid search over n_list x ctx_list x pad_list. Returns a list of dicts:
        {model, n, num_ctx, pad_words, status, error_message, vram_mb, elapsed_time_sec, ...}.
        Total measurements = len(n_list) * len(ctx_list) * len(pad_list); each one does a
        full unload + full generation, so size your lists accordingly before running a
        large grid unattended. Failed configurations do not stop the grid search.
        """
        rows = []
        combos = list(itertools.product(self.n_list, self.ctx_list, self.pad_list))
        self._log(f"Grid search: {len(combos)} combinations "
                   f"({len(self.n_list)} x {len(self.ctx_list)} x {len(self.pad_list)})...")
        for n, ctx, pad in combos:
            result = await self.measure_batch(n, num_ctx=ctx, pad_words=pad)
            row = {"model": self.model, "n": n, "num_ctx": ctx, "pad_words": pad, **result}
            rows.append(row)
            if result["status"] == "ok":
                self._log(f"  n={n:2d}  num_ctx={ctx:6d}  pad_words={pad:6d}  vram={result['vram_mb']:.0f}MB")
        n_failed = sum(1 for r in rows if r["status"] != "ok")
        if n_failed:
            self._log(f"Grid search done: {n_failed}/{len(rows)} configurations failed.")
        return rows

    async def run_all(self) -> list:
        """Convenience alias for run_grid_search()."""
        return await self.run_grid_search()

    def save_results(self, rows: list, filename: str = "vram_results.csv"):
        self.save_csv(rows, filename)

    def plot_results(self, rows: list, filename: str = "vram_benchmark.png"):
        """One line per (num_ctx, pad_words) combination, VRAM vs N on the x-axis.
        Rows with status != "ok" (no vram_mb reading) are excluded."""
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        n_skipped = len(rows) - len(ok_rows)
        if n_skipped:
            self._log(f"Skipping {n_skipped} failed row(s) when plotting.")
        if not ok_rows:
            self._log("No successful measurements to plot.")
            return

        combos = sorted(set((r["num_ctx"], r["pad_words"]) for r in ok_rows))

        fig, ax = plt.subplots(figsize=(10, 6))
        for ctx, pad in combos:
            sub = sorted([r for r in ok_rows if r["num_ctx"] == ctx and r["pad_words"] == pad],
                         key=lambda r: r["n"])
            ax.plot([r["n"] for r in sub], [r["vram_mb"] for r in sub],
                    marker="o", label=f"num_ctx={ctx}, pad_words={pad}")

        ax.set_title(f"VRAM Usage — {self.model}")
        ax.set_xlabel("Concurrent requests (N)")
        ax.set_ylabel("VRAM used (MB)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        self._save_plot(fig, filename)
