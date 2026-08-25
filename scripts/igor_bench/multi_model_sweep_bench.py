"""
ollama_bench.py
================
Class-based refactor of vram_bench.py and latency_bench.py.

- OllamaBenchmarkBase: shared client setup, prompt building, CSV/plot saving,
  error classification, and crash recovery.
- VRAMBenchmark: full grid search over (n, num_ctx, pad_words). Model is unloaded
  before every measurement for a clean read. No warmup (intentionally).
- LatencyBenchmark: sweeps concurrency (n), warms up the model once beforehand.
  Streams both "thinking" and "content" tokens (reasoning models emit both),
  and reports them separately in the latency metrics.

Both classes treat OOM / runner-crash / timeout / connection failures as valid
*failed measurement points* rather than letting the whole sweep crash: each
failure is logged, recorded as a row with status/error_message/elapsed_time,
the client attempts a best-effort recovery, and the sweep moves on to the next
configuration.

See the bottom of this file for a runnable example with explicit grid/sweep
parameters.
"""

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


# ── Base class ────────────────────────────────────────────────────────────
class OllamaBenchmarkBase:
    """Shared plumbing: client, prompt building, CSV/plot saving, logging,
    and error classification / recovery for crashed or OOM'd runners."""

    PROMPT_BASE = "Explain KV Cache in one sentence."
    LOREM = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
             "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim ") * 400

    # Substrings (checked case-insensitively) that indicate the llama runner
    # backend crashed or ran out of memory, as opposed to a generic HTTP error.
    OOM_SIGNATURES = (
        "llama runner process has terminated",
        "out of memory",
        "cudamalloc",
        "failed to allocate",
        "llama_new_context_with_model failed",
    )

    def __init__(self, host: str, model: str, output_dir: str = ".", verbose: bool = True,
                 request_timeout: float = 120.0, capture_ps_snapshots: bool = True):
        self.host = host
        self.model = model
        self.output_dir = output_dir
        self.verbose = verbose
        self.request_timeout = request_timeout
        self.capture_ps_snapshots = capture_ps_snapshots
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

    @staticmethod
    def _stream_tokens(chunk: dict, i: int, token_index: int, log_fn) -> tuple:
        """
        Reads a streamed chat chunk and logs both reasoning ("thinking") and
        regular ("content") tokens, since reasoning models can emit both in
        the same stream. Returns (token_index, got_thinking, got_content).
        """
        message = chunk.get("message", {})
        thinking = message.get("thinking", "")
        content = message.get("content", "")

        got_thinking = bool(thinking)
        got_content = bool(content)

        if got_thinking:
            token_index += 1
            log_fn(f"[R{i},T{token_index},thinking]: {thinking.strip()}", end=" | ", flush=True)
        if got_content:
            token_index += 1
            log_fn(f"[R{i},T{token_index},content]: {content.strip()}", end=" | ", flush=True)

        return token_index, got_thinking, got_content

    def _classify_error(self, exc: Exception) -> tuple:
        """
        Maps an exception to (status, error_message). Status is one of:
        'timeout', 'oom_or_runner_crash', 'connection_error',
        'server_error_500', or 'unknown_error'. Never raises.
        """
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout", f"Request exceeded timeout of {self.request_timeout}s"

        message = str(exc)
        # ollama.ResponseError (and similar) often carry the real server message
        # in an `.error` attribute distinct from the generic Python str(exc).
        error_attr = getattr(exc, "error", None)
        if error_attr:
            message = f"{message} | {error_attr}"

        lowered = message.lower()
        if any(sig in lowered for sig in self.OOM_SIGNATURES):
            return "oom_or_runner_crash", message

        if "connection" in lowered or isinstance(exc, ConnectionError):
            return "connection_error", message

        status_code = getattr(exc, "status_code", None)
        if status_code == 500:
            return "server_error_500", message

        return "unknown_error", message

    async def _ps_snapshot(self) -> str:
        """Best-effort `ollama ps` dump as a JSON string; never raises."""
        try:
            resp = await self.client.ps()
            models = [{"model": m.model, "size_vram_mb": round(m.size_vram / (1024 ** 2), 1)}
                      for m in resp.models]
            return json.dumps(models)
        except Exception as e:
            return f"ps_snapshot_failed: {e}"

    async def _attempt_recovery(self, extra_sleep: float = 3.0):
        """
        Best-effort recovery after a failed batch: give the server a moment,
        then ping the model so the next configuration starts from a known
        state. Recovery failures are logged but never raised — the sweep
        always continues to the next configuration regardless.
        """
        self._log(f"  Attempting recovery (sleeping {extra_sleep}s, then pinging model)...")
        await asyncio.sleep(extra_sleep)
        try:
            await asyncio.wait_for(
                self.client.chat(model=self.model, messages=[{"role": "user", "content": "ping"}],
                                  keep_alive=0),
                timeout=30,
            )
            self._log("  Recovery ping succeeded — runner appears to be back.")
        except Exception as e:
            self._log(f"  Recovery ping failed too ({e}); continuing to next configuration anyway.")

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


# ── Multi-model benchmark (combined VRAM + latency) ─────────────────────────
class MultiModelBenchmark(OllamaBenchmarkBase):
    """
    Measures combined VRAM usage AND per-model latency when two or more
    models are loaded and generating *simultaneously* on the same GPU.
    Generalizes VRAMBenchmark/LatencyBenchmark from a single (n, num_ctx,
    pad_words) grid to N independent per-model grids, cross-producted
    together.

    Each grid point: unload every model -> fire concurrent load at every
    model at once (one gather() per model, so failures are attributable to
    a specific model) -> read `ollama ps` once for combined VRAM, and
    average each model's own per-request latency (wall time, TTFT, tok/s)
    over its own concurrent requests.

    model_specs: list of dicts, one per model:
        {"model": str, "n_list": list[int], "ctx_list": list[int|None],
         "pad_list": list[int]}
    "ctx_list"/"pad_list" default to [None]/[0] per model if omitted, so
    adding a model doesn't silently multiply grid size unless you ask for it.

    Total grid points = product of len(n_list)*len(ctx_list)*len(pad_list)
    across ALL models — this grows fast with more models/dimensions, so
    run_grid_search() logs the total up front and warns above 200 points.

    Failed grid points (OOM/crash/timeout on any model) are recorded, not
    fatal — matching the recovery behavior of the rest of this file.

    output_dir is the user-provided BASE directory (default "."). Results
    are actually written to <output_dir>/results/<model1>_<model2>_.../ —
    that subfolder is computed automatically from model_specs and created
    for you, so runs against different model combinations never collide or
    overwrite each other's CSV/plots.
    """

    def __init__(self, host: str, model_specs: list, output_dir: str = ".", verbose: bool = True,
                 request_timeout: float = 240.0, capture_ps_snapshots: bool = True):
        label = "+".join(spec["model"] for spec in model_specs)
        results_dir = self._results_subdir(output_dir, model_specs)
        super().__init__(host, label, results_dir, verbose, request_timeout, capture_ps_snapshots)
        self.model_specs = model_specs
        self.base_output_dir = output_dir

    @staticmethod
    def _sanitize_model_name(name: str) -> str:
        """Model tags often contain ':' or '/' (e.g. 'qwen3:32b-q4_k_m',
        'org/model'), which aren't safe as path components on every OS."""
        return name.replace(":", "-").replace("/", "-").replace("\\", "-")

    @classmethod
    def _results_subdir(cls, base_dir: str, model_specs: list) -> str:
        names = "_".join(cls._sanitize_model_name(spec["model"]) for spec in model_specs)
        return os.path.join(base_dir, "results", names)

    def _configs_label(self, configs: list) -> str:
        return " | ".join(
            f"{c['model']}(n={c['n']},ctx={c.get('num_ctx')},pad={c.get('pad_words', 0)})"
            for c in configs
        )

    async def unload_all(self):
        await asyncio.gather(*[
            self.client.chat(model=spec["model"], messages=[], keep_alive=0)
            for spec in self.model_specs
        ])
        await asyncio.sleep(2)  # Give the server a moment to release VRAM

    async def combined_breakdown(self) -> dict:
        """
        One `ollama ps` read, sliced per model plus combined totals.
        Returns {"per_model": {model_name: {vram_mb, total_mb, gpu_pct,
        cpu_pct, fully_on_gpu, loaded}}, "vram_total_mb", "all_loaded",
        "all_fully_on_gpu"}.
        """
        resp = await self.client.ps()
        ps_map = {m.model: m for m in resp.models}

        per_model = {}
        vram_total = 0.0
        for spec in self.model_specs:
            name = spec["model"]
            m = ps_map.get(name)
            if m is None:
                per_model[name] = {"vram_mb": 0.0, "total_mb": 0.0, "gpu_pct": None,
                                    "cpu_pct": None, "fully_on_gpu": None, "loaded": False}
                continue
            size_vram = getattr(m, "size_vram", 0) or 0
            size_total = getattr(m, "size", 0) or 0
            vram_mb = size_vram / (1024 ** 2)
            total_mb = size_total / (1024 ** 2)
            if size_total > 0:
                gpu_pct = round(100 * size_vram / size_total, 1)
                cpu_pct = round(100 - gpu_pct, 1)
                fully_on_gpu = gpu_pct >= 99.95
            else:
                gpu_pct = cpu_pct = fully_on_gpu = None
            per_model[name] = {"vram_mb": vram_mb, "total_mb": total_mb, "gpu_pct": gpu_pct,
                                "cpu_pct": cpu_pct, "fully_on_gpu": fully_on_gpu, "loaded": True}
            vram_total += vram_mb

        all_loaded = all(per_model[spec["model"]]["loaded"] for spec in self.model_specs)
        all_fully_on_gpu = all_loaded and all(
            per_model[spec["model"]]["fully_on_gpu"] for spec in self.model_specs
        )
        return {"per_model": per_model, "vram_total_mb": vram_total,
                "all_loaded": all_loaded, "all_fully_on_gpu": all_fully_on_gpu}

    async def run_request(self, model: str, i: int, num_ctx: int = None, pad_words: int = 0) -> dict:
        """
        Runs one streaming request against `model` and returns its latency
        metrics (same shape as LatencyBenchmark.run_request): wall time,
        TTFT, tok/s, and thinking/content token counts.
        """
        options = {"temperature": 0.0}
        if num_ctx:
            options["num_ctx"] = num_ctx

        output_tokens = 0
        eval_duration_ns = 0
        ttft = None
        token_index = 0
        thinking_tokens = 0
        content_tokens = 0

        start = time.perf_counter()
        async for chunk in await self.client.chat(
            model=model,
            messages=[{"role": "user", "content": self.make_prompt(pad_words)}],
            stream=True,
            options=options,
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

    async def _attempt_recovery(self, extra_sleep: float = 3.0):
        """Pings every model in model_specs (not just one) before continuing."""
        self._log(f"  Attempting recovery (sleeping {extra_sleep}s, then pinging "
                   f"{len(self.model_specs)} model(s))...")
        await asyncio.sleep(extra_sleep)
        try:
            await asyncio.wait_for(
                asyncio.gather(*[
                    self.client.chat(model=spec["model"], messages=[{"role": "user", "content": "ping"}],
                                      keep_alive=0)
                    for spec in self.model_specs
                ]),
                timeout=30,
            )
            self._log("  Recovery ping succeeded for all models.")
        except Exception as e:
            self._log(f"  Recovery ping failed too ({e}); continuing to next grid point anyway.")

    @staticmethod
    def _empty_latency() -> dict:
        return {"wall_time_sec": None, "ttft_sec": None, "tokens_per_sec": None,
                "batch_tokens_per_sec": None, "thinking_tokens": None, "content_tokens": None}

    async def measure_batch(self, configs: list) -> dict:
        """
        configs: list of {"model", "n", "num_ctx", "pad_words"}, one per model
        (same order as self.model_specs). Fires one gather() per model so a
        failure is attributable to a specific model rather than "something
        failed". On success, returns combined VRAM plus each model's own
        averaged latency over its concurrent requests. Never raises.
        """
        batch_start = time.perf_counter()

        try:
            await asyncio.wait_for(self.unload_all(), timeout=self.request_timeout)
        except Exception as e:
            status, err_msg = self._classify_error(e)
            elapsed = time.perf_counter() - batch_start
            self._log(f"  ✗ Unload failed before {self._configs_label(configs)}: {status} — {err_msg}")
            await self._attempt_recovery()
            return {"status": status, "error_message": f"unload_failed: {err_msg}",
                    "vram_total_mb": None, "elapsed_time_sec": elapsed,
                    "per_model_status": {c["model"]: status for c in configs},
                    "per_model_latency": {c["model"]: self._empty_latency() for c in configs}}

        per_model_successes = {}
        per_model_errors = {}
        pending = []
        for cfg in configs:
            model, n = cfg["model"], cfg["n"]
            ctx, pad = cfg.get("num_ctx"), cfg.get("pad_words", 0)
            tasks = [asyncio.wait_for(self.run_request(model, i, ctx, pad), timeout=self.request_timeout)
                     for i in range(1, n + 1)]
            pending.append((model, asyncio.gather(*tasks, return_exceptions=True)))

        for model, coro in pending:
            results = await coro
            errors = [r for r in results if isinstance(r, Exception)]
            successes = [r for r in results if not isinstance(r, Exception)]
            per_model_successes[model] = successes
            per_model_errors[model] = errors[0] if errors else None
        self._log("")
        elapsed = time.perf_counter() - batch_start

        failed = {m: e for m, e in per_model_errors.items() if e is not None}
        if failed:
            statuses, msgs = {}, []
            for m, e in failed.items():
                status, err_msg = self._classify_error(e)
                statuses[m] = status
                msgs.append(f"{m}: {status} — {err_msg}")
            self._log(f"  ✗ FAILED {self._configs_label(configs)}: {'; '.join(msgs)}")
            row = {"status": "partial_or_full_failure", "error_message": "; ".join(msgs),
                   "vram_total_mb": None, "elapsed_time_sec": elapsed,
                   "per_model_status": {c["model"]: statuses.get(c["model"], "ok") for c in configs},
                   "per_model_latency": {c["model"]: self._empty_latency() for c in configs}}
            if self.capture_ps_snapshots:
                row["ps_after"] = await self._ps_snapshot()
            await self._attempt_recovery()
            return row

        try:
            breakdown = await asyncio.wait_for(self.combined_breakdown(), timeout=30)
            if not breakdown["all_fully_on_gpu"]:
                pcts = {m: d["gpu_pct"] for m, d in breakdown["per_model"].items()}
                self._log(f"  ⚠ Not all models fully on GPU for {self._configs_label(configs)}: {pcts}")

            per_model_latency = {}
            combined_batch_tps = 0.0
            for cfg in configs:
                model = cfg["model"]
                results = per_model_successes[model]
                total_tokens = sum(r["output_tokens"] for r in results)
                batch_tps = total_tokens / elapsed if elapsed > 0 else 0.0
                combined_batch_tps += batch_tps
                per_model_latency[model] = {
                    "wall_time_sec": statistics.mean(r["wall_time_sec"] for r in results),
                    "ttft_sec": statistics.mean(r["ttft_sec"] for r in results),
                    "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in results),
                    "batch_tokens_per_sec": batch_tps,
                    "thinking_tokens": statistics.mean(r["thinking_tokens"] for r in results),
                    "content_tokens": statistics.mean(r["content_tokens"] for r in results),
                }

            return {"status": "ok", "error_message": None, "elapsed_time_sec": elapsed,
                    "vram_total_mb": breakdown["vram_total_mb"],
                    "all_loaded": breakdown["all_loaded"],
                    "all_fully_on_gpu": breakdown["all_fully_on_gpu"],
                    "per_model_status": {c["model"]: "ok" for c in configs},
                    "per_model_breakdown": breakdown["per_model"],
                    "per_model_latency": per_model_latency,
                    "combined_batch_tokens_per_sec": combined_batch_tps}
        except Exception as e:
            status, err_msg = self._classify_error(e)
            self._log(f"  ✗ VRAM read failed {self._configs_label(configs)}: {status} — {err_msg}")
            return {"status": "ps_read_error", "error_message": err_msg,
                    "vram_total_mb": None, "elapsed_time_sec": elapsed,
                    "per_model_status": {c["model"]: "ok" for c in configs},
                    "per_model_latency": {c["model"]: self._empty_latency() for c in configs}}

    async def run_grid_search(self) -> list:
        """
        Full cross-product grid search: each model gets its own (n, num_ctx,
        pad_words) combo list, and every combination of every model's combos
        is measured. Returns a flat list of dicts with per-model columns
        prefixed m0_, m1_, ... (index-based, so model tags with ':' or '/'
        never collide with CSV field names).
        """
        per_model_combos = []
        for spec in self.model_specs:
            n_list = spec.get("n_list") or [1]
            ctx_list = spec.get("ctx_list") or [None]
            pad_list = spec.get("pad_list") or [0]
            per_model_combos.append(list(itertools.product(n_list, ctx_list, pad_list)))

        total = 1
        for c in per_model_combos:
            total *= len(c)
        model_names = [s["model"] for s in self.model_specs]
        self._log(f"Multi-model grid search: {total} combinations across "
                   f"{len(self.model_specs)} models ({', '.join(model_names)})...")
        if total > 200:
            self._log(f"  ⚠ {total} combinations means {total} full unload+load cycles across "
                       f"all models — this will take a while. Consider narrowing n_list/ctx_list/"
                       f"pad_list before running unattended.")

        rows = []
        for combo_tuple in itertools.product(*per_model_combos):
            configs = [
                {"model": spec["model"], "n": n, "num_ctx": ctx, "pad_words": pad}
                for spec, (n, ctx, pad) in zip(self.model_specs, combo_tuple)
            ]
            result = await self.measure_batch(configs)

            row = {
                "grid_point": self._configs_label(configs),
                "status": result["status"],
                "error_message": result.get("error_message"),
                "elapsed_time_sec": result.get("elapsed_time_sec"),
                "vram_total_mb": result.get("vram_total_mb"),
                "all_loaded": result.get("all_loaded"),
                "all_fully_on_gpu": result.get("all_fully_on_gpu"),
                "combined_batch_tokens_per_sec": result.get("combined_batch_tokens_per_sec"),
            }
            per_model_status = result.get("per_model_status", {})
            per_model_breakdown = result.get("per_model_breakdown", {})
            per_model_latency = result.get("per_model_latency", {})
            for idx, cfg in enumerate(configs):
                prefix = f"m{idx}"
                bd = per_model_breakdown.get(cfg["model"]) if per_model_breakdown else None
                lat = per_model_latency.get(cfg["model"]) if per_model_latency else self._empty_latency()
                row[f"{prefix}_model"] = cfg["model"]
                row[f"{prefix}_n"] = cfg["n"]
                row[f"{prefix}_num_ctx"] = cfg["num_ctx"]
                row[f"{prefix}_pad_words"] = cfg["pad_words"]
                row[f"{prefix}_status"] = per_model_status.get(cfg["model"], "unknown")
                row[f"{prefix}_vram_mb"] = bd["vram_mb"] if bd else None
                row[f"{prefix}_gpu_pct"] = bd["gpu_pct"] if bd else None
                row[f"{prefix}_wall_time_sec"] = lat["wall_time_sec"]
                row[f"{prefix}_ttft_sec"] = lat["ttft_sec"]
                row[f"{prefix}_tokens_per_sec"] = lat["tokens_per_sec"]
                row[f"{prefix}_batch_tokens_per_sec"] = lat["batch_tokens_per_sec"]
                row[f"{prefix}_thinking_tokens"] = lat["thinking_tokens"]
                row[f"{prefix}_content_tokens"] = lat["content_tokens"]
            rows.append(row)

            if result["status"] == "ok":
                ttft_bits = ", ".join(
                    f"{cfg['model']} ttft={result['per_model_latency'][cfg['model']]['ttft_sec']:.2f}s"
                    for cfg in configs
                )
                self._log(f"  {self._configs_label(configs)}  "
                          f"vram_total={result['vram_total_mb']:.0f}MB  ({ttft_bits})")

        n_failed = sum(1 for r in rows if r["status"] != "ok")
        if n_failed:
            self._log(f"Grid search done: {n_failed}/{len(rows)} grid points failed.")
        return rows

    async def run_all(self) -> list:
        """Convenience alias for run_grid_search()."""
        return await self.run_grid_search()

    def save_results(self, rows: list, filename: str = "results.csv"):
        """Saves to <output_dir>/results/<model1>_<model2>_.../<filename> —
        the model-combo subfolder was already created in __init__."""
        self.save_csv(rows, filename)

    def plot_results(self, rows: list, filename: str = "vram.png"):
        """
        Currently only plots the 2-model case cleanly (VRAM total vs. model
        A's n, one line per distinct model-B config + model-A ctx/pad combo).
        With 3+ models a single 2D plot stops being readable — skips
        plotting and points at the CSV instead rather than force a
        contorted N-dimensional chart.
        """
        if len(self.model_specs) != 2:
            self._log(f"plot_results only supports exactly 2 models today (got "
                       f"{len(self.model_specs)}); skipping plot — use the CSV directly.")
            return

        ok_rows = [r for r in rows if r.get("status") == "ok"]
        n_skipped = len(rows) - len(ok_rows)
        if n_skipped:
            self._log(f"Skipping {n_skipped} failed grid point(s) when plotting.")
        if not ok_rows:
            self._log("No successful measurements to plot.")
            return

        def group_key(r):
            return (r["m0_num_ctx"], r["m0_pad_words"], r["m1_n"], r["m1_num_ctx"], r["m1_pad_words"])

        groups = {}
        for r in ok_rows:
            groups.setdefault(group_key(r), []).append(r)

        model_a, model_b = self.model_specs[0]["model"], self.model_specs[1]["model"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, sub in sorted(groups.items(), key=lambda kv: str(kv[0])):
            sub = sorted(sub, key=lambda r: r["m0_n"])
            ctx0, pad0, n1, ctx1, pad1 = key
            label = f"A: ctx={ctx0},pad={pad0}  |  B: n={n1},ctx={ctx1},pad={pad1}"
            ax.plot([r["m0_n"] for r in sub], [r["vram_total_mb"] for r in sub], marker="o", label=label)

        ax.set_title(f"Combined VRAM — {model_a} + {model_b}")
        ax.set_xlabel(f"Concurrent requests to {model_a} (n)")
        ax.set_ylabel("Total VRAM used (MB)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1)

        self._save_plot(fig, filename)

    def plot_latency(self, rows: list, metric: str = "ttft_sec", filename: str = None):
        """
        Same 2-model-only structure as plot_results, but for a per-model
        latency metric instead of VRAM. metric is one of: "ttft_sec",
        "wall_time_sec", "tokens_per_sec", "batch_tokens_per_sec". Plots
        BOTH models' values (m0_<metric> and m1_<metric>) vs. model A's n,
        one line pair per distinct "other config" grouping.
        """
        if len(self.model_specs) != 2:
            self._log(f"plot_latency only supports exactly 2 models today (got "
                       f"{len(self.model_specs)}); skipping plot — use the CSV directly.")
            return

        valid_metrics = {"ttft_sec", "wall_time_sec", "tokens_per_sec", "batch_tokens_per_sec"}
        if metric not in valid_metrics:
            self._log(f"Unknown latency metric '{metric}'; expected one of {sorted(valid_metrics)}.")
            return

        ok_rows = [r for r in rows if r.get("status") == "ok"]
        if not ok_rows:
            self._log("No successful measurements to plot.")
            return

        if filename is None:
            filename = f"latency_{metric}.png"

        def group_key(r):
            return (r["m0_num_ctx"], r["m0_pad_words"], r["m1_n"], r["m1_num_ctx"], r["m1_pad_words"])

        groups = {}
        for r in ok_rows:
            groups.setdefault(group_key(r), []).append(r)

        model_a, model_b = self.model_specs[0]["model"], self.model_specs[1]["model"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, sub in sorted(groups.items(), key=lambda kv: str(kv[0])):
            sub = sorted(sub, key=lambda r: r["m0_n"])
            ctx0, pad0, n1, ctx1, pad1 = key
            label_suffix = f"ctx0={ctx0},pad0={pad0} | B: n={n1},ctx={ctx1},pad={pad1}"
            ax.plot([r["m0_n"] for r in sub], [r[f"m0_{metric}"] for r in sub],
                     marker="o", linestyle="-", label=f"{model_a} ({label_suffix})")
            ax.plot([r["m0_n"] for r in sub], [r[f"m1_{metric}"] for r in sub],
                     marker="s", linestyle="--", label=f"{model_b} ({label_suffix})")

        ax.set_title(f"{metric} under concurrent load — {model_a} + {model_b}")
        ax.set_xlabel(f"Concurrent requests to {model_a} (n)")
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=1)

        self._save_plot(fig, filename)


# Backward-compatible alias — the class used to be VRAM-only.
MultiModelVRAMBenchmark = MultiModelBenchmark

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

    HOST = "http://localhost:11458"
    REQUEST_TIMEOUT = 240.0

    MODEL_SPECS = [
        {
            "model": "qwen3-235b-a22b:q4_k_m",
            "n_list": [10, 20],
            "ctx_list": [32768, 65536],
            "pad_list": [0],
        },
        {
            "model": "mistral-small3.2:24b",
            "n_list": [5, 10],
            "ctx_list": [16384],
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
