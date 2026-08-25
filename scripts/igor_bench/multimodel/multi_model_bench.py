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

