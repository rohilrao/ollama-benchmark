"""Combined VRAM + latency benchmark for 1..N vLLM servers (podman containers).

One grid point =
    (optionally) release every request across every endpoint at the same instant
 -> sample VRAM per GPU index while they generate, and average each model's
    own latency.

There is no unload/load phase (see vllm_bench.py docstring for why): each
endpoint is assumed to already be up, serving one model, for the whole run.
With a single model spec this is the single-server VRAM + latency benchmark;
with two or more (each on its own base_url/GPU) it is the co-residency
benchmark. Same code path either way -- same shape as MultiModelBenchmark in
multi_model_bench.py, and the same column names in the CSV.
"""

import asyncio
import itertools
import statistics
import time

import matplotlib.pyplot as plt

from vllm_bench import VLLMBenchmarkBase

METRICS = ("wall_time_sec", "ttft_sec", "ttft_thinking_sec", "ttft_content_sec",
           "tokens_per_sec", "batch_tokens_per_sec")
PLOTTABLE = METRICS  # no load_sec here -- there's no load phase to time


class MultiEndpointVLLMBenchmark(VLLMBenchmarkBase):
    """
    model_specs: list of dicts, one per vLLM endpoint:
        {"model": str,            # id as served, e.g. "/models/deepseek-r1"
         "base_url": str,         # e.g. "http://localhost:8555/v1"
         "api_key": str,          # optional, default "EMPTY"
         "gpu_index": int|None,   # nvidia-smi index this container is pinned to
         "n_list": [int],         # concurrent requests at each grid point
         "pad_list": [int],       # filler words prepended to the prompt
         "max_tokens_list": [int]}# output length sweep (defaults to [default_max_tokens])

    Grid points = product of every model's own (n x pad x max_tokens) combos,
    so the count grows fast with more models; it's logged up front.

    Results land directly in output_dir (results.csv, vram.png,
    latency_<metric>.png). If you run multiple model combinations, give each
    its own output_dir so they don't overwrite each other.
    """

    def __init__(self, model_specs: list, output_dir: str = ".",
                 reps: int = 1, vram_sample_interval: float = 2.0, **kwargs):
        specs = [self._normalize(s) for s in model_specs]
        super().__init__(self._results_dir(output_dir, specs), **kwargs)
        self.model_specs = specs
        self.models = [s["model"] for s in specs]
        self.reps = reps
        self.vram_sample_interval = vram_sample_interval

    # ── setup helpers ─────────────────────────────────────────────────────
    def _normalize(self, spec: dict) -> dict:
        return {"model": spec["model"], "base_url": spec["base_url"],
                "api_key": spec.get("api_key", "EMPTY"),
                "gpu_index": spec.get("gpu_index"),
                "n_list": list(spec.get("n_list") or [1]),
                "pad_list": list(spec.get("pad_list") or [0]),
                "max_tokens_list": list(spec.get("max_tokens_list") or [self.default_max_tokens])}

    @staticmethod
    def _safe(name: str) -> str:
        for ch in ":/\\":
            name = name.replace(ch, "-")
        return name.strip("-")

    @classmethod
    def _results_dir(cls, base: str, specs: list) -> str:
        return base

    @classmethod
    def _cfg_tag(cls, cfg: dict) -> str:
        """Compact, comma-free label for a single model's config."""
        pad = f"-p{cfg['pad_words']}" if cfg["pad_words"] else ""
        return f"{cls._safe(cfg['model']).split('-')[0][:14]}-n{cfg['n']}-mt{cfg['max_tokens']}{pad}"

    @classmethod
    def _point_tag(cls, configs: list) -> str:
        return "+".join(cls._cfg_tag(c) for c in configs)

    @staticmethod
    def _blank_metrics() -> dict:
        return {k: None for k in METRICS + ("output_tokens", "thinking_tokens", "content_tokens")}

    def _spec_by_name(self, name: str) -> dict:
        return next(s for s in self.model_specs if s["model"] == name)

    # ── preflight ─────────────────────────────────────────────────────────
    async def preflight(self) -> bool:
        """Confirms every endpoint is reachable before spending time on a
        grid search that would just fail. Logs and returns False on failure."""
        ok = True
        for spec in self.model_specs:
            res = await self.check_endpoint(spec)
            if res["ok"]:
                self._log(f"  ✓ {spec['base_url']} serving {spec['model']}"
                          + (f" (gpu {spec['gpu_index']})" if spec["gpu_index"] is not None else ""))
            else:
                self._log(f"  ✗ {spec['base_url']}: {res['error']}")
                ok = False
        return ok

    # ── one grid point ────────────────────────────────────────────────────
    async def measure_point(self, configs: list) -> dict:
        """
        configs: [{"model", "n", "pad_words", "max_tokens"}, ...], one per model.
        Returns {status, error, elapsed_sec, vram_total_gb, per_model: {name: {...}}}.
        Never raises.
        """
        blank = {c["model"]: {"status": "not_run", "n_ok": 0, "n_failed": 0, "error": None,
                              **self._blank_metrics()} for c in configs}

        baseline = await self.vram_breakdown(self.model_specs)

        # ── generation phase — all requests released together ────────────
        gate = asyncio.Event()
        tasks, owners = [], []
        for cfg in configs:
            spec = self._spec_by_name(cfg["model"])
            for i in range(1, cfg["n"] + 1):
                tasks.append(asyncio.create_task(asyncio.wait_for(
                    self.stream_request(spec, i, cfg["pad_words"], cfg["max_tokens"], gate),
                    timeout=self.request_timeout)))
                owners.append(cfg["model"])

        stop = asyncio.Event()
        sampler = asyncio.create_task(self._sample_vram(stop, baseline))
        await asyncio.sleep(0)  # let every task reach the gate before it opens

        start = time.perf_counter()
        gate.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - start
        stop.set()
        peak = await sampler

        # ── per-model aggregation ────────────────────────────────────────
        per_model, failures = {}, []
        for cfg in configs:
            name = cfg["model"]
            mine = [r for r, owner in zip(results, owners) if owner == name]
            errs = [r for r in mine if isinstance(r, Exception)]
            oks = [r for r in mine if not isinstance(r, Exception)]

            placement = peak["per_model"].get(
                name, {"vram_gb": None, "total_gb": None, "gpu_util_pct": None, "per_gpu": {}})
            row = {"status": "ok", "n_ok": len(oks), "n_failed": len(errs), "error": None, **placement}
            if errs:
                status, msg = self._classify_error(errs[0])
                row["status"] = "partial_failure" if oks else status
                row["error"] = msg
                failures.append(f"{name}: {status}")

            if oks:
                # span, not point elapsed: a model that finishes early shouldn't
                # be charged for the slower model's tail.
                span = max(r["wall_time_sec"] for r in oks)
                tokens = sum(r["output_tokens"] for r in oks)
                row.update({m: statistics.mean(r[m] for r in oks)
                            for m in ("wall_time_sec", "ttft_sec", "ttft_thinking_sec",
                                     "ttft_content_sec", "tokens_per_sec")})
                row.update({"batch_tokens_per_sec": tokens / span if span > 0 else 0.0,
                            "output_tokens": tokens,
                            "thinking_tokens": statistics.mean(r["thinking_tokens"] for r in oks),
                            "content_tokens": statistics.mean(r["content_tokens"] for r in oks)})
                if tokens == 0:
                    # Request "succeeded" but produced nothing — a silent worker
                    # failure, which would otherwise land in the CSV as a very
                    # fast, very good-looking row.
                    row["status"] = "empty_response"
                    row["error"] = "completed with 0 output tokens"
                    failures.append(f"{name}: empty_response")
            else:
                row.update(self._blank_metrics())
            per_model[name] = row

        if failures:
            self._log(f"  ✗ {self._point_tag(configs)}: {'; '.join(failures)}")

        return {"status": "ok" if not failures else "partial_or_full_failure",
                "error": "; ".join(failures) or None,
                "elapsed_sec": elapsed,
                "vram_total_gb": peak["vram_total_gb"],
                "per_model": per_model}

    async def _sample_vram(self, stop: asyncio.Event, baseline: dict) -> dict:
        """Poll nvidia-smi during generation and keep the highest total seen."""
        peak = baseline
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.vram_sample_interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                bd = await self.vram_breakdown(self.model_specs)
                if bd["vram_total_gb"] > peak["vram_total_gb"]:
                    peak = bd
            except Exception:
                pass  # sampling must never break a measurement
        return peak

    # ── grid ──────────────────────────────────────────────────────────────
    def _grid(self) -> list:
        per_model = [list(itertools.product(s["n_list"], s["pad_list"], s["max_tokens_list"]))
                     for s in self.model_specs]
        return [[{"model": s["model"], "n": n, "pad_words": pad, "max_tokens": mt}
                 for s, (n, pad, mt) in zip(self.model_specs, combo)]
                for combo in itertools.product(*per_model)]

    async def run_grid_search(self) -> list:
        """
        Runs every grid point `reps` times. Returns long-format rows: one row per
        (point, rep, model), so the schema is identical for 1 model or 5 and every
        cell is a scalar -- and matches the Ollama benchmark's row shape.
        """
        self._log("Checking endpoints...")
        if not await self.preflight():
            self._log("  ⚠ one or more endpoints failed preflight; continuing anyway "
                      "(failures will show up as row-level errors).")

        grid = self._grid()
        self._log(f"Grid: {len(grid)} point(s) x {self.reps} rep(s) across "
                  f"{len(self.models)} model(s): {', '.join(self.models)}")
        if len(grid) * self.reps > 200:
            self._log("  ⚠ that's a lot of grid points — consider narrowing the lists.")

        rows = []
        for p, configs in enumerate(grid, 1):
            for rep in range(1, self.reps + 1):
                self._log(f"[{p}/{len(grid)} rep {rep}/{self.reps}] {self._point_tag(configs)}")
                res = await self.measure_point(configs)
                tag = self._point_tag(configs)
                for cfg in configs:
                    pm = res["per_model"][cfg["model"]]
                    rows.append({
                        "point": p, "rep": rep, "point_tag": tag,
                        "model": cfg["model"], "n": cfg["n"],
                        "max_tokens": cfg["max_tokens"], "pad_words": cfg["pad_words"],
                        "point_status": res["status"], "model_status": pm["status"],
                        "vram_gb": pm.get("vram_gb"), "total_gb": pm.get("total_gb"),
                        "gpu_util_pct": pm.get("gpu_util_pct"),
                        "vram_per_gpu": ";".join(f"gpu{i}={v}" for i, v in (pm.get("per_gpu") or {}).items()),
                        "vram_total_gb": res["vram_total_gb"],
                        "point_elapsed_sec": res["elapsed_sec"],
                        **{m: pm.get(m) for m in METRICS},
                        "output_tokens": pm.get("output_tokens"),
                        "thinking_tokens": pm.get("thinking_tokens"),
                        "content_tokens": pm.get("content_tokens"),
                        "n_ok": pm.get("n_ok"), "n_failed": pm.get("n_failed"),
                        "error": pm.get("error") or res["error"],
                    })
                if res["status"] == "ok":
                    detail = "  ".join(
                        "{m} ttft_think={tk} ttft_content={c:.2f}s tok/s={tps:.1f}".format(
                            m=c["model"],
                            tk=("-" if res["per_model"][c["model"]]["thinking_tokens"] in (0, None)
                                else f"{res['per_model'][c['model']]['ttft_thinking_sec']:.2f}s"),
                            c=res["per_model"][c["model"]]["ttft_content_sec"],
                            tps=res["per_model"][c["model"]]["tokens_per_sec"])
                        for c in configs)
                    self._log(f"  ok  vram_total={res['vram_total_gb']:.2f}GB  "
                              f"elapsed={res['elapsed_sec']:.1f}s  {detail}")

        failed = {r["point"] for r in rows if r["point_status"] != "ok"}
        if failed:
            self._log(f"{len(failed)}/{len(grid)} grid point(s) had failures.")
        return rows

    async def run_all(self) -> list:
        return await self.run_grid_search()

    # ── output ────────────────────────────────────────────────────────────
    def save_results(self, rows: list, filename: str = "results.csv"):
        self.save_csv(rows, filename)

    def _points(self, rows: list, need: str) -> tuple:
        """(ordered point tags, {model: [value per point]}) averaged over reps."""
        ok = [r for r in rows if r.get(need) is not None and r["model_status"] in ("ok", "partial_failure")]
        if not ok:
            return [], {}
        tags = list(dict.fromkeys(r["point_tag"] for r in sorted(ok, key=lambda r: r["point"])))
        series = {}
        for model in self.models:
            vals = []
            for tag in tags:
                v = [r[need] for r in ok if r["point_tag"] == tag and r["model"] == model]
                vals.append(statistics.mean(v) if v else float("nan"))
            series[model] = vals
        return tags, series

    def plot_vram(self, rows: list, filename: str = "vram.png"):
        """Stacked per-model VRAM per grid point. Works for any number of models."""
        tags, series = self._points(rows, "vram_gb")
        if not tags:
            self._log("No successful VRAM measurements to plot (check gpu_index is set).")
            return
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tags)), 6))
        bottom = [0.0] * len(tags)
        for model, vals in series.items():
            vals = [0.0 if v != v else v for v in vals]  # NaN -> 0 for stacking
            ax.bar(tags, vals, bottom=bottom, label=model)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"VRAM with {len(self.models)} vLLM endpoint(s) running")
        ax.set_ylabel("VRAM used (GB)")
        ax.set_xlabel("Grid point")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        self._save_plot(fig, filename)

    def plot_latency(self, rows: list, metric: str = "ttft_sec", filename: str = None):
        """One line per model across grid points, for any latency metric."""
        if metric not in PLOTTABLE:
            self._log(f"Unknown metric '{metric}'; expected one of {sorted(PLOTTABLE)}.")
            return
        tags, series = self._points(rows, metric)
        if not tags:
            self._log(f"No successful {metric} measurements to plot.")
            return
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tags)), 6))
        for model, vals in series.items():
            ax.plot(tags, vals, marker="o", label=model)
        ax.set_title(f"{metric} under concurrent load")
        ax.set_ylabel(metric)
        ax.set_xlabel("Grid point")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        self._save_plot(fig, filename or f"latency_{metric}.png")

    def plot_all(self, rows: list):
        self.plot_vram(rows)
        for metric in PLOTTABLE:
            self.plot_latency(rows, metric)

    # ── teardown ──────────────────────────────────────────────────────────
    async def cleanup(self):
        """
        Closes the HTTP clients. There's nothing to unload -- the podman
        containers keep running after this script exits, unlike Ollama's
        keep_alive-based eviction. Safe to call twice; never raises.
        """
        self._log("Closing HTTP clients (podman containers keep running)...")
        await self.aclose()
