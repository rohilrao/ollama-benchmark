"""Combined VRAM + latency benchmark for 1..N models sharing a GPU.

One grid point =
    unload everything
 -> load every model to residency, at the exact num_ctx it will be benched at
 -> verify via `ollama ps` that all of them are actually resident
 -> send one untimed warmup generation per model (resident is not warm)
 -> release every request across every model at the same instant
 -> sample VRAM while they generate, and average each model's own latency.

With a single model spec this is the old VRAM + latency benchmark; with two or
more it is the co-residency benchmark. Same code path either way.
"""

import asyncio
import itertools
import os
import statistics
import time

import matplotlib.pyplot as plt

from ollama_bench import OllamaBenchmarkBase

METRICS = ("wall_time_sec", "ttft_sec", "ttft_thinking_sec", "ttft_content_sec",
           "tokens_per_sec", "batch_tokens_per_sec")
PLOTTABLE = METRICS + ("load_sec",)   # load_sec is measured per model, not per request


class MultiModelBenchmark(OllamaBenchmarkBase):
    """
    model_specs: list of dicts, one per model:
        {"model": str, "n_list": [int], "ctx_list": [int|None], "pad_list": [int]}
    ctx_list/pad_list default to [None]/[0], so adding a model doesn't silently
    multiply the grid unless you ask for it.

    Grid points = product of every model's own (n x ctx x pad) combos, so the
    count grows fast with more models; it's logged up front.

    Results land in <output_dir>/<model1>_<model2>_.../ so different
    model combinations never overwrite each other.
    """

    def __init__(self, host: str, model_specs: list, output_dir: str = ".",
                 reps: int = 1, vram_sample_interval: float = 2.0,
                 min_gpu_pct: float = 99.9, skip_on_cpu_offload: bool = True, **kwargs):
        specs = [self._normalize(s) for s in model_specs]
        super().__init__(host, self._results_dir(output_dir, specs), **kwargs)
        self.model_specs = specs
        self.models = [s["model"] for s in specs]
        self.reps = reps
        self.vram_sample_interval = vram_sample_interval
        self.min_gpu_pct = min_gpu_pct                  # below this a model counts as offloaded
        self.skip_on_cpu_offload = skip_on_cpu_offload  # False = bench anyway, still flagged

    # ── setup helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _normalize(spec: dict) -> dict:
        return {"model": spec["model"],
                "n_list": list(spec.get("n_list") or [1]),
                "ctx_list": list(spec.get("ctx_list") or [None]),
                "pad_list": list(spec.get("pad_list") or [0])}

    @staticmethod
    def _safe(name: str) -> str:
        for ch in ":/\\":
            name = name.replace(ch, "-")
        return name

    @classmethod
    def _results_dir(cls, base: str, specs: list) -> str:
        return os.path.join(base, "_".join(cls._safe(s["model"]) for s in specs))

    @classmethod
    def _cfg_tag(cls, cfg: dict) -> str:
        """Compact, comma-free label for a single model's config."""
        ctx = cfg["num_ctx"]
        ctx_s = "def" if not ctx else (f"{ctx // 1024}k" if ctx % 1024 == 0 else str(ctx))
        pad = f"-p{cfg['pad_words']}" if cfg["pad_words"] else ""
        return f"{cls._safe(cfg['model']).split('-')[0][:14]}-n{cfg['n']}-c{ctx_s}{pad}"

    @classmethod
    def _point_tag(cls, configs: list) -> str:
        return "+".join(cls._cfg_tag(c) for c in configs)

    @staticmethod
    def _blank_metrics() -> dict:
        return {k: None for k in METRICS + ("output_tokens", "thinking_tokens", "content_tokens")}

    def _placement(self, bd: dict, load_sec: dict, name: str) -> dict:
        """Where the model physically sits, per `ollama ps`. Recorded on every
        row — including skipped ones, which is the whole point."""
        return {"load_sec": load_sec.get(name),
                "vram_gb": bd.get("vram_gb"), "total_gb": bd.get("total_gb"),
                "cpu_gb": bd.get("cpu_gb"), "gpu_pct": bd.get("gpu_pct"),
                "cpu_pct": bd.get("cpu_pct"), "ctx_reported": bd.get("ctx_reported")}

    # ── one grid point ────────────────────────────────────────────────────
    async def measure_point(self, configs: list) -> dict:
        """
        configs: [{"model", "n", "num_ctx", "pad_words"}, ...], one per model.
        Returns {status, error, elapsed_sec, vram_total_gb, cpu_total_gb,
                 all_on_gpu, per_model: {name: {...}}}. Never raises.
        """
        blank = {c["model"]: {"status": "not_run", "n_ok": 0, "n_failed": 0, "error": None,
                              **self._blank_metrics()} for c in configs}

        await self.unload(self.models)

        # ── load phase — nothing is timed until every model is resident ──
        load_sec = {}
        for cfg in configs:
            t = time.perf_counter()
            try:
                await self.load(cfg["model"], cfg["num_ctx"])
            except Exception as e:
                status, msg = self._classify_error(e)
                self._log(f"  ✗ load failed for {cfg['model']}: {status} — {msg}")
                await self.recover(self.models)
                return {"status": status, "error": f"load_failed[{cfg['model']}]: {msg}",
                        "elapsed_sec": None, "vram_total_gb": None, "cpu_total_gb": None,
                        "all_on_gpu": None, "per_model": blank}
            load_sec[cfg["model"]] = time.perf_counter() - t

        # ── placement check — refuse to bench a point that didn't fit ────
        loaded = await self.ps_breakdown(self.models)
        placed = {c["model"]: {**blank[c["model"]],
                               **self._placement(loaded["per_model"][c["model"]], load_sec, c["model"])}
                  for c in configs}
        head = {"elapsed_sec": None, "vram_total_gb": loaded["vram_total_gb"],
                "cpu_total_gb": loaded["cpu_total_gb"], "all_on_gpu": loaded["all_fully_on_gpu"]}

        missing = [m for m, p in loaded["per_model"].items() if not p["loaded"]]
        if missing:
            self._log(f"  ✗ not resident after loading: {' '.join(missing)}")
            self._log("    Ollama evicted them — raise OLLAMA_MAX_LOADED_MODELS or free VRAM. "
                      "Measuring now would time a load, not co-residency.")
            for m in missing:
                placed[m]["status"] = "not_loaded"
            return {"status": "eviction_or_oom", "error": f"not_resident: {' '.join(missing)}",
                    **head, "per_model": placed}

        # Ollama retries a load that OOM'd with a smaller context (sched.go
        # reduceAutoNumCtxForLoadOOM) and returns success, so the runner can be
        # holding a context we never asked for. `ollama ps` reports what it
        # actually loaded; anything smaller than requested makes the point a
        # measurement of a different configuration.
        shrunk = {}
        for c in configs:
            got = loaded["per_model"][c["model"]]["ctx_reported"]
            if c["num_ctx"] and got and got < c["num_ctx"]:
                shrunk[c["model"]] = (got, c["num_ctx"])
        if shrunk:
            detail = " ".join(f"{m}: loaded {got} not {want}" for m, (got, want) in shrunk.items())
            self._log(f"  ✗ context silently reduced after a load OOM — skipping: {detail}")
            for m in shrunk:
                placed[m]["status"] = "ctx_reduced"
            return {"status": "ctx_reduced", "error": f"ctx_reduced: {detail}",
                    **head, "per_model": placed}

        # A model that spilled into system RAM is running partly on CPU: its
        # numbers measure the spill, not the GPU, and generation can take
        # minutes-to-forever. Record how much went to CPU and skip the point.
        spilled = {m: p for m, p in loaded["per_model"].items()
                   if p["gpu_pct"] is not None and p["gpu_pct"] < self.min_gpu_pct}
        if spilled:
            detail = " ".join(f"{m}={p['cpu_gb']:.2f}GB/{p['cpu_pct']}% on CPU"
                              for m, p in spilled.items())
            if self.skip_on_cpu_offload:
                self._log(f"  ✗ CPU offload — skipping benchmark for this point: {detail}")
                for m in spilled:
                    placed[m]["status"] = "cpu_offload"
                return {"status": "cpu_offload", "error": f"cpu_offload: {detail}",
                        **head, "per_model": placed}
            self._log(f"  ⚠ CPU offload (benching anyway, skip_on_cpu_offload=False): {detail}")

        # ── warmup — resident is not warm; do this before anything is timed.
        # The first generation after a load still pays for CUDA kernel JIT /
        # cache warming, which otherwise inflates TTFT for whichever request
        # happens to run first.
        await asyncio.gather(*(self.warmup(c["model"], c["num_ctx"]) for c in configs))

        # ── generation phase — all requests released together ────────────
        gate = asyncio.Event()
        tasks, owners = [], []
        for cfg in configs:
            for i in range(1, cfg["n"] + 1):
                tasks.append(asyncio.create_task(asyncio.wait_for(
                    self.stream_request(cfg["model"], i, cfg["num_ctx"], cfg["pad_words"], gate),
                    timeout=self.request_timeout)))
                owners.append(cfg["model"])

        stop = asyncio.Event()
        sampler = asyncio.create_task(self._sample_vram(stop, loaded))
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

            row = {"status": "ok", "n_ok": len(oks), "n_failed": len(errs), "error": None,
                   **self._placement(peak["per_model"].get(name, {}), load_sec, name)}
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
                    # Request "succeeded" but produced nothing — a silent runner
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
            await self.recover(self.models)

        return {"status": "ok" if not failures else "partial_or_full_failure",
                "error": "; ".join(failures) or None,
                "elapsed_sec": elapsed,
                "vram_total_gb": peak["vram_total_gb"],
                "cpu_total_gb": peak["cpu_total_gb"],
                "all_on_gpu": peak["all_fully_on_gpu"],
                "per_model": per_model}

    async def _sample_vram(self, stop: asyncio.Event, baseline: dict) -> dict:
        """
        Poll `ollama ps` during generation and keep the highest total seen. Note
        that size_vram is a load-time allocation, not a live reading, so this
        will not show the KV cache filling — it catches a model being evicted or
        reloaded mid-run, which would otherwise go unnoticed.
        """
        peak = baseline
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.vram_sample_interval)
                break
            except asyncio.TimeoutError:
                pass
            try:
                bd = await self.ps_breakdown(self.models)
                if bd["vram_total_gb"] > peak["vram_total_gb"]:
                    peak = bd
            except Exception:
                pass  # sampling must never break a measurement
        return peak

    # ── grid ──────────────────────────────────────────────────────────────
    def _grid(self) -> list:
        per_model = [list(itertools.product(s["n_list"], s["ctx_list"], s["pad_list"]))
                     for s in self.model_specs]
        return [[{"model": s["model"], "n": n, "num_ctx": ctx, "pad_words": pad}
                 for s, (n, ctx, pad) in zip(self.model_specs, combo)]
                for combo in itertools.product(*per_model)]

    async def run_grid_search(self) -> list:
        """
        Runs every grid point `reps` times. Returns long-format rows: one row per
        (point, rep, model), so the schema is identical for 1 model or 5 and every
        cell is a scalar.
        """
        grid = self._grid()
        self._log(f"Grid: {len(grid)} point(s) x {self.reps} rep(s) across "
                  f"{len(self.models)} model(s): {', '.join(self.models)}")
        self._log(f"Check the server env: OLLAMA_MAX_LOADED_MODELS >= {len(self.models)}, "
                  f"OLLAMA_NUM_PARALLEL >= {max(max(s['n_list']) for s in self.model_specs)}, "
                  "or Ollama will evict models / queue requests behind each other.")
        if len(grid) * self.reps > 200:
            self._log("  ⚠ that's a lot of full unload+load cycles — consider narrowing the lists.")

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
                        "num_ctx": cfg["num_ctx"], "ctx_reported": pm.get("ctx_reported"),
                        "pad_words": cfg["pad_words"],
                        "point_status": res["status"], "model_status": pm["status"],
                        "vram_gb": pm.get("vram_gb"), "cpu_gb": pm.get("cpu_gb"),
                        "total_gb": pm.get("total_gb"),
                        "gpu_pct": pm.get("gpu_pct"), "cpu_pct": pm.get("cpu_pct"),
                        "vram_total_gb": res["vram_total_gb"],
                        "cpu_total_gb": res["cpu_total_gb"], "all_on_gpu": res["all_on_gpu"],
                        "load_sec": pm.get("load_sec"), "point_elapsed_sec": res["elapsed_sec"],
                        **{m: pm.get(m) for m in METRICS},
                        "output_tokens": pm.get("output_tokens"),
                        "thinking_tokens": pm.get("thinking_tokens"),
                        "content_tokens": pm.get("content_tokens"),
                        "n_ok": pm.get("n_ok"), "n_failed": pm.get("n_failed"),
                        "error": pm.get("error") or res["error"],
                    })
                if res["status"] == "ok":
                    detail = "  ".join(
                        "{m} load={l:.1f}s ttft_think={tk} ttft_content={c:.2f}s".format(
                            m=c["model"],
                            l=res["per_model"][c["model"]]["load_sec"] or 0.0,
                            tk=("-" if res["per_model"][c["model"]]["thinking_tokens"] in (0, None)
                                else f"{res['per_model'][c['model']]['ttft_thinking_sec']:.2f}s"),
                            c=res["per_model"][c["model"]]["ttft_content_sec"])
                        for c in configs)
                    self._log(f"  ok  vram_total={res['vram_total_gb']:.2f}GB  "
                              f"elapsed={res['elapsed_sec']:.1f}s  {detail}")

        offload = [r for r in rows if r["point_status"] == "cpu_offload"]
        if offload:
            seen = {}
            for r in offload:
                seen.setdefault((r["point_tag"], r["model"]), r)
            self._log(f"\n{len({r['point'] for r in offload})} grid point(s) skipped "
                      "— model spilled to CPU:")
            for (tag, model), r in seen.items():
                if (r["cpu_gb"] or 0) > 0:
                    self._log(f"  {tag}  {model}: {r['cpu_gb']:.2f}GB of "
                              f"{r['total_gb']:.2f}GB on CPU ({r['cpu_pct']}%)")
        failed = {r["point"] for r in rows
                  if r["point_status"] not in ("ok", "cpu_offload")}
        if failed:
            self._log(f"{len(failed)}/{len(grid)} grid point(s) had failures.")
        return rows

    async def run_all(self) -> list:
        return await self.run_grid_search()

    # ── output ────────────────────────────────────────────────────────────
    def save_results(self, rows: list, filename: str = "results.csv"):
        self.save_csv(rows, filename)

    def average_reps(self, rows: list) -> list:
        """
        Collapse the raw rows to one row per (grid point, model), averaged over
        reps. Placement (VRAM/CPU/load) is averaged over every rep, so a point
        skipped for CPU offload still reports how much spilled; latency is
        averaged over successful reps only, so a failed rep can't drag the mean
        toward zero. `_std` columns are sample stdev, 0.0 when there's one rep.
        """
        placement = ("vram_gb", "cpu_gb", "total_gb", "gpu_pct", "cpu_pct",
                     "vram_total_gb", "cpu_total_gb", "load_sec", "ctx_reported")
        perf = METRICS + ("point_elapsed_sec", "output_tokens",
                          "thinking_tokens", "content_tokens")
        spread = ("load_sec", "vram_gb", "wall_time_sec", "ttft_content_sec",
                  "tokens_per_sec", "batch_tokens_per_sec")

        def agg(reps, field, stat="mean"):
            vals = [r[field] for r in reps if r.get(field) is not None]
            if not vals:
                return None
            if stat == "std":
                return statistics.stdev(vals) if len(vals) > 1 else 0.0
            return statistics.mean(vals)

        groups = {}
        for r in rows:
            groups.setdefault((r["point"], r["model"]), []).append(r)

        out = []
        for (point, model), reps in sorted(groups.items()):
            first = reps[0]
            ok = [r for r in reps if r["model_status"] in ("ok", "partial_failure")]
            row = {"point": point, "point_tag": first["point_tag"], "model": model,
                   "n": first["n"], "num_ctx": first["num_ctx"], "pad_words": first["pad_words"],
                   "reps": len(reps), "reps_ok": len(ok),
                   "statuses": ";".join(sorted({r["model_status"] for r in reps})),
                   "all_on_gpu": all(bool(r["all_on_gpu"]) for r in reps)}
            row.update({f: agg(reps, f) for f in placement})
            row.update({f: agg(ok, f) for f in perf})
            row.update({f + "_std": agg(ok if f in perf else reps, f, "std") for f in spread})
            row["requests_ok"] = sum(r["n_ok"] or 0 for r in reps)
            row["requests_failed"] = sum(r["n_failed"] or 0 for r in reps)
            row["error"] = next((r["error"] for r in reps if r["error"]), None)
            out.append(row)
        return out

    def save_averaged(self, rows: list, filename: str = "results_avg.csv"):
        self.save_csv(self.average_reps(rows), filename)

    def save_config(self, filename: str = "config.txt"):
        """Plain-text record of what was actually swept, dropped in the results
        folder alongside the CSVs — so a run is self-describing without having
        to reread the script that produced it."""
        lines = [f"host: {self.host}",
                 f"reps: {self.reps}",
                 f"request_timeout_sec: {self.request_timeout}",
                 f"load_timeout_sec: {self.load_timeout}",
                 f"keep_alive: {self.keep_alive}",
                 f"min_gpu_pct: {self.min_gpu_pct}",
                 f"skip_on_cpu_offload: {self.skip_on_cpu_offload}",
                 "", "models:"]
        for s in self.model_specs:
            lines.append(f"  {s['model']}")
            lines.append(f"    n_list:   {s['n_list']}")
            lines.append(f"    ctx_list: {s['ctx_list']}")
            lines.append(f"    pad_list: {s['pad_list']}")
        n_points = 1
        for s in self.model_specs:
            n_points *= len(s["n_list"]) * len(s["ctx_list"]) * len(s["pad_list"])
        lines += ["", f"grid points: {n_points}", f"total measurements (points x reps): {n_points * self.reps}"]

        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self._log(f"Saved config → {path}")

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
            self._log("No successful VRAM measurements to plot.")
            return
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tags)), 6))
        bottom = [0.0] * len(tags)
        for model, vals in series.items():
            vals = [0.0 if v != v else v for v in vals]  # NaN -> 0 for stacking
            ax.bar(tags, vals, bottom=bottom, label=model)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"VRAM with {len(self.models)} model(s) co-resident")
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
        Unload every model and close the HTTP client, so the GPU is left free
        for whatever runs next instead of holding VRAM for keep_alive minutes.
        Safe to call twice; never raises.
        """
        self._log("Cleaning up — unloading all models...")
        try:
            await self.unload(self.models)
            left = await self.ps_breakdown(self.models)
            resident = [f"{m} ({p['vram_gb']:.2f}GB)"
                        for m, p in left["per_model"].items() if p["loaded"]]
            self._log(f"  ⚠ still resident: {', '.join(resident)}" if resident
                      else "  GPU clear — no benchmark models resident.")
        except Exception as e:
            self._log(f"  cleanup failed (ignored): {e}")
        client = getattr(self.client, "_client", None)   # underlying httpx client
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
