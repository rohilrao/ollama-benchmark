"""Shared plumbing for Ollama benchmarks.

Everything that is not specific to the grid-search logic lives here: the client,
prompt building, a single streaming request that returns latency metrics,
`ollama ps` parsing, model load/unload, error classification and recovery, and
CSV/plot output.
"""

import asyncio
import csv
import json
import os
import time
import uuid

import matplotlib
matplotlib.use("Agg")  # headless boxes / over ssh
import matplotlib.pyplot as plt
import ollama


class OllamaBenchmarkBase:
    PROMPT_BASE = "Explain KV Cache in one sentence."
    LOREM = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
             "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim ") * 400

    # Case-insensitive substrings meaning the llama runner crashed or OOM'd,
    # as opposed to a generic HTTP error.
    OOM_SIGNATURES = (
        "llama runner process has terminated",
        "out of memory",
        "cudamalloc",
        "failed to allocate",
        "llama_new_context_with_model failed",
    )

    def __init__(self, host: str, output_dir: str = ".", verbose: bool = True,
                 request_timeout: float = 240.0, load_timeout: float = 900.0,
                 keep_alive: str = "30m", log_tokens: bool = False):
        self.host = host
        self.output_dir = output_dir
        self.verbose = verbose
        self.request_timeout = request_timeout
        self.load_timeout = load_timeout      # loading a 235B from cold takes minutes
        self.keep_alive = keep_alive          # must outlive a grid point or models get evicted
        self.log_tokens = log_tokens          # per-token logging is unusable at n=40
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = ollama.AsyncClient(host=host)

    def _log(self, msg="", end="\n", flush=False):
        if self.verbose:
            print(msg, end=end, flush=flush)

    def make_prompt(self, pad_words: int = 0) -> str:
        """Unique (cache-busting) prompt, optionally padded with filler words."""
        pad = " ".join(self.LOREM.split()[:pad_words]) + "\n" if pad_words else ""
        return f"[{uuid.uuid4().hex}]\n{pad}{self.PROMPT_BASE}"

    # ── error handling ────────────────────────────────────────────────────
    def _classify_error(self, exc: Exception) -> tuple:
        """(status, message). Never raises."""
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout", str(exc) or f"exceeded request_timeout of {self.request_timeout}s"

        message = str(exc)
        detail = getattr(exc, "error", None)  # ollama.ResponseError carries the server message here
        if detail:
            message = f"{message} | {detail}"
        low = message.lower()

        if any(sig in low for sig in self.OOM_SIGNATURES):
            return "oom_or_runner_crash", message
        if "connection" in low or isinstance(exc, ConnectionError):
            return "connection_error", message
        if getattr(exc, "status_code", None) == 500:
            return "server_error_500", message
        return "unknown_error", message

    # ── ollama ps ─────────────────────────────────────────────────────────
    async def ps_breakdown(self, models: list) -> dict:
        """
        One `ollama ps` read, sliced per model plus totals. Reports not just VRAM
        but how it splits against the model's total footprint (the same size vs
        size_vram fields the CLI uses to print "100% GPU" / "43%/57% CPU/GPU").

        Returns {per_model: {name: {loaded, vram_gb, total_gb, cpu_gb, gpu_pct,
                 cpu_pct, fully_on_gpu}}, vram_total_gb, cpu_total_gb,
                 all_loaded, all_fully_on_gpu}.

        Sizes are decimal GB (1000^3), matching the `ollama ps` CLI's own SIZE
        column (format.HumanBytes), so a value here agrees with the CLI's printed
        size for the same model.

        `size`/`size_vram` come from the runner's MemorySize(), i.e. buffers
        allocated at load (weights + KV cache for num_ctx x OLLAMA_NUM_PARALLEL).
        They are not a live GPU reading and do not grow as the KV cache fills.
        """
        resp = await self.client.ps()
        found = {m.model: m for m in resp.models}

        per_model, vram_total, cpu_total = {}, 0.0, 0.0
        GB = 1000 ** 3  # decimal GB, matching the `ollama ps` CLI's own SIZE column
        for name in models:
            # `ollama ps` reports "qwen3:latest" for a bare "qwen3" request.
            m = found.get(name) or found.get(f"{name}:latest")
            if m is None:
                per_model[name] = {"loaded": False, "vram_gb": 0.0, "total_gb": 0.0, "cpu_gb": 0.0,
                                   "gpu_pct": None, "cpu_pct": None, "fully_on_gpu": None,
                                   "ctx_reported": None}
                continue
            vram_gb = (getattr(m, "size_vram", 0) or 0) / GB
            total_gb = (getattr(m, "size", 0) or 0) / GB
            if total_gb <= 0 or vram_gb > total_gb:
                # Same case `ollama ps` prints as "Unknown"; don't invent a split.
                cpu_gb, gpu_pct, cpu_pct, fully = 0.0, None, None, None
            else:
                cpu_gb = total_gb - vram_gb          # the part that spilled to system RAM
                gpu_pct = round(100 * vram_gb / total_gb, 1)
                cpu_pct = round(100 - gpu_pct, 1)
                fully = gpu_pct >= 99.95
            per_model[name] = {"loaded": True, "vram_gb": vram_gb, "total_gb": total_gb,
                               "cpu_gb": cpu_gb, "gpu_pct": gpu_pct, "cpu_pct": cpu_pct,
                               "fully_on_gpu": fully,
                               # What the runner actually loaded, vs what we asked for.
                               "ctx_reported": getattr(m, "context_length", None) or None}
            vram_total += vram_gb
            cpu_total += cpu_gb

        return {"per_model": per_model, "vram_total_gb": vram_total, "cpu_total_gb": cpu_total,
                "all_loaded": all(p["loaded"] for p in per_model.values()),
                "all_fully_on_gpu": all(bool(p["fully_on_gpu"]) for p in per_model.values())}

    async def ps_snapshot(self) -> str:
        """Best-effort `ollama ps` dump as a JSON string; never raises."""
        try:
            resp = await self.client.ps()
            return json.dumps([{"model": m.model,
                                "vram_gb": round((getattr(m, "size_vram", 0) or 0) / 1000 ** 3, 2)}
                               for m in resp.models])
        except Exception as e:
            return f"ps_snapshot_failed: {e}"

    # ── model lifecycle ───────────────────────────────────────────────────
    async def unload(self, models: list, settle: float = 2.0, verify_timeout: float = 60.0):
        """
        Evict every model and wait until `ollama ps` confirms it is gone, so one
        grid point never inherits the previous point's runner or its VRAM. A
        fixed sleep isn't enough: freeing a large model can take longer than it,
        and the next load would then be measured against stale VRAM.
        """
        for name in models:
            try:
                await asyncio.wait_for(
                    self.client.chat(model=name, messages=[], keep_alive=0), timeout=60)
            except Exception as e:
                self._log(f"  unload of {name} failed (ignored): {e}")

        deadline = time.perf_counter() + verify_timeout
        while True:
            await asyncio.sleep(settle)
            try:
                resident = [m for m, p in (await self.ps_breakdown(models))["per_model"].items()
                            if p["loaded"]]
            except Exception:
                return  # can't verify; the residency check after loading will still catch trouble
            if not resident:
                return
            if time.perf_counter() > deadline:
                self._log(f"  ⚠ still resident {verify_timeout}s after unload: {' '.join(resident)}")
                return

    async def load(self, model: str, num_ctx: int = None):
        """
        Force `model` resident *without generating*: an empty messages array is a
        load-only request. num_ctx is passed here as well as on the real requests,
        because a later request with a different num_ctx makes Ollama reload the
        runner — which would put a model load back inside the timed window.
        """
        options = {"num_ctx": num_ctx} if num_ctx else {}
        try:
            await asyncio.wait_for(
                self.client.chat(model=model, messages=[], keep_alive=self.keep_alive, options=options),
                timeout=self.load_timeout,
            )
        except asyncio.TimeoutError:
            # A load that never returns is the usual shape of an OOM that Ollama
            # neither reports nor recovers from; bound it and move on.
            raise asyncio.TimeoutError(
                f"load of {model} (num_ctx={num_ctx}) exceeded load_timeout of {self.load_timeout}s")

    async def warmup(self, model: str, num_ctx: int = None):
        """
        One throwaway generation, on top of `load()`. Being resident isn't the
        same as being warm: the first real generation after a load still pays
        for CUDA kernel JIT / cuBLAS autotune / cache warming that later
        generations don't, which otherwise shows up as an inflated TTFT on
        whichever request happens to run first. Errors are logged, not raised —
        a failed warmup shouldn't lose an otherwise-good grid point.
        """
        options = {"num_ctx": num_ctx} if num_ctx else {}
        try:
            async for _ in await self.client.chat(
                model=model, messages=[{"role": "user", "content": "hello"}],
                stream=True, options=options, keep_alive=self.keep_alive,
            ):
                pass
        except Exception as e:
            self._log(f"  warmup for {model} failed (ignored): {e}")

    async def recover(self, models: list, sleep: float = 3.0):
        """Best-effort post-failure recovery; logged, never raised."""
        self._log(f"  recovering (sleep {sleep}s, then pinging {len(models)} model(s))...")
        await asyncio.sleep(sleep)
        for name in models:
            try:
                await asyncio.wait_for(
                    self.client.chat(model=name, messages=[{"role": "user", "content": "ping"}],
                                     keep_alive=0),
                    timeout=60)
            except Exception as e:
                self._log(f"  recovery ping for {name} failed ({e}); continuing anyway.")

    # ── one request ───────────────────────────────────────────────────────
    async def stream_request(self, model: str, i: int, num_ctx: int = None,
                             pad_words: int = 0, gate: asyncio.Event = None) -> dict:
        """
        One streaming request, returning wall time, time-to-first-thinking-token,
        time-to-first-content-token, tok/s, and thinking vs content chunk counts.
        Reasoning models stream thinking before content, so these two TTFTs can
        differ a lot — "time to first token" usually means the content one.

        If `gate` is given the request blocks on it before starting, so every
        request across every model leaves the line at the same instant and the
        clock starts after the gate — not at task-creation time.
        """
        options = {"temperature": 0.0}
        if num_ctx:
            options["num_ctx"] = num_ctx
        prompt = self.make_prompt(pad_words)  # built before the gate; not timed

        if gate is not None:
            await gate.wait()

        out_tokens = eval_ns = thinking = content = idx = 0
        ttft_thinking = ttft_content = None
        start = time.perf_counter()
        async for chunk in await self.client.chat(
            model=model, messages=[{"role": "user", "content": prompt}],
            stream=True, options=options, keep_alive=self.keep_alive,
        ):
            msg = chunk.get("message", {}) or {}
            think_tok, content_tok = msg.get("thinking") or "", msg.get("content") or ""
            if think_tok:
                thinking += 1
                if ttft_thinking is None:
                    ttft_thinking = time.perf_counter() - start
            if content_tok:
                content += 1
                if ttft_content is None:
                    ttft_content = time.perf_counter() - start
            if think_tok or content_tok:
                idx += 1
                if self.log_tokens:
                    kind = "thinking" if think_tok else "content"
                    self._log(f"[{model} R{i} T{idx} {kind}]: {(think_tok or content_tok).strip()}",
                              end=" | ", flush=True)
            if chunk.get("done"):
                out_tokens = chunk.get("eval_count", 0) or 0
                eval_ns = chunk.get("eval_duration", 0) or 0

        wall = time.perf_counter() - start
        # Non-reasoning models never set ttft_thinking; fall back to whichever
        # arrived, then wall time, so downstream code always gets a float.
        ttft_first = next((t for t in (ttft_thinking, ttft_content) if t is not None), wall)
        return {"wall_time_sec": wall,
                "ttft_sec": ttft_first,
                "ttft_thinking_sec": ttft_thinking if ttft_thinking is not None else ttft_first,
                "ttft_content_sec": ttft_content if ttft_content is not None else wall,
                "tokens_per_sec": out_tokens / (eval_ns / 1e9) if eval_ns > 0 else 0.0,
                "output_tokens": out_tokens,
                "thinking_tokens": thinking,
                "content_tokens": content}

    # ── output ────────────────────────────────────────────────────────────
    @staticmethod
    def _csv_safe(v):
        """Keep commas and newlines out of cells so the CSV stays column-aligned."""
        if isinstance(v, str):
            return v.replace(",", ";").replace("\n", " ").replace("\r", " ").strip()
        if isinstance(v, float):
            return round(v, 4)
        return v

    def save_csv(self, rows: list, filename: str):
        if not rows:
            self._log(f"No rows to save for {filename}; skipping.")
            return
        path = os.path.join(self.output_dir, filename)
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({k: self._csv_safe(v) for k, v in r.items()} for r in rows)
        self._log(f"Saved CSV → {path}")

    def _save_plot(self, fig, filename: str):
        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self._log(f"Saved plot → {path}")
