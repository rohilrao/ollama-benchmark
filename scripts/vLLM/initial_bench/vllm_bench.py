"""Shared plumbing for vLLM (OpenAI-compatible) benchmarks, run via podman.

Mirrors ollama_bench.py as closely as the two servers' APIs allow, so the
resulting results.csv shares column names with the Ollama benchmark and the
two can be plotted/compared side by side.

Where this necessarily differs from Ollama, and why:
  - vLLM does not hot-swap models through the API. Each vLLM server (one
    podman container) is dedicated to a single model, and its context length
    is fixed at container startup (--max-model-len). So there is no
    unload()/load() lifecycle here, and no ctx_list sweep -- to compare
    context lengths, restart the container with a different --max-model-len,
    re-run this script, and record the ctx value yourself (e.g. in the label
    you give the model spec).
  - "Co-residency" of N models becomes N vLLM containers running at once,
    normally one per GPU. You point this script at all of them by giving each
    a base_url + gpu_index. VRAM is read per-GPU-index via `nvidia-smi`
    rather than per-model-name -- if two models ever share a GPU index this
    reports the GPU's total, not a per-model split.
  - Token accounting comes from the OpenAI streaming `usage` chunk
    (stream_options={"include_usage": True}) instead of Ollama's
    eval_count/eval_duration.
  - Reasoning models (e.g. deepseek-r1 on vLLM) stream `delta.reasoning_content`
    before `delta.content`, exactly like concurrent_requests.py already
    handles -- so ttft_thinking / ttft_content are captured the same way as
    the Ollama base.
  - tokens_per_sec is computed over (wall_time - ttft_content), i.e.
    decode-only, to stay comparable to Ollama's eval_count/eval_duration
    (which also excludes prompt processing).
"""

import asyncio
import csv
import os
import time
import uuid

import matplotlib
matplotlib.use("Agg")  # headless boxes / over ssh
import matplotlib.pyplot as plt
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)


class VLLMBenchmarkBase:
    PROMPT_BASE = "Explain KV Cache in one sentence."
    LOREM = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
             "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim ") * 400

    # Case-insensitive substrings meaning the vLLM worker crashed or OOM'd,
    # as opposed to a generic HTTP/connection error.
    OOM_SIGNATURES = (
        "cuda out of memory",
        "out of memory",
        "cudamalloc",
        "no available memory",
        "engine core has crashed",
        "enginedeaderror",
        "worker.*died",
        "kv cache",
    )

    def __init__(self, output_dir: str = ".", verbose: bool = True,
                 request_timeout: float = 240.0, log_tokens: bool = False,
                 default_max_tokens: int = 256):
        self.output_dir = output_dir
        self.verbose = verbose
        self.request_timeout = request_timeout
        self.log_tokens = log_tokens          # per-token logging is unusable at n=40
        self.default_max_tokens = default_max_tokens
        os.makedirs(self.output_dir, exist_ok=True)
        self._clients = {}  # base_url -> AsyncOpenAI

    def _log(self, msg="", end="\n", flush=False):
        if self.verbose:
            print(msg, end=end, flush=flush)

    def make_prompt(self, pad_words: int = 0) -> str:
        """Unique (cache-busting) prompt, optionally padded with filler words."""
        pad = " ".join(self.LOREM.split()[:pad_words]) + "\n" if pad_words else ""
        return f"[{uuid.uuid4().hex}]\n{pad}{self.PROMPT_BASE}"

    def _client(self, spec: dict) -> AsyncOpenAI:
        """One AsyncOpenAI client per base_url, reused across requests."""
        base_url = spec["base_url"]
        if base_url not in self._clients:
            self._clients[base_url] = AsyncOpenAI(
                base_url=base_url, api_key=spec.get("api_key", "EMPTY"))
        return self._clients[base_url]

    # ── error handling ────────────────────────────────────────────────────
    def _classify_error(self, exc: Exception) -> tuple:
        """(status, message). Never raises."""
        if isinstance(exc, (asyncio.TimeoutError, APITimeoutError)):
            return "timeout", str(exc) or f"exceeded request_timeout of {self.request_timeout}s"

        message = str(exc)
        low = message.lower()

        if any(sig in low for sig in self.OOM_SIGNATURES):
            return "oom_or_worker_crash", message
        if isinstance(exc, APIConnectionError) or "connection" in low:
            return "connection_error", message
        if isinstance(exc, RateLimitError):
            return "rate_limited", message
        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, APIStatusError) or status_code is not None:
            return f"server_error_{status_code}", message
        return "unknown_error", message

    # ── preflight / model check ──────────────────────────────────────────
    async def check_endpoint(self, spec: dict) -> dict:
        """
        Confirms base_url is reachable and the requested model id is being
        served. Never raises -- returns {ok, served_models, error}.
        """
        try:
            resp = await asyncio.wait_for(self._client(spec).models.list(), timeout=30)
            served = [m.id for m in resp.data]
            ok = (not served) or (spec["model"] in served)
            return {"ok": ok, "served_models": served,
                    "error": None if ok else f"{spec['model']} not in served models: {served}"}
        except Exception as e:
            status, msg = self._classify_error(e)
            return {"ok": False, "served_models": [], "error": f"{status}: {msg}"}

    # ── GPU VRAM via nvidia-smi ──────────────────────────────────────────
    @staticmethod
    async def _nvidia_smi_snapshot() -> dict:
        """
        {gpu_index: {"mem_used_mb": float, "mem_total_mb": float, "gpu_util_pct": float}}
        Best-effort; returns {} if nvidia-smi is unavailable (e.g. sandboxed
        container without access to the host GPU tooling).
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            snap = {}
            for line in out.decode().strip().splitlines():
                idx, used, total, util = (p.strip() for p in line.split(","))
                snap[int(idx)] = {"mem_used_mb": float(used), "mem_total_mb": float(total),
                                  "gpu_util_pct": float(util)}
            return snap
        except Exception:
            return {}

    @staticmethod
    def _gpu_indices(spec: dict) -> list:
        """spec['gpu_index'] can be a single int (one GPU) or a list of ints
        (tensor-parallel across several GPUs, e.g. [0, 1, 2, 3])."""
        idx = spec.get("gpu_index")
        if idx is None:
            return []
        return list(idx) if isinstance(idx, (list, tuple)) else [idx]

    async def vram_breakdown(self, specs: list) -> dict:
        """
        Per-spec VRAM. gpu_index may be a single int (one GPU per model, the
        normal way to pin separate vLLM containers) or a list of ints (one
        model tensor-parallel across several GPUs) -- in the latter case
        vram_gb/total_gb are summed across all of them and per_gpu holds the
        individual readings. Specs without a gpu_index report None -- they're
        still benched, just without a VRAM column.

        Returns {per_model: {name: {vram_gb, total_gb, gpu_util_pct, per_gpu}},
                 vram_total_gb} where vram_total_gb sums only the *distinct*
                 GPU indices referenced across all specs, so two specs sharing
                 a GPU (or overlapping TP ranks) don't double-count it.
        """
        snap = await self._nvidia_smi_snapshot()
        per_model, seen_idx = {}, set()
        vram_total = 0.0
        for s in specs:
            indices = self._gpu_indices(s)
            readings = [snap[i] for i in indices if i in snap]
            if not indices or len(readings) != len(indices):
                per_model[s["model"]] = {"vram_gb": None, "total_gb": None,
                                         "gpu_util_pct": None, "per_gpu": {}}
                continue
            vram_gb = sum(g["mem_used_mb"] for g in readings) / 1024
            total_gb = sum(g["mem_total_mb"] for g in readings) / 1024
            avg_util = sum(g["gpu_util_pct"] for g in readings) / len(readings)
            per_model[s["model"]] = {
                "vram_gb": vram_gb, "total_gb": total_gb, "gpu_util_pct": avg_util,
                "per_gpu": {i: round(g["mem_used_mb"] / 1024, 3) for i, g in zip(indices, readings)}}
            for i, g in zip(indices, readings):
                if i not in seen_idx:
                    vram_total += g["mem_used_mb"] / 1024
                    seen_idx.add(i)
        return {"per_model": per_model, "vram_total_gb": vram_total}

    # ── one request ───────────────────────────────────────────────────────
    async def stream_request(self, spec: dict, i: int, pad_words: int = 0,
                             max_tokens: int = None, gate: asyncio.Event = None) -> dict:
        """
        One streaming chat completion, returning wall time, time-to-first-
        thinking-token, time-to-first-content-token, decode tok/s, and
        thinking vs content chunk counts. Same semantics as
        OllamaBenchmarkBase.stream_request so the two are directly comparable.

        If `gate` is given the request blocks on it before starting, so every
        request across every model/endpoint leaves the line at the same
        instant and the clock starts after the gate -- not at task-creation
        time.
        """
        prompt = self.make_prompt(pad_words)  # built before the gate; not timed

        if gate is not None:
            await gate.wait()

        thinking = content = idx = 0
        ttft_thinking = ttft_content = None
        usage = None
        start = time.perf_counter()
        stream = await self._client(spec).chat.completions.create(
            model=spec["model"],
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.0,
            max_tokens=max_tokens or spec.get("max_tokens", self.default_max_tokens),
        )
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            think_tok = getattr(delta, "reasoning_content", None) or ""
            content_tok = getattr(delta, "content", None) or ""
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
                    self._log(f"[{spec['model']} R{i} T{idx} {kind}]: {(think_tok or content_tok).strip()}",
                              end=" | ", flush=True)

        wall = time.perf_counter() - start
        out_tokens = (usage.completion_tokens if usage is not None else content) or 0
        # Non-reasoning models never set ttft_thinking; fall back to whichever
        # arrived, then wall time, so downstream code always gets a float.
        ttft_first = next((t for t in (ttft_thinking, ttft_content) if t is not None), wall)
        # Decode-only speed (post-first-token), to line up with Ollama's
        # eval_count/eval_duration which also excludes prompt processing.
        decode_time = wall - ttft_content if (ttft_content is not None and wall > ttft_content) else wall
        tokens_per_sec = out_tokens / decode_time if decode_time > 0 and out_tokens else 0.0

        return {"wall_time_sec": wall,
                "ttft_sec": ttft_first,
                "ttft_thinking_sec": ttft_thinking if ttft_thinking is not None else ttft_first,
                "ttft_content_sec": ttft_content if ttft_content is not None else wall,
                "tokens_per_sec": tokens_per_sec,
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

    async def aclose(self):
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
