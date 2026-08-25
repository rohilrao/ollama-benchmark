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
