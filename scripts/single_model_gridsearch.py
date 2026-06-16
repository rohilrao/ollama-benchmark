#!/usr/bin/env python3
"""
Ollama grid-search benchmark
-----------------------------
Runs every combination of (parallel_requests × context_length) as a grid,
polls `ollama ps` throughout each cell, and writes results to CSV.

Failure modes handled:
  - HTTP 500 / OOM from Ollama (VRAM exceeded)
  - Connection errors / timeouts
  - Partial failures (some of N parallel requests fail)
  - Full cell failure → marked as VRAM_OOM or ERROR, grid search continues
"""

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11436"
MODEL       = "llama3.2"               # ← your model name

PROMPT      = "Explain gravity in one sentence."

# Grid axes — every combination is tested
PARALLEL_REQUESTS = [1, 4, 8, 12, 16]          # rows
CONTEXT_LENGTHS   = [8_192, 32_768, 65_536, 131_072]  # cols  (8k / 32k / 64k / 128k)

# Behaviour
POLL_INTERVAL_SEC  = 0.5     # ollama ps sampling rate during a cell run
REQUEST_TIMEOUT    = 300     # seconds per individual request before giving up
SETTLE_SEC         = 3       # pause between grid cells for GPU to settle
WARMUP_CTX         = 2_048   # context size for the throwaway warmup request
WARMUP_PROMPT      = "Say hello."
UNLOAD_AFTER_BENCH = True    # evict model from VRAM when done

OUTPUT_CSV = "ollama_grid_results.csv"

# OOM detection: strings that appear in Ollama error bodies when VRAM is exceeded
OOM_SIGNALS = [
    "out of memory", "oom", "cuda error", "not enough memory",
    "failed to allocate", "cudaerroroutofmemory", "no space left",
    "exit status 1",   # ollama process crash on OOM
]
# ───────────────────────────────────────────────────────────────────────────────

import asyncio
import csv
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from itertools import product as iterproduct


# ── helpers ─────────────────────────────────────────────────────────────────────

def parse_size(s: str) -> float:
    """'3.5 GiB' / '512 MiB' / '0 B'  →  float bytes."""
    s = s.strip()
    if s in ("", "-", "0 B", "0B"):
        return 0.0
    m = re.match(r"([\d.]+)\s*([KMGT]?i?B)", s, re.IGNORECASE)
    if not m:
        return 0.0
    value, unit = float(m.group(1)), m.group(2).upper()
    mult = {"B": 1, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
            "KB": 1000, "MB": 1e6, "GB": 1e9, "TB": 1e12}
    return value * mult.get(unit, 1)


def is_oom(error_str: str) -> bool:
    low = error_str.lower()
    return any(sig in low for sig in OOM_SIGNALS)


def gib(b: float) -> float:
    return round(b / 1024**3, 3)


# ── ollama ps ───────────────────────────────────────────────────────────────────

def sample_ollama_ps() -> dict:
    """
    Run `ollama ps`, parse SIZE + PROCESSOR columns, return aggregated stats.
    Splits size into VRAM and CPU RAM based on the GPU% reported per instance.
    """
    try:
        env = {**os.environ,
               "OLLAMA_HOST": OLLAMA_HOST.replace("http://", "").replace("https://", "")}
        r = subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                           timeout=5, env=env)
        lines = r.stdout.strip().splitlines()
        if len(lines) < 2:
            return {"vram_bytes": 0, "cpu_ram_bytes": 0, "processors": "", "instances": 0}

        total_vram, total_cpu = 0.0, 0.0
        processors, instances = [], 0

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            instances += 1
            size_bytes = parse_size(parts[2])
            proc_str   = " ".join(parts[3:-1])   # between SIZE and UNTIL
            processors.append(proc_str)

            m = re.search(r"(\d+)%\s*GPU", proc_str, re.IGNORECASE)
            gpu_pct = float(m.group(1)) / 100.0 if m else 0.0

            total_vram += size_bytes * gpu_pct
            total_cpu  += size_bytes * (1 - gpu_pct)

        return {"vram_bytes": total_vram, "cpu_ram_bytes": total_cpu,
                "processors": "; ".join(processors), "instances": instances}
    except Exception as e:
        return {"vram_bytes": 0, "cpu_ram_bytes": 0,
                "processors": f"ps_error:{e}", "instances": 0}


# ── single request ──────────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


async def call_ollama(session_id: int, context_len: int) -> dict:
    """One /api/generate call. Returns a result dict with success/error info."""
    payload = {
        "model":   MODEL,
        "prompt":  PROMPT,
        "stream":  False,
        "options": {"num_ctx": context_len},
    }
    loop = asyncio.get_event_loop()
    t0   = time.perf_counter()

    try:
        resp    = await loop.run_in_executor(
            None, lambda: _post_json(f"{OLLAMA_HOST}/api/generate",
                                     payload, timeout=REQUEST_TIMEOUT))
        elapsed = time.perf_counter() - t0

        # Ollama can return HTTP 200 but embed an error in the JSON body
        if "error" in resp:
            err = resp["error"]
            return {"session_id": session_id, "success": False,
                    "elapsed_sec": round(elapsed, 3),
                    "prompt_tokens": 0, "gen_tokens": 0,
                    "error": err, "oom": is_oom(err)}

        return {"session_id": session_id, "success": True,
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": resp.get("prompt_eval_count", 0),
                "gen_tokens":    resp.get("eval_count", 0),
                "error": "", "oom": False}

    except urllib.error.HTTPError as e:
        elapsed  = time.perf_counter() - t0
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            body = str(e)
        return {"session_id": session_id, "success": False,
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": 0, "gen_tokens": 0,
                "error": f"HTTP {e.code}: {body[:300]}", "oom": is_oom(body)}

    except Exception as e:
        elapsed = time.perf_counter() - t0
        err     = str(e)
        return {"session_id": session_id, "success": False,
                "elapsed_sec": round(elapsed, 3),
                "prompt_tokens": 0, "gen_tokens": 0,
                "error": err, "oom": is_oom(err)}


# ── model lifecycle ─────────────────────────────────────────────────────────────

async def warmup_model() -> None:
    """Fire one cheap request to load the model, then confirm it's resident."""
    print(f"⏳ Warming up '{MODEL}' (ctx={WARMUP_CTX}) …", flush=True)
    loop = asyncio.get_event_loop()
    t0   = time.perf_counter()
    try:
        resp = await loop.run_in_executor(None, lambda: _post_json(
            f"{OLLAMA_HOST}/api/generate",
            {"model": MODEL, "prompt": WARMUP_PROMPT, "stream": False,
             "options": {"num_ctx": WARMUP_CTX}, "keep_alive": "30m"},
            timeout=180,
        ))
        elapsed = time.perf_counter() - t0
        print(f"   ✓ Warmup done in {elapsed:.1f}s  "
              f"(gen_tokens={resp.get('eval_count', '?')})")
    except Exception as e:
        print(f"   ⚠ Warmup failed: {e} — continuing anyway")

    for _ in range(20):
        if sample_ollama_ps()["instances"] > 0:
            break
        await asyncio.sleep(0.5)
    else:
        print("   ⚠ Model not visible in `ollama ps` — check connection")


async def unload_model() -> None:
    """Evict model from VRAM via keep_alive=0."""
    if not UNLOAD_AFTER_BENCH:
        return
    print(f"\n🗑  Unloading '{MODEL}' (keep_alive=0) …", flush=True)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: _post_json(
            f"{OLLAMA_HOST}/api/generate",
            {"model": MODEL, "prompt": "", "keep_alive": 0},
            timeout=30,
        ))
    except Exception as e:
        if "Expecting value" not in str(e) and "EOF" not in str(e):
            print(f"   ⚠ Unload error (may be harmless): {e}")

    for _ in range(20):
        if sample_ollama_ps()["instances"] == 0:
            print("   ✓ Model unloaded — VRAM cleared")
            return
        await asyncio.sleep(0.5)
    print("   ⚠ Model still visible in `ollama ps` after unload")


# ── grid cell runner ────────────────────────────────────────────────────────────

# Failure classifications written into the CSV status column
STATUS_OK         = "OK"
STATUS_PARTIAL    = "PARTIAL"    # some requests failed, at least one succeeded
STATUS_OOM        = "VRAM_OOM"   # all failed and at least one error looks like OOM
STATUS_ERROR      = "ERROR"      # all failed, non-OOM reason


async def run_cell(n_requests: int, ctx_len: int, cell_idx: int, total: int) -> dict:
    """
    Run one grid cell: fire n_requests concurrently at ctx_len, poll ps,
    classify the outcome, return a CSV row dict.
    """
    label = f"[{cell_idx}/{total}] {n_requests}×{ctx_len//1024}k"
    print(f"\n{'='*62}")
    print(f"  {label}  ({datetime.now():%H:%M:%S})")
    print(f"{'='*62}")

    # ── ps polling ──────────────────────────────────────────────────────────────
    ps_samples: list[dict] = []
    stop_event = asyncio.Event()

    async def poller():
        while not stop_event.is_set():
            ps_samples.append(sample_ollama_ps())
            await asyncio.sleep(POLL_INTERVAL_SEC)

    poll_task = asyncio.create_task(poller())

    # ── fire all requests ────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    results = await asyncio.gather(
        *[call_ollama(i, ctx_len) for i in range(n_requests)]
    )
    t_total = time.perf_counter() - t_start

    stop_event.set()
    await poll_task

    # ── classify outcome ─────────────────────────────────────────────────────────
    successes  = [r for r in results if r["success"]]
    failures   = [r for r in results if not r["success"]]
    oom_hits   = [r for r in failures if r["oom"]]
    all_errors = [r["error"] for r in failures]

    if len(failures) == 0:
        status = STATUS_OK
    elif len(successes) > 0:
        status = STATUS_PARTIAL
    elif len(oom_hits) > 0:
        status = STATUS_OOM
    else:
        status = STATUS_ERROR

    # ── ps aggregation ───────────────────────────────────────────────────────────
    vram_vals = [s["vram_bytes"]    for s in ps_samples if s["vram_bytes"] > 0]
    cpu_vals  = [s["cpu_ram_bytes"] for s in ps_samples]
    procs     = "; ".join({s["processors"] for s in ps_samples if s["processors"]})

    peak_vram = gib(max(vram_vals))                       if vram_vals else 0
    mean_vram = gib(sum(vram_vals) / len(vram_vals))      if vram_vals else 0
    peak_cpu  = gib(max(cpu_vals))                        if cpu_vals  else 0
    mean_cpu  = gib(sum(cpu_vals)  / len(cpu_vals))       if cpu_vals  else 0

    mean_elapsed = (round(sum(r["elapsed_sec"] for r in successes) / len(successes), 3)
                    if successes else None)

    # Status indicator for console
    status_icon = {"OK": "✓", "PARTIAL": "⚡", "VRAM_OOM": "💥", "ERROR": "✗"}[status]
    print(f"  {status_icon} {status}  {len(successes)}/{n_requests} ok  "
          f"total={t_total:.1f}s  VRAM peak={peak_vram:.2f}G  CPU peak={peak_cpu:.2f}G")
    if failures:
        # Show first unique error
        unique_errs = list(dict.fromkeys(r["error"][:120] for r in failures))
        for ue in unique_errs[:2]:
            print(f"    error: {ue}")

    return {
        "timestamp":            datetime.now().isoformat(timespec="seconds"),
        "model":                MODEL,
        "n_requests":           n_requests,
        "context_k":            ctx_len // 1024,
        "status":               status,
        "successes":            len(successes),
        "failures":             len(failures),
        "oom_failures":         len(oom_hits),
        "total_elapsed_sec":    round(t_total, 3),
        "mean_req_elapsed_sec": mean_elapsed,
        "peak_vram_gib":        peak_vram,
        "mean_vram_gib":        mean_vram,
        "peak_cpu_ram_gib":     peak_cpu,
        "mean_cpu_ram_gib":     mean_cpu,
        "ps_samples":           len(ps_samples),
        "processors":           procs[:120],
        "first_error":          (all_errors[0][:300] if all_errors else ""),
    }


# ── main ────────────────────────────────────────────────────────────────────────

async def main():
    # Build the full grid and print it upfront
    grid = list(iterproduct(PARALLEL_REQUESTS, CONTEXT_LENGTHS))
    total = len(grid)

    print(f"Ollama grid benchmark — host={OLLAMA_HOST}  model={MODEL}")
    print(f"Grid: {len(PARALLEL_REQUESTS)} parallel_requests × "
          f"{len(CONTEXT_LENGTHS)} context_lengths = {total} cells")

    # Print the grid table
    ctx_header = "  ".join(f"{c//1024:>6}k" for c in CONTEXT_LENGTHS)
    print(f"\n  {'n_req':>5}  {ctx_header}")
    print(f"  {'─'*5}  {'──────  '*len(CONTEXT_LENGTHS)}")
    for n in PARALLEL_REQUESTS:
        print(f"  {n:>5}  {'  ·     '*len(CONTEXT_LENGTHS)}")

    print(f"\nOutput: {OUTPUT_CSV}\n")

    await warmup_model()
    await asyncio.sleep(2)

    rows = []
    try:
        for idx, (n_req, ctx) in enumerate(grid, 1):
            row = await run_cell(n_req, ctx, idx, total)
            rows.append(row)
            await asyncio.sleep(SETTLE_SEC)
    finally:
        await unload_model()

    if not rows:
        print("No results collected.")
        return

    # ── write CSV ────────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Results written to {OUTPUT_CSV}")

    # ── print grid summary ───────────────────────────────────────────────────────
    # Index rows by (n_req, ctx_k) for easy lookup
    lookup = {(r["n_requests"], r["context_k"]): r for r in rows}
    ctx_ks = [c // 1024 for c in CONTEXT_LENGTHS]

    STATUS_SYMBOL = {
        STATUS_OK:      "  OK  ",
        STATUS_PARTIAL: "PARTL ",
        STATUS_OOM:     " OOM  ",
        STATUS_ERROR:   " ERR  ",
        None:           "  -   ",
    }

    print(f"\n── VRAM peak (GiB) ── status ──────────────────────────────────")
    header = f"  {'n_req':>5}  " + "  ".join(f"{c:>7}k" for c in ctx_ks)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for n in PARALLEL_REQUESTS:
        cells = []
        for c in ctx_ks:
            r = lookup.get((n, c))
            if r is None:
                cells.append("   -   ")
            else:
                sym = STATUS_SYMBOL.get(r["status"], "  ?   ")
                vram = r["peak_vram_gib"]
                cells.append(f"{sym}({vram:.1f}G)")
        print(f"  {n:>5}  " + "  ".join(cells))

    print("\nStatuses: OK=all passed  PARTL=some failed  OOM=VRAM exceeded  ERR=other error")


if __name__ == "__main__":
    asyncio.run(main())
