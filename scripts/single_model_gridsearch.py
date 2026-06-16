#!/usr/bin/env python3
"""
Ollama grid-search benchmark
Runs every (parallel_requests × context_length) combination, polls `ollama ps`,
writes results to CSV.
"""

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST       = "http://localhost:11436"
MODEL             = "llama3.2"
PROMPT            = "Explain gravity in one sentence."

PARALLEL_REQUESTS = [1, 4, 8, 12, 16]
CONTEXT_LENGTHS   = [8_192, 32_768, 65_536, 131_072]

POLL_INTERVAL_SEC  = 0.5
REQUEST_TIMEOUT    = 300
SETTLE_SEC         = 3
OUTPUT_CSV         = "ollama_grid_results.csv"

OOM_SIGNALS = [
    "out of memory", "oom", "cuda error", "not enough memory",
    "failed to allocate", "no space left", "exit status 1",
]
# ───────────────────────────────────────────────────────────────────────────────

import asyncio, csv, json, re, subprocess, time, urllib.error, urllib.request
from datetime import datetime
from itertools import product as iterproduct


def parse_size_gib(s: str) -> float:
    """'3.5 GiB' / '512 MiB' → float GiB. Returns 0 on failure."""
    m = re.match(r"([\d.]+)\s*([KMGT]?i?B)", s.strip(), re.IGNORECASE)
    if not m:
        return 0.0
    v, u = float(m.group(1)), m.group(2).upper()
    bytes_ = v * {"B":1,"KIB":1024,"MIB":1024**2,"GIB":1024**3,
                  "KB":1000,"MB":1e6,"GB":1e9}.get(u, 1)
    return bytes_ / 1024**3


def is_oom(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in OOM_SIGNALS)


def sample_ps() -> tuple[float, float]:
    """Returns (vram_gib, cpu_gib) from `ollama ps`. Returns (0,0) on error."""
    try:
        env = {"OLLAMA_HOST": OLLAMA_HOST.replace("http://", "")}
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                             timeout=5, env={**__import__("os").environ, **env}).stdout
        vram = cpu = 0.0
        for line in out.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            size = parse_size_gib(parts[2])
            m = re.search(r"(\d+)%\s*GPU", " ".join(parts[3:]), re.IGNORECASE)
            gpu = float(m.group(1)) / 100 if m else 0.0
            vram += size * gpu
            cpu  += size * (1 - gpu)
        return vram, cpu
    except Exception:
        return 0.0, 0.0


def post(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


async def call_once(i: int, ctx: int) -> dict:
    loop = asyncio.get_event_loop()
    t0   = time.perf_counter()
    try:
        resp = await loop.run_in_executor(None, lambda: post(
            f"{OLLAMA_HOST}/api/generate",
            {"model": MODEL, "prompt": PROMPT, "stream": False,
             "options": {"num_ctx": ctx}},
            timeout=REQUEST_TIMEOUT,
        ))
        elapsed = time.perf_counter() - t0
        if "error" in resp:
            return {"ok": False, "elapsed": elapsed, "oom": is_oom(resp["error"]),
                    "error": resp["error"]}
        return {"ok": True, "elapsed": elapsed, "oom": False, "error": ""}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"ok": False, "elapsed": time.perf_counter()-t0,
                "oom": is_oom(body), "error": f"HTTP {e.code}: {body[:200]}"}
    except Exception as e:
        err = str(e)
        return {"ok": False, "elapsed": time.perf_counter()-t0,
                "oom": is_oom(err), "error": err}


async def run_cell(n: int, ctx: int, idx: int, total: int) -> dict:
    print(f"\n[{idx}/{total}]  {n} req × {ctx//1024}k ctx  "
          f"({datetime.now():%H:%M:%S})", flush=True)

    vram_samples, cpu_samples = [], []
    done = asyncio.Event()

    async def poller():
        while not done.is_set():
            v, c = sample_ps()
            vram_samples.append(v)
            cpu_samples.append(c)
            await asyncio.sleep(POLL_INTERVAL_SEC)

    poll = asyncio.create_task(poller())
    t0   = time.perf_counter()
    results = await asyncio.gather(*[call_once(i, ctx) for i in range(n)])
    elapsed  = time.perf_counter() - t0
    done.set(); await poll

    ok  = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    status = ("OK"       if not bad
              else "PARTIAL"  if ok
              else "VRAM_OOM" if any(r["oom"] for r in bad)
              else "ERROR")

    peak_vram = max(vram_samples, default=0)
    peak_cpu  = max(cpu_samples,  default=0)
    mean_elapsed = sum(r["elapsed"] for r in ok) / len(ok) if ok else None

    icon = {"OK":"Ok","PARTIAL":"Partial","VRAM_OOM":"OoM","ERROR":"Err"}[status]
    print(f"  {icon} {status}  {len(ok)}/{n} ok  "
          f"wall={elapsed:.1f}s  VRAM={peak_vram:.2f}G  CPU={peak_cpu:.2f}G")
    if bad:
        print(f"    error: {bad[0]['error'][:120]}")

    return {
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "model":             MODEL,
        "n_requests":        n,
        "context_k":         ctx // 1024,
        "status":            status,
        "successes":         len(ok),
        "failures":          len(bad),
        "wall_sec":          round(elapsed, 3),
        "mean_req_sec":      round(mean_elapsed, 3) if mean_elapsed else "",
        "peak_vram_gib":     round(peak_vram, 3),
        "peak_cpu_gib":      round(peak_cpu,  3),
        "first_error":       (bad[0]["error"][:300] if bad else ""),
    }


async def warmup():
    print(f" Warming up {MODEL} …", flush=True)
    try:
        r = await asyncio.get_event_loop().run_in_executor(None, lambda: post(
            f"{OLLAMA_HOST}/api/generate",
            {"model": MODEL, "prompt": "Hi.", "stream": False,
             "options": {"num_ctx": 2048}, "keep_alive": "30m"},
            timeout=180,
        ))
        print(f"    done (gen_tokens={r.get('eval_count','?')})")
    except Exception as e:
        print(f"    warmup failed: {e}")


async def unload():
    print(f"\n🗑  Unloading {MODEL} …", flush=True)
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: post(
            f"{OLLAMA_HOST}/api/generate",
            {"model": MODEL, "prompt": "", "keep_alive": 0}, timeout=30,
        ))
    except Exception:
        pass  # empty-body response is normal for unload


async def main():
    grid = list(iterproduct(PARALLEL_REQUESTS, CONTEXT_LENGTHS))
    print(f"Ollama benchmark  host={OLLAMA_HOST}  model={MODEL}")
    print(f"{len(PARALLEL_REQUESTS)} × {len(CONTEXT_LENGTHS)} = {len(grid)} cells\n")

    await warmup()
    await asyncio.sleep(2)

    rows = []
    try:
        for idx, (n, ctx) in enumerate(grid, 1):
            rows.append(await run_cell(n, ctx, idx, len(grid)))
            await asyncio.sleep(SETTLE_SEC)
    finally:
        await unload()

    if not rows:
        print("No results."); return

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # ── summary grid ─────────────────────────────────────────────────────────
    ctx_ks = [c // 1024 for c in CONTEXT_LENGTHS]
    lookup = {(r["n_requests"], r["context_k"]): r for r in rows}
    sym    = {"OK":"  OK  ","PARTIAL":"PARTL ","VRAM_OOM":" OOM  ","ERROR":" ERR  "}

    print(f"\n{OUTPUT_CSV}")
    print(f"\n── peak VRAM (GiB) / status ──")
    print("  req  " + "  ".join(f"{c:>6}k" for c in ctx_ks))
    for n in PARALLEL_REQUESTS:
        cells = []
        for c in ctx_ks:
            r = lookup.get((n, c))
            cells.append(f"{sym.get(r['status'],'  ?   ')}({r['peak_vram_gib']:.1f}G)"
                         if r else "   -   ")
        print(f"  {n:>3}  " + "  ".join(cells))


if __name__ == "__main__":
    asyncio.run(main())
