"""
benchmark_vram.py
==================
Measures VRAM usage for a single Ollama model across six sweeps:
concurrency (N) at min and max num_ctx, context window size (num_ctx)
at N=1 and N=max, and prompt length (padding) at min and max num_ctx.
Model is unloaded before each measurement for a clean read. VRAM is
read via `ollama ps`, which reports only what this Ollama instance
has loaded (unaffected by other processes sharing the GPU).
"""

import asyncio
import uuid

import matplotlib.pyplot as plt
import ollama

# ── Configuration ────────────────────────────────────────────────────────
HOST = "http://localhost:11441"
MODEL = "mistral-small3.2:24b-32k"

N_LIST = [1, 2, 4, 5, 10, 15, 20]
CTX_LIST = [8192, 16384, 32768, 32768*2]
PAD_LIST = [0, 2000, 6000, 12000]

PROMPT_BASE = "Write a sentence with each letter of the english alphabet used EXACTLY once:"
LOREM = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
         "tempor incididunt ut labore et dolore magna aliqua ut enim ad minim ") * 400

client = ollama.AsyncClient(host=HOST)


def make_prompt(pad_words=0):
    unique = f"[{uuid.uuid4().hex}]\n"
    padding = " ".join(LOREM.split()[:pad_words]) + "\n" if pad_words else ""
    return unique + padding + PROMPT_BASE


# ── Ollama helpers ───────────────────────────────────────────────────────
async def ollama_vram_mb():
    """VRAM (MB) Ollama reports for MODEL via `ollama ps`; 0 if not loaded."""
    resp = await client.ps()
    for m in resp.models:
        if m.model == MODEL:
            return m.size_vram / (1024 ** 2)
    return 0.0


async def unload_model():
    await client.chat(model=MODEL, messages=[], keep_alive=0)
    await asyncio.sleep(2)  # give the server a moment to release VRAM


async def run_request(num_ctx=None, pad_words=0):
    options = {"temperature": 0.0}
    if num_ctx:
        options["num_ctx"] = num_ctx
    async for chunk in await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": make_prompt(pad_words)}],
        stream=True,
        options=options,
    ):
        pass


async def measure_batch(n, num_ctx=None, pad_words=0):
    await unload_model()
    await asyncio.gather(*[run_request(num_ctx, pad_words) for _ in range(n)])
    return await ollama_vram_mb()


# ── Sweeps ──────────────────────────────────────────────────────────────
async def sweep_concurrency(num_ctx):
    print(f"Concurrency sweep (num_ctx={num_ctx})...")
    vram = []
    for n in N_LIST:
        v = await measure_batch(n, num_ctx=num_ctx)
        vram.append(v)
        print(f"  N={n:2d}  vram={v:.0f}MB")
    return vram


async def sweep_num_ctx(n=1):
    print(f"num_ctx sweep (N={n})...")
    vram = []
    for ctx in CTX_LIST:
        v = await measure_batch(n, num_ctx=ctx)
        vram.append(v)
        print(f"  num_ctx={ctx:6d}  vram={v:.0f}MB")
    return vram


async def sweep_prompt_padding(num_ctx):
    print(f"Prompt padding sweep (num_ctx={num_ctx})...")
    vram = []
    for pad in PAD_LIST:
        v = await measure_batch(1, num_ctx=num_ctx, pad_words=pad)
        vram.append(v)
        print(f"  pad_words={pad:6d}  vram={v:.0f}MB")
    return vram


# ── Plotting ────────────────────────────────────────────────────────────
def plot_results(conc_min, conc_max, ctx_vram_n1, ctx_vram_nmax, pad_min, pad_max):
    ctx_min, ctx_max = min(CTX_LIST), max(CTX_LIST)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f"VRAM Usage — {MODEL}")
    axes = axes.flatten()

    axes[0].plot(N_LIST, conc_min, marker="o")
    axes[0].set_title(f"VRAM vs. Concurrency (num_ctx={ctx_min})")
    axes[0].set_xlabel("Concurrent requests (N)")

    axes[1].plot(N_LIST, conc_max, marker="o")
    axes[1].set_title(f"VRAM vs. Concurrency (num_ctx={ctx_max})")
    axes[1].set_xlabel("Concurrent requests (N)")

    axes[2].plot(CTX_LIST, ctx_vram_n1, marker="o")
    axes[2].set_title("VRAM vs. num_ctx (N=1)")
    axes[2].set_xlabel("num_ctx (tokens)")

    axes[3].plot(CTX_LIST, ctx_vram_nmax, marker="o")
    axes[3].set_title(f"VRAM vs. num_ctx (N={max(N_LIST)})")
    axes[3].set_xlabel("num_ctx (tokens)")

    axes[4].plot(PAD_LIST, pad_min, marker="o")
    axes[4].set_title(f"VRAM vs. Prompt Length (num_ctx={ctx_min})")
    axes[4].set_xlabel("Extra input words")

    axes[5].plot(PAD_LIST, pad_max, marker="o")
    axes[5].set_title(f"VRAM vs. Prompt Length (num_ctx={ctx_max})")
    axes[5].set_xlabel("Extra input words")

    for ax in axes:
        ax.set_ylabel("VRAM used (MB)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("vram_benchmark.png", dpi=150)
    print("Saved → vram_benchmark.png")


# ── Main ────────────────────────────────────────────────────────────────
async def main():
    ctx_min, ctx_max = min(CTX_LIST), max(CTX_LIST)

    conc_min = await sweep_concurrency(num_ctx=ctx_min)
    conc_max = await sweep_concurrency(num_ctx=ctx_max)
    ctx_vram_n1 = await sweep_num_ctx(n=1)
    ctx_vram_nmax = await sweep_num_ctx(n=max(N_LIST))
    pad_min = await sweep_prompt_padding(num_ctx=ctx_min)
    pad_max = await sweep_prompt_padding(num_ctx=ctx_max)

    plot_results(conc_min, conc_max, ctx_vram_n1, ctx_vram_nmax, pad_min, pad_max)


if __name__ == "__main__":
    asyncio.run(main())
