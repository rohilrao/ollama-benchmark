"""
benchmark_vram.py
==================
Measures VRAM usage for a single Ollama model across three sweeps:
concurrency (N), context window size (num_ctx), and prompt length
(padding). Model is unloaded before each measurement for a clean read.
VRAM is read via `ollama ps`, which reports only what this Ollama
instance has loaded (unaffected by other processes sharing the GPU).
"""

import asyncio
import uuid

import matplotlib.pyplot as plt
import ollama

# ── Configuration ────────────────────────────────────────────────────────
HOST = "http://localhost:11440"
MODEL = "mistral-small3.2:24b"

N_LIST = [1, 2, 4, 5]
CTX_LIST = [8192, 16384, 32768]
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
async def sweep_concurrency():
    print("Concurrency sweep...")
    vram = []
    for n in N_LIST:
        v = await measure_batch(n, num_ctx=8192)
        vram.append(v)
        print(f"  N={n:2d}  vram={v:.0f}MB")
    return vram


async def sweep_num_ctx():
    print("num_ctx sweep...")
    vram = []
    for ctx in CTX_LIST:
        v = await measure_batch(1, num_ctx=ctx)
        vram.append(v)
        print(f"  num_ctx={ctx:6d}  vram={v:.0f}MB")
    return vram


async def sweep_prompt_padding():
    print("Prompt padding sweep...")
    vram = []
    for pad in PAD_LIST:
        v = await measure_batch(1, num_ctx=max(CTX_LIST), pad_words=pad)
        vram.append(v)
        print(f"  pad_words={pad:6d}  vram={v:.0f}MB")
    return vram


# ── Plotting ────────────────────────────────────────────────────────────
def plot_results(concurrency_vram, ctx_vram, padding_vram):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"VRAM Usage — {MODEL}")

    axes[0].plot(N_LIST, concurrency_vram, marker="o")
    axes[0].set_title("VRAM vs. Concurrency (num_ctx=8192)")
    axes[0].set_xlabel("Concurrent requests (N)")

    axes[1].plot(CTX_LIST, ctx_vram, marker="o")
    axes[1].set_title("VRAM vs. num_ctx (N=1)")
    axes[1].set_xlabel("num_ctx (tokens)")

    axes[2].plot(PAD_LIST, padding_vram, marker="o")
    axes[2].set_title(f"VRAM vs. Prompt Length (num_ctx={max(CTX_LIST)})")
    axes[2].set_xlabel("Extra input words")

    for ax in axes:
        ax.set_ylabel("VRAM used (MB)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("vram_benchmark.png", dpi=150)
    print("Saved → vram_benchmark.png")


# ── Main ────────────────────────────────────────────────────────────────
async def main():
    concurrency_vram = await sweep_concurrency()
    ctx_vram = await sweep_num_ctx()
    padding_vram = await sweep_prompt_padding()

    plot_results(concurrency_vram, ctx_vram, padding_vram)


if __name__ == "__main__":
    asyncio.run(main())
