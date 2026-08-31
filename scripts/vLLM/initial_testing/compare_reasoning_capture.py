"""
Compare: does the raw OpenAI client vs the LangChain ChatOpenAI client
surface Qwen3's reasoning trace (the `reasoning_content` field that
vLLM / sglang / etc. add for reasoning models like Qwen3-235B)?

Assumes an OpenAI-compatible server (e.g. vLLM running Qwen3-235B in your
podman container) is up and reachable, and that it was started with a
reasoning parser enabled (for vLLM: --enable-reasoning --reasoning-parser
qwen3, or the newer --reasoning-parser deepseek_r1 depending on version).

Install deps first:
    pip install --break-system-packages openai langchain-openai

Usage:
    python compare_reasoning_capture.py
"""

import json
import os

# ---- Config: point this at your podman-hosted vLLM/OpenAI-compatible server ----
BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("QWEN_API_KEY", "EMPTY")   # local servers usually ignore this
MODEL = os.environ.get("QWEN_MODEL", "qwen3-235b")

PROMPT = "A farmer has 17 sheep. All but 9 die. How many are left? Think step by step."


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_raw_openai_client():
    line("1) Raw OpenAI Python client")
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=1024,
    )

    msg = resp.choices[0].message
    # Dump everything the SDK actually got back (incl. any extra/non-standard fields)
    raw = msg.model_dump()
    print("message.model_dump() keys:", list(raw.keys()))

    reasoning = raw.get("reasoning_content") or raw.get("reasoning")
    if reasoning:
        print("\nreasoning_content FOUND via raw client")
        print("Reasoning (truncated):", reasoning[:300], "...")
    else:
        print("\nNo reasoning_content field on message object")

    print("\nFinal content (truncated):", (msg.content or "")[:300])

    # Also show the fully raw JSON in case the field is nested elsewhere
    print("\nFull raw JSON dump of choice[0]:")
    print(json.dumps(resp.choices[0].model_dump(), indent=2)[:1500])

    return reasoning


def test_langchain_openai_client():
    line("2) LangChain ChatOpenAI client")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=1024,
    )

    ai_msg = llm.invoke(PROMPT)

    print("AIMessage attributes checked:")
    print("  .content (truncated):", str(ai_msg.content)[:300])
    print("  .additional_kwargs keys:", list(ai_msg.additional_kwargs.keys()))
    print("  .response_metadata keys:", list(ai_msg.response_metadata.keys()))

    reasoning = (
        ai_msg.additional_kwargs.get("reasoning_content")
        or ai_msg.additional_kwargs.get("reasoning")
        or ai_msg.response_metadata.get("reasoning_content")
    )

    if reasoning:
        print("\nreasoning_content FOUND via LangChain (in additional_kwargs/response_metadata)")
        print("Reasoning (truncated):", reasoning[:300], "...")
    else:
        print("\nNo reasoning_content surfaced by LangChain wrapper")
        print("   (LangChain's OpenAI wrapper only promotes fields it explicitly")
        print("    knows about — vendor-specific extras like reasoning_content")
        print("    are silently dropped unless your langchain-openai version")
        print("    has been updated to pass them through.)")

    return reasoning


def main():
    print(f"Target server: {BASE_URL}  |  model: {MODEL}")

    raw_reasoning = None
    lc_reasoning = None

    try:
        raw_reasoning = test_raw_openai_client()
    except Exception as e:
        line("1) Raw OpenAI Python client — FAILED")
        print(f"Error: {e}")

    try:
        lc_reasoning = test_langchain_openai_client()
    except Exception as e:
        line("2) LangChain ChatOpenAI client — FAILED")
        print(f"Error: {e}")

    line("SUMMARY")
    print(f"Raw OpenAI client captured reasoning:      {bool(raw_reasoning)}")
    print(f"LangChain ChatOpenAI captured reasoning:   {bool(lc_reasoning)}")


if __name__ == "__main__":
    main()
