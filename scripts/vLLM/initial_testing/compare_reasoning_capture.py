"""
Compare three clients against your local Qwen3-235B (vLLM, OpenAI-compatible
server) and check which ones surface the `reasoning_content` field that
vLLM's reasoning parser adds for reasoning models:

  1. Raw OpenAI Python client       -> captures it (raw JSON, nothing hidden)
  2. LangChain ChatOpenAI           -> currently DROPS it (open LangChain bug,
                                        see github.com/langchain-ai/langchain/issues/35059)
  3. LangChain ChatDeepSeek         -> captures it (built for this response shape,
                                        works fine against any OpenAI-compatible
                                        server, not just DeepSeek's own API)

Assumes your podman-hosted vLLM server is up and was started with a reasoning
parser enabled (e.g. --reasoning-parser qwen3 or --reasoning-parser deepseek_r1,
flag names vary by vLLM version).

Install deps first:
    pip install --break-system-packages openai langchain-openai langchain-deepseek

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
        print("\n✅ reasoning_content FOUND via raw client")
        print("Reasoning (truncated):", reasoning[:300], "...")
    else:
        print("\n❌ No reasoning_content field on message object")

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
        print("\n✅ reasoning_content FOUND via LangChain (in additional_kwargs/response_metadata)")
        print("Reasoning (truncated):", reasoning[:300], "...")
    else:
        print("\n❌ No reasoning_content surfaced by LangChain wrapper")
        print("   (LangChain's OpenAI wrapper only promotes fields it explicitly")
        print("    knows about — vendor-specific extras like reasoning_content")
        print("    are silently dropped unless your langchain-openai version")
        print("    has been updated to pass them through.)")

    return reasoning


def test_langchain_deepseek_client():
    line("3) LangChain ChatDeepSeek client (pointed at your local server)")
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(
        api_base=BASE_URL,
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
        print("\n✅ reasoning_content FOUND via LangChain ChatDeepSeek")
        print("Reasoning (truncated):", reasoning[:300], "...")
    else:
        print("\n❌ No reasoning_content surfaced (unexpected for ChatDeepSeek — check")
        print("   that your server's reasoning parser is actually enabled and firing)")

    return reasoning


def main():
    print(f"Target server: {BASE_URL}  |  model: {MODEL}")

    raw_reasoning = None
    lc_openai_reasoning = None
    lc_deepseek_reasoning = None

    try:
        raw_reasoning = test_raw_openai_client()
    except Exception as e:
        line("1) Raw OpenAI Python client — FAILED")
        print(f"Error: {e}")

    try:
        lc_openai_reasoning = test_langchain_openai_client()
    except Exception as e:
        line("2) LangChain ChatOpenAI client — FAILED")
        print(f"Error: {e}")

    try:
        lc_deepseek_reasoning = test_langchain_deepseek_client()
    except Exception as e:
        line("3) LangChain ChatDeepSeek client — FAILED")
        print(f"Error: {e}")

    line("SUMMARY")
    print(f"Raw OpenAI client captured reasoning:        {bool(raw_reasoning)}")
    print(f"LangChain ChatOpenAI captured reasoning:     {bool(lc_openai_reasoning)}")
    print(f"LangChain ChatDeepSeek captured reasoning:   {bool(lc_deepseek_reasoning)}")


if __name__ == "__main__":
    main()
