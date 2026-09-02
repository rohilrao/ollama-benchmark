uv run src/vLLM/debug_reasoning.py 
=== RAW SSE (first 5 data lines) ===
{"id":"chatcmpl-919b13ab4ff836ac","object":"chat.completion.chunk","created":1788338477,"model":"/models/qwen235b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null,"prompt_text":null}
{"id":"chatcmpl-919b13ab4ff836ac","object":"chat.completion.chunk","created":1788338477,"model":"/models/qwen235b","choices":[{"index":0,"delta":{"reasoning":"\n"},"logprobs":null,"finish_reason":null,"token_ids":null}]}
{"id":"chatcmpl-919b13ab4ff836ac","object":"chat.completion.chunk","created":1788338477,"model":"/models/qwen235b","choices":[{"index":0,"delta":{"reasoning":"Okay"},"logprobs":null,"finish_reason":null,"token_ids":null}]}
{"id":"chatcmpl-919b13ab4ff836ac","object":"chat.completion.chunk","created":1788338477,"model":"/models/qwen235b","choices":[{"index":0,"delta":{"reasoning":","},"logprobs":null,"finish_reason":null,"token_ids":null}]}
{"id":"chatcmpl-919b13ab4ff836ac","object":"chat.completion.chunk","created":1788338477,"model":"/models/qwen235b","choices":[{"index":0,"delta":{"reasoning":" so"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

=== RAW: fields seen in delta across whole stream ===
{'reasoning', 'content', 'role'}
RAW reasoning_content text: ''
RAW content text:          ''

=== CLIENT (openai AsyncOpenAI) ===
getattr fields kept (model_dump): {'refusal', 'content', 'role', 'reasoning', 'function_call', 'tool_calls'}
getattr(delta, 'reasoning_content') text: ''
model_dump()['reasoning_content'] text:   ''
getattr(delta, 'content') text:           ''

=== DIAGNOSIS ===
Server never sent reasoning_content on the wire. Check for <think> tags inside RAW content_text above, and confirm enable_thinking is actually reaching the chat template.
an error occurred during closing of asynchronous generator <async_generator object PoolByteStream.__aiter__ at 0x7f1ee4643f10>
asyncgen: <async_generator object PoolByteStream.__aiter__ at 0x7f1ee4643f10>
Traceback (most recent call last):
  File "/home/rrao/projectcode/inference-bench/inference-bench/.venv/lib64/python3.13/site-packages/httpcore2/_async/connection_pool.py", line 427, in __aiter__
    yield chunk
GeneratorExit

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/rrao/projectcode/inference-bench/inference-bench/.venv/lib64/python3.13/site-packages/httpcore2/_async/connection_pool.py", line 425, in __aiter__
    async with safe_async_iterate(self._stream) as iterator:
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/usr/lib64/python3.13/contextlib.py", line 271, in __aexit__
    raise RuntimeError("generator didn't stop after athrow()")
RuntimeError: generator didn't stop after athrow()
"""
Runs the SAME request two ways and compares what each path sees:

  1. RAW    - httpx reads the SSE stream directly, parses each `data:` line
              as JSON ourselves. This is ground truth for what vLLM sends.
  2. CLIENT - openai.AsyncOpenAI parses the identical request. We check both
              getattr(delta, "reasoning_content", None) (what runner.py uses)
              and delta.model_dump() (every field the client actually kept).

If RAW sees reasoning_content but CLIENT's getattr doesn't -> client-side
parsing bug. If RAW never sees it either -> server isn't emitting it (check
<think> tags inside plain content, or the enable_thinking kwarg isn't
reaching the chat template). If CLIENT's getattr misses it but
model_dump() has it -> something is wrong with our getattr, not the client.

Run: python debug_compare_client_vs_raw.py
"""

import asyncio
import json

import httpx
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8556/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
MODEL_NAME = "/models/qwen3-235b"
PROMPT = "What is 17 * 24? Think step by step."

PAYLOAD_EXTRAS = {
    "chat_template_kwargs": {"enable_thinking": True},
}


async def run_raw():
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "max_tokens": 60,
        "temperature": 0,
        **PAYLOAD_EXTRAS,
    }

    seen_fields = set()
    reasoning_text = ""
    content_text = ""
    raw_samples = []

    async with httpx.AsyncClient(timeout=30) as http:
        async with http.stream("POST", CHAT_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                if len(raw_samples) < 5:
                    raw_samples.append(data)

                obj = json.loads(data)
                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                seen_fields.update(delta.keys())
                reasoning_text += delta.get("reasoning_content") or ""
                content_text += delta.get("content") or ""

    return {
        "seen_fields": seen_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "raw_samples": raw_samples,
    }


async def run_client():
    client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")

    getattr_reasoning = ""
    getattr_content = ""
    dump_fields = set()
    dump_reasoning = ""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        max_tokens=60,
        temperature=0,
        extra_body=PAYLOAD_EXTRAS,
    )

    async for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Path A: what runner.py actually does.
        r = getattr(delta, "reasoning_content", None)
        c = getattr(delta, "content", None)
        getattr_reasoning += r or ""
        getattr_content += c or ""

        # Path B: full dump, to see every field the client kept regardless
        # of whether our getattr call happens to name it right.
        dumped = delta.model_dump()
        dump_fields.update(dumped.keys())
        dump_reasoning += dumped.get("reasoning_content") or ""

    return {
        "getattr_reasoning": getattr_reasoning,
        "getattr_content": getattr_content,
        "dump_fields": dump_fields,
        "dump_reasoning": dump_reasoning,
    }


async def main():
    raw = await run_raw()
    client_res = await run_client()

    print("=== RAW SSE (first 5 data lines) ===")
    for line in raw["raw_samples"]:
        print(line)

    print("\n=== RAW: fields seen in delta across whole stream ===")
    print(raw["seen_fields"])
    print("RAW reasoning_content text:", repr(raw["reasoning_text"][:200]))
    print("RAW content text:         ", repr(raw["content_text"][:200]))

    print("\n=== CLIENT (openai AsyncOpenAI) ===")
    print("getattr fields kept (model_dump):", client_res["dump_fields"])
    print("getattr(delta, 'reasoning_content') text:", repr(client_res["getattr_reasoning"][:200]))
    print("model_dump()['reasoning_content'] text:  ", repr(client_res["dump_reasoning"][:200]))
    print("getattr(delta, 'content') text:          ", repr(client_res["getattr_content"][:200]))

    print("\n=== DIAGNOSIS ===")
    raw_has_reasoning = bool(raw["reasoning_text"])
    client_getattr_has_reasoning = bool(client_res["getattr_reasoning"])
    client_dump_has_reasoning = bool(client_res["dump_reasoning"])

    if not raw_has_reasoning:
        print("Server never sent reasoning_content on the wire. Check for "
              "<think> tags inside RAW content_text above, and confirm "
              "enable_thinking is actually reaching the chat template.")
    elif raw_has_reasoning and not client_dump_has_reasoning:
        print("Server sent it, but the openai client dropped it entirely "
              "(even via model_dump). Client-side parsing/version issue.")
    elif client_dump_has_reasoning and not client_getattr_has_reasoning:
        print("Client kept the field (model_dump has it) but getattr() "
              "missed it - bug in our getattr call, not the client itself.")
    else:
        print("Reasoning tokens are flowing correctly end to end.")


if __name__ == "__main__":
    asyncio.run(main())
