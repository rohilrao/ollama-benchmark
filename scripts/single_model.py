import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def bytes_to_gb(x: int | float | None) -> float:
    if not x:
        return 0.0
    return round(x / 1024**3, 2)


def generate_once(base_url: str, model: str, i: int, prompt: str, num_ctx: int | None):
    payload = {
        "model": model,
        "prompt": f"{prompt}\n\nRequest ID: {i}",
        "stream": False,
        "keep_alive": "10m",
    }

    if num_ctx is not None:
        payload["options"] = {"num_ctx": num_ctx}

    start = time.time()
    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=600)
    elapsed = time.time() - start

    r.raise_for_status()
    data = r.json()

    return {
        "request_id": i,
        "elapsed_sec": round(elapsed, 2),
        "response_preview": data.get("response", "")[:120].replace("\n", " "),
    }


def fetch_ollama_ps(base_url: str):
    r = requests.get(f"{base_url}/api/ps", timeout=30)
    r.raise_for_status()
    return r.json()


def print_ps(ps_data: dict):
    models = ps_data.get("models", [])

    if not models:
        print("\nNo models currently loaded according to /api/ps.")
        return

    print("\nOllama /api/ps memory view")
    print("-" * 80)

    for m in models:
        print(f"Model      : {m.get('name') or m.get('model')}")
        print(f"Processor  : {m.get('processor')}")
        print(f"Size       : {bytes_to_gb(m.get('size'))} GB")
        print(f"VRAM       : {bytes_to_gb(m.get('size_vram'))} GB")
        print(f"Expires at : {m.get('expires_at')}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Send N parallel requests to a containerized Ollama server and inspect /api/ps memory."
    )

    parser.add_argument(
        "--url",
        default="http://localhost:11436",
        help="Ollama host URL. Use the exposed host port, e.g. http://localhost:11436",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, e.g. qwen3:32b, llama3.1:8b, etc.",
    )

    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=8,
        help="Number of concurrent requests.",
    )

    parser.add_argument(
        "--prompt",
        default="Explain KV caching in LLM inference in 5 concise points.",
        help="Prompt to send to each request.",
    )

    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Optional context size, e.g. 8192 or 16384.",
    )

    args = parser.parse_args()

    print(f"Target Ollama server : {args.url}")
    print(f"Model                : {args.model}")
    print(f"Concurrent requests  : {args.num_requests}")

    print("\nBefore requests:")
    try:
        print_ps(fetch_ollama_ps(args.url))
    except Exception as e:
        print(f"Could not fetch /api/ps before requests: {e}")

    print("\nSending parallel requests...")
    results = []

    with ThreadPoolExecutor(max_workers=args.num_requests) as executor:
        futures = [
            executor.submit(
                generate_once,
                args.url,
                args.model,
                i,
                args.prompt,
                args.num_ctx,
            )
            for i in range(args.num_requests)
        ]

        for fut in as_completed(futures):
            try:
                result = fut.result()
                results.append(result)
                print(
                    f"[OK] request={result['request_id']} "
                    f"time={result['elapsed_sec']}s "
                    f"preview={result['response_preview']!r}"
                )
            except Exception as e:
                print(f"[FAIL] {e}")

    print("\nAfter requests:")
    try:
        print_ps(fetch_ollama_ps(args.url))
    except Exception as e:
        print(f"Could not fetch /api/ps after requests: {e}")

    if results:
        times = [r["elapsed_sec"] for r in results]
        print("\nLatency summary")
        print("-" * 80)
        print(f"Completed requests : {len(results)} / {args.num_requests}")
        print(f"Min time           : {min(times)} s")
        print(f"Max time           : {max(times)} s")
        print(f"Avg time           : {round(sum(times) / len(times), 2)} s")


if __name__ == "__main__":
    main()
