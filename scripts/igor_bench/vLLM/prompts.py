"""
Builds prompts sized to a target input-token length, and optionally checks
the real count via vLLM's /tokenize endpoint so the CSV records actual
input length, not just the requested target (word/token ratio is only an
estimate and will drift from the model's real tokenizer).
"""

import httpx

_FILLER = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
    "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
    "occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum. "
)
_FILLER_WORDS = _FILLER.split()
_INSTRUCTION = "\n\nBased on the text above, summarize it in exactly one sentence."


def build_prompt(target_tokens: int, words_per_token: float) -> str:
    target_words = max(1, int(target_tokens * words_per_token))
    reps_needed = target_words // len(_FILLER_WORDS) + 2
    words = (_FILLER_WORDS * reps_needed)[:target_words]
    return " ".join(words) + _INSTRUCTION


async def measure_input_tokens(http: httpx.AsyncClient, tokenize_url: str,
                                model: str, prompt: str) -> int:
    """Returns the real token count from /tokenize, or -1 if the endpoint
    isn't available (older vLLM builds, or a proxy that doesn't expose it)."""
    try:
        r = await http.post(tokenize_url, json={"model": model, "prompt": prompt}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "count" in data:
            return int(data["count"])
        if "tokens" in data:
            return len(data["tokens"])
        return -1
    except Exception:
        return -1
