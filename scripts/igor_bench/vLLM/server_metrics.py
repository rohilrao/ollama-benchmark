"""
Reads vLLM's own Prometheus /metrics endpoint plus NVML and /v1/models.

/metrics gives vLLM-internal state that a client can't otherwise see:
queue depth, KV-cache occupancy, and server-side histograms for TTFT /
per-token latency / end-to-end latency. Comparing these against the
client-measured timings in runner.py shows how much of the client's
wall-clock time is queueing vs. actual generation.

Metric names vary a bit across vLLM versions (e.g. gpu_cache_usage_perc
was renamed kv_cache_usage_perc in some builds); this parser is generic -
it just captures every "vllm:*" series it finds, stripped of labels, so
it keeps working across versions without a hardcoded metric list.
"""

import re
import httpx

try:
    import pynvml
    pynvml.nvmlInit()
    HAVE_NVML = True
except Exception:
    HAVE_NVML = False

_METRIC_LINE = re.compile(r"^(vllm:[\w]+)(\{[^}]*\})?\s+([-+0-9.eE]+)")


def parse_prometheus(text: str) -> dict:
    """Sum all label variants of each vllm:* series into one flat dict.
    Bucket lines (histogram buckets) are skipped - only _sum/_count/_total
    and plain gauges are kept, which is enough to derive averages."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, _labels, value = m.groups()
        if name.endswith("_bucket"):
            continue
        out[name] = out.get(name, 0.0) + float(value)
    return out


async def fetch_metrics(http: httpx.AsyncClient, url: str) -> dict:
    try:
        r = await http.get(url, timeout=5)
        r.raise_for_status()
        return parse_prometheus(r.text)
    except Exception:
        return {}


def compute_deltas(before: dict, after: dict) -> dict:
    """Deltas for counters/histograms (_total, _sum, _count). Gauges (queue
    depth, cache %) aren't diffed - use the 'after' snapshot for those."""
    deltas = {}
    for k in set(before) | set(after):
        if not (k.endswith("_total") or k.endswith("_sum") or k.endswith("_count")):
            continue
        b, a = before.get(k), after.get(k)
        if b is not None and a is not None:
            deltas[k] = a - b
    return deltas


def derive_averages(deltas: dict) -> dict:
    """For every '<x>_sum' / '<x>_count' pair, compute '<x>_avg' - this is
    the server-side average latency for that histogram over the batch."""
    out = {}
    for k in list(deltas):
        if not k.endswith("_sum"):
            continue
        base = k[: -len("_sum")]
        count_key = base + "_count"
        if deltas.get(count_key, 0) > 0:
            out[f"{base}_avg"] = deltas[k] / deltas[count_key]
    return out


def kv_cache_usage_pct(metrics: dict) -> float:
    for key in ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"):
        if key in metrics:
            return metrics[key] * 100
    return float("nan")


def queue_depth(metrics: dict) -> dict:
    return {
        "num_requests_running": metrics.get("vllm:num_requests_running", float("nan")),
        "num_requests_waiting": metrics.get("vllm:num_requests_waiting", float("nan")),
    }


async def read_vram_gb() -> float:
    if not HAVE_NVML:
        return float("nan")
    try:
        total_used = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            total_used += info.used
        return total_used / (1024 ** 3)
    except Exception:
        return float("nan")


async def get_ctx_reported(http: httpx.AsyncClient, models_url: str) -> int:
    try:
        r = await http.get(models_url, timeout=5)
        r.raise_for_status()
        data = r.json()
        return int(data["data"][0].get("max_model_len", -1))
    except Exception:
        return -1
