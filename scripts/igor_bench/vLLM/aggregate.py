"""
Turns one batch's raw RequestResults + before/after /metrics snapshots into
a flat dict (one CSV row per rep), then aggregates reps for one grid point
(one input_length x concurrency pair) into mean/std.
"""

import statistics

import server_metrics


def _mean(x):
    return statistics.mean(x) if x else float("nan")


def _mn(x):
    return min(x) if x else float("nan")


def _mx(x):
    return max(x) if x else float("nan")


def summarize_rep(results, batch_start, batch_end, vram_gb,
                   metrics_before, metrics_after, ctx_reported,
                   input_tokens_target, concurrency):
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    wall_time = batch_end - batch_start

    ttft_thinking = [r.first_thinking - r.start for r in ok if r.first_thinking is not None]
    ttft_content = [r.first_content - r.start for r in ok if r.first_content is not None]

    per_req_tps = []
    for r in ok:
        anchor = r.first_thinking if r.first_thinking is not None else r.first_content
        anchor = anchor if anchor is not None else r.start
        dur = r.end - anchor
        if dur > 0 and r.output_tokens:
            per_req_tps.append(r.output_tokens / dur)

    total_output_tokens = sum(r.output_tokens for r in ok)

    deltas = server_metrics.compute_deltas(metrics_before, metrics_after)
    server_avgs = server_metrics.derive_averages(deltas)
    queue = server_metrics.queue_depth(metrics_after)

    row = {
        "input_tokens_target": input_tokens_target,
        "concurrency": concurrency,
        "wall_time_sec": wall_time,
        "ttft_thinking_sec": _mean(ttft_thinking),
        "ttft_thinking_sec_min": _mn(ttft_thinking),
        "ttft_thinking_sec_max": _mx(ttft_thinking),
        "ttft_content_sec": _mean(ttft_content),
        "ttft_content_sec_min": _mn(ttft_content),
        "ttft_content_sec_max": _mx(ttft_content),
        "tokens_per_sec": _mean(per_req_tps),
        "batch_tokens_per_sec": (total_output_tokens / wall_time) if wall_time > 0 else float("nan"),
        "output_tokens": total_output_tokens,
        "thinking_chunks": sum(r.thinking_chunks for r in ok),
        "content_chunks": sum(r.content_chunks for r in ok),
        "vram_gb": vram_gb,
        "kv_cache_usage_pct": server_metrics.kv_cache_usage_pct(metrics_after),
        "ctx_reported": ctx_reported,
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "error": "; ".join(sorted({r.error for r in failed}))[:300],
    }
    row.update(queue)          # num_requests_running / num_requests_waiting
    row.update(server_avgs)    # vllm:*_avg for every histogram scraped
    for k, v in deltas.items():
        if k.endswith("_total"):
            row[f"server_{k.replace('vllm:', '')}"] = v
    return row


def aggregate_reps(rep_rows: list) -> dict:
    """Collapses REPS rows for one grid point into mean (+ _std where
    there's more than one sample) while keeping the grid-key columns."""
    agg = {
        "input_tokens_target": rep_rows[0]["input_tokens_target"],
        "concurrency": rep_rows[0]["concurrency"],
        "reps": len(rep_rows),
        "ctx_reported": rep_rows[0]["ctx_reported"],
    }
    numeric_keys = [
        k for k, v in rep_rows[0].items()
        if isinstance(v, (int, float)) and k not in agg
    ]
    for k in numeric_keys:
        vals = [r[k] for r in rep_rows if r[k] == r[k]]  # drop NaN
        agg[k] = _mean(vals)
        if len(vals) > 1:
            agg[f"{k}_std"] = statistics.stdev(vals)
    agg["reps_ok"] = sum(1 for r in rep_rows if r["requests_failed"] == 0)
    agg["error"] = "; ".join(sorted({r["error"] for r in rep_rows if r["error"]}))
    return agg
