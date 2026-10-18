"""Prometheus-style metrics.

The API process exports its own HTTP latency histogram with prometheus_client. Celery runs several
worker processes, so they record into Redis hashes instead (atomic HINCRBY), and the API's /metrics
renders those alongside live gauges (queue depth, dead letters, review queue, open incidents).
"""

import math

import redis

HIST_PREFIX = "triage:metrics:hist:"
COUNTERS_KEY = "triage:metrics:counters"
BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30, 60)

HELP = {
    "triage_processing_seconds": "Time from receipt to triage complete (dedup + classification) per ticket",
    "triage_embed_seconds": "Embedding time per ticket",
    "triage_dedup_seconds": "Duplicate search + incident assignment time per ticket",
    "triage_dedup_lock_seconds": "Time each ticket holds the global dedup lock (the serialized part)",
    "triage_classify_seconds": "Classification + routing time per ticket (includes LLM fallback)",
    "triage_draft_seconds": "Reply draft time per ticket (retrieval + LLM)",
    "triage_tickets_processed_total": "Tickets triaged successfully",
    "triage_tickets_failed_total": "Tickets dead-lettered after exhausting retries",
    "triage_task_retries_total": "Task attempts that failed and were retried",
    "triage_incidents_created_total": "New incidents started",
    "triage_tickets_deduplicated_total": "Tickets attached to an existing incident",
    "triage_routed_auto_total": "Tickets auto-routed",
    "triage_routed_review_total": "Tickets sent to the review queue",
    "triage_llm_calls_total": "LLM calls (classification fallback + drafts)",
    "triage_llm_cost_usd_total": "Equivalent LLM cost in USD",
}


class WorkerMetrics:
    def __init__(self, client: redis.Redis):
        self.redis = client

    def observe(self, name: str, seconds: float) -> None:
        pipe = self.redis.pipeline(transaction=False)
        key = HIST_PREFIX + name
        for bound in BUCKETS:
            if seconds <= bound:
                pipe.hincrby(key, repr(float(bound)), 1)
        pipe.hincrby(key, "+Inf", 1)
        pipe.hincrbyfloat(key, "sum", seconds)
        pipe.hincrby(key, "count", 1)
        pipe.execute()

    def inc(self, name: str, amount: float = 1) -> None:
        self.redis.hincrbyfloat(COUNTERS_KEY, name, amount)


def _decode(d: dict) -> dict[str, str]:
    return {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in d.items()}


def format_metrics(histograms: dict[str, dict], counters: dict, gauges: dict[str, float]) -> str:
    lines = []
    for name, raw in sorted(histograms.items()):
        h = _decode(raw)
        lines += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} histogram"]
        for bound in BUCKETS:
            lines.append(f'{name}_bucket{{le="{bound}"}} {int(h.get(repr(float(bound)), 0))}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {int(h.get("+Inf", 0))}')
        lines.append(f"{name}_sum {float(h.get('sum', 0))}")
        lines.append(f"{name}_count {int(h.get('count', 0))}")
    for name, value in sorted(_decode(counters).items()):
        lines += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} counter", f"{name} {float(value)}"]
    for name, value in sorted(gauges.items()):
        lines += [f"# TYPE {name} gauge", f"{name} {value}"]
    return "\n".join(lines) + "\n"


def quantile_from_histogram(raw: dict, q: float) -> float:
    """Upper-bound estimate of a quantile from cumulative buckets (what Prometheus' histogram_quantile approximates)."""
    h = _decode(raw)
    total = int(h.get("count", 0))
    if not total:
        return math.nan
    target = q * total
    for bound in BUCKETS:
        if int(h.get(repr(float(bound)), 0)) >= target:
            return bound
    return math.inf
