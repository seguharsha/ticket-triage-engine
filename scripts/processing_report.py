"""Measure what the worker actually did for recently received tickets (run after a replay or load test).

  python -m scripts.processing_report --minutes 15

Processing latency is processed_at - received_at (server clock), i.e. queue wait + embed + dedup +
classification. Also reports failures, routing split and LLM cost per ticket.
"""

import argparse

from sqlalchemy import text

from app.db import SyncSession

QUERY = text("""
WITH recent AS (
    SELECT * FROM tickets WHERE received_at >= now() - make_interval(mins => :minutes) AND source <> 'history'
      AND idempotency_key LIKE :key_like
)
SELECT
    count(*)                                                     AS received,
    count(*) FILTER (WHERE status = 'processed')                 AS processed,
    count(*) FILTER (WHERE status = 'failed')                    AS failed,
    count(*) FILTER (WHERE status IN ('received', 'processing')) AS pending,
    count(*) FILTER (WHERE routing = 'auto')                     AS auto_routed,
    count(*) FILTER (WHERE routing = 'review')                   AS to_review,
    count(DISTINCT incident_id)                                  AS incidents,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY extract(epoch FROM processed_at - received_at)) AS p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM processed_at - received_at)) AS p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY extract(epoch FROM processed_at - received_at)) AS p99,
    extract(epoch FROM max(processed_at) - min(received_at))     AS span_s
FROM recent
""")

COST = text("""
SELECT coalesce(sum(u.cost_usd), 0) AS cost, count(u.id) AS calls,
       count(*) FILTER (WHERE u.purpose = 'classify') AS classify_calls
FROM llm_usage u JOIN tickets t ON t.id = u.ticket_id
WHERE t.received_at >= now() - make_interval(mins => :minutes) AND t.idempotency_key LIKE :key_like
""")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, default=15)
    p.add_argument("--key-prefix", default="", help="only tickets whose idempotency key starts with this, e.g. replay: or load-")
    args = p.parse_args()
    with SyncSession() as session:
        params = {"minutes": args.minutes, "key_like": args.key_prefix + "%"}
        r = session.execute(QUERY, params).one()
        c = session.execute(COST, params).one()
    if not r.received:
        raise SystemExit("no tickets received in that window")
    print(f"received {r.received}  processed {r.processed}  failed {r.failed}  still pending {r.pending}")
    if r.p50 is not None:
        print(f"processing latency s: p50 {r.p50:.2f}  p95 {r.p95:.2f}  p99 {r.p99:.2f}")
        print(f"throughput: {r.processed / r.span_s:.1f} tickets/s over {r.span_s:.0f}s")
    routed = (r.auto_routed or 0) + (r.to_review or 0)
    if routed:
        print(f"routing: {r.auto_routed / routed:.1%} auto, {r.to_review / routed:.1%} review; "
              f"{r.incidents} incidents")
    print(f"LLM: {c.calls} calls ({c.classify_calls} classification), ${c.cost:.5f} total, "
          f"${c.cost / r.received:.6f} per ticket")


if __name__ == "__main__":
    main()
