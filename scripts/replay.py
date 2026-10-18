"""Replay the dataset into the API in created_at order, and report intake latency.

  python -m scripts.replay                         # all tickets, as fast as --concurrency allows
  python -m scripts.replay --split test --rate 50  # held-out set at 50 tickets/s

Writes data/replay_map.jsonl (dataset id -> API ticket id) so evaluation scripts can join results
back to labels. Idempotency keys are derived from dataset ids, so re-running is safe.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--data", type=Path, default=Path("data/tickets.jsonl"))
    p.add_argument("--split", choices=["train", "test"], help="only replay one split")
    p.add_argument("--limit", type=int)
    p.add_argument("--rate", type=float, help="tickets per second (default: unthrottled)")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--no-timestamps", action="store_true", help="let the API stamp created_at with now()")
    p.add_argument("--map-out", type=Path, default=Path("data/replay_map.jsonl"))
    args = p.parse_args()

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted((r for r in rows if not args.split or r["split"] == args.split), key=lambda r: r["created_at"])
    rows = rows[: args.limit] if args.limit else rows

    sem = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    results: list[dict] = []

    async def send(client: httpx.AsyncClient, row: dict) -> None:
        data = {"subject": row["subject"], "body": row["body"]}
        if not args.no_timestamps:
            data["created_at"] = row["created_at"]
        async with sem:
            start = time.perf_counter()
            try:
                resp = await client.post("/tickets", data=data, headers={"Idempotency-Key": f"replay:{row['id']}"})
                status, ticket_id = resp.status_code, resp.json().get("id") if resp.status_code < 300 else None
            except httpx.HTTPError as exc:
                status, ticket_id = f"error:{exc.__class__.__name__}", None
            latencies.append((time.perf_counter() - start) * 1000)
        results.append({"dataset_id": row["id"], "ticket_id": ticket_id, "status": status})

    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.api, timeout=30) as client:
        tasks = []
        for i, row in enumerate(rows):
            if args.rate:
                # Schedule against a fixed clock so slow responses don't lower the offered rate.
                delay = started + i / args.rate - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(client, row)))
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started

    args.map_out.parent.mkdir(parents=True, exist_ok=True)
    with args.map_out.open("w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda r: r["dataset_id"]):
            f.write(json.dumps(r) + "\n")

    by_status: dict = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"sent {len(rows)} tickets in {elapsed:.1f}s ({len(rows) / elapsed:.1f}/s); statuses {by_status}")
    if latencies:
        print(f"intake latency ms: p50 {statistics.median(latencies):.1f}  p95 {percentile(latencies, 0.95):.1f}  "
              f"p99 {percentile(latencies, 0.99):.1f}  max {max(latencies):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
