# Load test results

Measured 2026-10-18 with `loadtest/locustfile.py` (each simulated helpdesk sends 1 ticket/s) and
`scripts/processing_report.py`. Raw Locust CSVs: `reports/load_15_*.csv`, `reports/load_20_*.csv`
(git-ignored).

## Setup

| | |
|---|---|
| API | uvicorn, 2 processes, no `--reload` |
| Worker | Celery, 4 processes × 2 PyTorch threads (`WORKER_CONCURRENCY=4`, `TORCH_THREADS=2`) |
| LLM | `LLM_PROVIDER=fake` (fixed answer after 300 ms), so the test measures this system, not an LLM |
| Load generator | Locust on the same host (it competes for the same CPU) |
| Duration | 2 minutes per run, starting from an empty database plus the resolved-ticket knowledge base |

## Results

| Offered load | Achieved | Intake p50 / p95 / p99 | Errors | Queue during run | Processing p50 / p95 / p99 | Lost |
|---|---|---|---|---|---|---|
| 15 users | **14.9 tickets/s** | 110 / **170** / 280 ms | **0%** | flat (0–11) | 0.37 / **0.61** / 0.74 s | **0** |
| 20 users | **18.2 tickets/s** (Locust's limit) | 130 / 210 / 310 ms | 0.28% (connection resets) | flat (0–18) | 0.51 / **1.16** / 1.60 s | **0** |
| 50 users (3 runs) | 37–46 tickets/s offered | 140–550 / 460–1,600 ms | 0–0.95% | grew to ~3,700 | grew to p95 45–80 s | none in one run; 1 ticket stuck in another; one not checked |

"Lost" counts tickets accepted with 202 but never processed, checked by waiting for the queue to drain
and counting statuses. The 15 and 20-user runs were checked with the sweeper in place:
all `processed`, 0 dead letters. Of the three 50-user runs, one had none lost, one left 1 ticket stuck
in `received` (which led to the sweeper, below), and one was not checked before the database was reset.

**Max sustained throughput: ~18 tickets/s** (the highest rate where the queue stays
flat). The 50 tickets/s target is not met.

## Where the time goes (worker, per ticket, from `/metrics`)

| Stage | Mean |
|---|---|
| Embed (`all-MiniLM-L6-v2`, CPU) | 72 ms |
| Dedup (ANN search + assignment) | 65 ms |
| of which holding the global dedup lock | 27 ms |
| Classify + route | 27 ms |

About 165 ms of CPU per ticket: 50 tickets/s needs 8+ cores of worker CPU, more than the test host
had once the API, Postgres and Locust took their share. The limit is CPU, not the
design: the lock-held section (27 ms) would allow ~37 tickets/s on its own.

## Changes made because of these tests

1. **ANN search moved out of the dedup lock.** It first ran inside the global advisory lock, which
   capped throughput at ~18 tickets/s and would have got worse as the table grew. Now the search runs unlocked and only a delta check (assignments made during the
   search, tracked by a sequence number) runs under the lock. The concurrency test still passes.
2. **Reply drafts on their own queue and worker,** so slow LLM calls never occupy triage slots.
3. **PyTorch threads capped per worker process.** The default (6 threads × each process) oversubscribed
   the CPU and starved the API: intake p50 dropped from 310 ms to 140 ms at the same load.
4. **Stuck-ticket sweeper.** One run left a ticket in `received` (stored, but its enqueue was lost when
   the client connection was reset). A periodic task now re-enqueues tickets stuck in `received` (> 5
   min) or `processing` (> 10 min).

## Known issues

- The connection resets at ≥ 18 tickets/s are requests that hung ~30 s and were then dropped, with no
  error in the API logs. They did not occur at 15 tickets/s. Most likely the host's Docker port
  forwarding under load; the cause is not confirmed.
- Intake p95 (170 ms at 15/s) is above the 100 ms target. Inside the container, the database insert +
  commit is ~21 ms and the enqueue ~6 ms (p50); the rest is HTTP handling, Docker networking
  and CPU contention with the worker and Locust on the same host.
