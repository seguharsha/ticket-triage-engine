import logging
import time

from fastapi import FastAPI, Request
from prometheus_client import Histogram

from app.api import health, incidents, tickets, webhooks
from app.logging_config import configure_logging

configure_logging()
log = logging.getLogger("app.http")

HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "API request latency",
    ["method", "route", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5),
)

app = FastAPI(
    title="Support Ticket Triage Engine",
    version="1.0.0",
    description="Deduplicates, classifies, routes and drafts replies for support tickets.",
)
app.include_router(tickets.router)
app.include_router(webhooks.router)
app.include_router(incidents.router)
app.include_router(health.router)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    # Label by route template (/tickets/{ticket_id}), not raw path, to keep label cardinality bounded.
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    HTTP_LATENCY.labels(request.method, path, str(response.status_code)).observe(elapsed)
    if path not in ("/metrics", "/health"):
        log.info("request", extra={"method": request.method, "route": path, "status": response.status_code,
                                   "ms": round(elapsed * 1000, 2)})
    return response
