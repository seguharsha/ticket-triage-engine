import json

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.metrics import COUNTERS_KEY, HIST_PREFIX, format_metrics
from app.models import Incident, IncidentStatus, Routing, Ticket, TicketStatus
from app.queue import DEAD_LETTER_KEY, DRAFTS_QUEUE_KEY, QUEUE_KEY

router = APIRouter(tags=["ops"])
_redis = aioredis.from_url(settings.redis_url)


@router.get("/health")
async def health(response: Response, session: AsyncSession = Depends(get_session)) -> dict:
    checks = {}
    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc.__class__.__name__}"
    try:
        await _redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc.__class__.__name__}"

    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", **checks}


@router.get("/metrics")
async def metrics(session: AsyncSession = Depends(get_session)) -> Response:
    hist_keys = [k async for k in _redis.scan_iter(match=HIST_PREFIX + "*")]
    histograms = {}
    for key in hist_keys:
        name = (key.decode() if isinstance(key, bytes) else key).removeprefix(HIST_PREFIX)
        histograms[name] = await _redis.hgetall(key)
    counters = await _redis.hgetall(COUNTERS_KEY)

    review = await session.scalar(
        select(func.count()).where(Ticket.routing == Routing.REVIEW, Ticket.reviewed_at.is_(None))
    )
    open_incidents = await session.scalar(select(func.count()).where(Incident.status == IncidentStatus.OPEN))
    unprocessed = await session.scalar(
        select(func.count()).where(Ticket.status.in_([TicketStatus.RECEIVED, TicketStatus.PROCESSING]))
    )
    gauges = {
        "triage_queue_depth": await _redis.llen(QUEUE_KEY),
        "triage_drafts_queue_depth": await _redis.llen(DRAFTS_QUEUE_KEY),
        "triage_dead_letters": await _redis.llen(DEAD_LETTER_KEY),
        "triage_review_queue_size": review,
        "triage_open_incidents": open_incidents,
        "triage_tickets_unprocessed": unprocessed,
    }
    body = generate_latest().decode() + format_metrics(histograms, counters, gauges)
    return Response(body, media_type=CONTENT_TYPE_LATEST)


@router.get("/dead-letters")
async def dead_letters(limit: int = 50) -> list[dict]:
    """Most recent jobs that failed after all retries."""
    raw = await _redis.lrange(DEAD_LETTER_KEY, -limit, -1)
    return [json.loads(r) for r in reversed(raw)]
