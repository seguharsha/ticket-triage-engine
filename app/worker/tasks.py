import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import redis
from celery import Task
from celery.signals import setup_logging, worker_process_init
from sqlalchemy import and_, or_, select

from app.config import settings
from app.db import SyncSession
from app.logging_config import configure_logging
from app.metrics import WorkerMetrics
from app.models import Ticket, TicketStatus
from app.queue import DEAD_LETTER_KEY, DRAFT_REPLY_TASK, PROCESS_TICKET_TASK, REQUEUE_STUCK_TASK
from app.services import pipeline
from app.services.classifier import get_classifier
from app.services.embeddings import get_embedder
from app.worker.celery_app import celery_app

log = logging.getLogger(__name__)

_redis = redis.Redis.from_url(settings.redis_url)
metrics = WorkerMetrics(_redis)


@setup_logging.connect
def _configure_logging(**_):
    configure_logging()


@worker_process_init.connect
def _warm_up(**_):
    # Load models when each worker process starts, not on its first ticket (which would skew p95).
    # The drafts worker uses stored embeddings only, so it skips this and saves ~300 MB per process.
    if settings.warm_up_models:
        get_embedder().embed(["warm up"])
        get_classifier()


class TriageTask(Task):
    """Retries with exponential backoff + jitter; after the last attempt, records a dead letter."""

    autoretry_for = (Exception,)
    max_retries = settings.task_max_retries
    retry_backoff = 2
    retry_backoff_max = 60
    retry_jitter = True
    marks_ticket_failed = True

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        metrics.inc("triage_task_retries_total")
        log.warning("task retry", extra={"task": self.name, "ticket_id": args[0] if args else None,
                                         "error": repr(exc)})

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        ticket_id = args[0] if args else None
        _redis.rpush(DEAD_LETTER_KEY, json.dumps({
            "task": self.name,
            "task_id": task_id,
            "ticket_id": ticket_id,
            "error": repr(exc),
            "failed_at": datetime.now(UTC).isoformat(),
        }))
        metrics.inc("triage_tickets_failed_total")
        if ticket_id and self.marks_ticket_failed:
            with SyncSession() as session:
                ticket = session.get(Ticket, uuid.UUID(ticket_id))
                if ticket is not None:
                    ticket.status = TicketStatus.FAILED
                    ticket.error = repr(exc)[:2000]
                    session.commit()
        log.error("task dead-lettered", extra={"task": self.name, "ticket_id": ticket_id, "error": repr(exc)})


class DraftTask(TriageTask):
    # A failed draft doesn't undo triage: the ticket stays processed, the failure is dead-lettered.
    marks_ticket_failed = False


def _ticket_lock(ticket_id: str):
    # Guards against the same ticket being processed twice at once (duplicate enqueue, redelivery).
    return _redis.lock(f"triage:lock:{ticket_id}", timeout=300)


@celery_app.task(name=PROCESS_TICKET_TASK, base=TriageTask, bind=True)
def process_ticket(self, ticket_id: str) -> None:
    lock = _ticket_lock(ticket_id)
    if not lock.acquire(blocking=False):
        log.info("ticket already being processed", extra={"ticket_id": ticket_id})
        return
    try:
        finished = pipeline.process_ticket(uuid.UUID(ticket_id), metrics)
    finally:
        try:
            lock.release()
        except redis.exceptions.LockError:
            pass
    if finished:
        draft_reply.delay(ticket_id)


@celery_app.task(name=DRAFT_REPLY_TASK, base=DraftTask, bind=True)
def draft_reply(self, ticket_id: str) -> None:
    pipeline.draft_reply(uuid.UUID(ticket_id), metrics)


@celery_app.task(name=REQUEUE_STUCK_TASK)
def requeue_stuck() -> int:
    """Re-enqueue tickets whose job was lost.

    Intake commits before it enqueues, so a lost enqueue leaves a ticket in `received`; a dead
    worker leaves one in `processing`. A duplicate job is harmless: the per-ticket lock and the
    pipeline's step skipping see to that.
    """
    now = datetime.now(UTC)
    with SyncSession() as session:
        stuck = session.scalars(
            select(Ticket.id).where(
                or_(
                    and_(Ticket.status == TicketStatus.RECEIVED,
                         Ticket.received_at < now - timedelta(seconds=settings.requeue_received_after_seconds)),
                    and_(Ticket.status == TicketStatus.PROCESSING,
                         Ticket.received_at < now - timedelta(seconds=settings.requeue_processing_after_seconds)),
                )
            ).limit(1000)
        ).all()
    for ticket_id in stuck:
        process_ticket.delay(str(ticket_id))
    if stuck:
        metrics.inc("triage_requeued_total", len(stuck))
        log.warning("requeued stuck tickets", extra={"count": len(stuck)})
    return len(stuck)
