import uuid

from starlette.concurrency import run_in_threadpool

from app.worker.celery_app import celery_app

PROCESS_TICKET_TASK = "tickets.process"
DRAFT_REPLY_TASK = "tickets.draft_reply"
REQUEUE_STUCK_TASK = "tickets.requeue_stuck"
QUEUE_KEY = "celery"  # Redis list holding pending triage jobs (default queue)
DRAFTS_QUEUE_KEY = "drafts"  # Redis list holding pending reply-draft jobs
DEAD_LETTER_KEY = "triage:dead_letters"


async def enqueue_ticket(ticket_id: uuid.UUID) -> None:
    # send_task by name keeps the API from importing worker code (and the embedding model).
    await run_in_threadpool(celery_app.send_task, PROCESS_TICKET_TASK, args=[str(ticket_id)])
