from celery import Celery

from app.config import settings

celery_app = Celery("triage", broker=settings.redis_url, include=["app.worker.tasks"])
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    # Ack only after the task finishes, so a crashed worker's job is redelivered rather than lost.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # Drafts call the LLM (seconds each on CPU). A separate queue and worker keeps them from
    # occupying the slots triage needs, so processing latency doesn't depend on LLM speed.
    task_routes={"tickets.draft_reply": {"queue": "drafts"}},
    beat_schedule={
        "requeue-stuck-tickets": {"task": "tickets.requeue_stuck", "schedule": settings.requeue_interval_seconds},
    },
)
