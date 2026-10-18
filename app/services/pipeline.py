"""The worker's per-ticket pipeline: clean + embed -> dedup -> classify + route, then reply drafting.

Each step commits on its own and is skipped if already done, so a retry (say, after an LLM 429 in the
classification step) resumes where it failed instead of re-running dedup.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SyncSession
from app.metrics import WorkerMetrics
from app.models import Incident, LLMUsage, Prediction, ReplySuggestion, Ticket, TicketStatus
from app.services.classifier import EmbeddingClassifier, get_classifier
from app.services.cleaning import clean_ticket_text
from app.services.dedup import assign_incident
from app.services.embeddings import get_embedder
from app.services.llm import LLMClient, LLMResult, extract_json, get_llm_client
from app.services.replies import parse_draft, relevant, reply_prompt, similar_resolved_stmt, sources_json, to_sources
from app.services.routing import Decision, classification_prompt, decide, parse_llm_labels, uncertain_fields

log = logging.getLogger(__name__)


@contextmanager
def timed(metrics: WorkerMetrics, name: str):
    start = time.perf_counter()
    yield
    metrics.observe(name, time.perf_counter() - start)


def record_llm_usage(session: Session, metrics: WorkerMetrics, ticket_id, purpose: str, result: LLMResult) -> None:
    session.add(
        LLMUsage(
            ticket_id=ticket_id,
            purpose=purpose,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )
    )
    metrics.inc("triage_llm_calls_total")
    metrics.inc("triage_llm_cost_usd_total", result.cost_usd)


def classify_and_route(
    session: Session,
    metrics: WorkerMetrics,
    ticket: Ticket,
    embedding: list[float],
    classifier: EmbeddingClassifier | None,
    llm: LLMClient | None,
) -> Decision:
    primary = classifier.predict(embedding) if classifier else {}
    llm_labels, llm_model = None, None
    # The LLM is called only when the cheap classifier is unsure about a routing field.
    if llm and settings.llm_classify_fallback and uncertain_fields(
        primary, settings.routing_fields, settings.route_threshold
    ):
        result = llm.complete(classification_prompt(ticket.clean_text or ""), max_tokens=100, json_mode=True)
        record_llm_usage(session, metrics, ticket.id, "classify", result)
        llm_labels, llm_model = parse_llm_labels(extract_json(result.text)), result.model

    decision = decide(primary, llm_labels, settings.routing_fields, settings.route_threshold)

    for name, pred in primary.items():
        session.add(Prediction(ticket_id=ticket.id, field=name, predicted_value=pred.value,
                               confidence=pred.confidence, model_version=classifier.version))
    for name, value in (llm_labels or {}).items():
        session.add(Prediction(ticket_id=ticket.id, field=name, predicted_value=value,
                               confidence=None, model_version=llm_model))

    labels = decision.labels
    ticket.category = labels["category"].value if "category" in labels else None
    ticket.priority = labels["priority"].value if "priority" in labels else None
    ticket.team = labels["team"].value if "team" in labels else None
    ticket.routing = decision.routing

    # The incident takes the first ticket's labels, and escalates to the most urgent priority seen.
    incident = session.get(Incident, ticket.incident_id, with_for_update=True)
    if incident is not None:
        incident.category = incident.category or ticket.category
        incident.team = incident.team or ticket.team
        if ticket.priority and (incident.priority is None or ticket.priority < incident.priority):
            incident.priority = ticket.priority
    session.commit()
    log.info("ticket classified", extra={"ticket_id": str(ticket.id), "routing": decision.routing,
                                          "reason": decision.reason, "category": ticket.category,
                                          "team": ticket.team, "priority": ticket.priority})
    return decision


def process_ticket(ticket_id: uuid.UUID, metrics: WorkerMetrics) -> bool:
    """Triage one ticket. Returns True when it finished triage now (so a reply draft should be queued)."""
    with SyncSession() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None or ticket.status in (TicketStatus.PROCESSED, TicketStatus.RESOLVED):
            return False
        ticket.status = TicketStatus.PROCESSING

        if ticket.embedding is None:
            with timed(metrics, "triage_embed_seconds"):
                ticket.clean_text = clean_ticket_text(ticket.subject, ticket.body)
                ticket.embedding = get_embedder().embed([ticket.clean_text])[0].tolist()
        session.commit()
        embedding = np.asarray(ticket.embedding, dtype=float).tolist()

        if ticket.incident_id is None:
            with timed(metrics, "triage_dedup_seconds"):
                result = assign_incident(session, ticket, embedding)
            metrics.observe("triage_dedup_lock_seconds", result.locked_seconds)
            metrics.inc("triage_incidents_created_total" if result.created_incident
                        else "triage_tickets_deduplicated_total")
            log.info("ticket deduplicated", extra={"ticket_id": str(ticket.id),
                                                    "incident_id": str(result.incident_id),
                                                    "new_incident": result.created_incident,
                                                    "similarity": result.similarity})

        if ticket.routing is None:
            with timed(metrics, "triage_classify_seconds"):
                decision = classify_and_route(session, metrics, ticket, embedding, get_classifier(), get_llm_client())
            metrics.inc(f"triage_routed_{decision.routing}_total")

        ticket.status = TicketStatus.PROCESSED
        ticket.processed_at = datetime.now(UTC)
        ticket.error = None
        session.commit()

    metrics.observe("triage_processing_seconds", (ticket.processed_at - ticket.received_at).total_seconds())
    metrics.inc("triage_tickets_processed_total")
    return True


def draft_reply(ticket_id: uuid.UUID, metrics: WorkerMetrics) -> None:
    with SyncSession() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None or ticket.embedding is None:
            return
        if session.scalar(select(ReplySuggestion.id).where(ReplySuggestion.ticket_id == ticket_id)):
            return

        start = time.perf_counter()
        embedding = np.asarray(ticket.embedding, dtype=float).tolist()
        retrieved = to_sources(session.execute(similar_resolved_stmt(embedding, settings.reply_top_k)).all())
        usable = relevant(retrieved, settings.reply_min_similarity)
        llm = get_llm_client()

        if not usable:
            # No LLM call: nothing relevant to ground a reply in.
            suggestion = ReplySuggestion(
                ticket_id=ticket_id, status="no_suggestion", sources=sources_json(retrieved, []),
                reason=f"no resolved ticket with similarity >= {settings.reply_min_similarity}",
            )
        elif llm is None:
            suggestion = ReplySuggestion(ticket_id=ticket_id, status="no_suggestion",
                                         sources=sources_json(usable, []), reason="LLM disabled")
        else:
            result = llm.complete(reply_prompt(ticket.clean_text or "", usable), max_tokens=400)
            record_llm_usage(session, metrics, ticket_id, "draft", result)
            draft = parse_draft(result.text, len(usable))
            suggestion = ReplySuggestion(
                ticket_id=ticket_id, status=draft.status, reason=draft.reason, draft=draft.text,
                sources=sources_json(usable, draft.cited), model_version=result.model,
            )
        session.add(suggestion)
        try:
            session.commit()
        except IntegrityError:  # a concurrent duplicate job already stored one
            session.rollback()
            return
        metrics.observe("triage_draft_seconds", time.perf_counter() - start)
