"""Duplicate detection: attach a ticket to the incident of its nearest recent open neighbour.

The rule (single-link, nearest neighbour above a threshold, inside a time window) lives in
`group_online`, a pure function the evaluation script runs offline; `assign_incident` is the same
rule against Postgres.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DEDUP_SEQUENCE, Incident, IncidentStatus, Ticket
from app.services.cleaning import clean_subject

# Serializes incident assignment so two concurrent duplicates can't both start an incident.
# Merge/split take the same lock. The ANN search itself runs outside it (see assign_incident).
DEDUP_LOCK_KEY = 7_201_001


def lock_dedup(session) -> None:
    session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": DEDUP_LOCK_KEY})

@dataclass
class DedupResult:
    incident_id: uuid.UUID
    created_incident: bool
    matched_ticket_id: uuid.UUID | None
    similarity: float | None
    locked_seconds: float = 0.0  # time spent holding the lock (the serialized part)


def group_online(
    embeddings: np.ndarray, times: list[datetime], threshold: float, window: timedelta
) -> list[int]:
    """Assign group labels in time order, exactly as the live service does. Embeddings must be unit length."""
    order = sorted(range(len(times)), key=lambda i: times[i])
    labels = [-1] * len(times)
    seen: list[int] = []
    next_label = 0
    for i in order:
        candidates = [j for j in seen if abs(times[i] - times[j]) <= window]
        best_j, best_sim = None, -1.0
        if candidates:
            sims = embeddings[candidates] @ embeddings[i]
            k = int(np.argmax(sims))
            best_j, best_sim = candidates[k], float(sims[k])
        if best_j is not None and best_sim >= threshold:
            labels[i] = labels[best_j]
        else:
            labels[i] = next_label
            next_label += 1
        seen.append(i)
    return labels


def find_nearest_open(session: Session, ticket: Ticket, embedding: list[float], window: timedelta):
    session.execute(text(f"SET LOCAL hnsw.ef_search = {int(settings.hnsw_ef_search)}"))
    # pgvector >= 0.8: keep scanning the index until enough rows pass the filters.
    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    distance = Ticket.embedding.cosine_distance(embedding)
    open_incidents = select(Incident.id).where(Incident.status == IncidentStatus.OPEN)
    rows = session.execute(
        select(Ticket.id, Ticket.incident_id, distance.label("distance"))
        .where(
            Ticket.id != ticket.id,
            Ticket.embedding.is_not(None),
            Ticket.incident_id.in_(open_incidents),
            Ticket.created_at.between(ticket.created_at - window, ticket.created_at + window),
        )
        .order_by(distance)
        .limit(5)
    ).all()
    # relaxed_order may return slightly out-of-order rows; take the true best.
    return min(rows, key=lambda r: r.distance) if rows else None


def _open_incident_of(session: Session, ticket_id: uuid.UUID) -> uuid.UUID | None:
    """Current incident of a ticket if still open (a merge may have moved it since the search)."""
    return session.scalar(
        select(Ticket.incident_id)
        .join(Incident, Incident.id == Ticket.incident_id)
        .where(Ticket.id == ticket_id, Incident.status == IncidentStatus.OPEN)
    )


def assign_incident(session: Session, ticket: Ticket, embedding: list[float]) -> DedupResult:
    """Join the nearest open incident or start a new one. Commits.

    Optimistic scheme, correct under concurrency:
      1. Under the lock, read the high-water mark of assignments. Lock holders commit before
         releasing it, so every assignment numbered <= the mark is committed.
      2. Without the lock, run the ANN search. Its snapshot starts after step 1, so it sees every
         assignment <= the mark.
      3. Under the lock again, check only tickets assigned after the mark (those that committed
         while we searched; usually a handful), then join or create, and take the next number.
    """
    window = timedelta(hours=settings.dedup_window_hours)

    lock_dedup(session)
    mark = session.scalar(select(func.coalesce(func.max(Ticket.dedup_seq), 0)))
    session.commit()

    nearest = find_nearest_open(session, ticket, embedding, window)
    session.commit()
    best_id = nearest.id if nearest else None
    best_sim = 1 - nearest.distance if nearest else None

    lock_dedup(session)
    locked_at = time.perf_counter()
    distance = Ticket.embedding.cosine_distance(embedding)
    open_incidents = select(Incident.id).where(Incident.status == IncidentStatus.OPEN)
    assigned_meanwhile = session.execute(
        select(Ticket.id, distance.label("distance")).where(
            Ticket.dedup_seq > mark,
            Ticket.id != ticket.id,
            Ticket.incident_id.in_(open_incidents),
            Ticket.created_at.between(ticket.created_at - window, ticket.created_at + window),
        )
    ).all()
    for row in assigned_meanwhile:
        if best_sim is None or 1 - row.distance > best_sim:
            best_id, best_sim = row.id, 1 - row.distance

    incident_id = _open_incident_of(session, best_id) if best_id and best_sim >= settings.dedup_threshold else None
    if incident_id:
        session.execute(
            update(Incident).where(Incident.id == incident_id).values(ticket_count=Incident.ticket_count + 1)
        )
        result = DedupResult(incident_id, False, best_id, best_sim)
    else:
        incident = Incident(title=clean_subject(ticket.subject)[:200] or "Untitled", ticket_count=1)
        session.add(incident)
        session.flush()
        result = DedupResult(incident.id, True, None, best_sim)

    ticket.incident_id = result.incident_id
    ticket.dedup_similarity = best_sim
    ticket.dedup_seq = session.scalar(select(DEDUP_SEQUENCE.next_value()))
    session.commit()
    result.locked_seconds = time.perf_counter() - locked_at
    return result