import random
import uuid
from datetime import UTC, datetime, timedelta

from app.db import SyncSession
from app.models import Ticket, TicketStatus
from app.services.cleaning import clean_ticket_text
from app.services.embeddings import get_embedder


def isolated_time() -> datetime:
    """A random point far from other tests' tickets, so dedup windows never overlap between tests."""
    return datetime(2031, 1, 1, tzinfo=UTC) + timedelta(days=random.randrange(0, 2_000_000, 7))


def make_ticket(subject: str, body: str, created_at: datetime) -> uuid.UUID:
    with SyncSession() as session:
        ticket = Ticket(idempotency_key=f"db-test-{uuid.uuid4()}", subject=subject, body=body, created_at=created_at)
        session.add(ticket)
        session.commit()
        return ticket.id


def make_resolved(subject: str, body: str, reply: str) -> uuid.UUID:
    text = clean_ticket_text(subject, body)
    with SyncSession() as session:
        ticket = Ticket(
            idempotency_key=f"db-test-history-{uuid.uuid4()}", source="history", subject=subject, body=body,
            clean_text=text, embedding=get_embedder().embed([text])[0].tolist(), status=TicketStatus.RESOLVED,
            resolution=reply, category="login_auth", priority="P2", team="Platform",
        )
        session.add(ticket)
        session.commit()
        return ticket.id


def get(ticket_id: uuid.UUID) -> Ticket:
    with SyncSession() as session:
        return session.get(Ticket, ticket_id)


def words() -> str:
    """Unique tokens so a test's tickets only resemble each other."""
    return " ".join(uuid.uuid4().hex[:8] for _ in range(4))
