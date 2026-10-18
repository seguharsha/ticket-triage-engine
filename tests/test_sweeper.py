"""The stuck-ticket sweeper re-enqueues tickets whose job was lost (needs Postgres + Redis)."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import update

from app.db import SyncSession
from app.models import Ticket, TicketStatus
from app.worker import tasks
from tests.db_helpers import isolated_time, make_ticket, words


def set_state(ticket_id, status, received_ago: timedelta):
    with SyncSession() as session:
        session.execute(update(Ticket).where(Ticket.id == ticket_id)
                        .values(status=status, received_at=datetime.now(UTC) - received_ago))
        session.commit()


def test_requeues_only_tickets_stuck_past_their_threshold(monkeypatch):
    sent = []
    monkeypatch.setattr(tasks.process_ticket, "delay", lambda ticket_id: sent.append(ticket_id))

    stuck_received = make_ticket("a", words(), isolated_time())
    fresh_received = make_ticket("b", words(), isolated_time())
    stuck_processing = make_ticket("c", words(), isolated_time())
    fresh_processing = make_ticket("d", words(), isolated_time())
    done = make_ticket("e", words(), isolated_time())
    set_state(stuck_received, TicketStatus.RECEIVED, timedelta(minutes=10))
    set_state(fresh_received, TicketStatus.RECEIVED, timedelta(seconds=5))
    set_state(stuck_processing, TicketStatus.PROCESSING, timedelta(minutes=30))
    set_state(fresh_processing, TicketStatus.PROCESSING, timedelta(minutes=2))
    set_state(done, TicketStatus.PROCESSED, timedelta(hours=1))

    tasks.requeue_stuck()

    assert str(stuck_received) in sent and str(stuck_processing) in sent
    assert not {str(fresh_received), str(fresh_processing), str(done)} & set(sent)
