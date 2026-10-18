"""Worker pipeline against Postgres + pgvector (hashing embedder, fake LLM)."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from sqlalchemy import func, select

from app.db import SyncSession
from app.models import Incident, LLMUsage, Prediction, ReplySuggestion, TicketStatus
from app.services import pipeline
from app.services.classifier import FieldPrediction
from app.services.llm import FakeLLMClient, RateLimited
from tests.db_helpers import get, isolated_time, make_resolved, make_ticket, words


class ConfidentClassifier:
    version = "stub-v1"

    def predict(self, embedding):
        return {"category": FieldPrediction("login_auth", 0.95, {"login_auth": 0.95}),
                "team": FieldPrediction("Platform", 0.9, {"Platform": 0.9}),
                "priority": FieldPrediction("P1", 0.6, {"P1": 0.6})}


def incident_of(ticket_id):
    with SyncSession() as session:
        return session.get(Incident, get(ticket_id).incident_id)


def test_duplicates_join_one_incident_and_distinct_ticket_starts_another(metrics):
    t0, w = isolated_time(), words()
    a = make_ticket("SSO login loop", f"okta sign in loops back to the login page {w}", t0)
    b = make_ticket("Re: SSO login loop", f"okta sign in loops back to the login page again {w}", t0 + timedelta(minutes=5))
    c = make_ticket("Invoice", f"please send a VAT invoice for october {words()}", t0 + timedelta(minutes=6))
    for t in (a, b, c):
        assert pipeline.process_ticket(t, metrics) is True

    assert get(a).incident_id == get(b).incident_id != get(c).incident_id
    assert incident_of(a).ticket_count == 2
    assert get(b).dedup_similarity >= 0.75
    assert metrics.counters["triage_incidents_created_total"] == 2
    assert metrics.counters["triage_tickets_deduplicated_total"] == 1
    assert len(metrics.observed["triage_processing_seconds"]) == 3


def test_tickets_outside_window_are_not_grouped(metrics):
    t0, w = isolated_time(), words()
    a = make_ticket("export stuck", f"csv export never finishes {w}", t0)
    b = make_ticket("export stuck", f"csv export never finishes {w}", t0 + timedelta(hours=100))
    pipeline.process_ticket(a, metrics)
    pipeline.process_ticket(b, metrics)
    assert get(a).incident_id != get(b).incident_id


def test_merged_incident_is_not_a_match(metrics):
    t0, w = isolated_time(), words()
    a = make_ticket("dashboard slow", f"dashboard takes a minute to load {w}", t0)
    pipeline.process_ticket(a, metrics)
    with SyncSession() as session:
        session.get(Incident, get(a).incident_id).status = "merged"
        session.commit()
    b = make_ticket("dashboard slow", f"dashboard takes a minute to load {w}", t0 + timedelta(minutes=1))
    pipeline.process_ticket(b, metrics)
    assert get(b).incident_id != get(a).incident_id


def test_concurrent_duplicates_create_exactly_one_incident(metrics):
    """Race condition: without the advisory lock, simultaneous duplicates each start an incident."""
    t0, w = isolated_time(), words()
    ids = [make_ticket("app down", f"everything returns 502 bad gateway {w}", t0 + timedelta(seconds=i))
           for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda t: pipeline.process_ticket(t, metrics), ids))
    incidents = {get(t).incident_id for t in ids}
    assert len(incidents) == 1
    with SyncSession() as session:
        assert session.get(Incident, incidents.pop()).ticket_count == 8


def test_no_classifier_uses_llm_and_sends_to_review(metrics):
    t = make_ticket("upload fails", f"attachment upload hangs at 99 percent {words()}", isolated_time())
    pipeline.process_ticket(t, metrics)
    ticket = get(t)
    assert ticket.status == TicketStatus.PROCESSED and ticket.routing == "review"
    assert (ticket.category, ticket.team, ticket.priority) == ("bug", "Product", "P3")  # fake LLM answer
    with SyncSession() as session:
        preds = session.scalars(select(Prediction).where(Prediction.ticket_id == t)).all()
        assert {p.model_version for p in preds} == {"fake"} and all(p.confidence is None for p in preds)
        assert session.scalar(select(func.count()).where(LLMUsage.ticket_id == t, LLMUsage.purpose == "classify")) == 1


def test_confident_classifier_auto_routes_without_llm(metrics, monkeypatch):
    llm = FakeLLMClient()
    monkeypatch.setattr(pipeline, "get_classifier", lambda: ConfidentClassifier())
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: llm)
    t = make_ticket("sso", f"cannot sign in with okta {words()}", isolated_time())
    pipeline.process_ticket(t, metrics)
    ticket = get(t)
    assert ticket.routing == "auto" and (ticket.category, ticket.team, ticket.priority) == ("login_auth", "Platform", "P1")
    assert llm.prompts == []  # cost control: no LLM call when the classifier is confident
    incident = incident_of(t)
    assert (incident.category, incident.team, incident.priority) == ("login_auth", "Platform", "P1")


def test_retry_resumes_after_llm_failure_without_redoing_dedup(metrics, monkeypatch):
    class FailingOnce(FakeLLMClient):
        failed = False

        def complete(self, prompt, **kw):
            if not self.failed:
                self.failed = True
                raise RateLimited("429")
            return super().complete(prompt, **kw)

    llm = FailingOnce()
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: llm)
    t = make_ticket("search slow", f"search times out after 30 seconds {words()}", isolated_time())
    with pytest.raises(RateLimited):
        pipeline.process_ticket(t, metrics)
    first = get(t)
    assert first.status == TicketStatus.PROCESSING and first.incident_id is not None and first.routing is None

    assert pipeline.process_ticket(t, metrics) is True
    second = get(t)
    assert second.status == TicketStatus.PROCESSED and second.incident_id == first.incident_id
    assert metrics.counters["triage_incidents_created_total"] == 1


def test_processed_ticket_is_not_reprocessed(metrics):
    t = make_ticket("a", f"b {words()}", isolated_time())
    assert pipeline.process_ticket(t, metrics) is True
    assert pipeline.process_ticket(t, metrics) is False


def test_draft_reply_grounded_in_similar_resolved_ticket(metrics):
    w = words()
    source = make_resolved("Okta SSO redirect loop", f"okta sso redirect loop on login page {w}",
                           "Re-save the SAML settings under Admin > Security, then sign in again.")
    t = make_ticket("sso loop", f"okta sso redirect loop on the login page {w}", isolated_time())
    pipeline.process_ticket(t, metrics)
    pipeline.draft_reply(t, metrics)
    with SyncSession() as session:
        s = session.scalar(select(ReplySuggestion).where(ReplySuggestion.ticket_id == t))
    assert s.status == "ready" and "[S1]" in s.draft
    assert s.sources[0]["ticket_id"] == str(source) and s.sources[0]["cited"] is True
    pipeline.draft_reply(t, metrics)  # idempotent: no second suggestion / IntegrityError


def test_no_relevant_resolved_ticket_means_no_llm_call(metrics, monkeypatch):
    llm = FakeLLMClient()
    monkeypatch.setattr(pipeline, "get_llm_client", lambda: llm)
    t = make_ticket("zzz", f"{words()} {words()}", isolated_time())
    pipeline.process_ticket(t, metrics)
    calls_before = len(llm.prompts)
    pipeline.draft_reply(t, metrics)
    with SyncSession() as session:
        s = session.scalar(select(ReplySuggestion).where(ReplySuggestion.ticket_id == t))
    assert s.status == "no_suggestion" and s.draft is None
    assert len(llm.prompts) == calls_before
