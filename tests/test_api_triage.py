"""Triage endpoints: similar, suggested reply, corrections, review queue, incidents, ops."""

import uuid
from datetime import timedelta

import pytest

from app.services import pipeline
from tests.db_helpers import isolated_time, make_resolved, make_ticket, words


@pytest.fixture
def processed(metrics):
    def make(subject, body, at=None):
        t = make_ticket(subject, body, at or isolated_time())
        pipeline.process_ticket(t, metrics)
        return t
    return make


def test_ticket_detail_includes_triage_fields(client, processed):
    t = processed("upload broken", f"upload hangs {words()}")
    body = client.get(f"/tickets/{t}").json()
    assert body["status"] == "processed" and body["routing"] == "review"
    assert body["incident_id"] and body["team"] == "Product" and body["clean_text"]
    assert {p["field"] for p in body["predictions"]} == {"category", "priority", "team"}


def test_similar_returns_resolved_tickets_ranked(client, processed):
    w = words()
    source = make_resolved("export broken", f"csv export missing columns {w}", "Enable 'include custom fields'.")
    t = processed("export", f"csv export missing columns {w}")
    results = client.get(f"/tickets/{t}/similar?k=3").json()
    assert results[0]["ticket_id"] == str(source) and results[0]["similarity"] > 0.8
    assert results[0]["url"] == f"/tickets/{source}"


def test_similar_conflict_when_not_embedded(client):
    t = make_ticket("x", "y", isolated_time())
    assert client.get(f"/tickets/{t}/similar").status_code == 409


def test_suggested_reply_lifecycle(client, processed, metrics):
    w = words()
    make_resolved("2fa lost", f"lost my phone two factor codes {w}", "We disabled 2FA after ID check.")
    t = processed("2fa", f"lost my phone two factor codes {w}")
    assert client.get(f"/tickets/{t}/suggested-reply").json()["status"] == "pending"
    assert client.post(f"/tickets/{t}/suggested-reply/approve", json={"approved_by": "ana"}).status_code == 409

    pipeline.draft_reply(t, metrics)
    reply = client.get(f"/tickets/{t}/suggested-reply").json()
    assert reply["status"] == "ready" and reply["sources"][0]["cited"] and reply["sent_automatically"] is False

    approved = client.post(f"/tickets/{t}/suggested-reply/approve", json={"approved_by": "ana"}).json()
    assert approved["approved_by"] == "ana" and approved["approved_at"]


def test_corrections_update_ticket_and_leave_review_queue(client, processed):
    t = processed("billing", f"charged twice {words()}")
    queue_ids = [r["id"] for r in client.get("/review-queue?limit=500").json()]
    assert str(t) in queue_ids

    resp = client.post(f"/tickets/{t}/corrections",
                       json={"corrected_by": "lead", "category": "billing", "team": "Payments", "priority": "P3"})
    assert resp.status_code == 200
    changes = {c["field"]: (c["old_value"], c["new_value"]) for c in resp.json()}
    assert changes == {"category": ("bug", "billing"), "team": ("Product", "Payments")}  # priority unchanged

    ticket = client.get(f"/tickets/{t}").json()
    assert (ticket["category"], ticket["team"], ticket["reviewed_by"]) == ("billing", "Payments", "lead")
    assert str(t) not in [r["id"] for r in client.get("/review-queue?limit=500").json()]


def test_correction_rejects_unknown_label(client, processed):
    t = processed("x", f"y {words()}")
    assert client.post(f"/tickets/{t}/corrections", json={"corrected_by": "a", "team": "Marketing"}).status_code == 422


def test_incident_list_detail_merge_and_split(client, processed):
    t0, w1, w2 = isolated_time(), words(), words()
    a1 = processed("sso", f"okta login loop {w1}", t0)
    a2 = processed("sso", f"okta login loop again {w1}", t0 + timedelta(minutes=1))
    b1 = processed("sso other", f"google sign in error {w2}", t0 + timedelta(minutes=2))
    inc_a = client.get(f"/tickets/{a1}").json()["incident_id"]
    inc_b = client.get(f"/tickets/{b1}").json()["incident_id"]
    assert inc_a == client.get(f"/tickets/{a2}").json()["incident_id"] != inc_b

    detail = client.get(f"/incidents/{inc_a}").json()
    assert detail["ticket_count"] == 2 and {t["id"] for t in detail["tickets"]} == {str(a1), str(a2)}
    assert inc_a in [i["id"] for i in client.get("/incidents?status=open&min_tickets=2&limit=500").json()]

    merged = client.post(f"/incidents/{inc_a}/merge", json={"source_incident_id": inc_b, "corrected_by": "lead"})
    assert merged.status_code == 200 and merged.json()["ticket_count"] == 3
    assert client.get(f"/incidents/{inc_b}").json()["status"] == "merged"
    assert client.get(f"/tickets/{b1}").json()["incident_id"] == inc_a
    assert client.post(f"/incidents/{inc_a}/merge",
                       json={"source_incident_id": inc_b, "corrected_by": "lead"}).status_code == 409
    assert client.post(f"/incidents/{inc_a}/merge",
                       json={"source_incident_id": inc_a, "corrected_by": "lead"}).status_code == 422

    split = client.post(f"/incidents/{inc_a}/split", json={"ticket_ids": [str(b1)], "corrected_by": "lead"})
    assert split.status_code == 201 and split.json()["ticket_count"] == 1
    assert client.get(f"/tickets/{b1}").json()["incident_id"] == split.json()["id"]
    assert client.get(f"/incidents/{inc_a}").json()["ticket_count"] == 2
    # can't empty an incident, or split out a ticket that isn't in it
    assert client.post(f"/incidents/{inc_a}/split",
                       json={"ticket_ids": [str(a1), str(a2)], "corrected_by": "x"}).status_code == 422
    assert client.post(f"/incidents/{inc_a}/split",
                       json={"ticket_ids": [str(b1)], "corrected_by": "x"}).status_code == 422


def test_unknown_incident_404(client):
    assert client.get(f"/incidents/{uuid.uuid4()}").status_code == 404


def test_metrics_and_dead_letters(client):
    client.get("/health")
    body = client.get("/metrics").text
    for name in ["http_request_duration_seconds_bucket", "triage_queue_depth", "triage_review_queue_size",
                 "triage_open_incidents", "triage_dead_letters"]:
        assert name in body
    assert client.get("/dead-letters").status_code == 200
