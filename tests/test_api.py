import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import settings
from app.services.attachments import safe_filename
from app.services.signatures import sign

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def post_ticket(client, key, **extra):
    data = {"subject": "Can't log in", "body": "SSO page spins forever"}
    return client.post("/tickets", headers={"Idempotency-Key": key}, data=data, **extra)


def test_submit_returns_202_and_enqueues(client, enqueued, key):
    resp = post_ticket(client, key)
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "received"
    assert body["duplicate_submission"] is False
    assert enqueued == [uuid.UUID(body["id"])]


def test_same_idempotency_key_returns_original_ticket(client, enqueued, key):
    first = post_ticket(client, key).json()
    second = post_ticket(client, key)
    assert second.status_code == 200
    assert second.json()["id"] == first["id"]
    assert second.json()["duplicate_submission"] is True


def test_concurrent_same_key_creates_one_ticket(client, enqueued, key):
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: post_ticket(client, key), range(8)))
    assert {r.json()["id"] for r in responses} == {responses[0].json()["id"]}
    assert sorted(r.status_code for r in responses) == [200] * 7 + [202]


def test_missing_idempotency_key_rejected(client, enqueued):
    resp = client.post("/tickets", data={"subject": "x", "body": "y"})
    assert resp.status_code == 422
    assert enqueued == []


def test_created_at_override_is_stored(client, enqueued, key):
    resp = post_ticket(client, key)
    ticket_id = resp.json()["id"]
    resp = client.post(
        "/tickets",
        headers={"Idempotency-Key": key + "-ts"},
        data={"subject": "a", "body": "b", "created_at": "2026-10-20T10:12:00"},
    )
    ticket = client.get(f"/tickets/{resp.json()['id']}").json()
    assert ticket["created_at"].startswith("2026-10-20T10:12:00")
    assert ticket_id != resp.json()["id"]


def test_attachment_is_stored_with_metadata(client, enqueued, key):
    resp = post_ticket(client, key, files=[("attachments", ("../../evil shot.png", PNG, "image/png"))])
    assert resp.status_code == 202
    ticket = client.get(f"/tickets/{resp.json()['id']}").json()
    [attachment] = ticket["attachments"]
    assert attachment["filename"] == "evil_shot.png"
    assert attachment["content_type"] == "image/png"
    assert attachment["size_bytes"] == len(PNG)
    stored = list((Path(settings.attachment_dir) / ticket["id"]).iterdir())
    assert len(stored) == 1 and stored[0].read_bytes() == PNG


def test_attachment_wrong_type_rejected_before_ticket_created(client, enqueued, key):
    resp = post_ticket(client, key, files=[("attachments", ("x.exe", b"MZ", "application/x-msdownload"))])
    assert resp.status_code == 415
    assert enqueued == []
    # The key was not consumed, so a corrected retry still works.
    assert post_ticket(client, key).status_code == 202


def test_attachment_too_large_rejected(client, enqueued, key):
    big = b"0" * (settings.max_attachment_bytes + 1)
    resp = post_ticket(client, key, files=[("attachments", ("big.txt", big, "text/plain"))])
    assert resp.status_code == 413


def test_get_unknown_ticket_404(client):
    assert client.get(f"/tickets/{uuid.uuid4()}").status_code == 404


def test_safe_filename():
    assert safe_filename("..\\..\\windows\\a b.png") == "a_b.png"
    assert safe_filename(".htaccess") == "htaccess"
    assert safe_filename(None) == "file"


def signed_webhook(client, payload: dict, secret=None, timestamp=None):
    raw = json.dumps(payload).encode()
    ts = str(timestamp or int(time.time()))
    headers = {
        "Content-Type": "application/json",
        "X-Timestamp": ts,
        "X-Signature": sign(secret or settings.webhook_secret, ts, raw),
    }
    return client.post("/webhooks/tickets", content=raw, headers=headers)


def test_webhook_valid_signature_accepted_and_resend_deduplicated(client, enqueued):
    payload = {"external_id": f"zd-{uuid.uuid4()}", "subject": "Charged twice", "body": "Two charges on my card"}
    first = signed_webhook(client, payload)
    assert first.status_code == 202
    resend = signed_webhook(client, payload)
    assert resend.status_code == 200
    assert resend.json()["id"] == first.json()["id"]
    ticket = client.get(f"/tickets/{first.json()['id']}").json()
    assert ticket["source"] == "webhook"


def test_webhook_bad_signature_rejected(client, enqueued):
    payload = {"external_id": "zd-1", "subject": "x", "body": "y"}
    assert signed_webhook(client, payload, secret="wrong").status_code == 401
    assert enqueued == []


def test_webhook_stale_timestamp_rejected(client, enqueued):
    payload = {"external_id": "zd-2", "subject": "x", "body": "y"}
    assert signed_webhook(client, payload, timestamp=int(time.time()) - 3600).status_code == 401


def test_webhook_invalid_payload_422(client, enqueued):
    assert signed_webhook(client, {"external_id": "zd-3"}).status_code == 422


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "ok", "redis": "ok"}
