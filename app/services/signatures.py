"""HMAC webhook signatures: sha256=hex(HMAC(secret, "<timestamp>.<raw body>"))."""

import hashlib
import hmac
import time


class InvalidSignature(Exception):
    pass


def sign(secret: str, timestamp: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify(secret: str, timestamp: str, body: bytes, signature: str, tolerance: int, now: float | None = None) -> None:
    try:
        sent_at = int(timestamp)
    except ValueError as exc:
        raise InvalidSignature("bad timestamp") from exc
    # The timestamp is inside the signed payload, so an old captured request can't be replayed later.
    if abs((now or time.time()) - sent_at) > tolerance:
        raise InvalidSignature("timestamp outside tolerance")
    if not hmac.compare_digest(sign(secret, timestamp, body), signature):
        raise InvalidSignature("signature mismatch")
