import pytest

from app.services.signatures import InvalidSignature, sign, verify

SECRET = "s3cret"
BODY = b'{"external_id":"1"}'


def test_valid_signature_passes():
    verify(SECRET, "1000", BODY, sign(SECRET, "1000", BODY), tolerance=300, now=1100)


def test_tampered_body_fails():
    with pytest.raises(InvalidSignature):
        verify(SECRET, "1000", BODY + b" ", sign(SECRET, "1000", BODY), tolerance=300, now=1000)


def test_wrong_secret_fails():
    with pytest.raises(InvalidSignature):
        verify(SECRET, "1000", BODY, sign("other", "1000", BODY), tolerance=300, now=1000)


def test_stale_timestamp_fails():
    with pytest.raises(InvalidSignature):
        verify(SECRET, "1000", BODY, sign(SECRET, "1000", BODY), tolerance=300, now=2000)


def test_non_numeric_timestamp_fails():
    with pytest.raises(InvalidSignature):
        verify(SECRET, "abc", BODY, sign(SECRET, "abc", BODY), tolerance=300)
