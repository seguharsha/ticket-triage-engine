import os
import re

# Deterministic, fast, offline: must be set before app.config is imported.
os.environ["EMBEDDER"] = "hashing"
os.environ["LLM_PROVIDER"] = "fake"
os.environ["FAKE_LLM_DELAY_MS"] = "0"
os.environ["CLASSIFIER_PATH"] = "tests/.no-classifier.joblib"
os.environ["LOG_JSON"] = "false"

# Tests get their own database, so test rows never mix with replayed or load-test data.
_LIVE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://triage:triage@localhost:5433/triage")
TEST_DB = "triage_test"
os.environ["DATABASE_URL"] = re.sub(r"/[^/?]+(\?|$)", f"/{TEST_DB}\\1", _LIVE_URL, count=1)
# Same for Redis: a separate logical database, so tests never touch live queues, metrics or dead letters.
os.environ["REDIS_URL"] = re.sub(r"/\d+$", "", os.environ.get("REDIS_URL", "redis://localhost:6379/0")) + "/15"


def _ensure_test_database() -> None:
    """Create the test database and schema if Postgres is reachable; otherwise leave DB tests to fail."""
    try:
        import psycopg

        admin_url = re.sub(r"/[^/?]+(\?|$)", "/postgres\\1", _LIVE_URL, count=1).replace("+psycopg", "")
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=3) as conn:
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
                conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    except Exception:
        return
    from app.init_db import main as init_schema

    init_schema()


_ensure_test_database()

import uuid  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    # One client for the whole session keeps the async DB pool on a single event loop.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def enqueued(monkeypatch):
    """Capture enqueued ticket ids instead of sending them to Redis."""
    calls: list[uuid.UUID] = []

    async def fake_enqueue(ticket_id):
        calls.append(ticket_id)

    monkeypatch.setattr("app.api.tickets.enqueue_ticket", fake_enqueue)
    return calls


@pytest.fixture
def key():
    return f"test-{uuid.uuid4()}"


class RecordingMetrics:
    """Stands in for the Redis-backed WorkerMetrics."""

    def __init__(self):
        self.observed: dict[str, list[float]] = {}
        self.counters: dict[str, float] = {}

    def observe(self, name, seconds):
        self.observed.setdefault(name, []).append(seconds)

    def inc(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount


@pytest.fixture
def metrics():
    return RecordingMetrics()
