"""Create the pgvector extension and all tables. Run once: `python -m app.init_db`. Safe to re-run."""

from sqlalchemy import text

from app.db import sync_engine
from app.models import Base

# Columns added after the first release. create_all() creates missing tables but never alters
# existing ones, so each addition gets an idempotent statement here (a stand-in for Alembic).
MIGRATIONS = [
    "ALTER TABLE tickets ADD COLUMN IF NOT EXISTS dedup_seq BIGINT",
    "CREATE INDEX IF NOT EXISTS ix_tickets_dedup_seq ON tickets (dedup_seq)",
]


def main() -> None:
    with sync_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(sync_engine)
    with sync_engine.begin() as conn:
        for statement in MIGRATIONS:
            conn.execute(text(statement))
    print("schema ready")


if __name__ == "__main__":
    main()
