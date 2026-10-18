"""Load data/resolved.jsonl into the database as resolved tickets (the reply-suggestion knowledge base).

  python -m scripts.load_resolved

Idempotent: re-running skips tickets already loaded.
"""

from datetime import UTC

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import SyncSession
from app.models import Ticket, TicketStatus
from scripts.common import DATA, embed_rows, load_jsonl, parse_time, texts


def main() -> None:
    rows = load_jsonl(DATA / "resolved.jsonl")
    vectors = embed_rows(rows)
    cleaned = texts(rows)
    values = [
        {
            "idempotency_key": f"history:{r['id']}",
            "source": "history",
            "external_id": r["id"],
            "subject": r["subject"],
            "body": r["body"],
            "clean_text": text,
            "embedding": vec.tolist(),
            "status": TicketStatus.RESOLVED,
            "resolution": r["reply"],
            "category": r["category"],
            "priority": r["priority"],
            "team": r["team"],
            "created_at": parse_time(r["created_at"]).replace(tzinfo=UTC),
        }
        for r, text, vec in zip(rows, cleaned, vectors)
    ]
    inserted = 0
    with SyncSession() as session:
        for i in range(0, len(values), 200):
            stmt = (
                pg_insert(Ticket)
                .values(values[i : i + 200])
                .on_conflict_do_nothing(index_elements=["idempotency_key"])
                .returning(Ticket.id)  # rowcount is -1 for multi-row inserts; count returned ids
            )
            inserted += len(session.execute(stmt).all())
        session.commit()
    print(f"loaded {inserted} resolved tickets ({len(rows) - inserted} already present)")


if __name__ == "__main__":
    main()
