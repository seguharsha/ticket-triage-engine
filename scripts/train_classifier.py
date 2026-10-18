"""Train the embedding classifier on the train split and save it for the worker.

  python -m scripts.train_classifier
  python -m scripts.train_classifier --with-reviewed   # also learn from human-reviewed tickets in the DB

Restart the worker afterwards so it loads the new model.
"""

import argparse
from datetime import UTC, datetime

import numpy as np

from app.config import settings
from app.services.classifier import FIELDS, EmbeddingClassifier
from app.services.embeddings import get_embedder
from scripts.common import embed_rows, load_tickets


def reviewed_from_db() -> tuple[np.ndarray, dict[str, list[str]]]:
    """Tickets a human reviewed; their final labels include every correction."""
    from sqlalchemy import select

    from app.db import SyncSession
    from app.models import Ticket

    with SyncSession() as session:
        rows = session.execute(
            select(Ticket.embedding, Ticket.category, Ticket.priority, Ticket.team).where(
                Ticket.reviewed_at.is_not(None), Ticket.embedding.is_not(None), Ticket.category.is_not(None),
                Ticket.priority.is_not(None), Ticket.team.is_not(None),
            )
        ).all()
    X = np.asarray([np.asarray(r.embedding) for r in rows], dtype=np.float32).reshape(-1, 384)
    return X, {f: [getattr(r, f) for r in rows] for f in FIELDS}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=float, default=4.0, help="inverse regularization strength")
    p.add_argument("--with-reviewed", action="store_true")
    p.add_argument("--out", default=settings.classifier_path)
    args = p.parse_args()

    rows = load_tickets("train")
    X = embed_rows(rows)
    labels = {f: [r[f] for r in rows] for f in FIELDS}
    if args.with_reviewed:
        X_db, labels_db = reviewed_from_db()
        print(f"adding {len(X_db)} human-reviewed tickets from the database")
        X = np.vstack([X, X_db])
        for f in FIELDS:
            labels[f] += labels_db[f]

    version = f"logreg-{get_embedder().version}-{datetime.now(UTC):%Y%m%d%H%M}"
    clf = EmbeddingClassifier.train(X, labels, version, C=args.C)
    clf.save(args.out)

    preds = clf.predict_many(X)
    for f in FIELDS:
        acc = np.mean([p[f].value == y for p, y in zip(preds, labels[f])])
        print(f"train accuracy {f:9s} {acc:.3f}")
    print(f"saved {version} -> {args.out} ({len(X)} examples). Evaluate: python -m scripts.eval_classifier")


if __name__ == "__main__":
    main()
