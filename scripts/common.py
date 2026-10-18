"""Helpers shared by the offline scripts: dataset loading and cached embeddings."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from app.services.cleaning import clean_ticket_text
from app.services.embeddings import get_embedder

DATA = Path("data")
REPORTS = Path("reports")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Generate the dataset first: python -m scripts.generate_dataset")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tickets(split: str | None = None) -> list[dict]:
    rows = load_jsonl(DATA / "tickets.jsonl")
    return [r for r in rows if split is None or r["split"] == split]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def texts(rows: list[dict]) -> list[str]:
    return [clean_ticket_text(r["subject"], r["body"]) for r in rows]


def embed_rows(rows: list[dict], cache_dir: Path = DATA / "cache") -> np.ndarray:
    """Embed cleaned ticket text, caching on disk keyed by model + exact texts."""
    embedder = get_embedder()
    items = texts(rows)
    key = hashlib.sha256((embedder.version + "\x00" + "\x00".join(items)).encode()).hexdigest()[:16]
    path = cache_dir / f"{embedder.version}-{key}.npy"
    if path.exists():
        return np.load(path)
    vectors = embedder.embed(items).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def try_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("matplotlib not installed (pip install -r requirements-dev.txt); skipping chart")
        return None
