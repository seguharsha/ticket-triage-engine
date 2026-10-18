"""Primary classifier: one logistic regression per field on top of the ticket embedding.

Trained offline by scripts/train_classifier.py; predict_proba gives a confidence per field that the
router thresholds. Cheap (microseconds), local, and retrainable from stored human corrections.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.config import settings

FIELDS = ("category", "priority", "team")


@dataclass
class FieldPrediction:
    value: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


class EmbeddingClassifier:
    def __init__(self, models: dict[str, LogisticRegression], version: str):
        self.models = models
        self.version = version

    @classmethod
    def train(cls, X: np.ndarray, labels: dict[str, list[str]], version: str, C: float = 4.0):
        models = {}
        for name in FIELDS:
            model = LogisticRegression(C=C, max_iter=3000)
            model.fit(X, labels[name])
            models[name] = model
        return cls(models, version)

    def predict_many(self, X: np.ndarray) -> list[dict[str, FieldPrediction]]:
        out: list[dict[str, FieldPrediction]] = [{} for _ in range(len(X))]
        for name, model in self.models.items():
            probas = model.predict_proba(X)
            for row, p in enumerate(probas):
                best = int(np.argmax(p))
                out[row][name] = FieldPrediction(
                    value=str(model.classes_[best]),
                    confidence=float(p[best]),
                    probabilities={str(c): float(v) for c, v in zip(model.classes_, p)},
                )
        return out

    def predict(self, embedding) -> dict[str, FieldPrediction]:
        return self.predict_many(np.asarray([embedding], dtype=np.float32))[0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "version": self.version}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingClassifier":
        data = joblib.load(path)
        return cls(data["models"], data["version"])


@lru_cache
def get_classifier() -> EmbeddingClassifier | None:
    """None until scripts/train_classifier.py has been run; tickets then go to the LLM or to review."""
    path = Path(settings.classifier_path)
    return EmbeddingClassifier.load(path) if path.exists() else None
