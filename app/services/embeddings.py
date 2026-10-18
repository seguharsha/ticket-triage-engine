"""Text -> unit-length 384-d vectors. Cosine similarity is then a plain dot product."""

import hashlib
import os
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import settings
from app.models import EMBEDDING_DIM


class Embedder(Protocol):
    version: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        # Public model: don't send a locally saved Hugging Face token (an expired one breaks the download).
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        import torch
        from sentence_transformers import SentenceTransformer  # heavy import; only where used

        if settings.torch_threads:
            torch.set_num_threads(settings.torch_threads)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.version = model_name.rsplit("/", 1)[-1]

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32, convert_to_numpy=True)


class HashingEmbedder:
    """Feature-hashed bag of words + bigrams. No model download; for tests and fast local runs.

    Texts sharing words get similar vectors, so dedup logic is testable, but it doesn't understand
    meaning: never use it for evaluation numbers.
    """

    version = "hashing-v1"
    _token = re.compile(r"[a-z0-9]+")

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self._token.findall(text.lower())
            for feature in tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]:
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
                out[row, h % EMBEDDING_DIM] += 1.0 if (h >> 63) else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


@lru_cache
def get_embedder() -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)


def embed_one(text: str) -> list[float]:
    return get_embedder().embed([text])[0].tolist()
