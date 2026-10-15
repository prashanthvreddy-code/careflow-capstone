"""Turn text into vectors (embeddings) for the policy index. M3.

- `FastEmbedder`: the real model chosen in phase 2, BAAI/bge-small-en-v1.5 through fastembed.
  Runs locally (ONNX), no API calls, same output every run. The model files (~67 MB) download
  once into runtime/models/ (gitignored) on first use, then load from disk.
- `HashEmbedder`: a tiny deterministic stand-in for tests (counts hashed words). No download;
  useful for testing the index code, useless for real search quality.

Both report `model_name` and `dimension`; the index stores them and refuses to mix vectors from
different models (vectors from two models are not comparable). Passages and queries go through
separate methods because some models treat them differently; for bge-small the difference is
small, but keeping them apart means a model swap can't silently skip it.
"""
import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from capstone.settings import EmbeddingConfig, get_settings

MAX_INPUT_WORDS = 380   # bge-small reads at most 512 tokens; longer text would be cut silently


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _check_lengths(texts: Sequence[str]) -> None:
    too_long = [i for i, t in enumerate(texts) if len(t.split()) > MAX_INPUT_WORDS]
    if too_long:
        raise ValueError(f"{len(too_long)} text(s) exceed {MAX_INPUT_WORDS} words and would be truncated")


class FastEmbedder:
    def __init__(self, model_name: str, dimension: int, cache_dir: Path | None = None):
        self.model_name = model_name
        self.dimension = dimension
        self.cache_dir = Path(cache_dir) if cache_dir else get_settings().runtime_root / "models"
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported here: loading takes a moment

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def _vectors(self, arrays) -> list[list[float]]:
        vectors = [[float(x) for x in a] for a in arrays]
        for v in vectors:
            if len(v) != self.dimension:
                raise ValueError(f"model returned {len(v)} numbers, config says {self.dimension}")
        return vectors

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        _check_lengths(texts)
        return self._vectors(self._load().passage_embed(list(texts)))

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("empty query")
        _check_lengths([text])
        return self._vectors(self._load().query_embed([text]))[0]


class HashEmbedder:
    """Deterministic bag-of-words vectors for tests: same words → same vector, no model needed."""

    def __init__(self, dimension: int = 64):
        self.model_name = f"test-hash-{dimension}"
        self.dimension = dimension

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9-]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("empty query")
        return self._one(text)


def make_embedder(config: EmbeddingConfig, cache_dir: Path | None = None) -> Embedder:
    """The embedder named in configs/retrieval.yaml."""
    if config.provider == "fastembed":
        return FastEmbedder(config.model, config.dimension, cache_dir)
    raise ValueError(f"unknown embedding provider: {config.provider}")
