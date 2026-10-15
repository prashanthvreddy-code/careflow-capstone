"""Re-sort a short list of candidate passages with a cross-encoder. M4.

The embedder (bi-encoder) turns the question and each passage into vectors separately: fast
enough to search everything, but rough. A cross-encoder reads the question and ONE passage
together and scores how well that passage answers it: more accurate, but slow, so it only
re-scores the fused top candidates (A4: a cheap broad net, then an expensive precise sort).

- `FastReranker`: Xenova/ms-marco-MiniLM-L-6-v2 through fastembed (local ONNX, ~80 MB, downloaded
  once into runtime/models/). Scores are relevance logits: higher = more relevant; only their
  order matters, not their size.
- `OverlapReranker`: a tiny deterministic stand-in for tests (shared-word count). No download.
Like the embedder, over-long text is refused rather than silently cut (512-token model limit).
"""
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from capstone.settings import RerankerConfig, get_settings

MAX_INPUT_WORDS = 380


class RerankHit(BaseModel):
    chunk_id: str
    score: float
    rank: int          # 1 = most relevant


class Reranker(Protocol):
    model_name: str

    def scores(self, query: str, texts: Sequence[str]) -> list[float]: ...


def rerank(reranker: Reranker, query: str, candidates: Sequence[tuple[str, str]],
           top_k: int = 5) -> list[RerankHit]:
    """Score (chunk_id, text) candidates against the query; return the best top_k, best first."""
    if not candidates:
        return []
    if not query.strip():
        raise ValueError("empty query")
    ids = [cid for cid, _ in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids")
    texts = [text for _, text in candidates]
    too_long = [cid for cid, text in candidates if len(query.split()) + len(text.split()) > MAX_INPUT_WORDS]
    if too_long:
        raise ValueError(f"{len(too_long)} candidate(s) too long for the reranker: {too_long[:3]}")
    scored = sorted(zip(reranker.scores(query, texts), ids, strict=True), key=lambda s: (-s[0], s[1]))
    return [RerankHit(chunk_id=cid, score=round(float(s), 6), rank=r)
            for r, (s, cid) in enumerate(scored[:top_k], start=1)]


class FastReranker:
    def __init__(self, model_name: str, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else get_settings().runtime_root / "models"
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # loading takes a moment

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextCrossEncoder(self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(s) for s in self._load().rerank(query, list(texts))]


class OverlapReranker:
    """Test stand-in: score = number of distinct query words found in the passage."""

    model_name = "test-overlap"

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        words = set(re.findall(r"[a-z0-9-]+", query.lower()))
        return [float(len(words & set(re.findall(r"[a-z0-9-]+", t.lower())))) for t in texts]


def make_reranker(config: RerankerConfig, cache_dir: Path | None = None) -> Reranker:
    if config.provider == "fastembed":
        return FastReranker(config.model, cache_dir)
    raise ValueError(f"unknown reranker provider: {config.provider}")
