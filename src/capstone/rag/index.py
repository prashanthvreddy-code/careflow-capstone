"""The policy index: passages + their vectors in Qdrant, searchable by meaning. M3.

- Qdrant runs in embedded on-disk mode (a folder, no server; decided in phase 2). One process may
  open the folder at a time, so use `PolicyIndex` as a context manager (`with ... as index:`).
- Each passage is one Qdrant point: the vector from the embedder plus a payload with its source
  IDs (chunk_id, document title/slug, section, policy id, effective date, source path) and text.
  Point ids are derived from chunk_id, so rebuilding gives the same ids.
- A small sidecar file (<collection>.meta.json) records the embedding model, dimension and a
  fingerprint of the passages. `build()` skips work when nothing changed, rebuilds when the
  corpus or model changed, and `search()` refuses to run with a different model than the one
  the index was built with (vectors from two models are not comparable).
- Evaluation data is never indexed: any passage whose source is in data/careflow/eval is refused
  (the config loader already refuses an eval corpus folder; this is the second lock).
"""
import hashlib
import json
import uuid
from pathlib import Path
from typing import Self

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from capstone.rag.chunking import Chunk, chunk_corpus
from capstone.rag.embeddings import Embedder, make_embedder
from capstone.rag.ingestion import load_corpus
from capstone.settings import RetrievalConfig, get_retrieval_config

EVAL_MARKER = "data/careflow/eval"
POINT_NAMESPACE = uuid.UUID("6f1c2a52-3b1e-4d7a-9a51-0c5e2f7d9b10")


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model/dimension than the one in use."""


class SearchHit(BaseModel):
    chunk_id: str
    doc_title: str
    doc_slug: str
    section: str
    policy_id: str | None
    effective_date: str | None
    source: str
    score: float
    text: str


class BuildReport(BaseModel):
    status: str                 # "built" | "rebuilt" | "up_to_date"
    chunks: int
    documents: int
    model: str
    dimension: int
    fingerprint: str
    problems: list[str] = []


def fingerprint(chunks: list[Chunk]) -> str:
    """Changes whenever any passage id or embedded text changes."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.embed_text.encode())
    return h.hexdigest()[:16]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


class PolicyIndex:
    def __init__(self, path: Path, collection: str, embedder: Embedder):
        self.path = Path(path)
        self.collection = collection
        self.embedder = embedder
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

    # --- lifecycle -------------------------------------------------------------------------------
    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def _meta_path(self) -> Path:
        return self.path / f"{self.collection}.meta.json"

    def meta(self) -> dict | None:
        return json.loads(self._meta_path.read_text()) if self._meta_path.exists() else None

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    # --- build -----------------------------------------------------------------------------------
    def build(self, chunks: list[Chunk], rebuild: bool = False) -> BuildReport:
        bad = [c.chunk_id for c in chunks if EVAL_MARKER in c.source.replace("\\", "/")]
        if bad:
            raise ValueError(f"refusing to index evaluation data: {bad[:3]}")
        if not chunks:
            raise ValueError("no passages to index")
        fp = fingerprint(chunks)
        wanted = {"model": self.embedder.model_name, "dimension": self.embedder.dimension, "fingerprint": fp}
        old = self.meta()
        report = {"chunks": len(chunks), "documents": len({c.doc_slug for c in chunks}),
                  "model": self.embedder.model_name, "dimension": self.embedder.dimension, "fingerprint": fp}
        exists = self.client.collection_exists(self.collection)
        if exists and not rebuild and old and {k: old.get(k) for k in wanted} == wanted \
                and self.count() == len(chunks):
            return BuildReport(status="up_to_date", **report)

        vectors = self.embedder.embed_passages([c.embed_text for c in chunks])
        if exists:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.embedder.dimension, distance=models.Distance.COSINE),
        )
        self.client.upsert(self.collection, points=[
            models.PointStruct(id=point_id(c.chunk_id), vector=v, payload=_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ])
        self._meta_path.write_text(json.dumps({**wanted, "chunks": len(chunks)}, indent=1))
        return BuildReport(status="rebuilt" if exists else "built", **report)

    # --- search ----------------------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, doc_slug: str | None = None) -> list[SearchHit]:
        """The top_k passages closest in meaning to the query, best first, with their source IDs."""
        old = self.meta()
        if not old or not self.client.collection_exists(self.collection):
            raise RuntimeError("index not built yet")
        if (old["model"], old["dimension"]) != (self.embedder.model_name, self.embedder.dimension):
            raise IndexMismatchError(
                f"index built with {old['model']} ({old['dimension']}), searching with "
                f"{self.embedder.model_name} ({self.embedder.dimension}); rebuild the index")
        flt = None
        if doc_slug:
            flt = models.Filter(must=[models.FieldCondition(key="doc_slug", match=models.MatchValue(value=doc_slug))])
        result = self.client.query_points(self.collection, query=self.embedder.embed_query(query),
                                          limit=top_k, query_filter=flt, with_payload=True)
        return [SearchHit(score=p.score, **{k: p.payload.get(k) for k in SearchHit.model_fields if k != "score"})
                for p in result.points]


def _payload(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id, "doc_title": c.doc_title, "doc_slug": c.doc_slug, "section": c.section,
        "policy_id": c.policy_id, "effective_date": c.effective_date.isoformat() if c.effective_date else None,
        "source": c.source, "part": c.part, "n_parts": c.n_parts, "text": c.text,
    }


def open_policy_index(config: RetrievalConfig | None = None, embedder: Embedder | None = None) -> PolicyIndex:
    """The real index at runtime/qdrant (from configs/retrieval.yaml)."""
    cfg = config or get_retrieval_config()
    return PolicyIndex(cfg.index.path, cfg.index.collection, embedder or make_embedder(cfg.embedding))


def build_policy_index(rebuild: bool = False, config: RetrievalConfig | None = None,
                       embedder: Embedder | None = None) -> BuildReport:
    """Read the corpus, chunk it and (re)build the index. Ingestion problems come back in the report."""
    cfg = config or get_retrieval_config()
    loaded = load_corpus(cfg.corpus_dir, cfg.corpus.source)
    chunks = chunk_corpus(loaded.documents, cfg.chunking.max_words)
    with open_policy_index(cfg, embedder) as index:
        report = index.build(chunks, rebuild=rebuild)
    report.problems = loaded.problems
    return report


if __name__ == "__main__":       # M8 setup step: python -m capstone.rag.index [--rebuild]
    import sys
    print(build_policy_index(rebuild="--rebuild" in sys.argv).model_dump_json(indent=1))
