"""Find the evidence for a question: dense, hybrid (dense + keyword, RRF), or hybrid + rerank. M4.

    dense          Qdrant top-k by meaning (the M3 baseline)
    hybrid         Qdrant top dense_pool  +  BM25 top keyword_pool, merged with Reciprocal Rank
                   Fusion: each passage scores sum(1 / (rrf_k + rank)) over the lists it is in,
                   so a passage ranked high by EITHER search rises (ranks, not raw scores, are
                   combined, because cosine and BM25 scores are not comparable)
    hybrid_rerank  the fused top rerank_pool, re-sorted by the cross-encoder

Topic routes (M4 fixes X1/X1b): if the question uses a configured term (e.g. "deductible", "pay")
and the topic's anchor passage is not among the results, the anchor is added as one extra source,
marked `routed`. The cost-share anchor is the plan cost-share table (copay-schedule#4.2): every
plan cost question needs it, but search often returned the worked example or the calculation
rules instead (EV-019, NC-03). Adding a whole document's "best" passage was tried first (X1) and
picked the wrong passage.

Every returned passage carries its source IDs (chunk_id, document title, section, policy id,
effective date, file) and its rank in each stage, so a wrong answer can be traced to the stage
that went wrong. The keyword index is built from the same passages as the Qdrant index; if the
index was built from different passages (corpus changed, not rebuilt) retrieval refuses to run.
"""
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel

from capstone.rag.chunking import Chunk, chunk_corpus
from capstone.rag.embeddings import Embedder
from capstone.rag.index import PolicyIndex, fingerprint, open_policy_index
from capstone.rag.ingestion import load_corpus
from capstone.rag.keyword import BM25Index, tokenize
from capstone.rag.rerank import Reranker, make_reranker, rerank
from capstone.settings import RetrievalConfig, SearchConfig, get_retrieval_config

Mode = Literal["dense", "hybrid", "hybrid_rerank"]


class StageRanks(BaseModel):
    dense: int | None = None
    keyword: int | None = None
    fused: int | None = None
    rerank: int | None = None
    routed: bool = False      # added by a topic route, not by ranking


class EvidenceHit(BaseModel):
    chunk_id: str
    doc_title: str
    doc_slug: str
    section: str
    policy_id: str | None
    effective_date: str | None
    source: str
    text: str
    rank: int                 # final position, 1 = best
    ranks: StageRanks


class Evidence(BaseModel):
    query: str
    mode: Mode
    hits: list[EvidenceHit]

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Merge ranked id lists: score = sum of 1 / (k + rank). Ties broken by id (repeatable)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


class Retriever:
    def __init__(self, index: PolicyIndex, chunks: list[Chunk], search: SearchConfig, top_k: int = 5,
                 reranker: Reranker | None = None):
        meta = index.meta()
        if not meta or meta.get("fingerprint") != fingerprint(chunks):
            raise RuntimeError("the index was built from different passages; rebuild it first")
        self.index, self.search, self.top_k, self.reranker = index, search, top_k, reranker
        self.chunks = {c.chunk_id: c for c in chunks}
        unknown = [r.anchor for r in search.topic_routes if r.anchor not in self.chunks]
        if unknown:
            raise ValueError(f"topic route anchor is not a known passage: {unknown}")
        self.keyword = BM25Index((c.chunk_id, c.embed_text) for c in chunks)

    def retrieve(self, query: str, mode: Mode | None = None, top_k: int | None = None) -> Evidence:
        mode = mode or self.search.mode
        top_k = top_k or self.top_k
        if not query.strip():
            raise ValueError("empty query")
        ranks: dict[str, StageRanks] = {}

        def note(stage: str, ids: Sequence[str]) -> None:
            for r, cid in enumerate(ids, start=1):
                setattr(ranks.setdefault(cid, StageRanks()), stage, r)

        dense = [h.chunk_id for h in self.index.search(query, top_k=max(self.search.dense_pool, top_k))]
        note("dense", dense)
        if mode == "dense":
            final = dense[:top_k]
        else:
            keyword = [h.chunk_id for h in self.keyword.search(query, top_k=self.search.keyword_pool)]
            note("keyword", keyword)
            fused = rrf_fuse([dense, keyword], k=self.search.rrf_k)
            note("fused", fused)
            if mode == "hybrid":
                final = fused[:top_k]
            else:
                if self.reranker is None:
                    raise RuntimeError("mode hybrid_rerank needs a reranker")
                pool = fused[: max(self.search.rerank_pool, top_k)]
                reranked = rerank(self.reranker, query, [(cid, self.chunks[cid].embed_text) for cid in pool],
                                  top_k=len(pool))
                note("rerank", [h.chunk_id for h in reranked])
                final = [h.chunk_id for h in reranked][:top_k]
        for cid in self._route_topics(query, final):
            final.append(cid)
            ranks.setdefault(cid, StageRanks()).routed = True
        return Evidence(query=query, mode=mode, hits=[self._hit(cid, r, ranks[cid]) for r, cid in
                                                      enumerate(final, start=1)])

    def _route_topics(self, query: str, final: list[str]) -> list[str]:
        """Anchor passages of the topics the question names but the results miss."""
        words, added = set(tokenize(query)), []
        for route in self.search.topic_routes:
            terms = {t for term in route.terms for t in tokenize(term)}
            if words & terms and route.anchor not in final + added:
                added.append(route.anchor)
        return added

    def _hit(self, cid: str, rank: int, stage: StageRanks) -> EvidenceHit:
        c = self.chunks[cid]
        return EvidenceHit(chunk_id=cid, doc_title=c.doc_title, doc_slug=c.doc_slug, section=c.section,
                           policy_id=c.policy_id,
                           effective_date=c.effective_date.isoformat() if c.effective_date else None,
                           source=c.source, text=c.text, rank=rank, ranks=stage)


class PolicyRetriever:
    """The real retriever from configs/retrieval.yaml; use as `with PolicyRetriever() as r:`."""

    def __init__(self, config: RetrievalConfig | None = None, embedder: Embedder | None = None,
                 reranker: Reranker | None = None):
        cfg = config or get_retrieval_config()
        chunks = chunk_corpus(load_corpus(cfg.corpus_dir, cfg.corpus.source).documents, cfg.chunking.max_words)
        if reranker is None and cfg.reranker:
            reranker = make_reranker(cfg.reranker)      # lazy: the model loads on first rerank only
        self._index = open_policy_index(cfg, embedder)
        try:
            self.retriever = Retriever(self._index, chunks, cfg.retrieval, cfg.index.top_k, reranker)
        except Exception:
            self._index.close()
            raise

    def retrieve(self, query: str, mode: Mode | None = None, top_k: int | None = None) -> Evidence:
        return self.retriever.retrieve(query, mode, top_k)

    def close(self) -> None:
        self._index.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
