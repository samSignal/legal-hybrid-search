"""HybridSearcher: full-text + dense retrieval, fusion, filters and optional reranking."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .corpus import Document
from .embeddings import Encoder
from .filters import Filters
from .fulltext import FullTextIndex
from .fusion import reciprocal_rank_fusion, weighted_score_fusion
from .rerank import Candidate
from .vectorstore import NumpyVectorStore, VectorStore

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    doc: Document
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


@dataclass
class SearchResponse:
    query: str
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridSearcher:
    def __init__(self, encoder: Encoder, vector_store: VectorStore | None = None,
                 fulltext: FullTextIndex | None = None):
        self.encoder = encoder
        self.vectors = vector_store or NumpyVectorStore()
        self.fulltext = fulltext or FullTextIndex()
        self.docs: dict[str, Document] = {}
        self._encode_lock = threading.Lock()  # encoders (tokenizers, HTTP clients) are not guaranteed thread-safe
        self._embed_query = lru_cache(maxsize=1024)(self._embed_query_uncached)

    def index(self, docs: list[Document], batch_size: int = 256) -> None:
        self.fulltext.add(docs)
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            vecs = self.encoder.encode([f"{d.title}\n{d.text}" for d in batch])
            self.vectors.add([d.id for d in batch], vecs, [d.metadata for d in batch])
        self.docs.update({d.id: d for d in docs})

    def _embed_query_uncached(self, query: str) -> np.ndarray:
        with self._encode_lock:
            return self.encoder.encode([query])[0]

    def search(self, query: str, k: int = 10, mode: str = "hybrid", filters: Filters | None = None,
               candidates: int = 50, fusion: str = "rrf", weights: list[float] | None = None,
               reranker=None, rerank_top: int = 20) -> SearchResponse:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        t: dict[str, float] = {}
        n = max(k, candidates)
        bm25: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []

        if mode in ("bm25", "hybrid"):
            s = time.perf_counter()
            bm25 = self.fulltext.search(query, n, filters)
            t["bm25"] = (time.perf_counter() - s) * 1000
        if mode in ("dense", "hybrid"):
            s = time.perf_counter()
            qv = self._embed_query(query)
            t["embed"] = (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            dense = self.vectors.search(qv, n, filters)
            t["dense"] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        if mode == "bm25":
            ranked = bm25
        elif mode == "dense":
            ranked = dense
        elif fusion == "rrf":
            ranked = reciprocal_rank_fusion([bm25, dense], weights=weights)
        else:
            ranked = weighted_score_fusion([bm25, dense], weights=weights)
        t["fusion"] = (time.perf_counter() - s) * 1000

        bm_info = {d: (sc, r) for r, (d, sc) in enumerate(bm25, 1)}
        de_info = {d: (sc, r) for r, (d, sc) in enumerate(dense, 1)}

        def cand(doc_id: str) -> Candidate:
            b, d = bm_info.get(doc_id, (None, None)), de_info.get(doc_id, (None, None))
            return Candidate(self.docs[doc_id], b[0], b[1], d[0], d[1])

        if reranker is not None and ranked:
            s = time.perf_counter()
            top = [cand(d) for d, _ in ranked[:rerank_top]]
            scores = reranker.score(query, top)
            order = np.argsort(-scores, kind="stable")
            ranked = [(top[i].doc.id, float(scores[i])) for i in order] + ranked[rerank_top:]
            t["rerank"] = (time.perf_counter() - s) * 1000

        hits = []
        for doc_id, score in ranked[:k]:
            c = cand(doc_id)
            hits.append(Hit(c.doc, score, c.bm25_rank, c.dense_rank))
        t["total"] = sum(t.values())
        return SearchResponse(query, hits, {key: round(v, 3) for key, v in t.items()})

    def candidates_for_training(self, query: str, n: int = 20, filters: Filters | None = None) -> list[Candidate]:
        """Hybrid candidates with their first-stage signals, used to train the FeatureReranker."""
        resp = self.search(query, k=n, mode="hybrid", filters=filters, candidates=max(50, n))
        bm = dict(self.fulltext.search(query, max(50, n), filters))
        de = dict(self.vectors.search(self._embed_query(query), max(50, n), filters))
        return [Candidate(h.doc, bm.get(h.doc.id), h.bm25_rank, de.get(h.doc.id), h.dense_rank) for h in resp.hits]
