"""Benchmark retrieval configurations: quality (Recall, MRR, nDCG) and latency (p50/p95)."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

import numpy as np

from .corpus import Qrels, Query
from .metrics import dedupe, mrr_at_k, ndcg_at_k, percentile, recall_at_k
from .rerank import FeatureReranker, featurize
from .search import HybridSearcher


@dataclass
class Config:
    name: str
    mode: str = "hybrid"
    fusion: str = "rrf"
    weights: list[float] | None = None
    reranker: object = None


@dataclass
class Report:
    config: str
    queries: int
    recall_5: float
    recall_10: float
    mrr_10: float
    ndcg_10: float
    p50_ms: float
    p95_ms: float
    failures: list[str] = field(default_factory=list)  # query ids with no relevant doc in the top 10


def evaluate(searcher: HybridSearcher, queries: list[Query], qrels: Qrels, config: Config, k: int = 10) -> Report:
    r5, r10, mrr, ndcg, lat, failures = [], [], [], [], [], []
    searcher._embed_query.cache_clear()  # measure real query-embedding latency for every configuration
    for q in queries:
        resp = searcher.search(q.text, k=k * 3, mode=config.mode, fusion=config.fusion,
                               weights=config.weights, reranker=config.reranker)
        ranked = dedupe([h.doc.doc_id for h in resp.hits])  # chunk hits count once per document
        rel = qrels[q.id]
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, k))
        mrr.append(mrr_at_k(ranked, rel, k))
        ndcg.append(ndcg_at_k(ranked, rel, k))
        lat.append(resp.timings_ms["total"])
        if r10[-1] == 0:
            failures.append(q.id)
    return Report(config.name, len(queries), round(mean(r5), 3), round(mean(r10), 3), round(mean(mrr), 3),
                  round(mean(ndcg), 3), round(percentile(lat, 50), 2), round(percentile(lat, 95), 2), failures)


def train_reranker(searcher: HybridSearcher, queries: list[Query], qrels: Qrels, n: int = 20,
                   Cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0), folds: int = 4) -> FeatureReranker:
    """Fit the learning-to-rank model on first-stage candidates of the *training* queries only.

    The regularisation strength C is chosen by cross-validation over queries (grouped, so the
    candidates of one query never appear in both the training and validation fold), scored by nDCG@10.
    """
    data = []
    for q in queries:
        cands = searcher.candidates_for_training(q.text, n)
        if cands:
            data.append((q, cands, featurize(q.text, cands), np.array([int(c.doc.doc_id in qrels[q.id]) for c in cands])))

    def fit(rows, C):
        return FeatureReranker().fit(np.vstack([r[2] for r in rows]), np.concatenate([r[3] for r in rows]), C=C)

    def cv_score(C):
        scores = []
        for f in range(folds):
            train = [r for i, r in enumerate(data) if i % folds != f]
            val = [r for i, r in enumerate(data) if i % folds == f]
            if not val or len({y for r in train for y in r[3]}) < 2:
                continue
            model = fit(train, C)
            for q, cands, _, _ in val:
                order = np.argsort(-model.score(q.text, cands), kind="stable")
                scores.append(ndcg_at_k([cands[i].doc.doc_id for i in order], qrels[q.id], 10))
        return mean(scores) if scores else 0.0

    best_C = max(Cs, key=cv_score)
    model = fit(data, best_C)
    model.C = best_C
    return model


def to_markdown(reports: list[Report]) -> str:
    head = "| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |\n|---|---|---|---|---|---|---|"
    rows = [f"| {r.config} | {r.recall_5:.3f} | {r.recall_10:.3f} | {r.mrr_10:.3f} | {r.ndcg_10:.3f} | "
            f"{r.p50_ms:.2f} | {r.p95_ms:.2f} |" for r in reports]
    return "\n".join([head, *rows])
