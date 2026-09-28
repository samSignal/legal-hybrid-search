"""Combining ranked lists from different retrievers."""
from __future__ import annotations

Ranking = list[tuple[str, float]]


def reciprocal_rank_fusion(rankings: list[Ranking], k: int = 60, weights: list[float] | None = None) -> Ranking:
    """RRF (Cormack et al., 2009): score(d) = sum_i w_i / (k + rank_i(d)).

    Uses ranks only, so it needs no score calibration between BM25 (unbounded) and cosine ([-1, 1]).
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def _minmax(r: Ranking) -> dict[str, float]:
    if not r:
        return {}
    lo, hi = min(s for _, s in r), max(s for _, s in r)
    return {d: (s - lo) / (hi - lo) if hi > lo else 1.0 for d, s in r}


def weighted_score_fusion(rankings: list[Ranking], weights: list[float] | None = None) -> Ranking:
    """Convex combination of min-max normalised scores (a common alternative to RRF)."""
    weights = weights or [1.0 / len(rankings)] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for d, s in _minmax(ranking).items():
            fused[d] = fused.get(d, 0.0) + w * s
    return sorted(fused.items(), key=lambda x: -x[1])
