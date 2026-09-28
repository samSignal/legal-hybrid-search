"""Standard IR metrics. `ranked` is a list of doc ids, `rel` maps doc id -> graded relevance (>0)."""
from __future__ import annotations

import math


def dedupe(ranked: list[str]) -> list[str]:
    """Collapse passage hits to their first occurrence (chunks of the same document count once)."""
    seen, out = set(), []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & set(rel)) / len(rel) if rel else 0.0


def precision_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return len(set(ranked[:k]) & set(rel)) / k


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    """nDCG with graded gains (2^rel - 1), as in TREC and BEIR."""
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)
