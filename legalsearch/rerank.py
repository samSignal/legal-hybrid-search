"""Second-stage rerankers applied to the top candidates from first-stage retrieval.

* FeatureReranker: a learning-to-rank model (logistic regression) over cheap query/document
  features. Trained on judged queries, it runs in microseconds and needs no GPU or API.
* LLMReranker: asks an LLM to grade each candidate's relevance (listwise). Slower and costs
  tokens, but understands meaning; typically used on the top 10-20 only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .corpus import Document
from .fulltext import query_terms

FEATURES = ["bm25_norm", "bm25_rr", "dense_cos", "dense_rr", "rrf", "title_cov", "body_cov", "bigram", "log_len"]


def _stem(t: str) -> str:
    return t[:5]  # crude prefix stemming: "terminate"/"termination" -> "termi"


@dataclass
class Candidate:
    doc: Document
    bm25: float | None = None
    bm25_rank: int | None = None
    dense: float | None = None
    dense_rank: int | None = None


def featurize(query: str, cands: list[Candidate]) -> np.ndarray:
    q = [_stem(t) for t in query_terms(query)] or [""]
    bigrams = {f"{a} {b}" for a, b in zip(q, q[1:])}
    bm = [c.bm25 for c in cands if c.bm25 is not None]
    lo, hi = (min(bm), max(bm)) if bm else (0.0, 1.0)
    rows = []
    for c in cands:
        title = {_stem(t) for t in query_terms(c.doc.title)}
        body_list = [_stem(t) for t in query_terms(c.doc.text)]
        body = set(body_list)
        body_bigrams = {f"{a} {b}" for a, b in zip(body_list, body_list[1:])}
        rows.append([
            0.0 if c.bm25 is None else ((c.bm25 - lo) / (hi - lo) if hi > lo else 1.0),
            0.0 if c.bm25_rank is None else 1.0 / c.bm25_rank,
            0.0 if c.dense is None else c.dense,
            0.0 if c.dense_rank is None else 1.0 / c.dense_rank,
            sum(1.0 / (60 + r) for r in (c.bm25_rank, c.dense_rank) if r is not None) * 30,
            sum(t in title for t in q) / len(q),
            sum(t in body for t in q) / len(q),
            float(bool(bigrams & body_bigrams)),
            float(np.log1p(len(body_list))),
        ])
    return np.array(rows, dtype=np.float64)


class FeatureReranker:
    name = "ltr"

    def __init__(self, coef: np.ndarray | None = None, intercept: float = 0.0,
                 mean: np.ndarray | None = None, scale: np.ndarray | None = None):
        self.coef, self.intercept, self.mean, self.scale = coef, intercept, mean, scale

    def fit(self, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> "FeatureReranker":
        from sklearn.linear_model import LogisticRegression

        self.mean, self.scale = X.mean(axis=0), X.std(axis=0) + 1e-9
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
        model.fit((X - self.mean) / self.scale, y)
        self.coef, self.intercept = model.coef_[0], float(model.intercept_[0])
        return self

    def score(self, query: str, cands: list[Candidate]) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("Reranker is not trained; call fit() or load()")
        return ((featurize(query, cands) - self.mean) / self.scale) @ self.coef + self.intercept

    def weights(self) -> dict[str, float]:
        return {f: round(float(w), 3) for f, w in zip(FEATURES, self.coef)}

    # JSON instead of pickle: human-readable and safe to load
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"features": FEATURES, "coef": self.coef.tolist(), "intercept": self.intercept,
                                          "mean": self.mean.tolist(), "scale": self.scale.tolist()}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureReranker":
        d = json.loads(Path(path).read_text())
        if d["features"] != FEATURES:
            raise ValueError("Saved reranker was trained on a different feature set")
        return cls(np.array(d["coef"]), d["intercept"], np.array(d["mean"]), np.array(d["scale"]))


LLM_PROMPT = """Rate how well each passage answers the query, from 0 (irrelevant) to 3 (directly answers it).
Reply with JSON only, mapping passage number to score, e.g. {{"1": 3, "2": 0}}.

Query: {query}

{passages}"""


class LLMReranker:
    """`complete` is any function prompt -> text (OpenAI, Ollama, ...), which keeps this testable offline."""

    name = "llm"

    def __init__(self, complete: Callable[[str], str], max_chars: int = 600):
        self.complete, self.max_chars = complete, max_chars

    def score(self, query: str, cands: list[Candidate]) -> np.ndarray:
        passages = "\n\n".join(f"[{i}] {c.doc.title}\n{c.doc.text[:self.max_chars]}" for i, c in enumerate(cands, 1))
        raw = self.complete(LLM_PROMPT.format(query=query, passages=passages))
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            grades = {int(k): float(v) for k, v in json.loads(match.group(0)).items()} if match else {}
        except (ValueError, json.JSONDecodeError):
            grades = {}
        # Unparseable output falls back to the first-stage order instead of failing the search
        return np.array([grades.get(i, 0.0) - i * 1e-3 for i in range(1, len(cands) + 1)])
