"""Vector stores behind one interface: exact NumPy search, or the Qdrant vector database.

Qdrant runs embedded (in memory or on disk, no server) for development and CI, and the same
class connects to a Qdrant server or cluster by URL in production.
"""
from __future__ import annotations

import uuid
from typing import Any, Protocol

import numpy as np

from .filters import Filters, matches, to_qdrant, validate


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None: ...

    def search(self, vector: np.ndarray, k: int, filters: Filters | None = None) -> list[tuple[str, float]]: ...


class NumpyVectorStore:
    """Exact brute-force cosine search. Fast up to roughly a few hundred thousand vectors."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.meta: list[dict[str, Any]] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self._columns: dict[str, np.ndarray] = {}

    def add(self, ids, vectors, metadata) -> None:
        self.ids += list(ids)
        self.meta += list(metadata)
        self.matrix = vectors if self.matrix.size == 0 else np.vstack([self.matrix, vectors])
        self._columns.clear()

    def _column(self, key: str) -> np.ndarray:
        """Metadata field as a NumPy array (built once), so filters are vectorised instead of a Python loop."""
        if key not in self._columns:
            values = [m.get(key) for m in self.meta]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                self._columns[key] = np.array(values, dtype=np.float64)
            else:
                self._columns[key] = np.array(values, dtype=object)
        return self._columns[key]

    def _mask(self, filters: Filters) -> np.ndarray:
        mask = np.ones(len(self.ids), dtype=bool)
        for key, cond in filters.items():
            col = self._column(key)
            if isinstance(cond, dict):
                if col.dtype == object:  # mixed/missing values: fall back to exact per-row semantics
                    mask &= np.array([matches({key: v}, {key: cond}) for v in col])
                    continue
                ops = {"gt": np.greater, "gte": np.greater_equal, "lt": np.less, "lte": np.less_equal}
                for op, bound in cond.items():
                    mask &= ops[op](col, bound)
            elif isinstance(cond, (list, tuple, set)):
                mask &= np.isin(col, list(cond))
            else:
                mask &= col == cond
        return mask

    def search(self, vector, k, filters=None):
        if not self.ids:
            return []
        scores = self.matrix @ vector
        if filters:
            validate(filters)
            scores = np.where(self._mask(filters), scores, -np.inf)
        k = min(k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]


class QdrantVectorStore:
    """`location` is ":memory:", a local folder path, or a server URL such as http://localhost:6333."""

    def __init__(self, dim: int, location: str = ":memory:", collection: str = "documents"):
        from qdrant_client import QdrantClient, models

        if location.startswith("http"):
            self.client = QdrantClient(url=location)
        elif location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=location)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    @staticmethod
    def _point_id(doc_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))  # Qdrant ids must be ints or UUIDs

    def add(self, ids, vectors, metadata) -> None:
        from qdrant_client import models

        points = [models.PointStruct(id=self._point_id(i), vector=v.tolist(), payload={"doc_id": i, "metadata": m})
                  for i, v, m in zip(ids, vectors, metadata)]
        for start in range(0, len(points), 256):
            self.client.upsert(self.collection, points[start:start + 256])

    def search(self, vector, k, filters=None):
        res = self.client.query_points(self.collection, query=vector.tolist(), limit=k,
                                       query_filter=to_qdrant(filters), with_payload=True)
        return [(p.payload["doc_id"], float(p.score)) for p in res.points]
