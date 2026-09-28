"""REST API: GET /search?q=...&mode=hybrid&act=LC&year_min=2018"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from .cli import DEFAULT_DATA, build_searcher

app = FastAPI(title="Legal Hybrid Search", version="1.0.0")


@lru_cache
def searcher():
    return build_searcher(os.getenv("DATA_DIR", DEFAULT_DATA), os.getenv("ENCODER", "wordllama"),
                          os.getenv("VECTOR_STORE", "numpy"))


@app.get("/health")
def health() -> dict:
    s = searcher()
    return {"status": "ok", "passages": len(s.docs), "encoder": s.encoder.name}


@app.get("/search")
def search(q: str = Query(..., min_length=1, max_length=500), k: int = Query(10, ge=1, le=50),
           mode: str = "hybrid", act: list[str] | None = Query(None), year_min: int | None = None,
           year_max: int | None = None) -> dict:
    filters: dict = {}
    if act:
        filters["act_code"] = act
    if year_min is not None or year_max is not None:
        filters["year"] = {k2: v for k2, v in (("gte", year_min), ("lte", year_max)) if v is not None}
    try:
        resp = searcher().search(q, k=k, mode=mode, filters=filters or None)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "query": q,
        "timings_ms": resp.timings_ms,
        "hits": [{"id": h.doc.id, "title": h.doc.title, "text": h.doc.text, "metadata": h.doc.metadata,
                  "score": round(h.score, 5), "bm25_rank": h.bm25_rank, "dense_rank": h.dense_rank}
                 for h in resp.hits],
    }
