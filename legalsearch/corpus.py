"""Loading document collections and relevance judgements.

Uses the BEIR layout, the de-facto standard for retrieval benchmarks, so the same code runs on the
bundled Norland legal collection and on public datasets such as SciFact, NFCorpus or FiQA:

    corpus.jsonl   {"_id", "title", "text", "metadata"?}
    queries.jsonl  {"_id", "text", "metadata"?}
    qrels.tsv      query-id <tab> corpus-id <tab> score      (or qrels/<split>.tsv)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Qrels = dict[str, dict[str, int]]  # query id -> {doc id: graded relevance}


@dataclass
class Document:
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None  # set on chunks produced from a longer document

    @property
    def doc_id(self) -> str:
        """The id relevance judgements refer to (the parent for chunks)."""
        return self.parent_id or self.id


@dataclass
class Query:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(folder: str | Path, split: str | None = None) -> tuple[list[Document], list[Query], Qrels]:
    folder = Path(folder)
    docs = [Document(str(r["_id"]), r.get("title", ""), r.get("text", ""), r.get("metadata", {}))
            for r in _read_jsonl(folder / "corpus.jsonl")]
    queries = [Query(str(r["_id"]), r["text"], r.get("metadata", {})) for r in _read_jsonl(folder / "queries.jsonl")]

    qrels_path = folder / "qrels.tsv"
    if not qrels_path.exists():  # BEIR puts them in qrels/test.tsv etc.
        qrels_path = folder / "qrels" / f"{split or 'test'}.tsv"
    qrels: Qrels = {}
    with qrels_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score

    queries = [q for q in queries if q.id in qrels]
    if split and any("split" in q.metadata for q in queries):
        queries = [q for q in queries if q.metadata.get("split") == split]
    return docs, queries, qrels


def chunk_documents(docs: list[Document], max_words: int = 200, overlap: int = 40) -> list[Document]:
    """Split long documents into overlapping passages; short documents pass through unchanged.

    Each chunk keeps the parent's title and metadata, so filters and citations still work,
    and `parent_id` lets evaluation map passage hits back to the judged document.
    """
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    out: list[Document] = []
    for d in docs:
        words = d.text.split()
        if len(words) <= max_words:
            out.append(d)
            continue
        step = max_words - overlap
        for n, start in enumerate(range(0, len(words), step)):
            out.append(Document(f"{d.id}#p{n}", d.title, " ".join(words[start:start + max_words]),
                                dict(d.metadata), parent_id=d.id))
            if start + max_words >= len(words):
                break
    return out
