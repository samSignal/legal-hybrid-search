"""Full-text (lexical) retrieval with SQLite FTS5 and BM25 ranking.

FTS5 gives a real inverted index with Porter stemming and BM25, needs no server, and scales to
millions of passages on one machine. Titles are weighted higher than body text.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

from .corpus import Document
from .filters import Filters, to_sql

STOPWORDS = set(
    "a about am an and are as at be been but by can could did do does for from had has have how i if in into is it "
    "its me my of on or our should so than that the their them then there these they this to was we were what when "
    "where which who why will with would you your".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def query_terms(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in STOPWORDS]


class FullTextIndex:
    def __init__(self, path: str | Path = ":memory:", title_weight: float = 2.0):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        # One connection is shared by all threads (e.g. an API server or a parallel evaluation run).
        # sqlite3 connections are not safe for concurrent use, so every access is serialised.
        self._lock = threading.Lock()
        self.title_weight = title_weight
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS docs (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, meta TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                title, body, tokenize='porter unicode61 remove_diacritics 2');
        """)

    def add(self, docs: list[Document]) -> None:
        with self._lock, self.conn:
            for d in docs:
                old = self.conn.execute("SELECT rowid FROM docs WHERE id = ?", (d.id,)).fetchone()
                if old:  # re-indexing a document replaces it
                    self.conn.execute("DELETE FROM docs_fts WHERE rowid = ?", old)
                    self.conn.execute("DELETE FROM docs WHERE rowid = ?", old)
                cur = self.conn.execute("INSERT INTO docs (id, meta) VALUES (?, ?)",
                                        (d.id, json.dumps(d.metadata)))
                self.conn.execute("INSERT INTO docs_fts (rowid, title, body) VALUES (?, ?, ?)",
                                  (cur.lastrowid, d.title, d.text))

    def __len__(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def search(self, query: str, k: int = 10, filters: Filters | None = None) -> list[tuple[str, float]]:
        """Return (doc id, BM25 score) pairs, higher is better.

        Terms are quoted, so user input can never inject FTS5 syntax, and OR-ed so that
        documents matching only some words are still found (BM25 ranks fuller matches higher).
        """
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        where, params = to_sql(filters, "d.meta")
        with self._lock:
            rows = self._query(match, where, params, k)
        return [(r[0], float(r[1])) for r in rows]

    def _query(self, match: str, where: str, params: list, k: int) -> list[tuple]:
        return self.conn.execute(
            f"""SELECT d.id, -bm25(docs_fts, ?, 1.0) AS score
                FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid
                WHERE docs_fts MATCH ? AND {where}
                ORDER BY score DESC LIMIT ?""",
            [self.title_weight, match, *params, k],
        ).fetchall()
