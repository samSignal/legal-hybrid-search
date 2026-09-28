"""Command-line interface.

    legalsearch search "landlord kept my deposit" --act RT
    legalsearch eval --data data/norland            # benchmark all configurations
    legalsearch serve                               # REST API on :8000
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .corpus import chunk_documents, load_dataset
from .embeddings import get_encoder
from .evaluate import Config, evaluate, to_markdown, train_reranker
from .search import HybridSearcher
from .vectorstore import NumpyVectorStore, QdrantVectorStore

DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "data" / "norland")


def build_searcher(data: str, encoder: str = "wordllama", store: str = "numpy", chunk: int = 200) -> HybridSearcher:
    docs, _, _ = load_dataset(data)
    enc = get_encoder(encoder)
    vs = NumpyVectorStore() if store == "numpy" else QdrantVectorStore(enc.encode(["x"]).shape[1], store)
    searcher = HybridSearcher(enc, vs)
    searcher.index(chunk_documents(docs, max_words=chunk))
    return searcher


def cmd_search(a) -> None:
    s = build_searcher(a.data, a.encoder, a.store)
    filters = {"act_code": a.act} if a.act else None
    resp = s.search(a.query, k=a.k, mode=a.mode, filters=filters)
    for i, h in enumerate(resp.hits, 1):
        print(f"{i}. [{h.doc.id}] {h.doc.title}  (score {h.score:.4f}, bm25 #{h.bm25_rank}, dense #{h.dense_rank})")
        print(f"   {h.doc.text[:160]}...")
    print(f"\ntimings (ms): {resp.timings_ms}")


def cmd_eval(a) -> None:
    s = build_searcher(a.data, a.encoder, a.store)
    _, test, qrels = load_dataset(a.data, "test")
    try:
        _, train, train_qrels = load_dataset(a.data, "train")
    except FileNotFoundError:  # dataset without training judgements
        train, train_qrels = [], {}
    configs = [Config("BM25 (FTS5)", "bm25"), Config(f"Dense ({s.encoder.name})", "dense"),
               Config("Hybrid, RRF"), Config("Hybrid, weighted scores", fusion="weighted")]
    if train and not {q.id for q in train} & {q.id for q in test}:  # needs a separate training split
        rr = train_reranker(s, train, train_qrels)
        configs.append(Config("Hybrid, RRF + learned reranker", reranker=rr))
        print(f"reranker trained on {len(train)} queries (C={rr.C}); weights: {rr.weights()}\n")
    reports = [evaluate(s, test, qrels, c) for c in configs]
    print(f"{len(test)} test queries, {len(s.docs)} indexed passages\n")
    print(to_markdown(reports))
    for r in reports:
        if r.failures:
            print(f"\n{r.config}: no relevant document in top 10 for {', '.join(r.failures)}")
    if a.out:
        Path(a.out).write_text(json.dumps([asdict(r) for r in reports], indent=2))


def main() -> None:
    p = argparse.ArgumentParser(prog="legalsearch", description="Hybrid legal search")
    p.add_argument("--data", default=DEFAULT_DATA, help="folder with corpus.jsonl, queries.jsonl, qrels")
    p.add_argument("--encoder", default="wordllama", help="wordllama | openai[:model] | ollama[:model]")
    p.add_argument("--store", default="numpy", help="numpy | :memory: | ./qdrant_path | http://host:6333")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    s.add_argument("--act", help="filter by act code, e.g. LC, DP, CR, RT, CA")
    e = sub.add_parser("eval")
    e.add_argument("--out", help="write the results as JSON")
    sub.add_parser("serve")
    a = p.parse_args()
    if a.cmd == "search":
        cmd_search(a)
    elif a.cmd == "eval":
        cmd_eval(a)
    else:
        import uvicorn

        uvicorn.run("legalsearch.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
