"""Latency and indexing throughput at scale.

Generates N synthetic passages by recombining sentences from the Norland corpus (with random act
and year metadata), indexes them, and times queries per retrieval mode, with and without filters.
This measures speed only; quality is measured by `legalsearch eval` on judged queries.

    python scripts/scale_benchmark.py --n 100000
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legalsearch.corpus import Document, load_dataset  # noqa: E402
from legalsearch.embeddings import get_encoder  # noqa: E402
from legalsearch.metrics import percentile  # noqa: E402
from legalsearch.search import HybridSearcher  # noqa: E402


def synth(docs: list[Document], n: int, seed: int = 7) -> list[Document]:
    rng = random.Random(seed)
    sents = [s for d in docs for s in re.split(r"(?<=\.)\s+", d.text)]
    acts = sorted({d.metadata["act_code"] for d in docs})
    return [Document(f"S-{i}", f"Synthetic passage {i}", " ".join(rng.sample(sents, 4)),
                     {"act_code": rng.choice(acts), "year": rng.randint(2000, 2025)}) for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--encoder", default="wordllama")
    a = p.parse_args()

    base, queries, _ = load_dataset(Path(__file__).resolve().parent.parent / "data" / "norland")
    docs = synth(base, a.n)
    s = HybridSearcher(get_encoder(a.encoder))
    t = time.perf_counter()
    s.index(docs, batch_size=2048)
    secs = time.perf_counter() - t
    print(f"Indexed {a.n:,} passages in {secs:.1f}s ({a.n / secs:,.0f} passages/s) with {s.encoder.name}\n")

    print("| Mode | Filter | p50 ms | p95 ms |\n|---|---|---|---|")
    for mode in ("bm25", "dense", "hybrid"):
        for filt in (None, {"act_code": "RT", "year": {"gte": 2015}}):
            s._embed_query.cache_clear()
            lat = [s.search(q.text, k=10, mode=mode, filters=filt).timings_ms["total"] for q in queries]
            print(f"| {mode} | {'act + year' if filt else '-'} | {percentile(lat, 50):.1f} | {percentile(lat, 95):.1f} |")


if __name__ == "__main__":
    main()
