# Legal Hybrid Search

![tests](https://github.com/samSignal/legal-hybrid-search/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

The retrieval layer of a legal RAG system, built to be **measured, not just demoed**. It combines full-text BM25 search, dense vector search, metadata filters, rank fusion and learned reranking, and includes a benchmark that reports retrieval quality and latency for every configuration.

People ask legal questions in everyday words ("my landlord kept my deposit") while the law uses formal terms ("security deposit ... deducting only documented damage"). Keyword search misses paraphrases and embeddings miss exact terms, so this project combines both and measures which combination works best.

## What's inside

| Component | Implementation |
|---|---|
| Full-text search | SQLite **FTS5** inverted index, **BM25** ranking, Porter stemming, title boosting, injection-safe query building |
| Dense retrieval | Pluggable encoders: local **WordLlama** (CPU, no API key), **OpenAI** or **Ollama** embeddings |
| Vector database | **Qdrant** (embedded, on-disk or server URL) or exact NumPy search, behind one interface |
| Metadata filters | One filter syntax (`{"act_code": ["LC","DP"], "year": {"gte": 2018}}`) translated to SQL, NumPy masks and Qdrant filters |
| Fusion | **Reciprocal Rank Fusion** and weighted min-max score fusion |
| Reranking | **Learning-to-rank** model (logistic regression over 9 features, C chosen by grouped cross-validation) and a pluggable **LLM reranker** |
| Chunking | Overlapping passage chunking for long documents; hits map back to the parent document for evaluation and citation |
| Evaluation | Recall@k, MRR@10, graded **nDCG@10**, p50/p95 latency, per-query failure lists; loads the **BEIR** format |
| Serving | FastAPI `/search` endpoint and a CLI |

## Results

Benchmark on the bundled Norland legal collection (61 articles from 5 acts, 56 lay-language questions with graded relevance judgements, split 28 train / 28 held-out test). Run it yourself with `legalsearch eval`.

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| BM25 (FTS5) | 0.631 | 0.792 | 0.588 | 0.623 | 0.15 | 0.33 |
| Dense (WordLlama-256) | 0.810 | 0.857 | 0.741 | 0.758 | 0.32 | 0.49 |
| Hybrid, RRF | **0.863** | **0.875** | 0.689 | 0.722 | 0.43 | 0.71 |
| Hybrid, weighted scores | 0.857 | **0.875** | 0.715 | 0.743 | 0.47 | 0.69 |
| Hybrid, RRF + learned reranker | 0.810 | 0.845 | 0.738 | 0.746 | 1.66 | 1.91 |

**What the numbers show**

- **Hybrid search has the best recall.** BM25 alone misses 4 of 28 questions entirely, and RRF fusion cuts that to 2. Recall matters most for RAG, because the LLM can only cite what retrieval finds.
- **Dense retrieval ranks best on these queries** (highest MRR and nDCG). The questions are deliberately paraphrased, which favours embeddings.
- **The learned reranker improves ranking over plain RRF** (MRR 0.689 → 0.738) but slightly lowers recall. With only 28 training queries this is expected; the reranker is trained only on the training split and C is chosen by cross-validation, so the test numbers are not tuned on the test set.
- **Failure analysis:** two questions fail in every configuration. *"maximum penalty for breaking privacy rules"* needs the article titled "Administrative fines" (penalty vs. fine, privacy vs. data protection), and *"deadline for a company to tell me what information it holds about me"* needs "Right of access". Both are vocabulary gaps that a stronger embedding model, legal synonym expansion or the LLM reranker are designed to close. These are the next experiments.

The collection is small and fictional (see [data/norland](data/norland/README.md)), so treat these results as a regression baseline, not a claim about real legal search. The same command runs on public BEIR datasets:

```bash
python scripts/download_beir.py scifact
legalsearch --data data/scifact eval
```

### Scale

`scripts/scale_benchmark.py` indexes synthetic passages and times queries (2 vCPU cloud VM, WordLlama encoder, NumPy store):

| 100,000 passages | Filter | p50 ms | p95 ms |
|---|---|---|---|
| BM25 (FTS5) | - | 37.7 | 90.8 |
| BM25 (FTS5) | act + year | 26.5 | 56.3 |
| Dense | - | 2.6 | 3.6 |
| Dense | act + year | 4.1 | 5.8 |
| Hybrid | - | 43.4 | 96.2 |
| Hybrid | act + year | 34.1 | 63.5 |

Indexing ran at about 4,000 passages/s. Metadata filters in the NumPy store are vectorised (column arrays instead of a Python loop), which took filtered dense search at 100k passages from about 93 ms to 4 ms p50. Beyond a few hundred thousand passages the next step is an ANN index (Qdrant server with HNSW).

## Quick start

```bash
git clone https://github.com/samSignal/legal-hybrid-search.git
cd legal-hybrid-search
pip install -e ".[api,dev]"

legalsearch search "landlord will not give back my deposit" --k 3
legalsearch search "fine for a privacy breach" --act DP
legalsearch eval                                   # the benchmark above
legalsearch --store ./qdrant_data eval             # same, using the Qdrant vector DB
legalsearch --encoder openai:text-embedding-3-small eval   # needs OPENAI_API_KEY
legalsearch serve                                  # http://localhost:8000/docs
```

```text
1. [RT-2] Residential Tenancy Act (2017), Article 2: Security deposit  (score 0.0325, bm25 #1, dense #2)
2. [RT-7] Residential Tenancy Act (2017), Article 7: Termination by the landlord  (score 0.0318, bm25 #5, dense #1)
3. [RT-4] Residential Tenancy Act (2017), Article 4: Landlord's access to the premises  (score 0.0310, bm25 #6, dense #3)
timings (ms): {'bm25': 0.236, 'embed': 0.223, 'dense': 1.331, 'fusion': 0.03, 'total': 1.82}
```

Every hit shows its rank in each retriever, and every response includes per-stage timings, so you can see why a result ranked where it did and where the time went.

### Python API

```python
from legalsearch.corpus import load_dataset, chunk_documents
from legalsearch.embeddings import WordLlamaEncoder
from legalsearch.search import HybridSearcher
from legalsearch.vectorstore import QdrantVectorStore

docs, _, _ = load_dataset("data/norland")
searcher = HybridSearcher(WordLlamaEncoder(), QdrantVectorStore(dim=256, location=":memory:"))
searcher.index(chunk_documents(docs))
resp = searcher.search("can my employer film me at work?", k=5,
                       filters={"act_code": ["DP", "LC"], "year": {"gte": 2018}})
```

## Project layout

```
legalsearch/
  corpus.py       BEIR loader, Document/Query types, passage chunking
  fulltext.py     SQLite FTS5 BM25 index
  embeddings.py   WordLlama / OpenAI / Ollama encoders
  vectorstore.py  NumPy and Qdrant stores
  filters.py      one filter syntax -> SQL, NumPy, Qdrant
  fusion.py       RRF and weighted fusion
  rerank.py       learning-to-rank and LLM rerankers
  search.py       HybridSearcher (pipeline + per-stage timings)
  metrics.py      Recall, MRR, nDCG, percentiles
  evaluate.py     benchmark runner, reranker training with CV
  api.py, cli.py
scripts/          scale benchmark, BEIR downloader
data/norland/     bundled benchmark collection
```

## Tests

`pytest -q` runs 13 tests, and GitHub Actions runs them on every push, then posts the benchmark table to the run summary. The tests cover metric correctness against hand-computed values, filter behaviour across SQL/NumPy/Qdrant, FTS injection safety, **identical results from Qdrant and NumPy**, reranker save/load, LLM-reranker fallback on unparseable output, and a **quality regression gate** that fails the build if hybrid Recall@10 drops below 0.85 or stops beating BM25.

## License

MIT.
