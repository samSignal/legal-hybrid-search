import numpy as np

from legalsearch.evaluate import Config, evaluate, train_reranker
from legalsearch.rerank import FeatureReranker, LLMReranker
from legalsearch.search import HybridSearcher
from legalsearch.vectorstore import NumpyVectorStore, QdrantVectorStore


def test_hybrid_finds_the_right_article(searcher):
    top = searcher.search("landlord will not give back my deposit", k=3).hits
    assert top[0].doc.id == "RT-2"
    assert top[0].bm25_rank is not None and top[0].dense_rank is not None


def test_filters_restrict_results(searcher):
    hits = searcher.search("notice period", k=10, filters={"act_code": "RT"}).hits
    assert hits and all(h.doc.metadata["act_code"] == "RT" for h in hits)
    hits = searcher.search("notice", k=10, filters={"year": {"gte": 2019}}).hits
    assert all(h.doc.metadata["year"] >= 2019 for h in hits)


def test_qdrant_and_numpy_return_the_same_results(encoder, dataset):
    docs = dataset[0]
    vecs = encoder.encode([f"{d.title}\n{d.text}" for d in docs])
    stores = [NumpyVectorStore(), QdrantVectorStore(vecs.shape[1], ":memory:")]
    for s in stores:
        s.add([d.id for d in docs], vecs, [d.metadata for d in docs])
    q = encoder.encode(["can my employer read my emails"])[0]
    for filt in (None, {"act_code": ["DP", "LC"]}, {"year": {"gte": 2019}}):
        a, b = (s.search(q, 5, filt) for s in stores)
        assert [x[0] for x in a] == [x[0] for x in b]
        assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-4)


def test_quality_regression_gate(searcher, dataset):
    """Fails CI if a change makes retrieval worse than the recorded baseline on the held-out queries."""
    _, _, test, qrels = dataset
    hybrid = evaluate(searcher, test, qrels, Config("hybrid"))
    bm25 = evaluate(searcher, test, qrels, Config("bm25", "bm25"))
    assert hybrid.recall_10 >= 0.85 and hybrid.ndcg_10 >= 0.70
    assert hybrid.recall_10 > bm25.recall_10  # hybrid must beat lexical-only


def test_learned_reranker_trains_saves_and_loads(searcher, dataset, tmp_path):
    _, train, test, qrels = dataset
    rr = train_reranker(searcher, train, qrels)
    assert rr.C in (0.01, 0.1, 1.0, 10.0)
    path = tmp_path / "rr.json"
    rr.save(path)
    loaded = FeatureReranker.load(path)
    cands = searcher.candidates_for_training(test[0].text)
    assert np.allclose(rr.score(test[0].text, cands), loaded.score(test[0].text, cands))
    report = evaluate(searcher, test, qrels, Config("ltr", reranker=loaded))
    assert report.mrr_10 > 0.6


def test_llm_reranker_uses_grades_and_survives_bad_output(searcher):
    q = "who pays postage when returning an online order"
    graded = searcher.search(q, k=5, reranker=LLMReranker(lambda p: '{"3": 3, "1": 1}'), rerank_top=5).hits
    base = searcher.search(q, k=5).hits
    assert graded[0].doc.id == base[2].doc.id  # the passage the "LLM" graded 3 moves to the top
    garbage = searcher.search(q, k=5, reranker=LLMReranker(lambda p: "sorry, I can't"), rerank_top=5).hits
    assert [h.doc.id for h in garbage] == [h.doc.id for h in base]


def test_chunked_documents_are_searchable(encoder):
    from legalsearch.corpus import Document, chunk_documents

    long = Document("BIG", "Big act", ("filler " * 300) + "the secret clause about parking spaces " + ("filler " * 300))
    s = HybridSearcher(encoder)
    s.index(chunk_documents([long], max_words=200, overlap=40))
    hit = s.search("parking spaces", k=1, mode="bm25").hits[0]
    assert hit.doc.doc_id == "BIG" and "parking" in hit.doc.text
