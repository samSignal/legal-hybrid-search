import math
import sqlite3

import pytest

from legalsearch.corpus import Document, chunk_documents
from legalsearch.filters import matches, to_qdrant, to_sql
from legalsearch.fulltext import FullTextIndex
from legalsearch.fusion import reciprocal_rank_fusion, weighted_score_fusion
from legalsearch.metrics import dedupe, mrr_at_k, ndcg_at_k, percentile, recall_at_k


def test_metrics_against_hand_computed_values():
    ranked, rel = ["a", "b", "c", "d"], {"b": 2, "d": 1}
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert mrr_at_k(ranked, rel) == 0.5
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, rel) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["b", "d"], rel) == pytest.approx(1.0)
    assert dedupe(["a", "a", "b"]) == ["a", "b"]
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[("a", 9.0), ("b", 5.0)], [("b", 0.9), ("c", 0.8)]])
    assert fused[0][0] == "b"  # ranked by both retrievers
    assert weighted_score_fusion([[("a", 10), ("b", 0)], [("b", 1.0), ("a", 0.0)]], [0.3, 0.7])[0][0] == "b"


def test_filter_semantics_match_across_backends():
    meta = {"act_code": "RT", "year": 2017}
    assert matches(meta, {"act_code": ["RT", "LC"], "year": {"gte": 2017, "lt": 2018}})
    assert not matches(meta, {"year": {"gt": 2017}})
    assert not matches({}, {"year": {"gte": 2000}})
    where, params = to_sql({"act_code": ["RT"], "year": {"gte": 2017}})
    row = sqlite3.connect(":memory:").execute(f"SELECT {where} FROM (SELECT ? AS meta)",
                                              [*params, '{"act_code": "RT", "year": 2017}']).fetchone()
    assert row == (1,)
    assert len(to_qdrant({"act_code": "RT", "year": {"gte": 2017}}).must) == 2
    with pytest.raises(ValueError):
        matches(meta, {"year') OR 1=1 --": 1})
    with pytest.raises(ValueError):
        matches(meta, {"year": {"between": 1}})


def test_fulltext_stemming_filters_and_injection_safety():
    idx = FullTextIndex()
    idx.add([Document("1", "Termination", "The landlord terminated the lease.", {"act": "RT"}),
             Document("2", "Wages", "Wages are paid twice a month.", {"act": "LC"})])
    assert idx.search("terminating leases")[0][0] == "1"  # porter stemming
    assert idx.search("wages", filters={"act": "RT"}) == []
    for nasty in ['"', "NEAR(a b)", "wages OR *", "a AND NOT b", "'; DROP TABLE docs; --"]:
        idx.search(nasty)  # must never raise an FTS syntax error
    idx.add([Document("1", "Termination", "Updated text about eviction.", {"act": "RT"})])
    assert len(idx) == 2 and idx.search("eviction")[0][0] == "1"


def test_chunking_keeps_parent_and_metadata():
    long = Document("D", "T", " ".join(f"w{i}" for i in range(450)), {"act": "X"})
    chunks = chunk_documents([long, Document("S", "T", "short")], max_words=200, overlap=50)
    assert [c.id for c in chunks] == ["D#p0", "D#p1", "D#p2", "S"]
    assert {c.doc_id for c in chunks[:3]} == {"D"} and chunks[1].metadata == {"act": "X"}
    assert chunks[2].text.split()[-1] == "w449"
