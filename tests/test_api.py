from fastapi.testclient import TestClient

from legalsearch import api


def test_search_endpoint(searcher, monkeypatch):
    monkeypatch.setattr(api, "searcher", lambda: searcher)
    c = TestClient(api.app)
    body = c.get("/search", params={"q": "guide dog rented flat", "k": 3}).json()
    assert body["hits"][0]["id"] == "RT-12" and "total" in body["timings_ms"]
    body = c.get("/search", params={"q": "notice", "act": ["LC", "RT"], "year_min": 2018}).json()
    assert {h["metadata"]["act_code"] for h in body["hits"]} == {"LC"}
    assert c.get("/search", params={"q": "x", "mode": "magic"}).status_code == 400
    assert c.get("/health").json()["passages"] == 61
