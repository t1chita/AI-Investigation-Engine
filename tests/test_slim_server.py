"""The pieces that let the server fit 512 MB: REST Qdrant client, child-process BM25 build, web UI."""

import json

import httpx

from investigator.models import Chunk
from investigator.retrieval import SearchIndex
from investigator.vector_store import QdrantRESTStore, point_id

from .conftest import parsed


def fake_qdrant():
    """A tiny in-memory stand-in for Qdrant's HTTP API: enough to check our requests."""
    points: dict[str, dict] = {}
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        calls.append((request.method, request.url.path, body))
        path = request.url.path
        if path.endswith("/exists"):
            return httpx.Response(200, json={"result": {"exists": bool(calls and len(calls) > 1)}})
        if request.method == "PUT" and path.endswith("/points"):
            for p in body["points"]:
                points[p["id"]] = p
            return httpx.Response(200, json={"result": {}})
        if path.endswith("/points/delete"):
            doc = body["filter"]["must"][0]["match"]["value"]
            for pid in [k for k, p in points.items() if p["payload"]["doc_id"] == doc]:
                del points[pid]
            return httpx.Response(200, json={"result": {}})
        if path.endswith("/points/query"):
            q = body["query"]
            scored = sorted(points.values(), key=lambda p: -sum(a * b for a, b in zip(q, p["vector"])))
            return httpx.Response(200, json={"result": {"points": [
                {"payload": {"chunk_id": p["payload"]["chunk_id"]}, "score": sum(a * b for a, b in zip(q, p["vector"]))}
                for p in scored[: body["limit"]]
            ]}})
        if path.endswith("/points/count"):
            return httpx.Response(200, json={"result": {"count": len(points)}})
        return httpx.Response(200, json={"result": {}})

    client = httpx.Client(base_url="https://qdrant.test", transport=httpx.MockTransport(handler))
    return QdrantRESTStore("https://qdrant.test", "key", dim=2, http=client), points, calls


def chunk(cid, doc):
    return Chunk(cid, doc, 1, "sentence", 0, "t", "x", 0, 0)


def test_rest_store_upsert_search_delete_count():
    store, points, calls = fake_qdrant()
    store.upsert("sentence", [chunk("a:v1:sentence:0", "a"), chunk("b:v1:sentence:0", "b")], [[1, 0], [0, 1]])
    assert ("PUT", "/collections/chunks_sentence", {"vectors": {"size": 2, "distance": "Cosine"}}) in calls
    assert point_id("a:v1:sentence:0") in points
    assert store.search("sentence", [1, 0], k=1) == [("a:v1:sentence:0", 1.0)]
    store.delete_doc("sentence", "a")
    assert store.count("sentence") == 1


def test_rest_store_raises_readable_errors():
    client = httpx.Client(base_url="https://q", transport=httpx.MockTransport(lambda r: httpx.Response(403, text="forbidden")))
    store = QdrantRESTStore("https://q", "bad", dim=2, http=client)
    try:
        store.count("sentence")
    except RuntimeError as exc:
        assert "403" in str(exc)
    else:
        raise AssertionError("expected an error")


def test_bm25_built_in_child_process_matches_in_process(stack):
    registry, index, service = stack
    service.add(parsed("Outage", "Release v2.18 lowered the pool size."), doc_id="o")
    service.add(parsed("Fire", "The alarm sounded at 21:40."), doc_id="f")
    child = SearchIndex(registry, index.vectors, index.embedder)
    child.warm_bm25("sentence", in_subprocess=True)
    assert child.keyword_ids("pool v2.18", "sentence", 5) == index.keyword_ids("pool v2.18", "sentence", 5)
    assert child.keyword_ids("pool v2.18", "sentence", 5)[0][0].startswith("o:")


def test_web_ui_is_served(stack):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from investigator import api

    registry, index, service = stack
    api.STATE.clear()
    api.init(SimpleNamespace(settings=SimpleNamespace(chunk_strategy="sentence"), registry=registry,
                             index=index, service=service, vectors=index.vectors), llm=None)
    with TestClient(api.app) as c:
        page = c.get("/")
        assert page.status_code == 200 and "AI Investigation Engine" in page.text and "/investigate" in page.text
    api.STATE.clear()
