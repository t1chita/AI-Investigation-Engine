"""End-to-end through the HTTP API: upload -> search -> investigate -> inspect -> update -> delete."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from investigator import api
from investigator.llm import ScriptedLLM


@pytest.fixture
def client(stack):
    registry, index, service = stack
    components = SimpleNamespace(
        settings=SimpleNamespace(chunk_strategy="sentence"),
        registry=registry, index=index, service=service, vectors=index.vectors,
    )
    llm = ScriptedLLM([])
    api.STATE.clear()
    api.init(components, llm=llm)
    with TestClient(api.app) as c:
        yield c, llm
    api.STATE.clear()


def upload(c, name, content):
    return c.post("/documents", files={"file": (name, content)})


def test_full_document_workflow(client):
    c, llm = client
    r = upload(c, "outage.txt", b"Larkspur outage\n\nRelease v2.18 cut the pool to 20. Payments failed.")
    assert r.status_code == 201
    doc_id = r.json()["documents"][0]["doc_id"]

    hits = c.post("/search", json={"query": "pool v2.18", "mode": "hybrid"}).json()["hits"]
    assert hits and hits[0]["doc_id"] == doc_id and hits[0]["keyword_rank"] == 1

    llm.responses += [
        {"queries": []},
        {"action": "answer", "status": "answered", "answer": "v2.18",
         "claims": [{"text": "v2.18 cut the pool", "evidence": ["E1"]}]},
    ]
    result = c.post("/investigate", json={"question": "Which release cut the pool?"}).json()
    assert result["status"] == "answered" and result["evidence"][0]["doc_id"] == doc_id
    assert result["input_tokens"] > 0 and result["output_tokens"] == 100  # 2 scripted calls x 50

    stats = c.get("/analytics").json()
    assert stats["totals"]["questions"] == 1 and stats["totals"]["calls"] == 2
    assert stats["by_model"][0]["model"] == "scripted" and stats["prices"]

    detail = c.get(f"/documents/{doc_id}").json()
    assert detail["version"] == 1 and detail["text_sentences"][1]["text"].startswith("Release v2.18")

    r = c.put(f"/documents/{doc_id}", files={"file": ("outage.txt", b"Larkspur outage\n\nRelease v2.19 fixed it.")})
    assert r.json()["version"] == 2
    assert c.post("/search", json={"query": "v2.18", "mode": "keyword"}).json()["hits"] == []

    assert c.delete(f"/documents/{doc_id}").status_code == 204
    assert c.get(f"/documents/{doc_id}").status_code == 404
    assert c.delete(f"/documents/{doc_id}").status_code == 404


def test_upload_errors_and_duplicates(client):
    c, _ = client
    assert upload(c, "x.exe", b"MZ").status_code == 400
    assert "invalid JSON" in upload(c, "x.json", b"{oops").json()["detail"]
    docs = [{"title": "Alpha report", "text": "Same text."}, {"title": "Bravo report", "text": "Same text."}]
    out = upload(c, "pair.json", json.dumps(docs).encode()).json()["documents"]
    assert [d["status"] for d in out] == ["indexed", "duplicate"]
    assert c.get("/documents", params={"status": "duplicate"}).json()["total"] == 1
    assert c.get("/documents", params={"q": "Bravo"}).json()["total"] == 1  # total respects the filter
    assert c.get("/documents", params={"q": "report"}).json()["total"] == 2


def test_validation_and_health(client):
    c, _ = client
    assert c.post("/search", json={"query": ""}).status_code == 422
    assert c.post("/search", json={"query": "x", "strategy": "nope"}).status_code == 400
    health = c.get("/health").json()
    assert health["status"] == "ok" and health["llm"] == "scripted:scripted"
