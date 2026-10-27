"""Add / update / delete / duplicates across registry, Qdrant and BM25 together."""

import pytest

from investigator.ingestion import DocumentNotFound, IndexingError
from investigator.models import Status

from .conftest import parsed


def ids(hits):
    return [h.chunk.doc_id for h in hits]


def test_add_makes_document_searchable_in_every_mode(stack):
    registry, index, service = stack
    d = service.add(parsed("Larkspur outage", "The payment gateway failed after release v2.18."))
    assert registry.get(d.doc_id).status is Status.INDEXED
    for mode in ("semantic", "keyword", "hybrid"):
        assert ids(index.search("payment gateway release", "sentence", mode)) == [d.doc_id]
    assert index.vectors.count("fixed") == len(registry.chunks_for(d.doc_id, "fixed"))


def test_exact_duplicate_is_recorded_but_not_indexed_twice(stack):
    registry, index, service = stack
    a = service.add(parsed("A", "Same   words here."))
    b = service.add(parsed("B", "Same words here."))  # differs only in whitespace
    assert b.status is Status.DUPLICATE and b.duplicate_of == a.doc_id
    assert registry.chunks_for(b.doc_id) == []
    assert ids(index.search("same words", "sentence")) == [a.doc_id]


def test_update_bumps_version_and_removes_stale_chunks_everywhere(stack):
    registry, index, service = stack
    d = service.add(parsed("Doc", "Old content about kafka."))
    index.bm25("sentence")  # build BM25 so the in-memory path is exercised too
    new = service.update(d.doc_id, parsed("Doc", "New content about postgres."))
    assert new.version == 2
    assert all(c.version == 2 for c in registry.chunks_for(d.doc_id))
    assert index.search("kafka", "sentence", "keyword") == []
    assert ids(index.search("postgres", "sentence", "keyword")) == [d.doc_id]
    assert all(":v2:" in h.chunk.chunk_id for h in index.search("content", "fixed", "semantic"))
    assert service.update(d.doc_id, parsed("Doc", "New content about postgres.")).version == 2  # no-op


def test_delete_removes_from_all_stores_and_promotes_duplicate(stack):
    registry, index, service = stack
    a = service.add(parsed("A", "Shared evidence sentence."))
    b = service.add(parsed("B", "Shared evidence sentence."))
    index.bm25("sentence")
    service.delete(a.doc_id)
    assert registry.get(a.doc_id) is None
    assert registry.get(b.doc_id).status is Status.INDEXED  # promoted
    assert ids(index.search("shared evidence", "sentence", "hybrid")) == [b.doc_id]
    with pytest.raises(DocumentNotFound):
        service.delete(a.doc_id)


def test_failed_indexing_leaves_nothing_searchable_and_can_be_retried(stack, monkeypatch):
    registry, index, service = stack
    real_upsert = index.vectors.upsert

    def broken(*args, **kwargs):
        raise ConnectionError("qdrant unavailable")

    monkeypatch.setattr(index.vectors, "upsert", broken)
    with pytest.raises(IndexingError):
        service.add(parsed("X", "Unlucky document text."), doc_id="x1")
    assert registry.get("x1").status is Status.FAILED
    assert registry.chunks_for("x1") == [] and index.search("unlucky", "sentence") == []

    monkeypatch.setattr(index.vectors, "upsert", real_upsert)
    assert service.retry_failed() == 1
    assert registry.get("x1").status is Status.INDEXED
    assert ids(index.search("unlucky", "sentence")) == ["x1"]


def test_bm25_rebuilds_from_registry_after_restart(stack):
    registry, index, service = stack
    d = service.add(parsed("Persist", "Durable keyword zebra."))
    from investigator.retrieval import SearchIndex

    fresh = SearchIndex(registry, index.vectors, index.embedder)  # simulates a new process
    assert ids(fresh.search("zebra", "sentence", "keyword")) == [d.doc_id]
