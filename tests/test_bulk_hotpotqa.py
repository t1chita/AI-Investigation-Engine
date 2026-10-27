from types import SimpleNamespace

from investigator import bulk, hotpotqa
from investigator.models import Status


def row(qid, titles_and_sentences, facts):
    return {
        "id": qid,
        "question": "q?",
        "answer": "a",
        "type": "bridge",
        "context": {"title": [t for t, _ in titles_and_sentences], "sentences": [s for _, s in titles_and_sentences]},
        "supporting_facts": {"title": [t for t, _ in facts], "sent_id": [i for _, i in facts]},
    }


ROWS = [
    row("q1", [("Alpha", ["Alpha is a river. ", "It is long."]), ("Beta", ["Beta is a town."])], [("Alpha", 1)]),
    # "Beta" repeats across questions; "Gamma" has the same text as "Beta" (a real duplicate).
    row("q2", [("Beta", ["Beta is a town."]), ("Gamma", ["Beta is a town."])], [("Beta", 0), ("Beta", 9)]),
]


def test_corpus_keeps_each_passage_once_with_stable_ids():
    docs = list(hotpotqa.iter_corpus(ROWS))
    assert [d.title for _, d in docs] == ["Alpha", "Beta", "Gamma"]
    assert docs[0][0] == hotpotqa.doc_id_for("Alpha") and docs[0][0].startswith("hp-")
    assert [s.text for s in docs[0][1].sentences] == ["Alpha is a river.", "It is long."]


def test_gold_question_maps_facts_and_flags_invalid_labels():
    g1, g2 = (hotpotqa.gold_question(r) for r in ROWS)
    assert g1.valid and g1.supporting == [(hotpotqa.doc_id_for("Alpha"), 1)]
    assert not g2.valid and g2.supporting_docs == [hotpotqa.doc_id_for("Beta")]


def test_bulk_ingest_dedups_resumes_and_reconciles(stack):
    registry, index, service = stack
    components = SimpleNamespace(registry=registry, index=index, service=service, vectors=index.vectors)
    report = bulk.ingest(components, hotpotqa.iter_corpus(ROWS), batch_size=2, progress=lambda *_: None)
    assert (report.indexed, report.duplicates) == (2, 1)
    gamma = registry.get(hotpotqa.doc_id_for("Gamma"))
    assert gamma.status is Status.DUPLICATE and gamma.duplicate_of == hotpotqa.doc_id_for("Beta")

    # Simulate a crash that left one document half-indexed, then re-run.
    registry.set_status(hotpotqa.doc_id_for("Alpha"), Status.INDEXING)
    again = bulk.ingest(components, hotpotqa.iter_corpus(ROWS), batch_size=2, progress=lambda *_: None)
    assert (again.retried, again.indexed, again.skipped_done) == (1, 1, 2)

    r = bulk.reconcile(components)
    assert r["documents_total"] == 3 and r["documents_indexing"] == 0
    for name in service.chunkers:
        assert r[f"chunks_{name}_registry"] == r[f"chunks_{name}_qdrant"] > 0
