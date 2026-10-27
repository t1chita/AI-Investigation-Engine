"""Hand-checkable tests for our own algorithms: BM25 and Reciprocal Rank Fusion."""

import math

import pytest

from investigator.bm25 import BM25Index
from investigator.fusion import reciprocal_rank_fusion


def test_rrf_matches_hand_calculation():
    result = reciprocal_rank_fusion([["A", "B", "C"], ["B", "D"]], k=60)
    assert [item for item, _ in result] == ["B", "A", "D", "C"]
    scores = dict(result)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["A"] == pytest.approx(1 / 61)
    assert scores["D"] == pytest.approx(1 / 62)
    assert scores["C"] == pytest.approx(1 / 63)


def test_rrf_ties_break_by_best_rank_then_id():
    # X is 1st in list 1 and Y is 1st in list 2: equal scores, equal best rank -> id order.
    assert [i for i, _ in reciprocal_rank_fusion([["X"], ["Y"]])] == ["X", "Y"]
    # P (ranks 1 and 3) vs Q (rank 2 twice): 1/61 + 1/63 = 0.032266 > 2/62 = 0.032258 -> P first.
    assert [i for i, _ in reciprocal_rank_fusion([["P", "Q"], ["Z", "Q", "P"]])][:2] == ["P", "Q"]


def test_rrf_counts_duplicates_in_one_list_once_and_supports_weights():
    result = dict(reciprocal_rank_fusion([["A", "A", "B"]], k=1))
    assert result["A"] == pytest.approx(1 / 2)
    assert result["B"] == pytest.approx(1 / 4)  # B keeps its real position, 3
    weighted = dict(reciprocal_rank_fusion([["A"], ["B"]], k=1, weights=[2.0, 1.0]))
    assert weighted["A"] == pytest.approx(1.0) and weighted["B"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A"]], weights=[1.0, 2.0])


def test_bm25_matches_formula_by_hand():
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add("d1", "payment outage payment")  # tokens: payment, outage, payment (len 3)
    idx.add("d2", "database outage")  # len 2
    idx.add("d3", "cache warmup")  # len 2
    # Query "payment": df = 1, N = 3, avgdl = 7/3
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    norm = 1.5 * (1 - 0.75 + 0.75 * 3 / (7 / 3))
    expected = idf * 2 * 2.5 / (2 + norm)
    (top, score), = idx.search("payment", k=1)
    assert top == "d1"
    assert score == pytest.approx(expected)


def test_bm25_rare_terms_beat_common_terms_and_ignores_stop_words():
    idx = BM25Index()
    idx.add("a", "the outage hit payments")
    idx.add("b", "the outage hit search")
    idx.add("c", "the outage hit login")
    ranked = idx.search("outage payments", k=3)
    assert ranked[0][0] == "a"
    assert idx.search("the", k=3) == []  # stop word only


def test_compact_bm25_gives_identical_results_through_adds_and_removes():
    import random

    from investigator.bm25 import CompactBM25Index

    rng = random.Random(7)
    words = [f"w{i}" for i in range(60)] + ["payment", "outage", "v2.18", "pool"]
    corpus = {f"c{i:03d}": " ".join(rng.choices(words, k=rng.randint(3, 40))) for i in range(300)}
    plain = BM25Index()
    for cid, text in corpus.items():
        plain.add(cid, text)
    compact = CompactBM25Index.build(corpus.items())
    queries = ["payment outage", "v2.18 pool pool", "w1 w2 w3 w59", "nothing-matches", "w7"]

    def same():
        assert len(plain) == len(compact) and plain.total_len == compact.total_len
        for q in queries:
            a, b = plain.search(q, k=15), compact.search(q, k=15)
            assert [i for i, _ in a] == [i for i, _ in b]
            assert [s for _, s in a] == pytest.approx([s for _, s in b], rel=1e-12)

    same()
    for cid in ["c000", "c150", "c299", "c150", "missing"]:  # removes, a repeat and an unknown id
        plain.remove(cid)
        compact.remove(cid)
    plain.add("new1", "payment outage after v2.18 pool change")
    compact.add("new1", "payment outage after v2.18 pool change")
    plain.add("c010", "pool pool pool")  # replace a bulk-built chunk
    compact.add("c010", "pool pool pool")
    same()
    assert "c000" not in compact and "new1" in compact and "c010" in compact
    assert CompactBM25Index.build([]).search("anything") == []


def test_bm25_remove_and_replace_keep_statistics_consistent():
    idx = BM25Index()
    idx.add("a", "alpha beta")
    idx.add("b", "beta gamma")
    idx.remove("a")
    idx.remove("a")  # repeat is harmless
    assert "a" not in idx and len(idx) == 1 and idx.total_len == 2
    assert "alpha" not in idx.postings
    assert idx.search("alpha") == []
    idx.add("b", "delta")  # replace
    assert idx.search("gamma") == [] and idx.search("delta")[0][0] == "b"
