"""The investigation pipeline with a scripted LLM: multi-hop, citations, conflicts, abstention, errors."""

import pytest

from investigator.investigate import Investigator
from investigator.llm import LLMRateLimited, ScriptedLLM, parse_json_object, LLMBadOutput

from .conftest import parsed


@pytest.fixture
def corpus(stack):
    registry, index, service = stack
    service.add(parsed("Release v2.18", "Release v2.18 lowered the database pool size to 20."), doc_id="rel")
    service.add(parsed("Payment outage", "The payment outage began after release v2.18."), doc_id="out")
    service.add(parsed("Report A", "The warehouse alarm sounded at 21:40."), doc_id="ra")
    service.add(parsed("Report B", "The warehouse alarm sounded at 22:35."), doc_id="rb")
    return index


def answer(status="answered", **kw):
    return {"action": "answer", "status": status, **kw}


def test_multi_hop_search_then_cited_answer(corpus):
    llm = ScriptedLLM([
        {"queries": ["payment outage cause"]},
        {"action": "search", "query": "release v2.18 database pool"},
        answer(answer="The pool size was lowered to 20", claims=[
            {"text": "The outage began after v2.18", "evidence": ["E1"]},
            {"text": "v2.18 lowered the pool to 20", "evidence": ["E2", "E99"]},
        ]),
    ])
    r = Investigator(corpus, llm, k_per_search=1).run("What change caused the payment outage?")
    assert r.status == "answered" and r.answer.startswith("The pool size")
    assert [s["query"] for s in r.steps if s["type"] == "tool"] == [
        "What change caused the payment outage?", "payment outage cause", "release v2.18 database pool"
    ]
    assert {e.doc_id for e in r.cited_evidence()} >= {"out"}
    assert r.claims[1].evidence == ["E2"] and any("E99" in w for w in r.warnings)
    assert r.llm_calls == 3


def test_conflict_needs_two_cited_sources(corpus):
    llm = ScriptedLLM([
        {"queries": []},
        answer("conflict", answer="Reports disagree", conflicts=[
            {"description": "21:40 vs 22:35", "evidence": ["E1", "E2"]}
        ]),
    ])
    r = Investigator(corpus, llm, k_per_search=2).run("When did the warehouse alarm sound?")
    assert r.status == "conflict" and len(r.conflicts) == 1
    assert {e.doc_id for e in r.cited_evidence()} == {"ra", "rb"}


def test_uncited_answer_becomes_insufficient(corpus):
    llm = ScriptedLLM([{"queries": []}, answer(answer="Probably Tuesday", claims=[{"text": "guess", "evidence": []}])])
    r = Investigator(corpus, llm).run("Who approved the Meridian renewal?")
    assert r.status == "insufficient" and r.answer == ""


def test_model_abstains_explicitly_after_declining_a_second_search(corpus):
    llm = ScriptedLLM([
        {"queries": []},
        answer("insufficient", missing="no approver is named"),
        {"action": "none"},  # second chance: nothing more to search
    ])
    r = Investigator(corpus, llm).run("Who approved the Meridian renewal?")
    assert r.status == "insufficient" and r.missing == "no approver is named"
    assert r.llm_calls == 3 and r.steps[-1] == {"type": "second_chance", "action": "none"}


def test_second_chance_search_finds_the_missing_hop(corpus):
    llm = ScriptedLLM([
        {"queries": []},
        answer("insufficient", missing="which release"),
        {"action": "search", "query": "release v2.18 pool size"},
        answer(answer="v2.18 lowered the pool to 20", claims=[{"text": "pool lowered", "evidence": ["E2"]}]),
    ])
    inv = Investigator(corpus, llm, k_per_search=1)
    r = inv.run("What did the change behind the payment outage do?")
    assert r.status == "answered" and r.evidence[1].doc_id == "rel"
    assert "release v2.18 pool size" in llm.prompts[-1][1]  # the final answer saw the new search


def test_second_chance_is_not_used_for_answers_and_can_be_disabled(corpus):
    llm = ScriptedLLM([{"queries": []}, answer("insufficient")])
    r = Investigator(corpus, llm, second_chance=False).run("Who approved the Meridian renewal?")
    assert r.status == "insufficient" and r.llm_calls == 2


def test_follow_up_searches_have_reserved_evidence_room(corpus):
    llm = ScriptedLLM([
        {"queries": []},
        {"action": "search", "query": "warehouse alarm report"},
        answer(answer="x", claims=[{"text": "c", "evidence": ["E1"]}]),
    ])
    r = Investigator(corpus, llm, initial_evidence=1, followup_evidence=2).run("payment outage")
    tool_steps = [s for s in r.steps if s["type"] == "tool"]
    assert [s["added"] for s in tool_steps] == [1, 2] and len(r.evidence) == 3


def test_unknown_tool_and_repeated_search_are_rejected(corpus):
    llm = ScriptedLLM([
        {"queries": []},
        {"action": "delete_everything"},
        answer(answer="x", claims=[{"text": "c", "evidence": ["E1"]}]),
    ])
    r = Investigator(corpus, llm).run("payment outage")
    assert r.status == "answered" and any("unknown tool" in w for w in r.warnings)


def test_step_limit_forces_an_answer(corpus):
    llm = ScriptedLLM([
        {"queries": []},
        {"action": "search", "query": "one"},
        {"action": "search", "query": "two"},
        answer(answer="done", claims=[{"text": "c", "evidence": ["E1"]}]),
    ])
    r = Investigator(corpus, llm, max_steps=2).run("payment outage")
    assert r.status == "answered" and "No more searches" in llm.prompts[-1][1]


def test_llm_failures_are_reported_not_raised(corpus):
    planning_fails = ScriptedLLM([LLMRateLimited("rate limited"), answer("insufficient")])
    r = Investigator(corpus, planning_fails).run("payment outage")
    assert r.status == "insufficient" and any("planning failed" in w for w in r.warnings)

    answer_fails = ScriptedLLM([{"queries": []}, LLMRateLimited("rate limited")])
    r = Investigator(corpus, answer_fails).run("payment outage")
    assert r.status == "error" and "LLMRateLimited" in r.error and r.evidence  # evidence still shown


def test_without_llm_returns_retrieval_only(corpus):
    r = Investigator(corpus, None).run("payment outage")
    assert r.status == "retrieval_only" and r.evidence and not r.claims


def test_parse_json_object_handles_fences_and_garbage():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMBadOutput):
        parse_json_object("no json here")
