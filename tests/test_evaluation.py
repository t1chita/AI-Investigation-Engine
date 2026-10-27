"""Evaluation runner: metrics and the stop-and-resume behaviour on a used-up daily LLM quota."""

from investigator import evaluation
from investigator.evaluation import EvalQuestion
from investigator.investigate import Investigator
from investigator.llm import LLMQuotaExhausted, ScriptedLLM

from .conftest import parsed


def answer(text, eid="E1"):
    return {"action": "answer", "status": "answered", "answer": text, "claims": [{"text": text, "evidence": [eid]}]}


def test_metrics_on_small_examples():
    assert evaluation.exact_match("The Yoruba", ["Yoruba"]) == 1.0
    assert evaluation.token_f1("Dana Whitfield approved it", ["Dana Whitfield"]) > 0.5
    assert evaluation.recall_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.5
    assert evaluation.reciprocal_rank(["x", "y", "a"], ["a"]) == 1 / 3
    assert evaluation.sentence_coverage([("d", 0, 2)], [("d", 1), ("d", 5)]) == 0.5


def test_answer_run_stops_on_daily_quota_and_resumes(stack, tmp_path):
    registry, index, service = stack
    service.add(parsed("Release", "Release v2.18 lowered the pool size to 20."), doc_id="rel")
    qs = [EvalQuestion(f"q{i}", "simple", "What did release v2.18 lower?", "answered", ["pool size"], ["rel"])
          for i in range(3)]
    out = tmp_path / "answers.json"

    llm = ScriptedLLM([{"queries": []}, answer("the pool size"), {"queries": []},
                       LLMQuotaExhausted("groq: daily quota used up")])
    first = evaluation.evaluate_answers(Investigator(index, llm), qs, "cfg", save_to=out)
    assert first["complete"] is False and [r["qid"] for r in first["rows"]] == ["q0"]

    llm2 = ScriptedLLM([{"queries": []}, answer("the pool size"), {"queries": []}, answer("the pool size")])
    second = evaluation.evaluate_answers(Investigator(index, llm2), qs, "cfg", done_rows=first["rows"], save_to=out)
    assert second["complete"] is True and [r["qid"] for r in second["rows"]] == ["q0", "q1", "q2"]
    assert second["summary"]["answer_correctness"] == 1.0 and llm2.usage.calls == 4  # q0 was not re-run
