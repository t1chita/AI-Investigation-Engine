"""Run the evaluation and write results to eval/results/.

    python scripts/evaluate.py retrieval            # 46-question bank + 500 HotpotQA questions,
                                                    # 2 chunkers x 3 retrieval modes, no LLM
    python scripts/evaluate.py answers              # 46-question bank end to end with the LLM,
                                                    # comparing two configurations
    python scripts/evaluate.py answers --limit 5    # quick smoke run

The synthetic evaluation documents (eval/synthetic_corpus.json) are added to the corpus first if
they are missing, through the normal document service.
"""

import argparse
import json
from pathlib import Path

from investigator import evaluation, hotpotqa
from investigator.factory import build
from investigator.investigate import Investigator
from investigator.llm import make_llm

OUT = Path("eval/results")

# The two end-to-end configurations compared (same LLM, same questions).
CONFIGS = {
    "sentence-hybrid": {"strategy": "sentence", "mode": "hybrid"},
    "fixed-semantic": {"strategy": "fixed", "mode": "semantic"},
}


def run_retrieval(components, rows) -> None:
    bank = evaluation.question_bank(rows)
    bench = evaluation.retrieval_benchmark(rows, 500)
    table = []
    for name, questions in (("bank46", bank), ("hotpot500", bench)):
        for strategy in components.service.chunkers:
            for mode in ("semantic", "keyword", "hybrid"):
                result = evaluation.evaluate_retrieval(components.index, questions, strategy, mode)
                evaluation.save(result, OUT / f"retrieval_{name}_{strategy}_{mode}.json")
                table.append({"set": name, "strategy": strategy, "mode": mode, **result["summary"]})
                s = result["summary"]
                print(f"{name:9} {strategy:8} {mode:8} R@2 {s['recall@2']:.3f} R@5 {s['recall@5']:.3f} "
                      f"R@10 {s['recall@10']:.3f} MRR {s['mrr']:.3f} SF@5 {s['sf_coverage@5']:.3f}")
    (OUT / "retrieval_summary.json").write_text(json.dumps(table, indent=2))


def run_answers(components, rows, limit, pause) -> None:
    s = components.settings
    llm = make_llm(s.llm_provider, s.llm_api_key, s.llm_model)
    if llm is None:
        raise SystemExit("answers evaluation needs an LLM: set LLM_PROVIDER and LLM_API_KEY in .env")
    bank = evaluation.question_bank(rows)[:limit]
    summaries = {}
    for label, cfg in CONFIGS.items():
        investigator = Investigator(components.index, llm, strategy=cfg["strategy"], mode=cfg["mode"])
        result = evaluation.evaluate_answers(investigator, bank, label, pause=pause)
        result["llm"] = f"{llm.provider}:{llm.model}"
        evaluation.save(result, OUT / f"answers_{label}.json")
        summaries[label] = result["summary"]
        print(label, json.dumps(result["summary"], indent=2))
    summaries["usage"] = vars(llm.usage)
    (OUT / "answers_summary.json").write_text(json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["retrieval", "answers"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between questions (rate limits)")
    args = parser.parse_args()

    components = build()
    added, updated = evaluation.ensure_synthetic(components.service)
    if added or updated:
        print(f"synthetic evaluation documents: {added} added, {updated} updated")
    rows = hotpotqa.load_split("validation")
    if args.what == "retrieval":
        run_retrieval(components, rows)
    else:
        run_answers(components, rows, args.limit, args.pause)


if __name__ == "__main__":
    main()
