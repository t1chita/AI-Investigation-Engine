"""Automated evaluation: question bank, metrics and runners.

Question bank (46 questions, five categories):
- simple (10)         synthetic single-document facts
- multi_step (10)     8 HotpotQA "bridge" questions + 2 synthetic chains
- multi_document (10) 8 HotpotQA "comparison" questions + 2 synthetic
- unanswerable (8)    synthetic: the corpus talks about the topic but never states the answer
- contradictory (8)   synthetic: two sources state different values for the same fact
HotpotQA questions are picked deterministically: valid labels only, sorted by question id, first N
of each type. Synthetic documents and questions are in ``eval/`` and labelled synthetic.

Metrics (all relevance is judged at DOCUMENT level, so both chunking strategies are scored on
the same footing; chunk hits are collapsed to their documents in rank order):
- Recall@K   share of gold documents that appear in the top K retrieved documents
- MRR        mean over questions of 1 / rank of the first gold document (0 if none in top K)
- SF-coverage@K (HotpotQA only) share of gold supporting SENTENCES inside the top-K chunks'
             sentence spans; this is where chunk boundaries matter
- Answer EM / F1  SQuAD-style normalised exact match and token F1 against the gold answers
- Answer correctness  answerable: status "answered" and a gold answer contained in the answer
             (or F1 >= 0.5); unanswerable: status "insufficient"; contradictory: status
             "conflict" citing at least two gold documents
- Citation precision  share of cited documents that are gold documents
- Citation recall     share of gold documents that were cited
"""

import json
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import hotpotqa
from .models import ParsedDocument, Sentence
from .text import split_sentences

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
ANSWERABLE = ("simple", "multi_step", "multi_document")


@dataclass
class EvalQuestion:
    qid: str
    category: str
    question: str
    expected_status: str
    answers: list[str]
    gold_docs: list[str]
    gold_sentences: list[tuple[str, int]] = field(default_factory=list)
    source: str = "synthetic"


# ---- question bank ----------------------------------------------------------------------------
def synthetic_documents() -> list[tuple[str, ParsedDocument]]:
    items = json.loads((EVAL_DIR / "synthetic_corpus.json").read_text())
    return [
        (
            item["doc_id"],
            ParsedDocument(
                title=item["title"],
                sentences=[Sentence(i, s) for i, s in enumerate(split_sentences(item["text"]))],
                format="json",
                metadata=item.get("metadata", {}),
            ),
        )
        for item in items
    ]


def ensure_synthetic(service) -> tuple[int, int]:
    """Add missing synthetic documents and update changed ones, through the normal document service.
    Returns (added, updated)."""
    added = updated = 0
    for doc_id, parsed in synthetic_documents():
        existing = service.registry.get(doc_id)
        if existing is None:
            service.add(parsed, source_type="synthetic", doc_id=doc_id)
            added += 1
        elif existing.content_hash != parsed.content_hash or existing.title != parsed.title:
            service.update(doc_id, parsed)
            updated += 1
    return added, updated


def hotpot_questions(rows, n_bridge: int, n_comparison: int) -> list[EvalQuestion]:
    gold = sorted((hotpotqa.gold_question(r) for r in rows), key=lambda g: g.qid)
    picked = {"bridge": [], "comparison": []}
    limits = {"bridge": n_bridge, "comparison": n_comparison}
    for g in gold:
        if g.valid and len(picked[g.qtype]) < limits[g.qtype]:
            picked[g.qtype].append(g)
    category = {"bridge": "multi_step", "comparison": "multi_document"}
    return [
        EvalQuestion(g.qid, category[g.qtype], g.question, "answered", [g.answer], g.supporting_docs,
                     [tuple(p) for p in g.supporting], "hotpotqa")
        for kind in ("bridge", "comparison")
        for g in picked[kind]
    ]


def question_bank(rows) -> list[EvalQuestion]:
    synthetic = [EvalQuestion(**q) for q in json.loads((EVAL_DIR / "questions_synthetic.json").read_text())]
    return synthetic + hotpot_questions(rows, 8, 8)


def retrieval_benchmark(rows, n: int = 500) -> list[EvalQuestion]:
    """A larger HotpotQA-only set for retrieval metrics (no LLM needed, so it is cheap)."""
    gold = sorted((hotpotqa.gold_question(r) for r in rows), key=lambda g: g.qid)
    out = []
    for g in gold:
        if g.valid and len(out) < n:
            category = "multi_step" if g.qtype == "bridge" else "multi_document"
            out.append(EvalQuestion(g.qid, category, g.question, "answered", [g.answer], g.supporting_docs,
                                    [tuple(p) for p in g.supporting], "hotpotqa"))
    return out


# ---- metrics ----------------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    best = 0.0
    p_tokens = normalize_answer(prediction).split()
    for g in golds:
        g_tokens = normalize_answer(g).split()
        common = sum((Counter(p_tokens) & Counter(g_tokens)).values())
        if common == 0:
            continue
        precision, recall = common / len(p_tokens), common / len(g_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_answer(prediction: str, golds: list[str]) -> bool:
    p = f" {normalize_answer(prediction)} "
    return any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g))


def doc_ranking(chunk_doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(chunk_doc_ids))


def recall_at_k(ranked_docs: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked_docs[:k]) & set(gold)) / len(gold) if gold else 0.0


def reciprocal_rank(ranked_docs: list[str], gold: list[str]) -> float:
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def sentence_coverage(spans: list[tuple[str, int, int]], gold_sentences: list[tuple[str, int]]) -> float:
    """Share of gold (doc, sentence) pairs that fall inside any retrieved (doc, start, end) span."""
    if not gold_sentences:
        return 0.0
    hit = sum(any(d == doc and s <= sent <= e for d, s, e in spans) for doc, sent in gold_sentences)
    return hit / len(gold_sentences)


def answer_correct(q: EvalQuestion, status: str, answer: str, cited_docs: list[str]) -> bool:
    if q.category == "unanswerable":
        return status == "insufficient"
    if q.category == "contradictory":
        return status == "conflict" and len(set(cited_docs) & set(q.gold_docs)) >= 2
    return status == "answered" and (contains_answer(answer, q.answers) or token_f1(answer, q.answers) >= 0.5)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# ---- runners ----------------------------------------------------------------------------------
def evaluate_retrieval(index, questions: list[EvalQuestion], strategy: str, mode: str, k: int = 10) -> dict:
    rows = []
    for q in questions:
        hits = index.search(q.question, strategy, mode, k=k)
        ranked = doc_ranking([h.chunk.doc_id for h in hits])
        spans = [(h.chunk.doc_id, h.chunk.sentence_start, h.chunk.sentence_end) for h in hits]
        rows.append({
            "qid": q.qid,
            "category": q.category,
            "recall@2": recall_at_k(ranked, q.gold_docs, 2),
            "recall@5": recall_at_k(ranked, q.gold_docs, 5),
            "recall@10": recall_at_k(ranked, q.gold_docs, 10),
            "mrr": reciprocal_rank(ranked, q.gold_docs),
            "sf_coverage@5": sentence_coverage(spans[:5], q.gold_sentences) if q.gold_sentences else None,
            "top_docs": ranked[:5],
        })
    metrics = ("recall@2", "recall@5", "recall@10", "mrr", "sf_coverage@5")
    summary = {m: mean(r[m] for r in rows if r[m] is not None) for m in metrics}
    return {"strategy": strategy, "mode": mode, "n": len(rows), "summary": summary, "rows": rows}


def evaluate_answers(investigator, questions: list[EvalQuestion], label: str, pause: float = 0.0) -> dict:
    rows = []
    for q in questions:
        start = time.time()
        r = investigator.run(q.question)
        cited_docs = list(dict.fromkeys(e.doc_id for e in r.cited_evidence()))
        retrieved_docs = doc_ranking([e.doc_id for e in r.evidence])
        rows.append({
            "qid": q.qid,
            "category": q.category,
            "question": q.question,
            "expected_status": q.expected_status,
            "status": r.status,
            "answer": r.answer,
            "gold_answers": q.answers,
            "correct": answer_correct(q, r.status, r.answer, cited_docs),
            "em": exact_match(r.answer, q.answers) if q.answers else None,
            "f1": token_f1(r.answer, q.answers) if q.answers else None,
            "evidence_recall": recall_at_k(retrieved_docs, q.gold_docs, len(retrieved_docs)),
            "citation_precision": (len(set(cited_docs) & set(q.gold_docs)) / len(cited_docs)) if cited_docs else None,
            "citation_recall": len(set(cited_docs) & set(q.gold_docs)) / len(q.gold_docs),
            "cited_docs": cited_docs,
            "gold_docs": q.gold_docs,
            "llm_calls": r.llm_calls,
            "seconds": round(time.time() - start, 2),
            "error": r.error,
            "warnings": r.warnings,
        })
        if pause:
            time.sleep(pause)  # stay under free-tier rate limits
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    summary = {
        "answer_correctness": mean(r["correct"] for r in rows),
        "em_answerable": mean(r["em"] for r in rows if r["category"] in ANSWERABLE),
        "f1_answerable": mean(r["f1"] for r in rows if r["category"] in ANSWERABLE),
        "evidence_recall": mean(r["evidence_recall"] for r in rows),
        "citation_precision": mean(r["citation_precision"] for r in rows if r["citation_precision"] is not None),
        "citation_recall_answerable": mean(r["citation_recall"] for r in rows if r["category"] in ANSWERABLE),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "by_category": {c: mean(r["correct"] for r in rs) for c, rs in by_cat.items()},
    }
    return {"config": label, "n": len(rows), "summary": summary, "rows": rows}


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
