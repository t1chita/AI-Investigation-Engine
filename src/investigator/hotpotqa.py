"""HotpotQA ("distractor" configuration) as our source corpus and evaluation question bank.

Source record definition (see docs/dataset-inspection.md for the measurements behind it):
- One record = one Wikipedia passage (a title and its sentences) from a question's ``context``.
- The corpus is EVERY distinct passage in the validation split: 73,700 passage occurrences
  collapse to 66,581 distinct (title, text) records, which is above the 50,000 requirement.
- Titles are unique within that set, so ``doc_id = "hp-" + sha256(title)[:16]`` is stable.
- Sentences keep their original zero-based positions, so a supporting fact
  (title, sent_id) maps to exactly one (doc_id, sentence index).

Why validation and not train: our evaluation questions come from the validation split, so all of
their gold passages AND their distractor passages are in the corpus by construction, and nothing
is sampled at random. Questions, answers and supporting facts are kept OUT of the corpus; they are
only used as evaluation labels.
"""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

from .models import ParsedDocument, Sentence
from .text import normalize_text

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"


def doc_id_for(title: str) -> str:
    return "hp-" + hashlib.sha256(normalize_text(title).encode("utf-8")).hexdigest()[:16]


def load_split(split: str = "validation"):
    from datasets import load_dataset

    return load_dataset(DATASET, CONFIG)[split]


def iter_corpus(rows) -> Iterator[tuple[str, ParsedDocument]]:
    """Yield (doc_id, document) for each distinct passage, in first-seen order (deterministic)."""
    seen: set[str] = set()
    for row in rows:
        context = row["context"]
        for title, sentences in zip(context["title"], context["sentences"]):
            doc_id = doc_id_for(title)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            # Keep every original position (even if a sentence is blank) so sent_id stays valid.
            kept = [Sentence(i, normalize_text(s)) for i, s in enumerate(sentences)]
            yield doc_id, ParsedDocument(
                title=normalize_text(title),
                sentences=[s for s in kept if s.text],
                format="hotpotqa",
                metadata={"dataset": DATASET, "config": CONFIG, "first_question": row["id"]},
            )


@dataclass
class GoldQuestion:
    qid: str
    question: str
    answer: str
    qtype: str  # HotpotQA type: "bridge" or "comparison"
    supporting: list[tuple[str, int]]  # (doc_id, sentence index) pairs
    supporting_docs: list[str]  # distinct doc_ids, in label order
    valid: bool  # False when a supporting fact points outside its passage (label noise)


def gold_question(row) -> GoldQuestion:
    lengths = {doc_id_for(t): len(s) for t, s in zip(row["context"]["title"], row["context"]["sentences"])}
    pairs, valid = [], True
    for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        doc_id = doc_id_for(title)
        if doc_id not in lengths or not 0 <= sent_id < lengths[doc_id]:
            valid = False
            continue
        pairs.append((doc_id, sent_id))
    docs = list(dict.fromkeys(d for d, _ in pairs))
    return GoldQuestion(row["id"], row["question"], row["answer"], row["type"], pairs, docs, valid)
