"""Ingest the HotpotQA validation corpus (66,581 passages) into the registry, Qdrant and BM25.

    python scripts/ingest_hotpotqa.py              # everything (resumable: re-run to continue)
    python scripts/ingest_hotpotqa.py --limit 2000 # a quick partial run

Uses DATABASE_URL / QDRANT_* / EMBEDDING_MODEL / INDEX_STRATEGIES from the environment or .env.
Also writes the evaluation labels to data/hotpotqa_validation_questions.jsonl (never indexed).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from investigator import bulk, hotpotqa
from investigator.factory import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many passages")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    rows = hotpotqa.load_split("validation")
    labels = Path("data/hotpotqa_validation_questions.jsonl")
    labels.parent.mkdir(parents=True, exist_ok=True)
    with labels.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(hotpotqa.gold_question(row))) + "\n")
    print(f"wrote {len(rows)} question labels to {labels}")

    components = build()
    print(f"strategies: {list(components.service.chunkers)} | embedder: {components.index.embedder.model_name}")
    report = bulk.ingest(components, hotpotqa.iter_corpus(rows), batch_size=args.batch_size, limit=args.limit)
    print("\nREPORT", json.dumps(asdict(report), indent=2))
    print("RECONCILIATION", json.dumps(bulk.reconcile(components), indent=2))


if __name__ == "__main__":
    main()
