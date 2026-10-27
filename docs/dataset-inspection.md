# HotpotQA inspection and source-record definition

Measured on 27 October 2026 with `datasets` 5.1.0, `hotpotqa/hotpot_qa`, config `distractor`
(local cache fingerprint of the validation split: `6d38dc5e932fe842`). Commands:
`scripts/dataset-inspection.py` (train split) and the counts in this note (validation split).

## Structure

- One row = one **question**: `id`, `question`, `answer`, `type` (`bridge` / `comparison`),
  `level`, `supporting_facts`, `context`.
- `context.title[j]` and `context.sentences[j]` are parallel lists: position `j` is one passage.
- `supporting_facts` = pairs (`title`, `sent_id`), `sent_id` being a zero-based sentence position
  inside that passage.

## Measurements

| | train | validation |
| --- | ---: | ---: |
| Questions | 90,447 | 7,405 |
| Passage occurrences (with repeats) | 899,667 | 73,700 |
| Empty passages | 0 | 0 |
| Unique passage texts (SHA-256 of whitespace-normalised text) | 481,844 | 66,555 |
| Unique (title, text) passages | – | 66,581 |
| Titles with more than one text | – | 0 |
| `title`/`sentences` length mismatches | 0 in first 10 | 0 |
| Supporting facts / invalid ones | – | 18,005 / 1 |
| Question types | – | 5,918 bridge, 1,487 comparison |
| Passage length (characters) | – | median 486, p95 1,128, max 8,237 |

Findings:
- Passages repeat heavily across questions (73,700 occurrences, 66,581 distinct), so neither
  questions nor occurrences are valid counts of independent records.
- **18 texts appear under 44 different titles** (e.g. "Tara Strong" and "Tara Strong
  filmography" have identical text). That is real duplicate information: 66,581 records,
  66,555 distinct contents, 26 duplicates.
- One validation label is broken: question `5ae61bfd5542992663a4f261` cites sentence 902 of
  "Jimmy Butler (basketball)", which has 5 sentences. Such questions are flagged
  `valid = false` and excluded from evaluation.

## Decision: what one source record is

- **One record = one distinct Wikipedia passage** (title + its sentences).
- **Corpus = every distinct passage in the validation split: 66,581 records** (≥ 50,000). The
  selection is deterministic (no sampling) and reproducible from the dataset alone.
- Validation, not train, because the evaluation questions come from validation: every gold
  passage and every distractor of every evaluation question is in the corpus by construction.
- `doc_id = "hp-" + sha256(title)[:16]` (titles are unique). Sentence positions are kept, so a
  supporting fact maps to exactly one (doc_id, sentence index).
- Duplicates: the 26 passages whose text equals an earlier passage are stored with status
  `duplicate` and a pointer to the original; they are not indexed twice.
- Questions, answers and supporting facts are evaluation labels only and are never indexed.

## Chunking on this corpus (64-word budget for both strategies)

| Strategy | Chunks | Words per chunk (median / mean / max) |
| --- | ---: | --- |
| fixed (64 words, 16 overlap) | 132,424 | 64 / 51.8 / 64 |
| sentence (whole sentences, ≤ 64 words) | 130,886 | 47 / 44.4 / 319 |

The sentence chunker's maximum (319 words) is a single very long sentence that it refuses to cut;
MiniLM reads at most 256 tokens, so the end of such a chunk is not embedded (BM25 still sees it).
Passages are 78 words at the median (p95 177), so most passages become one or two chunks.
