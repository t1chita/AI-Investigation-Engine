# AI Investigation Engine

An evidence-driven investigation system. It searches a collection of 66,000+ documents, connects
facts across several sources, and answers questions with citations to exact sentences. When the
evidence is missing it says so, and when sources disagree it reports the conflict instead of
choosing one.

**Live demo: https://ai-investigation-engine-3faf.onrender.com** (free tier: if nobody used it in
the last 15 minutes, the first page load takes about 2 minutes while the server starts).

> **Status (28 October 2026).** Implemented, tested and deployed: ingestion of 3 formats with
> add/update/delete and deduplication, two chunkers, Qdrant + our BM25 + our RRF hybrid search,
> the multi-step RAG pipeline with tool calls, citations, conflict reporting and abstention, the
> FastAPI backend with its web page, the 66,581-record corpus in Neon + Qdrant Cloud, both
> evaluations (retrieval on 500 questions, end-to-end answers on 46 questions), and a verified
> public deployment on Render with data surviving container replacement.

Submission documents: [evaluation report](docs/evaluation-report.md),
[AI usage report](docs/ai-usage-report.md), [dataset inspection](docs/dataset-inspection.md).

## How it works

```text
INGESTION (when documents change)
file (TXT/MD, JSON, PDF) or dataset record
  -> parser: text split into sentences + metadata (title, pages, filename, ...)
  -> dedup: SHA-256 of normalised text; exact copies are stored as "duplicate", not re-indexed
  -> registry (Postgres/SQLite): document, version, status, sentences, chunks
  -> chunkers: "fixed" (64-word windows, 16 overlap) and "sentence" (whole sentences <= 64 words)
  -> embeddings (all-MiniLM-L6-v2) -> Qdrant        and        BM25 keyword index (our code)

INVESTIGATION (per question)
question -> LLM plan: up to 3 search queries
         -> search tool: semantic (Qdrant) + keyword (BM25) -> our Reciprocal Rank Fusion
         -> evidence E1..En (chunk text + doc id + sentence span)
         -> LLM decides: {"action": "search", ...} for a missing hop, or {"action": "answer", ...}
         -> our validation: citations must exist, uncited answers become "insufficient",
            conflicts need two cited sources
         -> answered | conflict | insufficient (+ evidence, trace, warnings)
```

| Module (`src/investigator/`) | Responsibility |
| --- | --- |
| `parsers.py` | TXT/MD, JSON (one or many documents), PDF (with page numbers); readable `ParseError`s |
| `models.py`, `text.py` | dataclasses, normalisation, SHA-256 fingerprints, tokenizer, sentence splitter |
| `registry.py` | durable source of truth (SQLAlchemy Core: SQLite locally, Postgres in deployment) |
| `chunking.py` | the two chunking strategies, both keeping sentence spans for citations |
| `bm25.py` | **our** Okapi BM25 (custom algorithm #1): a readable dictionary version and a compact NumPy version with identical scores |
| `fusion.py` | **our** Reciprocal Rank Fusion with a worked example (custom algorithm #2) |
| `vector_store.py` | Qdrant collection per chunker: `qdrant_client` locally, plain HTTP to Qdrant Cloud |
| `embeddings.py` | MiniLM via ONNX (`fastembed`, default) or PyTorch (`sentence-transformers`); same vectors |
| `retrieval.py` | semantic / keyword / hybrid search; BM25 rebuilt from the registry at start-up |
| `ingestion.py`, `bulk.py` | add / update / delete / duplicates / retry; resumable bulk load with reconciliation |
| `llm.py` | Groq and Gemini clients (JSON mode), retries with backoff, error types, token usage |
| `investigate.py` | the multi-step pipeline, tool registry, citation and conflict validation |
| `api.py` | FastAPI: `/documents` (CRUD + upload), `/search`, `/investigate`, `/health`, and the web page at `/` |
| `web/index.html` | UI (plain HTML + JS): investigate, inspect cited sentences in their source, manage documents, raw search |
| `hotpotqa.py`, `evaluation.py` | corpus and question bank, metrics, evaluation runners |

Why the stores stay consistent: the registry is the single source of truth. A document is
`indexing` until its chunks are in the registry, Qdrant and BM25; only then is it `indexed`. A
failure marks it `failed` and removes partial chunks; `retry_failed()` or re-running the bulk load
fixes it. BM25 is rebuilt from the registry on start-up, and Qdrant vectors are removed by
`doc_id`, so updates and deletes never leave stale evidence behind (covered by tests).

## Data

HotpotQA `distractor`, validation split. **One source record = one distinct Wikipedia passage.**
73,700 passage occurrences collapse to **66,581 records** (≥ 50,000), of which 26 are exact text
duplicates; all are processed and 66,555 are indexed. Evaluation questions come from the same
split, so their evidence is in the corpus by construction; answers and supporting facts are
never indexed. Details and measurements: [docs/dataset-inspection.md](docs/dataset-inspection.md).

17 synthetic investigation documents (`eval/synthetic_corpus.json`, all labelled
`synthetic`) add the cases HotpotQA lacks: an incident with change logs, conflicting fire
reports, a contract with no recorded approval, and similar.

## Run it locally

Python 3.11+ (developed on 3.12, Apple Silicon).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"    # server + local Qdrant + dataset tools
cp .env.example .env            # defaults: local SQLite + local Qdrant, no LLM
pytest                          # 60 tests, about 1 second, no network
python scripts/ingest_hotpotqa.py   # about 6.5 min on an M-series Mac; resumable
uvicorn investigator.api:app --port 8000   # UI at http://localhost:8000, API docs at /docs
```

Without an LLM key the app runs in "retrieval only" mode and says so. To get answers, set
`LLM_PROVIDER=groq` and `LLM_API_KEY` in `.env` (default model `openai/gpt-oss-120b`). Groq's
free tier for that model allows 1,000 requests/day, 8,000 tokens/minute **and 200,000 tokens/day**;
one answer-evaluation run of 46 questions × 2 configurations uses most of a day's tokens, and the
live demo shares the same key. When the daily quota is used up the app says so within seconds
and still shows the retrieved evidence. `gemini` is also supported (default `gemini-3.8-flash`).

## Evaluation

```bash
python scripts/evaluate.py retrieval    # no LLM: 2 chunkers x 3 modes on 46 + 500 questions
python scripts/evaluate.py answers      # LLM: 46 questions, 2 configurations
```

Retrieval, 500 HotpotQA questions (document-level relevance):

| Chunking + retrieval | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: |
| sentence + semantic | 0.626 | 0.684 | 0.776 |
| sentence + BM25 | 0.697 | 0.773 | 0.827 |
| sentence + hybrid (RRF) | **0.726** | **0.797** | 0.828 |
| fixed + hybrid (RRF) | 0.717 | 0.781 | **0.832** |

End to end, 46 questions, LLM `openai/gpt-oss-120b` on Groq (free tier). **Run 1** is the
official comparison of the two configurations:

| Configuration | Answer correctness | simple | multi-step | multi-doc | unanswerable | contradictory | Citation precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sentence chunks + hybrid | **0.717** | 1.00 | 0.40 | 0.60 | 0.875 | 0.75 | 0.946 |
| fixed chunks + semantic | 0.674 | 0.90 | 0.30 | 0.70 | 0.875 | 0.625 | 0.971 |

**Run 2** was made after fixing the failures found in run 1: a second chance to search before
abstaining, reserved evidence slots for follow-up searches, and one rewritten synthetic conflict
document (disclosed in the report). With sentence + hybrid it scores **0.804 (37/46)**:
multi-step 0.40 → 0.70, contradictory 0.75 → 0.875, citation precision 0.974. 4 questions went
from wrong to right and none from right to wrong. The fixed + semantic run stopped at 20/46 on the
free daily token limit, so it is reported as partial.

Full tables, the 46-question bank, failure analysis by cause and limitations:
[docs/evaluation-report.md](docs/evaluation-report.md).

## Deployment (free tiers)

```text
browser ──> https://ai-investigation-engine-3faf.onrender.com
            Render web service (Docker, free, 512 MB): FastAPI = web page + JSON API
                 │  keyword index (our BM25) rebuilt from Postgres at start-up, in memory
                 ├──> Neon Postgres (registry: 66,598 documents, 130,854 chunks; 160 MB of 0.5 GB)
                 ├──> Qdrant Cloud (130,854 vectors, 384-d; free 1 GB cluster)
                 └──> Groq API (LLM)
```

Measured on the live service on 28 October 2026 (numbers from its own `/health` endpoint):

| Piece | Service | Limit that matters | Measured |
| --- | --- | --- | --- |
| App | Render free web service (`render.yaml`, `Dockerfile`), shared CPU | 512 MB RAM; sleeps after 15 min idle; 750 free hours/month | **334–336 MB** in use; **start-up 93–102 s** (most of it rebuilding the keyword index from Neon); a push redeploys in 2–5 min |
| Registry | Neon Postgres (free) | 0.5 GB storage | 66,598 documents, 130,854 chunks, 160 MB |
| Vectors | Qdrant Cloud (free) | 1 GB cluster | 130,854 points, status green |
| LLM | Groq free tier | per model: 1,000 req/day, 8,000 tokens/min, 200,000 tokens/day (rolling 24 h) | ~990 tokens per LLM call, 2–5 calls per question; search alone needs no LLM |
| Cost | | | 0 |

The keyword search answers in about 2 s from outside, including network. The first request after
the service has slept waits for start-up (about 1.5–2 minutes); the page shows "Starting the
server…" and retries meanwhile.

**LLM fallback.** Groq gives each model its own daily quota. When `openai/gpt-oss-120b` runs out,
the live app continues with `openai/gpt-oss-20b`, then `qwen/qwen3.8-27b`, and the answer says
which model produced it. Verified live: with the 120b quota used up, "Which change caused incident
INC-7781, and who approved it?" was answered correctly ("CHG-0412 … approved by Dana Whitfield",
both sources cited) by `gpt-oss-20b` in 4 LLM calls. The evaluation never uses fallbacks.

**Persistence test (passed).** The container keeps no data: the registry is in Neon, vectors in
Qdrant, and the keyword index is rebuilt from Neon at every start.
1. Uploaded a marked test document (`persistence-test.txt`) through the live app on the container
   running commit `4525233`; keyword search found it.
2. Pushed a new commit, so Render replaced the container (the new one, `29bc998`, reported an
   uptime of 117 s).
3. On the new container the document was still there (status `indexed`, 1 chunk) and was found
   by both keyword and semantic search.
4. Deleted it through the app: the API returned 404 for it afterwards, hybrid search no longer
   returned it, and the document count went back to 66,598.

What made it fit 512 MB (all measured on the real data): MiniLM through ONNX instead of PyTorch
(identical vectors, ~480 → ~160 MB), BM25 stored as NumPy arrays (~365 → ~80 MB, identical top-10
on 500 questions), built in a short-lived child process, Qdrant over plain HTTP instead of
`qdrant_client` (−78 MB), and one process serving both the page and the API (Streamlit, which
needed its own ~150 MB process, was replaced by `web/index.html`).

The cloud index holds one strategy (`INDEX_STRATEGIES=sentence`); the chunker comparison runs
locally where both are indexed. Secrets are set in the Render dashboard, never in Git.

Tried and ruled out: Hugging Face Spaces (Docker/Gradio on free CPU now needs a paid plan; 402),
Oracle Cloud Always Free (no Ampere capacity in the account's home region, Paris).

## Known limitations

- The answer evaluation uses string matching, which under-counts answers that are correct but
  worded differently (5 cases are listed in the report).
- Run 2 of the answer evaluation for fixed + semantic is partial (20/46) because of the free
  LLM token limit. `python scripts/evaluate.py answers` resumes it.
- MiniLM is a small embedding model, and BM25 beats it on this data. Exact identifiers such as
  `CHG-0412` are found by keyword search, not by semantic search.
- Dotted identifiers are kept whole, so `pool` does not match `pool.max`.
- On the free tier the first request after 15 idle minutes waits about 2 minutes for start-up.
