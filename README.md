# AI Investigation Engine

An evidence-driven investigation system. It searches a collection of 66,000+ documents, connects
facts across several sources, and answers questions with citations to exact sentences. When the
evidence is missing it says so, and when sources disagree it reports the conflict instead of
choosing one.

> **Status (28 October 2026).** Implemented and tested: ingestion of 3 formats with
> add/update/delete and deduplication, two chunkers, Qdrant + our BM25 + our RRF hybrid search,
> the multi-step RAG pipeline with tool calls, citations, conflict reporting and abstention, the
> FastAPI backend with its web page, the 66,581-record corpus (loaded locally **and** into Neon +
> Qdrant Cloud), and both evaluations (retrieval on 500 questions, end-to-end answers on 46
> questions with Groq). The app runs against the cloud data locally in about 360 MB.
> **Not done yet:** the public deployment (Render, needs your account and a GitHub push) and the
> second answer-evaluation run (waits for Groq's daily token quota). See "What is left".

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
pytest                          # 56 tests, about 1 second, no network
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

End to end, 46 questions, LLM `openai/gpt-oss-120b` on Groq (free tier):

| Configuration | Answer correctness | simple | multi-step | multi-doc | unanswerable | contradictory | Citation precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sentence chunks + hybrid | **0.717** | 1.00 | 0.40 | 0.60 | 0.875 | 0.75 | 0.946 |
| fixed chunks + semantic | 0.674 | 0.90 | 0.30 | 0.70 | 0.875 | 0.625 | 0.971 |

Full tables, the 46-question bank, failure analysis by cause and limitations:
[docs/evaluation-report.md](docs/evaluation-report.md).

## Deployment (free tiers)

```text
browser ──> Render web service (Docker, free, 512 MB): FastAPI = web page + JSON API
                 │  keyword index (our BM25) rebuilt from Postgres at start-up, in memory
                 ├──> Neon Postgres (registry: 66,598 documents, 130,854 chunks; 160 MB of 0.5 GB)
                 ├──> Qdrant Cloud (130,854 vectors, 384-d; free 1 GB cluster)
                 └──> Groq API (LLM)
```

| Piece | Service | Limit that matters | Measured |
| --- | --- | --- | --- |
| App | Render free web service (`render.yaml`, `Dockerfile`) | 512 MB RAM; sleeps after 15 min idle | 359 MB after start-up; start-up 40 s (keyword index from Neon) |
| Registry | Neon Postgres | 0.5 GB storage | 160 MB used |
| Vectors | Qdrant Cloud | 1 GB cluster | 130,854 points, status green |
| LLM | Groq free tier | 1,000 req/day, 8,000 tokens/min, 200,000 tokens/day | ~990 tokens per LLM call, 2–5 calls per question |
| Cost | | | 0 |

What made it fit 512 MB (all measured on the real data): MiniLM through ONNX instead of PyTorch
(identical vectors, ~480 → ~160 MB), BM25 stored as NumPy arrays (~365 → ~80 MB, identical top-10
on 500 questions), built in a short-lived child process, Qdrant over plain HTTP instead of
`qdrant_client` (−78 MB), and one process serving both the page and the API (Streamlit, which
needed its own ~150 MB process, was replaced by `web/index.html`).

The cloud index holds one strategy (`INDEX_STRATEGIES=sentence`); the chunker comparison runs
locally where both are indexed. Secrets are set in the Render dashboard, never in Git.

Tried and ruled out: Hugging Face Spaces (Docker/Gradio on free CPU now needs a paid plan; 402),
Oracle Cloud Always Free (no Ampere capacity in the account's home region, Paris).

## What is left

1. **Deploy to Render**: push to GitHub, create a free Render account, create a Blueprint from
   the repo (`render.yaml`), enter the 4 secrets; then verify the public URL and that data
   survives a restart.
2. **Answer evaluation run 2** (second chance before abstaining, reserved evidence slots, fixed
   synthetic conflicts) once Groq's daily token quota resets.
3. The AI usage report.
