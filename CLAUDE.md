# AI Investigation Engine: project instructions for Claude Code

> **Mode (since 27 October 2026): build mode.** The developer asked Claude Code to build the
> product to the assignment's requirements. The earlier mentoring-only rules ("never write code
> unless asked") no longer apply. Honesty and verification rules below still apply in full.

## 1. How to work

1. **Build the product.** Write, refactor and test application code without asking first. Ask only
   for decisions that are genuinely the developer's: accounts, API keys, paid services, pushing to
   GitHub, deleting data.
2. **Mandatory requirements first** (section 3). Bonus features only once every mandatory item is
   verified.
3. **Never claim unverified progress.** Track *implemented* and *tested* separately. A requirement is
   done only when a test, a measured run or a live check proves it. Update the ledger (section 7).
4. **Keep code explainable.** The developer must explain, debug and extend it in an interview with
   no AI. Prefer small modules, plain Python, clear names, docstrings on public functions, and no
   framework where twenty lines of our own code are clearer. Every algorithm we own (BM25, RRF,
   chunkers, metrics) gets a docstring with its formula and a hand-checkable unit test.
5. **Secrets.** Keys live only in `.env` (git-ignored); `.env.example` lists names with no values.
   Never print, log, commit or ask for secret values.
6. **Git.** Commit in small, meaningful steps when asked or at milestone ends; never push without
   the developer's go-ahead.
7. **Honest data.** Synthetic documents and questions (needed for unanswerable and contradictory
   cases) are always labelled synthetic in data, reports and UI.

## 2. Sources of truth

- Official assignment PDF: `assignment-requirements.pdf` (Georgian, 8 pages, project root). It wins
  over this file if they ever disagree.
- Repository: https://github.com/t1chita/AI-Investigation-Engine (branch `main`).
- `frontend/` is a separate, clearly labelled **mock** UI preview ("Evidence Quest"). It satisfies
  no requirement and must not be presented as the product UI.

## 3. Mandatory acceptance requirements

### A. Document ingestion and lifecycle
- At least three file formats; extract text and important metadata; chunking.
- Handle duplicate information; add, update and delete documents.
- Compare at least two chunking strategies; justify the ingestion architecture.

### B. Retrieval
- A vector database (plain SQL text search does not count); embeddings and semantic search.
- Hybrid search combining semantic and keyword search.
- At least one search or ranking algorithm of our own (not only library calls).

### C. LLM and RAG
- At least one LLM in a RAG architecture; combine evidence from several documents.
- Multi-step processing of complex questions; answers grounded in evidence; sources and citations.
- Attempt to detect conflicting information; abstain when evidence is insufficient.
- Handle LLM errors and limits; at least one tool-calling or equivalent structured tool mechanism.

### D. Evaluation
- Own automated evaluation; at least 40 questions: simple, multi-step, multi-document,
  unanswerable, contradictory.
- Quantitative retrieval and answer quality (e.g. Recall@K, MRR, Answer Correctness, Citation
  Accuracy); compare at least two configurations; analyse failures, causes and improvements.

### E. Dataset and scale
- Use an assignment dataset (HotpotQA, RAGBench, 2WikiMultihopQA, danluu post-mortems); extra
  public or synthetic data allowed.
- Process at least 50,000 documents or independent text records (not chunks, not questions).
- Cross-document linking; keep source identity; handle incomplete and contradictory information.
- Reproducible evaluation; a justified reproducible subset is allowed.

### F. Application and deployment
- Python backend and a functional UI: add and manage documents, ask questions, inspect sources.
- Public URL on free hosting or free tiers; working DB integration; data persists across restarts.
- Protect keys; document deployment architecture, costs, resources and free-tier limits.

### G. Engineering and submission
- Modular Python, separated responsibilities, error handling, automated tests, safe config,
  technical docs, performance and LLM-cost awareness.
- Submit: GitHub repo with history, live demo, README, evaluation report, AI usage report.

Bonus (deferred): multiple LLM or embedding comparisons, agentic RAG, advanced re-ranking,
prompt-injection tests, caching, streaming, observability, large-scale load tests.

## 4. Technical decisions

| Concern | Choice | Notes |
| --- | --- | --- |
| Language | Python 3.12 (`.venv`) | Apple Silicon macOS |
| Package | `src/investigator/` | `pyproject.toml`, pytest |
| API | FastAPI | |
| UI | `src/investigator/web/index.html` (plain HTML + JS, served by FastAPI at `/`) | replaced Streamlit to fit 512 MB |
| Source registry | Postgres (Neon free tier); SQLite for local dev and tests | persists across restarts |
| Vector DB | Qdrant (Qdrant Cloud free tier; local mode in dev and tests) | |
| Embeddings | `all-MiniLM-L6-v2` (384-d) via ONNX `fastembed` (default) or PyTorch | identical vectors; ONNX for the server |
| Keyword search | own BM25 | custom algorithm #1 |
| Fusion | own Reciprocal Rank Fusion | custom algorithm #2 |
| LLM | provider chosen by the developer (Gemini or Groq free tier) | behind one interface; a fake LLM for tests |
| Dataset | HotpotQA `distractor` | unique passages = source records |
| Hosting | Render free web service (`render.yaml`, `Dockerfile`), Neon, Qdrant Cloud | HF Spaces needs PRO (402); Oracle Paris has no free capacity |
| LLM limits | Groq `openai/gpt-oss-120b`: 1,000 req/day, 8,000 tok/min, 200,000 tok/day | one 2-config answer eval ≈ a full day's tokens |

## 5. Architecture

- **Ingestion:** file or dataset record → parser (PDF, TXT, JSON) → normalised `SourceDocument`
  → dedup (SHA-256 of normalised text) and versioning → chunker (strategy A or B) → embeddings →
  Qdrant + BM25 index → registry status `indexed`.
- **Investigation:** question → planner (sub-questions) → `search` tool calls (hybrid: vector +
  BM25 → RRF) → evidence set → LLM answer with citation IDs → citation check, conflict check,
  abstention → answer or "insufficient evidence".
- **Identity rules:** `doc_id` is stable and not the filename; `version` increments on update;
  `chunk_id` = doc_id + version + strategy + position; chunks keep the original text and sentence
  offsets; deleted or replaced versions must be removed from every index.
- Evaluation labels (answers, supporting facts) are never indexed as evidence.

## 6. Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
HF_DATASETS_OFFLINE=1 python scripts/ingest_hotpotqa.py   # resumable bulk load
python scripts/evaluate.py retrieval | answers
uvicorn investigator.api:app --port 8000
```

UI at http://localhost:8000. `.env` points at the cloud stores (Neon + Qdrant Cloud, sentence
strategy only); for local work or the evaluation override with
`DATABASE_URL=sqlite:///data/registry.db QDRANT_URL= INDEX_STRATEGIES=fixed,sentence`.

The desktop preview launcher cannot read `~/Documents` (macOS privacy), so start servers from the
shell. Local Qdrant allows one process at a time: stop the API before running ingest or evaluate.

## 7. Progress ledger (update with evidence only)

| Area | Implemented | Tested | Evidence / next proof |
| --- | --- | --- | --- |
| HotpotQA inspection | yes | run 2026-10-27 | `docs/dataset-inspection.md` |
| Source record definition and 50k corpus | yes | yes | 66,581 validation passages; `hotpotqa.py` |
| Data models, registry | yes | yes | `tests/test_lifecycle.py` |
| Parsers (TXT/MD, JSON, PDF), lifecycle, dedup | yes | yes | `tests/test_chunking_parsing.py`, `test_lifecycle.py` |
| Two chunkers + comparison | yes | yes | retrieval tables in `docs/evaluation-report.md` |
| Embeddings + Qdrant | yes | yes | local Qdrant; 263,210 vectors reconciled |
| BM25 + RRF hybrid | yes | yes | `tests/test_ranking.py` (hand-computed) |
| 50k ingestion | yes | yes | 66,581 processed, 66,555 indexed, 26 duplicates, 0 failed |
| RAG, tools, citations, conflicts, abstention | yes | yes | `tests/test_investigate.py` + 92 real questions |
| LLM clients (Groq, Gemini), retries | yes | yes | Groq `openai/gpt-oss-120b` in use; Gemini key's project returns 402 (no credits) |
| API + UI | yes | yes (local) | `tests/test_api.py`; UI checked in browser |
| Evaluation (46 questions, 2 configs) | yes | yes | run 1: 0.717 vs 0.674; run 2 (after fixes): 0.804 vs 0.739; `docs/evaluation-report.md` |
| Cloud data (Neon + Qdrant Cloud) | yes | yes | 66,598 docs, 130,854 chunks = 130,854 vectors |
| Slim server (≤ 512 MB) | yes | yes (local, cloud data) | 359 MB; 56 tests; clean-venv install without torch |
| Deployment + persistence | yes | yes | https://ai-investigation-engine-3faf.onrender.com; 334–336 MB, start-up 93–102 s; test doc survived container replacement (README) |
| LLM fallback on daily quota | yes | yes (live) | gpt-oss-120b → gpt-oss-20b → qwen3.8-27b; evaluation uses no fallback |
| README, evaluation report, AI usage report | yes | – | all three submission documents complete (29 Oct) |

Local data (git-ignored): `data/registry.db` (SQLite, 208 MB), `data/qdrant/` (1 GB, both
strategies). Rebuild with `python scripts/ingest_hotpotqa.py` (about 6.5 min, resumable).
