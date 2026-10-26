# AI Investigation Engine

> **Project status (27 October 2026):** Initial Git and macOS Python environment setup completed/reported; architecture concepts studied. Application functionality has not been implemented or tested. Features below remain planned.

An evidence-driven AI investigation system designed to search large document collections, connect facts across multiple sources, and answer complex questions with traceable citations. Unlike a basic *chat with documents* application, the goal is to support multi-step investigation, identify conflicting evidence, and acknowledge when available sources cannot justify an answer.

## Current progress (27 October 2026)

| Milestone | Status |
| --- | --- |
| Architecture, document/chunk metadata, chunking, embeddings, cosine similarity, Qdrant | Studied conceptually; not implemented |
| GitHub repository and README | Created |
| Python environment | Python 3.12.x in `.venv` on Apple Silicon macOS, reported active |
| PyCharm interpreter and dependency verification | Pending |
| Dataset, ingestion, hybrid search, RAG, evaluation, deployment | Not started |

**Next:** Verify the PyCharm interpreter, inspect HotpotQA and identify source records, then implement and test ingestion. Mandatory requirements first; learn → implement → test → explain. No application code without explicit request.

## At a glance

| Item | Plan / requirement |
| --- | --- |
| Backend | Python + FastAPI (planned) |
| Interface | Streamlit (planned) |
| Vector search | Qdrant (planned) |
| Embeddings | Sentence Transformers (planned; model to be selected) |
| Keyword search | BM25 (planned) |
| Hybrid ranking | Custom reciprocal rank fusion (RRF) (planned) |
| LLM | To be selected after access, reliability, and cost checks |
| Dataset | Evaluate HotpotQA as the starting dataset; final choice pending |
| Hosting | Free-tier public deployment; platform pending |
| Evaluation | At least 40 test questions and two configurations |
| Scale target | At least 50,000 documents or independent text records |

## Why this project exists

Organizations often keep useful evidence in separate incident reports, change logs, operational notes, and technical documents. A question such as:

> Which system changes contributed to the March payment outage, which services were affected, and which documents support the conclusion?

may require reading and comparing multiple sources. The system should **retrieve evidence, connect supported facts, cite the documents, and avoid inventing explanations**.

The outage scenario is illustrative; it is not a claim about the selected dataset.

## How it is intended to work

We separate two workflows because document processing happens when content changes, while investigation happens when a user asks a question.

### 1. Document ingestion pipeline

```text
Upload / import documents
          |
          v
Parse text + source metadata
          |
          v
Detect duplicates / track versions
          |
          v
Chunk text (compare two strategies)
          |
          v
Generate embeddings
          |
          +--------------------+
          |                    |
          v                    v
  Vector index          Keyword index
          \                    /
           \                  /
           Searchable knowledge
```

Planned behavior:

- Accept **at least three file formats**; exact supported formats will be confirmed after implementation.
- Extract text and important metadata such as document identity, title, source, and available location information.
- Deduplicate repeated content while preserving source provenance.
- Compare **at least two chunking strategies** with measured retrieval outcomes.
- Add, update, and delete documents, including associated search-index entries.
- Preserve persistent document and chunk identifiers so citations remain traceable.

### 2. Investigation pipeline

```text
User question
     |
     v
Interpret / decompose complex question
     |
     v
Semantic search + BM25 keyword search
     |
     v
Custom hybrid result fusion / ranking
     |
     v
Collect and compare evidence across documents
     |
     v
Detect potential contradictions / missing support
     |
     v
LLM-generated grounded answer
     |
     v
Citations, caveats, or abstention
```

The LLM should reason over retrieved evidence rather than pretend to know the entire document collection. Contradiction detection is a **best-effort mechanism**, not a guarantee that all conflicts will be found. If the retrieved evidence is insufficient, the system should explicitly say so.

## Proposed architecture

| Component | Responsibility | Rationale |
| --- | --- | --- |
| Streamlit | Upload/manage documents, ask questions, inspect citations | Functional UI with low frontend overhead |
| FastAPI | Request validation and API endpoints | Clear separation of backend and interface |
| Ingestion services | Parsing, metadata, deduplication, chunking, index lifecycle | Testable document processing |
| Sentence Transformers | Create vectors for chunks and queries | Local embedding option without per-request embedding API charges |
| Qdrant | Store and retrieve vectors with source metadata | Semantic retrieval and filtering |
| BM25 index | Retrieve exact terms, names, and identifiers | Complements semantic retrieval |
| Custom RRF | Combine semantic and keyword rankings | Explainable custom algorithmic search component |
| Investigation service | Decompose questions, invoke retrieval tools, assemble evidence | Supports multi-step questions and tool calling |
| LLM integration | Generate answers constrained by retrieved evidence | Source-grounded responses and controlled failure handling |
| Evaluation harness | Score retrieval, answers, and citations | Reproducible comparison of configurations |

**Design note:** Exact persistence choices for original documents, metadata, BM25 indexes, and vector indexes must be finalized before deployment. We must verify that add/update/delete operations remain consistent across all stores.

## Assignment requirements checklist

All items below are **pending** until implemented and verified.

### Document processing

- [ ] Support at least three file formats.
- [ ] Extract text and meaningful metadata.
- [ ] Implement chunking and compare two strategies.
- [ ] Handle duplicate information.
- [ ] Add, update, and delete documents.

### Search

- [ ] Use a vector-capable database.
- [ ] Generate embeddings and perform semantic search.
- [ ] Implement hybrid semantic + keyword search.
- [ ] Implement at least one custom search or ranking algorithmic component.

### LLM and investigation

- [ ] Integrate at least one LLM and a RAG workflow.
- [ ] Combine information from multiple documents.
- [ ] Process complex questions in multiple steps.
- [ ] Generate evidence-grounded answers with citations.
- [ ] Attempt to identify contradictory information.
- [ ] Abstain when evidence is insufficient.
- [ ] Handle LLM errors, timeouts, and limits.
- [ ] Include tool calling or equivalent structured tool invocation.

### Evaluation and scale

- [ ] Process at least 50,000 documents or independent text records.
- [ ] Create at least 40 test questions covering simple, multi-step, multi-document, unanswerable, and contradictory cases.
- [ ] Quantitatively evaluate retrieval and answer quality.
- [ ] Compare at least two configurations.
- [ ] Analyze failures and improvement opportunities.

### Application, deployment, and quality

- [ ] Python backend and functional user interface.
- [ ] Public live URL on a free hosting plan.
- [ ] Persistent data after application restarts.
- [ ] Secure secrets and API keys.
- [ ] Document deployment architecture, cost, and free-tier limits.
- [ ] Automated tests, error handling, modular code, and technical documentation.

## Dataset plan

The assignment allows a reproducible subset of the supplied or suggested datasets and requires at least **50,000 documents or independent text records** to be processed. Our initial candidate is [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), which is designed for multi-hop question answering.

Before committing to the dataset, we will document:

1. Which split and subset we use and why.
2. What constitutes one independent text record (rather than counting chunks as separate documents).
3. How records are uniquely identified and linked to their source.
4. How evaluation questions are kept separate from development decisions where appropriate.
5. How the same corpus can be recreated reproducibly.

Other assignment-suggested resources include [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench), [2WikiMultihopQA](https://huggingface.co/datasets/xanhho/2WikiMultihopQA), and [post-mortems](https://github.com/danluu/post-mortems). Dataset suitability and licensing should be checked before use.

## Evaluation plan

A dedicated automated evaluation suite will contain **at least 40 questions**, including:

- Simple factual questions.
- Multi-step questions.
- Questions requiring evidence from several documents.
- Questions with no supported answer.
- Questions involving conflicting evidence.

We intend to measure:

| Metric | What it tells us |
| --- | --- |
| Recall@K | Whether relevant evidence appears among the top K retrieved results |
| MRR | How early the first relevant result appears |
| Answer correctness | Whether the generated answer matches supported ground truth |
| Citation accuracy | Whether cited sources actually support the claims |
| Abstention behavior | Whether the system avoids unsupported answers |

We will compare at least two configurations, potentially **different chunking strategies** or **semantic-only vs. hybrid retrieval**, while holding other variables constant where possible. The final evaluation report will include measured results, methodology, failure examples, and limitations. **No performance results are claimed yet.**

## Local development on macOS

**Current status:** Git and GitHub are configured. Python 3.12.x in `.venv` has been reported active; PyCharm's interpreter selection and dependencies still require verification. This is not yet a tested installation guide.

Prerequisites:

- macOS with a compatible Python installation (target: Python 3.11 or 3.12).
- Git.
- PyCharm or another Python IDE.
- Access to the chosen LLM provider, if a hosted LLM is used.

Planned setup sequence:

1. Verify Python and Git versions and locate the Python executable.
2. Create the project directory and a dedicated Python virtual environment.
3. Configure PyCharm to use that environment.
4. Install pinned dependencies from the repository's dependency manifest (to be added).
5. Configure environment variables using an example file (to be added), without committing secrets.
6. Initialize persistent local stores and ingest a small test corpus.
7. Run backend, UI, and tests using documented commands (to be added and verified).

> **Setup commands will be added only after we have run and verified them.** This avoids publishing misleading instructions for a project that has not yet been implemented.

## Planned API and user workflows

The backend will need operations for:

- Uploading/importing documents and checking ingestion status.
- Listing documents and inspecting their metadata.
- Updating and deleting documents.
- Submitting an investigation question.
- Returning an answer together with source identifiers and citations.

Specific endpoint paths, request/response schemas, and error codes will be documented after implementation. The UI will expose the corresponding operations.

## Reliability, security, and limitations

We plan to account for:

- Invalid or unsupported files and extraction failures.
- Duplicate uploads and partial indexing failures.
- Consistent document updates/deletions across indexes.
- LLM timeouts, rate limits, and malformed responses.
- Missing evidence and conflicting documents.
- Secret management through environment variables rather than committed credentials.
- Persistent storage after restarts and the resource limits of free hosting.

Retrieval and LLM generation are probabilistic: citations and abstention reduce risk, but do not guarantee factual correctness. Evaluation should make those limitations visible.

## Deployment

**Not deployed yet.** The assignment requires a public URL using free hosting, persistence across restarts, database integration, and protected secrets. We will select a hosting arrangement after checking model, storage, and compute requirements.

- **Live demo:** Coming after deployment.
- **Hosting provider:** To be decided.
- **Persistent storage:** To be designed and verified.
- **Free-tier constraints:** To be documented with the deployed configuration.

## Project deliverables

| Deliverable | Status |
| --- | --- |
| GitHub repository and history | Created; application code pending |
| Public live demo | Pending |
| README and architecture rationale | Initial draft |
| Evaluation report with results and analysis | Pending |
| AI usage report | Pending |

## Three-day execution plan

| Day | Primary focus |
| --- | --- |
| **1** | Environment, architecture, ingestion, embeddings, vector and hybrid retrieval |
| **2** | Investigation pipeline, multi-step RAG, citations, error handling, API and UI |
| **3** | Full-scale verification, 40+ question evaluation, deployment, documentation, interview preparation |

This is an aggressive plan. Mandatory requirements take priority over optional enhancements. Integration tests and documentation should be updated continuously rather than deferred entirely to the final day.

## Learning resources and credits

These resources were **recommended during project mentoring**. Their inclusion is not a claim that every video has been watched or that tutorial code was used. Actual AI/tool usage will be documented in the required AI Usage Report.

| Topic | Video | Creator |
| --- | --- | --- |
| RAG fundamentals | [What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M) | IBM Technology |
| Vector cosine similarity | [Cosine Similarity (YouTube search)](https://www.youtube.com/results?search_query=StatQuest+Cosine+Similarity+Clearly+Explained) | StatQuest |
| Dense/sparse hybrid retrieval | [Implementing Hybrid Search in Qdrant](https://www.youtube.com/watch?v=zaQYa7oa1a8) | Qdrant |
| Python APIs | [Python FastAPI Tutorial](https://www.youtube.com/watch?v=iukOehU5aF4) | Corey Schafer |

Concepts covered in mentoring: ingestion versus investigation, document IDs and metadata, versioning, fixed-size and semantic chunking, embeddings, cosine similarity, and Qdrant's role. BM25/RRF implementation and evaluation remain ahead.

## AI assistance disclosure

AI tools may assist with explanations, design review, debugging, and documentation. The final **AI Usage Report** will record the tools actually used, their contributions, and the implementation or design changes made independently. The author remains responsible for understanding and defending all submitted code.

## Future enhancements (not part of the mandatory-first plan)

After all mandatory requirements are met, possible enhancements include multi-agent orchestration, more advanced reranking, prompt-injection testing, caching, streaming responses, monitoring, and larger-scale performance experiments.

---

**Repository note:** This README is a living document. Replace planned descriptions with verified implementation details, real setup commands, evaluation results, and a working demo link as the project progresses.
