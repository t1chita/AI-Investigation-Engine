# Evaluation report

Status (28 October 2026):
- **Retrieval evaluation:** complete (500 + 46 questions, 2 chunkers × 3 modes).
- **Answer evaluation, run 1:** complete for both configurations. This is the official
  comparison of the two configurations.
- **Answer evaluation, run 2:** after the pipeline fixes. Complete for the main configuration
  (46/46). The comparison configuration stopped at 20/46 on Groq's free daily token limit (see
  "Run 2" below).

Every number was produced by `scripts/evaluate.py` on the local index. Raw per-question results:
`eval/results/run1/` (run 1), `eval/results/answers_*.json` (run 2), `eval/results/retrieval_*.json`.

## Setup

- Corpus: 66,581 HotpotQA validation passages (66,555 indexed, 26 exact duplicates) plus 17
  synthetic investigation documents (labelled `synthetic`), 66,598 documents in total.
- Embeddings: `all-MiniLM-L6-v2` (384-d); vector DB: Qdrant (local mode); keyword: our BM25
  (k1 = 1.5, b = 0.75); fusion: our RRF (k = 60, 50 candidates from each method).
- Chunking: `fixed` = 64-word windows with 16-word overlap; `sentence` = whole sentences packed to
  at most 64 words.

### Question sets

| Set | Size | Content |
| --- | ---: | --- |
| Question bank (`bank46`) | 46 | 10 simple, 10 multi-step, 10 multi-document, 8 unanswerable, 8 contradictory |
| HotpotQA benchmark (`hotpot500`) | 500 | first 500 valid validation questions by id (bridge + comparison) |

The bank mixes 16 HotpotQA questions (8 bridge as multi-step, 8 comparison as multi-document)
with 30 synthetic questions over the synthetic documents, because HotpotQA has no simple,
unanswerable or contradictory cases. Selection is deterministic (sorted by question id).

### Metrics

Relevance is judged per document, so both chunkers are compared fairly: chunk hits are collapsed
to their documents in rank order. **Recall@K** = share of gold documents in the top K documents.
**MRR** = mean of 1 / rank of the first gold document. **SF@5** = share of HotpotQA gold
supporting sentences that fall inside the sentence spans of the top 5 chunks (sensitive to
chunk boundaries). Answer metrics (EM, F1, answer correctness, citation precision and recall)
are defined in `src/investigator/evaluation.py` and are reported once the LLM run is done.

## Retrieval results (no LLM involved)

### HotpotQA benchmark, 500 questions

| Chunking | Retrieval | Recall@2 | Recall@5 | Recall@10 | MRR | SF@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed | semantic | 0.488 | 0.621 | 0.689 | 0.762 | 0.567 |
| fixed | keyword (BM25) | 0.528 | 0.692 | 0.766 | 0.828 | 0.657 |
| fixed | hybrid (RRF) | **0.564** | 0.717 | 0.781 | **0.832** | **0.676** |
| sentence | semantic | 0.494 | 0.626 | 0.684 | 0.776 | 0.575 |
| sentence | keyword (BM25) | 0.528 | 0.697 | 0.773 | 0.827 | 0.645 |
| sentence | hybrid (RRF) | 0.563 | **0.726** | **0.797** | 0.828 | 0.672 |

### Question bank, 46 questions

| Chunking | Retrieval | Recall@2 | Recall@5 | Recall@10 | MRR | SF@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed | semantic | 0.670 | 0.812 | 0.877 | 0.823 | 0.677 |
| fixed | keyword | 0.764 | 0.924 | 0.924 | 0.905 | 0.781 |
| fixed | hybrid | 0.779 | 0.884 | 0.946 | 0.891 | 0.781 |
| sentence | semantic | 0.681 | 0.855 | 0.877 | 0.837 | 0.760 |
| sentence | keyword | 0.775 | 0.913 | 0.924 | 0.900 | 0.740 |
| sentence | hybrid | **0.812** | 0.895 | **0.946** | **0.917** | **0.812** |

Recall@10 by category (sentence + hybrid, bank46): simple 1.00, multi-step 0.85,
multi-document 0.95, unanswerable 1.00 (the related documents are found), contradictory 0.94.

### Findings

1. **Hybrid beats either method alone.** On 500 HotpotQA questions, hybrid raises Recall@10 from
   0.684 (semantic) and 0.773 (BM25) to 0.797 with sentence chunks. The two methods fail on
   different questions, and RRF rewards agreement.
2. **BM25 is stronger than MiniLM semantic search on this data.** HotpotQA questions repeat the
   exact names found in the passage titles, which suits keyword matching; MiniLM is a small
   general model. This is why keyword search must not be dropped.
3. **Chunking matters less than retrieval mode here.** Fixed and sentence chunking are within
   about 0.01–0.02 of each other on HotpotQA because most passages are short (median 78 words) and
   become one or two chunks either way. Sentence chunking is slightly better on Recall@5/10 and on
   the question bank (SF@5 0.812 vs 0.781 with hybrid); fixed chunking is slightly better on SF@5
   for HotpotQA. The app uses `sentence` by default because chunks never cut a sentence, which
   makes citations cleaner.

### Failure analysis (sentence + hybrid)

- **Second-hop documents are missed when the question never names them.** Example: *"Which group
  of people in West Africa … use the Ida?"* finds "Yoruba people" but not "Ida (sword)". *"Renamed
  in 2014, what was the vehicle offered as a prize … The Amazing Race Canada?"* finds the season
  page but not "Chevrolet Corvette (C7)". Single-shot retrieval cannot know the bridge entity;
  this is what the multi-step pipeline's follow-up searches are for (the LLM reads the first
  passage, then searches the entity it found).
- **Comparison questions lose one side to near-duplicates.** *"Northwestern University or Johns
  Hopkins University …"* retrieves several Johns Hopkins pages (university, school of medicine)
  before Northwestern. Improvement: one search per compared entity (the planner does this), or
  diversify results per document.
- **Vocabulary mismatch.** For *"At what time did the INC-7781 payment outage begin?"* the status
  page that contradicts the incident report is not found, because it never mentions INC-7781
  ("elevated checkout errors since 16:25"). Improvements: query expansion by the planner, a
  stronger embedding model, or metadata linking (date, company).

## End-to-end answer evaluation, run 1 (official comparison)

LLM: `openai/gpt-oss-120b` on Groq's free tier (temperature 0, JSON mode). Same 46 questions,
same LLM, same pipeline limits (plan ≤ 3 searches, ≤ 2 follow-up searches, 5 results per search,
≤ 10 evidence items of ≤ 700 characters). Only chunking and retrieval change:

- **sentence-hybrid**: sentence chunks, semantic + BM25 fused with RRF (the app's default)
- **fixed-semantic**: fixed chunks, semantic search only

| Metric | sentence-hybrid | fixed-semantic |
| --- | ---: | ---: |
| Answer correctness (all 46) | **0.717** (33/46) | 0.674 (31/46) |
| simple (10) | **1.00** | 0.90 |
| multi-step (10) | **0.40** | 0.30 |
| multi-document (10) | 0.60 | **0.70** |
| unanswerable (8) | 0.875 | 0.875 |
| contradictory (8) | **0.75** | 0.625 |
| EM, answerable questions | **0.467** | 0.400 |
| F1, answerable questions | **0.628** | 0.565 |
| Evidence recall (gold docs among all retrieved evidence) | **0.946** | 0.895 |
| Citation precision (cited docs that are gold) | 0.946 | **0.971** |
| Citation recall, answerable (gold docs cited) | **0.756** | 0.656 |
| LLM errors | 0 | 0 |
| LLM calls | 92 | 93 |
| Wall time (median per question) | 10.6 min (14.2 s) | 11.7 min (15.1 s) |

Both runs together: 185 LLM calls, 142,396 input and 40,416 output tokens (about 990 tokens per
call), and 181 automatic retries after HTTP 429. The free key allows 8,000 tokens per minute,
so most of the wall time is waiting for the rate-limit window, not model latency. Cost: 0 (free
tier).

**Reading the comparison.** Hybrid retrieval with sentence chunks wins overall (+2 questions)
and on evidence recall (+0.05), which is what reaches the LLM. The difference on 46 questions is
small (one question = 2.2 points), so it supports, but does not prove, the larger and clearer
retrieval result on 500 questions above. fixed-semantic cites slightly more precisely because it
cites less (lower citation recall).

### Answer failures (sentence-hybrid, 13 of 46), by cause

| Cause | Count | Questions |
| --- | ---: | --- |
| A needed document was never retrieved | 4 | syn-d01 (CHG-0412 missing, so the model wrongly says only CHG-0415 exists), syn-c08 (the status page with the other start time is not found, so no conflict), Corvette and Kareena Kapoor bridge questions |
| A hop was missing and the model abstained instead of searching | 1 | syn-m01: the incident report names CHG-0412, but the model answered "insufficient" rather than issuing the follow-up `search` for it |
| Evidence was present, the model reasoned wrongly or abstained | 2 | Symphony No. 7 question (picked a different work), Bull Terrier comparison (all evidence present, answered "insufficient") |
| A real conflict was not reported | 1 | syn-c05 (see the label note below) |
| Grading or label problem, answer arguably right | 5 | see below |

Grading and label problems found while reading the outputs (kept as failures in the score; we do
not edit labels after seeing results):
- **syn-m02**: answer "at least 60 connections, 40 more than the 20 set by CHG-0412" is right,
  but our string matcher looks for "60 versus 20".
- **syn-u03** ("What caused the Pier 9 fire?"): the system answered "not determined, still under
  investigation", citing the insurer. That is a correct, cited *negative* answer; our label
  expected the status "insufficient". The pipeline has no separate status for "the sources say
  it is unknown".
- **Scanian dialect**: correct but verbose answer, F1 0.15 against "Scanian dialects".
- **Northwestern vs Johns Hopkins**: answered "Northwestern University"; HotpotQA's gold answer is
  the span "with other campuses located in Chicago and Doha, Qatar", which no short answer matches.
- **Finding Kraftland and Shikashika**: a yes/no question whose HotpotQA gold is the sentence
  "Shikashika is a shaved ice drink", so "No" cannot be scored.

Two contradictory test cases are themselves weaker than intended: in **syn-c04/c05** the status
page reports when *compute* returned (09:58) and the RCA when *all services* returned (11:40, with
storage resync). A careful reader can treat these as compatible, which is what the model did in
c05 (both configurations) and c04 (fixed-semantic). They should be rewritten as a clean conflict.

### Improvements, in order of expected value

Items 1 and 5 were done before run 2 (below); the others are still open.

1. **Make the follow-up search harder to skip.** When the model says "insufficient" but the
   evidence names an entity it has not searched (syn-m01), run one more search automatically, or
   prompt with the specific unresolved entity.
2. **Retrieval diversity.** Run one search per entity in comparison questions and cap results per
   document, so one entity's pages cannot crowd out the other's.
3. **A fourth status for verified negatives** ("the sources state it is not known"), separate from
   "insufficient".
4. **Lenient answer grading**: an LLM-as-judge or alias lists for the answerable questions, and
   excluding HotpotQA questions whose gold is not an answer span.
5. **Fix the two weak synthetic conflicts** (c04/c05) and re-run.
6. **A stronger embedding model** (BM25 beats MiniLM here), then re-measure retrieval.

## Answer evaluation, run 2 (after the fixes; partial)

What changed between run 1 and run 2 (same 46 questions, same LLM, same labels):
1. **Second chance before abstaining.** When the model answers "insufficient", the pipeline asks
   it once more to name one missing entity and runs that search before accepting the abstention.
2. **Reserved evidence slots.** The first searches may fill 8 evidence slots, and each follow-up
   search adds up to 3 more. In run 1, the first searches could fill all 10 slots, so follow-up
   searches could not add the missing hop.
3. **One synthetic document rewritten.** `syn-nw-status` now says that *all* services were
   restored at 09:58 UTC (46 minutes). This makes syn-c04/c05 a clean conflict with the RCA's
   11:40. This is a change to test data. It is disclosed here because it affects 2 of the
   46 questions.

How the run went: Groq's free tier allows 200,000 tokens per model per rolling 24 hours, so run 2
was resumed several times as tokens became available. Each answer is saved as soon as it is
produced. Two questions (syn-s01, syn-s02) first failed with a network error
(`LLMUnavailable: ConnectError`: the machine lost its connection, so these are not model
answers). They were removed and asked again. No other answer was edited or re-asked.

### sentence-hybrid: complete (46/46)

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Answer correctness (all 46) | 0.717 (33/46) | **0.804 (37/46)** |
| simple (10) | 1.00 | 1.00 |
| multi-step (10) | 0.40 | **0.70** |
| multi-document (10) | 0.60 | 0.60 |
| unanswerable (8) | 0.875 | 0.875 |
| contradictory (8) | 0.75 | **0.875** |
| EM, answerable | **0.467** | 0.433 |
| F1, answerable | 0.628 | **0.680** |
| Evidence recall | 0.946 | **0.967** |
| Citation precision | 0.946 | **0.974** |
| Citation recall, answerable | 0.756 | **0.822** |
| LLM calls | 92 | 106 |

Per question: **4 questions went from wrong to right, and none went from right to wrong.**
- syn-m01 (INC-7781 → CHG-0412 → approver): run 1 abstained. Run 2 searched for CHG-0412 and
  answered "CHG-0412, approved by Dana Whitfield", citing both documents.
- The Corvette and Kareena Kapoor bridge questions: the second hop was found by a follow-up search.
- syn-c05: now reported as a conflict, after the rewrite in change 3.

The cost is 14 more LLM calls (+15%) for the extra searches. EM fell slightly (0.467 → 0.433)
while F1 rose, because some answers are now longer and cite more.

Still wrong (9): syn-m02, syn-u03 and two HotpotQA questions are the grading or label cases
listed for run 1. The others are syn-d01 and syn-c08 (a needed document is never retrieved),
one more bridge question, and two comparison questions (one side is crowded out). These match
open items 2–4 above.

### fixed-semantic: partial (20/46, stopped by the daily token limit)

The 20 questions answered so far are 10 simple, 2 multi-step, 2 multi-document and 6
unanswerable. On these 20 questions, run 1 and run 2 are both correct on 16 of 20 (0.80). An
overall score from 20 questions would not be comparable with run 1's 46, so it is not given.

One result here is useful even though the run is partial. **The second chance does not help
when retrieval cannot find the hop.** On syn-m01 the extra search ran (4 LLM calls), but the
document `CHG-0412` was still not retrieved, so the answer stayed "insufficient". On syn-s03, a
semantic search for "CHG-0412 approval" also failed to find it. An exact identifier like
CHG-0412 is what BM25 matches and MiniLM does not. The same question succeeds with hybrid
retrieval. This supports the retrieval finding that keyword search must stay in the pipeline.

The remaining 26 questions can be run with `python scripts/evaluate.py answers`. The command
resumes from the saved file.

## Limitations

- 30 of the 46 bank questions are synthetic and were written by us; they test behaviour
  (abstention, conflict reporting) rather than open-domain difficulty.
- Relevance is document-level; a document can count as found even when the retrieved chunk is
  not the one with the answer (SF@5 measures that separately for HotpotQA).
- No confidence intervals. The 500-question set makes the retrieval numbers fairly stable, and
  the 46-question numbers less so: one question is about 2 percentage points. Run 2 changed the
  pipeline and one test document at the same time, so its gain cannot be split exactly between
  the two. Only syn-c04/c05 use the rewritten document, and of those only syn-c05 changed result.
- Answer correctness uses string matching (contains or F1 ≥ 0.5), which under-counts correct but
  differently worded answers (5 such cases listed above).
- Only one LLM was evaluated; the free-tier token limit makes larger answer runs slow (about
  11 minutes per configuration for 46 questions).
