"""
HotpotQA dataset inspection (Milestone 1)

PURPOSE
    Understand the real structure and quality of the HotpotQA "distractor"
    training split BEFORE building ingestion on top of it. This script only
    READS and PRINTS. It does not write files, build an index or change data.

WHAT IT DOES
    Part 1  Inspect the first 10 questions: fields, passage counts, empty
            passages/sentences, and repeated titles (inside one example and
            across examples).
    Part 2  Resolve each "supporting fact" (a title + sentence number) to the
            real sentence in the example's context, and report references
            that point to nothing.
    Part 3  Walk the whole training split, count every passage occurrence,
            and count unique non-empty passages by SHA-256 fingerprint of
            their normalized text.

HOW HOTPOTQA IS STRUCTURED (as used here)
    - One row = one QUESTION (id, question, answer, type, level).
    - row["context"] holds the passages for that question, as two parallel lists:
          context["title"][j]      title of passage j
          context["sentences"][j]  list of sentences of passage j
      Both lists must have the same length, and index j refers to the same
      passage in both.
    - row["supporting_facts"] lists the evidence needed to answer:
          title[k]    which passage
          sent_id[k]  zero-based sentence position inside that passage
    - Distractor passages are repeated across questions, so total passage
      occurrences is much larger than the number of unique passages.

IMPORTANT DEFINITIONS
    - A question is NOT a source record, and a chunk is NOT a source record.
    - Identical titles do not prove identical content, so uniqueness is
      judged by a hash of the passage TEXT, not by its title.

RUN
    From the project root, with the .venv active:
        python scripts/dataset-inspection.py
    The first run downloads the dataset from Hugging Face (needs internet).
    Part 3 loops over about 90k examples, so it can take a little while.

STATUS
    Partial inspection only. Not yet covered: the validation split, the
    reproducible 50,000-record subset, and a written inspection note.
"""

# Hugging Face library that downloads and caches the dataset.
from datasets import load_dataset
# Used to build a fixed-length fingerprint of each passage's text.
import hashlib

# Download (or load from cache) the HotpotQA "distractor" configuration.
# The result is a DatasetDict with a "train" and a "validation" split.
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

# Work only on the training split (the larger one).
train = dataset["train"]

# Basic size and schema check: how many questions, and which columns exist.
print("Total training examples:", len(train))
print("Available fields:", train.column_names)


# ========================================
# PART 1: Inspect the first 10 examples
# ========================================

# Every title seen so far across all 10 examples (used for cross-example
# repeats). It is created once, outside the loop, so it keeps growing.
seen_titles = set()

# Look at the first 10 questions only (indexes 0 to 9).
for i in range(10):
    # One row: a dict with id, question, answer, type, level, ...
    example = train[i]

    # "context" holds the passages shown for this question.
    context = example["context"]
    # List of passage titles, one per passage.
    titles = context["title"]
    # List of passages; each passage is itself a list of sentence strings.
    sentences = context["sentences"]

    print("\n-----------------------------")
    print("EXAMPLE:", i)
    print("-----------------------------")

    # Question-level metadata (not part of the searchable source text).
    print("ID:", example["id"])
    print("Question:", example["question"])
    print("Answer:", example["answer"])
    print("Type:", example["type"])
    print("Level:", example["level"])

    # Check that titles and sentence lists match
    # Both lists are parallel, so they must have the same length. If not,
    # position j would not refer to the same passage in both.
    print("Titles match passages:",
          len(titles) == len(sentences))

    # Count passages
    # Number of passages provided for this single question.
    print("Number of passages:", len(titles))

    # Check for repeated titles
    # Titles seen in THIS example only; reset for every question.
    titles_in_example = set()

    # Go through the passages by position so title j and passage j stay paired.
    for j in range(len(titles)):
        title = titles[j]
        passage = sentences[j]

        print("\nTitle:", title)
        # How many sentences this passage has.
        print("Sentence count:", len(passage))

        # Check empty passages
        # Empty = no sentences at all, or every sentence is only whitespace.
        if not passage or not any(s.strip() for s in passage):
            print("EMPTY PASSAGE FOUND!")

        # Check individual empty sentences
        # sentence_id is the original zero-based position; it is the same
        # number that supporting_facts["sent_id"] refers to.
        for sentence_id, sentence in enumerate(passage):
            if not sentence.strip():
                print("Empty sentence at:", sentence_id)

        # Detect duplicates within this example
        # Same title twice in one question's context.
        if title in titles_in_example:
            print("Repeated title in this example:", title)

        titles_in_example.add(title)

        # Detect duplicates across examples
        # Same title already appeared in an earlier question. This is
        # expected (distractors are reused) and is not an error by itself.
        if title in seen_titles:
            print("Title already seen in another example:", title)

        seen_titles.add(title)

    # ========================================
    # PART 2: Resolve supporting facts
    # ========================================

    # Evidence labels: parallel lists "title" and "sent_id". Each pair says
    # "sentence number sent_id of the passage titled title is needed".
    facts = example["supporting_facts"]

    print("\nSUPPORTING FACTS:")

    # Walk the labels pair by pair.
    for fact_title, sentence_id in zip(
        facts["title"], facts["sent_id"]
    ):
        # Find every passage in this example with the referenced title.
        # A list is used because a title could (in theory) appear twice.
        matching_positions = [
            j for j, title in enumerate(titles)
            if title == fact_title
        ]

        # The label points to a title that is not in the context at all.
        # Report it instead of silently skipping it.
        if not matching_positions:
            print("Source not found:", fact_title)
            continue

        for position in matching_positions:
            source_sentences = sentences[position]

            # The sentence number must exist inside that passage.
            if 0 <= sentence_id < len(source_sentences):
                # Valid reference: show the actual evidence sentence.
                print("Source:", fact_title)
                print("Sentence ID:", sentence_id)
                print("Evidence:", source_sentences[sentence_id])
            else:
                # The number is outside the passage: a broken reference.
                print("Invalid sentence ID:", sentence_id)

# Distinct titles over the first 10 examples (titles, not content).
print("\nUnique titles in first 10 examples:",
      len(seen_titles))


# ========================================
# PART 3: Count unique source passages
# ========================================

# Set of SHA-256 fingerprints of non-empty passages. A set stores each
# fingerprint once, so its size is the number of unique passages.
unique_hashes = set()
# Every passage in every example, including repeats.
total_passages = 0
# Passages that contain no real text.
empty_passages = 0

# Full pass over all training questions (not just the first 10).
for example in train:
    context = example["context"]

    # Only the passage text is needed here; titles are not used for counting.
    for passage in context["sentences"]:
        total_passages += 1

        # Join sentences into a complete passage
        # Turn the list of sentences into one string.
        text = " ".join(passage)

        # Normalize whitespace
        # split() with no argument drops leading/trailing space and collapses
        # runs of spaces/newlines, so trivial spacing differences do not count
        # as different content. Case and punctuation are NOT changed.
        normalized_text = " ".join(text.split())

        # Nothing left after normalizing: count it as empty and skip hashing.
        if not normalized_text:
            empty_passages += 1
            continue

        # Create an exact-content fingerprint
        # Identical normalized text always gives the same hash, so repeated
        # passages collapse into one entry in the set.
        content_hash = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()

        unique_hashes.add(content_hash)

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

# Questions in the training split.
print("Training examples:", len(train))
# Passages counted with repeats (one per appearance in a question's context).
print("Total passage occurrences:", total_passages)
# Passages with no real text.
print("Empty passages:", empty_passages)
# Distinct non-empty passages by exact normalized text. This is the number
# to compare with the assignment's 50,000 independent records.
print("Unique non-empty passages:", len(unique_hashes))

# Quick pass/fail against the 50,000 target. This only shows that enough
# unique passages EXIST in the data; it does not prove they were processed.
if len(unique_hashes) >= 50000:
    print("Enough unique passage candidates for 50,000 records!")
else:
    print("Fewer than 50,000 unique passages found.")
