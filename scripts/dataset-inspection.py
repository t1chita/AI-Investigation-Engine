
from datasets import load_dataset
import hashlib

# Load the HotpotQA dataset
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

train = dataset["train"]

print("Total training examples:", len(train))
print("Available fields:", train.column_names)


# ========================================
# PART 1: Inspect the first 10 examples
# ========================================

seen_titles = set()

for i in range(10):
    example = train[i]

    context = example["context"]
    titles = context["title"]
    sentences = context["sentences"]

    print("\n-----------------------------")
    print("EXAMPLE:", i)
    print("-----------------------------")

    print("ID:", example["id"])
    print("Question:", example["question"])
    print("Answer:", example["answer"])
    print("Type:", example["type"])
    print("Level:", example["level"])

    # Check that titles and sentence lists match
    print("Titles match passages:",
          len(titles) == len(sentences))

    # Count passages
    print("Number of passages:", len(titles))

    # Check for repeated titles
    titles_in_example = set()

    for j in range(len(titles)):
        title = titles[j]
        passage = sentences[j]

        print("\nTitle:", title)
        print("Sentence count:", len(passage))

        # Check empty passages
        if not passage or not any(s.strip() for s in passage):
            print("EMPTY PASSAGE FOUND!")

        # Check individual empty sentences
        for sentence_id, sentence in enumerate(passage):
            if not sentence.strip():
                print("Empty sentence at:", sentence_id)

        # Detect duplicates within this example
        if title in titles_in_example:
            print("Repeated title in this example:", title)

        titles_in_example.add(title)

        # Detect duplicates across examples
        if title in seen_titles:
            print("Title already seen in another example:", title)

        seen_titles.add(title)

    # ========================================
    # PART 2: Resolve supporting facts
    # ========================================

    facts = example["supporting_facts"]

    print("\nSUPPORTING FACTS:")

    for fact_title, sentence_id in zip(
        facts["title"], facts["sent_id"]
    ):
        matching_positions = [
            j for j, title in enumerate(titles)
            if title == fact_title
        ]

        if not matching_positions:
            print("Source not found:", fact_title)
            continue

        for position in matching_positions:
            source_sentences = sentences[position]

            if 0 <= sentence_id < len(source_sentences):
                print("Source:", fact_title)
                print("Sentence ID:", sentence_id)
                print("Evidence:", source_sentences[sentence_id])
            else:
                print("Invalid sentence ID:", sentence_id)

print("\nUnique titles in first 10 examples:",
      len(seen_titles))


# ========================================
# PART 3: Count unique source passages
# ========================================

unique_hashes = set()
total_passages = 0
empty_passages = 0

for example in train:
    context = example["context"]

    for passage in context["sentences"]:
        total_passages += 1

        # Join sentences into a complete passage
        text = " ".join(passage)

        # Normalize whitespace
        normalized_text = " ".join(text.split())

        if not normalized_text:
            empty_passages += 1
            continue

        # Create an exact-content fingerprint
        content_hash = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()

        unique_hashes.add(content_hash)

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print("Training examples:", len(train))
print("Total passage occurrences:", total_passages)
print("Empty passages:", empty_passages)
print("Unique non-empty passages:", len(unique_hashes))

if len(unique_hashes) >= 50000:
    print("Enough unique passage candidates for 50,000 records!")
else:
    print("Fewer than 50,000 unique passages found.")
