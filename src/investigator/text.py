"""Text helpers shared by ingestion and retrieval: normalisation, fingerprints, tokens, sentences."""

import hashlib
import re

# Words, plus dotted or dashed identifiers kept whole ("v2.18", "INC-4471", "2026-03-01"), because
# investigations often hinge on exact versions, ticket ids and dates.
_WORD = re.compile(r"\w+(?:[.\-]\w+)*", re.UNICODE)
# A sentence ends at . ! or ? followed by whitespace and an upper-case letter, digit or quote.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")

# Small English stop-word list for keyword search. Kept short on purpose: names, IDs and
# numbers matter in investigations, so only very common function words are dropped.
STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "which with what who whom when where why how did does do".split()
)


def normalize_text(text: str) -> str:
    """Collapse all runs of whitespace to one space and strip the ends. Case is kept."""
    return " ".join(text.split())


def content_hash(text: str) -> str:
    """SHA-256 hex digest of the normalised text: equal content gives an equal fingerprint."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens without stop words, used by BM25 for documents and queries."""
    return [t for t in (m.group().lower() for m in _WORD.finditer(text)) if t not in STOP_WORDS]


def split_sentences(text: str) -> list[str]:
    """Split plain text into sentences, keeping paragraph breaks as boundaries too."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = normalize_text(paragraph)
        if paragraph:
            sentences.extend(s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip())
    return sentences
