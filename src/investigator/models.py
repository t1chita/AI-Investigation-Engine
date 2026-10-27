"""Core data models. Plain dataclasses so every field is visible and easy to explain.

Identity rules:
- ``doc_id`` is stable and never derived from a filename alone.
- ``version`` starts at 1 and increments on every content change.
- ``chunk_id`` = ``{doc_id}:v{version}:{strategy}:{position}``, so a chunk always says which
  document version and chunking strategy produced it.
"""

from dataclasses import dataclass, field
from enum import Enum

from .text import content_hash


class Status(str, Enum):
    """Processing state of a document in the registry."""

    INDEXING = "indexing"  # written to the registry, indexes being built
    INDEXED = "indexed"  # searchable
    DUPLICATE = "duplicate"  # same content as another live document; not indexed again
    FAILED = "failed"  # indexing failed; safe to retry


@dataclass
class Sentence:
    """One sentence of a source, with its original zero-based position and optional page."""

    index: int
    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    """What a parser returns: text split into sentences plus metadata, before identity is assigned."""

    title: str
    sentences: list[Sentence]
    format: str  # "txt", "json", "pdf", "hotpotqa"
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    @property
    def content_hash(self) -> str:
        return content_hash(self.text)


@dataclass
class SourceDocument:
    """A registered document: a parsed document plus identity, version and status."""

    doc_id: str
    title: str
    sentences: list[Sentence]
    format: str
    source_type: str  # "hotpotqa" or "upload"
    content_hash: str
    version: int = 1
    status: Status = Status.INDEXING
    duplicate_of: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


@dataclass
class Chunk:
    """A searchable piece of one document version, keeping its sentence span for citations."""

    chunk_id: str
    doc_id: str
    version: int
    strategy: str
    position: int
    title: str
    text: str
    sentence_start: int  # first sentence index covered (inclusive)
    sentence_end: int  # last sentence index covered (inclusive)
    page: int | None = None

    @property
    def search_text(self) -> str:
        """Text used for embeddings and BM25: the title gives short passages their subject."""
        return f"{self.title}. {self.text}"


def make_chunk_id(doc_id: str, version: int, strategy: str, position: int) -> str:
    return f"{doc_id}:v{version}:{strategy}:{position}"
