import pytest
from qdrant_client import QdrantClient

from investigator.chunking import FixedWordChunker, SentenceChunker
from investigator.embeddings import HashEmbedder
from investigator.ingestion import DocumentService
from investigator.models import ParsedDocument, Sentence
from investigator.registry import Registry
from investigator.retrieval import SearchIndex
from investigator.vector_store import QdrantVectorStore


@pytest.fixture
def stack(tmp_path):
    """A full ingestion + search stack on a temporary SQLite file and in-memory Qdrant."""
    registry = Registry(f"sqlite:///{tmp_path / 'registry.db'}")
    embedder = HashEmbedder()
    vectors = QdrantVectorStore(QdrantClient(":memory:"), dim=embedder.dim)
    index = SearchIndex(registry, vectors, embedder)
    service = DocumentService(registry, index, {"fixed": FixedWordChunker(8, 2), "sentence": SentenceChunker(8)})
    return registry, index, service


def parsed(title: str, *sentences: str) -> ParsedDocument:
    return ParsedDocument(
        title=title, sentences=[Sentence(i, s) for i, s in enumerate(sentences)], format="txt"
    )
