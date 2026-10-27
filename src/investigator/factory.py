"""Builds the registry, vector store, search index and document service from settings."""

from dataclasses import dataclass

from .chunking import get_chunker
from .config import Settings, load_settings
from .embeddings import FastEmbedEmbedder, HashEmbedder, SentenceTransformerEmbedder
from .ingestion import DocumentService
from .registry import Registry
from .retrieval import SearchIndex
from .vector_store import QdrantRESTStore, QdrantVectorStore


@dataclass
class Components:
    settings: Settings
    registry: Registry
    vectors: QdrantVectorStore | QdrantRESTStore
    index: SearchIndex
    service: DocumentService


def make_qdrant(settings: Settings):
    """A qdrant_client for scripts that need the full client (imported only when called)."""
    from qdrant_client import QdrantClient

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60)
    return QdrantClient(path=settings.qdrant_path)


def make_vector_store(settings: Settings, dim: int):
    """Qdrant server (cloud) over plain HTTP; local on-disk Qdrant through qdrant_client."""
    if settings.qdrant_url:
        return QdrantRESTStore(settings.qdrant_url, settings.qdrant_api_key, dim)
    return QdrantVectorStore(make_qdrant(settings), dim)


def make_embedder(settings: Settings):
    if settings.embedding_model.startswith("hash-"):
        return HashEmbedder(int(settings.embedding_model.split("-")[1]))
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    return FastEmbedEmbedder(settings.embedding_model)


def build(settings: Settings | None = None) -> Components:
    settings = settings or load_settings()
    registry = Registry(settings.database_url)
    embedder = make_embedder(settings)
    vectors = make_vector_store(settings, embedder.dim)
    index = SearchIndex(registry, vectors, embedder)
    chunkers = {name: get_chunker(name) for name in settings.index_strategies}
    service = DocumentService(registry, index, chunkers)
    return Components(settings, registry, vectors, index, service)
