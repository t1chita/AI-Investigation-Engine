"""Settings read from environment variables (and a local ``.env`` file). No secrets in code."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _strategies() -> list[str]:
    raw = os.getenv("INDEX_STRATEGIES", "fixed,sentence")
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///data/registry.db"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""), repr=False)
    qdrant_path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", "data/qdrant"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "fastembed"))
    chunk_strategy: str = field(default_factory=lambda: os.getenv("CHUNK_STRATEGY", "sentence"))
    index_strategies: list[str] = field(default_factory=_strategies)
    # Build the keyword index in a child process at start-up (saves ~100 MB on small servers).
    bm25_build_subprocess: bool = field(default_factory=lambda: os.getenv("BM25_BUILD_SUBPROCESS", "0") == "1")
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "fake"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""), repr=False)


def normalize_database_url(url: str) -> str:
    """Neon and most hosts hand out ``postgresql://`` URLs; SQLAlchemy would then look for the
    psycopg2 driver. We install psycopg 3, so name it explicitly."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def load_settings() -> Settings:
    load_dotenv()
    settings = Settings()
    settings.database_url = normalize_database_url(settings.database_url)
    return settings
