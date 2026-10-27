"""Search over indexed chunks: semantic (Qdrant), keyword (our BM25) and hybrid (our RRF).

The BM25 index lives in memory (compact NumPy arrays) and is rebuilt from the registry on start-up
or first use for each strategy, so it can never drift from the durable data. Vector search uses
Qdrant, which persists itself.
"""

import threading
from dataclasses import dataclass

from .bm25 import CompactBM25Index
from .fusion import reciprocal_rank_fusion
from .models import Chunk
from .registry import Registry
from .vector_store import QdrantRESTStore, QdrantVectorStore

MODES = ("semantic", "keyword", "hybrid")


def _build_bm25(registry: Registry, strategy: str) -> CompactBM25Index:
    return CompactBM25Index.build((c.chunk_id, c.search_text) for c in registry.iter_chunks(strategy))


def _build_bm25_from_url(database_url: str, strategy: str) -> CompactBM25Index:
    """Entry point for the child process (it opens its own database connection)."""
    return _build_bm25(Registry(database_url), strategy)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    semantic_rank: int | None = None  # 1-based rank in the semantic list, if present
    keyword_rank: int | None = None  # 1-based rank in the BM25 list, if present


class SearchIndex:
    def __init__(self, registry: Registry, vectors: QdrantVectorStore | QdrantRESTStore, embedder, candidates: int = 50):
        self.registry = registry
        self.vectors = vectors
        self.embedder = embedder
        self.candidates = candidates  # how many results each method contributes before fusion
        self._bm25: dict[str, CompactBM25Index] = {}
        self._lock = threading.RLock()  # the in-memory BM25 index is not thread-safe

    # ---- writing ----------------------------------------------------------------------------
    def bm25(self, strategy: str) -> CompactBM25Index:
        with self._lock:
            if strategy not in self._bm25:
                self._bm25[strategy] = _build_bm25(self.registry, strategy)
            return self._bm25[strategy]

    def warm_bm25(self, strategy: str, in_subprocess: bool = False) -> None:
        """Build the keyword index before the first request.

        ``in_subprocess=True`` builds it in a short-lived child process and receives only the
        finished arrays. Python keeps the memory of the temporary build objects (tokens, counters)
        until the process ends, so building in a child saves about 100 MB on a small server.
        """
        if not in_subprocess:
            self.bm25(strategy)
            return
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        url = self.registry.engine.url.render_as_string(hide_password=False)
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            index = pool.submit(_build_bm25_from_url, url, strategy).result()
        with self._lock:
            self._bm25[strategy] = index

    def add(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        """Index chunks in both stores. Chunks must already be saved in the registry."""
        if not chunks:
            return
        if vectors is None:
            vectors = self.embedder.embed([c.search_text for c in chunks])
        self.vectors.upsert(strategy, chunks, vectors)
        with self._lock:
            if strategy in self._bm25:  # an unbuilt index will read these from the registry later
                for c in chunks:
                    self._bm25[strategy].add(c.chunk_id, c.search_text)

    def remove(self, strategy: str, doc_id: str, chunk_ids: list[str]) -> None:
        self.vectors.delete_doc(strategy, doc_id)
        with self._lock:
            if strategy in self._bm25:
                for chunk_id in chunk_ids:
                    self._bm25[strategy].remove(chunk_id)

    # ---- reading ----------------------------------------------------------------------------
    def semantic_ids(self, query: str, strategy: str, k: int) -> list[tuple[str, float]]:
        vector = self.embedder.embed([query])[0]
        return self.vectors.search(strategy, vector, k)

    def keyword_ids(self, query: str, strategy: str, k: int) -> list[tuple[str, float]]:
        index = self.bm25(strategy)
        with self._lock:
            return index.search(query, k)

    def search(self, query: str, strategy: str, mode: str = "hybrid", k: int = 10) -> list[SearchHit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not query.strip():
            return []
        semantic = self.semantic_ids(query, strategy, self.candidates) if mode != "keyword" else []
        keyword = self.keyword_ids(query, strategy, self.candidates) if mode != "semantic" else []
        sem_rank = {cid: r for r, (cid, _) in enumerate(semantic, start=1)}
        kw_rank = {cid: r for r, (cid, _) in enumerate(keyword, start=1)}

        if mode == "semantic":
            ranked = semantic[:k]
        elif mode == "keyword":
            ranked = keyword[:k]
        else:
            ranked = reciprocal_rank_fusion([[c for c, _ in semantic], [c for c, _ in keyword]])[:k]

        found = self.registry.get_chunks(cid for cid, _ in ranked)
        # A vector whose chunk is no longer in the registry is stale; skip it rather than cite it.
        return [
            SearchHit(found[cid], score, sem_rank.get(cid), kw_rank.get(cid)) for cid, score in ranked if cid in found
        ]
