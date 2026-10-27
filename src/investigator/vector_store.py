"""Qdrant vector store: one collection per chunking strategy (``chunks_fixed``, ``chunks_sentence``).

Each point stores the chunk vector plus a small payload (chunk_id, doc_id, version). The chunk text
itself lives in the registry; Qdrant only finds candidates. Point ids must be integers or UUIDs,
so we derive a UUID from the chunk_id (uuid5 is deterministic, so re-upserting overwrites).

Two implementations of the same five methods:
- ``QdrantVectorStore`` uses the official ``qdrant_client``; needed for local (on-disk or
  in-memory) Qdrant in development and tests. Importing it costs about 78 MB of memory.
- ``QdrantRESTStore`` talks to a Qdrant server (Qdrant Cloud) with plain HTTP calls through
  ``httpx``. Used by the deployed app so it fits a 512 MB free-tier server.
"""

import uuid
import warnings

import httpx

from .models import Chunk


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    def __init__(self, client, dim: int, prefix: str = "chunks"):
        self.client = client  # a qdrant_client.QdrantClient
        self.dim = dim
        self.prefix = prefix
        self._ready: set[str] = set()

    def collection(self, strategy: str) -> str:
        return f"{self.prefix}_{strategy}"

    def ensure(self, strategy: str) -> str:
        from qdrant_client import models

        name = self.collection(strategy)
        if name in self._ready:
            return name
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            )
            with warnings.catch_warnings():  # local Qdrant ignores payload indexes and warns
                warnings.simplefilter("ignore", UserWarning)
                self.client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)
        else:
            size = self.client.get_collection(name).config.params.vectors.size
            if size != self.dim:
                raise ValueError(f"collection {name} has dimension {size}, embedder gives {self.dim}")
        self._ready.add(name)
        return name

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models

        if not chunks:
            return
        name = self.ensure(strategy)
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector=v,
                payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "version": c.version},
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(name, points=points, wait=True)

    def delete_doc(self, strategy: str, doc_id: str) -> None:
        """Remove every vector of a document, whatever version, by payload filter."""
        from qdrant_client import models

        name = self.ensure(strategy)
        selector = models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        )
        self.client.delete(name, points_selector=selector, wait=True)

    def search(self, strategy: str, vector: list[float], k: int = 10) -> list[tuple[str, float]]:
        name = self.ensure(strategy)
        result = self.client.query_points(name, query=vector, limit=k, with_payload=["chunk_id"])
        return [(p.payload["chunk_id"], float(p.score)) for p in result.points]

    def count(self, strategy: str) -> int:
        return self.client.count(self.ensure(strategy), exact=True).count


class QdrantRESTStore:
    """The same operations as ``QdrantVectorStore`` over Qdrant's HTTP API (no qdrant_client)."""

    def __init__(self, url: str, api_key: str, dim: int, prefix: str = "chunks", http: httpx.Client | None = None):
        headers = {"api-key": api_key} if api_key else {}
        self.http = http or httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=30)
        self.dim = dim
        self.prefix = prefix
        self._ready: set[str] = set()

    def collection(self, strategy: str) -> str:
        return f"{self.prefix}_{strategy}"

    def _call(self, method: str, path: str, json: dict | None = None) -> dict:
        response = self.http.request(method, path, json=json)
        if response.status_code >= 400:
            raise RuntimeError(f"qdrant {method} {path} failed ({response.status_code}): {response.text[:200]}")
        return response.json()

    def ensure(self, strategy: str) -> str:
        name = self.collection(strategy)
        if name in self._ready:
            return name
        exists = self._call("GET", f"/collections/{name}/exists")["result"]["exists"]
        if not exists:
            self._call("PUT", f"/collections/{name}", {"vectors": {"size": self.dim, "distance": "Cosine"}})
            self._call("PUT", f"/collections/{name}/index", {"field_name": "doc_id", "field_schema": "keyword"})
        else:
            info = self._call("GET", f"/collections/{name}")["result"]
            size = info["config"]["params"]["vectors"]["size"]
            if size != self.dim:
                raise ValueError(f"collection {name} has dimension {size}, embedder gives {self.dim}")
        self._ready.add(name)
        return name

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        name = self.ensure(strategy)
        points = [
            {"id": point_id(c.chunk_id), "vector": v,
             "payload": {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "version": c.version}}
            for c, v in zip(chunks, vectors)
        ]
        self._call("PUT", f"/collections/{name}/points?wait=true", {"points": points})

    def delete_doc(self, strategy: str, doc_id: str) -> None:
        name = self.ensure(strategy)
        body = {"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}}
        self._call("POST", f"/collections/{name}/points/delete?wait=true", body)

    def search(self, strategy: str, vector: list[float], k: int = 10) -> list[tuple[str, float]]:
        name = self.ensure(strategy)
        body = {"query": vector, "limit": k, "with_payload": ["chunk_id"]}
        points = self._call("POST", f"/collections/{name}/points/query", body)["result"]["points"]
        return [(p["payload"]["chunk_id"], float(p["score"])) for p in points]

    def count(self, strategy: str) -> int:
        name = self.ensure(strategy)
        return self._call("POST", f"/collections/{name}/points/count", {"exact": True})["result"]["count"]
