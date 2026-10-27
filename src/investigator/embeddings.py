"""Embedding models behind one small interface: ``dim`` and ``embed(texts) -> vectors``.

Backends: ``fastembed`` (ONNX, default, light) and ``sentence-transformers`` (PyTorch, used for
the original bulk load and kept for comparison); both give the same MiniLM vectors.

Vectors are L2-normalised, so the dot product equals cosine similarity.
``HashEmbedder`` is a deterministic, dependency-free stand-in for tests (no model download); it is
a bag-of-words hashing vector, so it only matches shared words and is NOT semantic.
"""

import hashlib
import math

from .text import tokenize


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.tolist()


class FastEmbedEmbedder:
    """The same MiniLM model run with ONNX Runtime instead of PyTorch (``fastembed``).

    Produces the same vectors as ``SentenceTransformerEmbedder`` (cosine 1.00000 on 202 sample
    texts) in about 160 MB instead of about 480 MB, which is what lets the app fit a 512 MB
    free-tier server. Vectors already stored in Qdrant stay valid.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", threads: int | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name, threads=threads)
        self.dim = len(next(iter(self.model.embed(["dimension probe"]))))

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out = []
        for vec in self.model.embed(texts, batch_size=batch_size):
            norm = math.sqrt(float((vec * vec).sum())) or 1.0
            out.append((vec / norm).tolist())
        return out


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.model_name = f"hash-{dim}"
        self.dim = dim

    def embed(self, texts: list[str], batch_size: int = 0) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
