"""Okapi BM25 keyword index, implemented by us (custom algorithm #1).

For a query q with terms t and a chunk d:

    score(d, q) = sum over t in q of  IDF(t) * tf(t, d) * (k1 + 1)
                                      -----------------------------------------
                                      tf(t, d) + k1 * (1 - b + b * |d| / avgdl)

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

- tf(t, d)  how often t occurs in d
- df(t)     how many chunks contain t
- N         number of chunks; |d| chunk length in tokens; avgdl average chunk length
- k1 = 1.5  how fast repeated terms stop adding score (term-frequency saturation)
- b  = 0.75 how strongly long chunks are penalised (length normalisation)

The "+ 1" inside the log keeps IDF positive even for very common terms. A term repeated in the
query counts once per repetition, as in the standard formula.

The index is an inverted index (term -> {chunk_id: tf}), so a query only touches chunks that
contain at least one query term. It supports removal, which document update and delete need.

Two implementations with identical scores:
- ``BM25Index``: plain dictionaries. Simple to read; about 365 MB for our 130,854 chunks.
- ``CompactBM25Index``: what the server uses. The bulk of the corpus is stored as NumPy arrays
  ("posting lists" sorted by term, the layout search engines use), about 60 MB. Chunks added
  later go into a small ``BM25Index`` (the "delta"); deleted chunks are marked dead with a
  tombstone instead of rewriting the arrays. Search combines both, so the results are the same as
  if everything were in one index.
"""

import heapq
import math
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_len: dict[str, int] = {}
        self.doc_terms: dict[str, list[str]] = {}  # lets remove() touch only this chunk's terms
        self.total_len = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self.doc_len

    def add(self, chunk_id: str, text: str) -> None:
        """Index one chunk. Re-adding an existing id replaces it."""
        if chunk_id in self.doc_len:
            self.remove(chunk_id)
        tokens = tokenize(text)
        counts = Counter(tokens)
        for term, tf in counts.items():
            self.postings[term][chunk_id] = tf
        self.doc_terms[chunk_id] = list(counts)
        self.doc_len[chunk_id] = len(tokens)
        self.total_len += len(tokens)

    def remove(self, chunk_id: str) -> None:
        """Forget a chunk. Unknown ids are ignored, so deletes are safe to repeat."""
        length = self.doc_len.pop(chunk_id, None)
        if length is None:
            return
        self.total_len -= length
        for term in self.doc_terms.pop(chunk_id):
            docs = self.postings[term]
            del docs[chunk_id]
            if not docs:
                del self.postings[term]

    def idf(self, term: str) -> float:
        n = len(self.doc_len)
        df = len(self.postings.get(term, ()))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score) pairs, best first. Ties are broken by chunk_id for repeatability."""
        if not self.doc_len:
            return []
        avgdl = self.total_len / len(self.doc_len)
        scores: dict[str, float] = defaultdict(float)
        for term in tokenize(query):
            docs = self.postings.get(term)
            if not docs:
                continue
            idf = self.idf(term)
            for chunk_id, tf in docs.items():
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[chunk_id] / avgdl)
                scores[chunk_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))


class CompactBM25Index:
    """BM25 with the same scores as ``BM25Index``, stored compactly (see the module docstring).

    Layout of the bulk-built part, for the term with id t:
        post_doc[offsets[t] : offsets[t + 1]]   rows of the chunks containing t
        post_tf [offsets[t] : offsets[t + 1]]   how often t occurs in each of them
    A row is a chunk's position in ``ids``; ``lens[row]`` is its token count and ``alive[row]``
    becomes False when it is removed.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.offsets = np.zeros(1, dtype=np.int64)
        self.post_doc = np.zeros(0, dtype=np.int32)
        self.post_tf = np.zeros(0, dtype=np.uint16)
        self.ids: list[str] = []
        self.rows: dict[str, int] = {}
        self.lens = np.zeros(0, dtype=np.int32)
        self.alive = np.zeros(0, dtype=bool)
        self.main_count = 0  # live chunks in the arrays
        self.main_len = 0  # their total token count
        self.delta = BM25Index(k1, b)  # chunks added after the build

    @classmethod
    def build(cls, items: Iterable[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> "CompactBM25Index":
        """Index (chunk_id, text) pairs in one pass, then sort the postings by term id."""
        index = cls(k1, b)
        term_ids, rows, tfs, lens = array("i"), array("i"), array("H"), array("i")
        for chunk_id, text in items:
            if chunk_id in index.rows:
                continue
            row = len(index.ids)
            index.ids.append(chunk_id)
            index.rows[chunk_id] = row
            tokens = tokenize(text)
            lens.append(len(tokens))
            for term, tf in Counter(tokens).items():
                term_ids.append(index.vocab.setdefault(term, len(index.vocab)))
                rows.append(row)
                tfs.append(min(tf, 65535))
        t = np.frombuffer(term_ids, dtype=np.int32) if term_ids else np.zeros(0, dtype=np.int32)
        order = np.argsort(t, kind="stable")
        index.post_doc = (np.frombuffer(rows, dtype=np.int32) if rows else np.zeros(0, np.int32))[order]
        index.post_tf = (np.frombuffer(tfs, dtype=np.uint16) if tfs else np.zeros(0, np.uint16))[order]
        counts = np.bincount(t, minlength=len(index.vocab))
        index.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        index.lens = np.array(lens, dtype=np.int32)
        index.alive = np.ones(len(index.ids), dtype=bool)
        index.main_count = len(index.ids)
        index.main_len = int(index.lens.sum())
        return index

    @property
    def total_len(self) -> int:
        return self.main_len + self.delta.total_len

    def __len__(self) -> int:
        return self.main_count + len(self.delta)

    def __contains__(self, chunk_id: str) -> bool:
        row = self.rows.get(chunk_id)
        return (row is not None and bool(self.alive[row])) or chunk_id in self.delta

    def add(self, chunk_id: str, text: str) -> None:
        if chunk_id in self:
            self.remove(chunk_id)
        self.delta.add(chunk_id, text)

    def remove(self, chunk_id: str) -> None:
        row = self.rows.get(chunk_id)
        if row is not None and self.alive[row]:
            self.alive[row] = False  # tombstone; the arrays are not rewritten
            self.main_count -= 1
            self.main_len -= int(self.lens[row])
        else:
            self.delta.remove(chunk_id)

    def _main_postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        tid = self.vocab.get(term)
        if tid is None:
            return self.post_doc[:0], self.post_tf[:0]
        start, end = self.offsets[tid], self.offsets[tid + 1]
        rows, tfs = self.post_doc[start:end], self.post_tf[start:end]
        live = self.alive[rows]
        return rows[live], tfs[live]

    def idf(self, term: str) -> float:
        df = len(self._main_postings(term)[0]) + len(self.delta.postings.get(term, ()))
        return math.log((len(self) - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        n = len(self)
        if not n:
            return []
        avgdl = self.total_len / n
        main_scores = np.zeros(len(self.ids), dtype=np.float64)
        delta_scores: dict[str, float] = defaultdict(float)
        for term in tokenize(query):
            rows, tfs = self._main_postings(term)
            delta_docs = self.delta.postings.get(term, {})
            if not len(rows) and not delta_docs:
                continue
            idf = self.idf(term)
            if len(rows):
                tf = tfs.astype(np.float64)
                norm = self.k1 * (1 - self.b + self.b * self.lens[rows] / avgdl)
                main_scores[rows] += idf * tf * (self.k1 + 1) / (tf + norm)
            for chunk_id, tf in delta_docs.items():
                norm = self.k1 * (1 - self.b + self.b * self.delta.doc_len[chunk_id] / avgdl)
                delta_scores[chunk_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        hits = np.nonzero(main_scores)[0]
        candidates = [(self.ids[r], float(main_scores[r])) for r in hits]
        candidates.extend(delta_scores.items())
        return heapq.nsmallest(k, candidates, key=lambda item: (-item[1], item[0]))
