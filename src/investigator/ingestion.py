"""Document lifecycle: add, update, delete, duplicate handling and recovery.

Order of writes, so a crash never leaves searchable garbage:
1. the registry row is written with status ``indexing``;
2. chunks are written to the registry, then to Qdrant and BM25;
3. only then the status becomes ``indexed``.
If step 2 fails, the document is marked ``failed`` and its partial chunks and vectors are removed;
``retry_failed()`` (or re-running the bulk ingest) indexes it again.

Duplicates: a document whose normalised text has the same SHA-256 as a live document is stored
with status ``duplicate`` and ``duplicate_of`` pointing to the original. It is not indexed again, so
search returns the evidence once, but its title and metadata are kept. If the original is deleted,
the oldest duplicate is promoted and indexed in its place.
"""

import uuid

from .models import ParsedDocument, SourceDocument, Status
from .registry import Registry
from .retrieval import SearchIndex


class DocumentNotFound(KeyError):
    pass


class DocumentExists(ValueError):
    pass


class IndexingError(RuntimeError):
    pass


class DocumentService:
    def __init__(self, registry: Registry, index: SearchIndex, chunkers: dict):
        self.registry = registry
        self.index = index
        self.chunkers = chunkers  # {"fixed": FixedWordChunker(), "sentence": SentenceChunker()}

    # ---- public operations ------------------------------------------------------------------
    def add(self, parsed: ParsedDocument, source_type: str = "upload", doc_id: str | None = None) -> SourceDocument:
        doc_id = doc_id or f"up-{uuid.uuid4().hex[:12]}"
        if self.registry.get(doc_id):
            raise DocumentExists(doc_id)
        doc = SourceDocument(
            doc_id=doc_id,
            title=parsed.title,
            sentences=parsed.sentences,
            format=parsed.format,
            source_type=source_type,
            content_hash=parsed.content_hash,
            metadata=parsed.metadata,
        )
        original = self.registry.find_live_by_hash(doc.content_hash)
        if original:
            doc.status, doc.duplicate_of = Status.DUPLICATE, original.doc_id
            self.registry.add_document(doc)
            return doc
        self.registry.add_document(doc)
        self._index(doc)
        return doc

    def update(self, doc_id: str, parsed: ParsedDocument) -> SourceDocument:
        old = self._require(doc_id)
        if parsed.content_hash == old.content_hash and parsed.title == old.title:
            return old  # nothing changed; keep the version
        new = SourceDocument(
            doc_id=doc_id,
            title=parsed.title,
            sentences=parsed.sentences,
            format=parsed.format,
            source_type=old.source_type,
            content_hash=parsed.content_hash,
            version=old.version + 1,
            metadata={**old.metadata, **parsed.metadata},
        )
        self._unindex(doc_id)
        original = self.registry.find_live_by_hash(new.content_hash, exclude=doc_id)
        if original:
            new.status, new.duplicate_of = Status.DUPLICATE, original.doc_id
            self.registry.replace_document(new)
            self._promote_duplicates_of(doc_id)
            return new
        new.status = Status.INDEXING
        self.registry.replace_document(new)
        self._index(new)
        # Documents that duplicated the OLD content are now its only holders: promote one.
        self._promote_duplicates_of(doc_id)
        return new

    def delete(self, doc_id: str) -> None:
        self._require(doc_id)
        self._unindex(doc_id)
        self.registry.delete_document(doc_id)
        self._promote_duplicates_of(doc_id)

    def retry_failed(self) -> int:
        """Re-index every document left in ``failed`` or ``indexing`` (e.g. after a crash)."""
        n = 0
        for status in (Status.FAILED, Status.INDEXING):
            for doc_id in self.registry.doc_ids_with_status(status):
                doc = self.registry.get(doc_id)
                self._unindex(doc_id)
                self._index(doc)
                n += 1
        return n

    # ---- internals --------------------------------------------------------------------------
    def _require(self, doc_id: str) -> SourceDocument:
        doc = self.registry.get(doc_id)
        if not doc:
            raise DocumentNotFound(doc_id)
        return doc

    def _index(self, doc: SourceDocument) -> None:
        try:
            for strategy, chunker in self.chunkers.items():
                chunks = chunker.chunk(doc)
                self.registry.add_chunks(chunks)
                self.index.add(strategy, chunks)
        except Exception as exc:
            self._unindex(doc.doc_id)
            self.registry.set_status(doc.doc_id, Status.FAILED)
            doc.status = Status.FAILED
            raise IndexingError(f"indexing {doc.doc_id} failed: {exc}") from exc
        self.registry.set_status(doc.doc_id, Status.INDEXED)
        doc.status = Status.INDEXED

    def _unindex(self, doc_id: str) -> None:
        chunk_ids = self.registry.delete_chunks(doc_id)
        for strategy in self.chunkers:
            self.index.remove(strategy, doc_id, [c for c in chunk_ids if f":{strategy}:" in c])

    def _promote_duplicates_of(self, doc_id: str) -> None:
        dups = self.registry.duplicates_of(doc_id)
        if not dups:
            return
        heir, rest = dups[0], dups[1:]
        self.registry.set_status(heir.doc_id, Status.INDEXING)
        for d in rest:
            self.registry.set_status(d.doc_id, Status.DUPLICATE, duplicate_of=heir.doc_id)
        self._index(heir)
