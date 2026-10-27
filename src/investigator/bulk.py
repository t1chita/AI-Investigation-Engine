"""Bulk ingestion for the 50k+ corpus, with batching, resume, dedup and a reconciliation report.

Resumable: documents already ``indexed`` or ``duplicate`` are skipped; documents left ``indexing``
or ``failed`` by an interrupted run are cleaned up and processed again. Re-running the command is
therefore safe and only does the missing work.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from .models import ParsedDocument, SourceDocument, Status
from .retrieval import SearchIndex


@dataclass
class BulkReport:
    seen: int = 0
    skipped_done: int = 0
    indexed: int = 0
    duplicates: int = 0
    retried: int = 0
    failed_batches: int = 0
    chunks: dict = field(default_factory=dict)
    seconds: float = 0.0


def ingest(
    components,
    records: Iterable[tuple[str, ParsedDocument]],
    source_type: str = "hotpotqa",
    batch_size: int = 512,
    limit: int | None = None,
    progress=print,
) -> BulkReport:
    registry, index, chunkers = components.registry, components.index, components.service.chunkers
    report = BulkReport(chunks={name: 0 for name in chunkers})
    hashes = registry.live_hashes()
    start = time.time()
    batch: list[tuple[str, ParsedDocument]] = []

    def run(batch):
        done = registry.statuses(doc_id for doc_id, _ in batch)
        todo = []
        for doc_id, parsed in batch:
            status = done.get(doc_id)
            if status in (Status.INDEXED, Status.DUPLICATE):
                report.skipped_done += 1
                continue
            if status is not None:  # left behind by an interrupted run
                _forget(components, doc_id)
                hashes.pop(next((h for h, d in hashes.items() if d == doc_id), None), None)
                report.retried += 1
            todo.append((doc_id, parsed))
        if todo:
            try:
                _ingest_batch(registry, index, chunkers, todo, source_type, hashes, report)
            except Exception as exc:  # keep going; the failed docs are retried on the next run
                report.failed_batches += 1
                progress(f"batch failed ({exc}); it will be retried on the next run")
        progress(
            f"{report.seen:>6} seen | {report.indexed} indexed, {report.duplicates} duplicates, "
            f"{report.skipped_done} already done | {time.time() - start:.0f}s"
        )

    for doc_id, parsed in records:
        if limit is not None and report.seen >= limit:
            break
        report.seen += 1
        batch.append((doc_id, parsed))
        if len(batch) >= batch_size:
            run(batch)
            batch = []
    if batch:
        run(batch)
    report.seconds = time.time() - start
    return report


def _ingest_batch(registry, index: SearchIndex, chunkers, todo, source_type, hashes, report):
    unique: list[SourceDocument] = []
    dups: list[SourceDocument] = []
    for doc_id, parsed in todo:
        doc = SourceDocument(
            doc_id=doc_id,
            title=parsed.title,
            sentences=parsed.sentences,
            format=parsed.format,
            source_type=source_type,
            content_hash=parsed.content_hash,
            metadata=parsed.metadata,
        )
        original = hashes.get(doc.content_hash)
        if original:
            doc.status, doc.duplicate_of = Status.DUPLICATE, original
            dups.append(doc)
        else:
            hashes[doc.content_hash] = doc_id
            unique.append(doc)
    registry.add_documents(unique + dups)  # unique ones start as "indexing"
    try:
        for name, chunker in chunkers.items():
            chunks = [c for doc in unique for c in chunker.chunk(doc)]
            registry.add_chunks(chunks)
            index.add(name, chunks)
            report.chunks[name] += len(chunks)
    except Exception:
        registry.set_status_many([d.doc_id for d in unique], Status.FAILED)
        raise
    registry.set_status_many([d.doc_id for d in unique], Status.INDEXED)
    report.indexed += len(unique)
    report.duplicates += len(dups)


def _forget(components, doc_id: str) -> None:
    chunk_ids = components.registry.delete_chunks(doc_id)
    for name in components.service.chunkers:
        components.index.remove(name, doc_id, [c for c in chunk_ids if f":{name}:" in c])
    components.registry.delete_document(doc_id)


def reconcile(components) -> dict:
    """Compare what each store holds. Every number pair must match for a consistent index."""
    reg = components.registry
    out = {
        "documents_total": reg.count_documents(),
        "documents_indexed": reg.count_documents(status=Status.INDEXED.value),
        "documents_duplicate": reg.count_documents(status=Status.DUPLICATE.value),
        "documents_failed": reg.count_documents(status=Status.FAILED.value),
        "documents_indexing": reg.count_documents(status=Status.INDEXING.value),
    }
    for name in components.service.chunkers:
        out[f"chunks_{name}_registry"] = reg.count_chunks(name)
        out[f"chunks_{name}_qdrant"] = components.vectors.count(name)
    return out
