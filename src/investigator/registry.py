"""Source registry: the durable record of documents, versions, statuses and chunks.

It is the single source of truth. The vector store and the BM25 index are derived from it and can
be rebuilt from it, which is how we keep the three stores consistent after updates and deletes.
Uses SQLAlchemy Core so the same code runs on SQLite (local, tests) and Postgres (deployment).
"""

from collections.abc import Iterable, Iterator
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Engine

from .models import Chunk, Sentence, SourceDocument, Status

metadata_obj = MetaData()

documents = Table(
    "documents",
    metadata_obj,
    Column("doc_id", String(200), primary_key=True),
    Column("title", Text, nullable=False),
    Column("format", String(20), nullable=False),
    Column("source_type", String(20), nullable=False),
    Column("content_hash", String(64), nullable=False, index=True),
    Column("version", Integer, nullable=False),
    Column("status", String(20), nullable=False, index=True),
    Column("duplicate_of", String(200), nullable=True, index=True),
    Column("meta", JSON, nullable=False),
    Column("sentences", JSON, nullable=False),  # [[index, text, page], ...]
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

chunks = Table(
    "chunks",
    metadata_obj,
    Column("chunk_id", String(300), primary_key=True),
    Column("doc_id", String(200), nullable=False, index=True),
    Column("version", Integer, nullable=False),
    Column("strategy", String(20), nullable=False, index=True),
    Column("position", Integer, nullable=False),
    Column("title", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("sentence_start", Integer, nullable=False),
    Column("sentence_end", Integer, nullable=False),
    Column("page", Integer, nullable=True),
)


# One row per investigation that called the LLM: what it cost, never what was asked (the live
# page is public, so questions are not stored).
llm_usage = Table(
    "llm_usage",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("model", String(100), nullable=False),
    Column("status", String(20), nullable=False),
    Column("mode", String(20), nullable=False),
    Column("calls", Integer, nullable=False),
    Column("input_tokens", Integer, nullable=False),
    Column("output_tokens", Integer, nullable=False),
    Column("seconds", Float, nullable=False),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _doc_row(doc: SourceDocument) -> dict:
    now = _now()
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "format": doc.format,
        "source_type": doc.source_type,
        "content_hash": doc.content_hash,
        "version": doc.version,
        "status": doc.status.value,
        "duplicate_of": doc.duplicate_of,
        "meta": doc.metadata,
        "sentences": [[s.index, s.text, s.page] for s in doc.sentences],
        "created_at": now,
        "updated_at": now,
    }


def _row_doc(row) -> SourceDocument:
    return SourceDocument(
        doc_id=row.doc_id,
        title=row.title,
        sentences=[Sentence(index=i, text=t, page=p) for i, t, p in row.sentences],
        format=row.format,
        source_type=row.source_type,
        content_hash=row.content_hash,
        version=row.version,
        status=Status(row.status),
        duplicate_of=row.duplicate_of,
        metadata=row.meta or {},
    )


def _chunk_row(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "version": c.version,
        "strategy": c.strategy,
        "position": c.position,
        "title": c.title,
        "text": c.text,
        "sentence_start": c.sentence_start,
        "sentence_end": c.sentence_end,
        "page": c.page,
    }


def _row_chunk(row) -> Chunk:
    return Chunk(
        chunk_id=row.chunk_id,
        doc_id=row.doc_id,
        version=row.version,
        strategy=row.strategy,
        position=row.position,
        title=row.title,
        text=row.text,
        sentence_start=row.sentence_start,
        sentence_end=row.sentence_end,
        page=row.page,
    )


class Registry:
    def __init__(self, url_or_engine: str | Engine):
        self.engine = create_engine(url_or_engine) if isinstance(url_or_engine, str) else url_or_engine
        metadata_obj.create_all(self.engine)

    # ---- documents ------------------------------------------------------------------------
    def add_document(self, doc: SourceDocument) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(documents), [_doc_row(doc)])

    def add_documents(self, docs: Iterable[SourceDocument], batch: int = 2000) -> int:
        """Bulk insert (used for the 50k corpus). Returns the number inserted."""
        n = 0
        rows: list[dict] = []
        with self.engine.begin() as conn:
            for doc in docs:
                rows.append(_doc_row(doc))
                if len(rows) >= batch:
                    conn.execute(insert(documents), rows)
                    n += len(rows)
                    rows = []
            if rows:
                conn.execute(insert(documents), rows)
                n += len(rows)
        return n

    def replace_document(self, doc: SourceDocument) -> None:
        """Overwrite content, version and status of an existing document (keeps created_at)."""
        row = _doc_row(doc)
        row.pop("created_at")
        with self.engine.begin() as conn:
            conn.execute(update(documents).where(documents.c.doc_id == doc.doc_id).values(**row))

    def set_status(self, doc_id: str, status: Status, duplicate_of: str | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(documents)
                .where(documents.c.doc_id == doc_id)
                .values(status=status.value, duplicate_of=duplicate_of, updated_at=_now())
            )

    def get(self, doc_id: str) -> SourceDocument | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(documents).where(documents.c.doc_id == doc_id)).first()
        return _row_doc(row) if row else None

    def get_many(self, doc_ids: Iterable[str]) -> dict[str, SourceDocument]:
        ids = list(set(doc_ids))
        if not ids:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(select(documents).where(documents.c.doc_id.in_(ids))).all()
        return {r.doc_id: _row_doc(r) for r in rows}

    def find_live_by_hash(self, content_hash: str, exclude: str | None = None) -> SourceDocument | None:
        """A non-duplicate document with this exact content, if one exists."""
        q = select(documents).where(
            documents.c.content_hash == content_hash, documents.c.status != Status.DUPLICATE.value
        )
        if exclude:
            q = q.where(documents.c.doc_id != exclude)
        with self.engine.connect() as conn:
            row = conn.execute(q.order_by(documents.c.created_at, documents.c.doc_id)).first()
        return _row_doc(row) if row else None

    def duplicates_of(self, doc_id: str) -> list[SourceDocument]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(documents).where(documents.c.duplicate_of == doc_id).order_by(documents.c.created_at)
            ).all()
        return [_row_doc(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self.engine.begin() as conn:
            conn.execute(delete(chunks).where(chunks.c.doc_id == doc_id))
            result = conn.execute(delete(documents).where(documents.c.doc_id == doc_id))
        return result.rowcount > 0

    def list_documents(self, query: str = "", status: str | None = None, limit: int = 50, offset: int = 0):
        q = select(documents)
        if query:
            like = f"%{query}%"
            q = q.where(or_(documents.c.title.ilike(like), documents.c.doc_id.ilike(like)))
        if status:
            q = q.where(documents.c.status == status)
        q = q.order_by(documents.c.updated_at.desc(), documents.c.doc_id).limit(limit).offset(offset)
        with self.engine.connect() as conn:
            return [_row_doc(r) for r in conn.execute(q).all()]

    def count_documents(self, status: str | None = None, source_type: str | None = None, query: str = "") -> int:
        q = select(func.count()).select_from(documents)
        if query:  # same filter as list_documents, so totals match the listing
            like = f"%{query}%"
            q = q.where(or_(documents.c.title.ilike(like), documents.c.doc_id.ilike(like)))
        if status:
            q = q.where(documents.c.status == status)
        if source_type:
            q = q.where(documents.c.source_type == source_type)
        with self.engine.connect() as conn:
            return conn.execute(q).scalar_one()

    def statuses(self, doc_ids: Iterable[str]) -> dict[str, Status]:
        ids = list(set(doc_ids))
        out: dict[str, Status] = {}
        with self.engine.connect() as conn:
            for i in range(0, len(ids), 900):
                q = select(documents.c.doc_id, documents.c.status).where(documents.c.doc_id.in_(ids[i : i + 900]))
                out.update({r.doc_id: Status(r.status) for r in conn.execute(q)})
        return out

    def set_status_many(self, doc_ids: list[str], status: Status) -> None:
        with self.engine.begin() as conn:
            for i in range(0, len(doc_ids), 900):
                conn.execute(
                    update(documents)
                    .where(documents.c.doc_id.in_(doc_ids[i : i + 900]))
                    .values(status=status.value, updated_at=_now())
                )

    def live_hashes(self) -> dict[str, str]:
        """content_hash -> doc_id for every non-duplicate document (used to dedup bulk loads)."""
        q = select(documents.c.content_hash, documents.c.doc_id).where(documents.c.status != Status.DUPLICATE.value)
        with self.engine.connect() as conn:
            return {r.content_hash: r.doc_id for r in conn.execute(q)}

    def doc_ids_with_status(self, status: Status) -> list[str]:
        with self.engine.connect() as conn:
            return list(conn.execute(select(documents.c.doc_id).where(documents.c.status == status.value)).scalars())

    # ---- chunks ---------------------------------------------------------------------------
    def add_chunks(self, items: Iterable[Chunk], batch: int = 5000) -> int:
        n = 0
        rows: list[dict] = []
        with self.engine.begin() as conn:
            for c in items:
                rows.append(_chunk_row(c))
                if len(rows) >= batch:
                    conn.execute(insert(chunks), rows)
                    n += len(rows)
                    rows = []
            if rows:
                conn.execute(insert(chunks), rows)
                n += len(rows)
        return n

    def delete_chunks(self, doc_id: str) -> list[str]:
        """Delete every chunk of a document (all versions and strategies); returns their ids."""
        with self.engine.begin() as conn:
            ids = list(conn.execute(select(chunks.c.chunk_id).where(chunks.c.doc_id == doc_id)).scalars())
            conn.execute(delete(chunks).where(chunks.c.doc_id == doc_id))
        return ids

    def chunks_for(self, doc_id: str, strategy: str | None = None) -> list[Chunk]:
        q = select(chunks).where(chunks.c.doc_id == doc_id)
        if strategy:
            q = q.where(chunks.c.strategy == strategy)
        with self.engine.connect() as conn:
            return [_row_chunk(r) for r in conn.execute(q.order_by(chunks.c.strategy, chunks.c.position)).all()]

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
        ids = list(set(chunk_ids))
        out: dict[str, Chunk] = {}
        with self.engine.connect() as conn:
            for i in range(0, len(ids), 900):  # stay under SQLite's bound-parameter limit
                for r in conn.execute(select(chunks).where(chunks.c.chunk_id.in_(ids[i : i + 900]))).all():
                    out[r.chunk_id] = _row_chunk(r)
        return out

    def iter_chunks(self, strategy: str, batch: int = 5000) -> Iterator[Chunk]:
        """Stream all chunks of one strategy (used to rebuild BM25 at start-up)."""
        last = ""
        while True:
            q = (
                select(chunks)
                .where(chunks.c.strategy == strategy, chunks.c.chunk_id > last)
                .order_by(chunks.c.chunk_id)
                .limit(batch)
            )
            with self.engine.connect() as conn:
                rows = conn.execute(q).all()
            if not rows:
                return
            for r in rows:
                yield _row_chunk(r)
            last = rows[-1].chunk_id

    def count_chunks(self, strategy: str | None = None) -> int:
        q = select(func.count()).select_from(chunks)
        if strategy:
            q = q.where(chunks.c.strategy == strategy)
        with self.engine.connect() as conn:
            return conn.execute(q).scalar_one()

    # ---- LLM usage (analytics) ------------------------------------------------------------
    def record_usage(self, model: str, status: str, mode: str, calls: int, input_tokens: int,
                     output_tokens: int, seconds: float, at: datetime | None = None) -> None:
        row = dict(created_at=at or _now(), model=model or "unknown", status=status, mode=mode,
                   calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, seconds=seconds)
        with self.engine.begin() as conn:
            conn.execute(insert(llm_usage), row)

    def usage_rows(self, since: datetime | None = None) -> list[dict]:
        """Usage rows, oldest first. Small table (one row per question), so callers aggregate in Python."""
        q = select(llm_usage).order_by(llm_usage.c.created_at)
        if since is not None:
            q = q.where(llm_usage.c.created_at >= since)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(q)]
