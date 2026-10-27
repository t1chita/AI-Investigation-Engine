"""File parsers for the three supported formats: TXT (and Markdown), JSON and PDF.

Every parser returns a list of ``ParsedDocument`` (a JSON file may hold several documents) and
raises ``ParseError`` with a readable message for empty, malformed or unsupported input, so the API
can report the problem instead of crashing.

JSON shapes accepted (one object or a list of them):
    {"title": "...", "text": "..."}
    {"title": "...", "sentences": ["...", "..."]}
    optional: "metadata": {...}; any other top-level keys are kept as metadata too.
"""

import io
import json
from pathlib import PurePath

from .models import ParsedDocument, Sentence
from .text import normalize_text, split_sentences

MAX_BYTES = 20 * 1024 * 1024
SUPPORTED = {".txt": "txt", ".md": "txt", ".json": "json", ".pdf": "pdf"}


class ParseError(ValueError):
    """The file cannot be turned into a document (bad format, empty, too large, unsupported)."""


def parse_file(filename: str, data: bytes) -> list[ParsedDocument]:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ParseError(f"unsupported file type {suffix or '(none)'}; supported: {', '.join(SUPPORTED)}")
    if not data:
        raise ParseError("file is empty")
    if len(data) > MAX_BYTES:
        raise ParseError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    kind = SUPPORTED[suffix]
    stem = PurePath(filename).stem
    docs = {"txt": _parse_txt, "json": _parse_json, "pdf": _parse_pdf}[kind](data, stem)
    for doc in docs:
        doc.metadata.setdefault("filename", filename)
    return docs


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _sentences(texts: list[str], page: int | None = None, start: int = 0) -> list[Sentence]:
    return [Sentence(index=start + i, text=t, page=page) for i, t in enumerate(texts)]


def _parse_txt(data: bytes, stem: str) -> list[ParsedDocument]:
    text = _decode(data)
    sentences = split_sentences(text)
    if not sentences:
        raise ParseError("no text found in file")
    first_line = next((normalize_text(line) for line in text.splitlines() if line.strip()), "")
    title = first_line.lstrip("# ").strip()[:200] or stem
    return [ParsedDocument(title=title, sentences=_sentences(sentences), format="txt")]


def _parse_json(data: bytes, stem: str) -> list[ParsedDocument]:
    try:
        payload = json.loads(_decode(data))
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc.msg} (line {exc.lineno})") from None
    items = payload if isinstance(payload, list) else [payload]
    if not items:
        raise ParseError("JSON list is empty")

    docs = []
    for n, item in enumerate(items):
        where = f"item {n}" if isinstance(payload, list) else "document"
        if not isinstance(item, dict):
            raise ParseError(f"{where}: expected an object with 'title' and 'text' or 'sentences'")
        if isinstance(item.get("sentences"), list):
            raw = [normalize_text(str(s)) for s in item["sentences"]]
            texts = [s for s in raw if s]
        elif isinstance(item.get("text"), str):
            texts = split_sentences(item["text"])
        else:
            raise ParseError(f"{where}: needs a 'text' string or a 'sentences' list")
        if not texts:
            raise ParseError(f"{where}: no text")
        metadata = dict(item.get("metadata") or {})
        metadata.update({k: v for k, v in item.items() if k not in {"title", "text", "sentences", "metadata"}})
        title = normalize_text(str(item.get("title") or "")) or (stem if len(items) == 1 else f"{stem} #{n + 1}")
        docs.append(ParsedDocument(title=title, sentences=_sentences(texts), format="json", metadata=metadata))
    return docs


def _parse_pdf(data: bytes, stem: str) -> list[ParsedDocument]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        info = reader.metadata or {}
    except (PdfReadError, ValueError, OSError) as exc:
        raise ParseError(f"unreadable PDF: {exc}") from None

    sentences: list[Sentence] = []
    for page_number, page_text in enumerate(pages, start=1):
        sentences.extend(_sentences(split_sentences(page_text), page=page_number, start=len(sentences)))
    if not sentences:
        raise ParseError("PDF has no extractable text (it may be scanned images)")

    metadata = {"pages": len(pages)}
    for key in ("author", "subject", "creator"):
        value = getattr(info, key, None)
        if value:
            metadata[key] = str(value)
    title = normalize_text(str(getattr(info, "title", "") or "")) or stem
    return [ParsedDocument(title=title, sentences=sentences, format="pdf", metadata=metadata)]
