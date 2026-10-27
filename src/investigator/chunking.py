"""Two chunking strategies, compared in the evaluation.

``fixed``    Fixed-size word windows with overlap. Ignores sentence boundaries, so a window may
             start or end mid-sentence. Simple, predictable chunk sizes.
``sentence`` Sentence-aware packing. Whole sentences are packed into a chunk until the word budget
             would be exceeded; a sentence is never cut. A page change also starts a new chunk,
             so a citation never spans two PDF pages.

Both record the first and last sentence index each chunk covers, so any chunk can be traced back
to exact source sentences (needed for citations and for evaluation against HotpotQA's
sentence-level supporting facts). Words are whitespace-separated tokens here; this keeps both
strategies on the same unit, and MiniLM's 256-token limit is comfortably above 64 words.
"""

from .models import Chunk, SourceDocument, make_chunk_id


class FixedWordChunker:
    """Windows of ``size`` words; each window starts ``size - overlap`` words after the previous."""

    name = "fixed"

    def __init__(self, size: int = 64, overlap: int = 16):
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        # Flatten to words, remembering which sentence each word came from.
        words: list[str] = []
        owner: list[int] = []  # owner[i] = position in doc.sentences of words[i]
        for pos, sentence in enumerate(doc.sentences):
            for word in sentence.text.split():
                words.append(word)
                owner.append(pos)

        chunks: list[Chunk] = []
        step = self.size - self.overlap
        start = 0
        while start < len(words):
            end = min(start + self.size, len(words))
            first, last = doc.sentences[owner[start]], doc.sentences[owner[end - 1]]
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, doc.version, self.name, len(chunks)),
                    doc_id=doc.doc_id,
                    version=doc.version,
                    strategy=self.name,
                    position=len(chunks),
                    title=doc.title,
                    text=" ".join(words[start:end]),
                    sentence_start=first.index,
                    sentence_end=last.index,
                    page=first.page,
                )
            )
            if end == len(words):
                break
            start += step
        return chunks


class SentenceChunker:
    """Pack whole sentences up to ``max_words`` words; an over-long sentence becomes its own chunk."""

    name = "sentence"

    def __init__(self, max_words: int = 64):
        if max_words <= 0:
            raise ValueError("max_words must be > 0")
        self.max_words = max_words

    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        group: list = []
        group_words = 0

        def flush():
            if group:
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc.doc_id, doc.version, self.name, len(chunks)),
                        doc_id=doc.doc_id,
                        version=doc.version,
                        strategy=self.name,
                        position=len(chunks),
                        title=doc.title,
                        text=" ".join(s.text for s in group),
                        sentence_start=group[0].index,
                        sentence_end=group[-1].index,
                        page=group[0].page,
                    )
                )

        for sentence in doc.sentences:
            n = len(sentence.text.split())
            if n == 0:
                continue
            page_changed = bool(group) and sentence.page != group[0].page
            if group and (group_words + n > self.max_words or page_changed):
                flush()
                group, group_words = [], 0
            group.append(sentence)
            group_words += n
        flush()
        return chunks


CHUNKERS = {"fixed": FixedWordChunker, "sentence": SentenceChunker}


def get_chunker(name: str):
    try:
        return CHUNKERS[name]()
    except KeyError:
        raise ValueError(f"unknown chunking strategy {name!r}; choose from {sorted(CHUNKERS)}") from None
