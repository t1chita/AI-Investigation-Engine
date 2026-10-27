import json

import pytest

from investigator.chunking import FixedWordChunker, SentenceChunker, get_chunker
from investigator.models import Sentence, SourceDocument
from investigator.parsers import ParseError, parse_file
from investigator.text import content_hash, split_sentences


def doc(*sentences, pages=None):
    pages = pages or [None] * len(sentences)
    return SourceDocument(
        doc_id="d1",
        title="T",
        sentences=[Sentence(i, s, p) for i, (s, p) in enumerate(zip(sentences, pages))],
        format="txt",
        source_type="upload",
        content_hash="x",
    )


def test_fixed_chunker_windows_overlap_and_cover_every_word():
    d = doc("one two three four", "five six seven", "eight nine ten")
    chunks = FixedWordChunker(size=4, overlap=1).chunk(d)
    assert [c.text for c in chunks] == ["one two three four", "four five six seven", "seven eight nine ten"]
    assert [(c.sentence_start, c.sentence_end) for c in chunks] == [(0, 0), (0, 1), (1, 2)]
    assert chunks[1].chunk_id == "d1:v1:fixed:1"


def test_sentence_chunker_never_cuts_sentences_and_respects_pages():
    d = doc("a b c", "d e", "f g h i j k", "l m", pages=[1, 1, 1, 2])
    chunks = SentenceChunker(max_words=5).chunk(d)
    assert [c.text for c in chunks] == ["a b c d e", "f g h i j k", "l m"]  # long sentence kept whole
    assert [(c.sentence_start, c.sentence_end, c.page) for c in chunks] == [(0, 1, 1), (2, 2, 1), (3, 3, 2)]


def test_chunkers_validate_arguments():
    with pytest.raises(ValueError):
        FixedWordChunker(size=4, overlap=4)
    with pytest.raises(ValueError):
        get_chunker("semantic")


def test_tokenize_keeps_identifiers_whole():
    from investigator.text import tokenize

    assert tokenize("The v2.18 release, INC-4471, on 2026-03-01.") == ["v2.18", "release", "inc-4471", "2026-03-01"]


def test_split_sentences_and_hash_normalisation():
    assert split_sentences("First one. Second one!\n\nThird") == ["First one.", "Second one!", "Third"]
    assert content_hash("a  b\n c") == content_hash(" a b c ")


def test_parse_txt_uses_first_line_as_title():
    (d,) = parse_file("notes.txt", b"# Outage report\n\nThe API failed. It recovered.")
    assert d.title == "Outage report" and d.format == "txt"
    assert [s.text for s in d.sentences] == ["# Outage report", "The API failed.", "It recovered."]


def test_parse_json_list_and_metadata():
    payload = [
        {"title": "A", "text": "Alpha one. Alpha two.", "date": "2026-03-01"},
        {"title": "B", "sentences": ["Beta.", "  "]},
    ]
    a, b = parse_file("docs.json", json.dumps(payload).encode())
    assert a.metadata["date"] == "2026-03-01" and len(a.sentences) == 2
    assert [s.text for s in b.sentences] == ["Beta."]


@pytest.mark.parametrize(
    "name,data,message",
    [
        ("x.docx", b"hi", "unsupported"),
        ("x.txt", b"", "empty"),
        ("x.txt", b"   \n  ", "no text"),
        ("x.json", b"{bad", "invalid JSON"),
        ("x.json", b'{"title": "t"}', "needs a 'text'"),
        ("x.pdf", b"not a pdf", "unreadable PDF"),
    ],
)
def test_parse_errors_are_readable(name, data, message):
    with pytest.raises(ParseError, match=message):
        parse_file(name, data)


def _tiny_pdf(text: str) -> bytes:
    """A minimal one-page PDF with one line of text (hand-written, no extra dependency)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_parse_pdf_extracts_text_with_page_numbers():
    (d,) = parse_file("report.pdf", _tiny_pdf("Deploy v2.18 lowered the pool size. Payments failed."))
    assert d.format == "pdf" and d.title == "report" and d.metadata["pages"] == 1
    assert [s.text for s in d.sentences] == ["Deploy v2.18 lowered the pool size.", "Payments failed."]
    assert {s.page for s in d.sentences} == {1}
