"""Parsers and the parser registry."""

import pymupdf
import pytest

from app.ingestion import UnparsableDocumentError, get_parser
from app.ingestion.pdf import PdfParser
from app.ingestion.text import TextParser


def make_pdf(pages: list[str], *, password: str | None = None) -> bytes:
    """Build a real PDF in memory.

    A genuine PyMuPDF-produced file, not a fixture blob — so the test
    exercises the actual extraction path rather than a canned string.
    """
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 96), text, fontsize=11)

    if password:
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password
        )
    else:
        data = document.tobytes()

    document.close()
    return data


# ---- PDF -----------------------------------------------------------------


def test_pdf_text_is_extracted_per_page() -> None:
    data = make_pdf(["Alpha page content", "Bravo page content"])

    parsed = PdfParser().parse(data, filename="test.pdf")

    assert parsed.page_count == 2
    assert "Alpha" in parsed.pages[0].text
    assert "Bravo" in parsed.pages[1].text
    assert [page.number for page in parsed.pages] == [1, 2]


def test_pdf_pages_are_numbered_from_one() -> None:
    """Page 0 would mean nothing to a user reading a citation."""
    parsed = PdfParser().parse(make_pdf(["only page"]), filename="x.pdf")

    assert parsed.pages[0].number == 1


def test_empty_pages_are_skipped() -> None:
    data = make_pdf(["Real content", "", "More content"])

    parsed = PdfParser().parse(data, filename="x.pdf")

    assert all(page.text.strip() for page in parsed.pages)


def test_pdf_with_no_text_is_rejected_with_a_useful_message() -> None:
    """A scanned PDF should say "needs OCR", not fail mysteriously."""
    data = make_pdf(["", ""])

    with pytest.raises(UnparsableDocumentError, match="OCR"):
        PdfParser().parse(data, filename="scan.pdf")


def test_password_protected_pdf_is_reported_clearly() -> None:
    data = make_pdf(["secret"], password="hunter2")

    with pytest.raises(UnparsableDocumentError, match="password"):
        PdfParser().parse(data, filename="locked.pdf")


def test_garbage_bytes_are_rejected() -> None:
    with pytest.raises(UnparsableDocumentError):
        PdfParser().parse(b"this is definitely not a pdf", filename="fake.pdf")


def test_hyphenated_line_breaks_are_rejoined() -> None:
    """PDF layout splits words across lines; leaving them split breaks search."""
    from app.ingestion.pdf import _clean

    assert _clean("reten-\ntion policy") == "retention policy"


# ---- Text ----------------------------------------------------------------


def test_text_file_is_one_page() -> None:
    parsed = TextParser().parse(b"Hello, knowledge base.", filename="notes.txt")

    assert parsed.page_count == 1
    assert parsed.pages[0].number == 1
    assert parsed.pages[0].text == "Hello, knowledge base."


def test_utf8_bom_is_handled() -> None:
    """Windows editors write a BOM; it must not appear in the text."""
    parsed = TextParser().parse("﻿Hello".encode(), filename="notes.txt")

    assert parsed.pages[0].text == "Hello"


def test_cp1252_smart_quotes_do_not_crash() -> None:
    data = "He said “hello”".encode("cp1252")

    parsed = TextParser().parse(data, filename="notes.txt")

    assert "hello" in parsed.pages[0].text


def test_empty_text_file_is_rejected() -> None:
    with pytest.raises(UnparsableDocumentError, match="empty"):
        TextParser().parse(b"   \n  ", filename="blank.txt")


# ---- Registry ------------------------------------------------------------


def test_extension_selects_the_parser() -> None:
    assert isinstance(get_parser(filename="a.pdf", content_type=None), PdfParser)
    assert isinstance(get_parser(filename="a.md", content_type=None), TextParser)


def test_extension_wins_over_a_wrong_content_type() -> None:
    """Browsers report octet-stream for .md constantly."""
    parser = get_parser(filename="notes.md", content_type="application/octet-stream")

    assert isinstance(parser, TextParser)


def test_content_type_is_the_fallback_when_there_is_no_extension() -> None:
    parser = get_parser(filename="noextension", content_type="application/pdf")

    assert isinstance(parser, PdfParser)


def test_unsupported_type_is_rejected() -> None:
    with pytest.raises(UnparsableDocumentError, match="Unsupported"):
        get_parser(filename="virus.exe", content_type="application/x-msdownload")
