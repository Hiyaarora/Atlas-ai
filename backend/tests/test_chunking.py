"""The chunker.

Pure functions, no I/O — so this can be exhaustive. Worth it: chunking
silently determines the ceiling on retrieval quality. A bug here cannot be
compensated for by any downstream stage.
"""

import pytest

from app.ingestion.base import ParsedPage
from app.ingestion.chunking import chunk_pages, split_text

LOREM = (
    "Retrieval augmented generation combines search with generation. "
    "First the system retrieves relevant passages from a corpus. "
    "Then a language model writes an answer grounded in those passages. "
)


def test_short_text_is_one_chunk() -> None:
    assert split_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_empty_text_produces_no_chunks() -> None:
    assert split_text("", chunk_size=100, chunk_overlap=10) == []
    assert split_text("   \n\n  ", chunk_size=100, chunk_overlap=10) == []


def test_every_chunk_respects_the_size_limit() -> None:
    text = LOREM * 20

    chunks = split_text(text, chunk_size=200, chunk_overlap=40)

    assert chunks
    for chunk in chunks:
        assert len(chunk) <= 200, f"chunk of {len(chunk)} exceeds the limit"


def test_no_content_is_lost() -> None:
    """Every word of the source must survive into some chunk."""
    text = LOREM * 5

    chunks = split_text(text, chunk_size=150, chunk_overlap=30)

    joined = " ".join(chunks)
    for word in set(text.split()):
        assert word in joined, f"{word!r} was dropped by the splitter"


def test_paragraph_boundaries_are_preferred() -> None:
    """With paragraphs that fit, chunks should land on paragraph breaks."""
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."

    chunks = split_text(text, chunk_size=30, chunk_overlap=0)

    # Each paragraph is under 30 chars, so none should be split mid-sentence.
    assert all(not chunk.startswith(" ") for chunk in chunks)
    assert any("First paragraph" in chunk for chunk in chunks)
    assert any("Third paragraph" in chunk for chunk in chunks)


def test_overlap_repeats_content_between_chunks() -> None:
    """The point of overlap: a boundary-straddling sentence survives whole."""
    text = LOREM * 6

    with_overlap = split_text(text, chunk_size=200, chunk_overlap=80)
    without_overlap = split_text(text, chunk_size=200, chunk_overlap=0)

    assert sum(len(c) for c in with_overlap) > sum(len(c) for c in without_overlap)


def test_text_with_no_separators_is_still_split() -> None:
    """Pathological input must not loop forever or produce an oversized chunk."""
    text = "x" * 1000

    chunks = split_text(text, chunk_size=100, chunk_overlap=0)

    assert len(chunks) == 10
    assert all(len(chunk) == 100 for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """Otherwise each chunk restates the previous one and nothing advances."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text("some text", chunk_size=100, chunk_overlap=100)


def test_separator_punctuation_is_preserved() -> None:
    """Splitting on '. ' must not delete the full stops."""
    text = "One. Two. Three. " * 30

    chunks = split_text(text, chunk_size=60, chunk_overlap=0)

    assert "".join(chunks).count(".") >= 80


def test_chunk_pages_tags_each_chunk_with_its_page() -> None:
    """A chunk with no page number cannot be cited."""
    pages = [
        ParsedPage(number=1, text=LOREM * 3),
        ParsedPage(number=2, text=LOREM * 3),
        ParsedPage(number=7, text="A short final page."),
    ]

    results = chunk_pages(pages, chunk_size=150, chunk_overlap=30)

    page_numbers = {page.number for page, _ in results}
    assert page_numbers == {1, 2, 7}


def test_pages_are_chunked_independently() -> None:
    """No chunk may span two pages, or it could not be attributed to either."""
    pages = [
        ParsedPage(number=1, text="Alpha content here."),
        ParsedPage(number=2, text="Bravo content here."),
    ]

    results = chunk_pages(pages, chunk_size=500, chunk_overlap=0)

    for page, chunk in results:
        if page.number == 1:
            assert "Bravo" not in chunk.text
        else:
            assert "Alpha" not in chunk.text


def test_chunk_index_restarts_per_page() -> None:
    pages = [
        ParsedPage(number=1, text=LOREM * 4),
        ParsedPage(number=2, text=LOREM * 4),
    ]

    results = chunk_pages(pages, chunk_size=120, chunk_overlap=20)

    first_page = [chunk.index for page, chunk in results if page.number == 1]
    second_page = [chunk.index for page, chunk in results if page.number == 2]

    assert first_page == list(range(len(first_page)))
    assert second_page == list(range(len(second_page)))
