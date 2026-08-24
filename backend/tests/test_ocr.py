"""OCR of text locked inside images.

Two layers here, tested differently.

The *quality gates* and the *plumbing* are pure logic and are tested with a
stubbed engine — deterministic, fast, and they run whether or not Tesseract is
installed. That matters because the binary is a system package and CI or a
developer laptop may not have it.

The *real recognition* tests are skipped when the binary is absent. Asserting
that a neural OCR engine reads a particular string is a test of Tesseract, not
of this code; what is worth testing here is that a real image round-trips
through the real pipeline at all.
"""

from __future__ import annotations

import io

import pytest

from app.core.config import settings
from app.ingestion import ocr

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _image_with_text(text: str, *, size: tuple[int, int] = (900, 220)) -> bytes:
    """Render text to a PNG, the way a screenshot would look."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    # The default bitmap font is small, so it is drawn several times scaled up
    # rather than relying on a TrueType file existing on the machine.
    draw.text((20, 60), text, fill="black")
    image = image.resize((size[0] * 3, size[1] * 3), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _blank_image(size: tuple[int, int] = (600, 400)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


needs_tesseract = pytest.mark.skipif(
    not ocr.is_available(),
    reason="tesseract binary not installed; real-recognition tests need it",
)


# ---------------------------------------------------------------------------
# Availability and degradation
# ---------------------------------------------------------------------------


def test_missing_engine_degrades_instead_of_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingestion must never fail because an optional enhancement is absent.

    Tesseract is a system binary, so it can be missing on any machine. When it
    is, documents must parse exactly as they did before OCR existed.
    """
    monkeypatch.setattr(ocr, "is_available", lambda: False)

    result = ocr.extract_text(_image_with_text("Hello"))

    assert not result.usable
    assert result.rejected == "ocr_unavailable"


def test_disabling_ocr_turns_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_enabled", False)
    ocr.is_available.cache_clear()

    assert ocr.is_available() is False

    ocr.is_available.cache_clear()


def test_a_corrupt_blob_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A document can contain a truncated or unsupported image. That is an
    image to skip, not a document to reject."""
    monkeypatch.setattr(ocr, "is_available", lambda: True)

    result = ocr.extract_text(b"this is not an image at all")

    assert not result.usable
    assert result.rejected == "unreadable_image"


def test_tiny_images_are_skipped_before_the_engine_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bullet glyphs, spacers and logo marks are everywhere in documents. Each
    costs a subprocess launch to learn nothing."""
    monkeypatch.setattr(ocr, "is_available", lambda: True)

    result = ocr.extract_text(_blank_image(size=(20, 20)))

    assert result.rejected == "image_too_small"


# ---------------------------------------------------------------------------
# Quality gates — the reason OCR does not just index whatever it returns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", "too_short"),
        ("   \n  \n ", "too_short"),
        ("a", "too_short"),
        ("| , . - ' ` ~ ^ * _ = +", "too_few_words"),
        ("a b c d e f g h i j", "mostly_noise"),
    ],
)
def test_meaningless_output_is_rejected(raw: str, reason: str) -> None:
    """OCR on an image with no text does not return nothing — it returns
    punctuation and stray letters read out of noise and compression
    artefacts. Indexed, that becomes a retrievable chunk of gibberish handed
    to the model as evidence, which is worse than having no OCR at all."""
    assert ocr._clean_and_gate(raw).rejected == reason


def test_real_text_passes_the_gates() -> None:
    result = ocr._clean_and_gate(
        "Quarterly revenue rose by twelve percent\nacross the northern region."
    )

    assert result.usable
    assert "Quarterly revenue" in result.text


def test_ragged_whitespace_is_normalised_but_paragraphs_survive() -> None:
    """OCR output is full of stray spacing. Paragraph breaks are kept because
    the chunker splits on them."""
    result = ocr._clean_and_gate("First    paragraph  here\n\n\n\nSecond paragraph follows")

    assert result.usable
    assert "First paragraph here" in result.text
    assert "\n\n" in result.text


# ---------------------------------------------------------------------------
# Redundancy — a page can hold both a text layer and a picture of it
# ---------------------------------------------------------------------------


def test_text_already_present_is_not_indexed_twice() -> None:
    """What a scanner produces when it "makes a PDF searchable": the same
    passage as both text and image. Indexing both stores one fact twice, so
    retrieval hands the model two chunks that look like two sources."""
    page_text = "The deployment window is Tuesday and Thursday between 10:00 and 16:00 UTC."
    ocr_text = "The deployment window is Tuesday and Thursday between 10:00 and 16:00 UTC"

    assert ocr.is_redundant(ocr_text, page_text)


def test_new_text_from_an_image_is_kept() -> None:
    """A screenshot on an otherwise textual page holds content the text layer
    never had. That is the whole point of the feature."""
    page_text = "See the dashboard below for current utilisation figures."
    ocr_text = "CPU 84 percent   Memory 61 percent   Disk 12 percent"

    assert not ocr.is_redundant(ocr_text, page_text)


def test_empty_ocr_counts_as_redundant() -> None:
    assert ocr.is_redundant("", "anything at all")


def test_redundancy_against_an_empty_page_keeps_the_ocr_text() -> None:
    """A scanned page has no text layer to duplicate."""
    assert not ocr.is_redundant("Recognised text from the scan", "")


# ---------------------------------------------------------------------------
# Real recognition, when the engine is present
# ---------------------------------------------------------------------------


@needs_tesseract
def test_an_image_containing_text_is_read() -> None:
    result = ocr.extract_text(_image_with_text("INVOICE TOTAL 4290"))

    assert result.usable
    assert "INVOICE" in result.text.upper()


@needs_tesseract
def test_an_image_containing_no_text_produces_nothing() -> None:
    """The gates must hold against the real engine, not just against
    hand-written strings."""
    result = ocr.extract_text(_blank_image())

    assert not result.usable
