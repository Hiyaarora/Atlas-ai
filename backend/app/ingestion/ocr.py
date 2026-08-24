"""Optical character recognition for text locked inside images.

WHY
---
A parser reads the text layer of a file. Text that only exists as pixels — a
screenshot pasted into a report, a scanned page, a slide exported as an image
— is invisible to it, so the document either loses that content or, when the
whole file is images, is rejected as unreadable.

OCR recovers that text and feeds it into the same pipeline as everything else.
Nothing downstream changes: OCR output becomes ordinary page text, is chunked
by the same chunker, embedded by the same provider, and cited with the same
page numbers. The only distinction it carries is a `content_type` of "ocr", so
a reader can tell where a passage came from.

DELIBERATELY NOT
----------------
This does one thing: characters in an image become characters in the text. It
does not describe pictures, read charts, or interpret diagrams. Those need a
vision model and are a different feature with different costs and failure
modes.

DEGRADATION
-----------
Tesseract is a system binary, not a Python package, so it can be absent even
when `pytesseract` is installed. Everything here is written so that a missing
binary means documents parse exactly as they did before OCR existed — no
errors, no failed ingestion, just no OCR. Ingestion must never fail because an
optional enhancement is unavailable.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Marks a page whose text came from OCR rather than a text layer.
OCR_CONTENT_TYPE = "ocr"

#: Tokens used when comparing OCR output against text already extracted.
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class OcrResult:
    text: str
    #: Why the result was discarded, or None if it was kept. Logged rather
    #: than raised: a rejected image is normal, not an error.
    rejected: str | None = None

    @property
    def usable(self) -> bool:
        return self.rejected is None and bool(self.text)


@lru_cache(maxsize=1)
def is_available() -> bool:
    """Is OCR usable in this process?

    Cached because it shells out to the Tesseract binary, and the answer
    cannot change while the process runs. Without the cache every image in
    every document would pay for a subprocess launch just to ask.
    """
    if not settings.ocr_enabled:
        return False

    try:
        import pytesseract
    except ImportError:
        logger.info("ocr_unavailable", extra={"reason": "pytesseract not installed"})
        return False

    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001 - pytesseract raises several types
        # The usual cause is the tesseract binary being absent from PATH.
        logger.info("ocr_unavailable", extra={"reason": f"{type(exc).__name__}: {exc}"})
        return False

    logger.info("ocr_available", extra={"tesseract_version": str(version)})
    return True


def extract_text(image_bytes: bytes, *, context: str = "") -> OcrResult:
    """Read text out of one image.

    Returns an `OcrResult` rather than raising. A blank image, a photograph
    with no writing in it, and a corrupt blob are all ordinary occurrences
    during ingestion, and none of them should fail a document.
    """
    if not is_available():
        return OcrResult(text="", rejected="ocr_unavailable")

    import pytesseract
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow raises several types
        logger.info("ocr_image_unreadable", extra={"context": context, "error": str(exc)})
        return OcrResult(text="", rejected="unreadable_image")

    # Skip anything too small to hold readable text. Bullet glyphs, spacers
    # and logo marks are common in documents, individually cost a subprocess
    # launch, and never produce anything worth indexing.
    if image.width * image.height < settings.ocr_min_image_pixels:
        return OcrResult(text="", rejected="image_too_small")

    try:
        raw = pytesseract.image_to_string(
            image,
            lang=settings.ocr_language,
            timeout=settings.ocr_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - includes RuntimeError on timeout
        logger.warning(
            "ocr_failed", extra={"context": context, "error": f"{type(exc).__name__}: {exc}"}
        )
        return OcrResult(text="", rejected="ocr_error")
    finally:
        image.close()

    return _clean_and_gate(raw)


def _clean_and_gate(raw: str) -> OcrResult:
    """Tidy OCR output and reject it if it is not worth indexing.

    OCR on an image with no text does not return nothing. It returns a
    scattering of punctuation and single letters read out of noise, edges and
    compression artefacts. Indexed, that becomes a chunk of gibberish which
    can be retrieved and handed to the model as evidence — worse than having
    no OCR at all, because it is confidently wrong rather than absent.
    """
    # Collapse the ragged whitespace OCR produces, while keeping paragraph
    # breaks, which carry real structure the chunker splits on.
    text = re.sub(r"[ \t]+", " ", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()

    if len(text) < settings.ocr_min_characters:
        return OcrResult(text="", rejected="too_short")

    words = _WORD.findall(text.lower())
    if len(words) < settings.ocr_min_words:
        return OcrResult(text="", rejected="too_few_words")

    # Noise reads as isolated characters. Requiring some proportion of real
    # words separates "a diagram with labels" from "a photograph of a sunset".
    substantial = [word for word in words if len(word) >= 3]
    if len(substantial) / len(words) < settings.ocr_min_word_ratio:
        return OcrResult(text="", rejected="mostly_noise")

    return OcrResult(text=text)


def is_redundant(ocr_text: str, existing_text: str) -> bool:
    """Does `ocr_text` merely repeat text already extracted from the page?

    A PDF page can carry both a text layer and an image of that same text —
    scanners that "make searchable" a document do exactly this. Indexing both
    stores the passage twice, so retrieval returns two chunks saying the same
    thing and the model is handed one fact as though it were two sources.

    Compared on word sets rather than exact strings, because OCR of the same
    passage differs from the text layer in whitespace, hyphenation and the
    occasional misread character.
    """
    ocr_words = set(_WORD.findall(ocr_text.lower()))
    if not ocr_words:
        return True

    existing_words = set(_WORD.findall(existing_text.lower()))
    if not existing_words:
        return False

    overlap = len(ocr_words & existing_words) / len(ocr_words)
    return overlap >= settings.ocr_redundancy_threshold
