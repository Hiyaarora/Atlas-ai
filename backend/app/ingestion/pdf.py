"""PDF parsing via PyMuPDF."""

import pymupdf

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion import ocr
from app.ingestion.base import DocumentParser, ParsedDocument, ParsedPage, UnparsableDocumentError

logger = get_logger(__name__)


class PdfParser(DocumentParser):
    content_types = ("application/pdf",)
    extensions = (".pdf",)

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        try:
            document = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001 - pymupdf raises several types
            raise UnparsableDocumentError(f"Could not open {filename} as a PDF.") from exc

        # Encrypted PDFs open fine and then yield empty text on every page,
        # which would otherwise surface as "this document is empty" rather
        # than the actual problem.
        if document.needs_pass:
            document.close()
            raise UnparsableDocumentError(
                f"{filename} is password protected. Remove the password and upload again."
            )

        pages: list[ParsedPage] = []
        ocr_budget = settings.ocr_max_images_per_document
        ocr_pages = 0

        try:
            for index, page in enumerate(document, start=1):
                # "text" mode preserves reading order and inserts line breaks
                # at layout boundaries, which matters because those breaks are
                # what the chunker later splits on.
                raw = page.get_text("text")
                cleaned = _clean(raw)
                if cleaned:
                    pages.append(ParsedPage(number=index, text=cleaned))

                # --- OCR ------------------------------------------------
                # Runs only when there is something a text layer cannot
                # reach, and never replaces the extraction above.
                if ocr_budget <= 0 or not ocr.is_available():
                    continue

                ocr_text, used = _ocr_page(page, cleaned, budget=ocr_budget)
                ocr_budget -= used
                if ocr_text:
                    ocr_pages += 1
                    pages.append(
                        ParsedPage(
                            number=index,
                            text=ocr_text,
                            # Carried through chunking so a chunk knows its
                            # text was recognised rather than read.
                            metadata={"content_type": ocr.OCR_CONTENT_TYPE},
                        )
                    )

            metadata = {
                "title": (document.metadata or {}).get("title") or None,
                "author": (document.metadata or {}).get("author") or None,
                "source_page_count": document.page_count,
            }
        finally:
            # PyMuPDF holds a native handle; without this the file stays
            # mapped until garbage collection, which on Windows keeps the
            # file locked.
            document.close()

        if not pages:
            hint = (
                "It may be a scanned document whose pages are images with no readable text."
                if ocr.is_available()
                else "If it is a scanned document, OCR is required and is not available here."
            )
            raise UnparsableDocumentError(f"No text could be extracted from {filename}. {hint}")

        # NOT extra={"filename": ...}: `filename` is a reserved LogRecord
        # attribute and logging raises KeyError rather than overwriting it.
        # Same trap applies to name, module, msg, args, levelname, lineno.
        logger.info(
            "pdf_parsed",
            extra={
                "source_name": filename,
                "pages_with_text": len(pages),
                "ocr_pages": ocr_pages,
            },
        )
        return ParsedDocument(pages=pages, metadata=metadata)


def _clean(text: str) -> str:
    """Normalise extracted text.

    PDF extraction produces artefacts that damage chunking: a hyphen plus
    newline splitting a word across lines, and runs of blank lines from
    layout whitespace that the chunker would treat as paragraph breaks.
    """
    # Re-join words hyphenated across a line break: "reten-\ntion" -> "retention".
    text = text.replace("-\n", "")

    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse three or more blank lines into a single paragraph break.
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            cleaned.append(line)
        else:
            blank_run += 1
            if blank_run == 1:
                cleaned.append("")

    return "\n".join(cleaned).strip()


def _ocr_page(page, extracted_text: str, *, budget: int) -> tuple[str, int]:
    """Recover text this page holds only as pixels.

    Returns `(text, images_processed)` so the caller can keep a per-document
    budget. Two distinct cases, handled differently on purpose:

    * **A page with little or no text layer** is a scan. The whole page is one
      image, so it is rendered and read in one pass. Pulling its embedded
      images out individually would be the same work with more steps, and on
      some scanners produces several strips instead of one page.

    * **A page with text AND pictures** is an ordinary page containing a
      screenshot or a labelled figure. Only the pictures are read; rendering
      the whole page would re-recognise prose that was already extracted
      perfectly, wasting time to produce a worse copy of it.

    A page with text and no images is skipped entirely — there is nothing OCR
    could add, and it is the overwhelmingly common case.
    """
    images = page.get_images(full=True)

    if len(extracted_text) < settings.ocr_pdf_page_text_threshold:
        if not images and not extracted_text:
            # Genuinely blank — a separator page. Rendering it would cost a
            # second of OCR to recognise nothing.
            return "", 0
        return _ocr_rendered_page(page, extracted_text), 1

    if not images:
        return "", 0

    return _ocr_embedded_images(page, extracted_text, budget=budget)


def _ocr_rendered_page(page, extracted_text: str) -> str:
    """Render the page and read it as a single image."""
    try:
        pixmap = page.get_pixmap(dpi=settings.ocr_pdf_render_dpi)
        image_bytes = pixmap.tobytes("png")
    except Exception as exc:  # noqa: BLE001 - pymupdf raises several types
        logger.warning("pdf_page_render_failed", extra={"page": page.number, "error": str(exc)})
        return ""

    result = ocr.extract_text(image_bytes, context=f"page {page.number + 1}")
    if not result.usable:
        return ""

    # A "searchable" scan carries both a text layer and the image it was made
    # from. Indexing both would store one passage twice and let retrieval
    # present a single fact as two independent sources.
    if extracted_text and ocr.is_redundant(result.text, extracted_text):
        logger.info("ocr_skipped_redundant", extra={"page": page.number + 1})
        return ""

    return result.text


def _ocr_embedded_images(page, extracted_text: str, *, budget: int) -> tuple[str, int]:
    """Read each picture placed on an otherwise textual page."""
    document = page.parent
    collected: list[str] = []
    processed = 0

    for xref, *_ in page.get_images(full=True):
        if processed >= budget:
            break
        processed += 1

        try:
            extracted = document.extract_image(xref)
        except Exception as exc:  # noqa: BLE001
            logger.info("pdf_image_extract_failed", extra={"xref": xref, "error": str(exc)})
            continue

        result = ocr.extract_text(
            extracted["image"], context=f"page {page.number + 1} image {xref}"
        )
        if result.usable and not ocr.is_redundant(result.text, extracted_text):
            collected.append(result.text)

    return ("\n\n".join(collected), processed)
