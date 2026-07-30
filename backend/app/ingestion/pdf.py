"""PDF parsing via PyMuPDF."""

import pymupdf

from app.core.logging import get_logger
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
        try:
            for index, page in enumerate(document, start=1):
                # "text" mode preserves reading order and inserts line breaks
                # at layout boundaries, which matters because those breaks are
                # what the chunker later splits on.
                raw = page.get_text("text")
                cleaned = _clean(raw)
                if cleaned:
                    pages.append(ParsedPage(number=index, text=cleaned))

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
            raise UnparsableDocumentError(
                f"No text could be extracted from {filename}. "
                "If it is a scanned document it needs OCR, which Atlas AI does not do yet."
            )

        # NOT extra={"filename": ...}: `filename` is a reserved LogRecord
        # attribute and logging raises KeyError rather than overwriting it.
        # Same trap applies to name, module, msg, args, levelname, lineno.
        logger.info(
            "pdf_parsed",
            extra={"source_name": filename, "pages_with_text": len(pages)},
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
