"""Parser selection.

Adding a source type means writing a parser and appending it to `PARSERS`.
"""

from pathlib import PurePosixPath

from app.ingestion.base import DocumentParser, UnparsableDocumentError
from app.ingestion.pdf import PdfParser
from app.ingestion.powerpoint import PptxParser
from app.ingestion.tabular import CsvParser, XlsxParser
from app.ingestion.text import TextParser
from app.ingestion.web import HtmlParser
from app.ingestion.word import DocxParser

#: Order matters only for content-type fallback, where the first parser
#: claiming a MIME type wins. Extension matching is exact, so it is unaffected.
PARSERS: tuple[DocumentParser, ...] = (
    PdfParser(),
    DocxParser(),
    PptxParser(),
    XlsxParser(),
    CsvParser(),
    HtmlParser(),
    # Last: TextParser claims text/plain, which browsers report for all sorts
    # of files. Anything with a more specific parser must be matched first.
    TextParser(),
)


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted({ext for parser in PARSERS for ext in parser.extensions}))


def supported_content_types() -> tuple[str, ...]:
    return tuple(sorted({ct for parser in PARSERS for ct in parser.content_types}))


def get_parser(*, filename: str, content_type: str | None) -> DocumentParser:
    """Choose a parser for an upload.

    Extension is checked before content type on purpose. Browsers report
    wildly inconsistent MIME types for the same file — Markdown arrives as
    text/markdown, text/plain, or application/octet-stream depending on the
    OS and browser — whereas the extension the user typed is stable.
    """
    extension = PurePosixPath(filename).suffix.lower()

    for parser in PARSERS:
        if extension in parser.extensions:
            return parser

    if content_type:
        normalised = content_type.split(";")[0].strip().lower()
        for parser in PARSERS:
            if normalised in parser.content_types:
                return parser

    raise UnparsableDocumentError(
        f"Unsupported file type {extension or content_type or 'unknown'!r}. "
        f"Supported: {', '.join(supported_extensions())}"
    )
