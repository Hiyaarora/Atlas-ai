"""Plain text and Markdown parsing."""

from app.ingestion.base import DocumentParser, ParsedDocument, ParsedPage, UnparsableDocumentError

# Order matters. utf-8-sig comes FIRST because it is utf-8 that also strips a
# byte-order mark; plain utf-8 "succeeds" on a BOM-prefixed file and leaves a
# stray U+FEFF as the first character, which then shows up at the head of the
# first chunk and in every citation excerpt.
# cp1252 catches files from Windows editors that would otherwise fail on a
# single smart quote. latin-1 never fails, so it is the terminator.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class TextParser(DocumentParser):
    content_types = ("text/plain", "text/markdown", "text/x-markdown")
    extensions = (".txt", ".md", ".markdown")

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        text = _decode(data, filename)
        cleaned = text.strip()

        if not cleaned:
            raise UnparsableDocumentError(f"{filename} is empty.")

        # A text file has no pages. It is one unit, numbered 1, so citations
        # stay uniform across source types.
        return ParsedDocument(
            pages=[ParsedPage(number=1, text=cleaned)],
            metadata={"source_page_count": 1},
        )


def _decode(data: bytes, filename: str) -> str:
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    # latin-1 cannot fail, so reaching here means the file is not text at all.
    raise UnparsableDocumentError(f"Could not decode {filename} as text.")
