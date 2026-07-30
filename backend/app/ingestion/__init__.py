"""Knowledge ingestion: turning source files into chunks of text.

Import from here rather than from a concrete parser:

    from app.ingestion import chunk_pages, get_parser
"""

from app.ingestion.base import (
    DocumentParser,
    ParsedDocument,
    ParsedPage,
    UnparsableDocumentError,
)
from app.ingestion.chunking import TextChunk, chunk_pages, split_text
from app.ingestion.registry import get_parser, supported_content_types, supported_extensions

__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "ParsedPage",
    "TextChunk",
    "UnparsableDocumentError",
    "chunk_pages",
    "get_parser",
    "split_text",
    "supported_content_types",
    "supported_extensions",
]
