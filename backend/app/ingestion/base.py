"""The contract every knowledge source implements.

This is the extension point the whole platform is designed around. A PDF, a
web page, a GitHub repository and a Notion database differ only in how they
produce `ParsedPage` objects; everything downstream — chunking, embedding,
indexing, retrieval, generation — is identical and never learns where the
text came from.

Adding a source means writing one parser and registering it. Nothing else in
the codebase changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedPage:
    """One addressable unit of a source.

    "Page" is the PDF case. For a web page it is the whole document; for a
    repository, one file. The number is what a citation points at, so it must
    mean something a human can navigate to.
    """

    number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    pages: list[ParsedPage]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_characters(self) -> int:
        return sum(len(page.text) for page in self.pages)


class DocumentParser(ABC):
    """Turns raw bytes into pages of plain text."""

    #: MIME types this parser claims.
    content_types: tuple[str, ...]

    #: File extensions, lowercase and dotted, e.g. ".pdf".
    extensions: tuple[str, ...]

    @abstractmethod
    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        """Extract text.

        Implementations must be pure and synchronous. They are CPU-bound, so
        callers run them in a thread — see `document_service`.

        Raises:
            UnparsableDocumentError: the bytes are not readable as this type.
        """


class UnparsableDocumentError(Exception):
    """The file could not be read as its declared type."""
