"""HTML parsing via BeautifulSoup.

Also the groundwork for a URL knowledge source: fetching a page and ingesting
it is the same problem once the bytes are in hand, so a future
`WebPageLoader` reuses this parser unchanged.
"""

from bs4 import BeautifulSoup

from app.core.logging import get_logger
from app.ingestion.base import DocumentParser, ParsedDocument, ParsedPage, UnparsableDocumentError

logger = get_logger(__name__)

#: Elements whose text is never content. Leaving these in fills the index with
#: menu labels and cookie notices, which then match every query weakly and
#: crowd out real passages — the same failure that made a bibliography look
#: like the best answer to "summarise this".
_NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg")

#: Rendered as markdown so structure survives into the chunker.
_HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


class HtmlParser(DocumentParser):
    content_types = ("text/html", "application/xhtml+xml")
    extensions = (".html", ".htm")

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        try:
            # lxml over the stdlib parser: real-world HTML is malformed, and
            # lxml recovers from unclosed tags the way a browser does.
            soup = BeautifulSoup(data, "lxml")
        except Exception as exc:  # noqa: BLE001
            raise UnparsableDocumentError(f"Could not parse {filename} as HTML.") from exc

        title = soup.title.get_text(strip=True) if soup.title else None

        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # Prefer the semantic content root when the page has one; it excludes
        # sidebars and related-article lists that survive tag stripping.
        root = soup.find("main") or soup.find("article") or soup.body or soup

        blocks: list[str] = []
        for element in root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "th"]
        ):
            text = element.get_text(" ", strip=True)
            if not text:
                continue

            if element.name in _HEADINGS:
                blocks.append(f"{_HEADINGS[element.name]} {text}")
            elif element.name == "li":
                blocks.append(f"- {text}")
            else:
                blocks.append(text)

        # Fall back to a flat text dump for pages built entirely from divs.
        if not blocks:
            flat = root.get_text("\n", strip=True)
            blocks = [line for line in flat.splitlines() if line.strip()]

        body = "\n\n".join(_deduplicate(blocks)).strip()
        if not body:
            raise UnparsableDocumentError(
                f"No readable text was found in {filename}. "
                "If the page renders its content with JavaScript, the saved HTML will be empty."
            )

        if title:
            body = f"# {title}\n\n{body}"

        logger.info("html_parsed", extra={"source_name": filename, "blocks": len(blocks)})

        return ParsedDocument(
            pages=[ParsedPage(number=1, text=body)],
            metadata={"title": title, "source_page_count": 1},
        )


def _deduplicate(blocks: list[str]) -> list[str]:
    """Drop repeated blocks while preserving order.

    Nested selectors match the same text twice — a `<td>` inside a table that
    is itself inside a matched element — and duplicated text skews embeddings
    toward whatever got repeated.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for block in blocks:
        if block not in seen:
            seen.add(block)
            unique.append(block)
    return unique
