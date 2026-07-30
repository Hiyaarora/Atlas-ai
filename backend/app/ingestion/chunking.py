"""Recursive character text splitting, implemented from scratch.

This is the single most under-appreciated stage of a RAG pipeline. Retrieval
can only return what chunking produced: a chunk that splits a definition from
its subject makes that fact permanently unretrievable, no matter how good the
embedding model or the reranker is.

The algorithm
-------------
Split on the most semantically meaningful boundary that works, and only fall
back to a cruder one when a piece is still too large:

    paragraph  ->  line  ->  sentence  ->  clause  ->  word  ->  character

A naive `text[i:i+1000]` splitter cuts mid-word and mid-sentence. This one
cuts at a paragraph break when it can, and only ever cuts mid-word for text
that contains no whitespace at all.

Then adjacent pieces are merged back up to `chunk_size`, because a paragraph
of 40 characters is not a useful retrieval unit on its own.

Why overlap
-----------
A sentence spanning a chunk boundary is otherwise half-present in two chunks
and fully present in neither, so it matches poorly against a question about
it. Repeating the tail of each chunk at the head of the next means such a
sentence survives intact in at least one.
"""

from dataclasses import dataclass

# Ordered most- to least-semantic. The trailing "" is the guaranteed
# terminator: splitting on the empty string always succeeds, so recursion
# cannot fail to make progress on pathological input (a 50,000-character
# string with no spaces).
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


@dataclass(frozen=True)
class TextChunk:
    text: str
    #: Position of this chunk within its page, from 0.
    index: int


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` characters."""
    if chunk_overlap >= chunk_size:
        # Overlap at or above chunk size means each chunk restates the whole
        # previous one, so the splitter never advances.
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = _split_recursively(text, chunk_size, separators)
    return _merge(pieces, chunk_size, chunk_overlap)


def _split_recursively(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Break text into pieces that are each at or under `chunk_size`."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # Out of separators: hard-slice. Only reachable if "" was removed
        # from the separator list.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *remaining = separators

    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    if separator not in text:
        # This boundary does not occur; try a cruder one on the same text.
        return _split_recursively(text, chunk_size, tuple(remaining))

    parts = text.split(separator)

    results: list[str] = []
    for position, part in enumerate(parts):
        # Put the separator back, except after the final part. Dropping it
        # would silently delete the punctuation the text was split on.
        restored = part + separator if position < len(parts) - 1 else part
        if not restored.strip():
            continue

        if len(restored) <= chunk_size:
            results.append(restored)
        else:
            results.extend(_split_recursively(restored, chunk_size, tuple(remaining)))

    return results


def _merge(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Combine small pieces into chunks, carrying an overlap between them."""
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for piece in pieces:
        piece_length = len(piece)

        # Adding this piece would overflow: close the current chunk first.
        if current and current_length + piece_length > chunk_size:
            chunks.append("".join(current).strip())

            # Seed the next chunk with the tail of the one just closed.
            overlap_parts: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                if overlap_length + len(previous) > chunk_overlap:
                    break
                overlap_parts.insert(0, previous)
                overlap_length += len(previous)

            current = overlap_parts
            current_length = overlap_length

        current.append(piece)
        current_length += piece_length

    if current:
        tail = "".join(current).strip()
        if tail:
            chunks.append(tail)

    # A chunk can end up empty after stripping (a run of separators).
    return [chunk for chunk in chunks if chunk]


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, TextChunk]]:
    """Chunk each page independently, returning `(page_number, chunk)` pairs.

    Chunking per page rather than across the whole document costs a little
    efficiency at page boundaries but buys something worth more: every chunk
    has exactly one page number, so a citation can point at a real place in
    the document. A chunk spanning pages 4 and 5 can be cited as neither.
    """
    results: list[tuple[int, TextChunk]] = []

    for page_number, page_text in pages:
        for index, text in enumerate(
            split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            results.append((page_number, TextChunk(text=text, index=index)))

    return results
