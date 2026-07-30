"""Rendering tabular data as text for embedding.

The naive approach — dumping a row as `Ada,Lovelace,1815,London` — destroys
the thing that made it meaningful. Embedded, that row is a bag of
unconnected tokens: nothing associates "1815" with a birth year or "London"
with a city, so a question like "who was born in London?" has almost nothing
to match against.

Rendering each row as `Name: Ada | Surname: Lovelace | Born: 1815 | City:
London` re-attaches every value to its column. The row becomes a small
self-describing sentence, which is the shape embedding models are good at.

The cost is repetition — the header is restated on every row, inflating token
counts. For retrieval that is a good trade: chunks are matched individually,
so each one has to stand on its own.
"""

from typing import Any

#: Rows past this are dropped, with a note. A million-row export is not a
#: knowledge base; the right tool for it is SQL, not vector search.
MAX_ROWS = 5_000


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        # openpyxl returns numeric cells as floats, so "2024" arrives as
        # 2024.0 and embeds as a different token than the year it represents.
        return str(int(value))
    return str(value).strip()


def render_row(headers: list[str], values: list[Any]) -> str:
    """One row as `Header: value | Header: value`, skipping empty cells."""
    parts: list[str] = []

    for index, value in enumerate(values):
        text = format_value(value)
        if not text:
            continue
        header = headers[index].strip() if index < len(headers) and headers[index] else ""
        parts.append(f"{header}: {text}" if header else text)

    return " | ".join(parts)


def looks_like_header(values: list[Any]) -> bool:
    """Heuristic: a header row is all non-empty text with no numbers."""
    rendered = [format_value(value) for value in values]
    if not any(rendered):
        return False
    non_empty = [value for value in rendered if value]
    return all(not _is_number(value) for value in non_empty)


def _is_number(text: str) -> bool:
    try:
        float(text.replace(",", ""))
    except ValueError:
        return False
    return True


def render_table(rows: list[list[Any]]) -> str:
    """Render a whole table, using the first row as headers when it looks like one."""
    if not rows:
        return ""

    if looks_like_header(rows[0]):
        headers = [format_value(value) for value in rows[0]]
        body = rows[1:]
    else:
        headers = []
        body = rows

    lines = [render_row(headers, row) for row in body[:MAX_ROWS]]
    lines = [line for line in lines if line]

    if len(body) > MAX_ROWS:
        lines.append(f"[{len(body) - MAX_ROWS} further rows omitted]")

    return "\n".join(lines)
