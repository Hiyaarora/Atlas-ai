"""Shared vocabulary for the retrieval pipeline.

Every retriever answers the same question — "which chunks match this query?" —
and returns the same shape. That uniformity is what lets the pipeline add,
remove or reorder retrievers without touching generation or citations.

The one thing a `RetrievedChunk` deliberately does NOT promise is that `score`
means the same thing across retrievers. Cosine similarity is bounded in [0, 1]
and roughly calibrated; BM25 is unbounded and depends on corpus statistics.
Comparing or summing them is meaningless, and pretending otherwise is the most
common bug in hand-rolled hybrid search. Fusion therefore uses *rank*, never
score — see `fusion.py`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by one retriever."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    text: str

    #: Retriever-native relevance. Only comparable against scores from the
    #: SAME retriever for the SAME query.
    score: float

    #: Which retriever produced this, e.g. "dense" or "lexical". Carried
    #: through fusion so a citation can be traced back to why it surfaced.
    retriever: str


@runtime_checkable
class Retriever(Protocol):
    """What the pipeline requires of a retrieval strategy.

    A Protocol rather than an ABC: retrievers have genuinely different
    dependencies (the dense one needs a vector store and an embedding
    provider, the lexical one needs a database session), so they are
    constructed differently and only need to agree on this one method.
    """

    #: Stable identifier used in fusion bookkeeping and logs.
    name: str

    async def retrieve(
        self,
        query: str,
        *,
        owner_id: uuid.UUID,
        document_ids: list[uuid.UUID],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks, best first.

        `document_ids` is required and must be honoured by every
        implementation. It is the conversation-isolation boundary: a retriever
        that ignored it would silently reintroduce cross-document leakage that
        the rest of the system is built to prevent.
        """
        ...
