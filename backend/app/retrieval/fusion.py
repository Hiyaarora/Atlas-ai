"""Reciprocal Rank Fusion.

THE PROBLEM
-----------
Dense search returns cosine similarity in [0, 1], where 0.68 is a good match
and 0.58 is noise. BM25 returns an unbounded sum of idf-weighted terms, where
a good match might be 3.1 or 24.0 depending on corpus size, query length and
how rare the matched terms happen to be.

Combining them by addition is meaningless — one scale would simply dominate.
The obvious repair, min-max normalising each list into [0, 1], is worse than
it looks: the normalised value depends on which candidates happened to come
back, so the same chunk scores differently depending on its *competition*. Add
one strong result and every other score shifts. That is not a relevance
signal.

THE FIX
-------
Throw the scores away and use only rank:

    RRF(chunk) = SUM over retrievers of  1 / (k + rank)

Rank is ordinal and dimensionless, so it is directly comparable across any two
rankers regardless of what their scores meant. There is nothing to calibrate
and no per-corpus tuning, which is why RRF became the default fusion method
despite being this simple.

WHY k = 60
----------
`k` controls how sharply top ranks are rewarded. With k=60, rank 1 scores
1/61 and rank 2 scores 1/62 — a 1.6% difference. A small k (say 1) makes rank
1 worth twice rank 2, so a single confident retriever dictates the result. A
large k flattens everything toward equality, and agreement between retrievers
becomes the deciding factor.

60 comes from Cormack et al. (2009) and has held up well; it says "being found
by *both* retrievers matters more than being ranked first by one". For hybrid
search that is exactly the property we want — a chunk both methods surface is
much more likely to be relevant than one only BM25 loved.

WHAT SURVIVES FUSION
--------------------
`FusedChunk` keeps each retriever's original rank and score. Two reasons:
observability (a citation can be traced to *why* it surfaced), and gating —
the pipeline still needs the native scores to decide relevance, because an RRF
score of 0.016 says nothing about whether a chunk is worth showing.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import settings
from app.retrieval.base import RetrievedChunk


@dataclass(frozen=True)
class FusedChunk:
    """One chunk after fusion, with its provenance intact."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    text: str

    #: Combined RRF score. Use for ORDERING only — it is a rank artefact and
    #: carries no information about absolute relevance.
    score: float

    #: retriever name -> 1-based rank in that retriever's list.
    ranks: dict[str, int] = field(default_factory=dict)

    #: retriever name -> that retriever's native score.
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def retrievers(self) -> tuple[str, ...]:
        """Which retrievers found this chunk. Length > 1 means agreement."""
        return tuple(sorted(self.ranks))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int | None = None,
    top_k: int | None = None,
) -> list[FusedChunk]:
    """Fuse several ranked lists into one.

    Input lists must already be ordered best-first; rank is taken from
    position, so an unsorted list would silently produce a wrong fusion.
    """
    rrf_k = settings.retrieval_rrf_k if k is None else k

    accumulated: dict[uuid.UUID, FusedChunk] = {}

    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            contribution = 1.0 / (rrf_k + position)
            existing = accumulated.get(chunk.chunk_id)

            if existing is None:
                accumulated[chunk.chunk_id] = FusedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=contribution,
                    ranks={chunk.retriever: position},
                    scores={chunk.retriever: chunk.score},
                )
            else:
                # Frozen dataclass: rebuild rather than mutate. The dicts are
                # copied too, so a caller holding an earlier reference does not
                # observe it change underneath them.
                accumulated[chunk.chunk_id] = FusedChunk(
                    chunk_id=existing.chunk_id,
                    document_id=existing.document_id,
                    filename=existing.filename,
                    page_number=existing.page_number,
                    # Keep the first text seen. Retrievers read the same rows,
                    # so they agree; if they ever diverge, the earlier list in
                    # the argument order wins deterministically.
                    text=existing.text,
                    score=existing.score + contribution,
                    ranks={**existing.ranks, chunk.retriever: position},
                    scores={**existing.scores, chunk.retriever: chunk.score},
                )

    fused = sorted(
        accumulated.values(),
        # Tie-break on chunk_id so equal scores produce a stable order rather
        # than one that varies with dict iteration. Without this, evaluation
        # runs would not be reproducible.
        key=lambda chunk: (-chunk.score, str(chunk.chunk_id)),
    )

    return fused[:top_k] if top_k is not None else fused
