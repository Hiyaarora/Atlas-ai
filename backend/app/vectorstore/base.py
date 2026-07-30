"""Vector store contract.

Chroma is the right choice for a single-node app: embedded, zero-ops, good
enough to tens of millions of vectors. It is also the component most likely
to be replaced — by pgvector to collapse two datastores into one, or by
Qdrant/Weaviate at scale.

So retrieval depends on this interface, not on Chroma. The interface exposes
only what a vector index genuinely provides: upsert, filtered similarity
search, delete. No Chroma `Collection` object escapes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    """One indexed chunk."""

    id: str
    embedding: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    id: str
    text: str
    #: Cosine similarity in [0, 1]. Higher is more similar.
    #
    # Deliberately a *similarity*, not the distance Chroma returns. Every
    # store reports distance differently (cosine, L2, inner product), and
    # leaking that would put vendor-specific arithmetic into business logic.
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records. Must be idempotent by id, so a retried
        ingestion does not duplicate chunks."""

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Nearest neighbours, filtered by metadata.

        `where` is not optional in practice: it carries the tenant filter.
        Retrieval without it would return every user's chunks.
        """

    @abstractmethod
    async def delete(
        self, *, ids: list[str] | None = None, where: dict[str, Any] | None = None
    ) -> None:
        """Remove records by id or by metadata filter."""

    @abstractmethod
    async def count(self, *, where: dict[str, Any] | None = None) -> int:
        """Number of indexed records, for diagnostics."""
