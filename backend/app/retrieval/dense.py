"""Dense retrieval: the original vector search, extracted unchanged.

The behaviour here is deliberately identical to what `retrieval_service`
performed inline before. Extracting it without altering it is what makes the
comparison meaningful: "hybrid vs baseline" only means something if
the baseline is genuinely the old system.
"""

from __future__ import annotations

import uuid

from app.core.logging import get_logger
from app.embeddings import get_embedding_provider
from app.retrieval.base import RetrievedChunk
from app.vectorstore import get_vector_store

logger = get_logger(__name__)


class DenseRetriever:
    """Nearest-neighbour search over chunk embeddings."""

    name = "dense"

    async def retrieve(
        self,
        query: str,
        *,
        owner_id: uuid.UUID,
        document_ids: list[uuid.UUID],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not document_ids:
            logger.warning("dense_retrieval_without_scope", extra={"owner_id": str(owner_id)})
            return []

        provider = get_embedding_provider()

        # "query", not "document". The same text embedded for the two purposes
        # lands in different places, and matching a question embedded as a
        # document against passages embedded as documents retrieves noticeably
        # worse. This asymmetry is a property of the model, not a convention.
        query_vector = await provider.embed_one(query, purpose="query")

        where: dict[str, object] = {
            "$and": [
                {"owner_id": str(owner_id)},
                {"document_id": {"$in": [str(document_id) for document_id in document_ids]}},
            ]
        }

        hits = await get_vector_store().search(query_vector, top_k=top_k, where=where)

        logger.info(
            "dense_retrieval_completed",
            extra={
                "returned": len(hits),
                "best_score": round(hits[0].score, 3) if hits else None,
            },
        )

        return [
            RetrievedChunk(
                chunk_id=uuid.UUID(hit.id),
                document_id=uuid.UUID(str(hit.metadata.get("document_id"))),
                filename=str(hit.metadata.get("filename", "document")),
                page_number=int(hit.metadata.get("page_number", 1)),
                text=hit.text,
                score=hit.score,
                retriever=self.name,
            )
            for hit in hits
        ]
