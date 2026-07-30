"""ChromaDB-backed vector store.

Every method here wraps a synchronous Chroma call in `asyncio.to_thread`.
Chroma's client has no async API, and calling it directly from an async
endpoint would block the event loop for the duration of the query — the same
failure measured with bcrypt, where an unrelated health
check went from 2ms to 1459ms.
"""

import asyncio
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.vectorstore.base import SearchHit, VectorRecord, VectorStore

logger = get_logger(__name__)

COLLECTION_NAME = "atlas_chunks"


class ChromaVectorStore(VectorStore):
    def __init__(self, path: str | None = None) -> None:
        directory = Path(path or settings.chroma_dir)
        directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            # Cosine, not the L2 default. Embedding providers return
            # L2-normalised vectors, for which cosine similarity is the
            # meaningful comparison; L2 on normalised vectors is a monotone
            # transform of it, but the distances no longer map cleanly to a
            # [0,1] score we can threshold.
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("chroma_ready", extra={"path": str(directory)})

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        def _upsert() -> None:
            self._collection.upsert(
                ids=[record.id for record in records],
                embeddings=[record.embedding for record in records],
                documents=[record.text for record in records],
                metadatas=[record.metadata for record in records],
            )

        try:
            await asyncio.to_thread(_upsert)
        except Exception as exc:  # noqa: BLE001
            logger.error("chroma_upsert_failed", extra={"count": len(records), "error": str(exc)})
            raise VectorStoreError("Could not index the document.") from exc

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        def _query() -> Any:
            return self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        try:
            result = await asyncio.to_thread(_query)
        except Exception as exc:  # noqa: BLE001
            logger.error("chroma_query_failed", extra={"error": str(exc)})
            raise VectorStoreError("Knowledge search failed.") from exc

        # Chroma returns one list per query embedding; we always send one.
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for index, chunk_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else 1.0
            hits.append(
                SearchHit(
                    id=chunk_id,
                    text=documents[index] if index < len(documents) else "",
                    # Chroma's cosine "distance" is 1 - cosine_similarity.
                    # Clamped because floating-point error can push it a
                    # hair outside [0, 1], and a negative score would be
                    # nonsense to threshold against.
                    score=max(0.0, min(1.0, 1.0 - float(distance))),
                    metadata=dict(metadatas[index]) if index < len(metadatas) else {},
                )
            )

        return hits

    async def delete(
        self, *, ids: list[str] | None = None, where: dict[str, Any] | None = None
    ) -> None:
        if not ids and not where:
            # A delete with neither filter would wipe the collection.
            raise ValueError("delete requires ids or a where filter")

        def _delete() -> None:
            self._collection.delete(ids=ids, where=where)

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:  # noqa: BLE001
            logger.error("chroma_delete_failed", extra={"error": str(exc)})
            raise VectorStoreError("Could not remove the document from the index.") from exc

    async def count(self, *, where: dict[str, Any] | None = None) -> int:
        def _count() -> int:
            if where is None:
                return self._collection.count()
            found = self._collection.get(where=where, include=[])
            return len(found.get("ids") or [])

        try:
            return await asyncio.to_thread(_count)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("Could not read the index.") from exc
