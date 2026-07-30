"""Vector index layer."""

from functools import lru_cache

from app.core.config import settings
from app.vectorstore.base import SearchHit, VectorRecord, VectorStore
from app.vectorstore.chroma import ChromaVectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the process-wide store.

    Cached because Chroma's PersistentClient owns the on-disk index and its
    HNSW graph in memory. Constructing one per request would reload the index
    every time and, on Windows, contend for file locks.
    """
    return ChromaVectorStore(path=settings.chroma_dir)


__all__ = ["SearchHit", "VectorRecord", "VectorStore", "get_vector_store"]
