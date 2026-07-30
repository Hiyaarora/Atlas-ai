"""Provider-agnostic embedding layer."""

from app.embeddings.base import EmbeddingProvider, EmbeddingPurpose
from app.embeddings.factory import get_embedding_provider

__all__ = ["EmbeddingProvider", "EmbeddingPurpose", "get_embedding_provider"]
