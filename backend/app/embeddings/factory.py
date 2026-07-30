"""Embedding provider selection."""

from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider
from app.embeddings.fake import FakeEmbeddingProvider
from app.embeddings.gemini import GeminiEmbeddingProvider

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured provider.

    Unlike the chat provider, a missing API key here does NOT silently fall
    back to the fake one in production. Stub answers are visibly wrong; stub
    *embeddings* are invisibly wrong — retrieval would return plausible-looking
    but meaningless matches, and the vectors would be silently incompatible
    with anything indexed later by the real model.
    """
    if settings.embedding_provider == "fake":
        logger.info("embedding_provider_selected", extra={"provider": "fake"})
        return FakeEmbeddingProvider(dimensions=settings.embedding_dimensions)

    logger.info(
        "embedding_provider_selected",
        extra={"provider": "gemini", "model": settings.gemini_embedding_model},
    )
    return GeminiEmbeddingProvider()
