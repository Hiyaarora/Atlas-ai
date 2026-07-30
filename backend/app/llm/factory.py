"""Provider selection.

One function decides which implementation the whole application uses. Adding
Claude or a local model means a new module plus one branch here — nothing in
`services/` changes.
"""

from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.echo import EchoProvider
from app.llm.gemini import GeminiProvider

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return the configured provider.

    Cached because provider construction opens an HTTP client and its
    connection pool; building one per request would leak sockets.

    Falls back to `EchoProvider` when Gemini is selected but unconfigured, so
    a missing API key degrades the app to "answers are stubs" rather than
    "every request 500s". The warning is deliberately loud — silently serving
    fake answers in production would be far worse than failing.
    """
    if settings.llm_provider == "echo":
        logger.info("llm_provider_selected", extra={"provider": "echo"})
        return EchoProvider(chunk_delay_seconds=0.02)

    if not settings.gemini_api_key:
        logger.warning(
            "llm_provider_fallback",
            extra={
                "requested": "gemini",
                "using": "echo",
                "reason": "GEMINI_API_KEY is not set",
            },
        )
        return EchoProvider(chunk_delay_seconds=0.02)

    logger.info(
        "llm_provider_selected",
        extra={"provider": "gemini", "model": settings.gemini_model},
    )
    return GeminiProvider()
