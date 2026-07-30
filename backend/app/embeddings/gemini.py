"""Gemini embeddings."""

import asyncio
import random

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import settings
from app.core.exceptions import EmbeddingError, LLMConfigurationError, LLMRateLimitError
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider, EmbeddingPurpose

logger = get_logger(__name__)

# Google's task hints. Using RETRIEVAL_DOCUMENT for passages and
# RETRIEVAL_QUERY for questions places them in a shared space optimised for
# matching one against the other, rather than for general similarity.
_TASK_TYPES: dict[EmbeddingPurpose, str] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}

#: Substring identifying the per-day quota in a 429 body. Google returns the
#: same status code for the per-minute and per-day caps; only the quotaId
#: separates "wait a few seconds" from "come back tomorrow".
_PER_DAY_QUOTA = "PerDay"

_MAX_RETRIES = 4
_RETRY_BASE_DELAY_SECONDS = 2.0


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.gemini_api_key
        if not key:
            raise LLMConfigurationError("GEMINI_API_KEY is not set. Add it to .env.")

        self.model = model or settings.gemini_embedding_model
        self.dimensions = settings.embedding_dimensions
        self._client = genai.Client(api_key=key)

    async def embed(self, texts: list[str], *, purpose: EmbeddingPurpose) -> list[list[float]]:
        if not texts:
            return []

        response = await self._embed_with_retry(texts, purpose=purpose)

        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            # Vectors are zipped against chunk metadata downstream. A count
            # mismatch would attach vectors to the wrong chunks, producing
            # confidently wrong citations — fail loudly instead.
            raise EmbeddingError(
                f"Provider returned {len(embeddings)} vectors for {len(texts)} inputs."
            )

        return [list(embedding.values or []) for embedding in embeddings]

    async def _embed_with_retry(self, texts: list[str], *, purpose: EmbeddingPurpose):
        """Call the API, retrying transient rate limits with backoff.

        WHY THIS EXISTS
        ---------------
        Free-tier Gemini enforces a per-minute request cap separately from the
        per-day one, and a burst trips it in seconds. Without a retry, ordinary
        bursts fail: ingesting several documents, or the evaluation harness
        embedding twenty queries in a row.

        The consequence was not a visible error. `hybrid_search` degrades
        gracefully when a retriever raises, so a rate-limited dense branch
        silently reduced hybrid search to lexical-only — and the evaluation
        harness dutifully reported that hybrid retrieval had destroyed recall
        (1.000 -> 0.312) when the retrieval code was fine. Silent degradation
        plus measurement produces confident, wrong conclusions.

        Only the per-MINUTE limit is retried. Exhausting the daily quota is not
        transient, and sleeping through it would turn a clear error into a hung
        request, so that still raises immediately.
        """
        delay = _RETRY_BASE_DELAY_SECONDS

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await self._client.aio.models.embed_content(
                    model=self.model,
                    contents=list(texts),
                    config=types.EmbedContentConfig(
                        task_type=_TASK_TYPES[purpose],
                        output_dimensionality=self.dimensions,
                    ),
                )
            except genai_errors.ClientError as exc:
                if getattr(exc, "code", None) != 429:
                    logger.error("embedding_client_error", extra={"error": str(exc)})
                    raise EmbeddingError("The embedding request was rejected.") from exc

                # Google reports both caps as HTTP 429 and distinguishes them
                # only in the quotaId. Retrying a daily exhaustion would sleep
                # for nothing.
                if _PER_DAY_QUOTA in str(exc) or attempt == _MAX_RETRIES:
                    logger.warning(
                        "embedding_rate_limited",
                        extra={"model": self.model, "attempts": attempt},
                    )
                    raise LLMRateLimitError() from exc

                # Jitter: several ingestion batches retrying in lockstep would
                # otherwise re-collide on exactly the same schedule.
                wait = delay + random.uniform(0.0, delay / 2.0)
                logger.info(
                    "embedding_rate_limited_retrying",
                    extra={"attempt": attempt, "sleep_seconds": round(wait, 2)},
                )
                await asyncio.sleep(wait)
                delay *= 2
            except genai_errors.APIError as exc:
                logger.error("embedding_api_error", extra={"error": str(exc)})
                raise EmbeddingError() from exc

        raise EmbeddingError("Exhausted embedding retries.")
