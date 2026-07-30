"""Gemini provider.

The only module in Atlas AI permitted to import `google.genai`. Everything
above it speaks `LLMProvider`.
"""

import asyncio
from collections.abc import AsyncIterator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import settings
from app.core.exceptions import LLMConfigurationError, LLMError, LLMRateLimitError
from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMProvider

logger = get_logger(__name__)

# Gemini labels the assistant turn "model"; the rest of the industry says
# "assistant". Translating here keeps the vendor's vocabulary out of our
# domain model.
_ROLE_TO_GEMINI = {"user": "user", "assistant": "model"}


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.gemini_api_key
        if not key:
            raise LLMConfigurationError("GEMINI_API_KEY is not set. Add it to .env.")

        self.model = model or settings.gemini_model
        # The client is cheap to hold and manages its own connection pool, so
        # it is created once per provider instance rather than per request.
        self._client = genai.Client(api_key=key)

    def _build_config(self, system_prompt: str | None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    @staticmethod
    def _to_gemini_contents(messages: list[ChatMessage]) -> list[types.Content]:
        return [
            types.Content(
                role=_ROLE_TO_GEMINI[message.role],
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if not messages:
            raise LLMError("Cannot generate a reply from an empty conversation.")

        try:
            # `generate_content_stream` is an `async def` that RETURNS an
            # async iterator — it is not itself an async generator. So it must
            # be awaited first, then iterated. `async for` on the un-awaited
            # coroutine raises TypeError.
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=self._to_gemini_contents(messages),
                config=self._build_config(system_prompt),
            )

            async for response in stream:
                # A chunk can legitimately carry no text: safety blocks and
                # the final usage-metadata chunk both arrive with text=None.
                text = getattr(response, "text", None)
                if text:
                    yield text

        except asyncio.CancelledError:
            # The client disconnected. Propagate so the caller can persist
            # whatever was generated; swallowing this would hang the task.
            logger.info("gemini_stream_cancelled", extra={"model": self.model})
            raise

        except genai_errors.ClientError as exc:
            # 4xx from Google: our request was wrong, or we are over quota.
            status_code = getattr(exc, "code", None)
            if status_code == 429:
                logger.warning("gemini_rate_limited", extra={"model": self.model})
                raise LLMRateLimitError() from exc
            logger.error(
                "gemini_client_error",
                extra={"model": self.model, "status": status_code, "error": str(exc)},
            )
            raise LLMError("The language model rejected the request.") from exc

        except genai_errors.ServerError as exc:
            logger.error("gemini_server_error", extra={"model": self.model, "error": str(exc)})
            raise LLMError("The language model provider is unavailable.") from exc

        except genai_errors.APIError as exc:
            logger.error("gemini_api_error", extra={"model": self.model, "error": str(exc)})
            raise LLMError() from exc
