"""Deterministic test double for the LLM provider interface.

It earns its place two ways:

* The test suite exercises the *entire* streaming path — SSE framing, DB
  persistence, client-disconnect handling — with zero API cost, zero latency,
  and byte-identical output on every run. Tests that call a real LLM are slow,
  expensive, and flaky, because the model's reply changes between runs.
* It proves the abstraction: if this can be swapped in without the service
  layer noticing, so can any other provider.
"""

import asyncio
from collections.abc import AsyncIterator

from app.core.exceptions import LLMError
from app.llm.base import ChatMessage, LLMProvider


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self, model: str = "echo-1", chunk_delay_seconds: float = 0.0) -> None:
        self.model = model
        # A small delay makes streaming visible when developing the UI. Left
        # at zero in tests so the suite stays fast.
        self._chunk_delay = chunk_delay_seconds

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if not messages:
            raise LLMError("Cannot generate a reply from an empty conversation.")

        last_user_message = next(
            (message for message in reversed(messages) if message.role == "user"),
            None,
        )
        if last_user_message is None:
            raise LLMError("Cannot generate a reply without a user message.")

        reply = (
            f"Echo from {self.model}: {last_user_message.content} "
            f"(turn {len(messages)} of this conversation)"
        )

        # Emit word by word so consumers exercise real multi-chunk assembly
        # rather than receiving one convenient blob.
        for index, word in enumerate(reply.split(" ")):
            if self._chunk_delay:
                await asyncio.sleep(self._chunk_delay)
            yield word if index == 0 else f" {word}"
