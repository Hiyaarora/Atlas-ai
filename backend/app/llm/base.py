"""The contract every LLM provider implements.

This interface is the whole point of the `llm` package. Business logic depends
on `LLMProvider`, never on `google.genai`, so switching to Claude, OpenAI, or
a local model is a new file plus one config value — not a refactor.

Two rules keep the abstraction honest:

1. Nothing provider-specific leaks through. No Gemini `Content` objects, no
   OpenAI `ChatCompletionChunk`. Callers see `ChatMessage` and plain strings.
2. Provider errors are translated into Atlas domain errors here, so the rest
   of the app never imports a vendor exception type.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a conversation, provider-independent.

    Frozen because a message that has been sent is history: mutating it after
    the fact would silently desynchronise the model's view from the database.
    """

    role: Role
    content: str


class LLMProvider(ABC):
    """Abstract chat-completion provider."""

    #: Short identifier for logs and analytics, e.g. "gemini".
    name: str

    #: The concrete model in use, e.g. "gemini-2.5-flash". Recorded on every
    #: assistant message so answer quality can be attributed to a model later.
    model: str

    @abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield the reply incrementally, as text fragments.

        Note the signature: this is a plain method returning an async
        iterator, not an `async def`. That lets implementations be async
        generators while callers uniformly write `async for chunk in
        provider.stream_chat(...)`.

        Fragments are whatever the provider emits — a token, a word, or a
        sentence. Callers must concatenate rather than assume any boundary.

        Raises:
            LLMError: the provider failed. Never a vendor-specific exception.
        """

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Non-streaming convenience built on `stream_chat`.

        Query rewriting and retrieval evaluation need this,
        where partial output is useless and only the final string matters.
        Defined once here so no provider has to implement it twice.
        """
        parts = [chunk async for chunk in self.stream_chat(messages, system_prompt=system_prompt)]
        return "".join(parts)
