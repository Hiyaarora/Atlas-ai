"""The LLM abstraction itself.

These tests never touch the network. They verify the *contract* every
provider must satisfy, which is what lets the service layer stay
provider-agnostic.
"""

import pytest

from app.core.exceptions import LLMConfigurationError, LLMError
from app.llm.base import ChatMessage, LLMProvider
from app.llm.echo import EchoProvider
from app.llm.factory import get_llm_provider
from app.llm.gemini import GeminiProvider


async def test_echo_provider_streams_multiple_chunks() -> None:
    """Consumers must handle assembly, so a provider must not send one blob."""
    provider = EchoProvider()

    chunks = [c async for c in provider.stream_chat([ChatMessage(role="user", content="hello")])]

    assert len(chunks) > 1
    assert "hello" in "".join(chunks)


async def test_complete_is_the_concatenation_of_the_stream() -> None:
    """`complete()` is defined on the base class in terms of `stream_chat`."""
    provider = EchoProvider()
    messages = [ChatMessage(role="user", content="hello")]

    streamed = "".join([c async for c in provider.stream_chat(messages)])
    completed = await provider.complete(messages)

    assert streamed == completed


async def test_empty_history_is_rejected() -> None:
    provider = EchoProvider()

    with pytest.raises(LLMError):
        [c async for c in provider.stream_chat([])]


async def test_history_without_a_user_turn_is_rejected() -> None:
    provider = EchoProvider()

    with pytest.raises(LLMError):
        [c async for c in provider.stream_chat([ChatMessage(role="assistant", content="hi")])]


def test_gemini_requires_an_api_key() -> None:
    """Misconfiguration must fail loudly at construction, not mid-stream."""
    with pytest.raises(LLMConfigurationError):
        GeminiProvider(api_key="")


def test_factory_falls_back_to_echo_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key degrades to stub answers rather than 500ing every chat."""
    from app.core import config

    monkeypatch.setattr(config.settings, "llm_provider", "gemini")
    monkeypatch.setattr(config.settings, "gemini_api_key", None)
    get_llm_provider.cache_clear()

    try:
        provider = get_llm_provider()
        assert isinstance(provider, EchoProvider)
    finally:
        get_llm_provider.cache_clear()


def test_providers_satisfy_the_interface() -> None:
    """Every provider must expose the attributes the service layer reads."""
    provider = EchoProvider()

    assert isinstance(provider, LLMProvider)
    assert isinstance(provider.name, str) and provider.name
    assert isinstance(provider.model, str) and provider.model


def test_gemini_translates_roles_to_googles_vocabulary() -> None:
    """Gemini calls the assistant turn "model"; our domain says "assistant"."""
    contents = GeminiProvider._to_gemini_contents(
        [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
    )

    assert [c.role for c in contents] == ["user", "model"]
    assert contents[0].parts[0].text == "hi"
