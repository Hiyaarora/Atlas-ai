"""Query rewriting for conversational follow-ups.

The heuristic gate carries most of the weight here. Rewriting costs an LLM
request, so a gate that fires too often burns a user's daily quota on
questions that never needed it — and one that fires too rarely leaves
follow-ups retrieving noise. These tests pin both directions.

The LLM itself is stubbed: asserting that a model produces one particular
sentence is a test of someone else's weights, not of this code.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.llm.base import ChatMessage
from app.services import query_rewrite_service as qr


def _history(*turns: tuple[str, str]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in turns]


CONVERSATION = _history(
    ("user", "What is multi-head attention?"),
    ("assistant", "Multi-head attention runs several attention operations in parallel."),
)


# ---------------------------------------------------------------------------
# The gate: when is an LLM call worth making?
# ---------------------------------------------------------------------------


def test_a_first_message_is_never_rewritten() -> None:
    """No prior turns means nothing to resolve against. Also the most common
    case, so it is checked first and costs nothing."""
    assert not qr.needs_rewriting("What is multi-head attention?", [])


def test_history_without_an_assistant_turn_is_not_enough() -> None:
    """Two user messages and no reply is a user talking to themselves — there
    is no established subject for a pronoun to refer back to."""
    assert not qr.needs_rewriting("what about it?", _history(("user", "hello")))


def test_a_pronoun_triggers_rewriting() -> None:
    assert qr.needs_rewriting("what about its complexity?", CONVERSATION)


@pytest.mark.parametrize(
    "question",
    ["why?", "and the second one?", "in Python?", "how fast is that"],
)
def test_a_very_short_question_triggers_rewriting(question: str) -> None:
    """Elliptical questions cannot stand alone regardless of wording."""
    assert qr.needs_rewriting(question, CONVERSATION)


def test_a_self_contained_question_is_left_alone() -> None:
    """The expensive case to get wrong: this would work perfectly well as a
    retrieval query, so rewriting it spends a request for nothing."""
    assert not qr.needs_rewriting(
        "What learning rate schedule did the authors use for training?", CONVERSATION
    )


def test_referring_words_are_matched_whole() -> None:
    """ "its" must not fire on "digits" — substring matching would rewrite
    almost every question and quietly double the request cost."""
    assert not qr.needs_rewriting(
        "How many digits of precision does the encoder retain overall?", CONVERSATION
    )


def test_the_feature_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    assert not qr.needs_rewriting("what about its complexity?", CONVERSATION)


# ---------------------------------------------------------------------------
# The rewrite itself
# ---------------------------------------------------------------------------


class _StubProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, messages, *, system_prompt=None):  # noqa: ANN001
        self.prompts.append(messages[0].content)
        return self.reply


def _use(monkeypatch: pytest.MonkeyPatch, provider) -> None:  # noqa: ANN001
    monkeypatch.setattr(qr, "get_llm_provider", lambda: provider)


async def test_a_follow_up_is_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _StubProvider("What is the computational complexity of multi-head attention?")
    _use(monkeypatch, provider)

    result = await qr.rewrite_for_retrieval("what about its complexity?", CONVERSATION)

    assert result == "What is the computational complexity of multi-head attention?"


async def test_the_conversation_is_given_to_the_rewriter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the transcript the model cannot know what "it" refers to."""
    provider = _StubProvider("resolved")
    _use(monkeypatch, provider)

    await qr.rewrite_for_retrieval("what about it?", CONVERSATION)

    assert "multi-head attention" in provider.prompts[0]


async def test_a_gated_question_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _StubProvider("should not be used")
    _use(monkeypatch, provider)

    question = "What learning rate schedule did the authors use for training?"
    assert await qr.rewrite_for_retrieval(question, CONVERSATION) == question
    assert provider.prompts == []


async def test_a_provider_failure_falls_back_to_the_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieval must still happen. A degraded query beats no answer, and the
    original question is exactly what the system used before this existed."""
    from app.core.exceptions import LLMError

    class _Broken:
        async def complete(self, messages, *, system_prompt=None):  # noqa: ANN001
            raise LLMError("provider down")

    _use(monkeypatch, _Broken())

    assert await qr.rewrite_for_retrieval("what about it?", CONVERSATION) == "what about it?"


async def test_an_answer_masquerading_as_a_rewrite_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that ignores the instruction and answers instead would make the
    retrieval query the model's own words rather than the user's."""
    _use(monkeypatch, _StubProvider("Multi-head attention has complexity O(n^2 d). " * 20))

    assert await qr.rewrite_for_retrieval("what about it?", CONVERSATION) == "what about it?"


async def test_an_empty_rewrite_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, _StubProvider("   "))

    assert await qr.rewrite_for_retrieval("what about it?", CONVERSATION) == "what about it?"


async def test_surrounding_quotes_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models routinely wrap a rewritten question in quotes; searching for a
    quoted string is not the same query."""
    _use(monkeypatch, _StubProvider('"What is the complexity of attention?"'))

    result = await qr.rewrite_for_retrieval("what about it?", CONVERSATION)

    assert result == "What is the complexity of attention?"
