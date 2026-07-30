"""Conversation CRUD, ownership isolation, and SSE streaming."""

import json
import uuid
from contextlib import asynccontextmanager

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.llm.echo import EchoProvider
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.auth import RegisterRequest
from app.services import auth_service, chat_service
from tests.conftest import TEST_PASSWORD

PREFIX = settings.api_v1_prefix


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs.

    Written by hand rather than mocked, so the tests fail if the wire format
    is malformed - a missing blank-line terminator is invisible to a test that
    only inspects Python objects.
    """
    events: list[tuple[str, dict]] = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        name: str | None = None
        payload: str | None = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = line.removeprefix("data: ")
        if name and payload is not None:
            events.append((name, json.loads(payload)))
    return events


@pytest.fixture
async def conversation_id(client: AsyncClient, auth_headers: dict[str, str]) -> str:
    response = await client.post(f"{PREFIX}/conversations", json={}, headers=auth_headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ==========================================================================
# CRUD
# ==========================================================================


async def test_conversations_require_authentication(client: AsyncClient) -> None:
    assert (await client.get(f"{PREFIX}/conversations")).status_code == 401
    assert (await client.post(f"{PREFIX}/conversations", json={})).status_code == 401


async def test_create_and_list_conversations(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await client.post(
        f"{PREFIX}/conversations", json={"title": "Research notes"}, headers=auth_headers
    )
    listed = await client.get(f"{PREFIX}/conversations", headers=auth_headers)

    assert created.status_code == 201
    assert created.json()["title"] == "Research notes"
    assert [c["id"] for c in listed.json()] == [created.json()["id"]]


async def test_new_conversation_gets_a_default_title(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(f"{PREFIX}/conversations", json={}, headers=auth_headers)

    assert response.json()["title"] == "New conversation"


async def test_conversation_detail_starts_empty(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    response = await client.get(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["messages"] == []


async def test_rename_conversation(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    response = await client.patch(
        f"{PREFIX}/conversations/{conversation_id}",
        json={"title": "Renamed"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Renamed"


async def test_delete_conversation(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    deleted = await client.delete(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)
    fetched = await client.get(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)

    assert deleted.status_code == 204
    assert fetched.status_code == 404


async def test_unknown_conversation_returns_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(f"{PREFIX}/conversations/{uuid.uuid4()}", headers=auth_headers)

    assert response.status_code == 404


# ==========================================================================
# Ownership isolation - the security property that matters most
# ==========================================================================


@pytest.fixture
async def other_users_conversation(db_session: AsyncSession) -> Conversation:
    """A conversation belonging to a completely different account."""
    other = await auth_service.register_user(
        db_session,
        RegisterRequest(email="intruder-target@example.com", password=TEST_PASSWORD),
    )
    conversation = Conversation(user_id=other.id, title="Private")
    db_session.add(conversation)
    await db_session.commit()
    await db_session.refresh(conversation)
    return conversation


async def test_cannot_read_another_users_conversation(
    client: AsyncClient, auth_headers: dict[str, str], other_users_conversation: Conversation
) -> None:
    response = await client.get(
        f"{PREFIX}/conversations/{other_users_conversation.id}", headers=auth_headers
    )

    # 404 and not 403: a 403 would confirm the conversation exists.
    assert response.status_code == 404


async def test_cannot_delete_another_users_conversation(
    client: AsyncClient,
    auth_headers: dict[str, str],
    other_users_conversation: Conversation,
    db_session: AsyncSession,
) -> None:
    response = await client.delete(
        f"{PREFIX}/conversations/{other_users_conversation.id}", headers=auth_headers
    )

    assert response.status_code == 404

    still_there = await db_session.get(Conversation, other_users_conversation.id)
    assert still_there is not None, "another user's conversation must survive"


async def test_cannot_post_into_another_users_conversation(
    client: AsyncClient, auth_headers: dict[str, str], other_users_conversation: Conversation
) -> None:
    response = await client.post(
        f"{PREFIX}/conversations/{other_users_conversation.id}/messages",
        json={"content": "leak please"},
        headers=auth_headers,
    )

    assert response.status_code == 404


async def test_list_only_returns_your_own_conversations(
    client: AsyncClient, auth_headers: dict[str, str], other_users_conversation: Conversation
) -> None:
    response = await client.get(f"{PREFIX}/conversations", headers=auth_headers)

    ids = [c["id"] for c in response.json()]
    assert str(other_users_conversation.id) not in ids


# ==========================================================================
# Streaming
# ==========================================================================


async def test_send_message_streams_sse_events(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    response = await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "What is retrieval augmented generation?"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    names = [name for name, _ in events]

    assert names.count("token") > 1, "the reply must arrive in multiple chunks"
    assert names[-1] == "done"
    assert "error" not in names


async def test_streamed_tokens_reassemble_into_the_saved_message(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    """What the user watched appear must equal what was persisted."""
    response = await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "hello there"},
        headers=auth_headers,
    )
    events = parse_sse(response.text)

    streamed = "".join(data["text"] for name, data in events if name == "token")
    done = next(data for name, data in events if name == "done")

    assert streamed == done["content"]

    detail = await client.get(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)
    assistant = [m for m in detail.json()["messages"] if m["role"] == "assistant"]
    assert assistant[-1]["content"] == streamed


async def test_both_turns_are_persisted(
    client: AsyncClient,
    auth_headers: dict[str, str],
    conversation_id: str,
    db_session: AsyncSession,
) -> None:
    await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "first question"},
        headers=auth_headers,
    )

    result = await db_session.execute(
        select(Message)
        .where(Message.conversation_id == uuid.UUID(conversation_id))
        .order_by(Message.created_at, Message.id)
    )
    messages = list(result.scalars().all())

    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "first question"
    assert messages[1].model, "the assistant message must record which model produced it"


async def test_assistant_message_records_the_model(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "hi"},
        headers=auth_headers,
    )
    detail = await client.get(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)

    assistant = [m for m in detail.json()["messages"] if m["role"] == "assistant"][0]
    user_message = [m for m in detail.json()["messages"] if m["role"] == "user"][0]

    assert assistant["model"] is not None
    assert user_message["model"] is None, "user messages have no model"


async def test_first_message_titles_the_conversation(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "How does hybrid retrieval work?"},
        headers=auth_headers,
    )
    response = await client.get(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)

    assert response.json()["title"] == "How does hybrid retrieval work?"


async def test_long_first_message_is_truncated_for_the_title(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "word " * 100},
        headers=auth_headers,
    )
    response = await client.get(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)

    title = response.json()["title"]
    assert len(title) <= 60
    assert title.endswith("…")


async def test_conversation_history_is_replayed_to_the_model(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    """EchoProvider reports the turn count, proving history reaches it."""
    await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "first"},
        headers=auth_headers,
    )
    second = await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "second"},
        headers=auth_headers,
    )

    done = next(data for name, data in parse_sse(second.text) if name == "done")
    assert "turn 3 of this conversation" in done["content"]


async def test_blank_message_is_rejected(
    client: AsyncClient, auth_headers: dict[str, str], conversation_id: str
) -> None:
    response = await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "   "},
        headers=auth_headers,
    )

    assert response.status_code == 422


async def test_user_message_survives_when_generation_fails(
    client: AsyncClient,
    auth_headers: dict[str, str],
    conversation_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user's typing must never be lost to an upstream outage."""
    from app.core.exceptions import LLMError
    from app.llm.echo import EchoProvider

    async def exploding_stream(self, messages, *, system_prompt=None):  # noqa: ANN001
        raise LLMError("provider exploded")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(EchoProvider, "stream_chat", exploding_stream)

    response = await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "precious question"},
        headers=auth_headers,
    )

    # Status is still 200: headers were already sent before generation began.
    assert response.status_code == 200
    events = parse_sse(response.text)
    # `sources` is always emitted first, even when retrieval found nothing.
    assert [name for name, _ in events] == ["sources", "error"]
    assert events[1][1]["code"] == "llm_error"

    detail = await client.get(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)
    contents = [m["content"] for m in detail.json()["messages"]]
    assert "precious question" in contents


async def test_streaming_uses_its_own_session_not_the_request_session(
    db_session: AsyncSession,
    registered_user: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the design decision the `app` fixture's override otherwise hides.

    In production the generator runs *after* FastAPI has torn down the
    request's dependencies, so the request session is already closed. It must
    therefore open its own. This asserts it actually does — if someone
    "simplifies" the service to accept the request session, this fails.
    """
    conversation = Conversation(user_id=registered_user.id, title="t")
    db_session.add(conversation)
    # Flush first: `id` has a Python-side default that is applied on INSERT,
    # so it is still None on the un-flushed object.
    await db_session.flush()
    db_session.add(Message(conversation_id=conversation.id, role="user", content="hi"))
    await db_session.commit()

    opened: list[str] = []

    @asynccontextmanager
    async def spy_factory():
        opened.append("session")
        yield db_session

    monkeypatch.setattr("app.services.chat_service.AsyncSessionLocal", spy_factory)

    events = [
        event
        async for event in chat_service.stream_assistant_reply(
            conversation_id=conversation.id,
            user_id=registered_user.id,
            provider=EchoProvider(),
        )
    ]

    assert opened == ["session"], "the stream must open exactly one session of its own"
    assert any(isinstance(e, chat_service.DoneEvent) for e in events)


async def test_history_is_trimmed_to_the_configured_limit() -> None:
    """Context windows are finite; conversations are not."""
    messages = [
        Message(
            conversation_id=uuid.uuid4(),
            role="user" if i % 2 == 0 else "assistant",
            content=f"m{i}",
        )
        for i in range(100)
    ]

    history = chat_service.build_history(messages)

    assert len(history) == settings.llm_history_message_limit
    assert history[-1].content == "m99"


async def test_unanswered_user_turns_are_dropped_from_history() -> None:
    """Regression: a failed generation left a dangling question in history.

    The user's message is committed before generation begins, so a provider
    error (a 429, an outage, a closed tab) leaves a user turn with no reply
    after it. Replaying those made the next answer address every stranded
    question at once instead of the one just asked.
    """
    conversation_id = uuid.uuid4()
    messages = [
        Message(conversation_id=conversation_id, role="user", content="answered question"),
        Message(conversation_id=conversation_id, role="assistant", content="the reply"),
        Message(conversation_id=conversation_id, role="user", content="rate limited, no reply"),
        Message(conversation_id=conversation_id, role="user", content="also failed, no reply"),
        Message(conversation_id=conversation_id, role="user", content="the current question"),
    ]

    history = chat_service.build_history(messages)

    assert [m.content for m in history] == [
        "answered question",
        "the reply",
        "the current question",
    ]


async def test_the_latest_user_turn_is_always_kept() -> None:
    """It is unanswered by definition — it is the question being asked."""
    conversation_id = uuid.uuid4()

    history = chat_service.build_history(
        [Message(conversation_id=conversation_id, role="user", content="first ever message")]
    )

    assert [m.content for m in history] == ["first ever message"]


async def test_system_messages_are_not_replayed_as_turns() -> None:
    """The system prompt is passed separately, not as conversation history."""
    messages = [
        Message(conversation_id=uuid.uuid4(), role="system", content="ignore me"),
        Message(conversation_id=uuid.uuid4(), role="user", content="keep me"),
    ]

    history = chat_service.build_history(messages)

    assert [m.content for m in history] == ["keep me"]


async def test_deleting_a_conversation_cascades_to_messages(
    client: AsyncClient,
    auth_headers: dict[str, str],
    conversation_id: str,
    db_session: AsyncSession,
) -> None:
    await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "hi"},
        headers=auth_headers,
    )
    await client.delete(f"{PREFIX}/conversations/{conversation_id}", headers=auth_headers)

    result = await db_session.execute(
        select(Message).where(Message.conversation_id == uuid.UUID(conversation_id))
    )
    assert result.scalars().all() == []


async def test_deleting_a_user_cascades_to_conversations(
    db_session: AsyncSession, registered_user: User
) -> None:
    conversation = Conversation(user_id=registered_user.id, title="doomed")
    db_session.add(conversation)
    await db_session.commit()

    await db_session.delete(registered_user)
    await db_session.commit()

    result = await db_session.execute(
        select(Conversation).where(Conversation.user_id == registered_user.id)
    )
    assert result.scalars().all() == []
