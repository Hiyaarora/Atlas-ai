"""Conversation isolation — the most important property in the system.

If these pass, a conversation about one document can never surface content
from another. If any of them fail, the product is broken in a way users
cannot see and cannot report accurately.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.conversation import Conversation
from app.models.document import Document
from app.services import chat_service, document_service
from tests.test_chat import parse_sse
from tests.test_rag import COOKING_NOTES, POSTGRES_NOTES, upload

PREFIX = settings.api_v1_prefix


@pytest.fixture
async def two_documents(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> dict[str, dict]:
    """Two ingested documents, each with the conversation upload created."""
    documents: dict[str, dict] = {}
    for name, body in (("postgres.txt", POSTGRES_NOTES), ("cooking.txt", COOKING_NOTES)):
        document = await upload(client, auth_headers, name, body.encode())
        await document_service.ingest_document(uuid.UUID(document["id"]))
        documents[name] = document
    await db_session.commit()
    return documents


# ==========================================================================
# Binding
# ==========================================================================


async def test_upload_creates_a_conversation_bound_to_the_document(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    """One request, one intent: "I want to talk about this file"."""
    document = await upload(client, auth_headers, "notes.txt", POSTGRES_NOTES.encode())

    conversation = await db_session.get(Conversation, uuid.UUID(document["conversation_id"]))

    assert conversation is not None
    assert str(conversation.document_id) == document["id"]


async def test_conversation_is_titled_after_its_document(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    document = await upload(client, auth_headers, "quarterly-report.txt", POSTGRES_NOTES.encode())

    conversation = await db_session.get(Conversation, uuid.UUID(document["conversation_id"]))

    assert conversation is not None
    assert conversation.title == "quarterly-report.txt"


async def test_uploading_again_creates_a_separate_conversation(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    """Requirement 1: a new upload becomes the active document, in its own
    conversation. The previous one is untouched, not repurposed."""
    first = await upload(client, auth_headers, "first.txt", POSTGRES_NOTES.encode())
    second = await upload(client, auth_headers, "second.txt", COOKING_NOTES.encode())

    assert first["conversation_id"] != second["conversation_id"]

    original = await db_session.get(Conversation, uuid.UUID(first["conversation_id"]))
    assert original is not None
    assert str(original.document_id) == first["id"], "the first binding must not move"


# ==========================================================================
# Scope resolution
# ==========================================================================


async def test_scope_is_the_conversations_document(
    two_documents: dict[str, dict], db_session: AsyncSession
) -> None:
    postgres = two_documents["postgres.txt"]
    conversation = await db_session.get(Conversation, uuid.UUID(postgres["conversation_id"]))
    assert conversation is not None

    scope = await chat_service.resolve_scope(db_session, conversation)

    assert scope == [uuid.UUID(postgres["id"])]


async def test_conversation_without_a_document_has_empty_scope(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    """And empty scope means no retrieval — never a widened search."""
    response = await client.post(f"{PREFIX}/conversations", json={}, headers=auth_headers)
    conversation = await db_session.get(Conversation, uuid.UUID(response.json()["id"]))
    assert conversation is not None

    assert await chat_service.resolve_scope(db_session, conversation) == []


async def test_archived_document_drops_out_of_scope(
    two_documents: dict[str, dict], db_session: AsyncSession
) -> None:
    """An archived document must stop being retrievable, without falling back
    to searching anything else."""
    postgres = two_documents["postgres.txt"]
    document = await db_session.get(Document, uuid.UUID(postgres["id"]))
    assert document is not None
    document.lifecycle_status = "archived"
    await db_session.commit()

    conversation = await db_session.get(Conversation, uuid.UUID(postgres["conversation_id"]))
    assert conversation is not None

    assert await chat_service.resolve_scope(db_session, conversation) == []


@pytest.mark.parametrize("status", ["pending", "processing", "failed"])
async def test_document_that_is_not_ready_is_not_in_scope(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession, status: str
) -> None:
    """Uploaded but not embedded yet: there is nothing to retrieve, and
    quietly searching a neighbouring document would be worse than nothing.

    The state is set explicitly rather than relying on upload leaving the
    document pending — FastAPI background tasks *do* run under httpx's
    ASGITransport, so ingestion has already completed by the time the upload
    response is returned in tests.
    """
    uploaded = await upload(client, auth_headers, "pending.txt", POSTGRES_NOTES.encode())

    document = await db_session.get(Document, uuid.UUID(uploaded["id"]))
    assert document is not None
    document.ingestion_status = status
    await db_session.commit()

    conversation = await db_session.get(Conversation, uuid.UUID(uploaded["conversation_id"]))
    assert conversation is not None

    assert await chat_service.resolve_scope(db_session, conversation) == []


# ==========================================================================
# End to end: contexts must never mix
# ==========================================================================


async def _ask(client: AsyncClient, headers: dict[str, str], conversation_id: str, question: str):
    response = await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": question},
        headers=headers,
    )
    events = parse_sse(response.text)
    sources = next(data for name, data in events if name == "sources")
    return sources["citations"]


async def test_each_conversation_retrieves_only_its_own_document(
    client: AsyncClient, auth_headers: dict[str, str], two_documents: dict[str, dict]
) -> None:
    """The headline requirement, asked both ways round.

    The postgres conversation is asked a *baking* question and the cooking
    conversation a *postgres* question. Correct behaviour is that neither
    finds anything — not that each helpfully searches the other document.
    """
    postgres_citations = await _ask(
        client,
        auth_headers,
        two_documents["postgres.txt"]["conversation_id"],
        "How long should I autolyse the flour and water?",
    )
    cooking_citations = await _ask(
        client,
        auth_headers,
        two_documents["cooking.txt"]["conversation_id"],
        "Which view reports how often an index is used?",
    )

    assert all(c["filename"] == "postgres.txt" for c in postgres_citations)
    assert all(c["filename"] == "cooking.txt" for c in cooking_citations)


async def test_a_conversation_still_retrieves_its_document_after_newer_uploads(
    client: AsyncClient, auth_headers: dict[str, str], two_documents: dict[str, dict]
) -> None:
    """ "Reopen conversation A in six months" — compressed into one test.

    The binding lives in the database, so uploading other documents afterwards
    cannot change what an older conversation searches.
    """
    postgres_conversation = two_documents["postgres.txt"]["conversation_id"]

    # cooking.txt was uploaded after postgres.txt, and is now the newest.
    citations = await _ask(
        client,
        auth_headers,
        postgres_conversation,
        "Which view reports how often an index is used?",
    )

    assert citations
    assert all(c["filename"] == "postgres.txt" for c in citations)


async def test_general_conversation_retrieves_nothing_even_with_documents(
    client: AsyncClient, auth_headers: dict[str, str], two_documents: dict[str, dict]
) -> None:
    """A conversation created without a document stays document-free.

    Before this refactor it would have searched everything the user owned.
    """
    response = await client.post(f"{PREFIX}/conversations", json={}, headers=auth_headers)

    citations = await _ask(
        client,
        auth_headers,
        response.json()["id"],
        "Which view reports how often an index is used?",
    )

    assert citations == []


async def test_cannot_send_into_another_users_conversation(
    client: AsyncClient, auth_headers: dict[str, str], two_documents: dict[str, dict]
) -> None:
    """Isolation across users, not just across documents."""
    from app.schemas.auth import RegisterRequest
    from app.services import auth_service
    from tests.conftest import TEST_PASSWORD

    login = await client.post(
        f"{PREFIX}/auth/register",
        json={"email": "isolation-intruder@example.com", "password": TEST_PASSWORD},
    )
    intruder_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    del auth_service, RegisterRequest  # imported for symmetry with other tests

    response = await client.post(
        f"{PREFIX}/conversations/{two_documents['postgres.txt']['conversation_id']}/messages",
        json={"content": "leak please"},
        headers=intruder_headers,
    )

    assert response.status_code == 404


# ==========================================================================
# The API surface
# ==========================================================================


async def test_conversation_list_exposes_the_linked_document(
    client: AsyncClient, auth_headers: dict[str, str], two_documents: dict[str, dict]
) -> None:
    """So the UI can always show which source an answer will come from."""
    response = await client.get(f"{PREFIX}/conversations", headers=auth_headers)

    by_id = {c["id"]: c for c in response.json()}
    postgres = by_id[two_documents["postgres.txt"]["conversation_id"]]

    assert postgres["document_filename"] == "postgres.txt"
    assert postgres["document_id"] == two_documents["postgres.txt"]["id"]


async def test_there_is_no_api_to_rebind_a_conversation(
    client: AsyncClient, auth_headers: dict[str, str], two_documents: dict[str, dict]
) -> None:
    """Binding is immutable by construction.

    A conversation whose document could be swapped would break the guarantee
    that reopening it retrieves from the same source, so PATCH accepts only a
    title and ignores anything else.
    """
    postgres = two_documents["postgres.txt"]
    cooking = two_documents["cooking.txt"]

    response = await client.patch(
        f"{PREFIX}/conversations/{postgres['conversation_id']}",
        json={"title": "renamed", "document_id": cooking["id"]},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == postgres["id"]
