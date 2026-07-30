"""Document lifecycle: reference counting, archiving, and the cleanup sweep.

Time is driven by writing timestamps directly rather than by sleeping or
freezing the clock. A 90-day inactivity rule cannot be tested any other way,
and it keeps every test in this file instant and deterministic.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.jobs import maintenance
from app.models.conversation import Conversation
from app.models.document import Chunk, Document
from app.services import chat_service, document_lifecycle_service, document_service
from app.vectorstore import get_vector_store
from tests.test_rag import POSTGRES_NOTES, upload

PREFIX = settings.api_v1_prefix


async def age_conversation(session: AsyncSession, conversation_id: uuid.UUID, days: int) -> None:
    conversation = await session.get(Conversation, conversation_id)
    assert conversation is not None
    conversation.last_message_at = datetime.now(UTC) - timedelta(days=days)
    await session.commit()


@pytest.fixture
async def ingested(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> dict:
    document = await upload(client, auth_headers, "postgres.txt", POSTGRES_NOTES.encode())
    await document_service.ingest_document(uuid.UUID(document["id"]))
    await db_session.commit()
    return document


# ==========================================================================
# Reference counting
# ==========================================================================


async def test_upload_sets_reference_count_to_one(ingested: dict, db_session: AsyncSession) -> None:
    document = await db_session.get(Document, uuid.UUID(ingested["id"]))

    assert document is not None
    assert document.reference_count == 1


async def test_a_second_conversation_increases_the_count(
    ingested: dict, registered_user, db_session: AsyncSession
) -> None:
    from app.schemas.chat import ConversationCreate

    await chat_service.create_conversation(
        db_session,
        registered_user.id,
        ConversationCreate(title="second"),
        document_id=uuid.UUID(ingested["id"]),
    )

    document = await db_session.get(Document, uuid.UUID(ingested["id"]))
    assert document is not None
    assert document.reference_count == 2


async def test_deleting_one_conversation_decrements_but_does_not_delete(
    client: AsyncClient,
    auth_headers: dict[str, str],
    ingested: dict,
    registered_user,
    db_session: AsyncSession,
) -> None:
    """Requirement 6: count drops, nothing else happens."""
    from app.schemas.chat import ConversationCreate

    second = await chat_service.create_conversation(
        db_session,
        registered_user.id,
        ConversationCreate(title="second"),
        document_id=uuid.UUID(ingested["id"]),
    )

    await client.delete(f"{PREFIX}/conversations/{second.id}", headers=auth_headers)

    document = await db_session.get(Document, uuid.UUID(ingested["id"]))
    assert document is not None
    assert document.reference_count == 1
    assert document.lifecycle_status == "active"


async def test_reaching_zero_references_does_not_delete_immediately(
    client: AsyncClient, auth_headers: dict[str, str], ingested: dict, db_session: AsyncSession
) -> None:
    """The whole point of the grace period."""
    await client.delete(
        f"{PREFIX}/conversations/{ingested['conversation_id']}", headers=auth_headers
    )

    document = await db_session.get(Document, uuid.UUID(ingested["id"]))
    assert document is not None
    assert document.reference_count == 0
    assert document.lifecycle_status == "active", "nothing happens until the janitor runs"


async def test_reference_counts_are_recomputed_not_decremented(
    ingested: dict, db_session: AsyncSession
) -> None:
    """A drifted counter must heal itself.

    `count = count - 1` loses updates under concurrency and after any rolled
    back transaction. Recomputing from the conversations table cannot.
    """
    document = await db_session.get(Document, uuid.UUID(ingested["id"]))
    assert document is not None
    document.reference_count = 99  # simulate drift
    await db_session.commit()

    corrected = await document_lifecycle_service.reconcile_reference_counts(db_session)
    await db_session.commit()

    await db_session.refresh(document)
    assert corrected == 1
    assert document.reference_count == 1


# ==========================================================================
# Archiving
# ==========================================================================


async def test_idle_conversation_is_archived(ingested: dict, db_session: AsyncSession) -> None:
    await age_conversation(
        db_session,
        uuid.UUID(ingested["conversation_id"]),
        settings.conversation_inactive_days + 1,
    )

    report = await document_lifecycle_service.run_maintenance(db_session)

    conversation = await db_session.get(Conversation, uuid.UUID(ingested["conversation_id"]))
    assert conversation is not None
    assert conversation.is_archived
    assert report.conversations_archived == 1


async def test_active_conversation_is_left_alone(ingested: dict, db_session: AsyncSession) -> None:
    report = await document_lifecycle_service.run_maintenance(db_session)

    conversation = await db_session.get(Conversation, uuid.UUID(ingested["conversation_id"]))
    assert conversation is not None
    assert not conversation.is_archived
    assert report.conversations_archived == 0


async def test_archiving_removes_vectors_but_keeps_chunks(
    ingested: dict, db_session: AsyncSession
) -> None:
    """This is what makes archiving reversible.

    Vectors are a derived index; chunks are the source of truth. Restoring is
    therefore a re-embed, not a re-upload — the original file is not even
    required.
    """
    await age_conversation(
        db_session,
        uuid.UUID(ingested["conversation_id"]),
        settings.conversation_inactive_days + 1,
    )

    await document_lifecycle_service.run_maintenance(db_session)

    assert await get_vector_store().count(where={"document_id": ingested["id"]}) == 0

    result = await db_session.execute(
        select(Chunk).where(Chunk.document_id == uuid.UUID(ingested["id"]))
    )
    assert result.scalars().all(), "chunks must survive archiving"

    document = await db_session.get(Document, uuid.UUID(ingested["id"]))
    assert document is not None
    assert document.lifecycle_status == "archived"
    assert document.archived_at is not None


async def test_restoring_reindexes_from_stored_chunks(
    ingested: dict, registered_user, db_session: AsyncSession
) -> None:
    await age_conversation(
        db_session,
        uuid.UUID(ingested["conversation_id"]),
        settings.conversation_inactive_days + 1,
    )
    await document_lifecycle_service.run_maintenance(db_session)
    assert await get_vector_store().count(where={"document_id": ingested["id"]}) == 0

    document = await document_lifecycle_service.restore_document(
        db_session, registered_user.id, uuid.UUID(ingested["id"])
    )

    assert document.lifecycle_status == "active"
    assert document.archived_at is None
    assert await get_vector_store().count(where={"document_id": ingested["id"]}) > 0


# ==========================================================================
# The full pipeline
# ==========================================================================


async def test_marking_and_purging_can_never_happen_in_the_same_sweep(
    client: AsyncClient, auth_headers: dict[str, str], ingested: dict, db_session: AsyncSession
) -> None:
    """The guarantee that actually protects the data.

    A document may travel from `active` to `pending_deletion` within one
    sweep — that is fine, and nothing is destroyed. What must be impossible is
    marking and purging in the same run, because the grace period is what
    makes an accidental delete recoverable.

    It holds structurally: marking sets `deletion_scheduled_at = now()`, and
    purging requires it to be older than the grace period.
    """
    document_id = uuid.UUID(ingested["id"])
    await client.delete(
        f"{PREFIX}/conversations/{ingested['conversation_id']}", headers=auth_headers
    )

    report = await document_lifecycle_service.run_maintenance(db_session)

    document = await db_session.get(Document, document_id)
    assert document is not None
    await db_session.refresh(document)

    assert document.lifecycle_status == "pending_deletion"
    assert document.deletion_scheduled_at is not None
    assert report.documents_purged == 0, "the grace period must not be skippable"
    assert await db_session.get(Document, document_id) is not None

    # And repeated sweeps before the deadline change nothing.
    for _ in range(3):
        repeat = await document_lifecycle_service.run_maintenance(db_session)
        assert repeat.documents_purged == 0

    assert await db_session.get(Document, document_id) is not None


async def test_document_is_purged_after_the_grace_period(
    client: AsyncClient, auth_headers: dict[str, str], ingested: dict, db_session: AsyncSession
) -> None:
    document_id = uuid.UUID(ingested["id"])

    await client.delete(f"{PREFIX}/documents/{ingested['id']}", headers=auth_headers)

    document = await db_session.get(Document, document_id)
    assert document is not None
    document.deletion_scheduled_at = datetime.now(UTC) - timedelta(
        days=settings.document_deletion_grace_days + 1
    )
    await db_session.commit()

    report = await document_lifecycle_service.run_maintenance(db_session)

    assert report.documents_purged == 1
    assert await db_session.get(Document, document_id) is None

    result = await db_session.execute(select(Chunk).where(Chunk.document_id == document_id))
    assert result.scalars().all() == [], "chunks go with the document"


async def test_a_new_conversation_rescues_a_document_during_the_grace_period(
    client: AsyncClient,
    auth_headers: dict[str, str],
    ingested: dict,
    registered_user,
    db_session: AsyncSession,
) -> None:
    """Reference count is re-checked at the moment of deletion, not trusted
    from when the document was marked."""
    from app.schemas.chat import ConversationCreate

    document_id = uuid.UUID(ingested["id"])
    await client.delete(
        f"{PREFIX}/conversations/{ingested['conversation_id']}", headers=auth_headers
    )

    await document_lifecycle_service.run_maintenance(db_session)  # archive
    await document_lifecycle_service.run_maintenance(db_session)  # mark

    document = await db_session.get(Document, document_id)
    assert document is not None
    document.deletion_scheduled_at = datetime.now(UTC) - timedelta(
        days=settings.document_deletion_grace_days + 1
    )
    await db_session.commit()

    # Someone starts talking about it again before the purge lands.
    await chat_service.create_conversation(
        db_session,
        registered_user.id,
        ConversationCreate(title="revived"),
        document_id=document_id,
    )

    report = await document_lifecycle_service.run_maintenance(db_session)

    assert report.documents_purged == 0
    assert await db_session.get(Document, document_id) is not None


async def test_periods_are_configurable(
    ingested: dict, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 7: nothing about these durations is hardcoded."""
    monkeypatch.setattr(settings, "conversation_inactive_days", 1)

    await age_conversation(db_session, uuid.UUID(ingested["conversation_id"]), 2)

    report = await document_lifecycle_service.run_maintenance(db_session)

    assert report.conversations_archived == 1


async def test_a_failing_purge_does_not_abort_the_sweep(
    client: AsyncClient,
    auth_headers: dict[str, str],
    ingested: dict,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad document must not stop cleanup for every other document."""

    async def exploding_purge(session, document):  # noqa: ANN001
        raise RuntimeError("vector store unreachable")

    monkeypatch.setattr(document_lifecycle_service, "purge_document", exploding_purge)

    await client.delete(f"{PREFIX}/documents/{ingested['id']}", headers=auth_headers)
    document = await db_session.get(Document, uuid.UUID(ingested["id"]))
    assert document is not None
    document.deletion_scheduled_at = datetime.now(UTC) - timedelta(
        days=settings.document_deletion_grace_days + 1
    )
    await db_session.commit()

    report = await document_lifecycle_service.run_maintenance(db_session)

    assert report.documents_purged == 0
    assert report.errors, "the failure must be reported, not swallowed"


# ==========================================================================
# The scheduler
# ==========================================================================


async def test_advisory_lock_lets_only_one_worker_sweep() -> None:
    """Four uvicorn workers means four schedulers.

    Without coordination two would race on the same document — one succeeding
    and one erroring on a file that has already gone. The lock is what makes
    running the janitor in-process safe.
    """
    async with maintenance.maintenance_lock() as first:
        assert first is True

        async with maintenance.maintenance_lock() as second:
            assert second is False, "a second holder must be refused"

    # Released on exit, so the next sweep can acquire it.
    async with maintenance.maintenance_lock() as third:
        assert third is True
