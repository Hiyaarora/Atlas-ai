"""Document lifecycle: reference counting, archiving, restoring, purging.

Separate from `document_service`, which answers "how does a file become
searchable?". This module answers "how long does it stay that way?".

    active ──(no conversation activity for N days)──▶ archived
    archived ──(reference_count reaches 0)──────────▶ pending_deletion
    pending_deletion ──(grace period of M days)─────▶ purged

Each transition is separately reversible until the last one, and the last one
is the only destructive operation in the application.

Archiving is deliberately cheap to undo: it drops the document's *vectors*
but keeps its chunks in Postgres. Restoring is therefore a re-embed, not a
re-upload — the original file is not even required. That is the dividend of
treating the vector store as a derived index rather than a datastore.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.embeddings import get_embedding_provider
from app.models.conversation import Conversation
from app.models.document import Chunk, Document
from app.vectorstore import VectorRecord, get_vector_store

logger = get_logger(__name__)


@dataclass
class MaintenanceReport:
    """What one sweep did. Returned so it can be logged and asserted on."""

    conversations_archived: int = 0
    documents_archived: int = 0
    documents_marked_for_deletion: int = 0
    documents_purged: int = 0
    reference_counts_corrected: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def did_anything(self) -> bool:
        return bool(
            self.conversations_archived
            or self.documents_archived
            or self.documents_marked_for_deletion
            or self.documents_purged
            or self.reference_counts_corrected
        )


# ==========================================================================
# Reference counting
# ==========================================================================


async def sync_reference_count(session: AsyncSession, document_id: uuid.UUID) -> int:
    """Recompute one document's reference count from the conversations table.

    Recomputed, never incremented. `reference_count = reference_count - 1` is
    the version that drifts: two concurrent deletes both read 2 and both write
    1, and any transaction that rolls back after the decrement leaves the
    counter permanently wrong. Assigning the result of a subquery is a single
    atomic statement with no read-modify-write to lose.

    The stored column is therefore a *cache* of a query, kept for cheap
    filtering by the janitor, and reconstructible at any time.
    """
    subquery = (
        select(func.count())
        .select_from(Conversation)
        .where(Conversation.document_id == document_id)
        .scalar_subquery()
    )

    result = await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(reference_count=subquery)
        .returning(Document.reference_count)
    )
    count = result.scalar_one_or_none()
    return int(count or 0)


async def reconcile_reference_counts(session: AsyncSession) -> int:
    """Repair every drifted counter. Returns how many were wrong.

    Belt and braces. Even with atomic recomputation, a counter can fall out of
    step if a conversation row is ever changed by a path that forgets to call
    `sync_reference_count` — a migration, a manual fix in psql, a future
    feature. Reconciling on a schedule means the system heals itself instead
    of quietly making deletion decisions from bad data.
    """
    truth = (
        select(Conversation.document_id, func.count().label("actual"))
        .where(Conversation.document_id.is_not(None))
        .group_by(Conversation.document_id)
        .subquery()
    )

    result = await session.execute(
        update(Document)
        .where(
            Document.reference_count.is_distinct_from(
                func.coalesce(
                    select(truth.c.actual)
                    .where(truth.c.document_id == Document.id)
                    .scalar_subquery(),
                    0,
                )
            )
        )
        .values(
            reference_count=func.coalesce(
                select(truth.c.actual).where(truth.c.document_id == Document.id).scalar_subquery(),
                0,
            )
        )
        .returning(Document.id)
    )
    corrected = len(result.scalars().all())

    if corrected:
        logger.warning("reference_counts_corrected", extra={"documents": corrected})

    return corrected


async def touch_document(session: AsyncSession, document_id: uuid.UUID) -> None:
    """Record that a document was actually used for retrieval.

    Not `updated_at`, which any metadata write bumps. Inactivity must mean
    "nobody asked it anything", not "no column changed".
    """
    await session.execute(
        update(Document).where(Document.id == document_id).values(last_accessed_at=func.now())
    )


# ==========================================================================
# Archive / restore
# ==========================================================================


async def archive_document(session: AsyncSession, document: Document) -> None:
    """Take a document out of active retrieval without destroying anything.

    Removes vectors from the index; leaves chunks, the stored file, and every
    row untouched. The document simply stops being findable.
    """
    if document.lifecycle_status != "active":
        return

    await get_vector_store().delete(where={"document_id": str(document.id)})

    document.lifecycle_status = "archived"
    document.archived_at = datetime.now(UTC)

    logger.info(
        "document_archived",
        extra={"document_id": str(document.id), "references": document.reference_count},
    )


async def restore_document(
    session: AsyncSession, owner_id: uuid.UUID, document_id: uuid.UUID
) -> Document:
    """Bring an archived document back into active retrieval.

    Re-embeds from the chunks already in Postgres. No re-parse, no re-upload,
    and the original file is not required — only the derived index is rebuilt.

    Not exposed over HTTP yet; this is the primitive a future "Document
    Library" restore button calls.
    """
    result = await session.execute(
        select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise NotFoundError("Document not found.")

    if document.lifecycle_status == "active":
        return document

    chunk_result = await session.execute(
        select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.position)
    )
    chunks = list(chunk_result.scalars().all())
    if not chunks:
        raise ValidationError("This document has no stored content and cannot be restored.")

    provider = get_embedding_provider()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), settings.embedding_batch_size):
        batch = chunks[start : start + settings.embedding_batch_size]
        vectors.extend(await provider.embed([chunk.content for chunk in batch], purpose="document"))

    await get_vector_store().upsert(
        [
            VectorRecord(
                id=str(chunk.id),
                embedding=vector,
                text=chunk.content,
                metadata={
                    "owner_id": str(document.owner_id),
                    "document_id": str(document.id),
                    "filename": document.filename,
                    "page_number": chunk.page_number,
                    "position": chunk.position,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )

    document.lifecycle_status = "active"
    document.archived_at = None
    document.deletion_scheduled_at = None
    document.last_accessed_at = datetime.now(UTC)
    # The embedding model may have changed since the document was archived.
    document.embedding_model = provider.model

    await session.commit()
    logger.info("document_restored", extra={"document_id": str(document.id)})
    return document


# ==========================================================================
# Purge — the only destructive operation in the application
# ==========================================================================


async def purge_document(session: AsyncSession, document: Document) -> None:
    """Permanently delete a document: vectors, file, chunks, row.

    Order matters. Derived artefacts go first and the row goes last, so a
    crash midway leaves an orphaned file or index entry — recoverable, and
    invisible to users — rather than a row whose vectors still surface in
    search results.
    """
    document_id = document.id

    await get_vector_store().delete(where={"document_id": str(document_id)})

    path = Path(settings.storage_dir) / "documents" / document.storage_key
    if path.exists():
        await asyncio.to_thread(path.unlink)

    # Chunks go via the FK cascade when the document row is deleted.
    await session.execute(delete(Document).where(Document.id == document_id))

    logger.info("document_purged", extra={"document_id": str(document_id)})


# ==========================================================================
# The sweep
# ==========================================================================


async def run_maintenance(session: AsyncSession) -> MaintenanceReport:
    """Run every cleanup stage once.

    Called by the scheduler, and directly by tests — which is why it takes a
    session and returns a report rather than reading the clock and logging
    into the void. Nothing here is time-dependent beyond `now()`, so a test
    can create a conversation with an old `last_message_at` and assert the
    exact transition.

    Stages run oldest-state-first, so a document can move from `active` all
    the way to `pending_deletion` within a single sweep. That is deliberate
    and safe — and worth being precise about, because the tempting stronger
    claim ("one transition per sweep") is not what protects the data.

    What protects the data is that purging compares `now()` against
    `deletion_scheduled_at`, which the marking stage sets to `now()`. A
    document marked in this sweep therefore cannot be purged by this sweep,
    or by any sweep until the grace period has genuinely elapsed. Adding
    artificial per-sweep throttling on top would only slow archiving down
    without making deletion any harder to reverse.
    """
    report = MaintenanceReport()

    report.conversations_archived = await _archive_idle_conversations(session)
    report.reference_counts_corrected = await reconcile_reference_counts(session)
    report.documents_archived = await _archive_unreferenced_documents(session)
    report.documents_marked_for_deletion = await _mark_documents_for_deletion(session)
    report.documents_purged = await _purge_expired_documents(session, report)

    await session.commit()

    if report.did_anything:
        logger.info(
            "maintenance_completed",
            extra={
                "conversations_archived": report.conversations_archived,
                "documents_archived": report.documents_archived,
                "documents_marked": report.documents_marked_for_deletion,
                "documents_purged": report.documents_purged,
                "counts_corrected": report.reference_counts_corrected,
                "errors": len(report.errors),
            },
        )

    return report


async def _archive_idle_conversations(session: AsyncSession) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=settings.conversation_inactive_days)

    result = await session.execute(
        select(Conversation.id)
        .where(Conversation.is_archived.is_(False), Conversation.last_message_at < cutoff)
        .limit(settings.maintenance_batch_size)
    )
    ids = list(result.scalars().all())
    if not ids:
        return 0

    await session.execute(
        update(Conversation).where(Conversation.id.in_(ids)).values(is_archived=True)
    )
    logger.info("conversations_archived", extra={"count": len(ids)})
    return len(ids)


async def _archive_unreferenced_documents(session: AsyncSession) -> int:
    """Archive active documents whose conversations are all archived.

    Note the condition: not "no conversations", but "no *unarchived*
    conversations". A document whose only conversation was archived last night
    is no longer reachable by any user journey, so keeping its vectors in the
    index costs memory and pollutes search for no benefit.
    """
    live_conversations = (
        select(func.count())
        .select_from(Conversation)
        .where(
            Conversation.document_id == Document.id,
            Conversation.is_archived.is_(False),
        )
        .scalar_subquery()
    )

    result = await session.execute(
        select(Document)
        .where(Document.lifecycle_status == "active", live_conversations == 0)
        .limit(settings.maintenance_batch_size)
    )
    documents = list(result.scalars().all())

    for document in documents:
        await archive_document(session, document)

    return len(documents)


async def _mark_documents_for_deletion(session: AsyncSession) -> int:
    """Move archived, unreferenced documents into the grace period.

    Reaching zero references does NOT delete anything. It starts a clock —
    which is the whole point of the requirement: an accidental conversation
    delete must be survivable.
    """
    result = await session.execute(
        select(Document)
        .where(Document.lifecycle_status == "archived", Document.reference_count == 0)
        .limit(settings.maintenance_batch_size)
    )
    documents = list(result.scalars().all())

    now = datetime.now(UTC)
    for document in documents:
        document.lifecycle_status = "pending_deletion"
        document.deletion_scheduled_at = now
        logger.info("document_marked_for_deletion", extra={"document_id": str(document.id)})

    return len(documents)


async def _purge_expired_documents(session: AsyncSession, report: MaintenanceReport) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=settings.document_deletion_grace_days)

    result = await session.execute(
        select(Document)
        .where(
            Document.lifecycle_status == "pending_deletion",
            Document.deletion_scheduled_at.is_not(None),
            Document.deletion_scheduled_at < cutoff,
            # Re-checked at the moment of deletion, not trusted from when the
            # document was marked. A conversation created during the grace
            # period must rescue it.
            Document.reference_count == 0,
        )
        .limit(settings.maintenance_batch_size)
    )
    documents = list(result.scalars().all())

    purged = 0
    for document in documents:
        try:
            await purge_document(session, document)
            purged += 1
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop the sweep
            logger.exception("purge_failed", extra={"document_id": str(document.id)})
            report.errors.append(f"{document.id}: {type(exc).__name__}")

    return purged
