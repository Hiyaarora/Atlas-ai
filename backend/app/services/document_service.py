"""Document storage and the ingestion pipeline.

    validate -> store bytes -> parse -> chunk -> embed -> index

Ingestion runs in the background, after the upload response has been sent, so
it opens its own database session for the same reason the chat stream does:
FastAPI has already torn down the request's dependencies by then.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.embeddings import get_embedding_provider
from app.ingestion import UnparsableDocumentError, chunk_pages, get_parser, supported_extensions
from app.models.conversation import Conversation
from app.models.document import Chunk, Document
from app.schemas.documents import IngestionStats
from app.services import document_lifecycle_service
from app.vectorstore import VectorRecord, get_vector_store

logger = get_logger(__name__)


# ==========================================================================
# Upload
# ==========================================================================


def _storage_root() -> Path:
    root = Path(settings.storage_dir) / "documents"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_extension(filename: str) -> str:
    """Extract a suffix that is safe to append to a generated path.

    `PurePosixPath(...).suffix` on "../../etc/passwd" yields "" and on
    "report.pdf" yields ".pdf". The result is then whitelisted, so nothing
    from user input can influence the directory a file lands in.
    """
    suffix = PurePosixPath(filename).suffix.lower()
    return suffix if suffix in supported_extensions() else ""


async def create_document(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    filename: str,
    content_type: str,
    data: bytes,
) -> Document:
    """Validate an upload, write it to disk, and record it as pending."""
    if not data:
        raise ValidationError("The uploaded file is empty.")

    if len(data) > settings.max_upload_bytes:
        megabytes = settings.max_upload_bytes / (1024 * 1024)
        raise ValidationError(f"File is larger than the {megabytes:.0f} MB limit.")

    # Reject unsupported types now, at upload, where the user gets a real
    # status code — rather than in the background task, where the only
    # feedback is a row silently turning red.
    try:
        get_parser(filename=filename, content_type=content_type)
    except UnparsableDocumentError as exc:
        raise ValidationError(str(exc)) from exc

    document_id = uuid.uuid4()
    # The stored name is derived entirely from a UUID we generated. The user's
    # filename is kept as a label only and never touches the path.
    storage_key = f"{owner_id}/{document_id}{_safe_extension(filename)}"

    destination = _storage_root() / storage_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(destination.write_bytes, data)

    document = Document(
        id=document_id,
        owner_id=owner_id,
        filename=PurePosixPath(filename).name[:255],
        storage_key=storage_key,
        content_type=content_type or "application/octet-stream",
        size_bytes=len(data),
        ingestion_status="pending",
        lifecycle_status="active",
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    logger.info(
        "document_uploaded",
        extra={"document_id": str(document.id), "bytes": len(data)},
    )
    return document


# ==========================================================================
# Queries
# ==========================================================================


async def list_documents(
    session: AsyncSession,
    owner_id: uuid.UUID,
    *,
    include_archived: bool = False,
) -> list[Document]:
    """List a user's documents.

    `include_archived` is the seam a future Document Library uses; the chat
    sidebar never sets it, so archived documents simply stop appearing.
    """
    statement = select(Document).where(Document.owner_id == owner_id)
    if not include_archived:
        statement = statement.where(Document.lifecycle_status == "active")

    result = await session.execute(statement.order_by(Document.created_at.desc()))
    return list(result.scalars().all())


async def get_document(
    session: AsyncSession, owner_id: uuid.UUID, document_id: uuid.UUID
) -> Document:
    """Ownership is in the WHERE clause; a missing row and someone else's row
    both produce 404, so document ids cannot be enumerated."""
    result = await session.execute(
        select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise NotFoundError("Document not found.")
    return document


async def delete_document(
    session: AsyncSession, owner_id: uuid.UUID, document_id: uuid.UUID
) -> None:
    """Remove a document at the user's request.

    A *soft* delete, deliberately. The document leaves search and the UI
    immediately — its vectors are dropped, so it is unfindable within
    milliseconds — but the file, chunks and row survive the grace period
    before the janitor destroys them.

    This is the only reason an accidental delete is survivable, and it is what
    a future "Restore" button in the Document Library will hang off.

    Conversations are unlinked rather than deleted. Their history is the
    user's, not the document's; they become general chat and say so.
    """
    document = await get_document(session, owner_id, document_id)

    # Unlink first, so the reference count reflects the user's intent. Without
    # this the document would sit in `pending_deletion` forever, rescued on
    # every sweep by the conversations still pointing at it.
    await session.execute(
        update(Conversation).where(Conversation.document_id == document_id).values(document_id=None)
    )

    await document_lifecycle_service.archive_document(session, document)

    document.reference_count = 0
    document.lifecycle_status = "pending_deletion"
    document.deletion_scheduled_at = datetime.now(UTC)

    await session.commit()

    logger.info(
        "document_soft_deleted",
        extra={
            "document_id": str(document_id),
            "purge_after_days": settings.document_deletion_grace_days,
        },
    )


# ==========================================================================
# Ingestion
# ==========================================================================


async def ingest_document(document_id: uuid.UUID) -> IngestionStats | None:
    """Run the full pipeline for one document.

    Opens its own session: this executes as a background task, after the
    upload response has been sent and the request's session closed.

    Never raises. A failure is recorded on the row as
    `ingestion_status="failed"` with a message the user can act on, because
    there is no request left to return an error to.
    """
    async with AsyncSessionLocal() as session:
        document = await session.get(Document, document_id)
        if document is None:
            logger.warning("ingest_missing_document", extra={"document_id": str(document_id)})
            return None

        document.ingestion_status = "processing"
        document.error = None
        await session.commit()

        try:
            stats = await _run_pipeline(session, document)
        except UnparsableDocumentError as exc:
            await _fail(session, document, str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - background task; nothing above it
            logger.exception("ingest_failed", extra={"document_id": str(document_id)})
            await _fail(session, document, f"Ingestion failed: {type(exc).__name__}")
            return None

        return stats


async def _fail(session: AsyncSession, document: Document, message: str) -> None:
    document.ingestion_status = "failed"
    document.error = message
    await session.commit()
    logger.warning(
        "document_ingestion_failed",
        extra={"document_id": str(document.id), "reason": message},
    )


async def _run_pipeline(session: AsyncSession, document: Document) -> IngestionStats:
    path = _storage_root() / document.storage_key
    if not path.exists():
        raise UnparsableDocumentError("The stored file is missing.")

    data = await asyncio.to_thread(path.read_bytes)

    # --- parse ------------------------------------------------------------
    parser = get_parser(filename=document.filename, content_type=document.content_type)
    # PDF extraction is CPU-bound and can run for seconds on a large file.
    # Off the event loop it goes.
    parsed = await asyncio.to_thread(parser.parse, data, filename=document.filename)

    # --- chunk ------------------------------------------------------------
    chunked = chunk_pages(
        parsed.pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunked:
        raise UnparsableDocumentError("The document produced no text to index.")

    # --- persist chunks ---------------------------------------------------
    # Replace rather than append, so re-ingesting is idempotent.
    await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
    await get_vector_store().delete(where={"document_id": str(document.id)})

    chunks: list[Chunk] = []
    for position, (page, text_chunk) in enumerate(chunked):
        chunks.append(
            Chunk(
                document_id=document.id,
                position=position,
                page_number=page.number,
                content=text_chunk.text,
                char_count=len(text_chunk.text),
                # Set by the parser when the text was recognised from an image
                # rather than read from a text layer.
                content_type=page.metadata.get("content_type", "text"),
            )
        )
    session.add_all(chunks)
    await session.flush()  # assign ids before they are used as vector keys

    # --- embed ------------------------------------------------------------
    provider = get_embedding_provider()
    vectors: list[list[float]] = []

    # Batched: one request per chunk would be dominated by round-trip latency,
    # and one request for 5,000 chunks would exceed the payload limit.
    for start in range(0, len(chunks), settings.embedding_batch_size):
        batch = chunks[start : start + settings.embedding_batch_size]
        vectors.extend(await provider.embed([chunk.content for chunk in batch], purpose="document"))

    # --- index ------------------------------------------------------------
    records = [
        VectorRecord(
            id=str(chunk.id),
            embedding=vector,
            text=chunk.content,
            metadata={
                # Denormalised onto every vector because the filter runs
                # inside Chroma, which cannot join back to Postgres.
                "owner_id": str(document.owner_id),
                "document_id": str(document.id),
                "filename": document.filename,
                "page_number": chunk.page_number,
                "position": chunk.position,
                "content_type": chunk.content_type,
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    await get_vector_store().upsert(records)

    # --- finish -----------------------------------------------------------
    document.ingestion_status = "ready"
    document.page_count = parsed.page_count
    document.chunk_count = len(chunks)
    document.embedding_model = provider.model
    await session.commit()

    logger.info(
        "document_ingested",
        extra={
            "document_id": str(document.id),
            "pages": parsed.page_count,
            "chunks": len(chunks),
            "model": provider.model,
        },
    )

    return IngestionStats(
        document_id=document.id,
        pages=parsed.page_count,
        chunks=len(chunks),
        embedding_model=provider.model,
    )
