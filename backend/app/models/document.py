"""Document and Chunk models.

Chunk text lives in Postgres *and* in Chroma, which looks like duplication and
is not. Postgres is the source of truth; Chroma is a derived, disposable search
index. That separation is what makes the archive stage below reversible:
archiving drops a document's vectors but keeps its chunks, so restoring it is a
re-embed rather than a re-upload.

Two independent state machines
------------------------------
`ingestion_status` — how processing went:

    pending -> processing -> ready
                          -> failed

`lifecycle_status` — where the document is in its life:

    active -> archived -> pending_deletion -> (purged)

They are orthogonal. A document can be `ready` *and* `archived`. Keeping them
in one column would make illegal states representable and force every query to
disambiguate which meaning it wanted.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.user import User

#: How ingestion went.
INGESTION_STATUSES = ("pending", "processing", "ready", "failed")

#: Where the document is in its life.
LIFECYCLE_STATUSES = ("active", "archived", "pending_deletion")


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    #: The name the user recognises. Never used as a filesystem path.
    filename: Mapped[str] = mapped_column(String(255), nullable=False)

    #: Opaque UUID-based path under the storage root. Keeping the two separate
    #: is the path-traversal defence: a file called "../../.env" is a harmless
    #: label, never a location.
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)

    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # ---- Ingestion state -------------------------------------------------
    ingestion_status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    #: Which model produced the stored vectors. Without this, changing
    #: embedding model leaves a corpus of vectors from two incompatible spaces
    #: with no way to tell them apart.
    embedding_model: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ---- Lifecycle state -------------------------------------------------
    lifecycle_status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)

    #: Conversations pointing at this document.
    #:
    #: A cache of `SELECT count(*) FROM conversations WHERE document_id = id`,
    #: and always written by recomputing that query rather than by
    #: incrementing. An incremented counter drifts under concurrency and
    #: after any failed transaction; a recomputed one cannot.
    reference_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    #: Last time this document was actually used for retrieval. Distinct from
    #: `updated_at`, which any metadata write would bump.
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    #: When it was archived; NULL while active.
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: When it entered `pending_deletion`. The grace period is measured from
    #: here, so purging is a comparison against a stored fact rather than an
    #: inference from other timestamps.
    deletion_scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    owner: Mapped["User"] = relationship(lazy="raise")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="raise",
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="document",
        lazy="raise",
    )

    __table_args__ = (
        CheckConstraint("ingestion_status IN " + str(INGESTION_STATUSES), name="valid_ingestion"),
        CheckConstraint("lifecycle_status IN " + str(LIFECYCLE_STATUSES), name="valid_lifecycle"),
        CheckConstraint("reference_count >= 0", name="reference_count_non_negative"),
        Index("ix_documents_owner_id_created_at", "owner_id", "created_at"),
        # The janitor's sweeps. Without these they degrade into full scans as
        # the corpus grows — which is precisely when cleanup matters most.
        Index("ix_documents_lifecycle_archived", "lifecycle_status", "archived_at"),
        Index("ix_documents_lifecycle_deletion", "lifecycle_status", "deletion_scheduled_at"),
    )

    @property
    def is_searchable(self) -> bool:
        """Usable as a retrieval source: ingested, and still active."""
        return self.ingestion_status == "ready" and self.lifecycle_status == "active"

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} filename={self.filename!r} "
            f"ingestion={self.ingestion_status} lifecycle={self.lifecycle_status}>"
        )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    #: Position within the document, from 0. Ordering key for reassembly.
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    #: Where in the source this came from — what a citation points at.
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    #: How this text was obtained: "text" read from the file's text layer,
    #: "ocr" recognised from an image.
    #:
    #: Worth recording because the two are not equally trustworthy. A text
    #: layer is exact; OCR is a reading, and misreads a zero as an O. Marking
    #: it means a passage can be traced to its origin when an answer looks
    #: wrong, and lets OCR content be excluded later without re-ingesting.
    #: `server_default` as well as `default`, because the migration adds one
    #: and the two must agree — a model that omits it makes autogenerate
    #: see permanent drift, which is what test_migrations catches.
    content_type: Mapped[str] = mapped_column(
        String(16), default="text", server_default="text", nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="chunks", lazy="raise")

    __table_args__ = (
        # Re-ingesting a document must replace its chunks, never silently
        # create a second set at the same positions.
        UniqueConstraint("document_id", "position", name="uq_chunk_position"),
        Index("ix_chunks_document_id_position", "document_id", "position"),
    )

    def __repr__(self) -> str:
        return f"<Chunk doc={self.document_id} pos={self.position} page={self.page_number}>"
