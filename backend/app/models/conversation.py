"""Conversation and Message models."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.user import User

DEFAULT_CONVERSATION_TITLE = "New conversation"

#: Stored as text with a CHECK constraint rather than a Postgres ENUM type.
#: Native enums require an ALTER TYPE migration to add a value and cannot drop
#: one at all; a CHECK constraint is edited like any other constraint. When
#: "tool" and "function" roles may arrive later, this matters.
MESSAGE_ROLES = ("user", "assistant", "system")


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255), default=DEFAULT_CONVERSATION_TITLE, nullable=False
    )

    #: The document this conversation is about — the isolation guarantee.
    #:
    #: Retrieval resolves its scope from HERE, never from the logged-in user.
    #: Reopening this conversation in six months re-reads this column, so the
    #: binding survives restarts, new uploads, and everything else. There is
    #: no in-memory "currently selected document" that can go stale.
    #:
    #: Nullable: a conversation with no document is general chat, and existing
    #: conversations predate the column.
    #:
    #: ON DELETE SET NULL, deliberately not CASCADE. Purging a document must
    #: never destroy a user's conversation history; the conversation survives
    #: having lost its source.
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    #: Last time a message was added. Drives inactivity-based archiving.
    #: Distinct from `updated_at`, which a rename would also bump.
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    #: Archived conversations leave the sidebar but keep their history.
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(lazy="raise")
    document: Mapped["Document"] = relationship(back_populates="conversations", lazy="raise")

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="raise",
    )

    __table_args__ = (
        # The sidebar query is "my conversations, newest activity first".
        # A composite index serves both the filter and the sort in one scan.
        Index("ix_conversations_user_id_updated_at", "user_id", "updated_at"),
        # Reference counting and the archive sweep both group by document.
        Index("ix_conversations_document_id", "document_id"),
        # The janitor's inactivity scan.
        Index("ix_conversations_archived_last_message", "is_archived", "last_message_at"),
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} title={self.title!r}>"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(String(16), nullable=False)

    # Text, not String(n): model replies have no meaningful length ceiling,
    # and Postgres stores both identically anyway (TOAST handles large values).
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Which model produced this message. NULL for user messages. Recorded so
    # answer quality can be attributed to a model after the fact — essential
    # when comparing retrieval and generation strategies.
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # clock_timestamp(), NOT now().
    #
    # In PostgreSQL `now()` (and CURRENT_TIMESTAMP) return the *transaction
    # start* time and stay constant for the whole transaction. Two messages
    # inserted in one transaction therefore receive byte-identical
    # timestamps, ordering collapses to the random-UUID tiebreak, and the
    # transcript comes back scrambled — assistant before user.
    #
    # clock_timestamp() reads the actual wall clock on each call, so message
    # order is correct however the writes are batched.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp(), nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages", lazy="raise")

    __table_args__ = (
        CheckConstraint(
            "role IN " + str(MESSAGE_ROLES),
            name="valid_role",
        ),
        # Every read is "the messages of one conversation, in order".
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} chars={len(self.content)}>"
