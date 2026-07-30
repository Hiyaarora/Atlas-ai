"""Refresh token model.

Why refresh tokens live in the database at all, when JWTs are famously
stateless: statelessness is exactly what makes a JWT impossible to revoke.
"Log out everywhere", "this device was stolen", and "this account is
suspended" all require server-side state. Keeping that state on the
long-lived token - and only the long-lived token - preserves the performance
benefit of stateless access tokens while making sessions revocable.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        # Deleting a user must not strand their sessions. The cascade is
        # declared in the database, not just the ORM, so it holds even when
        # rows are deleted by hand in psql.
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # SHA-256 hex digest, never the token itself. 64 chars.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # NULL = still valid. Set on logout, so a used session leaves an audit
    # trail rather than vanishing.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="refresh_tokens", lazy="raise")

    @property
    def is_active(self) -> bool:
        """Usable only if neither revoked nor expired."""
        if self.revoked_at is not None:
            return False
        return self.expires_at > datetime.now(UTC)

    def __repr__(self) -> str:
        return f"<RefreshToken user_id={self.user_id} active={self.is_active}>"
