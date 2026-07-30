"""User model."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.refresh_token import RefreshToken


class User(Base, TimestampMixin):
    """An authenticated account.

    Note what is *not* here: no plaintext password column, ever, not even
    temporarily. The only credential stored is a bcrypt hash.
    """

    __tablename__ = "users"

    # UUID rather than a sequential integer. Sequential ids leak how many
    # users exist and invite enumeration (/users/1, /users/2, ...). UUIDs also
    # let the client generate ids offline later without collision risk.
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # 320 = 64 (local part) + 1 (@) + 255 (domain), the RFC 5321 maximum.
    # Stored lowercase; normalisation happens in the schema layer so that
    # "Hiya@x.com" and "hiya@x.com" cannot become two accounts.
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)

    # bcrypt output is always 60 characters.
    hashed_password: Mapped[str] = mapped_column(String(60), nullable=False)

    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Soft disable: lets an account be suspended without deleting its
    # documents and conversations.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        # Async SQLAlchemy cannot lazy-load on attribute access; loading this
        # collection must be explicit (selectinload) at the query site.
        lazy="raise",
    )

    def __repr__(self) -> str:
        # Deliberately excludes the password hash - reprs end up in logs.
        return f"<User id={self.id} email={self.email!r}>"
