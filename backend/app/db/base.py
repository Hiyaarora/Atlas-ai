"""Declarative base for every ORM model.

The naming convention matters more than it looks. Postgres auto-generates
constraint names like `users_email_key`; Alembic's autogenerate cannot
reliably drop or alter a constraint whose name it did not choose. Fixing the
convention up front means migrations stay deterministic for the life of the
project.

Concrete models are imported here so Alembic's autogenerate can see them
via `Base.metadata`.
"""

from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Adds audit timestamps maintained by the database, not by Python.

    `server_default=func.now()` means Postgres stamps the row. That stays
    correct even when a row is inserted by a migration or by hand in psql.

    `DateTime(timezone=True)` is not optional. Without it SQLAlchemy emits
    `TIMESTAMP WITHOUT TIME ZONE`, which stores a wall-clock reading with no
    record of the offset it was taken in. Two servers in different regions
    then write values that cannot be ordered, and the error is invisible until
    it matters. Always TIMESTAMPTZ.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
