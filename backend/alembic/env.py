"""Alembic migration environment.

Two deliberate choices:

1. The database URL comes from `app.core.config.settings`, not from
   alembic.ini. One source of truth for credentials, and nothing secret is
   ever committed.
2. Migrations run through the *synchronous* psycopg driver. Alembic's
   migration scripts are sync code; using the sync DSN avoids wrapping every
   command in an event loop for no benefit. The application itself still uses
   asyncpg.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import settings
from app.db.base import Base

# Importing this package registers every ORM model on Base.metadata, which is
# what autogenerate diffs against the live database.
import app.models  # noqa: F401  isort:skip

config = context.config

# Fall back to app settings, but never override a URL the caller already set.
# Tests point Alembic at a scratch database programmatically; clobbering that
# would silently run migrations against the developer's real database.
if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option("sqlalchemy.url", settings.sync_database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting - useful for DBA review."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # a migration run is short-lived; pooling adds nothing
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # detect column type changes
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
