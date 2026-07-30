"""Migrations must stay in sync with the models.

The most common Alembic failure in real projects is not a broken migration -
it is a *missing* one. Someone edits a model, tests pass locally because the
test schema is built from the models, and the change reaches production as a
column that does not exist.

This test closes that gap: it applies the real migration history to a scratch
database and asserts that Alembic sees no remaining difference between that
schema and `Base.metadata`. If you change a model and forget
`alembic revision --autogenerate`, this fails.
"""

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.db.base import Base

MIGRATION_TEST_DB = f"{settings.postgres_test_db}_migrations"

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _sync_url(database: str) -> str:
    return (
        f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{database}"
    )


def _assert_is_a_scratch_database(url: str) -> None:
    """Refuse to touch anything that is not the disposable migration database.

    This is not paranoia. During development, `env.py`
    overrode the URL passed in by these tests, and `command.downgrade(...,
    "base")` ran against the developer's real `atlas` database and dropped
    every table. env.py no longer does that - and this assertion means a
    future regression there fails loudly instead of destroying data.
    """
    if not url.endswith(f"/{MIGRATION_TEST_DB}"):
        raise RuntimeError(
            f"Refusing to run destructive migration tests against {url!r}. "
            f"Expected a database named {MIGRATION_TEST_DB!r}."
        )


@pytest.fixture
def migrated_database() -> str:
    """A database built by running the migration history from scratch."""
    admin_engine = create_engine(_sync_url("postgres"), isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        # Drop and recreate so the history is always applied from zero. A
        # migration that only works against a pre-existing schema is broken.
        conn.execute(text(f'DROP DATABASE IF EXISTS "{MIGRATION_TEST_DB}" WITH (FORCE)'))
        conn.execute(text(f'CREATE DATABASE "{MIGRATION_TEST_DB}"'))
    admin_engine.dispose()

    url = _sync_url(MIGRATION_TEST_DB)
    _assert_is_a_scratch_database(url)

    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")

    # Prove env.py honoured our URL rather than redirecting to the app
    # database. If it did not, the tables are missing here and we stop before
    # the downgrade test can do damage elsewhere.
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            tables = (
                connection.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                )
                .scalars()
                .all()
            )
    finally:
        engine.dispose()

    assert "users" in tables, (
        "Migrations did not run against the scratch database. "
        "Check that alembic/env.py does not overwrite a caller-supplied sqlalchemy.url."
    )

    return url


def test_migrations_produce_the_schema_the_models_describe(migrated_database: str) -> None:
    engine = create_engine(migrated_database)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={"compare_type": True, "compare_server_default": True},
            )
            differences = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert differences == [], (
        "Models and migrations have drifted. Run:\n"
        "  alembic revision --autogenerate -m 'describe your change'\n"
        f"Differences: {differences}"
    )


def test_migrations_can_be_rolled_all_the_way_back(migrated_database: str) -> None:
    """A downgrade path that was never run is a downgrade path that is broken.

    This matters the first time a bad deploy needs reverting at 2am.
    """
    _assert_is_a_scratch_database(migrated_database)

    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", migrated_database)

    command.downgrade(config, "base")

    engine = create_engine(migrated_database)
    try:
        with engine.connect() as connection:
            remaining = (
                connection.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name != 'alembic_version'"
                    )
                )
                .scalars()
                .all()
            )
    finally:
        engine.dispose()

    assert remaining == [], f"downgrade left tables behind: {remaining}"
