"""Shared pytest fixtures.

Test database strategy
----------------------
Every test runs against a **real PostgreSQL database** (`atlas_test`), not
SQLite and not mocks. Auth code depends on Postgres-specific behaviour -
UUID columns, `ON DELETE CASCADE`, unique-constraint violations raising
`IntegrityError` - and a SQLite stand-in would pass tests that production
fails.

Isolation comes from transactions, not from recreating the schema:

  1. Once per session, create the database and its tables (synchronously, so
     no event loop is involved and there is no fixture-scope juggling).
  2. Per test, open a connection, begin a transaction, and bind the session to
     it with `join_transaction_mode="create_savepoint"`.
  3. After the test, roll the outer transaction back.

`create_savepoint` is what makes this work: service code calls
`session.commit()` freely, and those commits become savepoint releases inside
our outer transaction rather than real commits. The rollback then erases
everything. Tests are fully isolated and cost no schema rebuild.
"""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.embeddings.factory import get_embedding_provider
from app.llm.factory import get_llm_provider
from app.main import create_app
from app.models.user import User
from app.retrieval.rerank import get_reranker
from app.schemas.auth import RegisterRequest
from app.services import auth_service
from app.vectorstore import get_vector_store

TEST_DB = settings.postgres_test_db


def _sync_url(database: str) -> str:
    return (
        f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{database}"
    )


def _async_test_url() -> str:
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{TEST_DB}"
    )


@pytest.fixture(scope="session", autouse=True)
def _isolate_storage(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    """Point uploads and the vector index at a throwaway directory.

    Without this the suite writes into the developer's real `storage/` and
    indexes test fixtures into the same Chroma collection the app serves —
    so a test document would start showing up in real search results.
    """
    root = tmp_path_factory.mktemp("atlas-storage")

    original_storage = settings.storage_dir
    original_chroma = settings.chroma_dir

    settings.storage_dir = str(root)
    settings.chroma_dir = str(root / "chroma")
    get_vector_store.cache_clear()

    yield

    settings.storage_dir = original_storage
    settings.chroma_dir = original_chroma
    get_vector_store.cache_clear()


@pytest.fixture(autouse=True)
def _never_call_a_real_llm(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Pin every test to a deterministic local test double.

    Without this, a developer with GEMINI_API_KEY set in `.env` runs the whole
    suite against the live Gemini API. That is slow, costs money on every run,
    depends on network reachability, and produces different text each time so
    assertions cannot check content. It also rate-limits: this suite started
    returning 429 the first time a real key was present.

    A test suite must never depend on a third party being up. `autouse` makes
    this the default rather than something each test has to remember.
    """
    monkeypatch.setattr(settings, "llm_provider", "echo")
    monkeypatch.setattr(settings, "gemini_api_key", None)
    # Same reasoning for embeddings, with sharper teeth: real embedding calls
    # are billed per token, and a full test corpus would embed on every run.
    monkeypatch.setattr(settings, "embedding_provider", "fake")

    # Relevance thresholds are a property of the EMBEDDING MODEL, not of the
    # retrieval code. The defaults in `config.py` were calibrated against
    # gemini-embedding-001 with `python -m app.eval.run --calibrate`, where a
    # genuinely relevant chunk scores 0.60-0.80. The fake provider is a
    # different model with a different score distribution, so applying those
    # numbers here rejects everything and every retrieval test fails.
    #
    # Restoring the pre-calibration floor keeps these tests measuring what they
    # are for — pipeline logic, isolation, citation shape — rather than
    # accidentally asserting a threshold tuned for a model they do not use.
    # Threshold quality is the evaluation harness's job, against real
    # embeddings and a labelled corpus.
    monkeypatch.setattr(settings, "retrieval_min_score", 0.35)
    monkeypatch.setattr(settings, "retrieval_min_margin", 0.0)

    # Reranking off by default: loading a cross-encoder costs seconds and
    # hundreds of megabytes, and would reorder results according to a model
    # these tests are not about. Tests that exercise the reranker enable it
    # explicitly with a stub model — see test_reranker.py.
    monkeypatch.setattr(settings, "rerank_enabled", False)
    get_reranker.cache_clear()

    # The factories memoise their choice, so a provider selected before this
    # patch would survive it.
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    yield
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _prepare_test_database() -> None:
    """Create `atlas_test` and its schema once per test session."""
    # CREATE DATABASE cannot run inside a transaction block, hence AUTOCOMMIT.
    admin_engine = create_engine(_sync_url("postgres"), isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DB}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    admin_engine.dispose()

    # Build the schema from the models. Note this is deliberately NOT
    # `alembic upgrade head`: tests assert against the models, and a separate
    # test verifies that the migrations actually match them.
    test_engine = create_engine(_sync_url(TEST_DB))
    Base.metadata.drop_all(test_engine)
    Base.metadata.create_all(test_engine)
    test_engine.dispose()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """A session whose writes are always rolled back."""
    engine = create_async_engine(_async_test_url(), poolclass=None)
    connection = await engine.connect()
    transaction = await connection.begin()

    session = AsyncSession(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield session
    finally:
        await session.close()
        if transaction.is_active:
            await transaction.rollback()
        await connection.close()
        await engine.dispose()


@pytest.fixture
def app(db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """An app whose database access all routes through the test transaction."""
    application = create_app()

    async def _get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    application.dependency_overrides[get_db] = _get_db

    # The chat streaming generator deliberately opens its OWN session, because
    # it runs after the request (and its session) has been torn down. That is
    # correct in production and wrong in tests: a second session on a second
    # connection cannot see rows written inside this test's still-open
    # transaction, and its writes would land in the real database.
    #
    # Redirecting the factory at the test session keeps streaming tests
    # hermetic. `test_streaming_uses_its_own_session` covers the production
    # behaviour this override hides.
    @asynccontextmanager
    async def _streaming_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    monkeypatch.setattr(
        "app.services.chat_service.AsyncSessionLocal",
        _streaming_session,
    )
    # Document ingestion is a background task and opens its own session for
    # exactly the same reason — and needs the same redirection, or it cannot
    # see the document row the test just created inside an open transaction.
    monkeypatch.setattr(
        "app.services.document_service.AsyncSessionLocal",
        _streaming_session,
    )

    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client wired straight to the ASGI app - no network, no server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ==========================================================================
# Failure-injection fixtures (health probes)
# ==========================================================================


class StubSession:
    """Async session double that always fails, to test degraded paths."""

    async def execute(self, *_: Any, **__: Any) -> Any:
        raise ConnectionError("simulated database outage")

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.fixture
async def degraded_client() -> AsyncGenerator[AsyncClient, None]:
    application = create_app()

    async def _get_db() -> AsyncGenerator[StubSession, None]:
        yield StubSession()

    application.dependency_overrides[get_db] = _get_db

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ==========================================================================
# Auth helpers
# ==========================================================================

TEST_PASSWORD = "correct-horse-battery-staple"


@pytest.fixture
async def registered_user(db_session: AsyncSession) -> User:
    return await auth_service.register_user(
        db_session,
        RegisterRequest(email="hiya@example.com", password=TEST_PASSWORD, full_name="Hiya"),
    )


@pytest.fixture
async def auth_headers(client: AsyncClient, registered_user: User) -> dict[str, str]:
    """Log in and return an Authorization header for the registered user."""
    response = await client.post(
        f"{settings.api_v1_prefix}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
