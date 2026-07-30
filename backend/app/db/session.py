"""Async database engine and session management.

Concepts worth internalising:

* **Engine** - owns the connection pool. One per process, created at import,
  disposed at shutdown. Creating engines per request is a classic
  resource-leak bug.
* **Session** - a unit of work. One per request, never shared across
  requests, because a Session is not concurrency-safe.
* **expire_on_commit=False** - after `commit()`, SQLAlchemy would normally
  mark loaded objects stale and re-fetch attributes on next access. In async
  code that lazy re-fetch happens outside the `await` you expected and raises
  `MissingGreenlet`. Disabling it lets us return ORM objects after commit.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,  # validate a pooled connection before handing it out
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,  # drop connections older than 30 min; proxies kill idle ones
    connect_args={
        # asyncpg defaults to a 60s connect timeout. On an unreachable host
        # that stalls startup and health probes for a full minute before
        # reporting the obvious. Fail fast and let the caller decide.
        "timeout": 5.0,
        # Ceiling on any single statement, so one pathological query cannot
        # hold a pooled connection open indefinitely.
        "command_timeout": 30.0,
    },
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a request-scoped session.

    Transaction policy: the *service layer* decides when to commit. This
    dependency only guarantees the session is rolled back on error and always
    closed, so a failed request can never leave a half-applied transaction
    holding locks.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
