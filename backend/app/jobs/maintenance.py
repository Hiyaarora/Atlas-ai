"""The scheduled cleanup job.

Runs as an asyncio task inside the application, not as an external cron entry
or a Celery worker. For a job that touches a few hundred rows an hour that is
the right size: no broker, no second deployment artifact, no drift between the
app's configuration and the job's.

The catch, and the reason for the advisory lock below: `uvicorn --workers 4`
starts four copies of this application, and therefore four schedulers. Without
coordination they would race — two workers purging the same document, one
succeeding and one erroring on a file that has already gone.

PostgreSQL advisory locks solve it with no new infrastructure. Exactly one
worker holds the lock and does the sweep; the others find it taken, log
nothing, and go back to sleep. If that worker dies, its connection drops and
the lock is released automatically — no stale-lock recovery to write.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal, engine
from app.services.document_lifecycle_service import MaintenanceReport, run_maintenance

logger = get_logger(__name__)

#: Arbitrary but fixed application-wide key. Any constant works as long as it
#: is unique among advisory locks this application uses.
MAINTENANCE_LOCK_KEY = 0x4A_54_4C_53  # "ATLS"


@asynccontextmanager
async def maintenance_lock() -> AsyncIterator[bool]:
    """Hold the cluster-wide maintenance lock, if it is free.

    Yields True when acquired. Session-scoped rather than transaction-scoped,
    because a sweep spans several transactions plus file and vector-store I/O
    that must not run inside an open transaction.
    """
    connection = await engine.connect()
    try:
        acquired = bool(
            await connection.scalar(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": MAINTENANCE_LOCK_KEY}
            )
        )
        try:
            yield acquired
        finally:
            if acquired:
                await connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": MAINTENANCE_LOCK_KEY}
                )
                await connection.commit()
    finally:
        await connection.close()


async def run_once() -> MaintenanceReport | None:
    """One guarded sweep. Returns None if another worker held the lock."""
    async with maintenance_lock() as acquired:
        if not acquired:
            logger.debug("maintenance_skipped_lock_held")
            return None

        async with AsyncSessionLocal() as session:
            return await run_maintenance(session)


async def maintenance_loop() -> None:
    """Sweep forever on the configured interval."""
    interval = settings.maintenance_interval_minutes * 60

    # Startup is the worst moment to compete for connections: migrations may
    # still be settling and the pool is cold.
    await asyncio.sleep(settings.maintenance_startup_delay_seconds)

    while True:
        try:
            await run_once()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a failed sweep must not kill the loop
            # Cleanup running an hour late is a non-event. Cleanup never
            # running again because one sweep raised is a slow data leak.
            logger.exception("maintenance_run_failed")

        await asyncio.sleep(interval)


def start(task_registry: list[asyncio.Task]) -> None:
    """Start the loop and register it for cancellation at shutdown."""
    if not settings.maintenance_enabled:
        logger.info("maintenance_disabled")
        return

    task = asyncio.create_task(maintenance_loop(), name="atlas-maintenance")
    task_registry.append(task)

    logger.info(
        "maintenance_scheduled",
        extra={
            "interval_minutes": settings.maintenance_interval_minutes,
            "inactive_days": settings.conversation_inactive_days,
            "grace_days": settings.document_deletion_grace_days,
        },
    )


async def stop(task_registry: list[asyncio.Task]) -> None:
    """Cancel the loop and wait for it to unwind."""
    for task in task_registry:
        task.cancel()
    for task in task_registry:
        with contextlib.suppress(asyncio.CancelledError):
            await task
