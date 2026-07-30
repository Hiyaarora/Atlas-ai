"""Health service.

Contains the actual dependency-probing logic, deliberately kept out of the
route function. The route's job is HTTP; this module's job is "is Postgres
answering?". As Atlas grows, ChromaDB and the LLM provider get probes here
too, and the route does not change.
"""

import asyncio
import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.schemas.health import DependencyCheck, ReadinessResponse

logger = get_logger(__name__)

# A probe that hangs is worse than one that fails: it ties up the readiness
# endpoint and the orchestrator's probe times out with no diagnostic.
_PROBE_TIMEOUT_SECONDS = 3.0

# Above this, the dependency is answering but unhealthily slowly.
_DEGRADED_THRESHOLD_MS = 500.0


async def check_database(session: AsyncSession) -> DependencyCheck:
    """Issue the cheapest possible query to prove the connection works.

    The timeout is the point: an orchestrator gives readiness a few seconds
    before giving up. A probe that hangs longer than that is indistinguishable
    from a crashed app, so we bound it ourselves and report the timeout
    explicitly.
    """
    started = time.perf_counter()
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            await session.execute(text("SELECT 1"))
    except TimeoutError:
        logger.warning("database_health_check_timeout", extra={"timeout_s": _PROBE_TIMEOUT_SECONDS})
        return DependencyCheck(
            name="postgres",
            status="down",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error="TimeoutError",
        )
    except Exception as exc:  # noqa: BLE001 - a probe reports any failure
        logger.warning("database_health_check_failed", extra={"error": str(exc)})
        return DependencyCheck(
            name="postgres",
            status="down",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error=type(exc).__name__,
        )

    latency_ms = (time.perf_counter() - started) * 1000
    # A slow-but-alive dependency is "degraded": still serve traffic, but make
    # the problem visible before it becomes an outage.
    status = "ok" if latency_ms < _DEGRADED_THRESHOLD_MS else "degraded"
    return DependencyCheck(name="postgres", status=status, latency_ms=round(latency_ms, 2))


async def get_readiness(session: AsyncSession) -> ReadinessResponse:
    """Aggregate every dependency probe into one verdict.

    Aggregation rule: the overall status is the worst individual status.
    """
    checks = [await check_database(session)]

    if any(check.status == "down" for check in checks):
        overall = "down"
    elif any(check.status == "degraded" for check in checks):
        overall = "degraded"
    else:
        overall = "ok"

    return ReadinessResponse(status=overall, dependencies=checks)
