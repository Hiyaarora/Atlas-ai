"""Health endpoint behaviour."""

import asyncio

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services import health_service

PREFIX = settings.api_v1_prefix


async def test_liveness_returns_ok(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_liveness_does_not_touch_the_database(degraded_client: AsyncClient) -> None:
    """The whole point of liveness: a DB outage must not restart the app."""
    response = await degraded_client.get(f"{PREFIX}/health/live")

    assert response.status_code == 200


async def test_readiness_reports_healthy_dependencies(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/health/ready")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert [dep["name"] for dep in body["dependencies"]] == ["postgres"]
    assert body["dependencies"][0]["latency_ms"] is not None


async def test_readiness_returns_503_when_database_is_down(
    degraded_client: AsyncClient,
) -> None:
    response = await degraded_client.get(f"{PREFIX}/health/ready")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "down"
    assert body["dependencies"][0]["status"] == "down"
    assert body["dependencies"][0]["error"] == "ConnectionError"


async def test_hanging_database_is_reported_as_down_not_left_to_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe must bound its own wait, or the orchestrator times out blind."""

    class HangingSession:
        async def execute(self, *_: object, **__: object) -> None:
            await asyncio.sleep(10)

    monkeypatch.setattr(health_service, "_PROBE_TIMEOUT_SECONDS", 0.05)

    check = await health_service.check_database(HangingSession())  # type: ignore[arg-type]

    assert check.status == "down"
    assert check.error == "TimeoutError"


async def test_slow_but_alive_database_is_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowSession:
        async def execute(self, *_: object, **__: object) -> None:
            await asyncio.sleep(0.05)

    monkeypatch.setattr(health_service, "_DEGRADED_THRESHOLD_MS", 10.0)

    check = await health_service.check_database(SlowSession())  # type: ignore[arg-type]

    assert check.status == "degraded"
    assert check.error is None


async def test_readiness_aggregates_to_the_worst_dependency_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.schemas.health import DependencyCheck

    async def fake_check(_: AsyncSession) -> DependencyCheck:
        return DependencyCheck(name="postgres", status="degraded", latency_ms=900.0, error=None)

    monkeypatch.setattr(health_service, "check_database", fake_check)

    result = await health_service.get_readiness(None)  # type: ignore[arg-type]

    assert result.status == "degraded"


async def test_info_reports_environment(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/health/info")
    body = response.json()

    assert response.status_code == 200
    assert body["name"] == settings.app_name
    assert body["api_version"] == "v1"


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/health/live")

    assert response.headers.get("X-Request-ID")


async def test_caller_supplied_request_id_is_preserved(client: AsyncClient) -> None:
    """Distributed tracing depends on the id surviving across hops."""
    response = await client.get(
        f"{PREFIX}/health/live",
        headers={"X-Request-ID": "trace-abc-123"},
    )

    assert response.headers["X-Request-ID"] == "trace-abc-123"
