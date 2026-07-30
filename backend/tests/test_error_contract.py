"""The error envelope is a contract; these tests pin it down.

If a future refactor changes the shape of an error response, the frontend
breaks silently. These tests make that break loud instead.
"""

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.exceptions import ConflictError, NotFoundError

PREFIX = settings.api_v1_prefix


async def test_unknown_route_uses_the_error_envelope(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/does-not-exist")
    body = response.json()

    assert response.status_code == 404
    assert set(body["error"]) == {"code", "message", "details", "request_id"}


async def test_domain_error_maps_to_its_status_code(app: FastAPI) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise NotFoundError("Document not found", details={"document_id": "42"})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/boom")

    body = response.json()
    assert response.status_code == 404
    assert body["error"]["code"] == "not_found"
    assert body["error"]["message"] == "Document not found"
    assert body["error"]["details"] == {"document_id": "42"}


async def test_conflict_error_maps_to_409(app: FastAPI) -> None:
    @app.get("/conflict")
    async def conflict() -> None:
        raise ConflictError()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/conflict")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_custom_validator_errors_are_serialisable(app: FastAPI) -> None:
    """Regression: a custom field_validator used to turn 422 into 500.

    Pydantic puts the raised ValueError *object* into each error's `ctx`, and
    json.dumps cannot serialise an exception. The handler now coerces exotic
    values with `default=str`.
    """
    from pydantic import BaseModel, field_validator

    class Payload(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def _reject(cls, v: str) -> str:
            raise ValueError("always invalid")

    @app.post("/validated")
    async def validated(payload: Payload) -> None:  # pragma: no cover - via HTTP
        return None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/validated", json={"value": "anything"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"
    assert response.json()["error"]["details"]["errors"], "error detail must survive"


async def test_error_response_includes_the_request_id(client: AsyncClient) -> None:
    """The id in the body must match the header, or support cannot trace it."""
    response = await client.get(f"{PREFIX}/nope", headers={"X-Request-ID": "trace-xyz"})

    assert response.json()["error"]["request_id"] == "trace-xyz"
