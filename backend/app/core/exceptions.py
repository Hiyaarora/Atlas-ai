"""Domain exceptions and the single error response contract.

Design rule: service-layer code raises *domain* exceptions (`NotFoundError`,
`ConflictError`, ...). It never raises `HTTPException`, because services must
not know they are being called over HTTP - the same service should be usable
from a CLI, a worker, or a test.

The translation from domain exception to HTTP status happens once, in the
exception handlers registered on the app.

Every error the API returns - domain, validation, or unhandled crash - uses
the same JSON envelope:

    {
      "error": {
        "code": "not_found",
        "message": "Document not found",
        "details": {...},
        "request_id": "0f9c..."
      }
    }

A stable envelope means the frontend writes error handling once.
"""

import json
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)

# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY to _CONTENT. Using the
# literal keeps this file working across both versions without a warning.
HTTP_422_UNPROCESSABLE = 422


# ==========================================================================
# Domain exceptions
# ==========================================================================


class AtlasError(Exception):
    """Base class for every error Atlas AI raises deliberately."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(AtlasError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "The requested resource was not found."


class ValidationError(AtlasError):
    """Business-rule violation, as opposed to a malformed request body."""

    status_code = HTTP_422_UNPROCESSABLE
    code = "validation_error"
    message = "The request could not be processed."


class ConflictError(AtlasError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    message = "The resource already exists or is in a conflicting state."


class AuthenticationError(AtlasError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "authentication_error"
    message = "Authentication is required or the credentials are invalid."


class RateLimitedError(AtlasError):
    """Too many failed attempts from this client or for this account.

    Deliberately not an AuthenticationError: it says nothing about whether the
    credentials were right, only that the caller must slow down. Reporting it
    as an auth failure would leak the same signal the limiter exists to
    protect.
    """

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Too many failed sign-in attempts. Please try again shortly."

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(details={"retry_after_seconds": retry_after_seconds})
        self.retry_after_seconds = retry_after_seconds


class AccountDisabledError(AuthenticationError):
    """Credentials were correct, but the account is deactivated.

    Only ever raised *after* the password verifies, so it cannot be used to
    probe which accounts are disabled.
    """

    code = "account_disabled"
    message = "This account has been deactivated."


class AuthorizationError(AtlasError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "authorization_error"
    message = "You do not have permission to perform this action."


class ExternalServiceError(AtlasError):
    """An upstream dependency (LLM provider, vector DB, ...) failed."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "external_service_error"
    message = "An upstream service failed to respond correctly."


class ServiceUnavailableError(AtlasError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"
    message = "The service is temporarily unavailable."


class LLMError(ExternalServiceError):
    """The language model provider failed.

    Distinct from a generic upstream failure so that LLM outages are visibly
    separable in logs and dashboards — they are the dependency most likely to
    degrade, and the one users notice first.
    """

    code = "llm_error"
    message = "The language model is currently unavailable."


class LLMRateLimitError(LLMError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "llm_rate_limited"
    message = "The language model is rate limited. Please retry shortly."


class EmbeddingError(ExternalServiceError):
    """The embedding provider failed."""

    code = "embedding_error"
    message = "Could not generate embeddings."


class VectorStoreError(ExternalServiceError):
    """The vector index failed."""

    code = "vector_store_error"
    message = "The knowledge index is unavailable."


class LLMConfigurationError(LLMError):
    """Misconfiguration, e.g. a missing API key.

    500, not 502: nothing upstream failed. We never sent the request.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "llm_not_configured"
    message = "The language model provider is not configured."


# ==========================================================================
# Response envelope
# ==========================================================================


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "request_id": request_id_ctx.get() or "-",
            }
        },
    )


# ==========================================================================
# Handlers
# ==========================================================================


async def atlas_error_handler(_: Request, exc: AtlasError) -> JSONResponse:
    logger.warning(
        "domain_error",
        extra={"error_code": exc.code, "status_code": exc.status_code, "detail": exc.message},
    )
    response = error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )

    # Retry-After is part of what 429 means. Without it a client has to guess
    # how long to wait, and the usual guess is "immediately", which keeps the
    # limiter saturated and the user locked out longer than necessary.
    if isinstance(exc, RateLimitedError):
        response.headers["Retry-After"] = str(exc.retry_after_seconds)

    return response


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Reshape framework-raised HTTP errors (404 routing, 405, ...)."""
    return error_response(
        status_code=exc.status_code,
        code="http_error",
        message=str(exc.detail),
    )


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Malformed request body/query/path - Pydantic rejected it.

    `exc.errors()` is not JSON-serialisable as-is. When a custom
    `field_validator` raises ValueError, Pydantic puts the *exception object*
    into each error's `ctx`, and json.dumps raises TypeError on it - turning a
    422 into a 500. Round-tripping through `default=str` coerces anything
    exotic to its string form and makes the handler total.
    """
    safe_errors = json.loads(json.dumps(exc.errors(), default=str))
    return error_response(
        status_code=HTTP_422_UNPROCESSABLE,
        code="request_validation_error",
        message="Request payload failed validation.",
        details={"errors": safe_errors},
    )


async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Last line of defence.

    Log the full traceback server-side, return a generic message to the
    client. Leaking stack traces to users is an information-disclosure bug.
    """
    logger.exception("unhandled_exception", exc_info=exc)
    return error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message="An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AtlasError, atlas_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
