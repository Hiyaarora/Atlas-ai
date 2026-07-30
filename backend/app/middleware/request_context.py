"""Request context middleware.

Responsibilities, in order:

1. Assign every request a unique id (or adopt the caller's `X-Request-ID`,
   which is how a trace survives across microservices).
2. Publish that id to the logging ContextVar so every log line emitted while
   handling the request carries it.
3. Emit one structured access log line with method, path, status, duration.
4. Echo the id back on the response so a user can quote it in a bug report
   and you can find their exact request in the logs.
"""

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, request_id_ctx

logger = get_logger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        # perf_counter, not time.time: monotonic, immune to clock adjustments.
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handler will build the response; we only need the
            # access log to record that this request ended in a failure.
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            # Deliberately NOT resetting the ContextVar here: the unhandled
            # exception handler sits *outside* this middleware and still needs
            # the id to put in the error envelope. No leak results, because
            # the ASGI server runs each request in its own task, and a task
            # gets a fresh copy of the context.
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id

        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        # Reset only after the access log is emitted, or the line that most
        # needs the id is the one line that does not have it.
        request_id_ctx.reset(token)
        return response
