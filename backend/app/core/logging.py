"""Structured logging.

Two formatters share one pipeline:

* `console` - readable, colour-free, for local development.
* `json`    - one JSON object per line, for production log aggregators
              (CloudWatch, Loki, Datadog) which parse fields, not prose.

Every record automatically carries the current `request_id`, so a single
user request can be traced across all the log lines it produced.
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

from app.core.config import settings

# A ContextVar is the async-safe equivalent of thread-local storage: each
# concurrently-running request coroutine sees its own value.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

# LogRecord attributes that exist on every record; anything else the caller
# passed via `extra=` is application context worth emitting.
_RESERVED_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "asctime",
    "message",
    "taskName",
}


class SafeExtraLogger(logging.Logger):
    """A logger whose `extra=` can never crash the caller.

    `logging` reserves attribute names on LogRecord — `filename`, `name`,
    `module`, `msg`, `args`, `lineno` and friends — and raises
    `KeyError: "Attempt to overwrite 'filename' in LogRecord"` if `extra`
    contains one. That turned a perfectly reasonable
    `extra={"filename": ...}` in the PDF parser into a crashed ingestion.

    A log line must never be able to break the operation it is describing, so
    colliding keys are prefixed instead of raising.
    """

    def makeRecord(  # noqa: PLR0913 - signature is fixed by the stdlib
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: Any,
        exc_info: Any,
        func: str | None = None,
        extra: dict[str, Any] | None = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        if extra:
            extra = {
                (f"ctx_{key}" if key in _RESERVED_ATTRS else key): value
                for key, value in extra.items()
            }
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


# Must run before any application logger is created, hence at import time of
# this module — which every other module reaches through `get_logger`.
logging.setLoggerClass(SafeExtraLogger)


class RequestIdFilter(logging.Filter):
    """Attach the active request id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single line, with any `extra=` fields appended."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_ATTRS and key != "request_id"
        }
        if extras:
            rendered = " ".join(f"{k}={v}" for k, v in extras.items())
            base = f"{base} | {rendered}"
        return base


def configure_logging() -> None:
    """Install handlers on the root logger. Call once, at startup."""
    formatter = JsonFormatter() if settings.log_format == "json" else ConsoleFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()  # drop uvicorn's default handlers so output stays uniform
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # uvicorn installs its own handlers; let records propagate to root instead
    # so every line goes through our formatter.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # RequestContextMiddleware already emits an access log with the request id
    # and a duration. uvicorn's version carries neither, so keeping both means
    # paying twice for strictly less information.
    logging.getLogger("uvicorn.access").disabled = True

    # SQLAlchemy is extremely chatty at INFO; opt in explicitly when debugging.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Module-level logger accessor: `logger = get_logger(__name__)`."""
    return logging.getLogger(name)
