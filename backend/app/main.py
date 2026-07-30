"""Application factory and ASGI entrypoint.

`create_app()` exists instead of a bare module-level `app = FastAPI()` so
tests can build an isolated instance with overridden settings, and so the
wiring order (logging -> middleware -> handlers -> routes) is explicit and
readable in one place.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.db.session import engine
from app.jobs import maintenance
from app.middleware.request_context import RequestContextMiddleware

logger = get_logger(__name__)

STARTUP_DB_TIMEOUT_SECONDS = 10.0


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup and shutdown hooks.

    Anything expensive and long-lived (DB pool warm-up, and later the vector
    store client and LLM client) is created here rather than per request.
    """
    logger.info(
        "application_starting",
        extra={"environment": settings.app_env, "debug": settings.debug},
    )

    # Verify connectivity once at boot. We log rather than crash: the process
    # staying up lets /health/ready report *why* it is not serving, which is
    # far easier to debug than a container in CrashLoopBackOff.
    try:
        # Bounded: an unreachable host must not stall the boot sequence.
        async with asyncio.timeout(STARTUP_DB_TIMEOUT_SECONDS):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        logger.info("database_connection_ok", extra={"host": settings.postgres_host})
    except TimeoutError:
        logger.error(
            "database_connection_timeout",
            extra={"host": settings.postgres_host, "timeout_s": STARTUP_DB_TIMEOUT_SECONDS},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "database_connection_failed",
            extra={"host": settings.postgres_host, "error": repr(exc)},
        )

    # Background jobs start after connectivity is known, so a failed sweep is
    # never the first thing a cold process does.
    background_tasks: list[asyncio.Task] = []
    maintenance.start(background_tasks)

    yield

    logger.info("application_shutting_down")
    await maintenance.stop(background_tasks)
    await engine.dispose()  # close pooled connections cleanly


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        description="AI Knowledge Intelligence Platform",
        version="0.1.0",
        lifespan=lifespan,
        # Interactive docs are a development convenience and an information
        # leak in production.
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # --- Middleware -------------------------------------------------------
    # Starlette runs middleware in reverse registration order, so the one
    # added last wraps the others. RequestContextMiddleware is added last on
    # purpose: it must be outermost to time the whole request and to have the
    # request id set before anything else logs.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    # --- Errors -----------------------------------------------------------
    register_exception_handlers(app)

    # --- Routes -----------------------------------------------------------
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
