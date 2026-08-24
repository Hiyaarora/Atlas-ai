# ==========================================================================
# Atlas AI backend image
#
# Two stages. `development` bind-mounts source and hot-reloads; `production`
# installs a fixed copy, runs as a non-root user, and applies migrations
# before serving.
# ==========================================================================

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl backs the compose healthcheck and is worth having when debugging a
# container that will not come up.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency manifest first and alone: Docker caches this layer, so editing
# application code does not reinstall the world.
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Application code, then the migration environment. `alembic/` and
# `alembic.ini` are NOT optional in the image — the container runs
# `alembic upgrade head` at startup, and without them it would boot against
# whatever schema happened to be there.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ---- development ---------------------------------------------------------
FROM base AS development

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---- production ----------------------------------------------------------
FROM base AS production

# Uploads and the vector index are written at runtime. Created here and owned
# by the app user so a mounted volume does not land root-owned and unwritable.
RUN mkdir -p /app/storage \
    && useradd --create-home --shell /bin/bash atlas \
    && chown -R atlas:atlas /app
USER atlas

EXPOSE 8000

# One worker, deliberately.
#
# Login rate limiting counts failures in process memory (see
# core/rate_limit.py). Four workers would not share that state, so the
# effective limit would silently become four times the configured one — a
# security control quietly weakened by a performance setting. Scaling out
# requires moving the counter to shared storage first.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
