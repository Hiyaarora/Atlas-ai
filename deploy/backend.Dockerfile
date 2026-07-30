# ==========================================================================
# Atlas AI backend image
#
# Multi-stage: `development` (hot reload, dev tools) and `production`
# (slim, non-root, no reload). Compose targets `development` today; the
# production target is not yet exercised.
# ==========================================================================

# ---- base ----------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is used by container healthchecks and debugging.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first. Docker caches this layer, so
# editing application code does not trigger a full reinstall.
COPY pyproject.toml ./
COPY app ./app

# ---- development ---------------------------------------------------------
FROM base AS development

RUN pip install -e ".[dev]"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---- production ----------------------------------------------------------
FROM base AS production

RUN pip install --no-cache-dir .

# Never run application code as root in a container.
RUN useradd --create-home --shell /bin/bash atlas \
    && chown -R atlas:atlas /app
USER atlas

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
