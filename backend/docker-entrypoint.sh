#!/usr/bin/env bash
# Applies migrations, then hands off to the container command.
#
# Migrations run here rather than in the application's startup event on
# purpose: FastAPI's lifespan runs once per worker, so schema changes would
# race between workers, and a migration failure would surface as a half-broken
# app rather than a container that refuses to start.
set -euo pipefail

echo "[entrypoint] applying database migrations"
alembic upgrade head

echo "[entrypoint] starting: $*"
exec "$@"
