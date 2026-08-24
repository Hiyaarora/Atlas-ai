# deploy/

Container definitions for a complete, reproducible Atlas AI deployment:
PostgreSQL, the API, and the built frontend behind nginx.

**Development does not need Docker.** Running natively is faster to iterate on
— see the root [`README.md`](../README.md). This stack exists to deploy the
finished application.

## Run it

```bash
cp .env.example .env          # from the repo root; set SECRET_KEY and GEMINI_API_KEY
cd deploy
docker compose --env-file ../.env up --build
```

Then open **http://localhost:8080**.

First start takes a few minutes: two images build, and the backend applies
migrations before it begins serving. `docker compose ps` shows each service
becoming healthy.

```bash
docker compose --env-file ../.env down       # stop, keep data
docker compose --env-file ../.env down -v    # stop and delete data
docker compose logs -f backend               # follow the API
```

## Contents

| File                    | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `docker-compose.yml`    | The stack: postgres + backend + nginx-served frontend |
| `backend.Dockerfile`    | Python image, `development` and `production` targets  |
| `frontend.Dockerfile`   | Node build, served from nginx in `production`         |

## Design notes

**One origin.** nginx serves the app and proxies `/api` to the backend, so the
browser never makes a cross-origin request. That removes CORS entirely and
makes the refresh cookie first-party — it does not need `SameSite=None`, which
is the setting that would otherwise be required and is exactly the one that
weakens CSRF protection.

**Migrations run in the entrypoint, not at app startup.** FastAPI's lifespan
runs once per worker, so schema changes would race between workers and a
failure would surface as a half-working app. Running them before the server
starts means a bad migration is a container that refuses to start.

**One uvicorn worker, deliberately.** Login rate limiting counts failures in
process memory. Additional workers would not share that state, so the
effective limit would silently become the configured limit times the worker
count — a security control weakened by a performance setting. Scaling out
requires moving the counter to shared storage first.

**Two named volumes.** `postgres_data` holds the database; `backend_storage`
holds uploaded files and the vector index. Both are needed: without the
second, a restart would leave document rows in Postgres pointing at files and
vectors that no longer exist, and every conversation would retrieve nothing.

**Nothing but the frontend is published.** Postgres and the API are reachable
only inside the compose network. That is what makes it safe for the backend to
trust `X-Forwarded-For` for per-IP rate limiting — nginx is the only thing
that can reach it.

## Before deploying anywhere real

The startup guard refuses to run in production with unsafe settings, so these
are enforced rather than merely recommended:

```ini
APP_ENV=production
DEBUG=false
LOG_FORMAT=json
SECRET_KEY=<generated, not the default>
REFRESH_COOKIE_SECURE=true    # requires HTTPS
```

Terminate TLS in front of nginx — a managed load balancer, or Caddy/Traefik.
`REFRESH_COOKIE_SECURE=true` without HTTPS means the browser never sends the
cookie and every session dies on reload.
