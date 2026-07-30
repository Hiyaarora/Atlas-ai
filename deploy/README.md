# deploy/

Container definitions for Atlas AI. **You do not need Docker installed to
develop or run this project** — everything runs natively (see the root
[`README.md`](../README.md)).

These files are written but **not yet exercised**: build paths, contexts and
the production targets have not been verified end to end. Treat them as a
starting point rather than a supported path.

## Contents

| File                    | Purpose                                       |
| ----------------------- | --------------------------------------------- |
| `docker-compose.yml`    | Full stack: postgres + backend + frontend     |
| `backend.Dockerfile`    | Multi-stage Python image (dev + production)   |
| `frontend.Dockerfile`   | Multi-stage Node image (dev + nginx static)   |

## Why the application is container-ready anyway

Containerisation is a packaging concern, and the application was built so that
adding it is mechanical rather than a rewrite:

- every tunable value is read from the environment, in one place
- no hostname or port is hardcoded
- nothing is written to a fixed path outside `storage/`
- the database schema is managed by migrations, not by application startup

Those four properties are what make the difference between "add a Dockerfile"
and "restructure the app".
