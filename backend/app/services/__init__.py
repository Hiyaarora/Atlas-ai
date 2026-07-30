"""Business logic.

Rules for this layer:
  * It may import from `models`, `schemas`, `db`, and `core`.
  * It must NOT import from `api` - dependencies point inward only.
  * It must NOT raise `HTTPException`; raise domain errors from
    `app.core.exceptions` instead so the logic stays transport-agnostic.
"""
