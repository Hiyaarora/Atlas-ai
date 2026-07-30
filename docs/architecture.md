# Atlas AI — Architecture

This document records *why* the system is shaped the way it is. Code shows
what was built; this shows what was decided and what was rejected.

---

## 1. Guiding principle: dependencies point inward

```
api  ──▶  services  ──▶  models / db
 │                            ▲
 └──────────  schemas  ───────┘
```

- `api/` knows about HTTP. It parses requests, calls a service, returns a schema.
- `services/` knows about the business domain. It must run unchanged if called
  from a CLI, a background worker, or a test.
- `models/` and `db/` know about persistence.

**Enforced rule:** a service must never `import` from `api/`, and must never
raise `HTTPException`. It raises domain errors from `core/exceptions.py`,
which a single handler maps to status codes.

Why it matters here specifically: the ingestion pipeline is called from an
HTTP upload endpoint and from a background job. That only works if the
pipeline never knew it was a web request.

---

## 2. Layers

### API layer (`app/api/`)

Versioned under `/api/v1`. Version is in the URL rather than a header because
it is greppable, cacheable, and visible in logs.

Route functions stay thin — validate, delegate, return. When a route grows an
`if` about business rules, that logic belongs in a service.

### Core (`app/core/`)

Cross-cutting concerns imported by every other layer:

- **`config.py`** — the only module that reads the environment. Pydantic
  validates at import, so bad configuration crashes at boot with a clear
  message instead of failing on line 400 an hour later.
- **`logging.py`** — one pipeline, two formatters. Every record carries a
  `request_id` pulled from a `ContextVar`.
- **`exceptions.py`** — domain exception hierarchy plus the single error
  envelope every failure response uses.

### Services (`app/services/`)

Where the interesting engineering lives:

| Service           | Responsibility                                      |
| ----------------- | --------------------------------------------------- |
| Knowledge Service | ingest, parse, chunk, embed, index, delete           |
| Retrieval Service | BM25 + dense search, fusion, reranking, evaluation   |
| LLM Service       | provider-agnostic generation and streaming           |

### Data layer

**PostgreSQL** for relational, transactional truth: users, documents,
collections, conversations, messages.

**ChromaDB** for vectors. Kept separate on purpose — vector search and
relational integrity are genuinely different workloads, and swapping Chroma
for pgvector, Qdrant, or Weaviate later must not touch business logic.

---

## 3. Decisions and trade-offs

### No LangChain (for now)

LangChain would have produced a working RAG pipeline in an afternoon. It would
also have hidden exactly the mechanics this project exists to demonstrate:
how chunk boundaries affect recall, why hybrid retrieval beats pure dense,
what reciprocal rank fusion actually computes, why a cross-encoder reranks
better than a bi-encoder retrieves.

Manual implementation also means the debugging surface is our own code rather
than a framework's abstraction stack.

**Revisit when:** we need many exotic loaders, or agentic tool-calling flows
where the orchestration cost genuinely exceeds the cost of understanding.

### Async SQLAlchemy over sync

A RAG request spends most of its wall-clock time waiting — on the embedding
API, the vector store, and the LLM. Async lets one worker handle many
concurrent requests during that wait. Sync would need a thread per request.

Cost: `asyncpg` for the app, `psycopg` for Alembic (migration scripts are sync
code), and the `MissingGreenlet` class of bug if lazy loading is used
carelessly. `expire_on_commit=False` in `db/session.py` prevents the most
common instance of it.

### Application factory (`create_app()`)

A module-level `app = FastAPI()` is simpler, but it makes the app a global
built once at import — impossible to construct twice with different settings.
Tests build isolated instances per test; the factory makes that trivial.

### Liveness and readiness are separate endpoints

- `/health/live` checks nothing external. Failing it means the orchestrator
  **restarts** the container.
- `/health/ready` checks every hard dependency. Failing it means the load
  balancer **stops routing** to the instance without killing it.

Collapsing them into one `/health` that pings the database is the common
mistake, and it turns a brief Postgres blip into a container restart storm
across every replica at once.

### One error envelope

```json
{
  "error": {
    "code": "not_found",
    "message": "Document not found",
    "details": {},
    "request_id": "0f9c1a…"
  }
}
```

Machine-readable `code` for client branching, human `message` for display,
`request_id` so a user's bug report maps to exact log lines. The frontend
parses this once, in `lib/api/client.ts`, and converts it to a typed
`ApiError`.

### Login reports unknown accounts explicitly (accepted risk)

**Decision, 2026-07-29.** Logging in with an unregistered email returns
`account_not_found` — "No account found for that email address" — rather than
a generic failure.

The security-conventional choice is a single generic message, because a
distinct one turns the login form into a **user-enumeration oracle**: an
attacker scripts it against leaked email lists to learn who holds an account,
which assists targeted phishing and cuts the cost of credential stuffing.

We accepted that risk deliberately, for clearer UX. Two consequences follow
and are already implemented:

* The constant-time dummy-hash defence was **removed**. Once the response body
  states whether an account exists, equalising response timing hides nothing —
  and it let an attacker force 250ms of bcrypt work per request. Restoring one
  without the other is pointless; `core/security.py` records this.
* Account-*disabled* status is still only revealed **after** the password
  verifies, so suspended accounts cannot be probed.

Mitigations that make this acceptable, still to be added: per-IP and
per-email rate limiting on `/auth/login`, plus alerting on
enumeration-shaped traffic.

### Refresh must be single-flight on the client

Refresh tokens rotate, so using one revokes it. Two concurrent refreshes with
the same cookie means the second presents a revoked token and gets a 401 —
logging the user out mid-session.

This bit us immediately: React's `<StrictMode>` invokes effects twice in
development, so the session-bootstrap effect fired two parallel refreshes and
signed the user out on every page reload. `features/auth/api.ts` now shares
one in-flight promise across all callers.

Known remaining gap: the promise is per-tab. Two tabs reloading at the same
moment still race. The fix is refresh-token *families* with reuse detection,
where replaying a revoked token revokes the whole family as a suspected
theft rather than silently failing.

### Streaming outlives the request

A `StreamingResponse` endpoint *returns* its generator and finishes. FastAPI
then tears down the request's dependencies — closing the session `get_db`
yielded — and only afterwards is the generator iterated.

So `chat_service.stream_assistant_reply` opens and owns its own session. The
route persists the user's message with the request session *before* streaming
starts, while a real HTTP status code can still be returned; everything after
the first byte belongs to the generator.

This is invisible to ordinary unit tests, which is why
`test_streaming_uses_its_own_session_not_the_request_session` asserts it
structurally.

Corollary: once headers are on the wire the status code is fixed. Mid-stream
failures are delivered as an SSE `error` event, never as a 500.

### CPU-bound work must leave the event loop

bcrypt at cost 12 takes ~280ms and is synchronous. Called directly from an
async endpoint it blocks the entire loop: measured, `/health/live` went from
**2ms idle to 1459ms** while five registrations were in flight — enough for an
orchestrator to fail the readiness probe and pull a healthy instance from
rotation during a login spike.

`auth_service` now wraps both hashing and verification in `asyncio.to_thread`.
The bcrypt extension releases the GIL, so the work genuinely parallelises;
health under the same load dropped to ~268ms.

The general rule for this codebase: **anything CPU-bound and longer than a few
milliseconds goes to a thread.** PDF parsing, embedding generation and
cross-encoder inference all fall squarely under it.

### Postgres `now()` is transaction time, not wall time

`now()` and `CURRENT_TIMESTAMP` return the *transaction start* and stay
constant for its duration. Two messages inserted in one transaction therefore
received byte-identical `created_at`, ordering fell through to a random-UUID
tiebreak, and transcripts came back scrambled — assistant before user.

`messages.created_at` uses `clock_timestamp()`, which reads the real clock on
every call. Ordering is now correct however the writes are batched.
`TimestampMixin` keeps `now()`: for audit columns, transaction-consistent
timestamps are the desirable behaviour.

### Chunks live in Postgres *and* Chroma

This looks like duplication. It is not — the two answer different questions.

Postgres is the **source of truth**: it survives a corrupted index, serves
citation lookups relationally, and lets the corpus be re-embedded with a
better model without re-parsing a single PDF. Chroma is a **derived search
index**: deletable and rebuildable from Postgres at any time.

Treating the vector store as disposable is what makes changing embedding model
a migration rather than a data-loss event — which matters, because changing
embedding model invalidates every stored vector. Old and new vectors occupy
different spaces and their similarities are meaningless against each other.
`documents.embedding_model` records which model produced the current vectors
so that mismatch is detectable rather than silent.

### Chunking happens per page

Chunking across the whole document is marginally more efficient at page
boundaries. Chunking per page means every chunk carries exactly one page
number — and a chunk spanning pages 4 and 5 can be cited as neither.

Citations are the product. Efficiency loses.

### Retrieval needs two thresholds, not one

Measured against `gemini-embedding-001`: for the question *"how do I stop the
primary discarding logs my replica needs"*, the correct passage scored
**0.677**, and two entirely unrelated passages scored **0.586** and **0.583**.

Embedding models have a high, model-specific baseline similarity between any
two texts. So an absolute threshold cannot separate those: anything excluding
0.586 would also exclude genuinely relevant passages on an easier question.

The **gap** is the signal, not the magnitude. Retrieval therefore applies:

1. an **absolute floor** — rejects everything when the corpus has no answer,
   where a relative cutoff would still keep the best of a uniformly
   irrelevant set;
2. a **relative cutoff** — keeps only hits within 90% of the best score,
   which an absolute floor cannot express.

Both defaults are evidence-based starting points, tuned against a labelled
evaluation set by the harness in `app/eval/`.

### Asymmetric embedding

Passages are embedded with `task_type=RETRIEVAL_DOCUMENT`, questions with
`RETRIEVAL_QUERY`. A question and the passage answering it are different kinds
of text; telling the model which is which places them in a space optimised for
matching one against the other rather than for general similarity.

### Ingestion runs in the background

Parsing, chunking and embedding a large PDF takes tens of seconds. Holding the
upload request open for that would time out behind most proxies and freeze the
UI, so upload returns **202 Accepted** with `status: pending` and the client
polls.

The consequence: the ingestion task runs after the response, so — like the
chat stream — it opens its own session, and it can never raise. There is no
request left to return an error to, so failures are recorded on the row as
`status="failed"` with a message the user can act on.

### Uploaded filenames never reach the filesystem

Files are stored under `{user_id}/{uuid}{whitelisted-extension}`. The user's
filename is kept in `documents.filename` as a display label only.

A file called `../../../../etc/passwd.txt` is then a harmless label rather
than a location. There is a test asserting exactly that.

### A conversation is bound to one document, permanently

`conversations.document_id` is the isolation mechanism. Retrieval resolves its
scope from the **conversation row**, never from the logged-in user:

    conversation_id → (ownership check) → document_id → filtered vector search

Three properties make it a guarantee rather than a convention:

1. **Unscoped retrieval is not expressible.** `retrieve()` requires a
   non-empty `document_ids`; "search everything this user owns" has no
   signature. An empty scope returns no results, never a widened search.
2. **The filter is defence in depth** — `owner_id` *and* `document_id` both go
   into the vector-store `where` clause.
3. **The binding is durable and immutable.** It lives in the database, so
   reopening a conversation months later searches the same document
   regardless of what was uploaded since — and there is no API to re-point it,
   because a re-pointable binding would break exactly that promise.

Scope is a *list* internally although a conversation binds to one document.
Multi-document workspaces then change only `resolve_scope`; retrieval,
generation and citations are untouched.

### Two state machines, not one enum

`documents` carries `ingestion_status` (pending → processing → ready → failed)
and `lifecycle_status` (active → archived → pending_deletion) as separate
columns. They are orthogonal — a document can be `ready` *and* `archived` —
and merging them would make illegal states representable while forcing every
query to disambiguate which meaning it wanted.

### Reference counts are recomputed, never decremented

`reference_count` is a cache of
`SELECT count(*) FROM conversations WHERE document_id = ?`, and it is only
ever written by assigning that subquery.

`count = count - 1` is the version that drifts: two concurrent deletes both
read 2 and both write 1, and any transaction rolling back after the decrement
leaves the counter permanently wrong. A single atomic recomputation has no
read-modify-write to lose. The janitor also reconciles every counter on each
sweep, so drift introduced by any other path heals itself.

### Deletion is a three-stage pipeline, never immediate

    active ──(no live conversations)──▶ archived
    archived ──(reference_count = 0)──▶ pending_deletion
    pending_deletion ──(grace period)──▶ purged

Reaching zero references starts a clock; it does not destroy anything. The
guarantee that matters is that **marking and purging can never happen in the
same sweep** — marking sets `deletion_scheduled_at = now()` and purging
requires it to be older than the grace period. Reference count is re-checked
at the moment of deletion, so a conversation created during the grace period
rescues the document.

Archiving drops **vectors only**; chunks stay in Postgres. Restoring is
therefore a re-embed, not a re-upload — the original file is not even
required. That is the dividend of treating the vector store as a derived
index, and it is what makes an accidental delete genuinely recoverable.

An explicit user delete is a *soft* delete: vectors go immediately (so the
document is unfindable within milliseconds) and everything else survives the
grace period.

### The janitor runs in-process, behind an advisory lock

Cleanup is an asyncio task in the app lifespan — no broker, no second
deployment artifact, no configuration drift. `uvicorn --workers 4` would
however start four schedulers, so each sweep is guarded by
`pg_try_advisory_lock`. Exactly one worker does the work; the rest find it
taken and go back to sleep. If that worker dies its connection drops and the
lock releases itself, so there is no stale-lock recovery to write.

Every interval is configuration (`conversation_inactive_days`,
`document_deletion_grace_days`, `maintenance_interval_minutes`); nothing is
hardcoded, and `maintenance_enabled=false` turns it off entirely for tests.

### Request ID propagation

`RequestContextMiddleware` adopts an inbound `X-Request-ID` or mints one,
stores it in a `ContextVar`, and echoes it on the response. `ContextVar`, not
thread-local: with async, many requests interleave on one thread, and
thread-locals would leak ids between them.

### Alembic from day one, before any models exist

Setting up migrations after tables already exist means hand-writing a
baseline. Ten minutes now saves an afternoon later. The naming convention in
`db/base.py` is the load-bearing part — without it, Postgres auto-names
constraints and autogenerate cannot reliably alter them.

---

## 4. Request lifecycle

```
Browser
  │  fetch() with typed client
  ▼
CORS middleware            ← is this origin allowed?
  ▼
RequestContextMiddleware   ← mint request_id, start timer
  ▼
Router                     ← match path, coerce params
  ▼
Dependencies               ← get_db() opens a session (later: get_current_user)
  ▼
Route handler              ← thin
  ▼
Service                    ← business logic; raises domain errors
  ▼
Database / vector store
  ▼
Pydantic response model    ← filters output to the declared contract
  ▼
Access log + X-Request-ID header
```

On failure the exception handler intercepts before the response is built, and
emits the same envelope regardless of where the error originated.

---

## 5. Planned extension points

**New knowledge source**: implement a loader satisfying the source
contract — `load() -> list[RawDocument]` — and register it. Chunking,
embedding, retrieval, and generation are untouched.

**New LLM provider**: implement the `LLMProvider` interface —
`generate()` and `stream()` — and change one config value. No business logic
changes.

**New retrieval strategy**: retrievers share one interface, so BM25,
dense, and hybrid are interchangeable and independently benchmarkable.

---

## 6. Known gaps (deliberate)

| Gap                                        | Notes                              |
| ------------------------------------------ | ---------------------------------- |
| No rate limiting on `/auth/login`          | required by the enumeration tradeoff |
| No metrics endpoint                        | counters exist, no exporter yet    |
| Citations are not persisted                | reloading a chat loses its sources |
| No OCR                                     | image-only PDFs and decks fail     |
| Not containerized                          | draft files in `deploy/`           |
| No CI pipeline                             | tests run locally only             |

### Why containers come last

Everything runs natively: Python, Node, and PostgreSQL installed on the
development machine. This is a deliberate sequencing decision, not an
oversight.

Docker solves *distribution* — "it works on my machine" — and it is genuinely
required before anyone deploys this. But it solves nothing about whether the
retrieval layer returns relevant chunks, or whether the prompt grounds the
model in its context. Introducing it early buys one benefit (uniform
environments for a single developer, who already has a uniform environment)
at the cost of a container rebuild between you and every experiment, and a
second system to debug whenever something breaks.

The cost of deferring is low and bounded: the application already reads all
configuration from the environment, binds no hostnames, and writes nothing to
hardcoded paths. Those are the properties that make containerization
mechanical rather than a rewrite: a Dockerfile is added to software that was
built container-ready from the first commit.
