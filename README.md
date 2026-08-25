# Atlas AI

**AI Knowledge Intelligence Platform**

Upload your documents, ask questions in natural language, get answers grounded
in those documents with citations back to the page.

Atlas AI is not a "chat with PDF" script. The RAG pipeline — chunking,
embedding, hybrid retrieval, rank fusion, reranking, grounded generation — is
implemented directly rather than assembled from a framework, and every
retrieval threshold in it was derived from measurement rather than guessed.

**▶ Live demo: https://13-206-126-184.sslip.io** — running on AWS. Create an
account and upload a document; nothing to install.

---

## Why this exists

Keyword search fails on the documents you actually need answers from —
contracts, runbooks, API docs, research papers, lecture notes. General-purpose
language models are fluent but have never seen your private files, and
confidently invent details when asked about them.

Atlas AI retrieves relevant passages first, then constrains the model to answer
from that evidence, with citations back to the source. When the documents do
not contain an answer, retrieval returns nothing and the model says so.

---

## What it does

- **Ingests** PDF, DOCX, PPTX, XLSX, CSV, HTML, Markdown and plain text, with
  tables rendered into readable text and archive-bomb guards on the OOXML
  formats.
- **Reads text inside images** — screenshots, scanned pages, diagrams with
  labels — via local OCR, indexed against the page or slide it came from and
  marked so a passage can be traced to how it was obtained.
- **Retrieves** with dense vector search and BM25 in parallel, fused by
  Reciprocal Rank Fusion, with an optional cross-encoder reranking the
  shortlist.
- **Refuses** when the corpus cannot answer the question, rather than returning
  the least-irrelevant paragraph and letting the model improvise from it.
- **Cites** every answer back to a filename and page.
- **Isolates** conversations: a conversation is bound to one document in the
  database, and no code path can express a search that crosses that boundary.
- **Streams** replies token by token over Server-Sent Events.
- **Manages document lifecycle** with reference counting, reversible archiving
  and a grace period before permanent deletion, swept by a background job.

---

## Architecture

```
┌───────────────────────────────────────────────┐
│  React + TypeScript + Tailwind                │
└───────────────────────┬───────────────────────┘
                        │  JSON / SSE over HTTP
┌───────────────────────▼───────────────────────┐
│  API Layer          FastAPI routers, schemas  │
├───────────────────────────────────────────────┤
│  Auth Layer         JWT, password hashing     │
├───────────────────────────────────────────────┤
│  Service Layer      business logic, no HTTP   │
│    ├── Document Service    ingest & manage    │
│    ├── Retrieval Service   search & rank      │
│    └── Chat Service        grounded answers   │
├───────────────────────────────────────────────┤
│  Data Layer                                   │
│    ├── PostgreSQL   users, docs, chats        │
│    └── ChromaDB     vectors, chunk metadata   │
└───────────────────────────────────────────────┘
```

Dependencies point **inward only**: routes may import services, services may
never import routes, and no service raises an HTTP exception. See
[`docs/architecture.md`](docs/architecture.md) for the reasoning behind each
layer.

### The retrieval pipeline

```
question
   │
   ├──────────────┬──────────────┐
   ▼              ▼              │  both scoped to the conversation's
 dense          BM25             │  document — enforced in each branch
 (Chroma)       (Postgres)       │
   │  top 20      │  top 20      │
   └──────┬───────┘              │
          ▼                      │
   Reciprocal Rank Fusion        │
          │                      │
          ▼                      │
   answerability gate  ──────────┘  returns nothing when the corpus
          │                         cannot answer the question
          ▼
   cross-encoder rerank (optional)
          │
          ▼
      top 6 passages  →  grounded generation  →  answer + citations
```

Fusion combines the two retrievers by **rank, never by score**. Cosine
similarity is bounded in [0, 1]; BM25 is unbounded and corpus-dependent. Adding
them is meaningless and normalising them per query is unstable, because the
normalised value then depends on which candidates happened to come back.

---

## Tech stack

| Layer      | Choice                                         |
| ---------- | ---------------------------------------------- |
| Backend    | Python 3.12, FastAPI, Uvicorn                  |
| Frontend   | React 19, TypeScript, Vite, Tailwind v4        |
| Database   | PostgreSQL 17, SQLAlchemy 2.0 (async), Alembic |
| Vectors    | ChromaDB behind a `VectorStore` interface      |
| Embeddings | Gemini behind an `EmbeddingProvider` interface |
| LLM        | Gemini behind an `LLMProvider` interface       |
| Reranking  | sentence-transformers cross-encoder, local     |
| Auth       | JWT + bcrypt, rotating refresh tokens          |
| Quality    | Pytest, Ruff, Black, isort                     |

**No LangChain.** Chunking, embedding, retrieval, fusion and reranking are
implemented directly. Every provider sits behind an interface, so swapping the
model or the vector store is a configuration change rather than a rewrite.

---

## Evaluation

Retrieval quality is measured, not asserted. `backend/app/eval/` holds a
reproducible harness: a labelled question set, a fixture corpus, and metrics.

```bash
python -m app.eval.run                  # compare configurations
python -m app.eval.run --calibrate      # score distributions, for thresholds
python -m app.eval.run --per-query      # per-question breakdown
```

It ingests its own corpus through the real pipeline, so a change to *chunking*
is measurable and not just a change to search. Questions are labelled by
**content marker** rather than chunk id, because chunk ids are regenerated on
every ingestion and would rot the moment chunk size changed.

Reported metrics:

| Metric        | Question it answers                                  |
| ------------- | ---------------------------------------------------- |
| recall@k      | did the evidence get retrieved at all?               |
| precision@k   | how much of what came back was evidence?             |
| MRR           | how near the top was the first correct passage?      |
| nDCG@k        | rank-weighted credit for finding several             |
| refusal rate  | how often does an unanswerable question return none? |
| latency       | p50 / p95 / max, end to end and per stage            |

The harness **withholds results** if a retriever or the reranker silently
degraded during a run — a rate-limited embedding API once made a healthy
pipeline look broken, and reporting those numbers would have been worse than
reporting nothing.

---

## Getting started

### Prerequisites

| Tool       | Version | Install                            |
| ---------- | ------- | ---------------------------------- |
| Python     | 3.12+   | https://www.python.org/downloads/  |
| Node.js    | 22+     | `winget install OpenJS.NodeJS.LTS` |
| PostgreSQL | 17      | see step 2 below                   |

### 1 · Configure

```powershell
git clone https://github.com/Hiyaarora/Atlas-ai.git atlas-ai
cd atlas-ai

copy .env.example .env
```

Open `.env` and set:

- `SECRET_KEY` — generate with
  `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- `POSTGRES_PASSWORD` — any password; this becomes the app's database password
- `GEMINI_API_KEY` — from https://aistudio.google.com/apikey

### 2 · PostgreSQL

**Portable install (recommended — no admin rights, no Windows service).**
Download the official binaries ZIP:

```powershell
$url = "https://get.enterprisedb.com/postgresql/postgresql-17.10-2-windows-x64-binaries.zip"
Invoke-WebRequest $url -OutFile "$env:USERPROFILE\Downloads\postgresql-17-binaries.zip"

.\scripts\install-postgres.ps1
```

This extracts to `~\pgsql`, initialises a cluster in `~\pgdata` with SCRAM
authentication, and starts the server on port 5432. Everything it touches
lives in those two directories — delete them to uninstall completely.

> EnterpriseDB's graphical `.exe` installer requires elevation and crashes with
> `0xc0000005` on some Windows 11 builds. The ZIP ships identical binaries
> without the installer wrapper. If you already have PostgreSQL installed by
> any means, skip this and go straight to `setup-db.ps1`.

Then create the application's role and database:

```powershell
.\scripts\setup-db.ps1
```

Reads your `.env`, creates the `atlas` role and `atlas` database, then
reconnects **as the app role** to prove the credentials work. Safe to re-run.

<details>
<summary>Doing it manually instead</summary>

```sql
CREATE ROLE atlas WITH LOGIN PASSWORD 'your-password';
CREATE DATABASE atlas OWNER atlas;
```
</details>

**Daily control.** The portable install registers no Windows service, so start
the server when you sit down to work:

```powershell
.\scripts\db.ps1 start     # also: stop | status | logs | psql
```

### 3 · Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

alembic upgrade head
uvicorn app.main:app --reload
```

If the API can't reach the database, check the server is up with
`.\scripts\db.ps1 status`.

API on **http://localhost:8000**, interactive docs at `/docs`.

### 4 · Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

App on **http://localhost:5173**.

### Verify

| URL                                        | Expected                          |
| ------------------------------------------ | --------------------------------- |
| http://localhost:5173                      | sign-in screen                    |
| http://localhost:8000/docs                 | interactive API documentation     |
| http://localhost:8000/api/v1/health/live   | `{"status":"ok"}`                 |
| http://localhost:8000/api/v1/health/ready  | `200` with postgres `ok`          |

If readiness returns `503`, PostgreSQL is not running or `.env` is wrong — the
response body names the failing dependency.

---

## Everyday commands

```powershell
# backend (from backend/, venv activated)
uvicorn app.main:app --reload      # run with hot reload
pytest -q                          # run tests
isort . ; black . ; ruff check --fix .    # format and autofix
ruff check . ; black --check .            # verify without changing files

# database (from backend/)
alembic revision --autogenerate -m "add users table"
alembic upgrade head
alembic downgrade -1

# frontend (from frontend/)
npm run dev          # dev server with HMR
npm run typecheck    # TypeScript, no emit
npm run build        # production bundle
```

---

## Project layout

```
atlas-ai/
├── backend/
│   ├── app/
│   │   ├── api/v1/routes/   HTTP endpoints — thin, no business logic
│   │   ├── core/            config · logging · error contract · security
│   │   ├── db/              engine, session lifecycle, declarative base
│   │   ├── embeddings/      provider interface + implementations
│   │   ├── eval/            retrieval evaluation harness and golden set
│   │   ├── ingestion/       parsers, chunking, table rendering
│   │   ├── jobs/            background maintenance
│   │   ├── llm/             provider interface + implementations
│   │   ├── middleware/      request id, access logging
│   │   ├── models/          SQLAlchemy ORM models
│   │   ├── retrieval/       dense · BM25 · fusion · rerank · pipeline
│   │   ├── schemas/         Pydantic request/response contracts
│   │   ├── services/        business logic — transport-agnostic
│   │   ├── vectorstore/     vector store interface + Chroma adapter
│   │   └── main.py          application factory
│   ├── alembic/             migration environment and versions
│   └── tests/
├── frontend/
│   └── src/
│       ├── config/          typed environment access
│       ├── features/        one folder per feature (api, hooks, components)
│       └── lib/api/         shared HTTP client and error types
├── scripts/                 local developer scripts
├── deploy/                  container definitions (draft)
├── docs/architecture.md
└── .env.example
```

---

## Deployment

The whole stack runs from one command:

```bash
cd deploy
docker compose --env-file ../.env up --build
```

Open **http://localhost:8080**. PostgreSQL, the API and the built frontend
come up together; nginx serves the app and proxies `/api` to the backend, so
everything is on one origin — no CORS, and the refresh cookie is first-party.
Migrations are applied by the container entrypoint before the API serves
anything.

See [`deploy/README.md`](deploy/README.md) for the design decisions and the
settings that must change before deploying anywhere public. The application
**refuses to start** in production with a default signing key, `DEBUG=true`,
or an insecure refresh cookie — a misconfigured deploy fails loudly instead of
serving quietly.

### Running on AWS

The live instance runs the same Compose stack on a single EC2 host:

```
Internet
   │  HTTPS
   ▼
Caddy ── automatic TLS, Let's Encrypt
   │  127.0.0.1:8080
   ▼
nginx ── serves the built frontend, proxies /api
   │
   ├── FastAPI backend ── Tesseract OCR, ChromaDB
   └── PostgreSQL
```

| | |
| ---------- | ---------------------------------------------------- |
| Compute    | EC2 `t4g.small` (AWS Graviton, arm64), ap-south-1     |
| Storage    | 30 GB gp3 — Docker volumes for Postgres, vectors and uploads |
| TLS        | Caddy, certificates issued and renewed automatically  |
| Networking | Elastic IP; SSH restricted to one address; only 80/443 public |

**Built for arm64.** The image is rebuilt on Graviton rather than emulated —
PyMuPDF, ChromaDB and Tesseract all run natively.

**Nothing stateful is ephemeral.** PostgreSQL, the Chroma index and uploaded
files live on named Docker volumes on the attached EBS disk, so they survive
container replacement and host reboots. Chroma stores a SQLite database and
memory-mapped index files, which is why it is given a block device rather than
network storage.

**Secrets stay on the host.** The production environment file is generated on
the instance, is never committed, and never reaches the frontend bundle. The
database and API listen on loopback only; the security group exposes nothing
but HTTP and HTTPS.

Redeploying after a push:

```bash
git pull && cd deploy && docker compose --env-file ../.env up -d --build
```

## License

MIT
