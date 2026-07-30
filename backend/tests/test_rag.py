"""The full pipeline: upload -> parse -> chunk -> embed -> index -> retrieve.

These run against real PostgreSQL and a real (temporary) Chroma index, with a
deterministic embedding provider. Nothing is mocked except the network.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.conversation import Conversation
from app.models.document import Chunk, Document
from app.models.user import User
from app.schemas.auth import RegisterRequest
from app.services import auth_service, document_service, retrieval_service
from app.vectorstore import get_vector_store
from tests.conftest import TEST_PASSWORD
from tests.test_chat import parse_sse
from tests.test_ingestion import make_pdf

PREFIX = settings.api_v1_prefix

POSTGRES_NOTES = (
    "PostgreSQL indexing guide. A B-tree index speeds up equality and range "
    "queries on ordered data. Use a partial index when queries always filter "
    "on the same predicate. The pg_stat_user_indexes view reports how often "
    "each index is actually used."
)

COOKING_NOTES = (
    "Sourdough baking notes. Feed the starter twice before mixing the dough. "
    "Autolyse the flour and water for thirty minutes. Bake at 240 degrees "
    "with steam for the first fifteen minutes to develop the crust."
)


async def upload(client: AsyncClient, headers: dict[str, str], name: str, body: bytes) -> dict:
    """Upload a file. Returns the document payload, with `conversation_id`
    folded in — upload now creates a conversation as part of the same
    request."""
    response = await client.post(
        f"{PREFIX}/documents",
        headers=headers,
        files={"file": (name, body, "text/plain")},
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    return {**payload["document"], "conversation_id": payload["conversation_id"]}


async def _all_document_ids(session: AsyncSession, owner_id: uuid.UUID) -> list[uuid.UUID]:
    """Every document a user owns.

    Retrieval no longer accepts an implicit "everything I own" scope — callers
    resolve scope from a conversation. These tests exercise retrieval itself
    rather than scope resolution, so they build the widest legitimate scope
    explicitly. `test_conversation_isolation.py` covers the resolution path.
    """
    result = await session.execute(select(Document.id).where(Document.owner_id == owner_id))
    return list(result.scalars().all())


@pytest.fixture
async def ingested_document(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> dict:
    """Upload a text file and run ingestion synchronously.

    The endpoint queues ingestion as a background task, which does not run
    under the ASGI test transport. Calling it directly is what makes the test
    deterministic — no polling, no sleeps.
    """
    document = await upload(client, auth_headers, "postgres.txt", POSTGRES_NOTES.encode())
    await document_service.ingest_document(uuid.UUID(document["id"]))
    await db_session.commit()
    return document


# ==========================================================================
# Upload validation
# ==========================================================================


async def test_upload_requires_authentication(client: AsyncClient) -> None:
    response = await client.post(
        f"{PREFIX}/documents", files={"file": ("a.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 401


async def test_upload_returns_202_and_pending_status(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """202, not 201: the document exists but is not yet queryable."""
    body = await upload(client, auth_headers, "notes.txt", POSTGRES_NOTES.encode())

    assert body["status"] == "pending"
    # A conversation is created in the same request, so the client can
    # navigate straight into it.
    assert body["conversation_id"]


async def test_upload_response_hides_implementation_details(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Chunk counts and embedding models mean nothing to a user."""
    body = await upload(client, auth_headers, "notes.txt", POSTGRES_NOTES.encode())

    for leaked in ("chunk_count", "embedding_model", "storage_key", "reference_count"):
        assert leaked not in body, f"{leaked} must not be in the user-facing contract"


async def test_empty_file_is_rejected(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    response = await client.post(
        f"{PREFIX}/documents",
        headers=auth_headers,
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 422


async def test_unsupported_type_is_rejected_at_upload(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Fail where the user sees a status code, not silently in the background."""
    response = await client.post(
        f"{PREFIX}/documents",
        headers=auth_headers,
        files={"file": ("payload.exe", b"MZ\x90\x00", "application/x-msdownload")},
    )

    assert response.status_code == 422


async def test_oversized_file_is_rejected(
    client: AsyncClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 10)

    response = await client.post(
        f"{PREFIX}/documents",
        headers=auth_headers,
        files={"file": ("big.txt", b"x" * 100, "text/plain")},
    )

    assert response.status_code == 422


async def test_malicious_filename_cannot_escape_the_storage_root(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    """Path traversal: the filename is a label, never a location."""
    response = await client.post(
        f"{PREFIX}/documents",
        headers=auth_headers,
        files={"file": ("../../../../etc/passwd.txt", b"root:x:0:0", "text/plain")},
    )

    assert response.status_code == 202
    document = await db_session.get(Document, uuid.UUID(response.json()["document"]["id"]))
    assert document is not None
    assert ".." not in document.storage_key
    assert document.storage_key.endswith(".txt")
    # The display name is sanitised down to a basename.
    assert "/" not in document.filename


# ==========================================================================
# Ingestion
# ==========================================================================


async def test_ingestion_produces_chunks(ingested_document: dict, db_session: AsyncSession) -> None:
    document = await db_session.get(Document, uuid.UUID(ingested_document["id"]))

    assert document is not None
    assert document.ingestion_status == "ready"
    assert document.chunk_count > 0
    assert document.page_count == 1
    assert document.embedding_model


async def test_chunks_are_persisted_with_page_numbers(
    ingested_document: dict, db_session: AsyncSession
) -> None:
    result = await db_session.execute(
        select(Chunk).where(Chunk.document_id == uuid.UUID(ingested_document["id"]))
    )
    chunks = list(result.scalars().all())

    assert chunks
    assert all(chunk.page_number >= 1 for chunk in chunks)
    assert [c.position for c in sorted(chunks, key=lambda c: c.position)] == list(
        range(len(chunks))
    )


async def test_chunks_are_indexed_in_the_vector_store(ingested_document: dict) -> None:
    count = await get_vector_store().count(where={"document_id": ingested_document["id"]})

    assert count > 0


async def test_pdf_ingestion_records_page_count(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    pdf = make_pdf(["First page about indexes.", "Second page about vacuum."])
    response = await client.post(
        f"{PREFIX}/documents",
        headers=auth_headers,
        files={"file": ("guide.pdf", pdf, "application/pdf")},
    )
    document_id = uuid.UUID(response.json()["document"]["id"])

    await document_service.ingest_document(document_id)
    await db_session.commit()

    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.ingestion_status == "ready"
    assert document.page_count == 2


async def test_ingestion_failure_is_recorded_not_raised(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    """There is no request left to return an error to, so it lands on the row."""
    response = await client.post(
        f"{PREFIX}/documents",
        headers=auth_headers,
        files={"file": ("broken.pdf", b"not a real pdf at all", "application/pdf")},
    )
    document_id = uuid.UUID(response.json()["document"]["id"])

    await document_service.ingest_document(document_id)
    await db_session.commit()

    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.ingestion_status == "failed"
    assert document.error


async def test_reingestion_replaces_chunks_rather_than_duplicating(
    ingested_document: dict, db_session: AsyncSession
) -> None:
    document_id = uuid.UUID(ingested_document["id"])
    before = (await db_session.get(Document, document_id)).chunk_count  # type: ignore[union-attr]

    await document_service.ingest_document(document_id)
    await db_session.commit()

    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.chunk_count == before

    result = await db_session.execute(select(Chunk).where(Chunk.document_id == document_id))
    assert len(list(result.scalars().all())) == before


# ==========================================================================
# Retrieval
# ==========================================================================


async def test_retrieval_finds_the_relevant_document(
    client: AsyncClient,
    auth_headers: dict[str, str],
    registered_user: User,
    db_session: AsyncSession,
) -> None:
    """Two documents, one question — the right one must come back."""
    for name, body in (("postgres.txt", POSTGRES_NOTES), ("cooking.txt", COOKING_NOTES)):
        document = await upload(client, auth_headers, name, body.encode())
        await document_service.ingest_document(uuid.UUID(document["id"]))
    await db_session.commit()

    citations = await retrieval_service.retrieve(
        db_session,
        owner_id=registered_user.id,
        document_ids=await _all_document_ids(db_session, registered_user.id),
        query="Which view reports how often an index is used?",
    )

    assert citations, "expected at least one relevant passage"
    assert citations[0].filename == "postgres.txt"
    assert "pg_stat_user_indexes" in citations[0].excerpt


async def test_citations_carry_a_page_number_and_score(
    ingested_document: dict, registered_user: User, db_session: AsyncSession
) -> None:
    citations = await retrieval_service.retrieve(
        db_session,
        owner_id=registered_user.id,
        document_ids=await _all_document_ids(db_session, registered_user.id),
        query="partial index predicate",
    )

    assert citations
    assert citations[0].index == 1
    assert citations[0].page_number >= 1
    assert 0.0 <= citations[0].score <= 1.0
    assert citations[0].chunk_id


async def test_retrieval_is_scoped_to_the_owner(
    client: AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    """The single most consequential property in the pipeline."""
    document = await upload(client, auth_headers, "private.txt", POSTGRES_NOTES.encode())
    await document_service.ingest_document(uuid.UUID(document["id"]))
    await db_session.commit()

    intruder = await auth_service.register_user(
        db_session,
        RegisterRequest(email="rag-intruder@example.com", password=TEST_PASSWORD),
    )

    citations = await retrieval_service.retrieve(
        db_session,
        owner_id=intruder.id,
        document_ids=await _all_document_ids(db_session, intruder.id),
        query="pg_stat_user_indexes B-tree partial index",
    )

    assert citations == [], "another user's chunks must never be retrievable"


async def test_only_the_best_cluster_of_matches_is_kept(
    client: AsyncClient,
    auth_headers: dict[str, str],
    registered_user: User,
    db_session: AsyncSession,
) -> None:
    """The relative cutoff must drop the long tail below the best hit.

    Embedding models have a high baseline similarity — with
    gemini-embedding-001, two unrelated passages still score ~0.58 — so an
    absolute floor alone lets irrelevant text into the prompt as "evidence".
    """
    for name, body in (("postgres.txt", POSTGRES_NOTES), ("cooking.txt", COOKING_NOTES)):
        document = await upload(client, auth_headers, name, body.encode())
        await document_service.ingest_document(uuid.UUID(document["id"]))
    await db_session.commit()

    citations = await retrieval_service.retrieve(
        db_session,
        owner_id=registered_user.id,
        document_ids=await _all_document_ids(db_session, registered_user.id),
        query="Which view reports how often an index is used?",
    )

    assert citations
    assert all(
        c.filename == "postgres.txt" for c in citations
    ), "the sourdough document must not be cited for a postgres question"

    best = citations[0].score
    floor = best * settings.retrieval_relative_cutoff
    assert all(c.score >= floor for c in citations)


async def test_weak_matches_are_filtered_out(
    ingested_document: dict, registered_user: User, db_session: AsyncSession
) -> None:
    """Irrelevant passages are worse than none: the model treats them as evidence."""
    citations = await retrieval_service.retrieve(
        db_session,
        owner_id=registered_user.id,
        document_ids=await _all_document_ids(db_session, registered_user.id),
        query="zebra migration patterns across the serengeti",
    )

    assert citations == []


# ==========================================================================
# Whole-document requests
# ==========================================================================


@pytest.mark.parametrize(
    "question",
    [
        "Summarise the key points of my document.",
        "give me a summary",
        "what is this paper about",
        "tl;dr",
        "What are the main findings?",
        "Can you outline the document?",
    ],
)
def test_overview_requests_are_recognised(question: str) -> None:
    assert retrieval_service.is_overview_request(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which view reports how often an index is used?",
        "What is a partial index?",
        "How do I configure autovacuum?",
    ],
)
def test_specific_questions_are_not_treated_as_overviews(question: str) -> None:
    """Misrouting a real question would bypass retrieval entirely."""
    assert not retrieval_service.is_overview_request(question)


async def test_overview_loads_the_document_in_order_not_by_similarity(
    ingested_document: dict, registered_user: User, db_session: AsyncSession
) -> None:
    """Regression for the bug that made "summarise" return the bibliography.

    Vector search answers "which passage resembles this query?", which is
    meaningless for a summarisation request — on a research paper it returned
    the reference list, because a bibliography is dense with topic words and
    weakly matches everything.
    """
    citations, truncated = await retrieval_service.load_document_overview(
        db_session,
        owner_id=registered_user.id,
        document_ids=await _all_document_ids(db_session, registered_user.id),
    )

    assert citations
    assert not truncated
    # Sequential, not ranked.
    assert [c.index for c in citations] == list(range(1, len(citations) + 1))


async def test_overview_is_scoped_to_the_owner(
    ingested_document: dict, db_session: AsyncSession
) -> None:
    """The overview path bypasses the vector store, so it needs its own
    tenant filter — a filter that lives in Chroma metadata does not apply."""
    intruder = await auth_service.register_user(
        db_session,
        RegisterRequest(email="overview-intruder@example.com", password=TEST_PASSWORD),
    )

    citations, _ = await retrieval_service.load_document_overview(
        db_session,
        owner_id=intruder.id,
        document_ids=await _all_document_ids(db_session, intruder.id),
    )

    assert citations == []


async def test_oversized_document_is_sampled_across_its_length(
    ingested_document: dict,
    registered_user: User,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncating from the start would summarise only the introduction."""
    monkeypatch.setattr(settings, "summary_max_chars", 200)

    citations, truncated = await retrieval_service.load_document_overview(
        db_session,
        owner_id=registered_user.id,
        document_ids=await _all_document_ids(db_session, registered_user.id),
    )

    assert truncated
    assert citations


async def test_summarise_request_does_not_refuse(
    client: AsyncClient, auth_headers: dict[str, str], ingested_document: dict
) -> None:
    """End to end: a summary request must reach the model with the document.

    Before the fix this produced an empty-handed refusal, because the grounded
    prompt's "refuse if the sources lack the answer" rule fired against a
    context full of irrelevant passages.
    """
    response = await client.post(
        f"{PREFIX}/conversations/{ingested_document['conversation_id']}/messages",
        json={"content": "Summarise the key points of my document."},
        headers=auth_headers,
    )

    events = parse_sse(response.text)
    sources = next(data for name, data in events if name == "sources")

    assert sources["citations"], "the document itself must be supplied as context"
    assert [name for name, _ in events][-1] == "done"


# ==========================================================================
# Deletion
# ==========================================================================


async def test_delete_removes_the_document_from_search_immediately(
    client: AsyncClient,
    auth_headers: dict[str, str],
    ingested_document: dict,
    db_session: AsyncSession,
) -> None:
    """Delete is a *soft* delete, and this is the half that must be instant.

    Vectors go straight away, so the document is unfindable within
    milliseconds. Everything else survives the grace period — which is the
    only reason an accidental delete is recoverable.
    """
    document_id = ingested_document["id"]

    response = await client.delete(f"{PREFIX}/documents/{document_id}", headers=auth_headers)
    assert response.status_code == 204

    assert await get_vector_store().count(where={"document_id": document_id}) == 0

    document = await db_session.get(Document, uuid.UUID(document_id))
    assert document is not None, "the row must survive for the grace period"
    assert document.lifecycle_status == "pending_deletion"
    assert document.deletion_scheduled_at is not None

    # Chunks are kept: they are what a restore would re-embed from.
    result = await db_session.execute(
        select(Chunk).where(Chunk.document_id == uuid.UUID(document_id))
    )
    assert result.scalars().all(), "chunks must survive so the delete is reversible"


async def test_delete_unlinks_conversations_without_deleting_them(
    client: AsyncClient,
    auth_headers: dict[str, str],
    ingested_document: dict,
    db_session: AsyncSession,
) -> None:
    """The history is the user's, not the document's."""
    conversation_id = uuid.UUID(ingested_document["conversation_id"])

    await client.delete(f"{PREFIX}/documents/{ingested_document['id']}", headers=auth_headers)

    conversation = await db_session.get(Conversation, conversation_id)
    assert conversation is not None
    assert conversation.document_id is None


async def test_deleted_document_disappears_from_the_list(
    client: AsyncClient, auth_headers: dict[str, str], ingested_document: dict
) -> None:
    await client.delete(f"{PREFIX}/documents/{ingested_document['id']}", headers=auth_headers)

    listed = await client.get(f"{PREFIX}/documents", headers=auth_headers)

    assert [d["id"] for d in listed.json()] == []


async def test_cannot_delete_another_users_document(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    other = await auth_service.register_user(
        db_session,
        RegisterRequest(email="doc-owner@example.com", password=TEST_PASSWORD),
    )
    document = Document(
        owner_id=other.id,
        filename="theirs.txt",
        storage_key=f"{other.id}/x.txt",
        content_type="text/plain",
        size_bytes=10,
        ingestion_status="ready",
        lifecycle_status="active",
    )
    db_session.add(document)
    await db_session.commit()

    response = await client.delete(f"{PREFIX}/documents/{document.id}", headers=auth_headers)

    assert response.status_code == 404


# ==========================================================================
# Grounded chat
# ==========================================================================


async def test_chat_emits_sources_before_tokens(
    client: AsyncClient, auth_headers: dict[str, str], ingested_document: dict
) -> None:
    """Note the phrasing of the question.

    `FakeEmbeddingProvider` is a bag-of-words hashing model: it has no notion
    of synonymy, so similarity comes purely from shared tokens. A natural
    question like "What does pg_stat_user_indexes report?" scores only 0.21
    here — the noise words dilute the vector — while a real embedding model
    would rank it highly.

    So the test uses a lexically overlapping phrasing. It still exercises the
    entire pipeline end to end; it just cannot exercise semantic matching,
    which no deterministic offline embedder can provide. Retrieval *quality*
    is measured properly by the evaluation harness, against real embeddings and a
    labelled evaluation set.
    """
    # The conversation created BY the upload — the one bound to the document.
    # A conversation created separately has no document and would correctly
    # retrieve nothing; see test_conversation_isolation.py.
    conversation_id = ingested_document["conversation_id"]

    response = await client.post(
        f"{PREFIX}/conversations/{conversation_id}/messages",
        json={"content": "Which view reports how often an index is used?"},
        headers=auth_headers,
    )

    events = parse_sse(response.text)
    names = [name for name, _ in events]

    assert names[0] == "sources", "the UI needs sources before the answer starts"

    citations = events[0][1]["citations"]
    assert citations
    assert citations[0]["filename"] == "postgres.txt"
    assert citations[0]["index"] == 1


async def test_chat_without_documents_sends_empty_sources(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """A user with an empty knowledge base gets a plain assistant, not a refusal."""
    conversation = await client.post(f"{PREFIX}/conversations", json={}, headers=auth_headers)

    response = await client.post(
        f"{PREFIX}/conversations/{conversation.json()['id']}/messages",
        json={"content": "hello there"},
        headers=auth_headers,
    )

    events = parse_sse(response.text)
    sources = next(data for name, data in events if name == "sources")

    assert sources["citations"] == []
    assert [name for name, _ in events][-1] == "done"
