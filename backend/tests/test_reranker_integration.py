"""Integration: the reranker inside the real retrieval pipeline.

The unit tests in `test_reranker.py` verify the reranker in isolation. These
verify it is *wired correctly* — that `hybrid_search` calls it at the right
point, with the right candidates, and that its ordering is what reaches the
caller. A reranker that works perfectly and is invoked after the final cut
would pass every unit test and do nothing.

The model is stubbed here too. Whether a cross-encoder ranks better than RRF
is measured by `python -m app.eval.run --config rerank` against a labelled
corpus; a test asserting it would be asserting a third party's weights.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Chunk, Document
from app.retrieval.fusion import FusedChunk
from app.retrieval.pipeline import hybrid_search
from app.retrieval.rerank import (
    NoopReranker,
    Reranker,
    get_reranker,
    rerank_latencies,
    reset_rerank_stats,
)
from app.services import document_service

PREFIX = "/api/v1"

# Five clearly distinct sections, each padded past the 1000-character chunk
# size so the document yields several chunks. Without that the pipeline has a
# single chunk and there is nothing for a reranker to reorder.
NOTES = "\n\n".join(
    f"## Section {index}\n\n{topic}\n\n" + (f"Supporting detail for {topic.lower()} " * 40)
    for index, topic in enumerate(
        [
            "Replication slots pin the write-ahead log until a consumer advances",
            "Rolling back a release means redeploying the previous image tag",
            "Latency budgets are stated at the ninety-ninth percentile",
            "Escalation pages the secondary after fifteen minutes",
            "Capacity is reviewed against the previous period's peak",
        ],
        start=1,
    )
)


class _ReverseReranker(Reranker):
    """Scores candidates so the fused order is exactly inverted.

    A deliberately unnatural ordering: if the final result matches it, the
    reranker's output is definitively what reached the caller rather than a
    coincidence of the retrieval order.
    """

    name = "reverse"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def rerank(self, query: str, candidates):  # noqa: ANN001
        self.seen = [str(chunk.chunk_id) for chunk in candidates]
        return [
            FusedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
                score=chunk.score,
                ranks=chunk.ranks,
                scores={**chunk.scores, "reranker": float(index)},
            )
            for index, chunk in enumerate(reversed(list(candidates)))
        ]


@pytest.fixture(autouse=True)
def _wiring_not_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the relevance gate fully for this module.

    The fake embedding provider scores even a good match around 0.2, well under
    the floor calibrated for gemini-embedding-001 — so the answerability gate
    refuses every query and there is nothing to rerank. That refusal is correct
    behaviour under a mismatched threshold, and it is not what these tests are
    about: they check that the reranker is invoked at the right point with the
    right candidates. Threshold quality is the harness's job.
    """
    monkeypatch.setattr(settings, "retrieval_min_score", 0.0)
    monkeypatch.setattr(settings, "retrieval_min_margin", 0.0)


@pytest.fixture
async def indexed_document(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> Document:
    response = await client.post(
        f"{PREFIX}/documents",
        files={"file": ("notes.md", NOTES.encode(), "text/markdown")},
        headers=auth_headers,
    )
    document_id = uuid.UUID(response.json()["document"]["id"])
    await document_service.ingest_document(document_id)
    await db_session.commit()
    return (
        await db_session.execute(select(Document).where(Document.id == document_id))
    ).scalar_one()


async def test_pipeline_returns_the_reranker_order(
    indexed_document: Document, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final ordering must come from the reranker, not from RRF."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    reranker = _ReverseReranker()
    monkeypatch.setattr("app.retrieval.pipeline.get_reranker", lambda: reranker, raising=False)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: reranker)

    without = await hybrid_search(
        db_session,
        owner_id=indexed_document.owner_id,
        document_ids=[indexed_document.id],
        query="how do we undo a bad release",
        top_k=3,
    )

    assert without, "precondition: retrieval must return something to reorder"
    assert reranker.seen, "the pipeline never called the reranker"
    # The reranker inverted the fused order, so the first result must be the
    # candidate the retrievers ranked last.
    assert str(without[0].chunk_id) == reranker.seen[-1]


async def test_reranker_sees_more_candidates_than_are_returned(
    indexed_document: Document, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reranker earns its cost by promoting a candidate retrieval ranked
    below the cut. It can only do that if more than `top_k` reach it — cutting
    first would make reranking a permutation of an already-final list."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    reranker = _ReverseReranker()
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: reranker)

    results = await hybrid_search(
        db_session,
        owner_id=indexed_document.owner_id,
        document_ids=[indexed_document.id],
        query="replication slot retention",
        top_k=2,
    )

    assert len(results) <= 2
    assert len(reranker.seen) > len(results), (
        f"reranker saw {len(reranker.seen)} candidates but {len(results)} were returned — "
        "the cut must happen after reranking, not before"
    )


async def test_reranking_is_skipped_entirely_when_disabled(
    indexed_document: Document, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With reranking off the A/B baseline must be genuinely untouched.

    Deliberately does NOT patch `get_reranker`: the factory returning a
    `NoopReranker` *is* the mechanism under test, and stubbing it would route
    around the decision this asserts.
    """
    monkeypatch.setattr(settings, "rerank_enabled", False)
    get_reranker.cache_clear()
    reset_rerank_stats()

    assert isinstance(get_reranker(), NoopReranker)

    results = await hybrid_search(
        db_session,
        owner_id=indexed_document.owner_id,
        document_ids=[indexed_document.id],
        query="latency budget percentile",
        top_k=3,
    )

    assert results, "precondition: retrieval should return something"
    # No model was consulted, so no inference latency was recorded. That is
    # what makes the disabled configuration a genuine A/B baseline rather than
    # a reranked run with the reranking silently applied.
    assert rerank_latencies() == []


async def test_a_failing_reranker_still_returns_results(
    indexed_document: Document, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or broken model must degrade ordering, never fail the search."""

    class _Broken(Reranker):
        name = "broken"

        async def rerank(self, query, candidates):  # noqa: ANN001
            raise RuntimeError("no such model")

    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: _Broken())

    results = await hybrid_search(
        db_session,
        owner_id=indexed_document.owner_id,
        document_ids=[indexed_document.id],
        query="how do we undo a bad release",
        top_k=3,
    )

    assert results, "the search must survive a reranker failure"


async def test_refused_queries_never_reach_the_reranker(
    indexed_document: Document, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward passes are the expensive part. Scoring candidates that the
    answerability gate already rejected would pay for nothing."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    # A floor nothing can clear forces the query-level refusal.
    monkeypatch.setattr(settings, "retrieval_min_score", 0.999)
    reranker = _ReverseReranker()
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: reranker)

    results = await hybrid_search(
        db_session,
        owner_id=indexed_document.owner_id,
        document_ids=[indexed_document.id],
        query="what is the refund policy for annual subscriptions",
        top_k=3,
    )

    assert results == []
    assert reranker.seen == []


async def test_retrieval_service_interface_is_unchanged(
    indexed_document: Document, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reranking must not alter the public contract: same call, same Citation
    shape, whether or not a reranker is configured."""
    from app.services import retrieval_service

    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: _ReverseReranker())

    citations = await retrieval_service.retrieve(
        db_session,
        owner_id=indexed_document.owner_id,
        document_ids=[indexed_document.id],
        query="how do we undo a bad release",
    )

    assert citations
    first = citations[0]
    assert first.index == 1
    assert isinstance(first.chunk_id, uuid.UUID)
    assert first.document_id == indexed_document.id
    assert first.filename == "notes.md"
    assert first.page_number >= 1
    assert first.excerpt


async def test_chunks_exist_for_the_fixture(
    indexed_document: Document, db_session: AsyncSession
) -> None:
    """Guards the fixture itself: a single-chunk document would make every
    reordering assertion above vacuously true."""
    chunks = (
        (await db_session.execute(select(Chunk).where(Chunk.document_id == indexed_document.id)))
        .scalars()
        .all()
    )

    assert len(chunks) >= 4
