"""Cross-encoder reranking.

The model itself is stubbed throughout. Loading a real cross-encoder costs
seconds and hundreds of megabytes, and asserting that a neural network ranks
one sentence above another turns a unit test into a slow, flaky check on a
third party's weights. These tests own the *contract* around the model:
ordering, non-truncation, determinism, thread offloading and fallback.

Whether the model actually improves relevance is a different question, and it
is answered by `python -m app.eval.run --config rerank` against a labelled
corpus — not by an assertion here.
"""

from __future__ import annotations

import asyncio
import threading
import uuid

import pytest

from app.core.config import settings
from app.retrieval.fusion import FusedChunk
from app.retrieval.rerank import (
    CrossEncoderReranker,
    NoopReranker,
    Reranker,
    apply_reranking,
    get_reranker,
    rerank_failures,
    rerank_latencies,
    reset_rerank_stats,
)


def _chunk(text: str, *, seed: int, fused_score: float = 0.5) -> FusedChunk:
    return FusedChunk(
        chunk_id=uuid.UUID(int=seed),
        document_id=uuid.UUID(int=999),
        filename="doc.md",
        page_number=1,
        text=text,
        score=fused_score,
        ranks={"dense": seed},
        scores={"dense": 0.6},
    )


class _StubModel:
    """Stands in for `sentence_transformers.CrossEncoder`."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []
        self.thread_names: list[str] = []

    def predict(self, pairs, batch_size=None):  # noqa: ANN001, ARG002
        self.calls.append(list(pairs))
        self.thread_names.append(threading.current_thread().name)
        return self._scores[: len(pairs)]


class _StubReranker(CrossEncoderReranker):
    """A real CrossEncoderReranker with the model swapped out."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__(model_name="stub")
        self.stub = _StubModel(scores)

    def _load(self):
        return self.stub


class _ExplodingReranker(Reranker):
    name = "exploding"

    async def rerank(self, query, candidates):  # noqa: ANN001
        raise RuntimeError("model file is corrupt")


@pytest.fixture(autouse=True)
def _clean_stats():
    reset_rerank_stats()
    yield
    reset_rerank_stats()


# ---------------------------------------------------------------------------
# Ordering contract
# ---------------------------------------------------------------------------


async def test_reranker_reorders_by_model_score_not_by_fused_score() -> None:
    """The entire point: a candidate retrieval ranked last can be promoted to
    first. If the fused order survived, the reranker would be a no-op."""
    candidates = [
        _chunk("first from retrieval", seed=1, fused_score=0.9),
        _chunk("second from retrieval", seed=2, fused_score=0.8),
        _chunk("third from retrieval", seed=3, fused_score=0.7),
    ]
    reranker = _StubReranker([0.1, 0.2, 0.9])

    result = await reranker.rerank("q", candidates)

    assert [chunk.text for chunk in result] == [
        "third from retrieval",
        "second from retrieval",
        "first from retrieval",
    ]


async def test_reranker_never_drops_candidates() -> None:
    """Selecting the top k is the pipeline's decision. A reranker that
    truncated would make retrieval_top_k mean something different depending on
    which reranker happened to be configured."""
    candidates = [_chunk(f"c{index}", seed=index) for index in range(1, 8)]
    reranker = _StubReranker([0.9, 0.1, 0.5, 0.4, 0.3, 0.2, 0.05])

    result = await reranker.rerank("q", candidates)

    assert len(result) == len(candidates)
    assert {chunk.chunk_id for chunk in result} == {chunk.chunk_id for chunk in candidates}


async def test_reranker_records_its_score_without_destroying_the_others() -> None:
    """The gate downstream was calibrated on native dense and lexical scores.
    Overwriting them with a reranker score would silently decalibrate it."""
    reranker = _StubReranker([0.77, 0.1])

    result = await reranker.rerank("q", [_chunk("a", seed=1), _chunk("b", seed=2)])

    assert result[0].scores["reranker"] == pytest.approx(0.77)
    assert result[0].scores["dense"] == pytest.approx(0.6)
    assert result[0].ranks == {"dense": 1}


async def test_ties_fall_back_to_the_fused_order() -> None:
    """Equal model scores must not produce an arbitrary order, or evaluation
    runs would not be reproducible."""
    candidates = [
        _chunk("lower fused", seed=1, fused_score=0.30),
        _chunk("higher fused", seed=2, fused_score=0.90),
    ]
    reranker = _StubReranker([0.5, 0.5])

    result = await reranker.rerank("q", candidates)

    assert [chunk.text for chunk in result] == ["higher fused", "lower fused"]


async def test_pairs_the_query_with_every_candidate() -> None:
    """A cross-encoder scores (query, passage) jointly — that is what
    distinguishes it from the bi-encoder. Passing passages alone would be a
    silently different model."""
    reranker = _StubReranker([0.4, 0.6])

    await reranker.rerank("how do I roll back", [_chunk("a", seed=1), _chunk("b", seed=2)])

    assert reranker.stub.calls == [[("how do I roll back", "a"), ("how do I roll back", "b")]]


async def test_single_candidate_skips_the_model_entirely() -> None:
    """Nothing to reorder, and no reason to pay for a model load."""
    reranker = _StubReranker([0.5])

    result = await reranker.rerank("q", [_chunk("only", seed=1)])

    assert len(result) == 1
    assert reranker.stub.calls == []


async def test_inference_runs_off_the_event_loop() -> None:
    """Transformer inference is CPU-bound C++ that does not yield. Running it
    inline would stall every other request on the worker — the same failure
    this project measured with bcrypt, where /health/live went from 2ms to
    1459ms under five concurrent registrations."""
    reranker = _StubReranker([0.5, 0.4])

    await reranker.rerank("q", [_chunk("a", seed=1), _chunk("b", seed=2)])

    assert reranker.stub.thread_names, "model was never called"
    assert reranker.stub.thread_names[0] != threading.current_thread().name


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


async def test_failure_falls_back_to_the_fused_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reranker is an optimisation, not a dependency. A missing model file
    must degrade ordering, not fail the user's search."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: _ExplodingReranker())

    candidates = [_chunk("a", seed=1), _chunk("b", seed=2), _chunk("c", seed=3)]
    result = await apply_reranking("q", candidates)

    assert [chunk.text for chunk in result] == ["a", "b", "c"]


async def test_failure_is_recorded_so_a_benchmark_can_refuse_to_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silent fallback is right for a request and a lie in a measurement — the
    same lesson as the rate-limited retriever."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: _ExplodingReranker())

    await apply_reranking("q", [_chunk("a", seed=1), _chunk("b", seed=2)])

    assert rerank_failures() == {"exploding": 1}


async def test_latency_is_recorded_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: _StubReranker([0.9, 0.1]))

    await apply_reranking("q", [_chunk("a", seed=1), _chunk("b", seed=2)])

    assert len(rerank_latencies()) == 1
    assert rerank_latencies()[0] >= 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


async def test_disabled_reranking_yields_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rerank_enabled", False)
    get_reranker.cache_clear()

    assert isinstance(get_reranker(), NoopReranker)


async def test_noop_preserves_order_exactly() -> None:
    candidates = [_chunk("a", seed=1), _chunk("b", seed=2), _chunk("c", seed=3)]

    assert await NoopReranker().rerank("q", candidates) == candidates


async def test_disabled_reranking_costs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With reranking off, apply_reranking must not touch a model or record
    latency — otherwise the A/B baseline is not actually a baseline."""
    monkeypatch.setattr(settings, "rerank_enabled", False)
    get_reranker.cache_clear()

    candidates = [_chunk("a", seed=1), _chunk("b", seed=2)]
    result = await apply_reranking("q", candidates)

    assert result == candidates
    assert rerank_latencies() == []


def test_reranker_factory_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cross-encoder holds hundreds of megabytes of weights; constructing one
    per request would be catastrophic."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    get_reranker.cache_clear()

    assert get_reranker() is get_reranker()

    get_reranker.cache_clear()


async def test_concurrent_reranks_share_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ten simultaneous searches must not load ten models."""
    shared = _StubReranker([0.9, 0.1])
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: shared)

    candidates = [_chunk("a", seed=1), _chunk("b", seed=2)]
    await asyncio.gather(*(apply_reranking("q", candidates) for _ in range(10)))

    assert len(rerank_latencies()) == 10
    assert rerank_failures() == {}
