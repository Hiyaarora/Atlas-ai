"""Cross-encoder reranking.

BI-ENCODER VS CROSS-ENCODER
---------------------------
Both answer "how relevant is this passage to this query?" and they differ in
*when* the two texts meet.

A bi-encoder — what `dense.py` uses — pushes the query and the passage through
the model separately and compares the two output vectors:

    vector(query)  ·  vector(passage)

The passage vector does not depend on the query, so it can be computed once at
ingestion and stored in an index. That is what makes similarity search over
millions of chunks possible: at query time you embed one string and do
approximate nearest-neighbour lookup. The price is that the model never sees
the two texts together. It must compress everything a passage could ever be
relevant to into one fixed vector, before knowing what will be asked.

A cross-encoder concatenates them and runs the pair through the model as a
single input:

    model("query [SEP] passage")  ->  relevance score

Now every layer of attention can relate query tokens to passage tokens
directly. The model can notice that the passage answers this exact question
rather than merely sharing a topic — and, critically for us, that a passage is
*about* the right subject while answering a different question.

The cost is total: nothing can be precomputed. The score depends on the pair,
so N passages means N forward passes at query time, and the index is useless
because you cannot sort by a score you have not computed yet.

WHY RERANK AFTER RETRIEVAL RATHER THAN INSTEAD OF IT
----------------------------------------------------
Because a cross-encoder cannot be an index.

To use one as the retriever you would score the query against every chunk the
user owns on every request. At this corpus's ~26 chunks that is ~30ms and
merely wasteful; at 100k chunks it is minutes per query, and no amount of
engineering fixes it — the work is inherent in scoring pairs.

So the two are composed rather than compared. Retrieval is cheap, indexable
and approximate: it reduces the corpus to a shortlist. Reranking is expensive,
exact and unindexable: it orders the shortlist properly. This funnel —
recall-oriented and broad, then precision-oriented and narrow — is the
standard shape of production search for exactly this reason.

The consequence worth stating: **reranking cannot fix a recall failure.** A
passage that retrieval never surfaced is a passage the reranker never sees.
That is why recall is measured first.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.fusion import FusedChunk

logger = get_logger(__name__)


class Reranker(ABC):
    """Reorders candidate passages by relevance to the query.

    Deliberately takes and returns `FusedChunk`, so a reranker is a pure
    reordering step that can be inserted or removed without any other part of
    the pipeline changing shape.
    """

    name: str

    @abstractmethod
    async def rerank(self, query: str, candidates: Sequence[FusedChunk]) -> list[FusedChunk]:
        """Return the candidates ordered best-first.

        Implementations MUST NOT drop candidates. Selecting the top k is the
        pipeline's decision, and a reranker that silently truncated would make
        `retrieval_top_k` mean something different depending on which reranker
        happened to be configured.
        """


class NoopReranker(Reranker):
    """Leaves the fused order untouched.

    Used when reranking is disabled, and as the fallback when the model cannot
    be loaded. Having an object rather than a `None` check means the pipeline
    has exactly one code path.
    """

    name = "noop"

    async def rerank(self, query: str, candidates: Sequence[FusedChunk]) -> list[FusedChunk]:
        return list(candidates)


class CrossEncoderReranker(Reranker):
    """A local sentence-transformers cross-encoder.

    Local rather than an API call, and that was a measured decision rather than
    a preference: an LLM-based reranker costs one request per query on top of
    generation, which on a 20-requests-per-day free tier would make the
    evaluation harness — 53 queries per run — impossible to operate. A local
    model runs offline, costs nothing per call, and makes sweeps cheap.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.rerank_model
        self._model = None

    def _load(self):
        """Load the model once, on first use.

        Not in `__init__`: constructing the reranker happens during app import,
        and a multi-second model load there would delay startup and run even
        for a process that never serves a search. Lazy loading moves that cost
        to the first query and skips it entirely in tests.
        """
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        started = time.perf_counter()
        self._model = CrossEncoder(self.model_name, max_length=settings.rerank_max_length)
        logger.info(
            "reranker_loaded",
            extra={
                "model": self.model_name,
                "load_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return self._model

    def _score(self, query: str, texts: list[str]) -> list[float]:
        """Synchronous inference. Called on a worker thread, never inline."""
        model = self._load()
        pairs = [(query, text) for text in texts]
        return [
            float(score) for score in model.predict(pairs, batch_size=settings.rerank_batch_size)
        ]

    async def rerank(self, query: str, candidates: Sequence[FusedChunk]) -> list[FusedChunk]:
        if len(candidates) < 2:
            # Nothing to reorder, and no reason to pay for a model load.
            return list(candidates)

        texts = [chunk.text for chunk in candidates]

        # `asyncio.to_thread` is not optional. Transformer inference is
        # CPU-bound C++ that does not yield, so calling it inline would block
        # the event loop for its whole duration — every other request on the
        # worker, including health checks and unrelated streams, would stall.
        # This project has measured that exact failure before, with bcrypt:
        # /health/live went from 2ms to 1459ms under five concurrent
        # registrations until the hashing moved onto a thread.
        scores = await asyncio.to_thread(self._score, query, texts)

        ordered = sorted(
            zip(candidates, scores, strict=True),
            # Tie-break on the fused score so equal reranker scores fall back
            # to the retrieval order rather than an arbitrary one. Without it,
            # evaluation runs would not be reproducible.
            key=lambda pair: (-pair[1], -pair[0].score),
        )

        return [
            FusedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
                score=chunk.score,
                ranks=chunk.ranks,
                # Recorded alongside the retrievers' own scores rather than
                # overwriting `score`. The RRF value stays meaningful for
                # debugging, and the gate downstream still needs the native
                # dense and lexical scores it was calibrated against.
                scores={**chunk.scores, "reranker": score},
            )
            for chunk, score in ordered
        ]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """The process-wide reranker.

    Cached because a cross-encoder holds hundreds of megabytes of weights;
    constructing one per request would be catastrophic. `lru_cache` also gives
    tests a `cache_clear()` to reset the choice, matching how the LLM and
    embedding factories already work.
    """
    if not settings.rerank_enabled:
        return NoopReranker()

    return CrossEncoderReranker()


async def apply_reranking(query: str, candidates: Sequence[FusedChunk]) -> list[FusedChunk]:
    """Rerank, falling back to the fused order on any failure.

    A reranker is an optimisation, not a dependency. If the model file is
    missing, the machine is out of memory, or the package is not installed,
    the correct behaviour is slightly worse ordering — not a failed search.
    The user asked a question; returning results in RRF order answers it.

    Catches broadly on purpose: the failure modes here come from a large
    third-party stack (torch, transformers, a HuggingFace download) and
    enumerating them would be a guess that goes stale.
    """
    reranker = get_reranker()
    if isinstance(reranker, NoopReranker):
        return list(candidates)

    started = time.perf_counter()
    try:
        reranked = await reranker.rerank(query, candidates)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning(
            "reranker_failed",
            extra={"reranker": reranker.name, "error": f"{type(exc).__name__}: {exc}"},
        )
        record_rerank_failure(reranker.name)
        return list(candidates)

    elapsed_ms = (time.perf_counter() - started) * 1000
    record_rerank_latency(elapsed_ms)
    logger.info(
        "reranked",
        extra={
            "reranker": reranker.name,
            "candidates": len(candidates),
            "latency_ms": round(elapsed_ms, 1),
        },
    )
    return reranked


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
#
# Plain module-level state, for the same reason as the degradation counter in
# `pipeline.py`: the only consumer is the evaluation harness, which has to
# report the latency reranking adds and to notice a run where the model
# silently failed and fell back. Real metrics plumbing comes later.

_latencies_ms: list[float] = []
_failures: dict[str, int] = {}


def record_rerank_latency(milliseconds: float) -> None:
    _latencies_ms.append(milliseconds)


def record_rerank_failure(name: str) -> None:
    _failures[name] = _failures.get(name, 0) + 1


def rerank_latencies() -> list[float]:
    return list(_latencies_ms)


def rerank_failures() -> dict[str, int]:
    return dict(_failures)


def reset_rerank_stats() -> None:
    _latencies_ms.clear()
    _failures.clear()
