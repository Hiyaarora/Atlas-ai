"""Hybrid retrieval: run the retrievers, fuse, then gate on evidence.

THE GATE IS THE INTERESTING PART
--------------------------------
The dense-only path gates on cosine similarity with an absolute floor plus a
relative cutoff. Neither transfers to a fused ranking: an RRF score is around 1/61 for
the top hit, so the 0.35 floor would reject every result, and a relative
cutoff on RRF scores measures nothing but rank spacing.

So relevance is decided on the retrievers' *native* scores, which fusion
preserved, with a different rule for each — matched to what that score
actually is:

    dense    absolute floor       cosine is bounded and roughly calibrated,
                                  so "0.35" means something across queries.

    lexical  relative to the best BM25 is unbounded and corpus-dependent;
             hit for THIS query   a score of 4.0 can be excellent for one
                                  query and mediocre for another, so only
                                  its ratio to the best hit is meaningful.

A chunk survives if EITHER retriever vouches for it. That is the whole point
of hybrid search: an exact-identifier match that dense scored 0.31 must still
reach the model, and a paraphrase match sharing no query terms must too.

Ordering then comes from RRF, which is what RRF is good at.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from statistics import median

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.base import RetrievedChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import FusedChunk, reciprocal_rank_fusion
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.rerank import apply_reranking

logger = get_logger(__name__)


#: Count of retriever failures since the last reset, by retriever name.
#:
#: Deliberately a plain module-level dict rather than a metrics library: its
#: only consumer is the evaluation harness, which needs to know whether the
#: numbers it just produced came from a healthy pipeline. Wiring it to
#: Prometheus is a later concern.
_degradations: dict[str, int] = {}


def record_degradation(retriever: str) -> None:
    _degradations[retriever] = _degradations.get(retriever, 0) + 1


def degradations() -> dict[str, int]:
    """Failures observed so far. Empty means every retriever ran."""
    return dict(_degradations)


def reset_degradations() -> None:
    _degradations.clear()


def _is_answerable(fused: Sequence[FusedChunk]) -> bool:
    """Does this query have an answer here at all?

    A QUERY-level decision, deliberately separate from the per-chunk gate
    below. The two ask different questions, and conflating them is why the
    system used to return six passages about database replication when asked
    about refund policy: every chunk individually cleared a low bar, and
    nothing ever asked whether the best of them was actually any good.

    Two conditions, both required, because measurement showed neither works
    alone — the score ranges for answerable and unanswerable queries overlap
    on each signal taken by itself:

      absolute   the best chunk must clear a floor
      relative   the best chunk must stand clear of the median chunk for the
                 same query, which removes the query-dependent baseline that
                 makes absolute scores incomparable across questions

    Uses dense scores only. BM25 has no equivalent notion of "nothing here
    matches" — it simply returns fewer rows, and its own coverage floor
    already handles that case in `lexical.py`.
    """
    dense_scores = [score for chunk in fused if (score := chunk.scores.get("dense")) is not None]
    if not dense_scores:
        # Lexical-only results. Exact-identifier matches are strong evidence on
        # their own and already passed the idf-coverage floor, so there is
        # nothing further to check.
        return True

    best = max(dense_scores)
    if best < settings.retrieval_min_score:
        return False

    # Too few candidates for a median to carry information — accept rather
    # than invent a verdict from noise.
    if len(dense_scores) < settings.retrieval_margin_min_candidates:
        return True

    return (best - median(dense_scores)) >= settings.retrieval_min_margin


def _passes_gate(chunk: FusedChunk, *, best_lexical: float) -> bool:
    """Does any retriever vouch for this chunk?"""
    dense_score = chunk.scores.get("dense")
    if dense_score is not None and dense_score >= settings.retrieval_min_score:
        return True

    lexical_score = chunk.scores.get("lexical")
    if lexical_score is not None and best_lexical > 0.0:
        return lexical_score >= best_lexical * settings.retrieval_lexical_relative_cutoff

    return False


async def hybrid_search(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    query: str,
    top_k: int,
    candidate_k: int | None = None,
) -> list[FusedChunk]:
    """Retrieve with dense + lexical, fuse, gate, and return the top `top_k`.

    `candidate_k` is how deep EACH retriever goes before fusion, and it is
    deliberately larger than `top_k`. Fusion can only reorder what it is given:
    a chunk ranked 8th by BM25 and 9th by dense is a strong consensus
    candidate, but with candidate_k == top_k == 6 neither list would have
    contained it and the consensus could never be observed.
    """
    if not document_ids:
        logger.warning("hybrid_search_without_scope", extra={"owner_id": str(owner_id)})
        return []

    depth = candidate_k or settings.retrieval_candidate_k

    dense = DenseRetriever()
    lexical = LexicalRetriever(session)

    # Concurrently: one call is network-bound (embedding the query), the other
    # database-bound. Running them in sequence would make every search cost the
    # sum rather than the max of two independent latencies.
    #
    # return_exceptions=True because a retriever failing is a degradation, not
    # an outage — if the embedding API is rate-limited, lexical results alone
    # are far better than an error. This is what "graceful degradation" means
    # concretely, and it matters on a 20-requests-per-day free tier.
    results = await asyncio.gather(
        dense.retrieve(query, owner_id=owner_id, document_ids=document_ids, top_k=depth),
        lexical.retrieve(query, owner_id=owner_id, document_ids=document_ids, top_k=depth),
        return_exceptions=True,
    )

    rankings: list[list[RetrievedChunk]] = []
    for retriever, result in zip((dense, lexical), results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "retriever_failed",
                extra={"retriever": retriever.name, "error": str(result)},
            )
            # Degradation is correct for a user request — lexical-only results
            # beat an error page. It is NOT correct for a measurement, which
            # would report the degraded quality as if it were the design's.
            # This counter is how the evaluation harness tells the difference.
            record_degradation(retriever.name)
            continue
        rankings.append(result)

    if not rankings:
        logger.error("all_retrievers_failed", extra={"owner_id": str(owner_id)})
        return []

    fused = reciprocal_rank_fusion(rankings)

    # Ask "is there an answer here?" before "which chunks are best?". Returning
    # the least-irrelevant passages of an irrelevant document is worse than
    # returning nothing: the model treats whatever it is handed as evidence,
    # and only the system prompt then stands between that and a fabricated
    # answer. Refusing here removes the temptation rather than resisting it.
    if not _is_answerable(fused):
        logger.info(
            "retrieval_refused",
            extra={
                "candidates": len(fused),
                "best_dense": round(
                    max((chunk.scores.get("dense", 0.0) for chunk in fused), default=0.0), 3
                ),
            },
        )
        return []

    # Computed across the whole fused set, before trimming — the best lexical
    # hit is the reference point for the relative cutoff, so it must not depend
    # on where we happen to cut.
    best_lexical = max((chunk.scores.get("lexical", 0.0) for chunk in fused), default=0.0)
    kept = [chunk for chunk in fused if _passes_gate(chunk, best_lexical=best_lexical)]

    # Rerank the survivors, then cut. The order of these three steps is the
    # whole design:
    #
    #   gate   -> decides what is admissible evidence  (calibrated on the
    #             retrievers' native scores, so it must run before anything
    #             overwrites the ordering they produced)
    #   rerank -> decides what is BEST among the admissible
    #   cut    -> takes the top k
    #
    # Reranking before the gate would waste forward passes on chunks about to
    # be discarded. Cutting before reranking would defeat the point entirely:
    # the reranker earns its cost by promoting a candidate retrieval ranked
    # 12th into the final six, which it cannot do if only six reach it.
    candidates = kept[: settings.rerank_candidate_k]
    reranked = await apply_reranking(query, candidates)

    logger.info(
        "hybrid_search_completed",
        extra={
            "retrievers": [ranking[0].retriever for ranking in rankings if ranking],
            "candidates": len(fused),
            "kept": len(kept),
            "reranked": len(reranked),
            "returned": min(len(reranked), top_k),
            "agreed": sum(1 for chunk in kept if len(chunk.ranks) > 1),
            "best_lexical": round(best_lexical, 3) if best_lexical else None,
        },
    )

    return reranked[:top_k]
