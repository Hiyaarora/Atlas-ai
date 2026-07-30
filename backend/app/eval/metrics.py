"""Retrieval metrics.

Four numbers, because each answers a different question and any one alone is
misleading:

    recall@k       Did we find the evidence at all?
    precision@k    How much of what we returned was evidence?
    MRR            How near the top was the first correct passage?
    nDCG@k         Rank-weighted credit for finding several relevant passages.

Recall is the headline for RAG — a passage the model never sees cannot be
cited — but optimising it alone is trivially gamed by returning everything.
Precision is the counterweight. MRR matters because context windows are
ordered and models weight early content more heavily. nDCG is the only one of
the four that rewards getting *two* relevant passages at ranks 1 and 2 over
getting them at ranks 1 and 6.

All functions are pure and take ids, not chunks, so they are testable without
a database, an embedding provider or a corpus.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass


def recall_at_k(retrieved: Sequence[uuid.UUID], relevant: set[uuid.UUID], k: int) -> float:
    """Fraction of the relevant passages that appear in the top k.

    Undefined with no relevant passages; returns 0.0 so a mislabelled query
    cannot silently inflate an average.
    """
    if not relevant:
        return 0.0
    found = len(set(retrieved[:k]) & relevant)
    return found / len(relevant)


def precision_at_k(retrieved: Sequence[uuid.UUID], relevant: set[uuid.UUID], k: int) -> float:
    """Fraction of the top k that is relevant.

    Divided by the number actually returned, not by k. A system returning two
    results, both correct, has precision 1.0 — penalising it for not filling
    six slots would reward padding, which is the behaviour we are trying to
    eliminate.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def reciprocal_rank(retrieved: Sequence[uuid.UUID], relevant: set[uuid.UUID]) -> float:
    """1 / rank of the first relevant result; 0 if none."""
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[uuid.UUID], relevant: set[uuid.UUID], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance.

    Each hit contributes 1/log2(rank + 1), so rank 1 is worth 1.0 and rank 6
    about 0.36. Dividing by the ideal ordering — every relevant passage packed
    into the top slots — bounds the result in [0, 1] and makes queries with
    different numbers of relevant passages comparable.
    """
    if not relevant:
        return 0.0

    gain = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))

    return gain / ideal if ideal else 0.0


@dataclass(frozen=True)
class QueryResult:
    """Scored outcome for one query."""

    query_id: str
    kind: str
    expect_empty: bool
    returned: int
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float

    #: For expect_empty queries only: did the system correctly return nothing?
    refused: bool | None = None


def score_query(
    *,
    query_id: str,
    kind: str,
    expect_empty: bool,
    retrieved: Sequence[uuid.UUID],
    relevant: set[uuid.UUID],
    k: int,
) -> QueryResult:
    """Score one query's retrieved list against its labels."""
    if expect_empty:
        # Ranking metrics are meaningless with no relevant set. The only
        # question that matters is whether the system stayed quiet.
        return QueryResult(
            query_id=query_id,
            kind=kind,
            expect_empty=True,
            returned=len(retrieved),
            recall=0.0,
            precision=0.0,
            reciprocal_rank=0.0,
            ndcg=0.0,
            refused=len(retrieved) == 0,
        )

    return QueryResult(
        query_id=query_id,
        kind=kind,
        expect_empty=False,
        returned=len(retrieved),
        recall=recall_at_k(retrieved, relevant, k),
        precision=precision_at_k(retrieved, relevant, k),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        ndcg=ndcg_at_k(retrieved, relevant, k),
    )


@dataclass(frozen=True)
class Summary:
    """Aggregate over a set of query results."""

    answerable: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    negatives: int
    refusal_rate: float

    #: Mean results returned for questions the corpus cannot answer. The
    #: number that exposes a retriever padding its output with noise.
    noise_per_negative: float


def summarise(results: Sequence[QueryResult]) -> Summary:
    """Aggregate, keeping answerable and unanswerable queries separate.

    Averaging them together would be meaningless: recall is undefined for a
    query with no correct answer, and a system that returns nothing for
    everything would score 0.0 recall and 100% refusal, which must be visible
    as the tradeoff it is rather than blended into one number.
    """
    answerable = [result for result in results if not result.expect_empty]
    negatives = [result for result in results if result.expect_empty]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return Summary(
        answerable=len(answerable),
        recall=mean([result.recall for result in answerable]),
        precision=mean([result.precision for result in answerable]),
        mrr=mean([result.reciprocal_rank for result in answerable]),
        ndcg=mean([result.ndcg for result in answerable]),
        negatives=len(negatives),
        refusal_rate=mean([1.0 if result.refused else 0.0 for result in negatives]),
        noise_per_negative=mean([float(result.returned) for result in negatives]),
    )
