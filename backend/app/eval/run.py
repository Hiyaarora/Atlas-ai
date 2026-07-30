"""Retrieval evaluation harness.

    python -m app.eval.run                  # compare dense-only vs hybrid
    python -m app.eval.run --config hybrid  # one configuration only
    python -m app.eval.run --per-query      # every query, not just aggregates
    python -m app.eval.run --keep           # leave the scratch corpus behind

WHY IT INGESTS ITS OWN CORPUS
-----------------------------
The harness creates a scratch account, ingests the fixture documents through
the real pipeline, runs the queries, and deletes everything. It does not reuse
whatever happens to be in the developer's database.

That costs a little embedding quota per run and buys reproducibility: the same
command gives the same numbers on any machine. It is also the only way to
evaluate a change to *chunking*, since chunk boundaries are decided during
ingestion — a harness that reused an existing index could never measure them.

WHAT IT DOES NOT DO
-------------------
It measures retrieval, not answers. Whether the model wrote a good reply from
the passages is a separate question needing a different method (an LLM judge,
which costs a request per query and is the reason it is not here). Retrieval
quality is the upstream bound: the model cannot cite what it never received.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, select

from app.core.config import settings
from app.core.security import hash_password
from app.db.session import AsyncSessionLocal
from app.embeddings import get_embedding_provider
from app.eval.dataset import QUERIES, EvalQuery, load_corpus, normalize, validate_dataset
from app.eval.embedding_cache import CachedEmbeddingProvider
from app.eval.metrics import QueryResult, Summary, score_query, summarise
from app.models.document import Chunk
from app.models.user import User
from app.retrieval.pipeline import degradations, reset_degradations
from app.retrieval.rerank import (
    get_reranker,
    rerank_failures,
    rerank_latencies,
    reset_rerank_stats,
)
from app.services import document_service, retrieval_service


@dataclass
class IngestedCorpus:
    owner_id: uuid.UUID
    email: str
    #: document name -> document id
    document_ids: dict[str, uuid.UUID]
    #: document name -> [(chunk_id, text)] in position order
    chunks: dict[str, list[tuple[uuid.UUID, str]]]


def _install_embedding_cache() -> CachedEmbeddingProvider:
    """Route every embedding call through the disk cache.

    Each consumer did `from app.embeddings import get_embedding_provider`, so
    the name is bound in that module's namespace and patching the factory
    alone would miss them. Rebinding per module is blunt but honest, and the
    list is short enough that a new consumer failing to be cached shows up
    immediately as a run that unexpectedly costs quota.
    """
    import app.retrieval.dense as dense_module
    import app.services.document_service as document_module
    import app.services.retrieval_service as retrieval_module

    cache = CachedEmbeddingProvider(get_embedding_provider())
    for module in (dense_module, document_module, retrieval_module):
        module.get_embedding_provider = lambda cache=cache: cache  # type: ignore[assignment]
    return cache


async def _ingest() -> IngestedCorpus:
    """Create a scratch account and push the fixture corpus through ingestion."""
    email = f"eval-{uuid.uuid4().hex[:10]}@atlas.local"

    async with AsyncSessionLocal() as session:
        user = User(
            email=email,
            hashed_password=hash_password("EvalHarness!0000"),
            full_name="Evaluation Harness",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        owner_id = user.id

    document_ids: dict[str, uuid.UUID] = {}
    for document in load_corpus():
        async with AsyncSessionLocal() as session:
            created = await document_service.create_document(
                session,
                owner_id=owner_id,
                filename=document.name,
                content_type="text/markdown",
                data=document.text.encode("utf-8"),
            )
            document_ids[document.name] = created.id

        # Ingestion normally runs as a background task. The harness awaits it
        # directly so the run is deterministic rather than racing a scheduler.
        # `ingest_document` opens its own session, as it must when FastAPI
        # calls it after the request's session has closed.
        stats = await document_service.ingest_document(document_ids[document.name])
        if stats is None:
            raise RuntimeError(f"ingestion failed for {document.name}")

    chunks: dict[str, list[tuple[uuid.UUID, str]]] = {}
    async with AsyncSessionLocal() as session:
        for name, document_id in document_ids.items():
            rows = (
                await session.execute(
                    select(Chunk.id, Chunk.content)
                    .where(Chunk.document_id == document_id)
                    .order_by(Chunk.position)
                )
            ).all()
            chunks[name] = [(row.id, row.content) for row in rows]

    return IngestedCorpus(owner_id=owner_id, email=email, document_ids=document_ids, chunks=chunks)


async def _teardown(corpus: IngestedCorpus) -> None:
    """Remove the scratch account and everything that cascades from it."""
    from app.vectorstore import get_vector_store

    store = get_vector_store()
    for document_id in corpus.document_ids.values():
        await store.delete(where={"document_id": str(document_id)})

    async with AsyncSessionLocal() as session:
        await session.execute(delete(User).where(User.id == corpus.owner_id))
        await session.commit()


def _relevant_chunk_ids(query: EvalQuery, corpus: IngestedCorpus) -> set[uuid.UUID]:
    """Resolve content markers to the chunk ids that contain them.

    Uses the same normalisation as `validate_dataset`, so a marker validated
    at startup is guaranteed to resolve here. Two different matching rules
    would let labels pass validation and then silently match nothing.
    """
    needles = [normalize(marker) for marker in query.markers]
    return {
        chunk_id
        for chunk_id, text in corpus.chunks[query.document]
        if any(needle in normalize(text) for needle in needles)
    }


async def _evaluate(
    corpus: IngestedCorpus, *, hybrid: bool, rerank: bool, k: int
) -> tuple[list[QueryResult], list[float]]:
    object.__setattr__(settings, "retrieval_hybrid_enabled", hybrid)
    object.__setattr__(settings, "rerank_enabled", rerank)
    # The factory memoises its choice, so a reranker selected under the
    # previous setting would survive the flip and silently invalidate the A/B.
    get_reranker.cache_clear()
    reset_degradations()
    reset_rerank_stats()

    results: list[QueryResult] = []
    query_latencies: list[float] = []

    async with AsyncSessionLocal() as session:
        for query in QUERIES:
            started = time.perf_counter()
            citations = await retrieval_service.retrieve(
                session,
                owner_id=corpus.owner_id,
                document_ids=[corpus.document_ids[query.document]],
                query=query.question,
                top_k=k,
            )
            query_latencies.append((time.perf_counter() - started) * 1000)
            results.append(
                score_query(
                    query_id=query.id,
                    kind=query.kind,
                    expect_empty=query.expect_empty,
                    retrieved=[citation.chunk_id for citation in citations],
                    relevant=_relevant_chunk_ids(query, corpus),
                    k=k,
                )
            )

    return results, query_latencies


def _describe_latency(label: str, values: Sequence[float]) -> None:
    if not values:
        print(f"    {label:<22} (none)")
        return
    ordered = sorted(values)
    # p95 by nearest-rank. The mean hides the model-load spike on the first
    # query, and the tail is what a user actually waits for.
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(
        f"    {label:<22} p50={ordered[len(ordered) // 2]:>7.1f}ms  "
        f"p95={p95:>7.1f}ms  max={ordered[-1]:>7.1f}ms"
    )


async def _calibrate(corpus: IngestedCorpus, *, depth: int) -> None:
    """Report the raw dense score distribution, ungated.

    A relevance threshold should come from where the scores actually separate,
    not from a plausible-looking constant. This runs dense retrieval with the
    gate bypassed and asks three questions:

      - how low does the score of a genuinely relevant chunk go?
      - how high does an irrelevant chunk score on an answerable query?
      - how high does the best chunk score when the answer is not there?

    The third is the one that matters for refusal. If the worst relevant score
    sits above the best unanswerable score, a clean separator exists and the
    only reason the system returns noise is that the floor was set too low.
    """
    from app.retrieval.dense import DenseRetriever

    dense = DenseRetriever()
    relevant_scores: list[float] = []
    distractor_scores: list[float] = []
    negative_scores: list[float] = []

    for query in QUERIES:
        chunks = await dense.retrieve(
            query.question,
            owner_id=corpus.owner_id,
            document_ids=[corpus.document_ids[query.document]],
            top_k=depth,
        )
        if not chunks:
            continue

        if query.expect_empty:
            negative_scores.append(max(chunk.score for chunk in chunks))
            continue

        relevant = _relevant_chunk_ids(query, corpus)
        hits = [chunk.score for chunk in chunks if chunk.chunk_id in relevant]
        misses = [chunk.score for chunk in chunks if chunk.chunk_id not in relevant]
        if hits:
            relevant_scores.append(max(hits))
        if misses:
            distractor_scores.append(max(misses))

    def describe(label: str, values: list[float]) -> None:
        if not values:
            print(f"  {label:<34} (none)")
            return
        ordered = sorted(values)
        print(
            f"  {label:<34} min={ordered[0]:.3f}  "
            f"p50={ordered[len(ordered) // 2]:.3f}  max={ordered[-1]:.3f}  n={len(ordered)}"
        )

    print("\ndense score distribution (gate bypassed)")
    describe("relevant chunk, answerable", relevant_scores)
    describe("best distractor, answerable", distractor_scores)
    describe("best chunk, UNANSWERABLE", negative_scores)

    if relevant_scores and negative_scores:
        worst_relevant = min(relevant_scores)
        best_noise = max(negative_scores)
        print()
        if worst_relevant > best_noise:
            floor = (worst_relevant + best_noise) / 2
            print(f"  separable: worst relevant {worst_relevant:.3f} > best noise {best_noise:.3f}")
            print(f"  suggested retrieval_min_score = {floor:.2f}")
        else:
            print(
                f"  NOT separable by an absolute floor: worst relevant "
                f"{worst_relevant:.3f} <= best noise {best_noise:.3f}"
            )
            print("  no single threshold can both keep recall and reject noise")

    await _calibrate_distinctiveness(corpus, depth=depth)


async def _calibrate_distinctiveness(corpus: IngestedCorpus, *, depth: int) -> None:
    """Is the TOP score unusually high for this query, relative to the rest?

    Absolute cosine similarity carries a large query-dependent baseline: a
    verbose question is closer to everything than a terse one, so 0.62 means
    different things for different queries and the ranges overlap.

    Distinctiveness removes that baseline by comparing each query only against
    itself. When the corpus contains the answer, one chunk stands out from its
    neighbours. When it does not, every chunk is uniformly unrelated and the
    best is barely better than the median — which is precisely the state the
    system currently fails to notice.

        margin = best - median(all scores for this query)

    Reported as a raw margin rather than a z-score: with a dozen chunks the
    standard deviation is itself noisy, and dividing by a noisy denominator
    manufactures false confidence.
    """
    from statistics import median

    from app.retrieval.dense import DenseRetriever

    dense = DenseRetriever()
    answerable: list[float] = []
    unanswerable: list[float] = []
    #: (best score, margin, is_unanswerable) per query, for the joint sweep.
    rows: list[tuple[float, float, bool]] = []

    for query in QUERIES:
        chunks = await dense.retrieve(
            query.question,
            owner_id=corpus.owner_id,
            document_ids=[corpus.document_ids[query.document]],
            top_k=depth,
        )
        if len(chunks) < 3:
            continue

        scores = [chunk.score for chunk in chunks]
        best = max(scores)
        margin = best - median(scores)
        rows.append((best, margin, query.expect_empty))
        (unanswerable if query.expect_empty else answerable).append(margin)

    def describe(label: str, values: list[float]) -> None:
        ordered = sorted(values)
        print(
            f"  {label:<34} min={ordered[0]:.3f}  "
            f"p50={ordered[len(ordered) // 2]:.3f}  max={ordered[-1]:.3f}  n={len(ordered)}"
        )

    print("\ndistinctiveness: best score minus median score, per query")
    describe("answerable", answerable)
    describe("UNANSWERABLE", unanswerable)

    worst = min(answerable)
    best_noise = max(unanswerable)
    print()
    if worst > best_noise:
        print(f"  separable: worst answerable {worst:.3f} > best noise {best_noise:.3f}")
        print(f"  suggested retrieval_min_margin = {(worst + best_noise) / 2:.3f}")
    else:
        # Report how much recall a threshold would cost, since a partial
        # separator may still be the right trade.
        for candidate in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08):
            kept = sum(1 for value in answerable if value >= candidate)
            rejected = sum(1 for value in unanswerable if value < candidate)
            print(
                f"  margin >= {candidate:.2f}  keeps {kept}/{len(answerable)} answerable"
                f"   refuses {rejected}/{len(unanswerable)} unanswerable"
            )

    _sweep_two_signals(rows)


def _sweep_two_signals(rows: list[tuple[float, float, bool]]) -> None:
    """Grid-search a two-signal refusal rule.

    Neither absolute score nor distinctiveness separates alone. They might
    still separate together: a query can be answerable because one passage
    scores highly in absolute terms, OR because one stands out sharply from
    its neighbours, and requiring both to fail before refusing is a weaker
    condition than either on its own.

    The rule under test is:

        answer if best_score >= score_floor AND margin >= margin_floor
        refuse otherwise

    Operating points are ranked by refusals gained subject to a recall budget,
    because the two errors are not symmetric and the choice of budget is a
    product decision rather than a mathematical one.
    """
    answerable = [row for row in rows if not row[2]]
    negatives = [row for row in rows if row[2]]
    if not answerable or not negatives:
        return

    best_points: list[tuple[int, int, float, float]] = []
    for score_step in range(50, 76):
        score_floor = score_step / 100
        for margin_step in range(0, 21):
            margin_floor = margin_step / 200
            kept = sum(
                1
                for best, margin, _ in answerable
                if best >= score_floor and margin >= margin_floor
            )
            refused = sum(
                1
                for best, margin, _ in negatives
                if not (best >= score_floor and margin >= margin_floor)
            )
            best_points.append((refused, kept, score_floor, margin_floor))

    print("\ntwo-signal rule: answer iff best_score >= S and margin >= M")
    total_answerable = len(answerable)
    total_negatives = len(negatives)

    for budget in (0.0, 0.02, 0.05, 0.10):
        minimum_kept = total_answerable * (1.0 - budget)
        feasible = [point for point in best_points if point[1] >= minimum_kept]
        if not feasible:
            print(f"  recall budget {budget:.0%}: no feasible operating point")
            continue
        refused, kept, score_floor, margin_floor = max(feasible)
        print(
            f"  recall budget {budget:>4.0%}:  S={score_floor:.2f} M={margin_floor:.3f}"
            f"   keeps {kept}/{total_answerable}"
            f"   refuses {refused}/{total_negatives}"
        )


def _print_summary(label: str, summary: Summary) -> None:
    print(f"\n  {label}")
    print(f"    answerable queries : {summary.answerable}")
    print(f"    recall@k           : {summary.recall:.3f}")
    print(f"    precision@k        : {summary.precision:.3f}")
    print(f"    MRR                : {summary.mrr:.3f}")
    print(f"    nDCG@k             : {summary.ndcg:.3f}")
    print(f"    unanswerable       : {summary.negatives}")
    print(f"    refusal rate       : {summary.refusal_rate:.3f}")
    print(f"    noise per negative : {summary.noise_per_negative:.2f} results")


def _print_by_kind(results: Sequence[QueryResult]) -> None:
    """Per-category recall.

    The aggregate can stay flat while lexical improves and semantic regresses.
    Splitting by category is what makes that visible instead of averaged away.
    """
    print("\n    recall by category")
    for kind in ("lexical", "semantic", "mixed"):
        subset = [result for result in results if result.kind == kind]
        if subset:
            mean = sum(result.recall for result in subset) / len(subset)
            print(f"      {kind:<9} {mean:.3f}   ({len(subset)} queries)")


def _print_per_query(label: str, results: Sequence[QueryResult]) -> None:
    print(f"\n  {label} — per query")
    print(f"    {'query':<18} {'kind':<11} {'ret':>4} {'recall':>7} {'rr':>6} {'ndcg':>6}")
    print("    " + "-" * 58)
    for result in results:
        if result.expect_empty:
            verdict = "REFUSED" if result.refused else f"{result.returned} NOISE"
            print(f"    {result.query_id:<18} {result.kind:<11} {verdict:>26}")
        else:
            print(
                f"    {result.query_id:<18} {result.kind:<11} {result.returned:>4} "
                f"{result.recall:>7.3f} {result.reciprocal_rank:>6.3f} {result.ndcg:>6.3f}"
            )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument(
        "--config",
        choices=("dense", "hybrid", "both", "rerank"),
        default="both",
        help="'rerank' A/Bs the cross-encoder against hybrid without it",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="enable reranking for the dense/hybrid configurations",
    )
    parser.add_argument("--k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--per-query", action="store_true")
    parser.add_argument("--keep", action="store_true", help="do not delete the scratch corpus")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="report the ungated dense score distribution and stop",
    )
    arguments = parser.parse_args()

    original = settings.retrieval_hybrid_enabled
    original_rerank = settings.rerank_enabled

    # Install the embedding cache before anything embeds. Without it, every
    # run re-embeds the same fixed corpus and the same fixed queries, and the
    # free tier's 1000-per-day embedding cap becomes the limit on how often
    # this project can measure itself — which it did, mid-A/B.
    cache = _install_embedding_cache()

    print("ingesting evaluation corpus…")
    corpus = await _ingest()

    try:
        total_chunks = sum(len(chunks) for chunks in corpus.chunks.values())
        print(f"corpus: {len(corpus.chunks)} documents, {total_chunks} chunks")
        for name, chunks in corpus.chunks.items():
            print(f"  {name:<20} {len(chunks):>3} chunks")

        # Validate before measuring. A marker that matches nothing would show
        # up as a retrieval failure and send us debugging the wrong system.
        problems = validate_dataset(
            {name: [text for _, text in chunks] for name, chunks in corpus.chunks.items()}
        )
        if problems:
            print("\nDATASET PROBLEMS — metrics below would be meaningless:")
            for problem in problems:
                print(f"  ! {problem}")
            return

        print(f"\ngolden set: {len(QUERIES)} queries, k={arguments.k}  (labels validated)")

        if arguments.calibrate:
            await _calibrate(corpus, depth=total_chunks)
            return

        # (label, hybrid, rerank). The reranker axis is compared against
        # hybrid-without-rerank rather than against dense, so the delta
        # attributes to reranking alone instead of to two changes at once.
        if arguments.config == "rerank":
            configurations = [
                ("hybrid, no rerank", True, False),
                ("hybrid + cross-encoder", True, True),
            ]
        elif arguments.config == "both":
            configurations = [
                # Not labelled a "baseline": the calibrated
                # retrieval_min_score applies to both paths, so this is the
                # dense-only topology with a measured threshold, not an
                # untouched earlier configuration.
                ("dense-only", False, arguments.rerank),
                ("hybrid + gate", True, arguments.rerank),
            ]
        else:
            configurations = [(arguments.config, arguments.config == "hybrid", arguments.rerank)]

        summaries: list[tuple[str, Summary]] = []
        for label, hybrid, rerank in configurations:
            results, query_latencies = await _evaluate(
                corpus, hybrid=hybrid, rerank=rerank, k=arguments.k
            )

            # A retriever that raised was silently skipped, so these numbers
            # describe a crippled pipeline rather than the one under test.
            # Reporting them anyway is how "hybrid destroys recall" gets
            # believed when the real cause was an API rate limit.
            failures = degradations()
            if failures:
                print(f"\n  {label}: ABORTED — retriever failures: {failures}")
                print("  Numbers withheld: they would measure a degraded pipeline.")
                continue

            # Same argument for the reranker: a fallback to fused order is the
            # right behaviour for a user and a silent lie in a benchmark.
            rerank_problems = rerank_failures()
            if rerank_problems:
                print(f"\n  {label}: ABORTED — reranker fell back: {rerank_problems}")
                print("  Numbers withheld: they would measure the fallback, not the reranker.")
                continue

            summary = summarise(results)
            summaries.append((label, summary))
            _print_summary(label, summary)
            _print_by_kind(results)

            print("\n    latency")
            _describe_latency("end-to-end retrieve", query_latencies)
            rerank_only = rerank_latencies()
            if rerank_only:
                _describe_latency("reranking alone", rerank_only)
                print(
                    f"    {'reranked queries':<22} {len(rerank_only)} of {len(QUERIES)}"
                    "   (refusals skip it)"
                )

            if arguments.per_query:
                _print_per_query(label, results)

        if len(summaries) == 2:
            (before_label, baseline), (after_label, candidate) = summaries
            print(f"\n  delta ({after_label} - {before_label})")
            for name, before, after in (
                ("recall@k", baseline.recall, candidate.recall),
                ("precision@k", baseline.precision, candidate.precision),
                ("MRR", baseline.mrr, candidate.mrr),
                ("nDCG@k", baseline.ndcg, candidate.ndcg),
                ("refusal rate", baseline.refusal_rate, candidate.refusal_rate),
            ):
                change = after - before
                print(f"    {name:<14} {before:.3f} -> {after:.3f}   {change:+.3f}")

    finally:
        object.__setattr__(settings, "retrieval_hybrid_enabled", original)
        object.__setattr__(settings, "rerank_enabled", original_rerank)
        get_reranker.cache_clear()
        print(f"\nembedding cache: {cache.stats()}")
        if arguments.keep:
            print(f"scratch corpus kept: {corpus.email}")
        else:
            await _teardown(corpus)
            print("scratch corpus removed")


if __name__ == "__main__":
    asyncio.run(main())
