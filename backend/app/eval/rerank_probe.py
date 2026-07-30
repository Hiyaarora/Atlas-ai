"""Measure the cross-encoder directly, without embeddings or a database.

    python -m app.eval.rerank_probe

WHY A SEPARATE PROBE
--------------------
The full harness ingests a corpus, which costs embedding quota — and Gemini's
free tier allows 1000 embedding requests per day, a limit a day of
benchmarking reached. A cross-encoder needs neither embeddings nor an index:
it scores (query, passage) pairs directly. So its two properties can be
measured offline, indefinitely, for free:

  1. RANKING     — given the corpus, does it put the answer at the top?
  2. SEPARATION  — do its scores distinguish "the answer is here" from
                   "nothing here is relevant"?

The second is the open question left by dense retrieval, where cosine could not
separate those two populations at all: relevant chunks scored 0.596-0.804 and
the best chunk on an unanswerable query scored 0.520-0.680, overlapping badly.
That overlap is why the refusal rate stalled at 0.250.

WHAT THIS IS NOT
----------------
Not the pipeline A/B. This scores every chunk in the document, whereas the
pipeline scores only the fused candidates. On this corpus the difference is
small — 12-14 chunks per document against a candidate depth of 20, so
retrieval passes nearly everything through — but on a large corpus the two
would diverge, and the harness remains the authority.
"""

from __future__ import annotations

import asyncio
import statistics
import time
import uuid

from app.core.config import settings
from app.eval.dataset import QUERIES, load_corpus, normalize
from app.ingestion.chunking import split_text
from app.retrieval.fusion import FusedChunk
from app.retrieval.rerank import CrossEncoderReranker


def _build_corpus() -> dict[str, list[FusedChunk]]:
    """Chunk the fixture documents with the real splitter."""
    corpus: dict[str, list[FusedChunk]] = {}
    for document in load_corpus():
        pieces = split_text(
            document.text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        corpus[document.name] = [
            FusedChunk(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid5(uuid.NAMESPACE_DNS, document.name),
                filename=document.name,
                page_number=1,
                text=piece,
                score=1.0 / (60 + position),
                ranks={"dense": position},
                scores={"dense": 0.6},
            )
            for position, piece in enumerate(pieces, start=1)
        ]
    return corpus


async def main() -> None:
    corpus = _build_corpus()
    total = sum(len(chunks) for chunks in corpus.values())
    print(f"corpus: {len(corpus)} documents, {total} chunks (chunk_size={settings.chunk_size})")

    reranker = CrossEncoderReranker()
    started = time.perf_counter()
    reranker._load()
    print(f"model: {reranker.model_name}")
    print(f"warm load: {(time.perf_counter() - started) * 1000:.0f} ms\n")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    answerable_best: list[float] = []
    unanswerable_best: list[float] = []
    misses: list[str] = []

    k = settings.retrieval_top_k

    for query in QUERIES:
        chunks = corpus[query.document]

        started = time.perf_counter()
        ranked = await reranker.rerank(query.question, chunks)
        latencies.append((time.perf_counter() - started) * 1000)

        best = ranked[0].scores["reranker"]

        if query.expect_empty:
            unanswerable_best.append(best)
            continue

        answerable_best.append(best)

        needles = [normalize(marker) for marker in query.markers]
        relevant = {
            chunk.chunk_id
            for chunk in chunks
            if any(needle in normalize(chunk.text) for needle in needles)
        }
        if not relevant:
            misses.append(f"{query.id}: marker matched no chunk at this chunk_size")
            continue

        top = [chunk.chunk_id for chunk in ranked[:k]]
        recalls.append(len(set(top) & relevant) / len(relevant))
        reciprocal_ranks.append(
            next(
                (
                    1.0 / position
                    for position, chunk in enumerate(ranked, start=1)
                    if chunk.chunk_id in relevant
                ),
                0.0,
            )
        )

    if misses:
        print("LABEL PROBLEMS (chunking differs from the ingestion path):")
        for miss in misses:
            print(f"  ! {miss}")
        print()

    def describe(label: str, values: list[float]) -> None:
        ordered = sorted(values)
        print(
            f"  {label:<28} min={ordered[0]:>8.3f}  p50={ordered[len(ordered) // 2]:>8.3f}  "
            f"max={ordered[-1]:>8.3f}  n={len(ordered)}"
        )

    print(f"ranking the whole document with the cross-encoder (k={k})")
    print(f"  recall@{k}                    {statistics.fmean(recalls):.3f}")
    print(f"  MRR                         {statistics.fmean(reciprocal_ranks):.3f}")
    print(
        f"  rank-1 accuracy             {sum(1 for r in reciprocal_ranks if r == 1.0)}"
        f"/{len(reciprocal_ranks)}"
    )

    print("\ntop cross-encoder score per query")
    describe("answerable", answerable_best)
    describe("UNANSWERABLE", unanswerable_best)

    worst = min(answerable_best)
    loudest = max(unanswerable_best)
    print()
    if worst > loudest:
        print(f"  SEPARABLE: worst answerable {worst:.3f} > best noise {loudest:.3f}")
        print(f"  a refusal threshold anywhere in ({loudest:.3f}, {worst:.3f}) is exact")
        print(f"  suggested rerank_min_score = {(worst + loudest) / 2:.2f}")
    else:
        print(f"  overlapping: worst answerable {worst:.3f} <= best noise {loudest:.3f}")
        for candidate in (-8.0, -6.0, -4.0, -2.0, 0.0):
            kept = sum(1 for value in answerable_best if value >= candidate)
            refused = sum(1 for value in unanswerable_best if value < candidate)
            print(
                f"    threshold {candidate:>6.1f}  keeps {kept}/{len(answerable_best)}"
                f"   refuses {refused}/{len(unanswerable_best)}"
            )

    ordered = sorted(latencies)
    print(
        f"\nlatency per query ({total // len(corpus)} chunks avg): "
        f"p50={ordered[len(ordered) // 2]:.0f}ms  "
        f"p95={ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]:.0f}ms  "
        f"max={ordered[-1]:.0f}ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
