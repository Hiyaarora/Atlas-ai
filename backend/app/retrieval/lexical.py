"""Lexical retrieval: Okapi BM25, implemented directly.

WHY THIS EXISTS AT ALL
----------------------
Dense embeddings and lexical matching fail on opposite inputs, which is the
entire justification for running both.

An embedding model maps text to a point in semantic space. Ask it for
"WM17S" and it has no useful representation — the token is rare, carries no
semantics, and its vector is close to arbitrary. Cosine similarity has no
concept of "this exact string appears in this passage".

BM25 has exactly that concept and nothing else. A term appearing in 1 of 43
chunks gets a large idf, so a single occurrence is a strong signal. Product
codes, error codes, function names, surnames, dates, acronyms — the things
users most often paste into a search box — are precisely where dense retrieval
is weakest and BM25 is strongest.

THE SCORING FUNCTION
--------------------
    idf(q)      = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(D, Q) = SUM over q in Q of
                    idf(q) * tf * (k1 + 1)
                    ---------------------------------------
                    tf + k1 * (1 - b + b * |D| / avgdl)

Two ideas are doing all the work:

*Saturation* (`k1`). The 10th occurrence of a word says barely more than the
3rd. Raw term frequency would let one keyword-stuffed chunk dominate; the
`tf / (tf + k1 * ...)` shape flattens out, so relevance is bounded.

*Length normalisation* (`b`). A long chunk contains more words and would win
on raw counts alone. Dividing by `|D| / avgdl` asks "is this term frequent
*for a chunk of this size*". `b=0` disables it, `b=1` applies it fully; 0.75
is the long-standing default and we have no evidence to beat it yet — stage
5.2 provides the harness that could produce such evidence.

SCALE, HONESTLY
---------------
The index is built per query, in memory, over the chunks of the documents in
scope. That is O(corpus) work per search, which is fine here and nowhere else:
a conversation is bound to one document, and the largest in this corpus is 43
chunks. At thousands of chunks per query this becomes the bottleneck and the
index belongs in Postgres (`tsvector` + GIN) or a dedicated search engine.
`LexicalRetriever` is the seam where that swap happens; nothing above it would
change.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import Chunk, Document
from app.retrieval.base import RetrievedChunk

logger = get_logger(__name__)


#: Tokens are alphanumeric runs, optionally joined by - _ . or /
#:
#: Keeping those joiners *inside* a token is the whole point: splitting on
#: them would turn "gpt-4" into "gpt" + "4", "config.py" into "config" + "py",
#: and "WM17S" survives only because digits are not separators. Those compound
#: identifiers are exactly what lexical search is for, so destroying them at
#: the tokeniser would defeat the retriever before scoring ever runs.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into scoring terms.

    Note what is absent: a stopword list. It would be redundant. A word in
    every chunk has df == N, so

        idf = ln(1 + (N - N + 0.5) / (N + 0.5))

    which for N=43 is ln(1.0115) ≈ 0.011 — already three orders of magnitude
    below a term appearing once. idf *is* the stopword filter, derived from the
    corpus rather than hardcoded per language. One less list to maintain, and
    it adapts to a corpus where "patient" or "transformer" is the real noise
    word.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over a fixed set of documents.

    Kept deliberately free of Atlas types — it indexes lists of tokens and
    returns `(position, score)`. That makes it trivially unit-testable and
    reusable for anything else that needs ranked lexical search.
    """

    def __init__(
        self,
        documents: Sequence[Sequence[str]],
        *,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self.k1 = settings.bm25_k1 if k1 is None else k1
        self.b = settings.bm25_b if b is None else b

        self._term_frequencies: list[Counter[str]] = [Counter(tokens) for tokens in documents]
        self._lengths: list[int] = [len(tokens) for tokens in documents]
        self._count = len(documents)
        # Guard against division by zero for an empty corpus, and note that an
        # empty document legitimately has length 0 and will score 0.
        self._average_length = (sum(self._lengths) / self._count) if self._count else 0.0

        # Document frequency: how many documents contain each term at least
        # once. Built from the *set* of terms per document, not the counts —
        # df is about presence, tf is about repetition.
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())

        # Precompute idf per term. Doing it here rather than per query matters
        # because idf depends only on the corpus, and a query repeating a term
        # would otherwise recompute a logarithm for every occurrence.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (self._count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def __len__(self) -> int:
        return self._count

    @property
    def _unseen_idf(self) -> float:
        """idf of a term that appears in NO document, i.e. df = 0.

        Needed because a query term absent from the corpus is not neutral —
        it is maximally rare, and a chunk failing to contain it is failing to
        match the most distinctive part of the query. Defaulting such terms to
        idf 0 (the obvious `dict.get` fallback) silently erases them from the
        relevance calculation, which is the bug this property exists to avoid.
        """
        return math.log(1.0 + (self._count + 0.5) / 0.5)

    def idf_coverage(self, query_tokens: Sequence[str], index: int) -> float:
        """What fraction of the query's *information* this document matched.

        WHY THIS IS NEEDED
        ------------------
        BM25 scores are unbounded and corpus-dependent, so there is no absolute
        floor that means the same thing twice. That pushed the relevance gate
        toward a purely relative rule — keep hits near the best one — which has
        a fatal hole: when every match is junk, the best junk still passes.

        Coverage closes it by normalising against the query itself rather than
        against the other candidates:

            sum of idf of query terms THIS chunk contains
            ---------------------------------------------
            sum of idf of ALL query terms

        Bounded in [0, 1] and comparable across queries and corpora, because
        both sides are measured in the same units.

        Crucially it weights by idf, not by word count. For "zebra migration
        patterns across the serengeti" against a database manual, the rare
        terms (zebra, serengeti) are absent and carry nearly all the idf mass,
        while the one term that does match (patterns) carries almost none — so
        coverage is near zero and the chunk is correctly rejected. Counting
        matched *words* instead would score that 1-of-5 and let it through.
        """
        frequencies = self._term_frequencies[index]

        matched = 0.0
        total = 0.0
        # A set: a term repeated in the query should not count twice toward
        # either side of the ratio.
        for term in set(query_tokens):
            idf = self._idf.get(term, self._unseen_idf)
            total += idf
            if frequencies.get(term, 0):
                matched += idf

        return matched / total if total > 0.0 else 0.0

    def score(self, query_tokens: Sequence[str], index: int) -> float:
        """BM25 score of one document against the query."""
        if not self._count or self._average_length == 0:
            return 0.0

        frequencies = self._term_frequencies[index]
        length = self._lengths[index]
        # The denominator's length factor is constant across query terms, so
        # it is hoisted out of the loop.
        length_norm = self.k1 * (1.0 - self.b + self.b * length / self._average_length)

        total = 0.0
        for term in query_tokens:
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                # A term absent from this document contributes exactly zero —
                # BM25 never penalises for absence, it only rewards presence.
                continue
            idf = self._idf.get(term, 0.0)
            total += idf * (term_frequency * (self.k1 + 1.0)) / (term_frequency + length_norm)

        return total

    def search(
        self,
        query_tokens: Sequence[str],
        *,
        top_k: int,
        min_coverage: float | None = None,
    ) -> list[tuple[int, float]]:
        """Rank every document, returning `(position, score)` best first.

        Two filters, and both matter:

        *Zero score* — a chunk sharing no terms with the query is not a weak
        match, it is not a match. Passing it on would hand fusion a rank to
        reward.

        *Low coverage* — a chunk matching only the throwaway words of a query
        scores above zero but is not evidence of anything. Rejecting it here,
        where the corpus statistics live, is what keeps an irrelevant question
        returning nothing at all rather than the least-irrelevant paragraph.
        """
        if not query_tokens:
            return []

        threshold = settings.bm25_min_coverage if min_coverage is None else min_coverage

        scored: list[tuple[int, float]] = []
        for index in range(self._count):
            score = self.score(query_tokens, index)
            if score <= 0.0:
                continue
            if self.idf_coverage(query_tokens, index) < threshold:
                continue
            scored.append((index, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class LexicalRetriever:
    """BM25 over the chunks of the documents in scope, read from Postgres."""

    name = "lexical"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def retrieve(
        self,
        query: str,
        *,
        owner_id: uuid.UUID,
        document_ids: list[uuid.UUID],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not document_ids:
            # Same contract as the dense retriever: no scope means no results,
            # never "search everything".
            logger.warning("lexical_retrieval_without_scope", extra={"owner_id": str(owner_id)})
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # owner_id is filtered here as well as document_id. Ids are unguessable
        # UUIDs so the document filter alone would do, but a bug that confused
        # one document id must still not be able to cross a tenant boundary.
        #
        # lifecycle_status == "active" is NOT optional, and the reason is
        # subtle. Archiving deletes a document's vectors and keeps its chunks,
        # treating the vector index as derived data. Dense retrieval therefore
        # goes silent on an archived document automatically — there is nothing
        # left to search. Lexical retrieval reads the *source of truth*, so
        # without this predicate it would happily return passages from
        # documents that are archived or already scheduled for deletion.
        #
        # That asymmetry is not a detail: it would mean adding a retriever
        # silently widened what users can see. Both retrievers must agree on
        # scope, or "hybrid search" changes visibility rather than just
        # ranking. Observed for real — the Transformer paper had 43 chunks in
        # Postgres and 0 vectors in Chroma, and only the lexical branch
        # returned it.
        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.page_number,
                Chunk.content,
                Document.filename,
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Document.owner_id == owner_id,
                Document.id.in_(document_ids),
                Document.ingestion_status == "ready",
                Document.lifecycle_status == "active",
            )
            .order_by(Chunk.document_id, Chunk.position)
        )

        rows = (await self._session.execute(statement)).all()
        if not rows:
            return []

        index = BM25Index([tokenize(row.content) for row in rows])
        ranked = index.search(query_tokens, top_k=top_k)

        logger.info(
            "lexical_retrieval_completed",
            extra={
                "corpus_chunks": len(rows),
                "query_terms": len(query_tokens),
                "matched": len(ranked),
                "best_score": round(ranked[0][1], 3) if ranked else None,
            },
        )

        return [
            RetrievedChunk(
                chunk_id=rows[position].id,
                document_id=rows[position].document_id,
                filename=rows[position].filename,
                page_number=rows[position].page_number,
                text=rows[position].content,
                score=score,
                retriever=self.name,
            )
            for position, score in ranked
        ]
