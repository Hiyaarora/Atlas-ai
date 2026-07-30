"""Retrieval and grounded-prompt construction.

The dense-only path is the simplest thing that works: embed the question,
take the nearest neighbours, drop the weak ones. The hybrid path replaces
the search step with lexical retrieval, fusion, and reranking — and because the
interface here is "question in, ranked passages out", that swap does not
touch generation.
"""

import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings import get_embedding_provider
from app.models.document import Chunk, Document
from app.retrieval import hybrid_search
from app.schemas.documents import Citation
from app.vectorstore import get_vector_store

logger = get_logger(__name__)

#: Longest excerpt shown back to the user per citation.
_EXCERPT_CHARS = 320


def _excerpt(text: str) -> str:
    """Trim a chunk to a citation-sized preview, cutting on a word boundary."""
    excerpt = text.strip()
    if len(excerpt) <= _EXCERPT_CHARS:
        return excerpt
    return excerpt[:_EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"


async def retrieve(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    query: str,
    top_k: int | None = None,
) -> list[Citation]:
    """Find passages relevant to `query` within an explicit set of documents.

    `document_ids` is required and must be non-empty. That is the enforcement
    point for conversation isolation: there is no signature that permits an
    unscoped search, so "search everything the user owns" is not an
    accidentally-reachable state — it is not expressible.

    A list rather than a single id, even though a conversation currently binds
    to exactly one document. Multi-document workspaces then change only how
    the list is resolved; this function, generation, and citations are
    untouched.
    """
    if not document_ids:
        # Belt and braces. Callers resolve scope from the conversation and
        # should never reach here empty, but returning [] is the only safe
        # interpretation — falling through to an unfiltered search would be
        # the exact leak this design exists to prevent.
        logger.warning("retrieval_called_without_scope", extra={"owner_id": str(owner_id)})
        return []

    if settings.retrieval_hybrid_enabled:
        return await _retrieve_hybrid(
            session,
            owner_id=owner_id,
            document_ids=document_ids,
            query=query,
            top_k=top_k or settings.retrieval_top_k,
        )

    return await _retrieve_dense_only(
        owner_id=owner_id,
        document_ids=document_ids,
        query=query,
        top_k=top_k or settings.retrieval_top_k,
    )


async def _retrieve_hybrid(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    query: str,
    top_k: int,
) -> list[Citation]:
    """Hybrid: dense + BM25, fused by RRF, gated on native scores."""
    fused = await hybrid_search(
        session,
        owner_id=owner_id,
        document_ids=document_ids,
        query=query,
        top_k=top_k,
    )

    return [
        Citation(
            index=index,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            page_number=chunk.page_number,
            # The RRF score is a rank artefact and would be meaningless in the
            # UI. Report the dense similarity when there is one — it is the
            # only score here on a scale a human can reason about — and fall
            # back to the fused value for lexical-only hits.
            score=round(chunk.scores.get("dense", chunk.score), 4),
            excerpt=_excerpt(chunk.text),
        )
        for index, chunk in enumerate(fused, start=1)
    ]


async def _retrieve_dense_only(
    *,
    owner_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    query: str,
    top_k: int,
) -> list[Citation]:
    """The dense-only path, kept intact as the evaluation baseline.

    Not dead code: the harness flips `retrieval_hybrid_enabled` to run the same
    labelled queries through both and produce a real before/after. Deleting it
    would make "hybrid retrieval improved recall" an unverifiable claim.
    """
    provider = get_embedding_provider()

    # "query", not "document": the same text embedded for the two purposes
    # lands in different places, and matching a question embedded as a
    # document against passages embedded as documents retrieves noticeably
    # worse.
    query_vector = await provider.embed_one(query, purpose="query")

    # Both filters, always. `document_id` alone would be sufficient — ids are
    # unguessable UUIDs — but a bug that mixed up one document id must still
    # not be able to cross a tenant boundary. Defence in depth costs nothing
    # here; the filter is evaluated inside Chroma either way.
    where: dict[str, object] = {
        "$and": [
            {"owner_id": str(owner_id)},
            {"document_id": {"$in": [str(document_id) for document_id in document_ids]}},
        ]
    }

    hits = await get_vector_store().search(query_vector, top_k=top_k, where=where)

    # Cosine similarity is always defined, so the store always returns
    # *something* — the nearest chunks even when nothing is relevant. Weak
    # matches are worse than none: the model treats whatever it is handed as
    # evidence and will answer from an unrelated passage.
    #
    # Two filters, because one is not enough:
    #
    #   1. Absolute floor — rejects everything when the corpus simply has no
    #      answer, where a relative cutoff would still keep the "best" of a
    #      uniformly irrelevant set.
    #   2. Relative cutoff — rejects the long tail when there IS a good match,
    #      where an absolute floor cannot, because embedding models have a
    #      high and model-specific baseline similarity between any two texts.
    relevant = [hit for hit in hits if hit.score >= settings.retrieval_min_score]

    if relevant:
        best_score = relevant[0].score
        floor = best_score * settings.retrieval_relative_cutoff
        relevant = [hit for hit in relevant if hit.score >= floor]

    logger.info(
        "retrieval_completed",
        extra={
            "returned": len(hits),
            "kept": len(relevant),
            "best_score": round(hits[0].score, 3) if hits else None,
            "worst_kept": round(relevant[-1].score, 3) if relevant else None,
        },
    )

    citations: list[Citation] = []
    for index, hit in enumerate(relevant, start=1):
        citations.append(
            Citation(
                index=index,
                chunk_id=uuid.UUID(hit.id),
                document_id=uuid.UUID(str(hit.metadata.get("document_id"))),
                filename=str(hit.metadata.get("filename", "document")),
                page_number=int(hit.metadata.get("page_number", 1)),
                score=round(hit.score, 4),
                excerpt=_excerpt(hit.text),
            )
        )

    return citations


async def load_document_overview(
    session: AsyncSession,
    *,
    owner_id: uuid.UUID,
    document_ids: list[uuid.UUID],
    max_chars: int | None = None,
) -> tuple[list[Citation], bool]:
    """Load a document's contents for a whole-document task.

    Deliberately does NOT use vector search. Similarity search answers "which
    passage is most like this question?" — a useless question when the request
    is "summarise the document", because no single passage is the summary.
    Asked anyway, it returns whatever text is superficially similar to a
    generic query: on the Transformer paper, it returned the *bibliography*,
    because a reference list is dense with topic words and matches everything
    weakly.

    So this reads the document in order instead. Modern context windows are
    large enough that most documents fit whole; when one does not, chunks are
    sampled evenly across it so the coverage is representative rather than
    just the beginning.

    Returns `(citations, was_truncated)`.
    """
    if not document_ids:
        return [], False

    budget = max_chars or settings.summary_max_chars

    # This path reads Postgres directly rather than the vector index, so it
    # needs its own scoping — the Chroma metadata filter does not apply here.
    statement = (
        select(Chunk, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Document.owner_id == owner_id,
            Document.id.in_(document_ids),
            Document.ingestion_status == "ready",
        )
        .order_by(Chunk.document_id, Chunk.position)
    )

    rows = (await session.execute(statement)).all()
    if not rows:
        return [], False

    total_chars = sum(len(chunk.content) for chunk, _ in rows)
    truncated = total_chars > budget

    if truncated:
        # Even sampling, not the first N chunks. Truncating from the start
        # would summarise only the introduction and silently miss the results
        # and conclusions — the parts a reader most wants.
        keep = max(1, int(len(rows) * budget / total_chars))
        step = len(rows) / keep
        rows = [rows[min(int(index * step), len(rows) - 1)] for index in range(keep)]

    logger.info(
        "document_overview_loaded",
        extra={"chunks": len(rows), "chars": total_chars, "truncated": truncated},
    )

    citations = [
        Citation(
            index=index,
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            page_number=chunk.page_number,
            score=1.0,  # not a similarity result; the whole document is "relevant"
            excerpt=chunk.content[:160].strip() + ("…" if len(chunk.content) > 160 else ""),
        )
        for index, (chunk, filename) in enumerate(rows, start=1)
    ]

    return citations, truncated


def build_context_block(citations: list[Citation], full_texts: list[str]) -> str:
    """Format retrieved passages for the prompt.

    Each source is numbered and labelled with its filename and page. The
    numbering is what makes citation possible: the model is told to write [1],
    and [1] resolves to a real page the user can open.
    """
    parts: list[str] = []
    for citation, text in zip(citations, full_texts, strict=True):
        parts.append(
            f"[{citation.index}] {citation.filename}, page {citation.page_number}\n{text.strip()}"
        )
    return "\n\n---\n\n".join(parts)


#: Instructions for answering from retrieved context.
#:
#: Every clause here exists to suppress a specific failure mode observed in
#: ungrounded RAG systems:
#:   - "only using the sources"      -> answering from pretraining instead
#:   - "say you could not find it"   -> inventing a plausible answer
#:   - "cite with [n]"               -> unverifiable claims
#:   - "do not merge sources"        -> attributing a fact to the wrong page
GROUNDED_SYSTEM_PROMPT = """You are Atlas AI, a knowledge assistant.

Answer the user's question using ONLY the numbered sources provided below.

Rules:
- Cite every factual claim with its source number in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, say so plainly and stop. Do not \
answer from general knowledge, and do not guess.
- Never attribute a statement to a source that does not contain it.
- Quote exact figures, names and dates from the sources rather than paraphrasing them.
- If the sources conflict, say so and cite both.

Be concise. Prefer a short, well-cited answer over a long, hedged one."""


#: Used when retrieval returns nothing. Without an explicit instruction the
#: model quietly falls back on pretraining, which is exactly the behaviour
#: this whole pipeline exists to prevent.
NO_CONTEXT_SYSTEM_PROMPT = """You are Atlas AI, a knowledge assistant.

The user has documents in their knowledge base, but none of them contain \
anything relevant to this question.

Tell the user you could not find an answer in their documents. You may then \
offer general knowledge, but you MUST label it clearly as not coming from \
their documents."""


#: For whole-document tasks. Note what it does NOT say: there is no "if the
#: sources do not contain the answer, refuse" clause. That instruction is
#: correct for a question about a fact, and actively wrong here — the sources
#: ARE the document, so refusing to summarise them is a contradiction.
SUMMARY_SYSTEM_PROMPT = """You are Atlas AI, a knowledge assistant.

Below is the content of the user's document, in order. Produce the summary or \
overview they asked for.

Rules:
- Summarise what the document actually says. Do not add outside knowledge.
- Cite sections with their source number in square brackets, like [1] or [4].
- Lead with what the document is and what it claims, then the key points.
- Prefer specifics — named methods, figures, results — over vague description.

Be well organised and concise."""


#: Requests that are about a document as a whole rather than a fact inside it.
#:
#: Keyword matching is crude and will misfire on edge cases. The alternative
#: is an extra LLM call to classify intent on every message; proper query
#: routing would replace this.
_OVERVIEW_PATTERNS = re.compile(
    r"\b("
    r"summari[sz]e|summary|tl;?dr|overview|abstract|"
    r"key (?:points|takeaways|findings|ideas)|main (?:points|ideas|findings)|"
    r"what (?:is|are) (?:this|the|these) (?:document|paper|pdf|file)s?\b|"
    r"what(?:'s| is) (?:it|this) about|"
    r"outline|gist|walk me through (?:this|the) (?:document|paper)"
    r")\b",
    re.IGNORECASE,
)


def is_overview_request(question: str) -> bool:
    """True when the user is asking about a document as a whole."""
    return bool(_OVERVIEW_PATTERNS.search(question))
