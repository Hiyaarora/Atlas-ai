"""Rewrite conversational follow-ups into standalone retrieval queries.

THE PROBLEM
-----------
Retrieval sees one string. A conversation does not work that way:

    user: What is multi-head attention?
    ai:   ...
    user: what about its complexity?

The second question retrieves nothing useful, because "its" carries all the
meaning and an embedding of a pronoun is close to noise. The same is true for
lexical search: "what", "about", "its" are stopwords with near-zero idf, so
BM25 has nothing to match either. Both retrievers fail for the same reason —
the question is not self-contained.

THE FIX
-------
Before retrieving, resolve the question against the conversation so far:

    "what about its complexity?"  ->  "What is the computational complexity
                                      of multi-head attention?"

The rewritten form is used ONLY for retrieval. The user's own words are what
reach the answering model, so a bad rewrite degrades which passages are found
rather than putting words in the user's mouth.

WHY IT IS GATED
---------------
Rewriting costs one LLM request. On a free tier allowing twenty generations a
day, rewriting every message would halve the number of questions a user can
ask. So a cheap local heuristic runs first and most questions skip the call
entirely: a first message has no history to resolve against, and a question
that already names its subject does not need resolving.

The heuristic is deliberately biased toward *not* rewriting. A missed rewrite
degrades one follow-up; an unnecessary one costs a request that the user
needed for an actual answer.
"""

from __future__ import annotations

import re

from app.core.config import settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm import get_llm_provider
from app.llm.base import ChatMessage

logger = get_logger(__name__)

#: Words that only mean something relative to earlier turns.
#:
#: Matched as whole words: "its" must not fire on "digits", and "that" as a
#: conjunction ("show me that it works") is a false positive we accept,
#: because the cost is one wasted request rather than a wrong answer.
_REFERRING = re.compile(
    r"\b("
    r"it|its|it's|this|that|these|those|them|they|their|the same|"
    r"he|she|his|her|one|ones|above|previous|earlier|former|latter"
    r")\b",
    re.IGNORECASE,
)

#: A question this short is almost always elliptical — "why?", "and the
#: second one?", "in Python?" — and cannot stand alone regardless of wording.
_SHORT_QUESTION_WORDS = 5

_SYSTEM_PROMPT = """You rewrite follow-up questions so they can be understood \
without the conversation.

Rules:
- Replace pronouns and references with the thing they refer to, taken from the \
conversation.
- Keep the user's intent and specificity exactly. Do not answer, expand, \
explain, or add detail that was not asked for.
- If the question already stands on its own, repeat it unchanged.
- Reply with the rewritten question and nothing else. No preamble, no quotes, \
no explanation."""


def needs_rewriting(question: str, history: list[ChatMessage]) -> bool:
    """Cheap local test for whether the LLM call is worth making."""
    if not settings.query_rewrite_enabled:
        return False

    # Nothing to resolve against. Also the common case — every conversation's
    # first message — so checking it first keeps the usual path free.
    if not any(message.role == "assistant" for message in history):
        return False

    stripped = question.strip()
    if not stripped:
        return False

    if len(stripped.split()) <= _SHORT_QUESTION_WORDS:
        return True

    return bool(_REFERRING.search(stripped))


async def rewrite_for_retrieval(question: str, history: list[ChatMessage]) -> str:
    """Return a self-contained version of `question`, or the original.

    Never raises. Retrieval must still happen if the rewrite fails: a
    degraded query beats no answer, and the original question is a perfectly
    reasonable fallback — it is what the system used before this existed.
    """
    if not needs_rewriting(question, history):
        return question

    # Only the last few turns. The referent of "it" is almost always recent,
    # and a longer window costs prompt tokens while making the model more
    # likely to resolve against something the user has moved on from.
    window = history[-settings.query_rewrite_history_turns :]
    transcript = "\n".join(
        f"{'User' if message.role == 'user' else 'Assistant'}: {message.content}"
        for message in window
    )

    prompt = (
        f"Conversation so far:\n{transcript}\n\n"
        f"Follow-up question to rewrite:\n{question}\n\n"
        "Rewritten standalone question:"
    )

    try:
        provider = get_llm_provider()
        rewritten = await provider.complete(
            [ChatMessage(role="user", content=prompt)],
            system_prompt=_SYSTEM_PROMPT,
        )
    except LLMError as exc:
        logger.warning("query_rewrite_failed", extra={"code": exc.code})
        return question

    rewritten = rewritten.strip().strip('"').strip()

    # Guard against a model that ignored the instructions and answered the
    # question instead of rewriting it. A reply several times longer than the
    # original is an answer, not a rewrite, and using it as the retrieval
    # query would search for the model's own words rather than the user's.
    if not rewritten or len(rewritten) > max(200, len(question) * 4):
        logger.info("query_rewrite_rejected", extra={"length": len(rewritten)})
        return question

    if rewritten.lower() != question.strip().lower():
        logger.info(
            "query_rewritten",
            extra={
                "original_words": len(question.split()),
                "rewritten_words": len(rewritten.split()),
            },
        )

    return rewritten
