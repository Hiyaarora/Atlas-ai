"""Conversation management and reply streaming.

The streaming path deserves attention, because it breaks an assumption that
holds everywhere else in the app.

A normal endpoint runs entirely inside the request, so the session yielded by
`get_db` is open for its whole lifetime. A `StreamingResponse` does not: the
endpoint function *returns* the generator and finishes, FastAPI tears down its
dependencies — closing the session — and only then does the generator run.
Touching the request session from inside it raises a closed-session error, and
only under real streaming, which is why it survives unit tests.

So the generator here opens and owns its own session. The route persists the
user's message with the request session before streaming begins; everything
after the first byte belongs to the generator.
"""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import LLMError, NotFoundError
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.llm import ChatMessage, LLMProvider
from app.models.conversation import DEFAULT_CONVERSATION_TITLE, Conversation, Message
from app.models.document import Chunk, Document
from app.schemas.chat import ConversationCreate
from app.schemas.documents import Citation
from app.services import document_lifecycle_service, retrieval_service

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Atlas AI, a knowledge assistant. Answer clearly and concisely. "
    "If you are unsure about something, say so plainly rather than guessing."
)

#: Longest first message used verbatim as a conversation title.
_TITLE_MAX_LENGTH = 60


# ==========================================================================
# Stream events
# ==========================================================================
# The service emits these; the route decides how to put them on the wire.
# Keeping SSE framing out of here means the same generator could feed a
# WebSocket or a CLI without modification.


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    message_id: uuid.UUID
    content: str
    # Which model produced the reply. Sent so the client can label the message
    # immediately; without it the badge only appears after a page reload,
    # which reads as a glitch.
    model: str


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    message: str


@dataclass(frozen=True)
class SourcesEvent:
    """Retrieved passages, emitted *before* the first token.

    Sending sources first lets the UI show what the answer is based on while
    it is still being written — and makes it obvious when an answer is
    ungrounded, because the source list is empty.
    """

    citations: list[Citation]


StreamEvent = TokenEvent | DoneEvent | ErrorEvent | SourcesEvent


# ==========================================================================
# Conversation CRUD
# ==========================================================================


async def create_conversation(
    session: AsyncSession,
    user_id: uuid.UUID,
    payload: ConversationCreate,
    *,
    document_id: uuid.UUID | None = None,
) -> Conversation:
    """Create a conversation, optionally bound to a document.

    The binding is set once, at creation, and never changes. A conversation
    that could be re-pointed at a different document would break the promise
    that reopening it retrieves from the same source — so there is no API to
    do so.
    """
    conversation = Conversation(
        user_id=user_id,
        title=payload.title or DEFAULT_CONVERSATION_TITLE,
        document_id=document_id,
    )
    session.add(conversation)
    await session.flush()

    if document_id is not None:
        # In the same transaction as the insert, so the count and the row it
        # counts can never disagree.
        await document_lifecycle_service.sync_reference_count(session, document_id)

    await session.commit()
    await session.refresh(conversation)

    logger.info(
        "conversation_created",
        extra={
            "conversation_id": str(conversation.id),
            "document_id": str(document_id) if document_id else None,
        },
    )
    return conversation


async def list_conversations(
    session: AsyncSession, user_id: uuid.UUID, *, include_archived: bool = False
) -> list[Conversation]:
    """List conversations, newest activity first.

    Archived conversations are hidden by default. `include_archived` is the
    seam a future history view uses; nothing in the chat UI sets it.
    """
    statement = select(Conversation).where(Conversation.user_id == user_id)
    if not include_archived:
        statement = statement.where(Conversation.is_archived.is_(False))

    result = await session.execute(statement.order_by(Conversation.last_message_at.desc()))
    return list(result.scalars().all())


async def list_conversations_with_documents(
    session: AsyncSession, user_id: uuid.UUID
) -> list[tuple[Conversation, str | None]]:
    """Sidebar rows with their document's filename.

    A LEFT JOIN rather than N follow-up queries: the sidebar shows every
    conversation, so lazy-loading the document per row would be a textbook
    N+1. The join is also why `Conversation.document` is `lazy="raise"` —
    accidental lazy loads fail loudly instead of quietly costing a query each.
    """
    result = await session.execute(
        select(Conversation, Document.filename)
        .outerjoin(Document, Document.id == Conversation.document_id)
        .where(Conversation.user_id == user_id, Conversation.is_archived.is_(False))
        .order_by(Conversation.last_message_at.desc())
    )
    return [(conversation, filename) for conversation, filename in result.all()]


async def get_document_filename(session: AsyncSession, conversation: Conversation) -> str | None:
    if conversation.document_id is None:
        return None
    document = await session.get(Document, conversation.document_id)
    return document.filename if document else None


async def get_conversation(
    session: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID
) -> Conversation:
    """Fetch a conversation the user owns.

    Ownership is part of the WHERE clause, not a check after loading. A
    missing row and someone else's row are indistinguishable from here, so
    404 is returned for both — a 403 would confirm the conversation exists,
    letting an attacker enumerate valid ids.
    """
    result = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return conversation


async def get_messages(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id)
        # id breaks ties: two rows written in the same microsecond would
        # otherwise come back in arbitrary order, scrambling the transcript.
        .order_by(Message.created_at, Message.id)
    )
    return list(result.scalars().all())


async def rename_conversation(
    session: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID, title: str
) -> Conversation:
    conversation = await get_conversation(session, user_id, conversation_id)
    conversation.title = title
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def delete_conversation(
    session: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID
) -> None:
    """Delete a conversation and update its document's reference count.

    Deleting the last conversation for a document does NOT delete the
    document. The count drops to zero, the janitor later moves it to
    `pending_deletion`, and only after the grace period is anything destroyed.
    That chain is what makes an accidental delete survivable.
    """
    # Ownership first, so this can never delete another user's row.
    conversation = await get_conversation(session, user_id, conversation_id)
    document_id = conversation.document_id

    await session.execute(
        delete(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    if document_id is not None:
        # Recomputed after the delete, in the same transaction.
        await document_lifecycle_service.sync_reference_count(session, document_id)

    await session.commit()
    logger.info(
        "conversation_deleted",
        extra={
            "conversation_id": str(conversation_id),
            "document_id": str(document_id) if document_id else None,
        },
    )


# ==========================================================================
# Messages
# ==========================================================================


async def add_user_message(
    session: AsyncSession, conversation: Conversation, content: str
) -> Message:
    """Persist the user's turn and title the conversation if it is new.

    Committed before generation starts, so a mid-stream failure or a closed
    browser tab never loses what the user typed.
    """
    message = Message(conversation_id=conversation.id, role="user", content=content)
    session.add(message)

    if conversation.title == DEFAULT_CONVERSATION_TITLE:
        conversation.title = _derive_title(content)

    # Drives inactivity-based archiving. Not `updated_at`, which a rename
    # would also bump — a renamed-but-unused conversation is still idle.
    conversation.last_message_at = func.now()

    await session.commit()
    await session.refresh(message)
    return message


def _derive_title(first_message: str) -> str:
    """Name a conversation after its opening message.

    Cheap and deterministic. Asking the model to summarise would cost a round
    trip and an API call for something a truncation handles adequately.
    """
    flattened = " ".join(first_message.split())
    if len(flattened) <= _TITLE_MAX_LENGTH:
        return flattened
    return flattened[: _TITLE_MAX_LENGTH - 1].rstrip() + "…"


def build_history(messages: list[Message]) -> list[ChatMessage]:
    """Convert stored rows into the provider-agnostic history.

    Only the most recent `llm_history_message_limit` turns are replayed.
    Conversations grow without bound while context windows do not, so an
    untrimmed history eventually produces a 400 from the provider — after
    you have already paid for the tokens.

    Token-aware trimming and summarisation of the dropped prefix would be
    better; message count is a crude but honest first pass.
    """
    turns = [message for message in messages if message.role in ("user", "assistant")]

    # Drop user turns that never received a reply.
    #
    # The user's message is committed before generation starts, so a failed
    # generation — a provider outage, a 429, a closed tab — leaves a user turn
    # with no assistant turn after it. Replayed as-is, the model sees several
    # apparently-pending questions and answers all of them at once.
    #
    # Observed exactly that: after two rate-limited requests, the next reply
    # opened by answering both earlier questions instead of the one asked.
    #
    # The final message is always kept: that is the question being asked now,
    # and it is unanswered by definition.
    answered: list[Message] = []
    for index, message in enumerate(turns):
        is_last = index == len(turns) - 1
        followed_by_reply = not is_last and turns[index + 1].role == "assistant"

        if message.role == "assistant" or is_last or followed_by_reply:
            answered.append(message)

    recent = answered[-settings.llm_history_message_limit :]
    return [
        ChatMessage(role="user" if message.role == "user" else "assistant", content=message.content)
        for message in recent
    ]


# ==========================================================================
# Streaming
# ==========================================================================


async def stream_assistant_reply(
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    provider: LLMProvider,
) -> AsyncIterator[StreamEvent]:
    """Generate a reply, emitting events as it goes, and persist the result.

    Opens its own database session — see the module docstring.
    """
    accumulated: list[str] = []

    async with AsyncSessionLocal() as session:
        try:
            # Re-verify ownership in this session. The route already checked,
            # but that was a different session and a different moment; the
            # generator must not assume it.
            conversation = await get_conversation(session, user_id, conversation_id)
            messages = await get_messages(session, conversation_id)
            history = build_history(messages)

            # --- retrieval ------------------------------------------------
            # The latest user turn is the retrieval query. Using the whole
            # conversation would drown the actual question in earlier topics;
            # Query rewriting would let follow-ups like "what about the
            # second one?" retrieve sensibly.
            question = next(
                (m.content for m in reversed(messages) if m.role == "user"),
                "",
            )
            citations, system_prompt = await _ground(session, conversation, user_id, question)
            yield SourcesEvent(citations=citations)

            async for chunk in provider.stream_chat(history, system_prompt=system_prompt):
                accumulated.append(chunk)
                yield TokenEvent(text=chunk)

        except GeneratorExit:
            # The consumer stopped iterating — in practice, the browser tab
            # closed. Save the partial answer so the user finds it on their
            # next visit rather than a conversation that dead-ends.
            await _persist_reply(session, conversation_id, accumulated, provider, partial=True)
            raise

        except LLMError as exc:
            logger.warning(
                "chat_stream_llm_error",
                extra={"conversation_id": str(conversation_id), "code": exc.code},
            )
            # Keep a partial answer if one exists; discard an empty one rather
            # than storing a blank assistant turn.
            await _persist_reply(session, conversation_id, accumulated, provider, partial=True)
            yield ErrorEvent(code=exc.code, message=exc.message)
            return

        except NotFoundError as exc:
            yield ErrorEvent(code=exc.code, message=exc.message)
            return

        message = await _persist_reply(session, conversation_id, accumulated, provider)
        if message is not None:
            yield DoneEvent(
                message_id=message.id,
                content="".join(accumulated),
                model=provider.model,
            )


async def resolve_scope(session: AsyncSession, conversation: Conversation) -> list[uuid.UUID]:
    """Which documents this conversation may retrieve from.

    THE isolation boundary. Scope comes from the conversation row, never from
    the logged-in user, so reopening a conversation months later searches
    exactly the document it was created for — regardless of what has been
    uploaded since.

    Returns a list, though a conversation binds to one document today. Adding
    multi-document workspaces means changing this function and nothing else:
    retrieval, generation and citations already take a list.

    An archived or half-ingested document yields an empty scope. There is no
    fallback to "search everything", because a silent widening of scope is
    exactly the bug this design forbids.
    """
    if conversation.document_id is None:
        return []

    document = await session.get(Document, conversation.document_id)
    if document is None or not document.is_searchable:
        return []

    return [document.id]


async def _ground(
    session: AsyncSession,
    conversation: Conversation,
    user_id: uuid.UUID,
    question: str,
) -> tuple[list[Citation], str]:
    """Retrieve context and build the system prompt for this turn.

    Three cases, deliberately distinct:

    * Conversation has no document -> behave as a plain assistant. Telling a
      user "nothing relevant was found" when they never attached anything is
      confusing.
    * Document attached, nothing relevant -> say so. Silently answering from
      pretraining is the failure this whole pipeline exists to prevent.
    * Relevant passages -> answer strictly from them, with citations.
    """
    document_ids = await resolve_scope(session, conversation)

    if not document_ids or not question.strip():
        return [], SYSTEM_PROMPT

    # Inactivity means "nobody asked it anything", so the clock is reset here,
    # at the point of actual use.
    await document_lifecycle_service.touch_document(session, document_ids[0])

    # Route before retrieving. "Summarise this document" is not a search
    # query, and running it through vector search returns whatever text is
    # superficially similar to a generic phrase — on a research paper, the
    # bibliography. The request needs the document, not a passage from it.
    if retrieval_service.is_overview_request(question):
        citations, truncated = await retrieval_service.load_document_overview(
            session, owner_id=user_id, document_ids=document_ids
        )
        if citations:
            full_texts = await _load_chunk_texts(session, citations)
            context = retrieval_service.build_context_block(citations, full_texts)

            note = (
                "\n\nNOTE: the document was too large to include in full. The "
                "sections below are sampled evenly across it. Say so in your "
                "answer."
                if truncated
                else ""
            )
            return citations, (
                f"{retrieval_service.SUMMARY_SYSTEM_PROMPT}{note}\n\n"
                f"--- DOCUMENT ---\n\n{context}\n\n--- END DOCUMENT ---"
            )

    citations = await retrieval_service.retrieve(
        session, owner_id=user_id, document_ids=document_ids, query=question
    )
    if not citations:
        return [], retrieval_service.NO_CONTEXT_SYSTEM_PROMPT

    # The excerpt on a Citation is truncated for display; the model gets the
    # full chunk text, read from Postgres rather than the search index so the
    # prompt is always built from the authoritative copy.
    full_texts = await _load_chunk_texts(session, citations)
    context = retrieval_service.build_context_block(citations, full_texts)

    return citations, (
        f"{retrieval_service.GROUNDED_SYSTEM_PROMPT}\n\n"
        f"--- SOURCES ---\n\n{context}\n\n--- END SOURCES ---"
    )


async def _load_chunk_texts(session: AsyncSession, citations: list[Citation]) -> list[str]:
    """Fetch full chunk text for the retrieved passages, in citation order."""
    ids = [citation.chunk_id for citation in citations]

    result = await session.execute(select(Chunk).where(Chunk.id.in_(ids)))
    by_id = {chunk.id: chunk.content for chunk in result.scalars().all()}

    # Fall back to the excerpt if a chunk vanished between search and read —
    # possible if the document was deleted mid-request.
    return [by_id.get(citation.chunk_id, citation.excerpt) for citation in citations]


async def _persist_reply(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    chunks: list[str],
    provider: LLMProvider,
    *,
    partial: bool = False,
) -> Message | None:
    """Write the assistant's turn. Returns None when there is nothing to save."""
    content = "".join(chunks)
    if not content.strip():
        return None

    message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=content,
        model=provider.model,
    )
    session.add(message)

    # Touch the parent so the sidebar sorts by real activity. `onupdate=now()`
    # does not fire here: inserting a child row is not a change to the parent,
    # so nothing on `conversations` is dirty and no UPDATE is emitted. Setting
    # the column explicitly is the honest way to say "this row changed".
    conversation = await session.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.updated_at = func.now()
        conversation.last_message_at = func.now()

    await session.commit()
    await session.refresh(message)

    logger.info(
        "assistant_message_saved",
        extra={
            "conversation_id": str(conversation_id),
            "chars": len(content),
            "model": provider.model,
            "partial": partial,
        },
    )
    return message
