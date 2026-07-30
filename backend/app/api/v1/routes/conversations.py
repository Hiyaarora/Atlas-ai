"""Conversation and chat endpoints.

Every route here is protected: `CurrentUser` resolves the caller, and the
service scopes each query by `user_id`. There is no endpoint that can read a
conversation the caller does not own.
"""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession
from app.core.logging import get_logger, request_id_ctx
from app.llm import LLMProvider, get_llm_provider
from app.models.conversation import Conversation
from app.schemas.chat import (
    ChatRequest,
    ConversationCreate,
    ConversationDetail,
    ConversationRename,
    ConversationSummary,
    MessageResponse,
)
from app.services import chat_service
from app.services.chat_service import DoneEvent, ErrorEvent, SourcesEvent, TokenEvent

logger = get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["chat"])


# ==========================================================================
# CRUD
# ==========================================================================


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    current_user: CurrentUser,
    session: DbSession,
) -> ConversationSummary:
    """Start a general conversation with no document attached.

    Note there is no way to attach a document here. Binding happens once, at
    upload, and is immutable — a conversation whose source could be swapped
    would break the guarantee that reopening it retrieves from the same place.
    """
    conversation = await chat_service.create_conversation(session, current_user.id, payload)
    return _to_summary(conversation, None)


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(
    current_user: CurrentUser,
    session: DbSession,
) -> list[ConversationSummary]:
    rows = await chat_service.list_conversations_with_documents(session, current_user.id)
    return [_to_summary(conversation, filename) for conversation, filename in rows]


def _to_summary(conversation: Conversation, filename: str | None) -> ConversationSummary:
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        document_id=conversation.document_id,
        document_filename=filename,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    responses={404: {"description": "No such conversation, or it belongs to someone else"}},
)
async def get_conversation(
    conversation_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> ConversationDetail:
    conversation = await chat_service.get_conversation(session, current_user.id, conversation_id)
    messages = await chat_service.get_messages(session, conversation_id)

    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        document_id=conversation.document_id,
        document_filename=await chat_service.get_document_filename(session, conversation),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
        messages=[MessageResponse.model_validate(m) for m in messages],
    )


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: uuid.UUID,
    payload: ConversationRename,
    current_user: CurrentUser,
    session: DbSession,
) -> ConversationSummary:
    conversation = await chat_service.rename_conversation(
        session, current_user.id, conversation_id, payload.title
    )
    filename = await chat_service.get_document_filename(session, conversation)
    return _to_summary(conversation, filename)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
) -> None:
    await chat_service.delete_conversation(session, current_user.id, conversation_id)


# ==========================================================================
# Streaming chat
# ==========================================================================


def _sse(event: str, data: dict[str, object]) -> str:
    """Format one Server-Sent Event frame.

    The wire format is strict: `event:` and `data:` lines, terminated by a
    BLANK line. Omitting the trailing newline is the classic SSE bug - the
    browser buffers the frame forever, waiting for a delimiter that never
    arrives, and the stream appears to hang.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _event_stream(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    provider: LLMProvider,
) -> AsyncIterator[str]:
    """Translate service events into SSE frames.

    Note what cannot happen here: raising an HTTPException. By the time this
    generator runs, the 200 status line and headers are already on the wire.
    Failures must therefore be *data* - an `error` event - not status codes.
    """
    request_id = request_id_ctx.get() or "-"

    try:
        async for event in chat_service.stream_assistant_reply(
            conversation_id=conversation_id,
            user_id=user_id,
            provider=provider,
        ):
            match event:
                case SourcesEvent(citations=citations):
                    yield _sse(
                        "sources",
                        {"citations": [citation.model_dump(mode="json") for citation in citations]},
                    )
                case TokenEvent(text=text):
                    yield _sse("token", {"text": text})
                case DoneEvent(message_id=message_id, content=content, model=model):
                    yield _sse(
                        "done",
                        {"message_id": str(message_id), "content": content, "model": model},
                    )
                case ErrorEvent(code=code, message=message):
                    yield _sse(
                        "error", {"code": code, "message": message, "request_id": request_id}
                    )

    except Exception:  # noqa: BLE001 - last resort inside a live stream
        # An unhandled error here would otherwise truncate the response with
        # no explanation, leaving the UI spinning forever.
        logger.exception(
            "chat_stream_failed",
            extra={"conversation_id": str(conversation_id)},
        )
        yield _sse(
            "error",
            {
                "code": "internal_error",
                "message": "The reply could not be completed.",
                "request_id": request_id,
            },
        )


@router.post(
    "/{conversation_id}/messages",
    summary="Send a message and stream the reply",
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE stream of reply tokens"},
        404: {"description": "No such conversation, or it belongs to someone else"},
    },
)
async def send_message(
    conversation_id: uuid.UUID,
    payload: ChatRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> StreamingResponse:
    """Persist the user's message, then stream the assistant's reply.

    Ownership and persistence happen *before* the response starts, while a
    real HTTP status code can still be returned. Everything after that point
    is the stream's problem.
    """
    conversation = await chat_service.get_conversation(session, current_user.id, conversation_id)
    await chat_service.add_user_message(session, conversation, payload.content)

    provider = get_llm_provider()

    return StreamingResponse(
        _event_stream(conversation_id, current_user.id, provider),
        media_type="text/event-stream",
        headers={
            # Proxies that buffer a response destroy streaming: the client
            # receives everything at once, at the end.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx
            "Connection": "keep-alive",
        },
    )
