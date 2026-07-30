"""Document upload and management.

Routes stay thin: validate, delegate, return. Every decision about lifecycle,
reference counting or cleanup lives in a service — nothing here knows what a
grace period is.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, status

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion import supported_extensions
from app.schemas.chat import ConversationCreate
from app.schemas.documents import DocumentResponse, DocumentUploadResponse
from app.services import chat_service, document_service

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["knowledge"])


@router.get("/supported-types", summary="File types Atlas AI can ingest")
async def supported_types() -> dict[str, object]:
    return {
        "extensions": list(supported_extensions()),
        "max_bytes": settings.max_upload_bytes,
    }


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document and start a conversation about it",
    responses={422: {"description": "Unsupported type, empty file, or over the size limit"}},
)
async def upload_document(
    current_user: CurrentUser,
    session: DbSession,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Store a document, open a conversation bound to it, and queue ingestion.

    One request expresses one user intent: "I want to talk about this file."
    The conversation is created here rather than by the client so the document
    and its conversation are linked inside a single transaction — there is no
    window in which an orphaned document or an unbound conversation exists.

    202, not 201: the document exists but is not yet queryable. Parsing,
    chunking and embedding a large file takes tens of seconds, and holding the
    request open would time out behind most proxies. The client navigates into
    the conversation immediately and watches `status` there.
    """
    data = await file.read()

    document = await document_service.create_document(
        session,
        owner_id=current_user.id,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        data=data,
    )

    conversation = await chat_service.create_conversation(
        session,
        current_user.id,
        ConversationCreate(title=document.filename),
        document_id=document.id,
    )

    # Runs after the response is sent, with its own session.
    background_tasks.add_task(document_service.ingest_document, document.id)

    return DocumentUploadResponse(
        document=DocumentResponse.from_document(document),
        conversation_id=conversation.id,
    )


@router.get("", response_model=list[DocumentResponse], summary="List your documents")
async def list_documents(current_user: CurrentUser, session: DbSession) -> list[DocumentResponse]:
    documents = await document_service.list_documents(session, current_user.id)
    return [DocumentResponse.from_document(document) for document in documents]


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={404: {"description": "No such document, or it belongs to someone else"}},
)
async def get_document(
    document_id: uuid.UUID, current_user: CurrentUser, session: DbSession
) -> DocumentResponse:
    document = await document_service.get_document(session, current_user.id, document_id)
    return DocumentResponse.from_document(document)


@router.post(
    "/{document_id}/reindex",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run ingestion",
)
async def reindex_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    session: DbSession,
    background_tasks: BackgroundTasks,
) -> DocumentResponse:
    """The escape hatch for a failed ingestion, and the mechanism for moving a
    corpus to a new embedding model without re-uploading anything."""
    document = await document_service.get_document(session, current_user.id, document_id)
    background_tasks.add_task(document_service.ingest_document, document.id)
    return DocumentResponse.from_document(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a document from your knowledge base",
)
async def delete_document(
    document_id: uuid.UUID, current_user: CurrentUser, session: DbSession
) -> None:
    """Soft delete. The document leaves search immediately; the janitor
    destroys it after the configured grace period."""
    await document_service.delete_document(session, current_user.id, document_id)
