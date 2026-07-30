"""Document and citation contracts."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

IngestionStatus = Literal["pending", "processing", "ready", "failed"]
LifecycleStatus = Literal["active", "archived", "pending_deletion"]


class DocumentResponse(BaseModel):
    """A document as the chat UI sees it.

    Note what is absent: `chunk_count`, `embedding_model`, `reference_count`,
    `storage_key`, `lifecycle_status`. Those are implementation details — how
    many pieces the text was cut into says nothing a user can act on, and
    showing it invites questions the product should never make them ask.

    They remain on the model and in the API's internal use; they are simply
    not part of the user-facing contract.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    size_bytes: int
    status: IngestionStatus
    error: str | None
    page_count: int
    created_at: datetime

    @classmethod
    def from_document(cls, document) -> "DocumentResponse":  # noqa: ANN001 - avoids a cycle
        return cls(
            id=document.id,
            filename=document.filename,
            size_bytes=document.size_bytes,
            status=document.ingestion_status,
            error=document.error,
            page_count=document.page_count,
            created_at=document.created_at,
        )


class DocumentUploadResponse(BaseModel):
    """Returned by upload.

    Carries the conversation created alongside the document, so the client
    navigates straight into it. Without this the frontend would have to
    upload, then create a conversation, then link them — three round trips to
    express one user intent, with two windows for a partial failure.
    """

    document: DocumentResponse
    conversation_id: uuid.UUID


class Citation(BaseModel):
    """One retrieved passage, as shown to the user.

    `index` is the number the model was told to cite — [1], [2] — so the UI
    can link a marker in the answer to the passage it came from. Without it a
    citation list is decorative rather than verifiable.
    """

    index: int
    #: The chunk this passage came from. Used to fetch the authoritative full
    #: text from Postgres when building the prompt.
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    score: float
    excerpt: str


class IngestionStats(BaseModel):
    """Returned by the reindex path, and useful in tests."""

    document_id: uuid.UUID
    pages: int
    chunks: int
    embedding_model: str
