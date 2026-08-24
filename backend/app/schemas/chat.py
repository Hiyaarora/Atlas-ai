"""Conversation and chat contracts."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.documents import Citation

MAX_MESSAGE_LENGTH = 16_000


class ConversationCreate(BaseModel):
    title: str | None = Field(None, max_length=255)


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: Literal["user", "assistant", "system"]
    content: str
    model: str | None
    created_at: datetime

    # The sources this answer was grounded in, as recorded when it was
    # written. Empty for user turns, and for assistant turns produced before
    # citations were stored — normalised from NULL here so the client never
    # has to distinguish "no sources" from "sources unknown". Both mean the
    # same thing to a reader: there is nothing to show.
    citations: list[Citation] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _null_is_empty(cls, value: object) -> object:
        return value or []


class ConversationSummary(BaseModel):
    """Sidebar row — deliberately excludes messages.

    Returning full message history for every conversation in the list would
    turn one screen into megabytes of JSON.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    #: The document this conversation is bound to, if any. The UI shows the
    #: name so it is always obvious which source an answer will come from.
    document_id: uuid.UUID | None
    document_filename: str | None = None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[MessageResponse]


class ChatRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("content")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """`min_length` alone would accept "   " — whitespace has length."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message cannot be empty.")
        return stripped
