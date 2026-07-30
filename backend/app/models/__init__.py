"""SQLAlchemy ORM models.

Every model module must be imported here so that `Base.metadata` is fully
populated before Alembic autogenerates a migration. A model that is never
imported is invisible to autogenerate, which then cheerfully writes a
migration that drops the table.
"""

from app.models.conversation import Conversation, Message
from app.models.document import Chunk, Document
from app.models.refresh_token import RefreshToken
from app.models.user import User

__all__ = ["Chunk", "Conversation", "Document", "Message", "RefreshToken", "User"]
