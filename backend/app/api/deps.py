"""Reusable FastAPI dependencies.

`get_current_user` is the single gate every protected route passes through.
Centralising it means an auth rule change happens in one place, and adding
`Depends(get_current_user)` to a route is the entire cost of protecting it.
"""

from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError
from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User
from app.services import auth_service

# auto_error=False so a missing header raises our own AuthenticationError and
# produces the standard Atlas error envelope, rather than Starlette's default
# 403 with a different response shape.
bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    session: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    """Resolve the caller from their bearer token, or reject the request."""
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication credentials were not provided.")

    user_id = decode_access_token(credentials.credentials)

    user = await session.get(User, user_id)
    if user is None:
        # The token signature was valid but the account is gone - e.g. deleted
        # while a token was still live.
        raise AuthenticationError("Access token is invalid.")

    if not user.is_active:
        raise AuthenticationError("This account has been deactivated.")

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Alias kept separate so future checks (email verified, subscription
    active) have an obvious home that does not disturb `get_current_user`."""
    return current_user


CurrentUser = Annotated[User, Depends(get_current_active_user)]

__all__ = ["CurrentUser", "DbSession", "auth_service", "get_current_user"]
