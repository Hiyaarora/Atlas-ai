"""Authentication business logic.

Transport-agnostic on purpose: nothing here knows about HTTP, cookies, or
FastAPI. The route layer decides how a refresh token reaches the client; this
module only decides whether one should be issued.
"""

import asyncio
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    AccountDisabledError,
    AuthenticationError,
    ConflictError,
    NotFoundError,
)
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest

logger = get_logger(__name__)

# Returned to the client on any failed login, regardless of cause. Telling an
# attacker "no account with that email" hands them a free user-enumeration
# oracle.
_INVALID_CREDENTIALS = "Incorrect email or password."

#: A real bcrypt hash of a value nobody can supply, used only to spend the
#: same CPU time on an unknown email as on a real one. Computed once at
#: import: doing it per request would double the cost of every failed login.
_TIMING_EQUALISER_HASH = hash_password(secrets.token_urlsafe(32))


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    result = await session.execute(select(User).where(User.email == email.strip().lower()))
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: uuid.UUID) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise NotFoundError("User not found.")
    return user


async def register_user(session: AsyncSession, payload: RegisterRequest) -> User:
    """Create a new account.

    Uniqueness is enforced by the database, not by a pre-check. A
    SELECT-then-INSERT has a race window in which two concurrent requests both
    see "email is free" and both insert; the unique constraint is the only
    thing that actually holds under concurrency.
    """
    # bcrypt is deliberately slow (~300ms at cost 12) and CPU-bound. Called
    # directly it would block the event loop for that entire time, stalling
    # EVERY other request in the process - measured: /health/live went from
    # 2ms to 1459ms under five concurrent registrations.
    #
    # `to_thread` moves it to the default executor. The bcrypt extension
    # releases the GIL while hashing, so the work genuinely runs in parallel
    # and the loop stays responsive.
    hashed_password = await asyncio.to_thread(hash_password, payload.password)

    user = User(
        email=payload.email,
        hashed_password=hashed_password,
        full_name=payload.full_name,
    )
    session.add(user)

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        logger.info("registration_conflict", extra={"email": payload.email})
        raise ConflictError("An account with this email already exists.") from exc

    await session.refresh(user)
    logger.info("user_registered", extra={"user_id": str(user.id)})
    return user


async def authenticate_user(session: AsyncSession, payload: LoginRequest) -> User:
    """Verify credentials and return the user.

    Every credential failure reports the SAME error, whether the email is
    unknown or the password is wrong. Distinguishing them is friendlier, and
    Atlas AI did exactly that until this change, but it turns the endpoint
    into a user-enumeration oracle: anyone can script it to learn which
    addresses are registered, which makes targeted phishing and credential
    stuffing cheaper. Friendliness is not worth handing out the user list.

    Ordering matters and is not arbitrary: the account-disabled check runs
    only *after* the password verifies, so an attacker cannot probe which
    accounts are suspended without already knowing the credentials.
    """
    user = await get_user_by_email(session, payload.email)

    if user is None:
        # Verify against a throwaway hash rather than returning immediately.
        # Identical responses are not enough on their own — bcrypt takes
        # ~250ms, so an early return would make "unknown email" answer in
        # microseconds and "wrong password" in a quarter of a second. The
        # timing alone would rebuild the oracle the message no longer leaks.
        await asyncio.to_thread(verify_password, payload.password, _TIMING_EQUALISER_HASH)
        logger.info("login_failed", extra={"reason": "unknown_email"})
        raise AuthenticationError(_INVALID_CREDENTIALS)

    # Offloaded for the same reason as hashing - see `register_user`.
    if not await asyncio.to_thread(verify_password, payload.password, user.hashed_password):
        logger.info("login_failed", extra={"reason": "bad_password", "user_id": str(user.id)})
        raise AuthenticationError(_INVALID_CREDENTIALS)

    if not user.is_active:
        logger.warning("login_failed", extra={"reason": "inactive", "user_id": str(user.id)})
        raise AccountDisabledError()

    logger.info("login_succeeded", extra={"user_id": str(user.id)})
    return user


async def issue_tokens(session: AsyncSession, user: User) -> tuple[str, datetime, str]:
    """Mint an access token and a persisted refresh token.

    Returns `(access_token, access_expires_at, raw_refresh_token)`. The raw
    refresh token is returned once and never stored - only its hash is.
    """
    access_token, access_expires_at = create_access_token(user.id)
    raw_refresh, refresh_hash = generate_refresh_token()

    session.add(
        RefreshToken(
            user_id=user.id,
            token_hash=refresh_hash,
            expires_at=datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days),
        )
    )
    await session.commit()

    return access_token, access_expires_at, raw_refresh


async def rotate_refresh_token(
    session: AsyncSession, raw_refresh_token: str
) -> tuple[User, str, datetime, str]:
    """Exchange a refresh token for a new pair, revoking the old one.

    Rotation matters: a refresh token that stays valid after use can be
    replayed indefinitely by anyone who captured it. Rotating means a stolen
    token works at most once, and its use invalidates the legitimate user's
    session - which surfaces the compromise instead of hiding it.
    """
    token_hash = hash_refresh_token(raw_refresh_token)

    result = await session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    stored = result.scalar_one_or_none()

    if stored is None or not stored.is_active:
        logger.warning("refresh_rejected", extra={"reason": "missing_or_inactive"})
        raise AuthenticationError("Refresh token is invalid or expired.")

    user = await session.get(User, stored.user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Refresh token is invalid or expired.")

    stored.revoked_at = datetime.now(UTC)
    await session.flush()

    access_token, access_expires_at, raw_refresh = await issue_tokens(session, user)
    logger.info("token_refreshed", extra={"user_id": str(user.id)})
    return user, access_token, access_expires_at, raw_refresh


async def revoke_refresh_token(session: AsyncSession, raw_refresh_token: str) -> None:
    """Log out one session.

    Idempotent and silent: logging out with an already-invalid token is not an
    error worth surfacing, and reporting "that token did not exist" would leak
    information to an attacker probing stolen values.
    """
    token_hash = hash_refresh_token(raw_refresh_token)

    result = await session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    stored = result.scalar_one_or_none()

    if stored is not None and stored.revoked_at is None:
        stored.revoked_at = datetime.now(UTC)
        await session.commit()
        logger.info("logout", extra={"user_id": str(stored.user_id)})
