"""Authentication endpoints.

The route layer owns one thing the service layer does not: how the refresh
token reaches the browser. That is an HTTP transport concern (cookies), so it
lives here.
"""

from fastapi import APIRouter, Cookie, Request, Response, status

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import AuthenticationError, RateLimitedError
from app.core.logging import get_logger
from app.core.rate_limit import SlidingWindowRateLimiter
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.services import auth_service

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

#: Failed-login counter, shared by every request in this process.
#:
#: Module level rather than per-request because the whole point is memory
#: across requests. See `app/core/rate_limit.py` for why this is in-process
#: and what that does and does not cover.
_login_limiter = SlidingWindowRateLimiter(
    max_attempts=settings.login_rate_limit_attempts,
    window_seconds=settings.login_rate_limit_window_seconds,
)


def _login_keys(request: Request, email: str) -> tuple[str, str]:
    """The two independent buckets a login attempt is counted against.

    Client IP is read from the socket, not from X-Forwarded-For: that header
    is attacker-controlled unless a trusted proxy overwrites it, and trusting
    it blindly would let anyone reset their own limit by inventing an address.
    A deployment behind a real proxy should enable uvicorn's --proxy-headers
    so the socket address is already the true client.
    """
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}", f"email:{email.strip().lower()}"


def _enforce_login_rate_limit(request: Request, email: str) -> None:
    for key in _login_keys(request, email):
        retry_after = _login_limiter.retry_after(key)
        if retry_after is not None:
            logger.warning("login_rate_limited", extra={"bucket": key.split(":", 1)[0]})
            raise RateLimitedError(retry_after_seconds=int(retry_after) + 1)


def _set_refresh_cookie(response: Response, raw_refresh_token: str) -> None:
    """Attach the refresh token as a hardened cookie.

    * httponly - JavaScript cannot read it, so XSS cannot exfiltrate it.
    * secure   - HTTPS only. Off in local dev because localhost is plain HTTP.
    * samesite - the browser will not attach it to cross-site requests, which
                 is the CSRF defence for this endpoint.
    * path     - sent only to /auth, so it never rides along with ordinary
                 API calls that have no use for it.
    """
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=raw_refresh_token,
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite=settings.refresh_cookie_samesite,
        max_age=settings.refresh_token_expire_days * 24 * 60 * 60,
        path=f"{settings.api_v1_prefix}/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite=settings.refresh_cookie_samesite,
        path=f"{settings.api_v1_prefix}/auth",
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    responses={409: {"description": "Email already registered"}},
)
async def register(
    payload: RegisterRequest,
    response: Response,
    session: DbSession,
) -> TokenResponse:
    user = await auth_service.register_user(session, payload)
    access_token, expires_at, raw_refresh = await auth_service.issue_tokens(session, user)

    _set_refresh_cookie(response, raw_refresh)
    return TokenResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for tokens",
    responses={
        401: {"description": "Incorrect email or password"},
        429: {"description": "Too many failed attempts"},
    },
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: DbSession,
) -> TokenResponse:
    # Checked before the password is verified. Doing it afterwards would let
    # an attacker spend a bcrypt hash per guess regardless of the limit, which
    # is a cheap denial-of-service against the whole worker.
    _enforce_login_rate_limit(request, payload.email)

    try:
        user = await auth_service.authenticate_user(session, payload)
    except AuthenticationError:
        for key in _login_keys(request, payload.email):
            _login_limiter.record_failure(key)
        raise

    # A correct password clears the record. Someone who mistyped four times
    # and then succeeded is not mid-attack, and carrying those failures
    # forward would lock out a legitimate user on their next slip.
    for key in _login_keys(request, payload.email):
        _login_limiter.reset(key)

    access_token, expires_at, raw_refresh = await auth_service.issue_tokens(session, user)

    _set_refresh_cookie(response, raw_refresh)
    return TokenResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh cookie for a new access token",
    responses={401: {"description": "Refresh token missing, expired, or revoked"}},
)
async def refresh(
    response: Response,
    session: DbSession,
    atlas_refresh: str | None = Cookie(default=None),
) -> TokenResponse:
    if not atlas_refresh:
        raise AuthenticationError("No refresh token was provided.")

    user, access_token, expires_at, raw_refresh = await auth_service.rotate_refresh_token(
        session, atlas_refresh
    )

    _set_refresh_cookie(response, raw_refresh)
    return TokenResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the current session",
)
async def logout(
    response: Response,
    session: DbSession,
    atlas_refresh: str | None = Cookie(default=None),
) -> None:
    """Always succeeds.

    Logging out is not an operation a client should be able to fail at. If the
    token is already gone, the desired end state is the same.
    """
    if atlas_refresh:
        await auth_service.revoke_refresh_token(session, atlas_refresh)
    _clear_refresh_cookie(response)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="The authenticated user",
    responses={401: {"description": "Missing or invalid access token"}},
)
async def me(current_user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(current_user)
