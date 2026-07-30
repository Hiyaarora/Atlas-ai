"""Authentication endpoints.

The route layer owns one thing the service layer does not: how the refresh
token reaches the browser. That is an HTTP transport concern (cookies), so it
lives here.
"""

from fastapi import APIRouter, Cookie, Response, status

from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


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
    responses={401: {"description": "Incorrect email or password"}},
)
async def login(
    payload: LoginRequest,
    response: Response,
    session: DbSession,
) -> TokenResponse:
    user = await auth_service.authenticate_user(session, payload)
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
