"""End-to-end authentication flows against a real database."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.refresh_token import RefreshToken
from app.models.user import User
from tests.conftest import TEST_PASSWORD

PREFIX = settings.api_v1_prefix
COOKIE = settings.refresh_cookie_name


# ==========================================================================
# Registration
# ==========================================================================


async def test_register_creates_user_and_returns_tokens(client: AsyncClient) -> None:
    response = await client.post(
        f"{PREFIX}/auth/register",
        json={"email": "new@example.com", "password": TEST_PASSWORD, "full_name": "New User"},
    )
    body = response.json()

    assert response.status_code == 201
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["email"] == "new@example.com"
    assert body["user"]["is_active"] is True


async def test_register_never_returns_the_password_hash(client: AsyncClient) -> None:
    """The response model is the thing preventing this leak."""
    response = await client.post(
        f"{PREFIX}/auth/register",
        json={"email": "leak@example.com", "password": TEST_PASSWORD},
    )

    assert "hashed_password" not in response.text
    assert "password" not in response.json()["user"]


async def test_password_is_hashed_in_the_database(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await client.post(
        f"{PREFIX}/auth/register",
        json={"email": "stored@example.com", "password": TEST_PASSWORD},
    )

    result = await db_session.execute(select(User).where(User.email == "stored@example.com"))
    user = result.scalar_one()

    assert user.hashed_password != TEST_PASSWORD
    assert user.hashed_password.startswith("$2b$")


async def test_duplicate_email_returns_409(client: AsyncClient, registered_user: User) -> None:
    response = await client.post(
        f"{PREFIX}/auth/register",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_email_is_case_insensitive(client: AsyncClient, registered_user: User) -> None:
    """`Hiya@example.com` must not become a second account."""
    response = await client.post(
        f"{PREFIX}/auth/register",
        json={"email": registered_user.email.upper(), "password": TEST_PASSWORD},
    )

    assert response.status_code == 409


async def test_short_password_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        f"{PREFIX}/auth/register",
        json={"email": "short@example.com", "password": "abc"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"


async def test_invalid_email_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        f"{PREFIX}/auth/register",
        json={"email": "not-an-email", "password": TEST_PASSWORD},
    )

    assert response.status_code == 422


# ==========================================================================
# Login
# ==========================================================================


async def test_login_with_valid_credentials(client: AsyncClient, registered_user: User) -> None:
    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )

    assert response.status_code == 200
    assert response.json()["access_token"]


async def test_login_sets_httponly_refresh_cookie(
    client: AsyncClient, registered_user: User
) -> None:
    """If JavaScript can read this cookie, XSS can steal the session."""
    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )

    set_cookie = response.headers["set-cookie"]
    assert COOKIE in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite" in set_cookie


async def test_refresh_token_is_not_in_the_response_body(
    client: AsyncClient, registered_user: User
) -> None:
    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )
    raw_cookie = client.cookies.get(COOKIE)

    assert raw_cookie is not None
    assert raw_cookie not in response.text


async def test_wrong_password_returns_401(client: AsyncClient, registered_user: User) -> None:
    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": "wrong-password"},
    )

    assert response.status_code == 401


async def test_unknown_email_is_indistinguishable_from_a_wrong_password(
    client: AsyncClient, registered_user: User
) -> None:
    """The login endpoint must not reveal which emails are registered.

    Atlas AI previously reported `account_not_found` for a friendlier message.
    That made the endpoint a user-enumeration oracle: scripted against a
    leaked address list it identifies who holds an account, which assists
    targeted phishing and cuts the cost of credential stuffing.

    This test exists so reintroducing the distinction is a conscious act
    rather than an accident.
    """
    unknown = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": "nobody@example.com", "password": TEST_PASSWORD},
    )
    wrong_password = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": "definitely-not-it"},
    )

    assert unknown.status_code == wrong_password.status_code == 401
    assert unknown.json()["error"]["code"] == wrong_password.json()["error"]["code"]
    assert unknown.json()["error"]["message"] == wrong_password.json()["error"]["message"]


async def test_wrong_password_does_not_reveal_that_the_password_was_wrong(
    client: AsyncClient, registered_user: User
) -> None:
    """Existence may be revealed; which *credential* failed still is not."""
    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": "wrong-password"},
    )
    body = response.json()

    assert response.status_code == 401
    assert body["error"]["code"] == "authentication_error"
    assert body["error"]["message"] == "Incorrect email or password."


async def test_inactive_user_cannot_log_in(
    client: AsyncClient, registered_user: User, db_session: AsyncSession
) -> None:
    registered_user.is_active = False
    await db_session.commit()

    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "account_disabled"


async def test_disabled_status_is_not_revealed_without_the_password(
    client: AsyncClient, registered_user: User, db_session: AsyncSession
) -> None:
    """A wrong password must not tell an attacker the account is suspended."""
    registered_user.is_active = False
    await db_session.commit()

    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": "wrong-password"},
    )

    assert response.json()["error"]["code"] == "authentication_error"


# ==========================================================================
# Protected routes
# ==========================================================================


async def test_me_requires_a_token(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/auth/me")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_error"


async def test_me_returns_the_authenticated_user(
    client: AsyncClient, auth_headers: dict[str, str], registered_user: User
) -> None:
    response = await client.get(f"{PREFIX}/auth/me", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["email"] == registered_user.email


async def test_me_rejects_a_garbage_token(client: AsyncClient) -> None:
    response = await client.get(
        f"{PREFIX}/auth/me", headers={"Authorization": "Bearer not.a.token"}
    )

    assert response.status_code == 401


async def test_me_rejects_a_malformed_authorization_header(client: AsyncClient) -> None:
    response = await client.get(f"{PREFIX}/auth/me", headers={"Authorization": "Basic abc123"})

    assert response.status_code == 401


# ==========================================================================
# Refresh and logout
# ==========================================================================


async def test_refresh_issues_a_new_access_token(
    client: AsyncClient, registered_user: User
) -> None:
    login = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )
    response = await client.post(f"{PREFIX}/auth/refresh")

    assert response.status_code == 200
    assert response.json()["access_token"]
    assert response.json()["user"]["email"] == registered_user.email
    assert login.json()["access_token"]


async def test_refresh_rotates_the_token(client: AsyncClient, registered_user: User) -> None:
    """A refresh token must be single-use."""
    await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )
    original = client.cookies.get(COOKIE)

    await client.post(f"{PREFIX}/auth/refresh")
    rotated = client.cookies.get(COOKIE)

    assert rotated != original


async def test_reusing_a_rotated_refresh_token_fails(
    client: AsyncClient, registered_user: User
) -> None:
    """Replaying a captured token must not work."""
    await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )
    stolen = client.cookies.get(COOKIE)

    await client.post(f"{PREFIX}/auth/refresh")  # rotates, revoking `stolen`

    # Put the captured cookie back, as an attacker replaying it would.
    client.cookies.set(COOKIE, str(stolen))
    replay = await client.post(f"{PREFIX}/auth/refresh")

    assert replay.status_code == 401


async def test_refresh_without_a_cookie_returns_401(client: AsyncClient) -> None:
    response = await client.post(f"{PREFIX}/auth/refresh")

    assert response.status_code == 401


async def test_logout_revokes_the_refresh_token(
    client: AsyncClient, registered_user: User, db_session: AsyncSession
) -> None:
    await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )

    response = await client.post(f"{PREFIX}/auth/logout")

    assert response.status_code == 204

    result = await db_session.execute(
        select(RefreshToken).where(RefreshToken.user_id == registered_user.id)
    )
    tokens = result.scalars().all()
    assert tokens, "a refresh token row should exist"
    assert all(token.revoked_at is not None for token in tokens)


async def test_refresh_after_logout_fails(client: AsyncClient, registered_user: User) -> None:
    await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )
    cookie_before_logout = client.cookies.get(COOKIE)

    await client.post(f"{PREFIX}/auth/logout")

    client.cookies.set(COOKIE, str(cookie_before_logout))
    response = await client.post(f"{PREFIX}/auth/refresh")

    assert response.status_code == 401


async def test_logout_without_a_session_still_succeeds(client: AsyncClient) -> None:
    """Logging out is not an operation a client should be able to fail at."""
    response = await client.post(f"{PREFIX}/auth/logout")

    assert response.status_code == 204


# ==========================================================================
# Login rate limiting
# ==========================================================================


@pytest.fixture(autouse=True)
def _clean_login_limiter():
    """Reset the shared counter around every test.

    The limiter lives at module scope because it must remember across
    requests. That also means one test's failed logins would count against
    the next, making the suite order-dependent.
    """
    from app.api.v1.routes.auth import _login_limiter

    _login_limiter.clear()
    yield
    _login_limiter.clear()


async def test_repeated_failures_are_eventually_rate_limited(
    client: AsyncClient, registered_user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.v1.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes._login_limiter, "max_attempts", 3)

    for _ in range(3):
        response = await client.post(
            f"{PREFIX}/auth/login",
            json={"email": registered_user.email, "password": "wrong"},
        )
        assert response.status_code == 401

    blocked = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": "wrong"},
    )

    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    # Without Retry-After a client has to guess, and the usual guess is
    # "immediately", which keeps the limiter saturated.
    assert int(blocked.headers["Retry-After"]) > 0


async def test_the_limit_applies_before_the_password_is_checked(
    client: AsyncClient, registered_user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked caller must be rejected even with the CORRECT password.

    Verifying first would let an attacker spend a bcrypt hash per guess no
    matter what the limiter said — a cheap denial of service on the worker.
    """
    from app.api.v1.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes._login_limiter, "max_attempts", 2)

    for _ in range(2):
        await client.post(
            f"{PREFIX}/auth/login",
            json={"email": registered_user.email, "password": "wrong"},
        )

    response = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )

    assert response.status_code == 429


async def test_a_successful_login_clears_the_record(
    client: AsyncClient, registered_user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One forgotten password must not count against someone for an hour."""
    from app.api.v1.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes._login_limiter, "max_attempts", 3)

    for _ in range(2):
        await client.post(
            f"{PREFIX}/auth/login",
            json={"email": registered_user.email, "password": "wrong"},
        )

    good = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": registered_user.email, "password": TEST_PASSWORD},
    )
    assert good.status_code == 200

    # The budget is full again, so two more failures still do not block.
    for _ in range(2):
        again = await client.post(
            f"{PREFIX}/auth/login",
            json={"email": registered_user.email, "password": "wrong"},
        )
        assert again.status_code == 401


async def test_unknown_emails_are_also_counted(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enumeration is exactly a stream of unknown-email attempts, so those
    must consume the budget too."""
    from app.api.v1.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes._login_limiter, "max_attempts", 2)

    for index in range(2):
        response = await client.post(
            f"{PREFIX}/auth/login",
            json={"email": f"probe{index}@example.com", "password": TEST_PASSWORD},
        )
        assert response.status_code == 401

    blocked = await client.post(
        f"{PREFIX}/auth/login",
        json={"email": "probe3@example.com", "password": TEST_PASSWORD},
    )

    # Counted by IP, so a different email from the same host is still blocked.
    assert blocked.status_code == 429
