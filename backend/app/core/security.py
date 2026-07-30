"""Password hashing and token issuance.

This is the most security-sensitive module in Atlas AI. Everything here is
deliberately boring: no clever optimisations, no caching, no shortcuts. The
threat model is an attacker who has obtained a full copy of the database.

Design summary:

* Passwords are hashed with bcrypt (slow by design, per-password salt).
* Access tokens are signed JWTs (stateless, short-lived, not revocable).
* Refresh tokens are opaque random strings; only a SHA-256 hash is stored, so
  a database dump yields nothing usable.
"""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
import jwt

from app.core.config import settings
from app.core.exceptions import AuthenticationError

TokenType = Literal["access", "refresh"]

# bcrypt hashes at most the first 72 bytes of a password and silently ignores
# the rest. Truncating on the user's behalf would mean a 100-character
# passphrase is no stronger than its first 72 bytes, while telling the user it
# was accepted in full. We reject instead - see `schemas/auth.py`.
PASSWORD_MAX_BYTES = 72

# 256 bits of entropy. Not guessable, so it needs no slow hash.
REFRESH_TOKEN_BYTES = 32


# ==========================================================================
# Passwords
# ==========================================================================


def hash_password(plain_password: str) -> str:
    """Hash a password for storage.

    bcrypt generates a fresh random salt per call and embeds it in the output,
    so two identical passwords produce different hashes and precomputed
    rainbow tables are useless.
    """
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > PASSWORD_MAX_BYTES:
        raise ValueError(f"Password exceeds bcrypt's {PASSWORD_MAX_BYTES}-byte limit")

    salt = bcrypt.gensalt(rounds=settings.bcrypt_rounds)
    return bcrypt.hashpw(password_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a password against a stored hash.

    `bcrypt.checkpw` compares in constant time, so an attacker cannot learn
    the hash byte-by-byte from response timing.
    """
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > PASSWORD_MAX_BYTES:
        return False

    try:
        return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))
    except ValueError:
        # Malformed hash in the database - treat as a failed login rather than
        # a 500, but it means something is wrong with the stored data.
        return False


# NOTE (2026-07-29): this module previously exposed a
# `waste_time_like_a_real_verification()` helper that ran a dummy bcrypt hash
# when an email did not exist, so that "no such user" (~1ms) and "wrong
# password" (~250ms) could not be told apart by response timing.
#
# It was removed when Atlas AI chose to report "no account found" explicitly
# on login (see `AccountNotFoundError`). Once the response body states whether
# an account exists, equalising the timing hides nothing - and it let an
# attacker force 250ms of bcrypt work per request.
#
# If that product decision is ever reverted, restore the helper *and* the
# generic error message together. Either alone is useless.


# ==========================================================================
# Access tokens (JWT)
# ==========================================================================


def create_access_token(user_id: uuid.UUID) -> tuple[str, datetime]:
    """Return a signed access token and its expiry.

    Claims:
      sub  - subject, the user id
      exp  - expiry (verified by PyJWT automatically)
      iat  - issued at
      jti  - unique token id, so individual tokens can be denylisted later
      type - guards against a refresh token being replayed as an access token
    """
    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=settings.access_token_expire_minutes)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "exp": expires_at,
        "iat": now,
        "jti": uuid.uuid4().hex,
        "type": "access",
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    return token, expires_at


def decode_access_token(token: str) -> uuid.UUID:
    """Verify an access token and return the user id it identifies.

    Raises `AuthenticationError` for anything wrong: bad signature, expired,
    wrong token type, malformed subject. The caller never needs to distinguish
    these, and telling a client *why* a token failed helps only an attacker.
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],  # a list, never `None`
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Access token is invalid.") from exc

    if payload.get("type") != "access":
        raise AuthenticationError("Access token is invalid.")

    try:
        return uuid.UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("Access token is invalid.") from exc


# ==========================================================================
# Refresh tokens (opaque)
# ==========================================================================


def generate_refresh_token() -> tuple[str, str]:
    """Return `(raw_token, token_hash)`.

    The raw token goes to the client in an httpOnly cookie and is never
    persisted. Only the hash is stored, so a leaked database cannot be used to
    mint sessions.
    """
    raw_token = secrets.token_urlsafe(REFRESH_TOKEN_BYTES)
    return raw_token, hash_refresh_token(raw_token)


def hash_refresh_token(raw_token: str) -> str:
    """SHA-256, not bcrypt.

    bcrypt exists to make *guessable* secrets expensive to attack. A 256-bit
    random token is not guessable, so the slow hash buys nothing and would
    make every authenticated refresh 250ms slower. SHA-256 also lets us look
    the token up by exact value with an indexed query.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
