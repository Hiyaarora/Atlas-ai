"""Unit tests for the security primitives.

These are pure functions with no database, so they are fast and exhaustive.
Anything security-critical gets tested at this level as well as end to end.
"""

import time
import uuid

import jwt
import pytest

from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.core.security import (
    PASSWORD_MAX_BYTES,
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)

# ---- Passwords -----------------------------------------------------------


def test_hash_is_not_the_password() -> None:
    hashed = hash_password("hunter2")

    assert hashed != "hunter2"
    assert "hunter2" not in hashed


def test_same_password_hashes_differently_each_time() -> None:
    """Per-hash salt: identical passwords must not produce identical hashes."""
    assert hash_password("hunter2") != hash_password("hunter2")


def test_verify_accepts_correct_password() -> None:
    assert verify_password("hunter2", hash_password("hunter2")) is True


def test_verify_rejects_wrong_password() -> None:
    assert verify_password("hunter3", hash_password("hunter2")) is False


def test_verify_rejects_malformed_hash_without_raising() -> None:
    """Corrupt data in the column must be a failed login, not a 500."""
    assert verify_password("hunter2", "not-a-bcrypt-hash") is False


def test_password_over_bcrypt_limit_is_rejected_not_truncated() -> None:
    """Silent truncation would make these two passwords interchangeable."""
    too_long = "a" * (PASSWORD_MAX_BYTES + 1)

    with pytest.raises(ValueError, match="72-byte limit"):
        hash_password(too_long)


def test_multibyte_characters_count_as_bytes_not_characters() -> None:
    """20 emoji are 80 bytes - over the limit despite being 20 characters."""
    emoji_password = "🔐" * 20

    assert len(emoji_password) == 20
    with pytest.raises(ValueError):
        hash_password(emoji_password)


# ---- Access tokens -------------------------------------------------------


def test_access_token_roundtrip() -> None:
    user_id = uuid.uuid4()
    token, _ = create_access_token(user_id)

    assert decode_access_token(token) == user_id


def test_token_signed_with_another_key_is_rejected() -> None:
    """The whole point of a signature."""
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": time.time() + 600, "type": "access"},
        "an-attacker-key-that-is-long-enough-to-be-plausible",
        algorithm="HS256",
    )

    with pytest.raises(AuthenticationError):
        decode_access_token(forged)


def test_expired_token_is_rejected() -> None:
    expired = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": time.time() - 1, "type": "access"},
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AuthenticationError, match="expired"):
        decode_access_token(expired)


def test_refresh_token_cannot_be_used_as_an_access_token() -> None:
    """Without the `type` claim check, this would authenticate a request."""
    wrong_type = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": time.time() + 600, "type": "refresh"},
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AuthenticationError):
        decode_access_token(wrong_type)


def test_token_without_required_claims_is_rejected() -> None:
    incomplete = jwt.encode(
        {"sub": str(uuid.uuid4())},  # no exp, no type
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AuthenticationError):
        decode_access_token(incomplete)


def test_garbage_token_is_rejected() -> None:
    with pytest.raises(AuthenticationError):
        decode_access_token("not.a.jwt")


def test_alg_none_attack_is_rejected() -> None:
    """The classic JWT vulnerability: an unsigned token claiming alg=none."""
    unsigned = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": time.time() + 600, "type": "access"},
        key="",
        algorithm="none",
    )

    with pytest.raises(AuthenticationError):
        decode_access_token(unsigned)


# ---- Refresh tokens ------------------------------------------------------


def test_refresh_token_hash_is_deterministic_and_not_the_token() -> None:
    raw, hashed = generate_refresh_token()

    assert hashed == hash_refresh_token(raw)
    assert raw not in hashed
    assert len(hashed) == 64  # sha256 hex


def test_refresh_tokens_are_unique() -> None:
    tokens = {generate_refresh_token()[0] for _ in range(100)}

    assert len(tokens) == 100
