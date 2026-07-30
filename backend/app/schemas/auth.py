"""Authentication request and response contracts."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.security import PASSWORD_MAX_BYTES

# 8 is a floor, not a recommendation. Length beats complexity rules: NIST
# SP 800-63B explicitly advises against forced character-class requirements,
# which push users toward "Password1!" and away from long passphrases.
PASSWORD_MIN_LENGTH = 8


class _PasswordMixin:
    """Shared password validation."""

    @field_validator("password", check_fields=False)
    @classmethod
    def _validate_password_bytes(cls, value: str) -> str:
        # Length in *bytes*, not characters: bcrypt's limit is bytes, and a
        # single emoji costs four of them.
        encoded = value.encode("utf-8")
        if len(encoded) > PASSWORD_MAX_BYTES:
            raise ValueError(
                f"Password must be at most {PASSWORD_MAX_BYTES} bytes "
                "(non-ASCII characters count as more than one)."
            )
        return value


class RegisterRequest(_PasswordMixin, BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=PASSWORD_MIN_LENGTH)
    full_name: str | None = Field(None, max_length=255)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        """Lowercase so `Hiya@x.com` and `hiya@x.com` are one account."""
        return value.strip().lower()


class LoginRequest(_PasswordMixin, BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


class UserResponse(BaseModel):
    """A user as seen by the outside world.

    `from_attributes` lets this be built straight from an ORM object. Note
    that `hashed_password` is absent - the response model is what stops it
    leaking, which is precisely why responses are not just serialised models.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """Issued on register, login, and refresh.

    The refresh token is deliberately NOT in this body - it travels in an
    httpOnly cookie so that JavaScript, and therefore any XSS payload, cannot
    read it.
    """

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse
