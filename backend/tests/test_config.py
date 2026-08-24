"""Configuration parsing and validation.

Note on test design: passing values as init kwargs (`Settings(foo=...)`)
exercises a *different* code path from loading them out of a .env file.
pydantic-settings JSON-decodes "complex" types like `list[str]` on the env
path only. An earlier version of these tests used init kwargs exclusively and
happily passed while the app crashed at import against a real .env — so the
env-file tests below are the ones that matter.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def write_env(tmp_path: Path, body: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


def test_cors_origins_parsed_from_comma_separated_string() -> None:
    settings = Settings(cors_origins="http://a.com, http://b.com")  # type: ignore[arg-type]

    assert settings.cors_origins == ["http://a.com", "http://b.com"]


def test_blank_cors_entries_are_dropped() -> None:
    settings = Settings(cors_origins="http://a.com,,  ,http://b.com")  # type: ignore[arg-type]

    assert settings.cors_origins == ["http://a.com", "http://b.com"]


def test_cors_origins_loaded_from_env_file_are_not_json_decoded(tmp_path: Path) -> None:
    """Regression: a bare comma-separated list is not valid JSON.

    Without `NoDecode` on the field, pydantic-settings calls json.loads() on
    this value before any validator runs, and the app dies at import.
    """
    env_file = write_env(tmp_path, "CORS_ORIGINS=http://a.com,http://b.com\n")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.cors_origins == ["http://a.com", "http://b.com"]


def test_env_file_inline_comments_are_stripped(tmp_path: Path) -> None:
    """`.env.example` documents values with trailing `# ...` comments."""
    env_file = write_env(
        tmp_path,
        "APP_ENV=development   # development | staging | production\n"
        "LOG_LEVEL=INFO        # DEBUG | INFO | WARNING\n",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"


def test_full_env_example_template_loads(tmp_path: Path) -> None:
    """The committed template must actually boot the app.

    A broken `.env.example` means every new developer's first experience is a
    stack trace.
    """
    template = Path(__file__).resolve().parents[2] / ".env.example"
    env_file = write_env(tmp_path, template.read_text(encoding="utf-8"))

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.app_name == "Atlas AI"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]
    assert settings.database_url.startswith("postgresql+asyncpg://")


def test_log_level_is_normalised_to_uppercase() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"


def test_invalid_log_level_fails_fast() -> None:
    """Bad config should crash at startup, not at the first log call."""
    with pytest.raises(ValidationError):
        Settings(log_level="verbose")


def test_invalid_environment_fails_fast() -> None:
    with pytest.raises(ValidationError):
        Settings(app_env="prod")  # type: ignore[arg-type]


def test_async_and_sync_dsns_use_different_drivers() -> None:
    settings = Settings(
        postgres_user="atlas",
        postgres_password="secret",
        postgres_host="postgres",
        postgres_port=5432,
        postgres_db="atlas",
    )

    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.sync_database_url.startswith("postgresql+psycopg://")
    assert settings.database_url.endswith("/atlas")


#: The minimum a production configuration must supply. Anything less is
#: rejected at construction — see `_refuse_unsafe_production_settings`.
SAFE_PRODUCTION = {
    "app_env": "production",
    "secret_key": "a" * 64,
    "refresh_cookie_secure": True,
    "debug": False,
}


def test_is_production_flag() -> None:
    assert Settings(**SAFE_PRODUCTION).is_production is True
    assert Settings(app_env="development").is_production is False


def test_production_refuses_the_default_signing_key() -> None:
    """Shipping the development key means anyone with the source can forge a
    token for any account. The app must not start rather than serve that."""
    unsafe = {**SAFE_PRODUCTION, "secret_key": "insecure-development-key"}

    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(**unsafe)


def test_production_refuses_an_insecure_refresh_cookie() -> None:
    with pytest.raises(ValidationError, match="REFRESH_COOKIE_SECURE"):
        Settings(**{**SAFE_PRODUCTION, "refresh_cookie_secure": False})


def test_production_refuses_debug_mode() -> None:
    with pytest.raises(ValidationError, match="DEBUG"):
        Settings(**{**SAFE_PRODUCTION, "debug": True})


def test_every_problem_is_reported_at_once() -> None:
    """Reporting one at a time turns a misconfigured deploy into several
    failed attempts, each revealing the next thing that was already wrong."""
    with pytest.raises(ValidationError) as caught:
        Settings(
            app_env="production",
            secret_key="insecure-development-key",
            refresh_cookie_secure=False,
            debug=True,
        )

    message = str(caught.value)
    assert "SECRET_KEY" in message
    assert "REFRESH_COOKIE_SECURE" in message
    assert "DEBUG" in message


def test_development_is_left_alone() -> None:
    """The same settings that are refused in production are the correct
    defaults locally, and must not be second-guessed there."""
    settings = Settings(app_env="development", debug=True, refresh_cookie_secure=False)

    assert settings.debug is True
