"""Thin wrapper around :class:`openai.OpenAI` with env-aware construction."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from .config import Settings
from .exceptions import ConfigError


def build_openai_client(settings: Settings | None = None) -> OpenAI:
    """Construct an :class:`~openai.OpenAI` client from :class:`Settings`.

    Raises :class:`ConfigError` if no API key is available.
    """
    settings = settings or Settings()
    if not settings.openai_api_key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )
    return OpenAI(
        api_key=settings.openai_api_key,
        organization=settings.openai_org_id,
        project=settings.openai_project_id,
    )


@lru_cache(maxsize=1)
def default_client() -> OpenAI:
    """Return a process-wide cached default client.

    The first call reads :class:`Settings`; subsequent calls return the cached
    instance. Call :func:`reset_default_client` in tests to clear it.
    """
    return build_openai_client()


def reset_default_client() -> None:
    """Clear the cached default client (useful in tests)."""
    default_client.cache_clear()
