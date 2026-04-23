"""Centralized logging configuration using Rich for pretty terminal output."""

from __future__ import annotations

import logging
import os
from typing import Final

from rich.logging import RichHandler

_LOGGER_NAME: Final[str] = "openai_ft"
_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root framework logger.

    Safe to call multiple times; only the first call applies handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = level or os.environ.get("OPENAI_FT_LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False

    handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_path=False,
        show_time=True,
        log_time_format="[%X]",
    )
    handler.setLevel(resolved_level)
    logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger under the framework root."""
    configure_logging()
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
