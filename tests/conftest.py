"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Return a clean temporary project directory."""
    return tmp_path


@pytest.fixture()
def sample_examples() -> list[dict[str, str]]:
    """Small deterministic dataset used across tests."""
    return [
        {"text": "I love this product!", "label": "positive"},
        {"text": "Terrible experience.", "label": "negative"},
        {"text": "It was fine, nothing special.", "label": "neutral"},
    ]
