"""Dataset loading utilities.

Supports loading examples from:

* Hugging Face datasets (requires the ``huggingface`` extra).
* Local JSONL files (one JSON object per line).
* Local JSON files (a single JSON array).
* In-memory Python objects (pass-through).

All loaders return plain Python lists of examples (usually dicts) so that
user-defined formatters can consume them without coupling to a particular
dataset library.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import DataConfig
from .exceptions import DataError
from .logging_utils import get_logger

log = get_logger("datasets")

Example = Any


def load_jsonl(path: str | Path) -> list[Example]:
    """Read a JSONL file and return a list of parsed objects."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"JSONL file not found: {p}")

    examples: list[Example] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                examples.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid JSON on {p}:{lineno}: {exc}") from exc
    return examples


def load_json_array(path: str | Path) -> list[Example]:
    """Read a JSON file that contains a top-level array."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"JSON file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Invalid JSON in {p}: {exc}") from exc
    if not isinstance(data, list):
        raise DataError(f"{p} must contain a JSON array at the top level")
    return data


def load_hf_dataset(
    name: str,
    train_split: str,
    val_split: str,
    test_split: str | None,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Load a Hugging Face dataset and return (train, val, test) lists."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataError(
            "datasets is not installed. Install the 'huggingface' extra: "
            "`uv sync --extra huggingface`"
        ) from exc

    log.info("Loading Hugging Face dataset %s", name)
    ds = load_dataset(name)
    train = [dict(row) for row in ds[train_split]]
    val = [dict(row) for row in ds[val_split]] if val_split in ds else []
    test = [dict(row) for row in ds[test_split]] if test_split and test_split in ds else []
    return train, val, test


def _truncate(seq: Sequence[Example], limit: int | None) -> list[Example]:
    if limit is None:
        return list(seq)
    return list(seq[:limit])


def load_from_config(
    config: DataConfig,
    *,
    in_memory: tuple[Sequence[Example], Sequence[Example], Sequence[Example]] | None = None,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Resolve a :class:`DataConfig` into concrete (train, val, test) lists.

    ``in_memory`` is only used when ``config.source == "python"``.
    """
    if config.source == "python":
        if in_memory is None:
            raise DataError(
                "DataConfig.source='python' requires in-memory examples "
                "to be passed to load_from_config()"
            )
        train, val, test = in_memory
    elif config.source == "hf":
        assert config.path is not None
        train, val, test = load_hf_dataset(
            config.path, config.train_split, config.val_split, config.test_split
        )
    elif config.source == "jsonl":
        assert config.train_path is not None
        train = load_jsonl(config.train_path)
        val = load_jsonl(config.val_path) if config.val_path else []
        test = load_jsonl(config.test_path) if config.test_path else []
    elif config.source == "json":
        assert config.train_path is not None
        train = load_json_array(config.train_path)
        val = load_json_array(config.val_path) if config.val_path else []
        test = load_json_array(config.test_path) if config.test_path else []
    else:  # pragma: no cover - exhaustiveness
        raise DataError(f"Unsupported data source: {config.source!r}")

    train = _truncate(train, config.max_train)
    val = _truncate(val, config.max_val)
    test = _truncate(test, config.max_test)
    log.info(
        "Loaded %d train / %d val / %d test examples", len(train), len(val), len(test)
    )
    return train, val, test
