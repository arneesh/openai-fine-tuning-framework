"""Build OpenAI-compatible JSONL training files from examples + a formatter."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .exceptions import FormatterError
from .formatters import BaseFormatter
from .logging_utils import get_logger

log = get_logger("jsonl")


def build_training_records(
    examples: Iterable[Any],
    formatter: BaseFormatter,
) -> list[dict[str, Any]]:
    """Render examples into the ``{"messages": [...]}`` records expected by OpenAI."""
    records: list[dict[str, Any]] = []
    for i, example in enumerate(examples):
        try:
            messages = formatter.training_messages(example)
        except FormatterError:
            raise
        except Exception as exc:
            raise FormatterError(f"Failed to format example #{i}: {exc}") from exc
        records.append({"messages": messages})
    return records


def serialize_records(records: Iterable[dict[str, Any]]) -> str:
    """Serialize records into a JSONL string."""
    return "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)


def write_jsonl(
    examples: Iterable[Any],
    formatter: BaseFormatter,
    output_path: str | Path,
) -> Path:
    """Build and write JSONL for ``examples`` to ``output_path``.

    Creates parent directories as needed. Returns the resolved path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = build_training_records(examples, formatter)
    if not records:
        raise FormatterError(f"No records to write to {path}")

    payload = serialize_records(records)
    path.write_text(payload + "\n", encoding="utf-8")
    log.info("Wrote %d JSONL records to %s", len(records), path)
    return path
