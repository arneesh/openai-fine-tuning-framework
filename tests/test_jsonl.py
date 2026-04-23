"""Tests for JSONL builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_ft.exceptions import FormatterError
from openai_ft.formatters import BaseFormatter
from openai_ft.jsonl import build_training_records, serialize_records, write_jsonl


class _SimpleFormatter(BaseFormatter):
    def user_message(self, example):
        return f"Q: {example['q']}"

    def assistant_message(self, example):
        return example["a"]


def test_build_training_records_produces_messages_key():
    records = build_training_records(
        [{"q": "hi", "a": "hello"}, {"q": "bye", "a": "goodbye"}],
        _SimpleFormatter(),
    )
    assert len(records) == 2
    for rec in records:
        assert set(rec.keys()) == {"messages"}
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["user", "assistant"]


def test_serialize_records_one_per_line():
    records = build_training_records([{"q": "hi", "a": "hello"}], _SimpleFormatter())
    payload = serialize_records(records)
    lines = payload.splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["messages"][0]["content"] == "Q: hi"


def test_write_jsonl_round_trip(tmp_path: Path):
    out = tmp_path / "sub" / "train.jsonl"
    path = write_jsonl(
        [{"q": "hi", "a": "hello"}, {"q": "bye", "a": "goodbye"}],
        _SimpleFormatter(),
        out,
    )
    assert path == out
    contents = path.read_text().strip().splitlines()
    assert len(contents) == 2
    assert json.loads(contents[0])["messages"][0]["content"] == "Q: hi"


def test_build_training_records_bubbles_formatter_error():
    class Bad(_SimpleFormatter):
        def user_message(self, example):
            return ""

    with pytest.raises(FormatterError):
        build_training_records([{"q": "x", "a": "y"}], Bad())


def test_write_jsonl_rejects_empty_input(tmp_path: Path):
    with pytest.raises(FormatterError):
        write_jsonl([], _SimpleFormatter(), tmp_path / "empty.jsonl")
