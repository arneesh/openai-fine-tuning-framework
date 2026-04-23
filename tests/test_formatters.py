"""Tests for formatters."""

from __future__ import annotations

import pytest

from openai_ft.exceptions import FormatterError
from openai_ft.formatters import BaseFormatter, TemplateFormatter, formatter_from_functions


class _SentimentFormatter(BaseFormatter):
    def system_message(self, example):
        return "You are a sentiment classifier."

    def user_message(self, example):
        return f"Classify: {example['text']}"

    def assistant_message(self, example):
        return example["label"]


def test_base_formatter_builds_training_messages(sample_examples):
    fmt = _SentimentFormatter()
    msgs = fmt.training_messages(sample_examples[0])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "Classify" in msgs[1]["content"]
    assert msgs[2]["content"] == "positive"


def test_base_formatter_builds_inference_messages(sample_examples):
    fmt = _SentimentFormatter()
    msgs = fmt.inference_messages(sample_examples[0])
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_base_formatter_without_system_message(sample_examples):
    class NoSystem(_SentimentFormatter):
        def system_message(self, example):
            return None

    msgs = NoSystem().training_messages(sample_examples[0])
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_base_formatter_rejects_empty_user_content():
    class BadFormatter(_SentimentFormatter):
        def user_message(self, example):
            return "  "

    with pytest.raises(FormatterError):
        BadFormatter().training_messages({"text": "x", "label": "y"})


def test_template_formatter(sample_examples):
    fmt = TemplateFormatter(
        user_template="Text: {text}",
        assistant_template="{label}",
        system_template="Be concise.",
    )
    msgs = fmt.training_messages(sample_examples[1])
    assert msgs[0]["content"] == "Be concise."
    assert msgs[1]["content"] == "Text: Terrible experience."
    assert msgs[2]["content"] == "negative"


def test_template_formatter_missing_placeholder_raises():
    fmt = TemplateFormatter(user_template="{missing}", assistant_template="x")
    with pytest.raises(FormatterError):
        fmt.training_messages({"text": "hi", "label": "pos"})


def test_formatter_from_functions(sample_examples):
    fmt = formatter_from_functions(
        user_fn=lambda e: f"u:{e['text']}",
        assistant_fn=lambda e: e["label"],
        system_fn=lambda e: None,
    )
    msgs = fmt.training_messages(sample_examples[0])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"].startswith("u:")
