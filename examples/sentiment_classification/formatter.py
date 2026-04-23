"""Formatter + metric for a simple sentiment-classification task.

Each example is a dict with ``text`` and ``label`` (``"positive"`` |
``"negative"`` | ``"neutral"``).
"""

from __future__ import annotations

from typing import Any

from openai_ft import Accuracy, BaseFormatter

_ALLOWED = {"positive", "negative", "neutral"}


class SentimentFormatter(BaseFormatter):
    def system_message(self, example: Any) -> str:  # noqa: ARG002
        return (
            "You are a sentiment classifier. Respond with exactly one word: "
            "positive, negative, or neutral."
        )

    def user_message(self, example: Any) -> str:
        return f"Classify the sentiment of this text:\n\n{example['text']}"

    def assistant_message(self, example: Any) -> str:
        label = str(example["label"]).strip().lower()
        if label not in _ALLOWED:
            raise ValueError(f"Unknown label {label!r}; expected one of {_ALLOWED}")
        return label


def sentiment_accuracy() -> Accuracy:
    return Accuracy(target_fn=lambda e: e["label"])
