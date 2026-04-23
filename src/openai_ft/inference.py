"""Inference helpers for calling fine-tuned chat models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import InferenceConfig
from .exceptions import InferenceError
from .formatters import BaseFormatter
from .logging_utils import get_logger

log = get_logger("inference")


class ChatPredictor:
    """Thin wrapper around ``client.chat.completions.create`` with retries.

    Use via :meth:`predict` for a single example or as a callable:

        predictor = ChatPredictor(client, formatter, model="ft:...", config=cfg)
        text = predictor(example)
    """

    def __init__(
        self,
        client: OpenAI,
        formatter: BaseFormatter,
        *,
        model: str,
        config: InferenceConfig | None = None,
    ) -> None:
        if not model:
            raise InferenceError("ChatPredictor requires a non-empty model id")
        self._client = client
        self._formatter = formatter
        self._model = model
        self._config = config or InferenceConfig()

        self._call_with_retry = retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(max(1, self._config.max_retries + 1)),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )(self._call_api)

    @property
    def model(self) -> str:
        return self._model

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "timeout": self._config.timeout,
        }
        if self._config.top_p is not None:
            kwargs["top_p"] = self._config.top_p

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content
        return content or ""

    def predict(self, example: Any) -> str:
        """Run a single prediction for ``example`` and return the model text."""
        messages = self._formatter.inference_messages(example)
        try:
            return self._call_with_retry(messages)
        except Exception as exc:
            raise InferenceError(f"Inference failed for example: {exc}") from exc

    def __call__(self, example: Any) -> str:
        return self.predict(example)


def make_predictor(
    client: OpenAI,
    formatter: BaseFormatter,
    model: str,
    config: InferenceConfig | None = None,
) -> Callable[[Any], str]:
    """Convenience factory returning a callable predictor."""
    return ChatPredictor(client, formatter, model=model, config=config)
