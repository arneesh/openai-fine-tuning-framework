"""Formatters convert raw user examples into OpenAI chat-format messages.

A "formatter" is the piece of logic that knows how to turn a single domain
example (a dict, dataclass, pydantic model, etc.) into:

* **training messages** -- a full conversation including the ground-truth
  assistant response, written to JSONL for fine-tuning.
* **inference messages** -- a conversation that stops *before* the assistant
  turn, used when calling the fine-tuned model.

Users subclass :class:`BaseFormatter` and implement :meth:`user_message` and
:meth:`assistant_message`. Optionally override :meth:`system_message` to add
a system prompt.

For simple cases, :func:`formatter_from_functions` lets you build a formatter
from plain callables without subclassing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .exceptions import FormatterError

Message = dict[str, str]


class BaseFormatter(ABC):
    """Abstract base class for example-to-messages formatters.

    Subclasses must implement :meth:`user_message` and :meth:`assistant_message`.
    Override :meth:`system_message` to add an optional system prompt.
    """

    @abstractmethod
    def user_message(self, example: Any) -> str:
        """Return the user-turn text for ``example``."""

    @abstractmethod
    def assistant_message(self, example: Any) -> str:
        """Return the assistant-turn text (ground-truth) for ``example``."""

    def system_message(self, example: Any) -> str | None:  # noqa: ARG002
        """Return an optional system prompt. Return ``None`` to omit."""
        return None

    def training_messages(self, example: Any) -> list[Message]:
        """Build the full training conversation for ``example``."""
        messages: list[Message] = []
        sys = self.system_message(example)
        if sys is not None:
            if not isinstance(sys, str) or not sys.strip():
                raise FormatterError("system_message must return a non-empty string or None")
            messages.append({"role": "system", "content": sys})

        user = self.user_message(example)
        if not isinstance(user, str) or not user.strip():
            raise FormatterError("user_message must return a non-empty string")
        messages.append({"role": "user", "content": user})

        assistant = self.assistant_message(example)
        if not isinstance(assistant, str) or not assistant.strip():
            raise FormatterError("assistant_message must return a non-empty string")
        messages.append({"role": "assistant", "content": assistant})
        return messages

    def inference_messages(self, example: Any) -> list[Message]:
        """Build a conversation without the assistant turn, for inference."""
        messages: list[Message] = []
        sys = self.system_message(example)
        if sys is not None:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": self.user_message(example)})
        return messages


class TemplateFormatter(BaseFormatter):
    """A simple formatter driven by ``str.format``-style templates.

    Example::

        fmt = TemplateFormatter(
            user_template="Translate to French: {text}",
            assistant_template="{translation}",
            system_template="You are a professional translator.",
        )

    Each template is ``str.format(**example)``-formatted; ``example`` must be a
    mapping.
    """

    def __init__(
        self,
        user_template: str,
        assistant_template: str,
        system_template: str | None = None,
    ) -> None:
        self._user_template = user_template
        self._assistant_template = assistant_template
        self._system_template = system_template

    @staticmethod
    def _as_mapping(example: Any) -> dict[str, Any]:
        if isinstance(example, dict):
            return example
        if hasattr(example, "model_dump"):
            dumped: dict[str, Any] = example.model_dump()
            return dumped
        if hasattr(example, "__dict__"):
            return dict(vars(example))
        raise FormatterError(
            f"TemplateFormatter requires mapping-like examples, got {type(example).__name__}"
        )

    def _render(self, template: str, example: Any) -> str:
        try:
            return template.format(**self._as_mapping(example))
        except KeyError as exc:
            raise FormatterError(f"Template placeholder {exc} missing in example") from exc

    def user_message(self, example: Any) -> str:
        return self._render(self._user_template, example)

    def assistant_message(self, example: Any) -> str:
        return self._render(self._assistant_template, example)

    def system_message(self, example: Any) -> str | None:
        if self._system_template is None:
            return None
        return self._render(self._system_template, example)


def formatter_from_functions(
    user_fn: Callable[[Any], str],
    assistant_fn: Callable[[Any], str],
    system_fn: Callable[[Any], str | None] | None = None,
) -> BaseFormatter:
    """Adapt plain callables into a :class:`BaseFormatter`."""

    class _FunctionFormatter(BaseFormatter):
        def user_message(self, example: Any) -> str:
            return user_fn(example)

        def assistant_message(self, example: Any) -> str:
            return assistant_fn(example)

        def system_message(self, example: Any) -> str | None:
            return system_fn(example) if system_fn is not None else None

    return _FunctionFormatter()
