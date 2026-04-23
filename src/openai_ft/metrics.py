"""Evaluation metrics.

Metrics consume model predictions (strings) plus the original example to compute
aggregated scores. Subclass :class:`BaseMetric` for custom metrics.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

from .exceptions import EvaluationError

TargetFn = Callable[[Any], Any]
ParserFn = Callable[[str], Any]


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    A metric is stateful: call :meth:`update` for each prediction, then
    :meth:`compute` to obtain a mapping of metric name → value.
    """

    name: str = "metric"

    @abstractmethod
    def update(self, prediction: str, example: Any) -> None:
        """Accumulate statistics from a single prediction."""

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return the aggregated metric value(s)."""

    def reset(self) -> None:  # noqa: B027 - intentional default no-op
        """Reset internal state. Default: no-op; override if stateful."""


class ExactMatch(BaseMetric):
    """Case-insensitive exact-match accuracy.

    ``target_fn`` extracts the expected string from the example. Both sides are
    stripped and lowercased by default.
    """

    name = "exact_match"

    def __init__(
        self,
        target_fn: TargetFn,
        *,
        case_sensitive: bool = False,
        strip: bool = True,
    ) -> None:
        self._target_fn = target_fn
        self._case_sensitive = case_sensitive
        self._strip = strip
        self._correct = 0
        self._total = 0

    def _normalize(self, value: Any) -> str:
        s = str(value)
        if self._strip:
            s = s.strip()
        if not self._case_sensitive:
            s = s.lower()
        return s

    def update(self, prediction: str, example: Any) -> None:
        target = self._target_fn(example)
        self._total += 1
        if self._normalize(prediction) == self._normalize(target):
            self._correct += 1

    def compute(self) -> dict[str, float]:
        if self._total == 0:
            return {self.name: 0.0}
        return {self.name: self._correct / self._total}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


class Accuracy(ExactMatch):
    """Alias for :class:`ExactMatch` with a more familiar name."""

    name = "accuracy"


class NumericRegressionMetric(BaseMetric):
    """MAE / RMSE / RMSLE for numeric prediction tasks.

    The prediction text is parsed by ``parser_fn`` (default: extract the first
    floating-point number in the string). ``target_fn`` extracts the ground-truth
    numeric value from the example.

    If prediction parsing fails, the example counts as an error and is recorded
    in the ``parse_errors`` output field.
    """

    name = "regression"

    def __init__(
        self,
        target_fn: TargetFn,
        parser_fn: ParserFn | None = None,
    ) -> None:
        self._target_fn = target_fn
        self._parser_fn = parser_fn or _default_numeric_parser
        self._errors_abs: list[float] = []
        self._errors_sq: list[float] = []
        self._errors_log_sq: list[float] = []
        self._parse_errors = 0
        self._total = 0

    def update(self, prediction: str, example: Any) -> None:
        self._total += 1
        target = float(self._target_fn(example))
        try:
            pred = float(self._parser_fn(prediction))
        except (ValueError, TypeError):
            self._parse_errors += 1
            return

        diff = pred - target
        self._errors_abs.append(abs(diff))
        self._errors_sq.append(diff * diff)

        if pred > 0 and target > 0:
            log_diff = math.log1p(pred) - math.log1p(target)
            self._errors_log_sq.append(log_diff * log_diff)

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {
            "count": float(self._total),
            "parse_errors": float(self._parse_errors),
        }
        if self._errors_abs:
            result["mae"] = sum(self._errors_abs) / len(self._errors_abs)
            result["rmse"] = math.sqrt(sum(self._errors_sq) / len(self._errors_sq))
        if self._errors_log_sq:
            result["rmsle"] = math.sqrt(sum(self._errors_log_sq) / len(self._errors_log_sq))
        return result

    def reset(self) -> None:
        self._errors_abs.clear()
        self._errors_sq.clear()
        self._errors_log_sq.clear()
        self._parse_errors = 0
        self._total = 0


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _default_numeric_parser(text: str) -> float:
    """Extract the first number from ``text``. Handles ``$1,234.56`` etc."""
    match = _NUMBER_RE.search(text.replace(",", ""))
    if not match:
        raise ValueError(f"No numeric value found in {text!r}")
    return float(match.group(0))


class MetricSuite:
    """Run a collection of metrics over the same prediction stream."""

    def __init__(self, metrics: Iterable[BaseMetric]) -> None:
        self._metrics = list(metrics)
        if not self._metrics:
            raise EvaluationError("MetricSuite requires at least one metric")

    def update(self, prediction: str, example: Any) -> None:
        for metric in self._metrics:
            metric.update(prediction, example)

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for metric in self._metrics:
            scoped = metric.compute()
            for k, v in scoped.items():
                key = k if len(scoped) > 1 or k != metric.name else metric.name
                results[f"{metric.name}/{k}" if len(scoped) > 1 else key] = v
        return results

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()
