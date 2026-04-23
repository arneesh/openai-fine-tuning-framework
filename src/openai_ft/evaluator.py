"""Run a prediction function over a dataset and aggregate metrics."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .exceptions import EvaluationError
from .logging_utils import get_logger
from .metrics import BaseMetric, MetricSuite

log = get_logger("evaluator")

PredictFn = Callable[[Any], str]


@dataclass
class EvaluationResult:
    """Result of an evaluation pass."""

    metrics: dict[str, float]
    predictions: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    num_examples: int = 0
    num_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "elapsed_seconds": self.elapsed_seconds,
            "num_examples": self.num_examples,
            "num_errors": self.num_errors,
            "predictions": self.predictions,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p


class Evaluator:
    """Iterate over examples, collect predictions, and compute metrics."""

    def __init__(
        self,
        predict_fn: PredictFn,
        metrics: Iterable[BaseMetric],
        *,
        store_predictions: bool = True,
        continue_on_error: bool = True,
    ) -> None:
        self._predict_fn = predict_fn
        self._suite = MetricSuite(metrics)
        self._store_predictions = store_predictions
        self._continue_on_error = continue_on_error

    def evaluate(
        self,
        examples: Iterable[Any],
        *,
        show_progress: bool = True,
        console: Console | None = None,
    ) -> EvaluationResult:
        """Run the predictor over ``examples`` and return an :class:`EvaluationResult`."""
        self._suite.reset()
        examples_list = list(examples)
        if not examples_list:
            raise EvaluationError("Cannot evaluate: received zero examples")

        predictions: list[dict[str, Any]] = []
        errors = 0
        start = time.monotonic()

        console = console or Console()
        progress_cm: Any
        if show_progress:
            progress_cm = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
        else:
            progress_cm = _NullProgress()

        with progress_cm as progress:
            task_id = progress.add_task("Evaluating", total=len(examples_list))
            for idx, example in enumerate(examples_list):
                try:
                    pred = self._predict_fn(example)
                except Exception as exc:
                    errors += 1
                    log.warning("Prediction failed for example %d: %s", idx, exc)
                    if not self._continue_on_error:
                        raise EvaluationError(f"Prediction failed: {exc}") from exc
                    progress.advance(task_id)
                    continue

                try:
                    self._suite.update(pred, example)
                except Exception as exc:
                    errors += 1
                    log.warning("Metric update failed for example %d: %s", idx, exc)
                    if not self._continue_on_error:
                        raise EvaluationError(f"Metric update failed: {exc}") from exc

                if self._store_predictions:
                    predictions.append({"index": idx, "prediction": pred, "example": example})
                progress.advance(task_id)

        elapsed = time.monotonic() - start
        metrics = self._suite.compute()
        log.info("Evaluation complete in %.1fs: %s", elapsed, metrics)

        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            elapsed_seconds=elapsed,
            num_examples=len(examples_list),
            num_errors=errors,
        )


class _NullProgress:
    """No-op progress context used when ``show_progress=False``."""

    def __enter__(self) -> _NullProgress:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def add_task(self, *_args: Any, **_kwargs: Any) -> int:
        return 0

    def advance(self, *_args: Any, **_kwargs: Any) -> None:
        return None
