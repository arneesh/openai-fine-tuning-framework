"""High-level orchestrator that runs the entire fine-tuning pipeline.

Typical usage::

    pipeline = FineTuningPipeline(config=cfg, formatter=MyFormatter(), metrics=[Accuracy(...)])
    result = pipeline.run()
    print(result.fine_tuned_model, result.evaluation.metrics)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import datasets as datasets_mod
from . import jsonl as jsonl_mod
from . import trainer as trainer_mod
from .client import build_openai_client
from .config import FineTuneConfig
from .evaluator import EvaluationResult, Evaluator
from .exceptions import JobError
from .formatters import BaseFormatter
from .inference import ChatPredictor
from .logging_utils import get_logger
from .metrics import BaseMetric
from .uploader import upload_file

log = get_logger("pipeline")


@dataclass
class PipelineState:
    """Persistent run state written to ``<artifacts>/state.json``."""

    created_at: str
    project_name: str
    train_path: str | None = None
    val_path: str | None = None
    training_file_id: str | None = None
    validation_file_id: str | None = None
    job_id: str | None = None
    job_status: str | None = None
    fine_tuned_model: str | None = None
    evaluation: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "project_name": self.project_name,
            "train_path": self.train_path,
            "val_path": self.val_path,
            "training_file_id": self.training_file_id,
            "validation_file_id": self.validation_file_id,
            "job_id": self.job_id,
            "job_status": self.job_status,
            "fine_tuned_model": self.fine_tuned_model,
            "evaluation": self.evaluation,
        }


@dataclass
class PipelineResult:
    """Outcome of :meth:`FineTuningPipeline.run`."""

    fine_tuned_model: str | None
    job_id: str | None
    training_file_id: str | None
    validation_file_id: str | None
    evaluation: EvaluationResult | None
    state: PipelineState


class FineTuningPipeline:
    """End-to-end pipeline: prepare → upload → train → (evaluate)."""

    def __init__(
        self,
        config: FineTuneConfig,
        formatter: BaseFormatter,
        *,
        metrics: Sequence[BaseMetric] | None = None,
        client: OpenAI | None = None,
        in_memory_data: tuple[Sequence[Any], Sequence[Any], Sequence[Any]] | None = None,
    ) -> None:
        self._config = config
        self._formatter = formatter
        self._metrics = list(metrics or [])
        self._client = client or build_openai_client()
        self._in_memory_data = in_memory_data

        self._artifacts_dir = Path(config.artifacts_dir) / config.project_name
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._artifacts_dir / "state.json"
        self._state = PipelineState(
            created_at=datetime.now(timezone.utc).isoformat(),
            project_name=config.project_name,
        )

    @property
    def artifacts_dir(self) -> Path:
        return self._artifacts_dir

    @property
    def state(self) -> PipelineState:
        return self._state

    def _save_state(self) -> None:
        self._state_path.write_text(
            json.dumps(self._state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_data(self) -> tuple[list[Any], list[Any], list[Any]]:
        """Resolve :attr:`FineTuneConfig.data` into ``(train, val, test)`` lists."""
        return datasets_mod.load_from_config(self._config.data, in_memory=self._in_memory_data)

    def prepare(
        self,
        train: Iterable[Any],
        val: Iterable[Any] | None = None,
    ) -> tuple[Path, Path | None]:
        """Build JSONL training (and optional validation) files."""
        jsonl_dir = self._artifacts_dir / "jsonl"
        train_path = jsonl_mod.write_jsonl(train, self._formatter, jsonl_dir / "train.jsonl")
        val_path: Path | None = None
        val_list = list(val) if val is not None else []
        if val_list:
            val_path = jsonl_mod.write_jsonl(val_list, self._formatter, jsonl_dir / "validation.jsonl")

        self._state.train_path = str(train_path)
        self._state.val_path = str(val_path) if val_path else None
        self._save_state()
        return train_path, val_path

    def upload(self, train_path: Path, val_path: Path | None) -> tuple[str, str | None]:
        """Upload JSONL files to OpenAI and remember their ids."""
        train_file = upload_file(self._client, train_path, purpose="fine-tune")
        validation_id: str | None = None
        if val_path is not None:
            val_file = upload_file(self._client, val_path, purpose="fine-tune")
            validation_id = val_file.id

        self._state.training_file_id = train_file.id
        self._state.validation_file_id = validation_id
        self._save_state()
        return train_file.id, validation_id

    def train(self, training_file_id: str, validation_file_id: str | None) -> Any:
        """Launch a fine-tuning job. Optionally waits for completion."""
        job = trainer_mod.create_job(
            self._client,
            training_file_id=training_file_id,
            validation_file_id=validation_file_id,
            model=self._config.model,
            hyperparameters=self._config.hyperparameters,
        )
        self._state.job_id = job.id
        self._state.job_status = job.status
        self._save_state()

        if self._config.trainer.wait_for_completion:
            job = trainer_mod.wait_for_completion(
                self._client, job.id, config=self._config.trainer
            )
            self._state.job_status = job.status
            ft_model = getattr(job, "fine_tuned_model", None)
            if not ft_model:
                raise JobError(f"Job {job.id} finished but has no fine_tuned_model field")
            self._state.fine_tuned_model = ft_model
            self._save_state()

        return job

    def evaluate(
        self,
        test: Iterable[Any],
        *,
        model: str | None = None,
    ) -> EvaluationResult:
        """Evaluate the fine-tuned model against ``test`` using configured metrics."""
        if not self._metrics:
            raise JobError(
                "Cannot evaluate: no metrics were supplied when constructing the pipeline"
            )
        model_id = model or self._state.fine_tuned_model
        if not model_id:
            raise JobError(
                "Cannot evaluate: no fine-tuned model id available. "
                "Pass model=... or run training with wait_for_completion=True."
            )

        predictor = ChatPredictor(
            self._client,
            self._formatter,
            model=model_id,
            config=self._config.inference,
        )
        evaluator = Evaluator(
            predictor,
            metrics=self._metrics,
            store_predictions=True,
            continue_on_error=True,
        )
        result = evaluator.evaluate(test)
        result.save(self._artifacts_dir / "evaluation.json")

        self._state.evaluation = {
            "metrics": result.metrics,
            "num_examples": result.num_examples,
            "num_errors": result.num_errors,
            "elapsed_seconds": result.elapsed_seconds,
        }
        self._save_state()
        return result

    def run(self) -> PipelineResult:
        """Execute the full pipeline end-to-end."""
        log.info("=== Running pipeline '%s' ===", self._config.project_name)
        train, val, test = self.load_data()

        train_path, val_path = self.prepare(train, val)
        training_file_id, validation_file_id = self.upload(train_path, val_path)
        self.train(training_file_id, validation_file_id)

        evaluation: EvaluationResult | None = None
        if test and self._metrics and self._state.fine_tuned_model:
            evaluation = self.evaluate(test)

        return PipelineResult(
            fine_tuned_model=self._state.fine_tuned_model,
            job_id=self._state.job_id,
            training_file_id=self._state.training_file_id,
            validation_file_id=self._state.validation_file_id,
            evaluation=evaluation,
            state=self._state,
        )
