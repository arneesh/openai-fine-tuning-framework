"""Configuration models for the fine-tuning pipeline.

All user-provided configuration flows through validated Pydantic models.
Configs can be constructed programmatically or loaded from YAML / JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigError


class Settings(BaseSettings):
    """Runtime secrets and environment-level settings.

    Resolved from environment variables and ``.env`` files.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: str | None = Field(default=None, description="OpenAI API key.")
    openai_org_id: str | None = Field(default=None, description="Optional OpenAI organization id.")
    openai_project_id: str | None = Field(default=None, description="Optional OpenAI project id.")

    openai_ft_log_level: str = Field(default="INFO")
    openai_ft_artifacts_dir: Path = Field(default=Path("./artifacts"))


class HyperParameters(BaseModel):
    """Hyperparameters forwarded to ``openai.fine_tuning.jobs.create``.

    Any field left as ``"auto"`` (the OpenAI default) is omitted from the payload
    so the server can choose a sensible value.
    """

    model_config = ConfigDict(extra="forbid")

    n_epochs: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"
    learning_rate_multiplier: float | Literal["auto"] = "auto"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in ("n_epochs", "batch_size", "learning_rate_multiplier"):
            value = getattr(self, name)
            if value != "auto":
                payload[name] = value
        return payload


class DataConfig(BaseModel):
    """Where training/validation/test examples come from.

    Supports:

    * ``source="hf"`` -- a Hugging Face dataset (requires the ``huggingface``
      extra). ``path`` is the dataset repo name.
    * ``source="jsonl"`` / ``source="json"`` -- local files, one example per line
      or a JSON array. Use ``train_path``, ``val_path``, ``test_path``.
    * ``source="python"`` -- examples provided in-process to
      :class:`~openai_ft.pipeline.FineTuningPipeline`. No paths required.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["hf", "jsonl", "json", "python"] = "jsonl"
    path: str | None = None
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None
    max_train: int | None = Field(default=None, ge=1)
    max_val: int | None = Field(default=None, ge=1)
    max_test: int | None = Field(default=None, ge=1)


class ModelConfig(BaseModel):
    """Base model identifier and the suffix used for the fine-tuned variant."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = "gpt-4o-mini-2024-07-18"
    suffix: str | None = Field(default=None, max_length=40)
    seed: int | None = None


class InferenceConfig(BaseModel):
    """Settings used when calling the fine-tuned model for evaluation."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class TrainerConfig(BaseModel):
    """Controls for launching and polling the fine-tuning job."""

    model_config = ConfigDict(extra="forbid")

    poll_interval_seconds: float = Field(default=30.0, gt=0)
    wait_for_completion: bool = True
    wait_timeout_seconds: float | None = Field(default=None, gt=0)


class FineTuneConfig(BaseModel):
    """Top-level configuration for an end-to-end fine-tuning pipeline."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(default="openai-ft-run", min_length=1)
    artifacts_dir: Path = Field(default=Path("./artifacts"))

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    hyperparameters: HyperParameters = Field(default_factory=HyperParameters)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @field_validator("artifacts_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Any:
        return Path(v) if v is not None else v

    @model_validator(mode="after")
    def _validate_data_source(self) -> FineTuneConfig:
        if self.data.source == "hf" and not self.data.path:
            raise ValueError("data.path is required when data.source='hf'")
        if self.data.source in {"jsonl", "json"} and self.data.train_path is None:
            raise ValueError(
                "data.train_path is required when data.source is 'jsonl' or 'json'"
            )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> FineTuneConfig:
        """Load a config from a YAML or JSON file."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        text = p.read_text()
        if p.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        elif p.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raise ConfigError(
                f"Unsupported config extension {p.suffix!r}. Use .yaml, .yml, or .json"
            )
        if not isinstance(raw, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
        try:
            return cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"Invalid config in {p}: {exc}") from exc
