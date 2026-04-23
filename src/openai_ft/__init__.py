"""openai-fine-tuning-framework

A generic, production-grade toolkit for fine-tuning OpenAI chat models.
"""

from __future__ import annotations

from .client import build_openai_client, default_client
from .config import (
    DataConfig,
    FineTuneConfig,
    HyperParameters,
    InferenceConfig,
    ModelConfig,
    Settings,
    TrainerConfig,
)
from .evaluator import EvaluationResult, Evaluator
from .exceptions import (
    ConfigError,
    DataError,
    EvaluationError,
    FormatterError,
    InferenceError,
    JobError,
    OpenAIFineTuneError,
)
from .formatters import BaseFormatter, TemplateFormatter, formatter_from_functions
from .inference import ChatPredictor, make_predictor
from .metrics import (
    Accuracy,
    BaseMetric,
    ExactMatch,
    MetricSuite,
    NumericRegressionMetric,
)
from .pipeline import FineTuningPipeline, PipelineResult, PipelineState

__all__ = [
    "Accuracy",
    "BaseFormatter",
    "BaseMetric",
    "ChatPredictor",
    "ConfigError",
    "DataConfig",
    "DataError",
    "EvaluationError",
    "EvaluationResult",
    "Evaluator",
    "ExactMatch",
    "FineTuneConfig",
    "FineTuningPipeline",
    "FormatterError",
    "HyperParameters",
    "InferenceConfig",
    "InferenceError",
    "JobError",
    "MetricSuite",
    "ModelConfig",
    "NumericRegressionMetric",
    "OpenAIFineTuneError",
    "PipelineResult",
    "PipelineState",
    "Settings",
    "TemplateFormatter",
    "TrainerConfig",
    "build_openai_client",
    "default_client",
    "formatter_from_functions",
    "make_predictor",
]

__version__ = "0.1.0"
