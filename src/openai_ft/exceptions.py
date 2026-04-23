"""Custom exceptions for the framework."""

from __future__ import annotations


class OpenAIFineTuneError(Exception):
    """Base exception for all framework errors."""


class ConfigError(OpenAIFineTuneError):
    """Raised when configuration is invalid or missing."""


class DataError(OpenAIFineTuneError):
    """Raised when a dataset or example cannot be loaded or formatted."""


class FormatterError(OpenAIFineTuneError):
    """Raised when a formatter produces invalid output."""


class JobError(OpenAIFineTuneError):
    """Raised when a fine-tuning job fails or enters an unexpected state."""


class InferenceError(OpenAIFineTuneError):
    """Raised when inference against a model fails."""


class EvaluationError(OpenAIFineTuneError):
    """Raised when evaluation cannot be completed."""
