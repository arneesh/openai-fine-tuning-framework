"""Create and monitor OpenAI fine-tuning jobs."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from openai import OpenAI

from .config import HyperParameters, ModelConfig, TrainerConfig
from .exceptions import JobError
from .logging_utils import get_logger

log = get_logger("trainer")

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
SUCCESS_STATES = frozenset({"succeeded"})


def create_job(
    client: OpenAI,
    *,
    training_file_id: str,
    validation_file_id: str | None,
    model: ModelConfig,
    hyperparameters: HyperParameters,
) -> Any:
    """Create a fine-tuning job and return the initial job object."""
    payload: dict[str, Any] = {
        "training_file": training_file_id,
        "model": model.base_model,
    }
    if validation_file_id:
        payload["validation_file"] = validation_file_id
    if model.suffix:
        payload["suffix"] = model.suffix
    if model.seed is not None:
        payload["seed"] = model.seed

    hp_payload = hyperparameters.to_payload()
    if hp_payload:
        payload["hyperparameters"] = hp_payload

    log.info("Creating fine-tuning job on base model %s", model.base_model)
    job = client.fine_tuning.jobs.create(**payload)
    log.info("Created fine-tuning job %s (status=%s)", job.id, job.status)
    return job


def retrieve_job(client: OpenAI, job_id: str) -> Any:
    """Retrieve the latest job state."""
    return client.fine_tuning.jobs.retrieve(job_id)


def list_events(client: OpenAI, job_id: str, limit: int = 20) -> list[Any]:
    """Return recent events for a fine-tuning job (most recent first)."""
    return list(client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit).data)


def wait_for_completion(
    client: OpenAI,
    job_id: str,
    *,
    config: TrainerConfig,
) -> Any:
    """Block until the job reaches a terminal state.

    Returns the final job object. Raises :class:`JobError` on failure, cancel,
    or timeout.
    """
    start = time.monotonic()
    seen_events: set[str] = set()
    last_status: str | None = None

    while True:
        job = retrieve_job(client, job_id)
        status = job.status

        if status != last_status:
            log.info("Job %s -> %s", job_id, status)
            last_status = status

        for event in reversed(list_events(client, job_id, limit=20)):
            eid = getattr(event, "id", None)
            if eid and eid not in seen_events:
                seen_events.add(eid)
                msg = getattr(event, "message", "")
                level = getattr(event, "level", "info")
                log.info("[%s] %s", level, msg)

        if status in TERMINAL_STATES:
            if status not in SUCCESS_STATES:
                raise JobError(f"Fine-tuning job {job_id} ended with status={status}")
            return job

        if (
            config.wait_timeout_seconds is not None
            and (time.monotonic() - start) > config.wait_timeout_seconds
        ):
            raise JobError(
                f"Timed out after {config.wait_timeout_seconds}s waiting for job {job_id}"
            )

        time.sleep(config.poll_interval_seconds)


def list_recent_jobs(client: OpenAI, limit: int = 10) -> Iterable[Any]:
    """Return the ``limit`` most recent fine-tuning jobs."""
    return client.fine_tuning.jobs.list(limit=limit).data
