"""Upload local JSONL files to the OpenAI Files API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .exceptions import JobError
from .logging_utils import get_logger

log = get_logger("uploader")

FilePurpose = Literal["assistants", "batch", "fine-tune", "vision", "user_data", "evals"]


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def upload_file(
    client: OpenAI,
    path: str | Path,
    *,
    purpose: FilePurpose = "fine-tune",
) -> Any:
    """Upload a single file and return the OpenAI ``FileObject``."""
    p = Path(path)
    if not p.exists():
        raise JobError(f"Cannot upload missing file: {p}")

    log.info("Uploading %s (%.2f KB) to OpenAI as purpose=%s", p, p.stat().st_size / 1024, purpose)
    with p.open("rb") as f:
        file_obj = client.files.create(file=f, purpose=purpose)
    log.info("Uploaded %s -> file id %s", p.name, file_obj.id)
    return file_obj
