"""Typer-based command-line interface for the framework.

The CLI is intentionally thin: it wires up :class:`FineTuningPipeline` and
exposes individual pipeline stages as subcommands. Users point commands at a
YAML/JSON :class:`FineTuneConfig` and tell them where their formatter / metrics
live using dotted import paths (e.g. ``mypkg.formatters:MyFormatter``).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from . import trainer as trainer_mod
from .client import build_openai_client
from .config import FineTuneConfig
from .evaluator import Evaluator
from .exceptions import ConfigError, OpenAIFineTuneError
from .formatters import BaseFormatter
from .inference import ChatPredictor
from .logging_utils import configure_logging, get_logger
from .metrics import BaseMetric
from .pipeline import FineTuningPipeline

app = typer.Typer(
    name="openai-ft",
    help="Production-grade framework for fine-tuning OpenAI chat models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
log = get_logger("cli")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"openai-fine-tuning-framework {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    configure_logging(log_level)


def _import_object(dotted: str) -> Any:
    """Import ``pkg.module:attr`` or ``pkg.module.attr`` and return the attribute."""
    if ":" in dotted:
        module_path, attr = dotted.split(":", 1)
    elif "." in dotted:
        module_path, _, attr = dotted.rpartition(".")
    else:
        raise ConfigError(
            f"Cannot import {dotted!r}: expected 'package.module:attr' or 'package.module.attr'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigError(f"Could not import module {module_path!r}: {exc}") from exc
    if not hasattr(module, attr):
        raise ConfigError(f"Module {module_path!r} has no attribute {attr!r}")
    return getattr(module, attr)


def _resolve_formatter(spec: str) -> BaseFormatter:
    obj = _import_object(spec)
    if callable(obj) and not isinstance(obj, BaseFormatter):
        obj = obj()
    if not isinstance(obj, BaseFormatter):
        raise ConfigError(f"{spec!r} did not resolve to a BaseFormatter instance")
    return obj


def _resolve_metrics(specs: list[str]) -> list[BaseMetric]:
    metrics: list[BaseMetric] = []
    for spec in specs:
        obj = _import_object(spec)
        if callable(obj) and not isinstance(obj, BaseMetric):
            obj = obj()
        if not isinstance(obj, BaseMetric):
            raise ConfigError(f"{spec!r} did not resolve to a BaseMetric instance")
        metrics.append(obj)
    return metrics


def _load_config(path: Path) -> FineTuneConfig:
    log.info("Loading config from %s", path)
    return FineTuneConfig.from_file(path)


def _handle(exc: BaseException) -> None:
    """Convert framework exceptions into a nice CLI error and exit 1."""
    if isinstance(exc, OpenAIFineTuneError):
        console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    raise exc


@app.command("run")
def run_cmd(
    config_path: Path = typer.Argument(..., exists=True, readable=True, help="Config YAML/JSON."),
    formatter: str = typer.Option(
        ...,
        "--formatter",
        "-f",
        help="Dotted path to a BaseFormatter subclass or instance (e.g. 'mypkg.fmt:MyFormatter').",
    ),
    metric: list[str] = typer.Option(
        [],
        "--metric",
        "-m",
        help="Dotted path to a BaseMetric (repeat to add multiple).",
    ),
) -> None:
    """Run the full pipeline: prepare → upload → train → evaluate."""
    try:
        config = _load_config(config_path)
        fmt = _resolve_formatter(formatter)
        metrics = _resolve_metrics(metric)
        pipeline = FineTuningPipeline(config, fmt, metrics=metrics)
        result = pipeline.run()
    except Exception as exc:
        _handle(exc)
        return

    console.print()
    table = Table(title="Pipeline Result", show_header=False)
    table.add_row("Job id", result.job_id or "-")
    table.add_row("Fine-tuned model", result.fine_tuned_model or "-")
    table.add_row("Training file id", result.training_file_id or "-")
    table.add_row("Validation file id", result.validation_file_id or "-")
    if result.evaluation:
        table.add_row("Evaluation metrics", json.dumps(result.evaluation.metrics, indent=2))
    console.print(table)
    console.print(f"\nArtifacts: [cyan]{pipeline.artifacts_dir}[/cyan]")


@app.command("prepare")
def prepare_cmd(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    formatter: str = typer.Option(..., "--formatter", "-f"),
) -> None:
    """Load data and write JSONL files without uploading or training."""
    try:
        config = _load_config(config_path)
        fmt = _resolve_formatter(formatter)
        pipeline = FineTuningPipeline(config, fmt)
        train, val, _ = pipeline.load_data()
        train_path, val_path = pipeline.prepare(train, val)
    except Exception as exc:
        _handle(exc)
        return
    console.print(f"[green]Wrote[/green] {train_path}")
    if val_path:
        console.print(f"[green]Wrote[/green] {val_path}")


@app.command("upload")
def upload_cmd(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    formatter: str = typer.Option(..., "--formatter", "-f"),
) -> None:
    """Upload previously-prepared JSONL files to OpenAI."""
    try:
        config = _load_config(config_path)
        fmt = _resolve_formatter(formatter)
        pipeline = FineTuningPipeline(config, fmt)
        train_path = pipeline.artifacts_dir / "jsonl" / "train.jsonl"
        val_path = pipeline.artifacts_dir / "jsonl" / "validation.jsonl"
        if not train_path.exists():
            raise ConfigError(f"Run `openai-ft prepare` first; missing {train_path}")
        train_id, val_id = pipeline.upload(
            train_path, val_path if val_path.exists() else None
        )
    except Exception as exc:
        _handle(exc)
        return
    console.print(f"[green]training_file_id[/green]  = {train_id}")
    if val_id:
        console.print(f"[green]validation_file_id[/green] = {val_id}")


@app.command("train")
def train_cmd(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    formatter: str = typer.Option(..., "--formatter", "-f"),
    training_file_id: Optional[str] = typer.Option(None, "--training-file-id"),
    validation_file_id: Optional[str] = typer.Option(None, "--validation-file-id"),
) -> None:
    """Launch a fine-tuning job (optionally waits for completion)."""
    try:
        config = _load_config(config_path)
        fmt = _resolve_formatter(formatter)
        pipeline = FineTuningPipeline(config, fmt)
        tf = training_file_id or pipeline.state.training_file_id
        vf = validation_file_id or pipeline.state.validation_file_id
        if not tf:
            raise ConfigError(
                "No training_file_id known. Run `openai-ft upload` first or pass --training-file-id."
            )
        job = pipeline.train(tf, vf)
    except Exception as exc:
        _handle(exc)
        return
    console.print(f"[green]job_id[/green]            = {job.id}")
    console.print(f"[green]status[/green]            = {job.status}")
    if getattr(job, "fine_tuned_model", None):
        console.print(f"[green]fine_tuned_model[/green]  = {job.fine_tuned_model}")


@app.command("status")
def status_cmd(
    job_id: str = typer.Argument(..., help="Fine-tuning job id."),
    events: int = typer.Option(10, "--events", "-n", help="Number of recent events to show."),
) -> None:
    """Retrieve the status of a fine-tuning job."""
    try:
        client = build_openai_client()
        job = trainer_mod.retrieve_job(client, job_id)
        event_list = trainer_mod.list_events(client, job_id, limit=events)
    except Exception as exc:
        _handle(exc)
        return

    console.print(f"[bold]Job[/bold] {job.id}: status=[cyan]{job.status}[/cyan]")
    if getattr(job, "fine_tuned_model", None):
        console.print(f"fine_tuned_model = [green]{job.fine_tuned_model}[/green]")
    if event_list:
        console.print("\n[bold]Recent events:[/bold]")
        for event in event_list:
            level = getattr(event, "level", "info")
            msg = getattr(event, "message", "")
            console.print(f"  [{level}] {msg}")


@app.command("evaluate")
def evaluate_cmd(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    formatter: str = typer.Option(..., "--formatter", "-f"),
    metric: list[str] = typer.Option(..., "--metric", "-m"),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override fine-tuned model id (defaults to saved state)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to write evaluation JSON."
    ),
) -> None:
    """Evaluate a fine-tuned model on the configured test split."""
    try:
        config = _load_config(config_path)
        fmt = _resolve_formatter(formatter)
        metrics = _resolve_metrics(metric)
        pipeline = FineTuningPipeline(config, fmt, metrics=metrics)
        _, _, test = pipeline.load_data()
        if not test:
            raise ConfigError("No test examples found; set data.test_path / data.max_test.")

        model_id = model or pipeline.state.fine_tuned_model
        if not model_id:
            raise ConfigError(
                "No model id known. Pass --model or run `openai-ft train` first."
            )

        predictor = ChatPredictor(
            build_openai_client(),
            fmt,
            model=model_id,
            config=config.inference,
        )
        evaluator = Evaluator(predictor, metrics=metrics)
        result = evaluator.evaluate(test)
        if output is not None:
            result.save(output)
    except Exception as exc:
        _handle(exc)
        return

    console.print()
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in result.metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)
    console.print(
        f"\n{result.num_examples} examples, {result.num_errors} errors, "
        f"{result.elapsed_seconds:.1f}s"
    )


def main() -> None:
    """Entry point used by the ``openai-ft`` console script."""
    app()


if __name__ == "__main__":
    main()
