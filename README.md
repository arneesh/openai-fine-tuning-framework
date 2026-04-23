# openai-fine-tuning-framework

A generic Python framework for fine-tuning OpenAI chat models
on arbitrary tasks. Bring your own data, plug in a formatter and (optionally) a
metric, and run the full pipeline — data preparation → upload → training →
evaluation — from one CLI command or a single `FineTuningPipeline.run()` call.

## Features

- **Pluggable formatters** — subclass `BaseFormatter` to turn your domain
  examples into OpenAI chat messages (training _and_ inference).
- **Pluggable metrics** — built-in accuracy / exact-match / numeric regression
  (MAE, RMSE, RMSLE). Custom metrics are a subclass away.
- **Config-driven** — everything is a validated Pydantic model, loadable from
  YAML or JSON.
- **End-to-end pipeline** — `prepare` → `upload` → `train` → `evaluate`, with
  persistent state written to `artifacts/<project>/state.json` so you can
  resume individual stages.
- **Typer CLI + Rich output** — `openai-ft run ...`, `openai-ft status ...`, etc.
- **Retries with tenacity** — resilient uploads and inference against the
  OpenAI API.
- **Typed, tested, linted** — strict `mypy`, `ruff` config, and a `pytest`
  suite.
- **`uv`-native** — modern dependency management.

## Install

```bash
cd openai-fine-tuning-framework
uv sync                       # core deps + project
uv sync --extra dev           # + tests/lint/type-check
uv sync --extra huggingface   # + datasets / huggingface-hub (optional)
```

Copy `.env.example` to `.env` and set at least `OPENAI_API_KEY`.

## Quickstart

### 1. Define a formatter

```python
from openai_ft import BaseFormatter, Accuracy

class SentimentFormatter(BaseFormatter):
    def system_message(self, example):
        return "You are a sentiment classifier. Respond with positive/negative/neutral."

    def user_message(self, example):
        return f"Classify: {example['text']}"

    def assistant_message(self, example):
        return example["label"]

def sentiment_accuracy():
    return Accuracy(target_fn=lambda e: e["label"])
```

### 2. Describe the run in YAML

```yaml
# config.yaml
project_name: sentiment-v1
artifacts_dir: ./artifacts

data:
  source: jsonl
  train_path: ./data/train.jsonl
  val_path: ./data/validation.jsonl
  test_path: ./data/test.jsonl

model:
  base_model: gpt-4o-mini-2024-07-18
  suffix: sentiment
  seed: 7

hyperparameters:
  n_epochs: 2

inference:
  max_tokens: 4
  temperature: 0.0
```

### 3. Run it

```bash
openai-ft run config.yaml \
    --formatter mypkg.formatters:SentimentFormatter \
    --metric   mypkg.formatters:sentiment_accuracy
```

or programmatically:

```python
from openai_ft import FineTuneConfig, FineTuningPipeline
from mypkg.formatters import SentimentFormatter, sentiment_accuracy

config = FineTuneConfig.from_file("config.yaml")
pipeline = FineTuningPipeline(
    config=config,
    formatter=SentimentFormatter(),
    metrics=[sentiment_accuracy()],
)
result = pipeline.run()

print(result.fine_tuned_model)
print(result.evaluation.metrics)
```

## CLI

```
openai-ft run       CONFIG.yaml  -f pkg.mod:Formatter  -m pkg.mod:metric
openai-ft prepare   CONFIG.yaml  -f pkg.mod:Formatter
openai-ft upload    CONFIG.yaml  -f pkg.mod:Formatter
openai-ft train     CONFIG.yaml  -f pkg.mod:Formatter  [--training-file-id ...]
openai-ft status    JOB_ID       [--events 10]
openai-ft evaluate  CONFIG.yaml  -f pkg.mod:Formatter  -m pkg.mod:metric  [--model ft:...]
```

Each stage persists its output in `artifacts/<project>/state.json`, so you
can run them independently (e.g. prepare offline, then upload+train later).

## Configuration reference

Top-level `FineTuneConfig` fields:

| field             | description                                                               |
| ----------------- | ------------------------------------------------------------------------- |
| `project_name`    | Used for the artifacts subdirectory.                                      |
| `artifacts_dir`   | Root directory for JSONL, state, evaluation output.                       |
| `data`            | `DataConfig` — where examples come from.                                  |
| `model`           | `ModelConfig` — base model id, suffix, seed.                              |
| `hyperparameters` | `HyperParameters` — `n_epochs`, `batch_size`, `learning_rate_multiplier`. |
| `trainer`         | Polling and timeout controls.                                             |
| `inference`       | Decoding settings used during evaluation.                                 |

Data sources:

- **`jsonl`** — `train_path`, `val_path`, `test_path` point to JSONL files.
- **`json`** — same, but each file is a single JSON array.
- **`hf`** — `path` is a Hugging Face dataset repo (requires the
  `huggingface` extra).
- **`python`** — pass `in_memory_data=(train, val, test)` to
  `FineTuningPipeline` directly.

## Built-in abstractions

**Formatters**

- `BaseFormatter` — subclass and implement `user_message` / `assistant_message`.
- `TemplateFormatter` — quick `str.format`-based formatter for simple cases.
- `formatter_from_functions(user_fn, assistant_fn, system_fn=None)` — zero-class
  adapter.

**Metrics**

- `Accuracy`, `ExactMatch` — case-insensitive exact-match accuracy.
- `NumericRegressionMetric` — MAE / RMSE / RMSLE with a pluggable number
  parser (default: first number in the string, handles `$1,234.56`).
- `MetricSuite` — aggregate multiple metrics in one pass.
- `BaseMetric` — subclass for custom logic (call `update` per example, then
  `compute`).

## Examples

Worked examples in `examples/`:

- [`price_estimation/`](examples/price_estimation/) — fine-tuning a model for product price estimation (regression task).
- [`sentiment_classification/`](examples/sentiment_classification/) — fine-tuning a model for sentiment analysis (classification task).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```

## Project layout

```
openai-fine-tuning-framework/
├── pyproject.toml
├── README.md
├── .env.example
├── src/openai_ft/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI
│   ├── client.py           # OpenAI client factory
│   ├── config.py           # Pydantic configs
│   ├── datasets.py         # JSONL / JSON / HF loaders
│   ├── evaluator.py        # Evaluator + EvaluationResult
│   ├── exceptions.py
│   ├── formatters.py       # BaseFormatter, TemplateFormatter
│   ├── inference.py        # ChatPredictor with retries
│   ├── jsonl.py            # Build + write training JSONL
│   ├── logging_utils.py
│   ├── metrics.py          # BaseMetric, Accuracy, NumericRegressionMetric
│   ├── pipeline.py         # End-to-end orchestrator
│   ├── trainer.py          # Create/monitor fine-tuning jobs
│   └── uploader.py         # Upload files to OpenAI
├── tests/
└── examples/
    ├── price_estimation/
    └── sentiment_classification/
```

## Author

Arneesh Aima

## License

MIT
