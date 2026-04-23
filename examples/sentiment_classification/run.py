"""Programmatic runner for the sentiment-classification example."""

from __future__ import annotations

from pathlib import Path

from openai_ft import FineTuneConfig, FineTuningPipeline

from .formatter import SentimentFormatter, sentiment_accuracy


def main() -> None:
    config = FineTuneConfig.from_file(Path(__file__).parent / "config.yaml")
    pipeline = FineTuningPipeline(
        config=config,
        formatter=SentimentFormatter(),
        metrics=[sentiment_accuracy()],
    )
    result = pipeline.run()

    print(f"Fine-tuned model: {result.fine_tuned_model}")
    if result.evaluation:
        print("Metrics:")
        for k, v in result.evaluation.metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
