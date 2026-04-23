"""Programmatic runner for the price-estimation example.

Equivalent to::

    openai-ft run config.yaml \\
        --formatter examples.price_estimation.formatter:PriceFormatter \\
        --metric   examples.price_estimation.formatter:price_regression_metric
"""

from __future__ import annotations

from pathlib import Path

from openai_ft import FineTuneConfig, FineTuningPipeline

from .formatter import PriceFormatter, price_regression_metric


def main() -> None:
    config = FineTuneConfig.from_file(Path(__file__).parent / "config.yaml")
    pipeline = FineTuningPipeline(
        config=config,
        formatter=PriceFormatter(),
        metrics=[price_regression_metric()],
    )
    result = pipeline.run()

    print(f"Fine-tuned model: {result.fine_tuned_model}")
    if result.evaluation:
        print("Metrics:")
        for k, v in result.evaluation.metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
