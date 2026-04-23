"""Formatter + metric for the price-estimation example.

Each example is expected to expose:

* ``summary`` -- the product description used as model input
* ``price``   -- the ground-truth numeric price
"""

from __future__ import annotations

from typing import Any

from openai_ft import BaseFormatter, NumericRegressionMetric


class PriceFormatter(BaseFormatter):
    """Ask the model to estimate a product price given its description."""

    def _price(self, example: Any) -> float:
        return float(example["price"] if isinstance(example, dict) else example.price)

    def _summary(self, example: Any) -> str:
        return str(example["summary"] if isinstance(example, dict) else example.summary)

    def user_message(self, example: Any) -> str:
        return (
            "Estimate the price of this product. "
            "Respond with the price, no explanation.\n\n"
            f"{self._summary(example)}"
        )

    def assistant_message(self, example: Any) -> str:
        return f"${self._price(example):.2f}"


def price_regression_metric() -> NumericRegressionMetric:
    """Numeric regression metric (MAE/RMSE/RMSLE) keyed on the ``price`` field."""
    return NumericRegressionMetric(
        target_fn=lambda e: e["price"] if isinstance(e, dict) else e.price
    )
