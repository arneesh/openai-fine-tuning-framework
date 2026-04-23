"""Tests for metrics."""

from __future__ import annotations

import math

import pytest

from openai_ft.exceptions import EvaluationError
from openai_ft.metrics import Accuracy, ExactMatch, MetricSuite, NumericRegressionMetric


def test_exact_match_counts_matches(sample_examples):
    m = ExactMatch(target_fn=lambda e: e["label"])
    m.update("positive", sample_examples[0])
    m.update("POSITIVE", sample_examples[0])
    m.update("wrong", sample_examples[0])
    assert m.compute() == {"exact_match": pytest.approx(2 / 3)}


def test_exact_match_case_sensitive():
    m = ExactMatch(target_fn=lambda e: e["label"], case_sensitive=True)
    m.update("Positive", {"label": "positive"})
    assert m.compute()["exact_match"] == 0.0


def test_accuracy_alias_has_name():
    m = Accuracy(target_fn=lambda e: e["label"])
    assert m.name == "accuracy"
    m.update("positive", {"label": "positive"})
    assert m.compute() == {"accuracy": 1.0}


def test_exact_match_reset():
    m = ExactMatch(target_fn=lambda e: e["label"])
    m.update("positive", {"label": "positive"})
    m.reset()
    m.update("wrong", {"label": "positive"})
    assert m.compute() == {"exact_match": 0.0}


def test_numeric_regression_basic():
    m = NumericRegressionMetric(target_fn=lambda e: e["price"])
    m.update("$10.00", {"price": 12.0})
    m.update("$14", {"price": 12.0})
    scores = m.compute()
    assert scores["count"] == 2
    assert scores["parse_errors"] == 0
    assert scores["mae"] == pytest.approx(2.0)
    assert scores["rmse"] == pytest.approx(2.0)


def test_numeric_regression_parse_error():
    m = NumericRegressionMetric(target_fn=lambda e: e["price"])
    m.update("not a number", {"price": 10.0})
    scores = m.compute()
    assert scores["parse_errors"] == 1
    assert "mae" not in scores


def test_numeric_regression_handles_commas():
    m = NumericRegressionMetric(target_fn=lambda e: e["price"])
    m.update("$1,234.56", {"price": 1234.56})
    scores = m.compute()
    assert scores["mae"] == pytest.approx(0.0, abs=1e-6)


def test_numeric_regression_rmsle_only_positive():
    m = NumericRegressionMetric(target_fn=lambda e: e["price"])
    m.update("10", {"price": 10})
    scores = m.compute()
    assert scores["rmsle"] == pytest.approx(0.0, abs=1e-6)


def test_metric_suite_aggregates_multiple():
    suite = MetricSuite(
        [
            ExactMatch(target_fn=lambda e: e["label"]),
            NumericRegressionMetric(target_fn=lambda e: e["price"]),
        ]
    )
    suite.update("positive", {"label": "positive", "price": 5.0})
    suite.update("$5", {"label": "positive", "price": 5.0})
    scores = suite.compute()
    assert "exact_match" in scores
    assert any(k.startswith("regression/") for k in scores)


def test_metric_suite_requires_at_least_one():
    with pytest.raises(EvaluationError):
        MetricSuite([])


def test_metric_suite_reset():
    m = ExactMatch(target_fn=lambda e: e["label"])
    suite = MetricSuite([m])
    suite.update("positive", {"label": "positive"})
    suite.reset()
    assert math.isclose(m.compute()["exact_match"], 0.0)
