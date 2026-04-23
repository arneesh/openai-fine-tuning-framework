"""Tests for the evaluator."""

from __future__ import annotations

from openai_ft.evaluator import Evaluator
from openai_ft.metrics import ExactMatch


def test_evaluator_runs_and_computes_metrics():
    examples = [
        {"text": "a", "label": "x"},
        {"text": "b", "label": "y"},
        {"text": "c", "label": "z"},
    ]
    predict = lambda ex: ex["label"]  # noqa: E731
    evaluator = Evaluator(predict, [ExactMatch(target_fn=lambda e: e["label"])])
    result = evaluator.evaluate(examples, show_progress=False)
    assert result.metrics["exact_match"] == 1.0
    assert result.num_examples == 3
    assert result.num_errors == 0
    assert len(result.predictions) == 3


def test_evaluator_continues_on_error():
    examples = [{"label": "a"}, {"label": "b"}]
    calls = {"n": 0}

    def flaky(ex):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return ex["label"]

    evaluator = Evaluator(
        flaky,
        [ExactMatch(target_fn=lambda e: e["label"])],
        continue_on_error=True,
    )
    result = evaluator.evaluate(examples, show_progress=False)
    assert result.num_errors == 1
    assert result.metrics["exact_match"] == 1.0


def test_evaluator_save_writes_json(tmp_path):
    examples = [{"label": "a"}]
    predict = lambda ex: ex["label"]  # noqa: E731
    evaluator = Evaluator(predict, [ExactMatch(target_fn=lambda e: e["label"])])
    result = evaluator.evaluate(examples, show_progress=False)
    out = result.save(tmp_path / "eval.json")
    assert out.exists()
    assert '"exact_match"' in out.read_text()
