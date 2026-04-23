"""Tests for dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_ft.config import DataConfig
from openai_ft.datasets import load_from_config, load_json_array, load_jsonl
from openai_ft.exceptions import DataError


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_jsonl(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    _write_jsonl(p, [{"a": 1}, {"a": 2}])
    assert load_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert load_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_missing(tmp_path: Path):
    with pytest.raises(DataError):
        load_jsonl(tmp_path / "missing.jsonl")


def test_load_jsonl_invalid(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}")
    with pytest.raises(DataError):
        load_jsonl(p)


def test_load_json_array(tmp_path: Path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    assert load_json_array(p) == [{"a": 1}, {"a": 2}]


def test_load_json_array_not_list(tmp_path: Path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"a": 1}))
    with pytest.raises(DataError):
        load_json_array(p)


def test_load_from_config_jsonl(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, [{"x": i} for i in range(5)])
    _write_jsonl(val, [{"x": i} for i in range(3)])
    cfg = DataConfig(source="jsonl", train_path=train, val_path=val, max_train=2)
    tr, v, te = load_from_config(cfg)
    assert len(tr) == 2
    assert len(v) == 3
    assert te == []


def test_load_from_config_python_requires_in_memory():
    cfg = DataConfig(source="python")
    with pytest.raises(DataError):
        load_from_config(cfg)


def test_load_from_config_python_passthrough():
    cfg = DataConfig(source="python", max_train=2)
    tr, v, te = load_from_config(
        cfg, in_memory=([1, 2, 3, 4], [5, 6], [7])
    )
    assert tr == [1, 2]
    assert v == [5, 6]
    assert te == [7]
