"""Tests for configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openai_ft.config import DataConfig, FineTuneConfig, HyperParameters
from openai_ft.exceptions import ConfigError


def test_hyperparameters_only_emits_explicit_values():
    hp = HyperParameters(n_epochs=2)
    assert hp.to_payload() == {"n_epochs": 2}


def test_hyperparameters_all_auto_emits_empty():
    assert HyperParameters().to_payload() == {}


def test_hyperparameters_reject_unknown_field():
    with pytest.raises(Exception):
        HyperParameters(foo=1)  # type: ignore[call-arg]


def test_data_config_jsonl_requires_train_path():
    with pytest.raises(Exception):
        FineTuneConfig(data=DataConfig(source="jsonl"))


def test_data_config_hf_requires_path():
    with pytest.raises(Exception):
        FineTuneConfig(data=DataConfig(source="hf"))


def test_data_config_python_ok_without_paths():
    cfg = FineTuneConfig(data=DataConfig(source="python"))
    assert cfg.data.source == "python"


def test_from_file_loads_yaml(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project_name": "sample",
                "data": {
                    "source": "jsonl",
                    "train_path": str(tmp_path / "train.jsonl"),
                },
                "model": {"base_model": "gpt-4o-mini-2024-07-18", "suffix": "demo"},
                "hyperparameters": {"n_epochs": 1},
            }
        )
    )
    cfg = FineTuneConfig.from_file(cfg_path)
    assert cfg.project_name == "sample"
    assert cfg.data.source == "jsonl"
    assert cfg.hyperparameters.n_epochs == 1


def test_from_file_missing_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        FineTuneConfig.from_file(tmp_path / "missing.yaml")


def test_from_file_unsupported_extension(tmp_path: Path):
    p = tmp_path / "cfg.toml"
    p.write_text("x = 1")
    with pytest.raises(ConfigError):
        FineTuneConfig.from_file(p)
