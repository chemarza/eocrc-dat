from __future__ import annotations

import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

from eocrc_dat.agency.schema import DataConfig, ExperimentConfig, ModelConfig, TrainConfig


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _filter(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def _build_model(data: dict[str, Any]) -> ModelConfig:
    clean = _filter(ModelConfig, data)
    if "mlp_hidden" in clean:
        clean["mlp_hidden"] = tuple(clean["mlp_hidden"])
    return ModelConfig(**clean)


def _build_data(data: dict[str, Any]) -> DataConfig:
    clean = _filter(DataConfig, data)
    if "drop_domains" in clean:
        clean["drop_domains"] = tuple(clean["drop_domains"])
    return DataConfig(**clean)


def _build_train(data: dict[str, Any]) -> TrainConfig:
    return TrainConfig(**_filter(TrainConfig, data))


def _resolve_group(configs_dir: Path, group: str, name: str | None) -> dict[str, Any]:
    if name is None:
        return {}
    return _read_toml(configs_dir / group / f"{name}.toml")


def load_experiment(name: str, configs_dir: str | Path = "configs") -> ExperimentConfig:
    root = Path(configs_dir)
    raw = _read_toml(root / "experiment" / f"{name}.toml")
    parent = raw.get("extends")
    if isinstance(parent, str):
        base = _read_toml(root / "experiment" / f"{parent}.toml")
        raw = _deep_merge(base, raw)

    use = raw.get("use", {})
    model_dict = _deep_merge(_resolve_group(root, "model", use.get("model")), raw.get("model", {}))
    data_dict = _deep_merge(_resolve_group(root, "data", use.get("data")), raw.get("data", {}))
    train_dict = _deep_merge(_resolve_group(root, "train", use.get("train")), raw.get("train", {}))

    thresholds = tuple(raw.get("thresholds", (0.01, 0.02, 0.05)))
    return ExperimentConfig(
        name=str(raw.get("name", name)),
        seed=int(raw.get("seed", 7)),
        model=_build_model(model_dict),
        data=_build_data(data_dict),
        train=_build_train(train_dict),
        thresholds=thresholds,
    )
