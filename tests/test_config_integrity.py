from __future__ import annotations

from pathlib import Path

from eocrc_dat.agency.config import load_experiment
from eocrc_dat.objective.constraints import SelectionConstraints


def test_main_matches_reported_hyperparameters(configs_dir: Path) -> None:
    cfg = load_experiment("main", configs_dir)
    assert cfg.model.method == "dat"
    assert cfg.model.k_steps == 5
    assert abs(cfg.model.gamma - 1.3) < 1e-9
    assert cfg.model.tabnet_dim == 64
    assert cfg.model.ft_dim == 192
    assert cfg.model.ft_heads == 8
    assert cfg.model.ft_layers == 3
    assert abs(cfg.train.learning_rate - 1e-3) < 1e-12
    assert abs(cfg.train.weight_decay - 1e-5) < 1e-12
    assert abs(cfg.train.lam - 1e-5) < 1e-12
    assert cfg.train.patience == 20


def test_effective_batch_product(configs_dir: Path) -> None:
    cfg = load_experiment("main", configs_dir)
    assert (
        cfg.train.effective_batch
        == cfg.train.batch_size * cfg.train.grad_accum * cfg.train.world_size
    )


def test_smoke_config_is_labelled(configs_dir: Path) -> None:
    text = (configs_dir / "experiment" / "_smoke.toml").read_text(encoding="utf-8")
    assert "smoke" in text.lower()
    cfg = load_experiment("_smoke", configs_dir)
    assert cfg.train.max_steps == 2


def test_selection_constraints_defaults() -> None:
    constraints = SelectionConstraints()
    assert abs(constraints.tau1_auroc - 0.85) < 1e-9
    assert abs(constraints.tau2_slope - 0.15) < 1e-9
    assert abs(constraints.tau3_citl - 0.05) < 1e-9
    assert abs(constraints.tau4_dpr - 1.25) < 1e-9
    assert abs(constraints.t_inf_ms - 10.0) < 1e-9
    passing = {"auroc": 0.9, "calibration_slope": 1.05, "citl": 0.01, "dpr": 1.1, "latency_ms": 4.0}
    assert constraints.passes(passing)
    assert not constraints.passes({**passing, "auroc": 0.5})
