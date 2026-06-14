from __future__ import annotations

from collections.abc import Callable
from typing import Any


def tune(objective: Callable[[Any], float], n_trials: int = 100, seed: int = 7) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is an optional extra; install eocrc-dat[tuning] to run the search."
        ) from exc
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    best: dict[str, Any] = dict(study.best_params)
    best["value"] = study.best_value
    return best
