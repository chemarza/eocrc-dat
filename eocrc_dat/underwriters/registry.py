from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from eocrc_dat.underwriters import boosting, ensemble, instance, kernel, linear, probabilistic

Backend = Literal["torch", "sklearn"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    backend: Backend
    proposed: bool = False


_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("dat", "deep_tabular", "torch", proposed=True),
    MethodSpec("ft_transformer", "deep_tabular", "torch"),
    MethodSpec("tabnet", "deep_tabular", "torch"),
    MethodSpec("saint", "deep_tabular", "torch"),
    MethodSpec("mlp", "deep_tabular", "torch"),
    MethodSpec("xgboost", "gradient_boosting", "sklearn"),
    MethodSpec("lightgbm", "gradient_boosting", "sklearn"),
    MethodSpec("catboost", "gradient_boosting", "sklearn"),
    MethodSpec("gbdt", "gradient_boosting", "sklearn"),
    MethodSpec("random_forest", "ensemble_tree", "sklearn"),
    MethodSpec("extra_trees", "ensemble_tree", "sklearn"),
    MethodSpec("adaboost", "boosted", "sklearn"),
    MethodSpec("svm_rbf", "kernel", "sklearn"),
    MethodSpec("logistic_regression", "linear", "sklearn"),
    MethodSpec("svm_linear", "linear", "sklearn"),
    MethodSpec("naive_bayes", "probabilistic", "sklearn"),
    MethodSpec("knn", "instance", "sklearn"),
)

REGISTRY: dict[str, MethodSpec] = {spec.name: spec for spec in _METHODS}


def families() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for spec in _METHODS:
        grouped.setdefault(spec.family, []).append(spec.name)
    return {family: tuple(names) for family, names in grouped.items()}


FAMILIES: dict[str, tuple[str, ...]] = families()


def baselines() -> tuple[str, ...]:
    return tuple(spec.name for spec in _METHODS if not spec.proposed)


_SKLEARN_BUILDERS: dict[str, Callable[[int], Any]] = {
    "xgboost": boosting.build_xgboost,
    "gbdt": boosting.build_gbdt,
    "lightgbm": boosting.build_lightgbm,
    "catboost": boosting.build_catboost,
    "random_forest": ensemble.build_random_forest,
    "extra_trees": ensemble.build_extra_trees,
    "adaboost": ensemble.build_adaboost,
    "svm_rbf": kernel.build_svm_rbf,
    "logistic_regression": linear.build_logistic_regression,
    "svm_linear": linear.build_svm_linear,
    "naive_bayes": probabilistic.build_naive_bayes,
    "knn": instance.build_knn,
}


def build_sklearn(name: str, seed: int) -> Any:
    if name not in _SKLEARN_BUILDERS:
        raise KeyError(f"no scikit-learn builder for method '{name}'")
    return _SKLEARN_BUILDERS[name](seed)
