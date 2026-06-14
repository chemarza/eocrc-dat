from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier


def build_xgboost(seed: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
    )


def build_gbdt(seed: int) -> Any:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4, random_state=seed
    )


def build_lightgbm(seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is an optional extra; install eocrc-dat[boosting] to enable it."
        ) from exc
    return LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, random_state=seed, verbosity=-1
    )


def build_catboost(seed: int) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is an optional extra; install eocrc-dat[boosting] to enable it."
        ) from exc
    return CatBoostClassifier(
        iterations=400, depth=4, learning_rate=0.05, random_seed=seed, verbose=False
    )
