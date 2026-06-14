from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier


def build_random_forest(seed: int) -> Any:
    return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)


def build_extra_trees(seed: int) -> Any:
    return ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1)


def build_adaboost(seed: int) -> Any:
    return AdaBoostClassifier(n_estimators=300, random_state=seed)
