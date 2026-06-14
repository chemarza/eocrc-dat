from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_logistic_regression(seed: int) -> Any:
    return LogisticRegression(max_iter=2000, C=1.0)


def build_svm_linear(seed: int) -> Any:
    return SVC(kernel="linear", probability=True, random_state=seed)
