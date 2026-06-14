from __future__ import annotations

from typing import Any

from sklearn.svm import SVC


def build_svm_rbf(seed: int) -> Any:
    return SVC(kernel="rbf", gamma="scale", probability=True, random_state=seed)
