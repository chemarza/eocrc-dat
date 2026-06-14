from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier


def build_knn(seed: int) -> Any:
    return KNeighborsClassifier(n_neighbors=25)
