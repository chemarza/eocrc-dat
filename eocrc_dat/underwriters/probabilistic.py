from __future__ import annotations

from typing import Any

from sklearn.naive_bayes import GaussianNB


def build_naive_bayes(seed: int) -> Any:
    return GaussianNB()
