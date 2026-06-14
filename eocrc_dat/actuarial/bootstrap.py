from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

FloatArray = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]
Metric = Callable[[Labels, FloatArray], float]


def _auroc(y: Labels, p: FloatArray) -> float:
    return float(roc_auc_score(y, p))


def bootstrap_ci(
    y: Labels,
    p: FloatArray,
    metric: Metric = _auroc,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    stratified: bool = True,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    estimates: list[float] = []
    for _ in range(resamples):
        if stratified and pos.size and neg.size:
            draw = np.concatenate(
                [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
            )
        else:
            draw = rng.integers(0, y.shape[0], y.shape[0])
        sample_y = y[draw]
        if sample_y.min() == sample_y.max():
            continue
        estimates.append(metric(sample_y, p[draw]))
    array = np.asarray(estimates, dtype=np.float64)
    return {
        "point": metric(y, p),
        "lower": float(np.quantile(array, alpha / 2.0)),
        "upper": float(np.quantile(array, 1.0 - alpha / 2.0)),
    }
