from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]


def net_benefit(y: Labels, p: FloatArray, threshold: float) -> float:
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    n = y.shape[0]
    flagged = p >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    odds = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * odds


def decision_curve(y: Labels, p: FloatArray, thresholds: Sequence[float]) -> dict[str, FloatArray]:
    prevalence = float(np.mean(y))
    model = np.array([net_benefit(y, p, t) for t in thresholds], dtype=np.float64)
    treat_all = np.array(
        [prevalence - (1.0 - prevalence) * (t / (1.0 - t)) for t in thresholds],
        dtype=np.float64,
    )
    return {
        "thresholds": np.asarray(thresholds, dtype=np.float64),
        "model": model,
        "treat_all": treat_all,
        "treat_none": np.zeros(len(thresholds), dtype=np.float64),
    }
