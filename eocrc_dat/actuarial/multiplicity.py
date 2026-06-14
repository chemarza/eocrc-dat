from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def holm_bonferroni(pvalues: FloatArray) -> FloatArray:
    n = pvalues.shape[0]
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues: FloatArray, q: float = 0.05) -> tuple[BoolArray, FloatArray]:
    n = pvalues.shape[0]
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=np.float64)
    prev = 1.0
    for position in range(n - 1, -1, -1):
        idx = order[position]
        rank = position + 1
        value = pvalues[idx] * n / rank
        prev = min(prev, value)
        adjusted[idx] = min(prev, 1.0)
    rejected = adjusted <= q
    return rejected, adjusted
