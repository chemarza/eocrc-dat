from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]


def _midrank(x: FloatArray) -> FloatArray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _fast_delong(predictions: FloatArray, positives: int) -> tuple[FloatArray, FloatArray]:
    m = positives
    n = predictions.shape[1] - m
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    k = predictions.shape[0]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r, :] = _midrank(positive[r, :])
        ty[r, :] = _midrank(negative[r, :])
        tz[r, :] = _midrank(predictions[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(y: Labels, p1: FloatArray, p2: FloatArray) -> dict[str, float]:
    positives = int(y.sum())
    if positives == 0 or positives == y.shape[0]:
        raise ValueError("DeLong test requires both classes present")
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    idx = np.concatenate([pos_idx, neg_idx])
    predictions = np.vstack([p1[idx], p2[idx]]).astype(np.float64)
    aucs, cov = _fast_delong(predictions, positives)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ cov @ contrast.T)[0, 0])
    z = 0.0 if variance <= 0.0 else float((aucs[0] - aucs[1]) / np.sqrt(variance))
    pvalue = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return {"z": z, "p_value": pvalue, "auc_1": float(aucs[0]), "auc_2": float(aucs[1])}
