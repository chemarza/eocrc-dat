from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from eocrc_dat.actuarial.metrics import calibration, discrimination


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(np.int64)
    p = np.clip(0.15 + 0.6 * y + rng.normal(0, 0.2, 500), 1e-3, 1 - 1e-3)
    return y, p


def test_discrimination_matches_sklearn() -> None:
    y, p = _data()
    metrics = discrimination(y, p)
    assert abs(metrics["auroc"] - roc_auc_score(y, p)) < 1e-9
    assert abs(metrics["auprc"] - average_precision_score(y, p)) < 1e-9


def test_brier_matches_sklearn() -> None:
    y, p = _data()
    assert abs(calibration(y, p)["brier"] - brier_score_loss(y, p)) < 1e-9


def test_perfect_calibration_slope_near_one() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(np.int64)
    result = calibration(y, p)
    assert abs(result["calibration_slope"] - 1.0) < 0.2
    assert abs(result["citl"]) < 0.2
