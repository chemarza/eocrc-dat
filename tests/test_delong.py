from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from eocrc_dat.actuarial.delong import delong_test


def test_auc_estimate_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600).astype(np.int64)
    p1 = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.3, 600), 0, 1)
    p2 = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.3, 600), 0, 1)
    result = delong_test(y, p1, p2)
    assert abs(result["auc_1"] - roc_auc_score(y, p1)) < 1e-6
    assert abs(result["auc_2"] - roc_auc_score(y, p2)) < 1e-6


def test_identical_predictions_not_significant() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400).astype(np.int64)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 400), 0, 1)
    result = delong_test(y, p, p.copy())
    assert result["p_value"] > 0.99
    assert abs(result["z"]) < 1e-6
