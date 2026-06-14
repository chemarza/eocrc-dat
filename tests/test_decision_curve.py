from __future__ import annotations

import numpy as np

from eocrc_dat.actuarial.decision_curve import decision_curve, net_benefit


def test_treat_none_is_zero_and_all_matches_formula() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(np.int64)
    p = rng.uniform(0, 1, 300)
    curve = decision_curve(y, p, [0.01, 0.02, 0.05])
    assert np.allclose(curve["treat_none"], 0.0)
    prevalence = float(y.mean())
    expected_all = prevalence - (1 - prevalence) * (0.02 / 0.98)
    assert abs(curve["treat_all"][1] - expected_all) < 1e-9


def test_net_benefit_bounded_by_prevalence() -> None:
    y = np.array([1, 1, 0, 0, 0], dtype=np.int64)
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.05])
    value = net_benefit(y, p, 0.5)
    assert value <= float(y.mean()) + 1e-9
