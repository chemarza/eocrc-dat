from __future__ import annotations

import numpy as np

from eocrc_dat.actuarial.fairness import demographic_parity_ratio


def test_equal_rates_give_unit_ratio() -> None:
    flagged = np.array([1, 0, 1, 0], dtype=np.int64)
    group = np.array(["a", "a", "b", "b"], dtype=np.str_)
    summary = demographic_parity_ratio(flagged, group)
    assert abs(summary["dpr"] - 1.0) < 1e-9


def test_imbalanced_rates_exceed_one() -> None:
    flagged = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    group = np.array(["a", "a", "a", "b", "b", "b"], dtype=np.str_)
    summary = demographic_parity_ratio(flagged, group)
    assert summary["dpr"] > 1.0
    assert summary["rate[a]"] == 1.0
