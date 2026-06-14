from __future__ import annotations

import numpy as np

from eocrc_dat.actuarial.multiplicity import benjamini_hochberg, holm_bonferroni


def test_holm_is_monotone_and_bounded() -> None:
    pvals = np.array([0.001, 0.02, 0.03, 0.5], dtype=np.float64)
    adjusted = holm_bonferroni(pvals)
    assert adjusted.max() <= 1.0
    assert adjusted[0] >= pvals[0]


def test_benjamini_hochberg_rejects_small() -> None:
    pvals = np.array([0.001, 0.008, 0.2, 0.7, 0.9], dtype=np.float64)
    rejected, adjusted = benjamini_hochberg(pvals, q=0.05)
    assert rejected[0]
    assert not rejected[-1]
    assert adjusted.max() <= 1.0
