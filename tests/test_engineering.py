from __future__ import annotations

import numpy as np

from eocrc_dat.intake.synthetic import make_bundle
from eocrc_dat.rating.engineering import apply_rating, fit_rating


def test_zscore_standardises_training_continuous() -> None:
    bundle = make_bundle(7)
    train = bundle["train"]
    fitted = fit_rating(train)
    engineered = apply_rating(fitted, train)
    block = engineered.x[:, fitted.cont_idx]
    assert abs(float(block.mean())) < 1e-3
    assert abs(float(block.std()) - 1.0) < 0.1


def test_winsor_bounds_respected_on_holdout() -> None:
    bundle = make_bundle(7)
    fitted = fit_rating(bundle["train"])
    engineered = apply_rating(fitted, bundle["external"])
    lo = (fitted.lo - fitted.mean) / fitted.std
    hi = (fitted.hi - fitted.mean) / fitted.std
    block = engineered.x[:, fitted.cont_idx]
    assert np.all(block >= lo - 1e-4)
    assert np.all(block <= hi + 1e-4)
