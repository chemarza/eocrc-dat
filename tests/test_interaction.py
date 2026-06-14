from __future__ import annotations

import numpy as np

from eocrc_dat.actuarial.interaction import (
    expected_additive_degradation,
    interaction_ratio,
    isr,
    top_interactions,
)


def test_isr_self_is_one_and_symmetric() -> None:
    assert abs(isr(4.0, 4.0, 4.0) - 1.0) < 1e-9
    assert abs(isr(2.0, 1.0, 4.0) - isr(2.0, 4.0, 1.0)) < 1e-9


def test_interaction_ratio_super_additive() -> None:
    assert interaction_ratio(1.19, 0.5, 0.5) > 1.0


def test_expected_additive_degradation_reduces_to_norm() -> None:
    deltas = np.array([0.3, 0.4], dtype=np.float64)
    zero_cov = np.zeros((2, 2), dtype=np.float64)
    assert abs(expected_additive_degradation(deltas, zero_cov) - 0.5) < 1e-9


def test_top_interactions_orders_by_strength() -> None:
    attention = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]], dtype=np.float64)
    pairs = top_interactions(attention, ("a", "b", "c"), k=3)
    assert (pairs[0].feature_a, pairs[0].feature_b) == ("a", "b")
    assert pairs[0].strength >= pairs[1].strength
