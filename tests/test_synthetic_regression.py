from __future__ import annotations

from eocrc_dat.intake.synthetic import make_bundle
from eocrc_dat.ledger.schema import FEATURE_NAMES


def test_feature_count_is_91() -> None:
    assert len(FEATURE_NAMES) == 91


def test_prevalence_targets_match_table_one() -> None:
    bundle = make_bundle(7)
    assert abs(bundle["train"].prevalence - 0.167) < 0.04
    assert abs(bundle["temporal"].prevalence - 0.083) < 0.04
    assert abs(bundle["external"].prevalence - 0.041) < 0.03


def test_generation_is_deterministic() -> None:
    first = make_bundle(7)["internal_test"]
    second = make_bundle(7)["internal_test"]
    assert (first.x == second.x).all()
    assert (first.y == second.y).all()
