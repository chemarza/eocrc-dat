from __future__ import annotations

import numpy as np

from eocrc_dat.tariff.scorecard import (
    MAX_SCORE,
    SCORECARD,
    grade,
    item_points,
    score_array,
    score_rows,
)


def test_scorecard_dimensions() -> None:
    assert MAX_SCORE == 30
    assert len(SCORECARD) == 10


def test_item_points_monotone() -> None:
    item = SCORECARD[1]
    assert item.feature == "crp"
    assert item_points(item, 1.0) == 0
    assert item_points(item, 5.0) == 1
    assert item_points(item, 15.0) == 2
    assert item_points(item, 40.0) == 3


def test_grade_bands() -> None:
    assert grade(0) == "low"
    assert grade(8) == "low"
    assert grade(9) == "moderate"
    assert grade(15) == "moderate"
    assert grade(16) == "high"
    assert grade(24) == "high"


def test_score_rows_and_array_agree() -> None:
    values = {item.feature: float(item.cutpoints[-1]) for item in SCORECARD}
    assert score_rows(values) == MAX_SCORE
    raw = {item.feature: np.array([item.cutpoints[-1]], dtype=np.float32) for item in SCORECARD}
    assert int(score_array(raw)[0]) == MAX_SCORE
