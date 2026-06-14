from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Grade = Literal["low", "moderate", "high"]


@dataclass(frozen=True)
class ScoreItem:
    feature: str
    cutpoints: tuple[float, float, float]


SCORECARD: tuple[ScoreItem, ...] = (
    ScoreItem("iron_deficiency_anemia", (1.0, 2.0, 3.0)),
    ScoreItem("crp", (3.0, 10.0, 30.0)),
    ScoreItem("gi_hemorrhage", (1.0, 2.0, 3.0)),
    ScoreItem("obesity_bmi", (27.0, 30.0, 35.0)),
    ScoreItem("hemoglobin_severity", (1.0, 2.0, 3.0)),
    ScoreItem("type2_diabetes", (1.0, 2.0, 3.0)),
    ScoreItem("ibd", (1.0, 2.0, 3.0)),
    ScoreItem("family_history_crc", (1.0, 2.0, 3.0)),
    ScoreItem("processed_meat", (1.0, 3.0, 7.0)),
    ScoreItem("age", (30.0, 35.0, 45.0)),
)

MAX_SCORE = 3 * len(SCORECARD)
REFERENCE_AUROC: dict[str, float] = {
    "internal_test": 0.862,
    "temporal": 0.841,
    "external": 0.819,
}


def item_points(item: ScoreItem, value: float) -> int:
    return int(sum(value >= cut for cut in item.cutpoints))


def score_rows(values: Mapping[str, float]) -> int:
    total = 0
    for item in SCORECARD:
        total += item_points(item, values[item.feature])
    return total


def grade(total: int) -> Grade:
    if total <= 8:
        return "low"
    if total <= 15:
        return "moderate"
    return "high"


def score_array(raw: Mapping[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.int64]:
    n = next(iter(raw.values())).shape[0]
    totals = np.zeros(n, dtype=np.int64)
    for item in SCORECARD:
        column = raw[item.feature]
        for cut in item.cutpoints:
            totals += (column >= cut).astype(np.int64)
    return totals
