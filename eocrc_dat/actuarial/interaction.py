from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def isr(w_ab: float, w_a: float, w_b: float) -> float:
    denom = np.sqrt(max(w_a, 1e-12) * max(w_b, 1e-12))
    return float(w_ab / denom)


def interaction_ratio(delta_combined: float, delta_a: float, delta_b: float) -> float:
    additive = delta_a + delta_b
    if additive == 0.0:
        return float("inf")
    return float(delta_combined / additive)


def expected_additive_degradation(deltas: FloatArray, covariance: FloatArray) -> float:
    diagonal = float(np.sum(deltas**2))
    cross = 0.0
    n = deltas.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            cross += covariance[i, j]
    return float(np.sqrt(max(diagonal + 2.0 * cross, 0.0)))


@dataclass(frozen=True)
class InteractionPair:
    feature_a: str
    feature_b: str
    strength: float


def top_interactions(
    attention: FloatArray, feature_names: tuple[str, ...], k: int
) -> list[InteractionPair]:
    n = len(feature_names)
    pairs: list[InteractionPair] = []
    for i in range(n):
        for j in range(i + 1, n):
            w_ab = 0.5 * (attention[i, j] + attention[j, i])
            strength = isr(w_ab, attention[i, i], attention[j, j])
            pairs.append(InteractionPair(feature_names[i], feature_names[j], strength))
    pairs.sort(key=lambda pair: pair.strength, reverse=True)
    return pairs[:k]
