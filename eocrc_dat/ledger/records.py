from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

SplitName = Literal["train", "internal_test", "temporal", "external"]
SPLIT_ORDER: tuple[SplitName, ...] = ("train", "internal_test", "temporal", "external")

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CohortArrays:
    x: FloatArray
    y: IntArray
    feature_names: tuple[str, ...]
    groups: dict[str, npt.NDArray[np.str_]] = field(default_factory=dict)
    raw: dict[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("x must be a 2-D matrix")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y disagree on sample count")
        if self.x.shape[1] != len(self.feature_names):
            raise ValueError("x columns must match feature_names")

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def prevalence(self) -> float:
        return float(self.y.mean()) if self.n else 0.0


@dataclass(frozen=True)
class CohortBundle:
    splits: dict[SplitName, CohortArrays]

    def __getitem__(self, name: SplitName) -> CohortArrays:
        return self.splits[name]

    def names(self) -> tuple[SplitName, ...]:
        return tuple(name for name in SPLIT_ORDER if name in self.splits)
