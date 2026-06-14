from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eocrc_dat.ledger.records import CohortArrays, FloatArray
from eocrc_dat.ledger.schema import continuous_mask


def winsorize(
    x: FloatArray, cont_idx: npt.NDArray[np.int64], p: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    lo = np.zeros(cont_idx.shape[0], dtype=np.float32)
    hi = np.zeros(cont_idx.shape[0], dtype=np.float32)
    out = x.copy()
    for k, col in enumerate(cont_idx):
        lo[k] = np.quantile(x[:, col], p)
        hi[k] = np.quantile(x[:, col], 1.0 - p)
        out[:, col] = np.clip(x[:, col], lo[k], hi[k])
    return out.astype(np.float32), lo, hi


def mice_impute(x: FloatArray, seed: int = 0) -> FloatArray:
    if not np.isnan(x).any():
        return x
    importlib.import_module("sklearn.experimental.enable_iterative_imputer")
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(random_state=seed, sample_posterior=False, max_iter=10)
    return imputer.fit_transform(x).astype(np.float32)


def zscore(
    x: FloatArray, cont_idx: npt.NDArray[np.int64]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    mean = x[:, cont_idx].mean(axis=0).astype(np.float32)
    std = (x[:, cont_idx].std(axis=0) + 1e-6).astype(np.float32)
    out = x.copy()
    out[:, cont_idx] = (x[:, cont_idx] - mean) / std
    return out.astype(np.float32), mean, std


@dataclass(frozen=True)
class FittedRating:
    cont_idx: npt.NDArray[np.int64]
    lo: FloatArray
    hi: FloatArray
    mean: FloatArray
    std: FloatArray


def fit_rating(arrays: CohortArrays, p: float = 0.01) -> FittedRating:
    cont_idx = np.flatnonzero(np.asarray(continuous_mask())).astype(np.int64)
    base = mice_impute(arrays.x)
    clipped, lo, hi = winsorize(base, cont_idx, p)
    _, mean, std = zscore(clipped, cont_idx)
    return FittedRating(cont_idx=cont_idx, lo=lo, hi=hi, mean=mean, std=std)


def apply_rating(fitted: FittedRating, arrays: CohortArrays) -> CohortArrays:
    base = mice_impute(arrays.x)
    out = base.copy()
    for k, col in enumerate(fitted.cont_idx):
        out[:, col] = np.clip(base[:, col], fitted.lo[k], fitted.hi[k])
    out[:, fitted.cont_idx] = (out[:, fitted.cont_idx] - fitted.mean) / fitted.std
    return CohortArrays(
        x=out.astype(np.float32),
        y=arrays.y,
        feature_names=arrays.feature_names,
        groups=arrays.groups,
        raw=arrays.raw,
    )
