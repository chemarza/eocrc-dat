from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from eocrc_dat.intake.synthetic import DEFAULT_BUNDLE, SplitSpec, make_bundle
from eocrc_dat.ledger.records import CohortArrays, CohortBundle
from eocrc_dat.ledger.schema import index_of


def subset(arrays: CohortArrays, idx: npt.NDArray[np.int64]) -> CohortArrays:
    groups = {key: value[idx] for key, value in arrays.groups.items()}
    raw = {key: value[idx] for key, value in arrays.raw.items()}
    return CohortArrays(
        x=arrays.x[idx],
        y=arrays.y[idx],
        feature_names=arrays.feature_names,
        groups=groups,
        raw=raw,
    )


def _propensity(arrays: CohortArrays) -> npt.NDArray[np.float64]:
    cols = [
        arrays.x[:, index_of("age")],
        arrays.x[:, index_of("sex_male")],
        arrays.x[:, index_of("race_nonwhite")],
        arrays.x[:, index_of("region_south")],
    ]
    design = np.column_stack(cols).astype(np.float64)
    design[:, 0] = (design[:, 0] - design[:, 0].mean()) / (design[:, 0].std() + 1e-9)
    model = LogisticRegression(max_iter=1000)
    model.fit(design, arrays.y)
    return model.predict_proba(design)[:, 1]


def propensity_match(arrays: CohortArrays, ratio: int = 5, seed: int = 0) -> CohortArrays:
    if arrays.y.sum() == 0:
        return arrays
    scores = _propensity(arrays)
    cases = np.flatnonzero(arrays.y == 1)
    controls = np.flatnonzero(arrays.y == 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(cases)
    used: set[int] = set()
    chosen: list[int] = []
    control_scores = scores[controls]
    for case in order:
        gaps = np.abs(control_scores - scores[case])
        ranked = np.argsort(gaps, kind="stable")
        picked = 0
        for pos in ranked:
            cidx = int(controls[pos])
            if cidx in used:
                continue
            used.add(cidx)
            chosen.append(cidx)
            picked += 1
            if picked >= ratio:
                break
    keep = np.concatenate([order, np.asarray(chosen, dtype=np.int64)])
    keep.sort()
    return subset(arrays, keep.astype(np.int64))


def make_splits(seed: int, specs: tuple[SplitSpec, ...] = DEFAULT_BUNDLE) -> CohortBundle:
    return make_bundle(seed, specs)


def cohort_summary(arrays: CohortArrays) -> dict[str, float]:
    age = arrays.raw.get("age", arrays.x[:, index_of("age")])
    return {
        "n": float(arrays.n),
        "cases": float(arrays.y.sum()),
        "case_pct": round(100.0 * arrays.prevalence, 2),
        "median_age": round(float(np.median(age)), 1),
        "male_pct": round(100.0 * float(arrays.x[:, index_of("sex_male")].mean()), 1),
        "nonwhite_pct": round(100.0 * float(arrays.x[:, index_of("race_nonwhite")].mean()), 1),
        "private_insurance_pct": round(
            100.0 * float(arrays.x[:, index_of("insurance_private")].mean()), 1
        ),
    }
