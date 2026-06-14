from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from eocrc_dat.ledger.records import CohortArrays, CohortBundle, FloatArray, SplitName
from eocrc_dat.ledger.schema import FEATURE_CATALOGUE, FEATURE_NAMES, index_of

_CONT_PARAMS: dict[str, tuple[float, float]] = {
    "age": (40.0, 8.5),
    "crp": (4.0, 5.0),
    "hemoglobin": (13.4, 1.6),
    "platelet_count": (270.0, 70.0),
    "albumin": (4.2, 0.45),
    "cea": (2.4, 1.8),
    "mcv": (89.0, 6.0),
    "ferritin": (90.0, 55.0),
    "wbc": (7.2, 2.0),
    "neutrophils": (4.3, 1.4),
    "lymphocytes": (2.1, 0.7),
    "alt": (26.0, 12.0),
    "ast": (24.0, 10.0),
    "alkaline_phosphatase": (78.0, 22.0),
    "bilirubin": (0.7, 0.3),
    "creatinine": (0.9, 0.2),
    "glucose": (98.0, 18.0),
    "hba1c": (5.5, 0.7),
    "ldh": (190.0, 40.0),
}

_BIN_PREV: dict[str, float] = {
    "sex_male": 0.47,
    "race_nonwhite": 0.38,
    "insurance_private": 0.58,
    "region_south": 0.36,
    "iron_deficiency_anemia": 0.08,
    "gi_hemorrhage": 0.05,
    "inflammatory_bowel_disease": 0.04,
    "type2_diabetes": 0.12,
    "anal_fissure": 0.03,
    "obesity": 0.28,
    "metabolic_syndrome": 0.15,
    "nsaid_use": 0.30,
    "ppi_use": 0.20,
    "metformin_use": 0.10,
    "statin_use": 0.18,
    "immunosuppressant_use": 0.03,
    "aspirin_use": 0.22,
    "prior_colonoscopy": 0.12,
    "prior_upper_endoscopy": 0.08,
    "prior_abdominal_imaging": 0.20,
}

_LINEAR_WEIGHTS: dict[str, float] = {
    "crp": 0.75,
    "hemoglobin": -0.65,
    "platelet_count": 0.32,
    "iron_deficiency_anemia": 0.95,
    "gi_hemorrhage": 1.15,
    "inflammatory_bowel_disease": 0.85,
    "type2_diabetes": 0.62,
    "obesity": 0.52,
    "anal_fissure": 0.40,
    "metabolic_syndrome": 0.50,
    "nsaid_use": -0.18,
    "metformin_use": -0.12,
}


@dataclass(frozen=True)
class SplitSpec:
    name: SplitName
    n: int
    prevalence: float
    signal_scale: float
    noise: float
    seed_offset: int


DEFAULT_BUNDLE: tuple[SplitSpec, ...] = (
    SplitSpec("train", 2400, 0.167, 1.00, 0.55, 0),
    SplitSpec("internal_test", 600, 0.167, 1.00, 0.55, 1),
    SplitSpec("temporal", 600, 0.083, 0.92, 0.70, 2),
    SplitSpec("external", 500, 0.041, 0.84, 0.95, 3),
)


def _padded_prevalence(name: str) -> float:
    if name.startswith("dx_ccs_"):
        return 0.05
    if name.startswith("rx_class_"):
        return 0.10
    if name.startswith("proc_"):
        return 0.05
    return 0.05


def _standardize(column: FloatArray, name: str) -> FloatArray:
    loc, scale = _CONT_PARAMS[name]
    return ((column - loc) / scale).astype(np.float32)


def _solve_intercept(linear: FloatArray, target: float) -> float:
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rate = float((1.0 / (1.0 + np.exp(-(linear + mid)))).mean())
        if rate < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class SyntheticCohort:
    def __init__(self, catalogue_size: int = len(FEATURE_CATALOGUE)) -> None:
        self.catalogue_size = catalogue_size

    def generate(self, spec: SplitSpec, seed: int) -> CohortArrays:
        rng = np.random.default_rng((seed + 1) * 7919 + spec.seed_offset)
        n = spec.n
        x = np.zeros((n, self.catalogue_size), dtype=np.float32)
        for feature in FEATURE_CATALOGUE:
            col = index_of(feature.name)
            if feature.kind == "continuous":
                loc, scale = _CONT_PARAMS[feature.name]
                values = rng.normal(loc, scale, size=n)
                if feature.name in {"crp", "cea", "ferritin"}:
                    values = np.clip(values, 0.0, None)
                x[:, col] = values.astype(np.float32)
            else:
                prev = _BIN_PREV.get(feature.name, _padded_prevalence(feature.name))
                x[:, col] = (rng.random(n) < prev).astype(np.float32)

        crp_z = _standardize(x[:, index_of("crp")], "crp")
        hb_z = _standardize(x[:, index_of("hemoglobin")], "hemoglobin")
        plt_z = _standardize(x[:, index_of("platelet_count")], "platelet_count")
        age_z = _standardize(x[:, index_of("age")], "age")

        linear = np.asarray(0.70 * age_z, dtype=np.float64)
        for name, weight in _LINEAR_WEIGHTS.items():
            spec_obj = FEATURE_CATALOGUE[index_of(name)]
            column = x[:, index_of(name)]
            contribution = _standardize(column, name) if spec_obj.kind == "continuous" else column
            linear = linear + weight * contribution
        ida = x[:, index_of("iron_deficiency_anemia")]
        t2d = x[:, index_of("type2_diabetes")]
        fissure = x[:, index_of("anal_fissure")]
        obese = x[:, index_of("obesity")]
        linear = linear + 1.70 * crp_z * ida
        linear = linear + 1.05 * np.clip(-hb_z, 0.0, None) * plt_z
        linear = linear + 1.30 * t2d * fissure
        linear = linear + 0.80 * obese * t2d
        linear = linear + 0.90 * (np.abs(crp_z) > 1.0).astype(np.float64) * (age_z > 0.0).astype(
            np.float64
        )
        linear = spec.signal_scale * linear
        linear = linear + rng.normal(0.0, spec.noise, size=n)

        intercept = _solve_intercept(linear.astype(np.float32), spec.prevalence)
        probs = 1.0 / (1.0 + np.exp(-(linear + intercept)))
        y = (rng.random(n) < probs).astype(np.int64)

        groups = self._groups(x, rng)
        raw = self._raw(x, rng)
        return CohortArrays(
            x=x,
            y=y.astype(np.int64),
            feature_names=FEATURE_NAMES,
            groups=groups,
            raw=raw,
        )

    def _groups(self, x: FloatArray, rng: np.random.Generator) -> dict[str, npt.NDArray[np.str_]]:
        age = x[:, index_of("age")]
        band = np.where(age < 30, "18-29", np.where(age < 40, "30-39", "40-49"))
        sex = np.where(x[:, index_of("sex_male")] > 0.5, "male", "female")
        race = np.where(x[:, index_of("race_nonwhite")] > 0.5, "nonwhite", "white")
        region = np.where(x[:, index_of("region_south")] > 0.5, "south", "other")
        return {
            "age_band": band.astype(np.str_),
            "sex": sex.astype(np.str_),
            "race": race.astype(np.str_),
            "region": region.astype(np.str_),
        }

    def _raw(self, x: FloatArray, rng: np.random.Generator) -> dict[str, FloatArray]:
        n = x.shape[0]
        obesity = x[:, index_of("obesity")]
        bmi = np.where(obesity > 0.5, rng.normal(32.5, 3.0, n), rng.normal(25.5, 3.0, n))
        bmi = np.clip(bmi, 16.0, 55.0)
        hb = x[:, index_of("hemoglobin")]
        male = x[:, index_of("sex_male")] > 0.5
        floor = np.where(male, 13.0, 12.0)
        hb_sev = np.where(
            hb >= floor,
            0.0,
            np.where(hb >= 11.0, 1.0, np.where(hb >= 9.0, 2.0, 3.0)),
        )

        def ordinal(flag_name: str, scale: float) -> FloatArray:
            flag = x[:, index_of(flag_name)]
            draw = rng.random(n)
            level = np.where(flag > 0.5, 1.0 + np.floor(draw * 3.0), 0.0)
            return np.clip(level * scale, 0.0, 3.0).astype(np.float32)

        return {
            "iron_deficiency_anemia": ordinal("iron_deficiency_anemia", 1.0),
            "crp": x[:, index_of("crp")].astype(np.float32),
            "gi_hemorrhage": ordinal("gi_hemorrhage", 1.0),
            "obesity_bmi": bmi.astype(np.float32),
            "hemoglobin_severity": hb_sev.astype(np.float32),
            "type2_diabetes": ordinal("type2_diabetes", 1.0),
            "ibd": ordinal("inflammatory_bowel_disease", 1.0),
            "family_history_crc": np.clip(np.floor(rng.random(n) * 1.4), 0.0, 3.0).astype(
                np.float32
            ),
            "processed_meat": np.clip(
                np.asarray(rng.poisson(1.6, n), dtype=np.float32), 0.0, 12.0
            ).astype(np.float32),
            "age": x[:, index_of("age")].astype(np.float32),
        }


def make_bundle(seed: int, specs: tuple[SplitSpec, ...] = DEFAULT_BUNDLE) -> CohortBundle:
    cohort = SyntheticCohort()
    splits: dict[SplitName, CohortArrays] = {}
    for spec in specs:
        splits[spec.name] = cohort.generate(spec, seed)
    return CohortBundle(splits=splits)
