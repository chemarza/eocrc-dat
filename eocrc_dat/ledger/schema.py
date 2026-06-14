from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Kind = Literal["continuous", "binary"]


class Domain(str, Enum):
    DEMOGRAPHICS = "demographics"
    DIAGNOSES = "diagnoses"
    LABORATORY = "laboratory"
    MEDICATIONS = "medications"
    PROCEDURES = "procedures"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    domain: Domain
    kind: Kind


_DEMOGRAPHICS: tuple[tuple[str, Kind], ...] = (
    ("age", "continuous"),
    ("sex_male", "binary"),
    ("race_nonwhite", "binary"),
    ("insurance_private", "binary"),
    ("region_south", "binary"),
)

_LABORATORY: tuple[str, ...] = (
    "crp",
    "hemoglobin",
    "platelet_count",
    "albumin",
    "cea",
    "mcv",
    "ferritin",
    "wbc",
    "neutrophils",
    "lymphocytes",
    "alt",
    "ast",
    "alkaline_phosphatase",
    "bilirubin",
    "creatinine",
    "glucose",
    "hba1c",
    "ldh",
)

_DIAGNOSES_NAMED: tuple[str, ...] = (
    "iron_deficiency_anemia",
    "gi_hemorrhage",
    "inflammatory_bowel_disease",
    "type2_diabetes",
    "anal_fissure",
    "obesity",
    "metabolic_syndrome",
)

_MEDICATIONS_NAMED: tuple[str, ...] = (
    "nsaid_use",
    "ppi_use",
    "metformin_use",
    "statin_use",
    "immunosuppressant_use",
    "aspirin_use",
)

_PROCEDURES_NAMED: tuple[str, ...] = (
    "prior_colonoscopy",
    "prior_upper_endoscopy",
    "prior_abdominal_imaging",
)


def _build_catalogue() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    for name, kind in _DEMOGRAPHICS:
        specs.append(FeatureSpec(name, Domain.DEMOGRAPHICS, kind))
    for name in _LABORATORY:
        specs.append(FeatureSpec(name, Domain.LABORATORY, "continuous"))
    for name in _DIAGNOSES_NAMED:
        specs.append(FeatureSpec(name, Domain.DIAGNOSES, "binary"))
    for idx in range(len(_DIAGNOSES_NAMED) + 1, 51):
        specs.append(FeatureSpec(f"dx_ccs_{idx:02d}", Domain.DIAGNOSES, "binary"))
    for name in _MEDICATIONS_NAMED:
        specs.append(FeatureSpec(name, Domain.MEDICATIONS, "binary"))
    for idx in range(len(_MEDICATIONS_NAMED) + 1, 13):
        specs.append(FeatureSpec(f"rx_class_{idx:02d}", Domain.MEDICATIONS, "binary"))
    for name in _PROCEDURES_NAMED:
        specs.append(FeatureSpec(name, Domain.PROCEDURES, "binary"))
    for idx in range(len(_PROCEDURES_NAMED) + 1, 7):
        specs.append(FeatureSpec(f"proc_{idx:02d}", Domain.PROCEDURES, "binary"))
    return tuple(specs)


FEATURE_CATALOGUE: tuple[FeatureSpec, ...] = _build_catalogue()
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_CATALOGUE)
_INDEX: dict[str, int] = {spec.name: i for i, spec in enumerate(FEATURE_CATALOGUE)}


def index_of(name: str) -> int:
    return _INDEX[name]


def names_in(domain: Domain) -> tuple[str, ...]:
    return tuple(spec.name for spec in FEATURE_CATALOGUE if spec.domain is domain)


def continuous_mask() -> tuple[bool, ...]:
    return tuple(spec.kind == "continuous" for spec in FEATURE_CATALOGUE)


PROTECTED_GROUPS: tuple[str, ...] = ("age_band", "sex", "race", "region")
