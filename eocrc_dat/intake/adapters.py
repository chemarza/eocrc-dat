from __future__ import annotations

from dataclasses import dataclass

from eocrc_dat.ledger.records import CohortArrays


@dataclass(frozen=True)
class CredentialedSource:
    key: str
    title: str
    portal: str

    def load(self, manifest: str) -> CohortArrays:
        raise RuntimeError(
            f"{self.title} is a credentialed cohort and cannot be redistributed. "
            f"Obtain access via {self.portal}, then point a custom loader at your "
            f"approved extract ('{manifest}'). This release ships only the synthetic "
            f"generator for code verification."
        )


SOURCES: dict[str, CredentialedSource] = {
    "trinetx": CredentialedSource("trinetx", "TriNetX Research Network", "https://trinetx.com"),
    "all_of_us": CredentialedSource(
        "all_of_us", "NIH All of Us Researcher Workbench", "https://researchallofus.org"
    ),
    "uk_biobank": CredentialedSource("uk_biobank", "UK Biobank", "https://www.ukbiobank.ac.uk"),
}


def get_source(key: str) -> CredentialedSource:
    if key not in SOURCES:
        raise KeyError(f"unknown cohort source: {key}")
    return SOURCES[key]
