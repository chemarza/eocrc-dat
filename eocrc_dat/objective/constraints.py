from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConstraints:
    tau1_auroc: float = 0.85
    tau2_slope: float = 0.15
    tau3_citl: float = 0.05
    tau4_dpr: float = 1.25
    t_inf_ms: float = 10.0

    def report(self, metrics: Mapping[str, float]) -> dict[str, bool]:
        dpr = metrics.get("dpr", 1.0)
        return {
            "auroc": metrics.get("auroc", 0.0) >= self.tau1_auroc,
            "calibration_slope": abs(metrics.get("calibration_slope", 1.0) - 1.0)
            <= self.tau2_slope,
            "citl": abs(metrics.get("citl", 0.0)) <= self.tau3_citl,
            "demographic_parity": (1.0 / self.tau4_dpr) <= dpr <= self.tau4_dpr,
            "latency": metrics.get("latency_ms", 0.0) <= self.t_inf_ms,
        }

    def passes(self, metrics: Mapping[str, float]) -> bool:
        return all(self.report(metrics).values())
