from __future__ import annotations

import numpy as np
import numpy.typing as npt

Labels = npt.NDArray[np.int64]
Groups = npt.NDArray[np.str_]


def demographic_parity_ratio(flagged: Labels, group: Groups) -> dict[str, float]:
    rates: dict[str, float] = {}
    for value in np.unique(group):
        mask = group == value
        if mask.sum() == 0:
            continue
        rates[str(value)] = float(flagged[mask].mean())
    if not rates:
        ratio = 1.0
    else:
        values = list(rates.values())
        ratio = max(values) / max(min(values), 1e-6)
    summary: dict[str, float] = {f"rate[{key}]": value for key, value in rates.items()}
    summary["dpr"] = ratio
    return summary
