from __future__ import annotations

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FloatArray = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]


def _logit(p: FloatArray) -> FloatArray:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def youden_operating_point(y: Labels, p: FloatArray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    youden = tpr - fpr
    return float(thresholds[int(np.argmax(youden))])


def discrimination(y: Labels, p: FloatArray) -> dict[str, float]:
    threshold = youden_operating_point(y, p)
    predicted = (p >= threshold).astype(np.int64)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else 0.0
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "nns": 1.0 / ppv if ppv else float("inf"),
        "threshold": threshold,
    }


def calibration(y: Labels, p: FloatArray) -> dict[str, float]:
    brier = float(np.mean((p - y) ** 2))
    offset = _logit(p)
    citl_model = sm.GLM(
        y.astype(np.float64),
        np.ones((y.shape[0], 1)),
        family=sm.families.Binomial(),
        offset=offset,
    ).fit()
    design = sm.add_constant(offset)
    slope_model = sm.GLM(y.astype(np.float64), design, family=sm.families.Binomial()).fit()
    return {
        "brier": brier,
        "citl": float(citl_model.params[0]),
        "calibration_slope": float(slope_model.params[1]),
    }
