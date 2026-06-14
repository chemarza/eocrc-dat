from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from eocrc_dat.actuarial.delong import delong_test
from eocrc_dat.actuarial.interaction import top_interactions
from eocrc_dat.actuarial.metrics import calibration, discrimination
from eocrc_dat.agency.schema import ExperimentConfig, ModelConfig, TrainConfig
from eocrc_dat.intake.cohorts import cohort_summary, propensity_match
from eocrc_dat.intake.synthetic import DEFAULT_BUNDLE, SplitSpec, make_bundle
from eocrc_dat.ledger.records import CohortArrays, CohortBundle, SplitName
from eocrc_dat.ledger.schema import Domain, index_of, names_in
from eocrc_dat.rating.engineering import apply_rating, fit_rating
from eocrc_dat.reserving.trainer import TrainResult, TrainSettings, predict_proba, train_model
from eocrc_dat.support.devices import pick_device
from eocrc_dat.support.seeding import set_seed
from eocrc_dat.tariff.scorecard import REFERENCE_AUROC, grade, score_array
from eocrc_dat.underwriters.dat import DATConfig, DualAttentionTransformer
from eocrc_dat.underwriters.ft_transformer import FTConfig, FTTransformerClassifier
from eocrc_dat.underwriters.mlp import MLPClassifier, MLPConfig
from eocrc_dat.underwriters.registry import REGISTRY, build_sklearn
from eocrc_dat.underwriters.saint import SaintClassifier, SaintConfig
from eocrc_dat.underwriters.tabnet import TabNetClassifier, TabNetConfig

EVAL_SPLITS: tuple[SplitName, ...] = ("internal_test", "temporal", "external")
ProbaFn = Callable[[CohortArrays], npt.NDArray[np.float64]]


def build_torch_model(cfg: ModelConfig, n_features: int) -> nn.Module:
    if cfg.method == "dat":
        return DualAttentionTransformer(
            DATConfig(
                n_features=n_features,
                k_steps=cfg.k_steps,
                gamma=cfg.gamma,
                tabnet_dim=cfg.tabnet_dim,
                n_glu=cfg.n_glu,
                q_select=cfg.q_select,
                ft_dim=cfg.ft_dim,
                ft_heads=cfg.ft_heads,
                ft_layers=cfg.ft_layers,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
            )
        )
    if cfg.method == "ft_transformer":
        return FTTransformerClassifier(
            FTConfig(
                n_features=n_features,
                dim=cfg.ft_dim,
                heads=cfg.ft_heads,
                layers=cfg.ft_layers,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
            )
        )
    if cfg.method == "tabnet":
        return TabNetClassifier(
            TabNetConfig(
                n_features=n_features,
                n_steps=cfg.k_steps,
                gamma=cfg.gamma,
                feature_dim=cfg.tabnet_dim,
                n_glu=cfg.n_glu,
            )
        )
    if cfg.method == "saint":
        return SaintClassifier(
            SaintConfig(
                n_features=n_features,
                dim=cfg.saint_dim,
                heads=cfg.saint_heads,
                layers=cfg.saint_layers,
                dropout=cfg.dropout,
            )
        )
    if cfg.method == "mlp":
        return MLPClassifier(
            MLPConfig(n_features=n_features, hidden=cfg.mlp_hidden, dropout=cfg.dropout)
        )
    raise KeyError(f"no torch model for method '{cfg.method}'")


def _scaled_specs(scale: float) -> tuple[SplitSpec, ...]:
    if scale >= 1.0:
        return DEFAULT_BUNDLE
    scaled: list[SplitSpec] = []
    for spec in DEFAULT_BUNDLE:
        size = max(int(spec.n * scale), 64)
        scaled.append(
            SplitSpec(
                spec.name, size, spec.prevalence, spec.signal_scale, spec.noise, spec.seed_offset
            )
        )
    return tuple(scaled)


def _drop_columns(arrays: CohortArrays, domains: tuple[str, ...]) -> CohortArrays:
    if not domains:
        return arrays
    masked = arrays.x.copy()
    for domain in domains:
        for name in names_in(Domain(domain)):
            masked[:, index_of(name)] = 0.0
    return CohortArrays(
        x=masked,
        y=arrays.y,
        feature_names=arrays.feature_names,
        groups=arrays.groups,
        raw=arrays.raw,
    )


def materialize(cfg: ExperimentConfig) -> CohortBundle:
    bundle = make_bundle(cfg.data.seed, _scaled_specs(cfg.data.scale))
    train = bundle["train"]
    if cfg.data.match:
        train = propensity_match(train, ratio=5, seed=cfg.data.seed)
    fitted = fit_rating(train, p=cfg.data.winsor_p)
    domains = cfg.data.drop_domains
    engineered: dict[SplitName, CohortArrays] = {
        "train": _drop_columns(apply_rating(fitted, train), domains)
    }
    for name in EVAL_SPLITS:
        engineered[name] = _drop_columns(apply_rating(fitted, bundle[name]), domains)
    return CohortBundle(splits=engineered)


def _settings(train_cfg: TrainConfig, seed: int) -> TrainSettings:
    return TrainSettings(
        max_epochs=train_cfg.max_epochs,
        batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        patience=train_cfg.patience,
        grad_accum=train_cfg.grad_accum,
        amp=train_cfg.amp,
        seed=seed,
        max_steps=train_cfg.max_steps,
    )


def fit_method(
    cfg: ExperimentConfig, bundle: CohortBundle
) -> tuple[ProbaFn, TrainResult | None, nn.Module | None]:
    method = cfg.model.method
    spec = REGISTRY[method]
    train = bundle["train"]
    if spec.backend == "torch":
        device = pick_device(cfg.train.device)
        model = build_torch_model(cfg.model, train.n_features)
        result = train_model(
            model,
            train,
            bundle["internal_test"],
            _settings(cfg.train, cfg.seed),
            device,
            cfg.train.lam,
        )
        if result.best_state:
            model.load_state_dict(result.best_state)

        def proba(arrays: CohortArrays) -> npt.NDArray[np.float64]:
            return predict_proba(model, arrays, device)

        return proba, result, model

    estimator: Any = build_sklearn(method, cfg.seed)
    estimator.fit(train.x, train.y)

    def proba_sklearn(arrays: CohortArrays) -> npt.NDArray[np.float64]:
        return np.asarray(estimator.predict_proba(arrays.x)[:, 1], dtype=np.float64)

    return proba_sklearn, None, None


def run_train(cfg: ExperimentConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    bundle = materialize(cfg)
    proba, result, _ = fit_method(cfg, bundle)
    probabilities = proba(bundle["internal_test"])
    auroc = float(roc_auc_score(bundle["internal_test"].y, probabilities))
    summary: dict[str, Any] = {
        "method": cfg.model.method,
        "experiment": cfg.name,
        "internal_test_auroc": round(auroc, 4),
        "effective_batch": cfg.train.effective_batch,
    }
    if result is not None:
        summary["best_val_auroc"] = round(result.best_auroc, 4)
        summary["epochs_run"] = result.epochs_run
    return summary


def run_evaluate(
    cfg: ExperimentConfig, with_interactions: bool = False, compare: str | None = None
) -> dict[str, Any]:
    set_seed(cfg.seed)
    bundle = materialize(cfg)
    proba, _, model = fit_method(cfg, bundle)
    report: dict[str, Any] = {"method": cfg.model.method, "experiment": cfg.name, "splits": {}}
    cached: dict[SplitName, npt.NDArray[np.float64]] = {}
    for name in EVAL_SPLITS:
        arrays = bundle[name]
        predictions = proba(arrays)
        cached[name] = predictions
        metrics = discrimination(arrays.y, predictions)
        metrics.update(calibration(arrays.y, predictions))
        report["splits"][name] = {key: round(value, 4) for key, value in metrics.items()}

    if compare is not None and compare in REGISTRY:
        rival_cfg = ExperimentConfig(
            name=cfg.name,
            seed=cfg.seed,
            model=ModelConfig(method=compare),
            data=cfg.data,
            train=cfg.train,
            thresholds=cfg.thresholds,
        )
        rival_proba, _, _ = fit_method(rival_cfg, bundle)
        rival = rival_proba(bundle["internal_test"])
        report["delong_internal"] = {
            key: round(value, 4)
            for key, value in delong_test(
                bundle["internal_test"].y, cached["internal_test"], rival
            ).items()
        }

    if with_interactions and isinstance(model, DualAttentionTransformer):
        report["interactions"] = _interaction_report(model, bundle["internal_test"])
    return report


def _interaction_report(
    model: DualAttentionTransformer, arrays: CohortArrays
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    features = torch.from_numpy(arrays.x[: min(256, arrays.n)]).to(device)
    trace = model.trace(features)
    attention = trace.attention.mean(dim=(0, 1)).detach().cpu().numpy()[1:, 1:]
    selected = trace.selected.detach().cpu().numpy()
    sub = attention[np.ix_(selected, selected)]
    names = tuple(arrays.feature_names[int(idx)] for idx in selected)
    pairs = top_interactions(sub.astype(np.float64), names, k=10)
    return [
        {"feature_a": pair.feature_a, "feature_b": pair.feature_b, "isr": round(pair.strength, 4)}
        for pair in pairs
    ]


def run_infer(cfg: ExperimentConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    bundle = materialize(cfg)
    return {name: cohort_summary(bundle[name]) for name in bundle.names()}


def run_score(cfg: ExperimentConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    bundle = materialize(cfg)
    report: dict[str, Any] = {
        "experiment": cfg.name,
        "reference_auroc": REFERENCE_AUROC,
        "splits": {},
    }
    for name in EVAL_SPLITS:
        arrays = bundle[name]
        totals = score_array(arrays.raw)
        grades = {"low": 0, "moderate": 0, "high": 0}
        for total in totals:
            grades[grade(int(total))] += 1
        report["splits"][name] = {
            "auroc": round(float(roc_auc_score(arrays.y, totals)), 4),
            "grade_counts": grades,
        }
    return report


def run_export(cfg: ExperimentConfig, out: str) -> dict[str, Any]:
    set_seed(cfg.seed)
    bundle = materialize(cfg)
    _, result, model = fit_method(cfg, bundle)
    if model is None:
        raise RuntimeError(
            f"method '{cfg.model.method}' is not a torch model and cannot be exported"
        )
    from eocrc_dat.reserving.checkpoint import save_checkpoint

    save_checkpoint(
        model,
        cfg.seed,
        {"method": cfg.model.method, "experiment": cfg.name},
        out,
    )
    return {"saved": out, "method": cfg.model.method}
