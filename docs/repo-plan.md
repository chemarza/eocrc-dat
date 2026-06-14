# Repo Plan — EOCRC-DAT

## Organising metaphor
The package is laid out as an underwriting workflow: clinical records enter a
`ledger`, pass through `intake` and `rating`, are assessed by `underwriters`
against an `objective`, priced by a `reserving` engine and a point `tariff`, and
audited by the `actuarial` desk; the `agency` exposes the operations.

## Directory tree

    eocrc_dat/
    ├── __init__.py
    ├── py.typed
    ├── ledger/
    │   ├── __init__.py
    │   ├── schema.py          # Domain enum, FEATURE_CATALOGUE (~90 features)
    │   └── records.py         # CohortArrays, Split, group ids
    ├── intake/
    │   ├── __init__.py
    │   ├── synthetic.py       # SyntheticCohort (deterministic, signal + interactions)
    │   ├── cohorts.py         # propensity_match (1:5), make_splits, cohort summary
    │   └── adapters.py        # TriNetX / All of Us / UK Biobank credentialed stubs
    ├── rating/
    │   ├── __init__.py
    │   └── engineering.py     # winsorize, mice_impute, zscore, engineer
    ├── underwriters/
    │   ├── __init__.py
    │   ├── sparsemax.py       # Formula 2
    │   ├── tabnet.py          # Eq.2 sparse mask + feature transformer
    │   ├── ft_transformer.py  # Eq.3 + Formula 4 tokeniser
    │   ├── dat.py             # Algorithm 1 (proposed model)
    │   ├── saint.py
    │   ├── mlp.py
    │   ├── boosting.py        # XGBoost / LightGBM / CatBoost / GBDT
    │   ├── ensemble.py        # RandomForest / ExtraTrees / AdaBoost
    │   ├── kernel.py          # SVM-RBF
    │   ├── linear.py          # LR / SVM-Linear
    │   ├── probabilistic.py   # Naive Bayes
    │   ├── instance.py        # KNN
    │   └── registry.py        # DAT + 16 baselines, family map
    ├── objective/
    │   ├── __init__.py
    │   ├── losses.py          # Eq.1 BCE + lambda*||Theta||^2
    │   └── constraints.py     # Eq.1 tau1..tau4, t_inf gates
    ├── reserving/
    │   ├── __init__.py
    │   ├── schedule.py        # cosine LR + warmup
    │   ├── checkpoint.py      # atomic save/load, seed persisted
    │   ├── trainer.py         # Algorithm 1 loop, early stop on val AUROC
    │   └── tuning.py          # Optuna 100x15x5 harness
    ├── tariff/
    │   ├── __init__.py
    │   └── scorecard.py       # Table V 10-item score, grade banding
    ├── actuarial/
    │   ├── __init__.py
    │   ├── metrics.py         # discrimination + calibration
    │   ├── delong.py          # DeLong test
    │   ├── bootstrap.py       # 2000-resample CI
    │   ├── multiplicity.py    # Holm-Bonferroni, BH-FDR
    │   ├── decision_curve.py  # Eq.4 net benefit
    │   ├── fairness.py        # demographic-parity ratio
    │   └── interaction.py     # Eq.5 ISR + IR, Eq.5b additive degradation
    ├── agency/
    │   ├── __init__.py
    │   ├── schema.py          # frozen config dataclasses
    │   ├── config.py          # TOML loader with extends chain
    │   ├── commands.py        # train / evaluate / infer / score / export
    │   └── cli.py             # Tap typed args + verb dispatch
    └── support/
        ├── __init__.py
        ├── logging.py
        ├── seeding.py
        ├── io.py
        └── devices.py

    configs/
    ├── model/        {dat,ft_transformer,tabnet,saint,mlp,boosting,...}.toml
    ├── data/         {trinetx,all_of_us,uk_biobank,synthetic}.toml
    ├── train/        {default,_smoke}.toml
    └── experiment/   main.toml, ablation_*.toml, supplementary_*.toml, _smoke.toml

    tests/            (varied kinds — see below)
    docs/             project-context.md, implementation-map.md, repo-plan.md, deviations.md
    scripts/          launch_train.sh, launch_eval.sh, prepare_data.sh
    assets/

## Module responsibilities (one line each)
- `ledger` — the canonical typed feature schema and cohort containers; no logic beyond shape contracts.
- `intake` — produce a cohort (synthetic generator or credentialed adapter), match, split.
- `rating` — turn a raw cohort into the engineered model matrix.
- `underwriters` — every model in Table II behind a uniform fit/predict surface; DAT is the proposed model.
- `objective` — the training loss and the Eq.1 acceptance constraints.
- `reserving` — train one model, schedule the LR, checkpoint atomically, tune.
- `tariff` — the deployable 10-point score.
- `actuarial` — all reported statistics and curves.
- `agency` — configuration and command-line surface.
- `support` — logging, seeding, device, atomic IO.

## Dependencies (pinned ranges)
- python >= 3.10
- torch >= 2.1
- numpy >= 1.24, scipy >= 1.11, pandas >= 2.0
- scikit-learn >= 1.3
- xgboost >= 2.0
- typed-argument-parser >= 1.10  (Tap)
- optional extras: lightgbm >= 4.0, catboost >= 1.2, optuna >= 3.4
- dev: pytest, ruff, mypy, black, isort, pre-commit

## Test plan (kinds)
- shape — tensor shapes through tokeniser / TabNet / FT-Transformer / DAT.
- gradient-flow — every DAT parameter receives a finite non-zero gradient.
- overfit-single-batch — DAT drives BCE near zero on one batch.
- invariant — sparsemax sums to 1 with exact zeros; ISR symmetry and self=1; net benefit at treat-all/treat-none; calibration identity on perfect probabilities.
- metric-correctness — AUROC / AUPRC / Brier vs scikit-learn; DeLong vs a reference variance.
- numerical-regression — synthetic-cohort summary and a fixed-seed forward pass pinned to stored values.
- config-integrity — effective batch = batch x grad_accum x world_size; Eq.1 tau thresholds; smoke config flagged unit-test-only.
- style-guard — no inline comments, no docstrings, no forbidden phrases / emoji across package + README + Makefile.
- e2e smoke — `tests/test_training_smoke.py` trains 2 steps on `_smoke.toml` and asserts loss decreases.
