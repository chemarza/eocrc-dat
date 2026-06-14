# Implementation Map — EOCRC-DAT

Source provenance lives here, not in the source files. Every numbered equation,
the algorithm box, every reported table, every ablation and every metric in the
manuscript maps to a module below. Section/equation/table identifiers are the
manuscript's own.

## Models and core mathematics

| paper anchor | item | file | symbol / API | notes |
|---|---|---|---|---|
| §IV-D-1, Eq.2, Formula 1-2 | TabNet sparse attention mask M_k = sparsemax(P_k · h_k(a_{k-1})) | `eocrc_dat/underwriters/tabnet.py` | `TabNetEncoder` | K=5 steps, relaxation gamma=1.3, embedding dim 64; prior update P_{k+1}=P_k·(gamma−M_k) |
| §IV-D, Formula 2 | sparsemax (exact-zero projection onto the simplex) | `eocrc_dat/underwriters/sparsemax.py` | `sparsemax` | row sums to 1, produces exact zeros (distinct from softmax) |
| §IV-D-2, Eq.3, Formula 3 | FT-Transformer self-attention softmax(QKᵀ/√d_h)V | `eocrc_dat/underwriters/ft_transformer.py` | `FTTransformerEncoder` | L=3 layers, H=8 heads, token dim d=192; pairwise interactions over C(n+1,2) tokens |
| §IV-D-2, Formula 4 | feature tokenisation T=[t_CLS; t_1..t_n], t_j=W_j x_j + b_j | `eocrc_dat/underwriters/ft_transformer.py` | `FeatureTokenizer` | continuous + categorical to unified d-dim; prepend [CLS] |
| §IV-D-3, Algorithm 1 | DAT forward pass + training loop | `eocrc_dat/underwriters/dat.py` | `DualAttentionTransformer` | TabNet selects top-q features → FT-Transformer pairwise attention → ŷ=σ(w·Z_CLS); K=5,L=3,H=8,d=192,gamma=1.3,eta=1e-3,lambda=1e-5 |
| §IV-D | SAINT baseline (row+column attention) | `eocrc_dat/underwriters/saint.py` | `SaintEncoder` | deep-tabular baseline |
| §IV-D | MLP baseline | `eocrc_dat/underwriters/mlp.py` | `MLPClassifier` | deep-tabular baseline |
| Table II | gradient-boosting family (XGBoost, LightGBM, CatBoost, GBDT) | `eocrc_dat/underwriters/boosting.py` | `build_boosting` | LightGBM/CatBoost optional, guarded import; XGBoost + sklearn GBDT always available |
| Table II | ensemble trees (Random Forest, Extra Trees) + AdaBoost | `eocrc_dat/underwriters/ensemble.py` | `build_ensemble`, `build_adaboost` | sklearn |
| Table II | SVM-RBF kernel method | `eocrc_dat/underwriters/kernel.py` | `build_kernel` | sklearn SVC(rbf), probability |
| Table II | LR, SVM-Linear | `eocrc_dat/underwriters/linear.py` | `build_linear` | sklearn |
| Table II | Naive Bayes | `eocrc_dat/underwriters/probabilistic.py` | `build_naive_bayes` | sklearn GaussianNB |
| Table II | KNN | `eocrc_dat/underwriters/instance.py` | `build_knn` | sklearn |
| Table II caption | method registry: DAT (ours) + 16 baselines | `eocrc_dat/underwriters/registry.py` | `REGISTRY`, `FAMILIES` | the manuscript's "16 techniques" = the 16 baselines; DAT is the proposed model |

## Objective, constraints, training

| paper anchor | item | file | symbol / API | notes |
|---|---|---|---|---|
| §IV-A, Eq.1 | constrained objective Θ*=argmin L(f_Θ(X),y)+λΩ(Θ) | `eocrc_dat/objective/losses.py` | `bce_with_l2` | L = binary cross-entropy; Ω = ‖Θ‖² penalised by lambda=1e-5 |
| §IV-A, Eq.1 | selection constraints (tau1..tau4, t_inf) | `eocrc_dat/objective/constraints.py` | `SelectionConstraints`, `passes` | AUROC≥0.85, |slope−1|≤0.15, |CITL|≤0.05, DPR∈[1/1.25,1.25], latency≤10 ms |
| Algorithm 1 | AdamW + cosine LR + warmup, early stop patience 20 on val AUROC | `eocrc_dat/reserving/trainer.py`, `eocrc_dat/reserving/schedule.py` | `Trainer`, `cosine_warmup` | monitors validation AUROC; AMP optional; DDP-ready |
| R4 | atomic checkpoint write (tmp + os.replace), seed stored + restored | `eocrc_dat/reserving/checkpoint.py`, `eocrc_dat/support/seeding.py` | `save_atomic`, `set_seed` | seed persisted in every checkpoint |
| §II-A, §IV-B, §IV-E | Optuna tuning protocol: 100 trials/method, 15 seeds, 5-fold CV | `eocrc_dat/reserving/tuning.py` | `tune` | search-space hooks per family; coordinated compute budget |

## Data and feature engineering

| paper anchor | item | file | symbol / API | notes |
|---|---|---|---|---|
| §IV-B, §IV-C, Fig.1 | five clinical-domain feature catalogue (~90 features) | `eocrc_dat/ledger/schema.py` | `FEATURE_CATALOGUE`, `Domain` | Demographics(5), Diagnoses(50), Laboratory(15-20), Medications(10-15), Procedures(5-8) |
| §IV-B | typed cohort batch / arrays | `eocrc_dat/ledger/records.py` | `CohortArrays`, `Split` | feature matrix X∈R^{N×p}, label y∈{0,1}^N, group ids |
| §IV-C | feature-engineering protocol: winsorise 1/99, MICE impute (<40% missing), z-score | `eocrc_dat/rating/engineering.py` | `winsorize`, `mice_impute`, `zscore`, `engineer` | standardisation uses training statistics only |
| §IV-B, Fig.1b | propensity matching 1:5 on age/sex/race/region; temporal + geographic splits | `eocrc_dat/intake/cohorts.py` | `propensity_match`, `make_splits` | dev 2016-2020, temporal 2021-2022, external held-out |
| §IV-B, Table I, Fig.1 | epidemiology-grounded synthetic cohort generator (cross-domain signal) | `eocrc_dat/intake/synthetic.py` | `SyntheticCohort` | deterministic; recoverable EOCRC signal + CRP×IDA / Hb×platelet interactions |
| §Data Availability | credentialed adapters (TriNetX, All of Us, UK Biobank) | `eocrc_dat/intake/adapters.py` | `CredentialedSource` | raises informative error; no redistributable data |

## Evaluation and statistics

| paper anchor | item | file | symbol / API | notes |
|---|---|---|---|---|
| §IV-E, Table II | AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, NNS, Brier | `eocrc_dat/actuarial/metrics.py` | `discrimination`, `youden_operating_point`, `nns` | NNS at Youden-optimal threshold |
| §IV-E, Fig.2c | calibration: Brier, calibration-in-the-large (CITL), slope | `eocrc_dat/actuarial/metrics.py` | `calibration` | CITL and slope from logistic recalibration |
| §IV-E, Table II, Fig.3k | DeLong test for correlated AUROCs | `eocrc_dat/actuarial/delong.py` | `delong_test` | DAT vs XGBoost p=0.003 (internal) |
| §IV-E | bootstrap 95% CI (2,000 resamples) | `eocrc_dat/actuarial/bootstrap.py` | `bootstrap_ci` | stratified resampling |
| §IV-E | Holm-Bonferroni (6 primary), Benjamini-Hochberg FDR q=0.05 (exploratory) | `eocrc_dat/actuarial/multiplicity.py` | `holm_bonferroni`, `benjamini_hochberg` | |
| §IV-E, Eq.4, Fig.2/3 | decision-curve net benefit NB(p_t)=TP/N − FP/N · p_t/(1−p_t) | `eocrc_dat/actuarial/decision_curve.py` | `net_benefit`, `decision_curve` | thresholds 1%/2%/5%; treat-all / treat-none references |
| §II / fairness, Fig.3n | demographic-parity ratio across protected groups | `eocrc_dat/actuarial/fairness.py` | `demographic_parity_ratio` | DPR within [1/1.25, 1.25] |
| §II-C, Eq.5, Table IV | interaction strength ratio ISR(A,B)=w(A,B)/√(w(A)·w(B)); IR=Δ_combined/(Δ_TabNet+Δ_FT)=1.19 | `eocrc_dat/actuarial/interaction.py` | `isr`, `interaction_ratio` | top pair CRP×IDA ISR=2.34, attn 0.087 |
| §II-A note, Eq.5b | expected additive degradation Δ_exp=√(Σ Δ_k² + 2 Σ_{i<j} Cov_f(Δ_i,Δ_j)) | `eocrc_dat/actuarial/interaction.py` | `expected_additive_degradation` | used in component-ablation reasoning |

## Simplified score

| paper anchor | item | file | symbol / API | notes |
|---|---|---|---|---|
| §II-D, Table V | 10-item DAT-EOCRC point score (0-24), grades low 0-8 / moderate 9-15 / high ≥16 | `eocrc_dat/tariff/scorecard.py` | `SCORECARD`, `score_rows`, `grade` | AUROC 0.862 / 0.841 / 0.819 (internal / temporal / external) |

## Tables / ablations coverage

| table / ablation | where reproduced | command |
|---|---|---|
| Table I (cohort characteristics) | `eocrc_dat/intake/cohorts.py` summary + synthetic generator targets | `eocrc-dat infer --describe-cohort` |
| Table II (16 baselines + DAT × dev/temporal/external) | `agency` evaluate over `configs/experiment/main.toml` + family configs | `eocrc-dat evaluate --experiment main` |
| Table III (component / pairwise / input-feature ablation) | `configs/experiment/ablation_*.toml` | `eocrc-dat train --experiment ablation_component_tabnet` |
| Table IV (top-10 cross-domain interactions) | `eocrc_dat/actuarial/interaction.py` | `eocrc-dat evaluate --experiment main --interactions` |
| Table V (risk-score elements) | `eocrc_dat/tariff/scorecard.py` | `eocrc-dat score --experiment main` |

## Config and CLI

| concern | file | notes |
|---|---|---|
| frozen config dataclasses (Model/Data/Train/Experiment) | `eocrc_dat/agency/schema.py` | mypy-strict typed |
| TOML loader with `extends` chain → dataclasses | `eocrc_dat/agency/config.py` | stdlib `tomllib` |
| Tap typed CLI + verb dispatch (train/evaluate/infer/score/export) | `eocrc_dat/agency/cli.py`, `eocrc_dat/agency/commands.py` | entry point `eocrc-dat` |

## Code-availability statement (verbatim, manuscript p. 9-10) — kept here, not in README

> The implementation code (PyTorch 2.x for DAT/TabNet/FT-Transformer; scikit-learn
> 1.x classical baseline; XGBoost 2.x, LightGBM 4; CatBoost 1.x) and analysis
> notebooks are available to peer reviewers and editors upon request during the
> peer review process (private repository access). Upon acceptance of the
> manuscript, the implementation will be made available as an open public
> repository with an archived version of the complete code and data (Zenodo DOI)
> for the community to reuse and to reproduce this study.
