# EOCRC-DAT — Risk Stratification Policy Document

This README is written as an insurance policy: it states what the code release
covers, the terms under which it operates, and the benefits you can claim by
running it. The "insured asset" is the Dual-Attention Transformer (DAT) risk
model for early-onset colorectal cancer (EOCRC) built on nationwide electronic
health records.

---

## Declarations

| Item | Value |
|------|-------|
| Policy form | `eocrc_dat` (Python package) |
| Insured model | Dual-Attention Transformer (DAT) = TabNet sparse selection + FT-Transformer pairwise attention |
| Class of risk | EOCRC (age < 50) binary risk stratification on tabular EHR |
| Reporting standard | TRIPOD+AI |
| Named perils covered | discrimination, calibration, decision-curve utility, subgroup fairness, cross-domain interaction discovery |
| Schedule of cover | training, evaluation, single-record inference, point-score grading, model export |
| Period of cover | from `pip install` to a reproduced main-result table |

## Insuring agreement

This release implements the risk framework described in the accompanying
manuscript. DAT first uses TabNet's sparse sequential attention to narrow the
feature space, then applies FT-Transformer pairwise self-attention over the
retained features, producing a single risk probability and an attention map that
ranks cross-domain feature interactions. The package also carries the sixteen
baseline methods the manuscript compares against, the simplified ten-item
DAT-EOCRC point score, and the statistical machinery (DeLong, bootstrap
intervals, Holm-Bonferroni, Benjamini-Hochberg, decision-curve net benefit,
demographic-parity ratio) used to judge the model.

## Definitions

- **DAT** — the proposed model (Algorithm 1); `eocrc_dat/underwriters/dat.py`.
- **EOCRC** — early-onset colorectal cancer, the binary outcome.
- **ISR** — interaction strength ratio, `w(A,B)/sqrt(w(A)*w(B))` (Table IV).
- **CITL** — calibration-in-the-large; **slope** — calibration slope.
- **NNS** — number needed to screen at the operating threshold.
- **DPR** — demographic-parity ratio across protected subgroups.

## Section I — How to bind cover (installation)

pip:

    pip install -e .
    pip install -e ".[boosting,tuning,dev]"   # optional: LightGBM/CatBoost, Optuna, dev tools

conda:

    conda env create -f environment.yml
    conda activate eocrc-dat

Docker:

    docker build -t eocrc-dat .
    docker run --rm eocrc-dat train --experiment main

A default install runs on CPU and needs only PyTorch, scikit-learn and XGBoost.
LightGBM and CatBoost are optional and guarded at import.

## Schedule of insured data

No raw cohort is redistributable; access is credentialed. The release ships a
deterministic synthetic cohort that mirrors the Table I prevalence structure and
the cross-domain signal, so the pipeline can be exercised end to end without the
private data.

| Cohort | Role | N | Cases | Access |
|--------|------|---|-------|--------|
| TriNetX | development + temporal | 41,052 | 6,842 | TriNetX Research Network (credentialed) |
| All of Us | external (US) validation | 4,782 | 196 | NIH All of Us Researcher Workbench (registered + controlled tier) |
| UK Biobank | international validation | 8,124 | 327 | UK Biobank application (project approval) |

Generate and describe the synthetic cohort:

    eocrc-dat infer --experiment main --describe_cohort

## Schedule of benefits (expected results)

Run each command on the approved cohort to claim the corresponding benefit. The
expected values are the manuscript's reported figures; reproduce them within the
stated tolerance on a five-seed average.

| Benefit (paper item) | Command | Expected |
|----------------------|---------|----------|
| DAT internal test AUROC (Table II) | `eocrc-dat evaluate --experiment main` | 0.896 (95% CI 0.884-0.908) |
| DAT temporal AUROC | `eocrc-dat evaluate --experiment main` | 0.879 +/- 0.015 |
| DAT external (All of Us) AUROC | `eocrc-dat evaluate --experiment main` | 0.854 +/- 0.019 |
| DAT international (UK Biobank) AUROC | `eocrc-dat evaluate --experiment main` | 0.834 |
| Best baseline (XGBoost) internal AUROC | `eocrc-dat evaluate --experiment main --compare xgboost` | 0.871 |
| DAT vs XGBoost (DeLong) | `eocrc-dat evaluate --experiment main --compare xgboost` | p = 0.003 |
| Top cross-domain interaction CRP x IDA (Table IV) | `eocrc-dat evaluate --experiment main --interactions` | ISR 2.34 |
| Simplified DAT-EOCRC score AUROC (Table V) | `eocrc-dat score --experiment main` | 0.862 / 0.841 / 0.819 |

## Premium and limits (compute budget)

The manuscript reports no GPU type, count, wall-clock or storage, so this release
does not invent any. The honest position:

| Resource | Value |
|----------|-------|
| Hardware | not reported by the manuscript (`COMPUTE_NOT_REPORTED`) |
| One DAT fit | minutes on a single consumer GPU; feasible on CPU |
| Tuning budget | 100 Optuna trials x 15 seeds x 5-fold CV per method (the dominant cost) |
| Operating limit | inference latency target <= 10 ms per record (Eq. 1 constraint) |
| Model size | order 10^6 parameters (L=3, H=8, d=192, K=5 over ~90 features) |

## Conditions

- The selection constraints of Eq. 1 are enforced in
  `eocrc_dat/objective/constraints.py`: AUROC >= 0.85, |slope - 1| <= 0.15,
  |CITL| <= 0.05, DPR within [1/1.25, 1.25], latency <= 10 ms.
- Every checkpoint stores its seed; `set_seed` restores it on resume.
- Continuous features are winsorised at the 1st/99th percentile, imputed by MICE
  when missing, and z-scored using training statistics only.
- Ethics: only de-identified TriNetX and All of Us data (IRB-approved or exempt)
  and a UK Biobank application were analysed; no patients were contacted and no
  new clinical data were gathered. No competing interests are declared.

## Exclusions (what this policy does not cover)

- Absolute metrics from the bundled synthetic cohort are not the manuscript's
  numbers; those require the approved credentialed data.
- No trained weights are shipped; checkpoints are available upon reasonable
  request.
- LightGBM and CatBoost results need the optional `boosting` extra installed.
- `docker build` was not verified on this machine (no Docker daemon present).

## Endorsements (ablations and supplementary runs)

| Endorsement | Command |
|-------------|---------|
| Component: TabNet only | `eocrc-dat train --experiment ablation_component_tabnet_only` |
| Component: FT-Transformer only | `eocrc-dat train --experiment ablation_component_ft_only` |
| Pairwise: no attention (MLP) | `eocrc-dat train --experiment ablation_pairwise_no_attention` |
| Input: drop a clinical domain | `eocrc-dat train --experiment ablation_input_no_laboratory` |
| Supplementary: unmatched cohort | `eocrc-dat evaluate --experiment supplementary_unmatched_cohort` |

## Filing a claim (tests and gates)

    make test     # pytest: shape, gradient-flow, overfit, metric, invariant, regression, smoke
    make lint     # ruff + isort + black
    make type     # mypy --strict
    make smoke    # two-step training on the unit-test configuration

Provenance for every equation, table and metric is recorded in
`docs/implementation-map.md`.
