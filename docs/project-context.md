# Project Context — EOCRC-DAT

project_name       : eocrc_dat                                        [HIGH]
domain             : clinical risk stratification — deep tabular      [HIGH]
                     learning on nationwide EHR for early-onset
                     colorectal cancer (EOCRC)
framework          : PyTorch 2.x + plain torch.nn (DAT/TabNet/FT-T);  [HIGH]
                     scikit-learn 1.x; XGBoost 2.x; LightGBM 4;
                     CatBoost 1.x (classical baselines)
venue              : IEEE Journal of Biomedical and Health            [HIGH]
                     Informatics (JBHI)
primary_datasets   : 3 cohorts (TriNetX, All of Us, UK Biobank),      [HIGH]
                     all credentialed/restricted; see §6
compute_target     : COMPUTE_NOT_REPORTED (upper-bound estimate §7)   [LOW]
hparams_reference  : Algorithm 1 + Methods §IV-D + §II-A              [HIGH]
supp_path          : none
extra_signals      : 1 algorithm box; Eq.1-5 + ISR + Eq.5b; Tables
                     I-V; Fig.1-5; operational constraints; 10-point
                     simplified score; double-blind manuscript

config_cli_stack   : Tap (typed-argument-parser) + TOML (tomllib)     [decided]
                     materialised into frozen dataclasses
package_layout     : actuarial / underwriting metaphor (ledger,       [decided]
                     intake, rating, underwriters, objective,
                     reserving, tariff, actuarial, agency, support)
readme_form        : insurance policy document; omits the verbatim
                     Code-Availability quote and the Citation section
                     (both kept only in docs/implementation-map.md)

NEEDS_USER_DECISION: 0 unresolved (1 resolved by logged default; see §8)

---

## 1. project_name — `eocrc_dat`  [HIGH]
Derived from the title content words ("early-onset colorectal cancer") plus the
named model "Dual-Attention Transformer" (DAT) and the deployable score
"DAT-EOCRC" (Results §II-D, Table V). Package import name is `eocrc_dat`.

## 2. supp_path — none  [HIGH]
No supplementary PDF, appendix file, or SI archive accompanies the manuscript;
the only appendix material ("Appendix Theorem 1") is inline in the main text
(Results §II-A). A scan of sibling locations returned no match.

## 3. domain — EOCRC risk stratification via deep tabular learning  [HIGH]
Population-level risk assessment for early-onset (age < 50) colorectal cancer
from structured EHR features, framed as a binary classification / risk-grade
task with calibration, decision-curve, and subgroup-fairness analysis
(Abstract; §I; §II; §IV). Not generic "machine learning".

## 4. framework — PyTorch 2.x deep tabular + classical baselines  [HIGH]
Stated in the manuscript's Code-Availability paragraph (p. 9): PyTorch 2.x for
DAT / TabNet / FT-Transformer; scikit-learn 1.x for the linear / kernel /
instance baselines; XGBoost 2.x, LightGBM 4, CatBoost 1.x for the
gradient-boosting family. This release uses plain `torch.nn` (no Lightning),
consistent with the small tabular model size. LightGBM and CatBoost are optional
extras guarded at import; the smoke path needs only torch + scikit-learn +
XGBoost.

## 5. venue — IEEE JBHI  [HIGH]
Running header on every page: "IEEE JOURNAL OF BIOMEDICAL AND HEALTH
INFORMATICS". The TRIPOD+AI reporting frame, DeLong comparison protocol, and
two-column layout are consistent with a JBHI submission. Authors withheld for
double-blind review.

## 6. primary_datasets — 3 credentialed cohorts (+ SEER incidence)  [HIGH]
All access is credentialed / restricted; no raw data is redistributable.

| name        | role                         | N      | cases | access / license |
|-------------|------------------------------|--------|-------|------------------|
| TriNetX     | development + temporal       | 41,052 | 6,842 | TriNetX Research Network, credentialed institutional access; de-identified (Table I, §IV-B) |
| All of Us   | external (US) validation     | 4,782  | 196   | NIH All of Us Researcher Workbench, registered + controlled tier |
| UK Biobank  | international validation      | 8,124  | 327   | UK Biobank application access (project approval) |
| SEER        | population incidence anchors | —      | —     | SEER public incidence rates (incidence framing only) |

Cohort split (Table I, §IV-B): TriNetX development window 2016-2020
(dev N=34,894 = 70% train + 15% internal test; temporal validation 2021-2022,
N=6,158 = final 15%), a geographically held-out region maps to AoU external and
UKBB international. Cases are propensity-matched 1:5 to controls on age, sex,
race/ethnicity, region. Because no cohort is redistributable, this release ships
an epidemiology-grounded synthetic cohort generator and a manifest adapter that
raises an informative error when pointed at real credentialed data.

## 7. compute_target — COMPUTE_NOT_REPORTED  [LOW]
The manuscript reports no GPU type, count, VRAM, wall-clock, or storage. The
model is small for a transformer: token dimension d=192, L=3 encoder layers,
H=8 heads, K=5 TabNet steps over ~90 features, batch on the order of 10^2,
development cohort N≈3.5x10^4. Upper-bound estimate: parameter count is on the
order of 10^6; one training run is minutes on a single consumer GPU and is
feasible on CPU. The tuning protocol (100 Optuna trials per method x 15 seeds x
5-fold CV, Table II / §II-A) dominates total cost: order 100 x 15 x 5 model fits
per method, i.e. a few thousand short fits, the only part needing a GPU pool. The
sole hard runtime figure is the operational inference-latency constraint
t_inf <= 10 ms (§IV-A). Marked COMPUTE_NOT_REPORTED; the documentation states
this honestly rather than inventing hardware.

## 8. hparams_reference — Algorithm 1 + §IV-D + §II-A  [HIGH for stated]
Stated values (used verbatim in `configs/`):

- TabNet branch: K=5 attention steps, relaxation gamma=1.3, embedding dim 64
  (§IV-D-1, Algorithm 1).
- FT-Transformer branch: L=3 layers, H=8 heads, token dim d=192 (§IV-D-2, Eq.3).
  Note the two embedding widths are distinct components, not a conflict: the
  TabNet sparse-selection branch uses width 64; the FT-Transformer / DAT trunk
  reported in Algorithm 1 uses width 192.
- Optimisation: AdamW, learning rate eta=1e-3, weight decay lambda=1e-5,
  cosine LR schedule, early stopping patience 20 on validation AUROC, loss
  BCE + lambda*||Theta||^2 (Algorithm 1).
- Tuning protocol: Optuna 100 trials per method, 15 seeds, 5-fold CV on the
  training partition, 2,000 bootstrap resamples for CIs (§II-A, §IV-B, §IV-E).
- Selection constraints (Eq.1): AUROC >= tau1=0.85, |slope-1| <= tau2=0.15,
  |CITL| <= tau3=0.05, demographic-parity ratio in [1/tau4, tau4] with
  tau4=1.25, inference latency t_inf <= 10 ms (§IV-A).

NEEDS_USER_DECISION (resolved by logged default): batch_size, max_epochs,
warmup, and grad_accum are not stated in the manuscript. Defaults adopted in
`configs/experiment/main.toml` and recorded in `docs/deviations.md`:
batch_size=256, max_epochs=200 (early-stop patience 20 governs actual length),
warmup_ratio=0.05, grad_accum=1. These are standard tabular-transformer choices
and are flagged as not paper-reported.

## 9. extra_signals
- Algorithm boxes: 1 (Algorithm 1 — DAT forward pass + training loop).
- Equations: Eq.1 (constrained objective), Eq.2 (TabNet sparsemax mask),
  Eq.3 (FT-Transformer self-attention), Eq.4 (net benefit), Eq.5 (interaction
  ratio IR=1.19) and Eq.5b (expected additive degradation); plus the interaction
  strength ratio ISR(A,B)=w(A,B)/sqrt(w(A)*w(B)).
- Tables: I (cohort characteristics), II (16 methods x 3 cohorts),
  III (component / pairwise / input-feature ablation), IV (top-10 cross-domain
  feature interactions), V (10-point risk-score elements).
- Reported headline numbers: DAT AUROC 0.896 (internal test; 95% CI 0.884-0.908),
  0.879 (temporal), 0.854 (external All of Us), 0.834 (UK Biobank); best baselines
  XGBoost 0.871, CatBoost 0.870, LightGBM 0.868; DAT vs XGBoost DeLong p=0.003
  (Holm-Bonferroni retained). Simplified DAT-EOCRC score AUROC 0.862 / 0.841 /
  0.819 (internal / temporal / external). Super-additive interaction ratio
  IR=1.19; top interaction CRP x iron-deficiency anaemia ISR=2.34, attention
  weight 0.087.
- 10-point DAT-EOCRC score, grades low 0-8 / moderate 9-15 / high >=16.
- A verbatim Code-Availability statement exists in the manuscript; per the
  release convention and the standing instruction it is recorded only in
  `docs/implementation-map.md`, not in the README.
- Ethics: TriNetX / All of Us IRB-approved or exempt, UK Biobank under
  application; de-identified data only; no competing interests declared.
