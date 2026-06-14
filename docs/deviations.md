# Deviations

Intentional departures from the manuscript, each with a paper anchor and a
justification. Reviewers should read this alongside `implementation-map.md`.

## D1. Synthetic cohort stands in for the credentialed data
- Anchor: §IV-B, Data Availability, Table I.
- The three cohorts (TriNetX, All of Us, UK Biobank) are credentialed and not
  redistributable, so no raw record ships here. `intake/synthetic.py` generates a
  deterministic, epidemiology-grounded cohort that reproduces the Table I
  prevalence structure (dev 16.7%, temporal 8.3%, external 4.1%) and embeds the
  cross-domain signal the model is meant to find (CRP x iron-deficiency anaemia,
  low-haemoglobin x platelet, type-2-diabetes x anal-fissure). `intake/adapters.py`
  raises an informative error when pointed at real credentialed extracts.
- Consequence: absolute metrics produced from the synthetic cohort are not the
  manuscript's numbers; the reported values in the README come from the paper and
  are reproduced only when the pipeline is run on the approved data.

## D2. Sparse selection realised as a stable importance gate
- Anchor: §IV-D-3, Algorithm 1 ("X_sel <- top-q features by sum_k M_k").
- `underwriters/dat.py` applies the aggregated TabNet attention as a multiplicative
  gate over the full feature set (token positions stay aligned to original
  features) instead of a hard top-q gather. A hard, per-batch gather makes the
  FT-Transformer's per-position tokeniser see a different feature at each position
  across batches, which prevents stable learning. The top-q ranking is retained and
  exposed through `DualAttentionTransformer.trace` for the interaction table
  (Table IV) so the reported selection semantics are preserved for analysis.

## D3. Simplified-score range
- Anchor: §II-D, Table V ("Total = 0-24"; bands low 0-8 / moderate 9-15 / high >=16).
- The ten scorecard items each contribute 0-3 points, so the arithmetic maximum is
  30. The manuscript states an operational range of 0-24. We keep the per-item
  0-3 mapping from Table V and the published grade bands verbatim in
  `tariff/scorecard.py`; `grade()` follows the manuscript's low/moderate/high cuts.
  The 0-24 figure is treated as the manuscript's reported empirical range, not as a
  hard cap.

## D4. Component ablations
- Anchor: §II-B, Table III.
- Architecture-level rows are runnable as separate experiments: "w/o FT
  self-attention" -> `ablation_component_tabnet_only`, "w/o TabNet attention" ->
  `ablation_component_ft_only`, "w/o both attention" -> `ablation_pairwise_no_attention`.
  Input-feature rows are runnable via `data.drop_domains`
  (`ablation_input_no_{diagnoses,laboratory,demographics,medications,procedures}`).
  The within-training rows (feature-mask regularisation, class weighting, cosine
  annealing) are documented by their reported deltas rather than re-exposed as
  toggles, to keep the trainer surface small.

## D5. Unreported hyperparameters
- Anchor: Algorithm 1, §IV-D (these four values are not stated).
- Adopted defaults in `configs/train/default.toml`: batch_size=256, max_epochs=200
  (early-stop patience 20 governs the realised length), warmup_ratio=0.05,
  grad_accum=1. Standard tabular-transformer choices, flagged as not paper-reported.

## D6. Optional gradient-boosting backends
- Anchor: Table II, Code Availability (LightGBM 4, CatBoost 1.x).
- LightGBM and CatBoost are optional extras guarded at import. XGBoost and the
  scikit-learn `HistGradientBoosting` GBDT cover the boosting family on a default
  install; the smoke path and CI need only torch + scikit-learn + XGBoost.

## D7. Haemoglobin scorecard band
- Anchor: Table V, row 5 (sex-specific haemoglobin cut-offs).
- The sex-dependent haemoglobin thresholds are pre-reduced to a 0-3 ordinal severity
  upstream (`intake/synthetic.py::_raw`) so the scorecard item uses a single ordinal
  cut sequence, matching the point structure of the other ordinal rows.
