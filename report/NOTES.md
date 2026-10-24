# Working notes: EDA findings and experiment log

Material for the report (P5). All scores are 100 × AP from repeated grouped CV on `train/`
(5 folds × 3 seeds, grouped by `scenario_id`). "Paired" = per-fold difference against the
previous row on identical folds; we keep a change only if gain > 2 × SE.

## EDA findings (train split)

- 40,131 payments, 700 wallets, 7 days; 1,250 suspicious (3.1%): 800 easy, 225 medium,
  225 hard. Test: 40,101 payments, a different week, new wallets **and a different token
  issuer** → asset identities can't be features.
- Every motif family (chain, cycle, fan_in/out, fan_out_in, bipartite, stack, multi_stage,
  random) has **legitimate decoy scenarios of the same shape** (~1,080 decoy txs).
- **Speed separates suspicious from decoy.** Suspicious motifs complete in a median 0.13h vs
  1.0h for decoys (scenario-level AUC 0.79). Medium = the same motifs spread over 1–3h
  (no faster than decoys, so still hard).
- **Hard relays**: a hub collects from several wallets, then pays out in waves over ~6 days,
  inside 12 legitimate "cohorts" that are exactly 1% suspicious (200 of 20,000 txs). Only 12
  suspicious relay scenarios exist in train, so anything tuned on them would overfit.
  Faster hub forwarding (~3h vs ~30h) is the only hint we found.
- 25 "ambiguous" hard txs are random by design (they rank at the median, as expected).
- **Conversions, legs and currencies carry no signal**: equal rates for suspicious and
  legitimate txs in every family, consistent with the organisers' statement. They are also
  the features that shift most between train and test, so they are not used.
- Wallets are reused across many scenarios (487/700 appear in some suspicious motif), so
  "this wallet was bad before" doesn't help.

## Experiment log

| # | Change | CV AP | Paired vs previous | Decision |
|---|---|---|---|---|
| 0 | Organisers' starter notebook (logistic regression, 12 rules) | 30.0 | — | baseline |
| 1 | Starter features + LightGBM | 35.0 | — | keep |
| 2 | + tx / account / graph / rules modules | 44.1 | — | keep |
| 3 | + `flow` (multi-hour dwell & forwarding) | 46.0 | +2.06 ± 0.43, 13/15 | keep |
| 4 | + `motif` (temporal motif reconstruction, 10m/1h/4h) | 49.6 | +4.00 ± 1.25, 12/15 | keep |
| 5 | motif: value-similar long windows (10m/1h/2h) | 48.2 | −1.34 ± 0.78 | reject |
| 6 | motif: 10-minute window only (1h/4h percolate into giant components) | 49.2 | −0.44 ± 0.38 (noise) | keep: simpler |
| 7 | Drop conversion / legs / currency / spread features | 49.0 | −0.31 ± 0.37 (noise) | keep: removes drift |
| 8 | LightGBM more regularised (7 leaves, min 80/leaf, 500 trees) | 49.7 | +0.87 ± 0.38, 10/15 | keep |
| 9 | Drop `motif_value_cv_10m` (only feature with train/test KS > 0.1) | 50.1 | +0.21 ± 0.39 (noise) | keep: removes drift |

| — | **Leaderboard v1** (commit d13e6d0): 57.3, top-1% P 0.90 R 0.29 | CV 50.1 | LB inside CV fold range (41–60) | no overfit sign |
| 10 | Burst shape: share of wallets that pass money through, hub share | 50.4 | −0.04 ± 0.59 (noise) | reject |
| 11 | Wallet repetition (# bursts a wallet joins per week) | — | no univariate signal (all ~22) | not built |

Rows 10–11 targeted the legitimate relay txs that crowd the top 3% (ceiling analysis: removing
the worst quarter of them would add ~5 AP). `through_frac` separates them from true positives
on its own (AUC 0.78) and the model used it heavily, but only as a substitute for existing
features: the top-3% composition didn't change (relay false positives 257 → 258).

Other LightGBM settings (slower/longer, less regularised, `is_unbalance`) were all within
±0.4 of each other: the model is not sensitive to its hyperparameters.

Current model: AP 50.1 ± 5.4 (folds 41–60); easy 68.5, medium 5.4, hard 0.9; top-1%
precision 0.81, recall 0.26. No feature drifts above KS 0.1 between train and test.

## Where the remaining errors are

Top 3% of scores: most false positives are legitimate relay-cohort txs (~280) and busy
background txs (~100); motif decoys are a smaller share. Medium and hard positives rank
low: medium motifs are spread out like decoys, and hard relays look like their cohort.
