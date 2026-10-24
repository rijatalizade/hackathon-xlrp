# [Team name] — Report

Copy to `report/REPORT.md` and fill in. Owner: P5.

## Problem
[One paragraph: score each test tx for suspicion; metric = [metric]; minimise false positives.]

## Data & assumptions
[Rows, accounts, time span, currencies/tx types, label rate, train/test split. Schema mapping
choices. Assumptions made where the data was ambiguous.]

## Method
### Features
[tx_, acct_, graph_, rule_ — what each captures and which typology it targets.]
### Model
[LightGBM params / IsolationForest; supervised vs unsupervised mode.]
### Ensemble
[Rank-average weights and why.]

## Validation & results
| Submission | Commit | CV [metric] | Leaderboard | Notes |
|---|---|---|---|---|
| [v0 baseline] | | | | |

[CV vs leaderboard gap and what it suggests.]

## What the model flags and why
[Top features (importance), 2–3 worked examples of high-scoring txs and the typology they match.]

## Uncertainty & false positives
[Score calibration, what legitimate behaviour scores high, how sensitive results are to thresholds.]

## Data limitations & bias
[Synthetic data caveats, leakage risks, representativeness. **Not a compliance tool.**]

## Reproducibility
[`./run.sh`, runtime, hardware, seed, clean-clone check on a second laptop.]

## What we'd do next
[…]
